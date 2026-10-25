"""Execute the finite preregistered segregated spot/perpetual account comparison."""

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from factors.crypto.spot_perpetual_carry import schedule
from research_tools.carry_feasibility.account_audit import audit_account
from research_tools.factor_lab.io import read, save, sha
from research_tools.factor_lab.metrics import portfolio_metrics
from research_tools.factor_lab.study import executor
from research_tools.github_factor_study.run import snapshot_sources
from research_tools.regime_program.module_gate import assess_module
from research_tools.regime_program.reporting import cluster_uncertainty, partitions


def verify(folder):
    for name, digest in read(folder / "completion.json")["manifest"].items():
        p = folder / name
        if folder.resolve() not in p.resolve().parents or sha(p) != digest:
            raise ValueError("Input manifest changed " + name)


def restart(engine, inputs, signals, config, window, full, split):
    first = engine.replay(*inputs, signals, config, window["start"], split, final_end=window["end"])
    restored = json.loads(json.dumps(first["resume_state"], allow_nan=False))
    second = engine.replay(
        *inputs,
        signals,
        config,
        split,
        window["end"],
        initial_state=restored,
        final_end=window["end"],
    )
    pd.testing.assert_frame_equal(
        pd.concat([first["curve"], second["curve"].iloc[1:]], ignore_index=True), full["curve"]
    )
    for key in ["events", "requests", "trades"]:
        if first[key] + second[key] != full[key]:
            raise ValueError("Actual JSON continuation changes " + key)
    if second["resume_state"] != full["resume_state"]:
        raise ValueError("Actual JSON final state differs")
    return {
        "passed": True,
        "split": split,
        "open_positions_at_split": len(first["positions"]),
        "pending_exits_at_split": sum("exit_request" in p for p in first["positions"].values()),
        "exact_json_roundtrip": True,
        "curve_rows": len(full["curve"]),
    }


def run(root, previous_attempt=None):
    root.mkdir(parents=True, exist_ok=False)
    (root / "accounts").mkdir()
    (root / "source_snapshot").mkdir()
    plan_path = Path(__file__).with_name("account_plan.json")
    plan = read(plan_path)
    protocol = read(Path("docs/research/regime_program/targets.json"))
    save(root / "protocol.json", protocol)
    spot_root, mark_root, perp_root = [
        Path(plan[k]) for k in ["spot_data", "mark_data", "perpetual_data"]
    ]
    for folder in [spot_root, mark_root]:
        verify(folder)
    source_files = [
        Path(__file__),
        plan_path,
        Path(plan["engine"]),
        Path(plan["factor"]),
        Path("research_tools/carry_feasibility/account_audit.py"),
        Path("research_tools/factor_lab/io.py"),
        Path("research_tools/factor_lab/metrics.py"),
        Path("research_tools/factor_lab/study.py"),
        Path("research_tools/github_factor_study/run.py"),
        Path("research_tools/regime_program/module_gate.py"),
        Path("research_tools/regime_program/reporting.py"),
        root / "protocol.json",
    ]
    source_files += [
        Path(plan[k]) for k in ["legacy_hold_plan", "legacy_hold_train", "legacy_hold_scope"]
    ]
    if previous_attempt:
        source_files += (
            [previous_attempt / "plan.json"]
            + list((previous_attempt / "accounts").glob("*/result.json"))
            + list((previous_attempt / "accounts").glob("*/equity_5m.parquet"))
        )
    source_files += [
        spot_root / "completion.json",
        mark_root / "completion.json",
        perp_root / "completion.json",
    ]
    source_files += [perp_root / f"{s}_5m.parquet" for s in plan["symbols"]]
    source_files += [
        Path("results/research/regime_states_20261025_v002") / n
        for n in ["completion.json", "states_daily.parquet", "event_risk_hourly.parquet"]
    ]
    plan["source_hashes"] = {str(p): sha(p) for p in source_files}
    plan["registered_at_utc"] = datetime.now(UTC).isoformat()
    plan["pid"] = os.getpid()
    save(root / "plan.json", plan)
    snapshot_sources(source_files, root / "source_snapshot")
    save(
        root / "legacy_evidence.json",
        {
            "training": read(Path(plan["legacy_hold_train"])),
            "scope": read(Path(plan["legacy_hold_scope"])),
            "interpretation": "Existing2024training hold baseline, not new independent alpha and not current-period profit evidence",
        },
    )
    sources = read(perp_root / "completion.json")["manifest"]
    for s in plan["symbols"]:
        if sha(perp_root / f"{s}_5m.parquet") != sources[f"{s}_5m.parquet"]:
            raise ValueError("Perpetual source changed")
    spot = {s: pd.read_parquet(spot_root / f"{s}_spot_5m.parquet") for s in plan["symbols"]}
    perp = {s: pd.read_parquet(perp_root / f"{s}_5m.parquet") for s in plan["symbols"]}
    marks = {s: pd.read_parquet(mark_root / f"{s}_mark_5m.parquet") for s in plan["symbols"]}
    funding = {
        s: pd.read_parquet(spot_root / f"{s}_funding_exact.parquet") for s in plan["symbols"]
    }
    inputs = (spot, perp, marks, funding)
    engine = executor(plan["engine"])
    state_root = Path("results/research/regime_states_20261025_v002")
    verify(state_root)
    states = pd.read_parquet(state_root / "states_daily.parquet")
    event_flags = pd.read_parquet(
        state_root / "event_risk_hourly.parquet"
    ).covered_fomc_post_release
    rows, audits, restarts = [], [], []
    reproduced = []

    def one(candidate, window, cost, phase, capital=10000):
        config = {**plan["account"], "capital": capital, "slippage_bps": cost}
        clocks = sorted(
            set(
                [pd.Timestamp(window["start"], tz="UTC")]
                + list(
                    pd.date_range(
                        window["start"], window["end"], freq="W-MON", inclusive="left", tz="UTC"
                    )
                )
            )
        )
        signals = schedule(
            funding, clocks, plan["condition"], read(spot_root / "plan.json")["start"], candidate
        )
        a = engine.replay(*inputs, signals, config, window["start"], window["end"])
        dates = pd.date_range(window["start"], window["end"], freq="D", tz="UTC")
        benchmark = spot["BTCUSDT"].close.reindex(dates - pd.Timedelta(minutes=5))
        benchmark.index = dates
        compatible = a["curve"].assign(
            cash=a["curve"].quote_cash + a["curve"].futures_wallet,
            gross_exposure=a["curve"].gross_notional,
        )
        m = portfolio_metrics(compatible, a["trades"], a["events"], capital, benchmark)
        last = a["curve"].iloc[-1]
        m["open_pnl_with_exit_reserve"] = (
            sum(p["cash_flow"] for p in a["positions"].values())
            + last.spot_assets
            + last.perp_unrealized
            - last.exit_reserve
        )
        m["matched_closed_pairs"] = sum(t["matched_hedge"] for t in a["trades"])
        m["maximum_conservative_stress_drawdown"] = float(
            (
                1
                - a["curve"].stress_nav_envelope.iloc[1:].to_numpy()
                / a["curve"].equity.cummax().iloc[:-1].to_numpy()
            ).max()
        )
        m["mean_absolute_net_exposure"] = float(
            (a["curve"].net_notional / a["curve"].equity).abs().mean()
        )
        identity = f"{phase}_{candidate}_cap{capital}_slip{cost}"
        folder = root / "accounts" / identity
        folder.mkdir()
        a["curve"].to_parquet(folder / "equity_5m.parquet", index=False)
        for key in ["events", "requests", "trades", "positions", "resume_state"]:
            save(folder / (key + ".json"), a[key])
        save(folder / "signals.json", signals)
        result = {
            "config": config,
            "window": window,
            "metrics": m,
            "capacity": a["capacity"],
            "flags": a["flags"],
            "wallets": a["wallets"],
            "risk_halted": a["risk_halted"],
            "engine": a["engine"],
            "production_eligible": False,
            "orders_enabled": False,
            "independent_oos": False,
            "source_hashes": {p.name: sha(p) for p in folder.iterdir()},
        }
        save(folder / "result.json", result)
        if previous_attempt and (previous_attempt / "accounts" / identity / "result.json").exists():
            old = previous_attempt / "accounts" / identity
            pd.testing.assert_frame_equal(a["curve"], pd.read_parquet(old / "equity_5m.parquet"))
            if m != read(old / "result.json")["metrics"]:
                raise ValueError("Journal repair changed prior economic metrics")
            normalised = [
                {
                    k: v
                    for k, v in e.items()
                    if not (e["type"] == "transfer" and k == "effective_time")
                }
                for e in a["events"]
            ]
            if normalised != read(old / "events.json"):
                raise ValueError("Journal repair changed prior economic ledger")
            reproduced.append(
                {
                    "id": identity,
                    "passed": True,
                    "equity_and_all_curve_fields_exact": True,
                    "metrics_exact": True,
                    "economic_ledger_exact_except_explicit_transfer_valuation_time": True,
                    "previous_attempt_audit_passed": False,
                }
            )
        audit = audit_account(folder, *inputs)
        save(folder / "independent_audit.json", audit)
        row = {
            "id": identity,
            "candidate": candidate,
            "phase": phase,
            "capital": capital,
            "slippage_bps": cost,
            "start": window["start"],
            "end": window["end"],
            "metrics": m,
            "capacity": a["capacity"],
            "flags": a["flags"],
            "risk_halted": a["risk_halted"],
            "audit_passed": audit["passed"],
            "result_sha256": sha(folder / "result.json"),
            "lifecycle_uncertainty": cluster_uncertainty(a["trades"]),
            "reused_reference": False,
        }
        save(
            folder / "periods.json",
            partitions(compatible, a["trades"], states, event_flags, capital),
        )
        rows.append(row)
        audits.append({"id": identity, **audit})
        if phase == "continuous2025_2026" and cost == 10:
            entries = [e for e in a["events"] if e["type"] == "entry"]
            if entries:
                split = pd.Timestamp(entries[0]["time"]).isoformat()
                restarts.append(
                    {"id": identity, **restart(engine, inputs, signals, config, window, a, split)}
                )
            restarts.append(
                {
                    "id": identity,
                    **restart(engine, inputs, signals, config, window, a, "2026-01-01"),
                }
            )
        print(
            json.dumps(
                {
                    "completed": identity,
                    "net_return": m["net_return"],
                    "closed_cycles": m["closed_trades"],
                    "halted": a["risk_halted"],
                    "audit_passed": audit["passed"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        (root / "status.json").write_text(
            json.dumps(
                {
                    "status": "running",
                    "pid": os.getpid(),
                    "completed_accounts": len(rows),
                    "last_account": identity,
                }
            )
            + "\n"
        )
        return row

    for window in plan["windows"]:
        for candidate in plan["candidates"]:
            for cost in plan["costs_slippage_bps"]:
                one(candidate, window, cost, window["id"])
    capacity_followups = []
    for candidate in plan["candidates"]:
        cases = [
            r for r in rows if r["candidate"] == candidate and r["phase"] == "continuous2025_2026"
        ]
        if all(
            r["metrics"]["net_return"] > 0
            and r["metrics"]["closed_net_pnl"] > 0
            and not r["risk_halted"]
            and not any(r["flags"].values())
            for r in cases
        ):
            capacity_followups.append(candidate)
    save(
        root / "capacity_selection.json",
        {
            "candidates": capacity_followups,
            "used_continuous_returns": True,
            "role": "capacity diagnostics only; not admission or untouched OOS",
        },
    )
    for candidate in capacity_followups:
        for capital in plan["capacity_capitals"]:
            for cost in plan["costs_slippage_bps"]:
                one(candidate, plan["windows"][-1], cost, "capacity", capital)
    if len(rows) > plan["budgets"]["new_total_accounts_max"]:
        raise ValueError("Account budget exceeded")
    assessments = []
    for candidate in plan["candidates"]:
        cases = [
            r for r in rows if r["candidate"] == candidate and r["phase"] == "continuous2025_2026"
        ]
        cap = [
            r
            for r in rows
            if r["candidate"] == candidate and r["phase"] == "capacity" and r["capital"] == 100000
        ]
        cap_pass = len(cap) == 2 and all(
            r["metrics"]["net_return"] > 0
            and not r["risk_halted"]
            and not any(r["flags"].values())
            and not r["capacity"]["hedge_failures"]
            for r in cap
        )
        clusters = min(
            r["lifecycle_uncertainty"]["by_entry_month"]["clusters"]
            if r["lifecycle_uncertainty"]["by_entry_month"]
            else 0
            for r in cases
        )
        assessment = assess_module(
            cases,
            protocol["module_acceptance"],
            domain_registered=True,
            sleeve_capacity_passed=cap_pass,
            cluster_groups=clusters,
        )
        if any(any(r["flags"].values()) for r in cases):
            assessment.update(
                status="risk_failed_current_domain",
                eligible_for_portfolio_test=False,
                reasons=["collateral_or_marked_leverage_path_failed"],
            )
        assessments.append(
            {"candidate": candidate, "capacity_100k_passed_research_proxy": cap_pass, **assessment}
        )
    for p, digest in plan["source_hashes"].items():
        if sha(Path(p)) != digest:
            raise ValueError("Frozen source changed during run " + p)
    summary = {
        "schema": "segregated-spot-carry-accounts.v1",
        "actual_accounts": len(rows),
        "account_rows": rows,
        "audit": {
            "passed": all(a["passed"] for a in audits),
            "accounts": len(audits),
            "maximum_nav_error": max(a["maximum_nav_error"] for a in audits),
        },
        "audits": audits,
        "restart_checks": restarts,
        "reproduced_prior_attempt": reproduced,
        "module_assessments": assessments,
        "capacity_selection": capacity_followups,
        "profit_goal_met": False,
        "portfolio_actual": None,
        "independent_oos": False,
        "production_eligible": False,
        "orders_enabled": False,
        "mechanism_count": 1,
        "warning": "Same funding family; old2024baseline retained; actual margin tiers/lots/orderbook not verified. No total multistrategy portfolio run.",
    }
    save(root / "summary.json", summary)
    save(
        root / "completion.json",
        {
            "summary_sha256": sha(root / "summary.json"),
            "plan_sha256": sha(root / "plan.json"),
            "audit_passed": True,
            "new_accounts": len(rows),
        },
    )
    (root / "status.json").write_text(
        json.dumps({"status": "complete", "pid": os.getpid(), "completed_accounts": len(rows)})
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous-attempt", type=Path)
    args = parser.parse_args()
    try:
        run(args.output, args.previous_attempt)
    except Exception:
        if args.output.exists():
            (args.output / "status.json").write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "pid": os.getpid(),
                        "completion_exists": (args.output / "completion.json").exists(),
                    }
                )
                + "\n"
            )
        raise
