"""Bounded registered two-leg accounts; every result independently reconstructed."""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.factor_lab.metrics import portfolio_metrics
from research_tools.factor_lab.study import executor
from research_tools.github_factor_study.run import snapshot_sources
from research_tools.regime_program.pair_account_audit import audit_account
from research_tools.regime_program.pair_storage import write_account


def assess(rows, protocol):
    reasons = []
    p = protocol["admission"]
    for row in rows:
        m = row["metrics"]
        checks = {
            "cagr": m["cagr"] is not None and m["cagr"] >= p["minimum_cagr"],
            "drawdown": m["max_drawdown"] <= p["maximum_drawdown"],
            "sharpe": m["sharpe"] is not None and m["sharpe"] >= p["minimum_sharpe"],
            "matched_lifecycles": m["matched_closed_pairs"] >= p["minimum_complete_lifecycles"],
            "profit_factor": m["profit_factor"] is not None
            and m["profit_factor"] >= p["minimum_closed_profit_factor"],
            "closed_pnl": m["closed_net_pnl"] > 0,
            "no_halt": not row["risk_halted"],
            "no_marked_leverage": m["max_gross_exposure"] <= 1,
        }
        reasons.extend(
            f"slip{row['slippage_bps']}:{key}" for key, passed in checks.items() if not passed
        )
    return {"passed": not reasons, "reasons": reasons}


def restart(engine, bars, rates, signals, config, window, full, split):
    first = engine.replay(bars, rates, signals, config, window["start"], split)
    state = json.loads(json.dumps(first["resume_state"], allow_nan=False))
    second = engine.replay(bars, rates, signals, config, split, window["end"], state)
    joined = pd.concat([first["curve"], second["curve"].iloc[1:]], ignore_index=True)
    pd.testing.assert_frame_equal(joined, full["curve"])
    for key in ("events", "requests", "trades"):
        if first[key] + second[key] != full[key]:
            raise ValueError("Pair JSON continuation changed " + key)
    if second["resume_state"] != full["resume_state"]:
        raise ValueError("Pair JSON final state changed")
    return {
        "passed": True,
        "split": split,
        "open_pairs_at_split": len(first["pair_positions"]),
        "pending_exits_at_split": len(first["pending_exits"]),
        "halted_at_split": first["risk_halted"],
        "curve_rows": len(joined),
        "ledger_events": len(full["events"]),
        "exact_json_roundtrip": True,
    }


def run(root, previous_attempt=None):
    started = time.monotonic()
    root.mkdir(parents=True, exist_ok=False)
    plan_path = Path(__file__).with_name("pair_execution_plan.json")
    plan = read(plan_path)
    protocol = read(Path(plan["formation_plan"]))
    inputs = Path(plan["inputs"])
    data = Path(plan["data"])
    input_audit_path = inputs.with_name(inputs.name + "_independent_audit.json")
    input_audit = read(input_audit_path)
    if not input_audit["passed"] or input_audit["input_completion_sha256"] != sha(
        inputs / "completion.json"
    ):
        raise ValueError("Pair input independent audit changed")
    for name, digest in read(inputs / "completion.json")["manifest"].items():
        if Path(name).name != name or sha(inputs / name) != digest:
            raise ValueError("Signals changed")
    paths = [
        Path(__file__),
        plan_path,
        Path(plan["formation_plan"]),
        Path(plan["engine"]),
        Path("research_tools/regime_program/pair_account_audit.py"),
        Path("research_tools/regime_program/pair_storage.py"),
        Path("research_tools/factor_lab/io.py"),
        Path("research_tools/factor_lab/metrics.py"),
        Path("research_tools/factor_lab/study.py"),
        Path("research_tools/github_factor_study/run.py"),
        inputs / "completion.json",
        input_audit_path,
        data / "completion.json",
    ]
    plan["registered_at"] = datetime.now(UTC).isoformat()
    plan["previous_incomplete_attempt"] = str(previous_attempt) if previous_attempt else None
    plan["source_hashes"] = {str(p.resolve()): sha(p) for p in paths}
    save(root / "plan.json", plan)
    save(
        root / "registration.json",
        {"plan_sha256": sha(root / "plan.json"), "before_pair_account_returns": True},
    )
    (root / "source_snapshot").mkdir()
    snapshot_sources(paths, root / "source_snapshot")
    for name, digest in read(data / "completion.json")["manifest"].items():
        if sha(data / name) != digest:
            raise ValueError("Official data changed")
    # Full declared40-name universe is retained as an input identity, even though
    # each strategy only touches its causally announced, disjoint pair names.
    symbols = read(data / "prepare_plan.json")["symbols"]
    bars = {s: pd.read_parquet(data / (s + "_5m.parquet")) for s in symbols}
    rates = {s: pd.read_parquet(data / (s + "_funding.parquet")) for s in symbols}
    signals = {c: read(inputs / (c + ".json")) for c in plan["candidates"]}
    engine = executor(plan["engine"])
    rows = []
    reproduced = []
    audits = []
    restarts = []

    def one(candidate, window, cost, phase, capital=10000):
        config = {**plan["account"], "capital": capital, "slippage_bps": cost}
        a = engine.replay(bars, rates, signals[candidate], config, window["start"], window["end"])
        dates = pd.date_range(window["start"], window["end"], freq="D", tz="UTC")
        benchmark = bars["BTCUSDT"].close.reindex(dates - pd.Timedelta(minutes=5))
        benchmark.index = dates
        a["metrics"] = portfolio_metrics(a["curve"], a["trades"], a["events"], capital, benchmark)
        m = a["metrics"]
        m["matched_closed_pairs"] = sum(t["hedge_matched"] for t in a["trades"])
        m["failed_hedge_closed_attempts"] = len(a["trades"]) - m["matched_closed_pairs"]
        m["open_campaign_settled_net_pnl"] = sum(p["net_pnl"] for p in a["pair_positions"].values())
        m["open_campaign_total_net_pnl"] = (
            m["open_campaign_settled_net_pnl"] + m["open_pnl_with_exit_reserve"]
        )
        m["mean_absolute_net_exposure"] = float(
            (a["curve"].net_exposure / a["curve"].equity).abs().mean()
        )
        m["max_absolute_net_exposure"] = float(
            (a["curve"].net_exposure / a["curve"].equity).abs().max()
        )
        m["lifecycle_unit"] = (
            "one pair campaign; failed hedge attempts included in PnL but excluded from matched100-cycle gate"
        )
        key = f"{phase}_{candidate}_cap{capital}_slip{cost}"
        write_account(root / "accounts" / key, a, config, window)
        row = {
            "id": key,
            "candidate": candidate,
            "phase": phase,
            "capital": capital,
            "slippage_bps": cost,
            "start": window["start"],
            "end": window["end"],
            "metrics": m,
            "capacity": a["capacity"],
            "risk_halted": a["risk_halted"],
            "result_sha256": sha(root / "accounts" / key / "result.json"),
        }
        rows.append(row)
        audits.append(audit_account(root, row, bars, rates, signals[candidate]))
        old = previous_attempt / "accounts" / key if previous_attempt else None
        if old and (old / "result.json").is_file():
            previous = read(old / "result.json")
            for name, expected in previous["source_hashes"].items():
                if sha(old / name) != expected:
                    raise ValueError("Previous incomplete attempt evidence changed")
            if previous["metrics"] != m:
                raise ValueError("Timestamp-only repair changed metrics")
            for filename in ("events.json", "requests.json", "trades.json", "resume_state.json"):
                if read(old / filename) != read(root / "accounts" / key / filename):
                    raise ValueError("Timestamp-only repair changed economics or state")
            reproduced.append(key)
        if (
            candidate != "cash"
            and cost == 10
            and capital == 10000
            and phase in ("screen2025", "continuous2025_2026")
        ):
            first_entry = next((e for e in a["events"] if e["type"] == "entry"), None)
            split = (
                first_entry["time"]
                if phase == "screen2025" and first_entry
                else "2026-01-01"
                if phase == "continuous2025_2026"
                else "2025-07-01"
            )
            check = restart(engine, bars, rates, signals[candidate], config, window, a, split)
            check.update(candidate=candidate, phase=phase)
            restarts.append(check)
            save(root / (key + "_restart.json"), check)
        print(
            {
                "account": len(rows),
                "phase": phase,
                "candidate": candidate,
                "slip": cost,
                "capital": capital,
                "net_return": round(m["net_return"], 6),
                "pairs": m["closed_trades"],
                "matched": m["matched_closed_pairs"],
                "hedge_failures": a["capacity"]["hedge_failures"],
                "audit": True,
                "seconds": round(time.monotonic() - started, 1),
            },
            flush=True,
        )

    for candidate in plan["candidates"]:
        for cost in protocol["costs_slippage_bps"]:
            one(candidate, protocol["screen"], cost, "screen2025")
    selection = {
        "assessments": [
            {"candidate": c, **assess([r for r in rows if r["candidate"] == c], protocol)}
            for c in plan["candidates"]
        ],
        "followup_cohort": plan["candidates"],
        "followup_basis": "both profiles and cash frozen before returns; no later return ranking",
        "later_returns_used": False,
        "production_eligible": False,
    }
    save(root / "selection.json", selection)
    for window in protocol["diagnostics"]:
        for candidate in plan["candidates"]:
            for cost in protocol["costs_slippage_bps"]:
                one(candidate, window, cost, window["id"])
    for candidate in ("pairs_fixed", "pairs_rolling"):
        for capital in protocol["capacity_capitals"]:
            for cost in protocol["costs_slippage_bps"]:
                one(candidate, protocol["diagnostics"][-1], cost, "capacity", capital)
    if (
        len(rows) != 26
        or len(rows) > plan["budget_accounts_max"]
        or len({r["id"] for r in rows}) != len(rows)
    ):
        raise ValueError("Pair account budget violated")
    economics = []
    for candidate in ("pairs_fixed", "pairs_rolling"):
        continuous = [
            r for r in rows if r["candidate"] == candidate and r["phase"] == "continuous2025_2026"
        ]
        verdict = assess(continuous, protocol)
        capacity = [
            r
            for r in rows
            if r["candidate"] == candidate and r["phase"] == "capacity" and r["capital"] == 100000
        ]
        if not all(
            r["metrics"]["net_return"] > 0
            and r["metrics"]["max_drawdown"] <= 0.15
            and not r["risk_halted"]
            for r in capacity
        ):
            verdict["passed"] = False
            verdict["reasons"].append("100k_both_cost_capacity")
        economics.append({"candidate": candidate, **verdict})
    save(
        root / "summary.json",
        {
            "schema": "two-leg-pair-account-results.v1",
            "account_rows": rows,
            "selection": selection,
            "economics": economics,
            "audit": {
                "passed": all(a["passed"] for a in audits),
                "accounts": len(audits),
                "maximum_nav_error": max(a["max_nav_error"] for a in audits),
            },
            "audit_accounts": audits,
            "restart_checks": restarts,
            "input_validation": input_audit,
            "timestamp_repair_reproduced_accounts": reproduced,
            "profit_goal_met": any(e["passed"] for e in economics),
            "independent_oos": False,
            "production_eligible": False,
            "orders_enabled": False,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    save(
        root / "completion.json",
        {
            "summary_sha256": sha(root / "summary.json"),
            "plan_sha256": sha(root / "plan.json"),
            "selection_sha256": sha(root / "selection.json"),
            "accounts": len(rows),
        },
    )
    print({"finished": True, "accounts": len(rows), "economics": economics}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous-attempt", type=Path)
    args = parser.parse_args()
    run(args.output.resolve(), args.previous_attempt.resolve() if args.previous_attempt else None)
