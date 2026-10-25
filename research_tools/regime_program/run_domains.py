"""Bounded conditional-module study with frozen screening and matched controls."""

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
from research_tools.regime_program.account_audit import audit_account
from research_tools.regime_program.module_gate import assess_module
from research_tools.regime_program.reporting import cluster_uncertainty
from research_tools.regime_program.storage import write_account


def select_followups(rows, modules):
    results = []
    for candidate in modules:
        cases = [
            r
            for r in rows
            if r["kind"] == "module" and r["phase"] == "screen2025" and r["candidate"] == candidate
        ]
        if len(cases) != 2 or {r["slippage_bps"] for r in cases} != {3, 10}:
            raise ValueError("Incomplete two-cost domain screen")
        if all(
            r["metrics"]["net_return"] > 0 and r["metrics"]["closed_net_pnl"] > 0 for r in cases
        ):
            results.append(
                {
                    "candidate": candidate,
                    "minimum_complete_cycles": min(r["metrics"]["closed_trades"] for r in cases),
                    "minimum_expectancy": min(r["metrics"]["expectancy"] for r in cases),
                }
            )
    return [
        r["candidate"]
        for r in sorted(
            results,
            key=lambda r: (-r["minimum_complete_cycles"], -r["minimum_expectancy"], r["candidate"]),
        )[:3]
    ]


def restart(engine, bars, rates, targets, config, window, full):
    entry = next((e for e in full["events"] if e["type"] == "entry"), None)
    # Old frozen engine expects both endpoints to have the same string form.
    # Choose a real nonflat boundary without using returns to select strategies.
    split = pd.Timestamp(entry["time"]).strftime("%Y-%m-%d %H:%M") if entry else "2025-07-01"
    first = engine.replay(bars, rates, targets, config, window["start"], split)
    state = json.loads(json.dumps(first["resume_state"], allow_nan=False))
    second = engine.replay(bars, rates, targets, config, split, window["end"], state)
    joined = pd.concat([first["curve"], second["curve"].iloc[1:]], ignore_index=True)
    pd.testing.assert_frame_equal(joined, full["curve"])
    for key in ("events", "requests", "trades"):
        if first[key] + second[key] != full[key]:
            raise ValueError("Domain JSON restart changed " + key)
    if second["resume_state"] != full["resume_state"]:
        raise ValueError("Domain JSON final state differs")
    return {
        "passed": True,
        "split": split,
        "open_positions_at_split": len(first["positions"]),
        "pending_exits_at_split": len(first["pending_exits"]),
        "curve_rows": len(joined),
        "exact_json_roundtrip": True,
    }


def run(root, inputs):
    started = time.monotonic()
    root.mkdir(parents=True, exist_ok=False)
    plan = read(inputs / "plan.json")
    parent = Path(plan["parent_accounts"])
    data = Path(plan["data"])
    protocol = read(Path(plan["protocol_snapshot"]))
    cards = {c["id"]: c for c in read(Path(plan["registry_snapshot"]))["cards"]}
    for name, digest in read(inputs / "completion.json")["manifest"].items():
        if Path(name).name != name or sha(inputs / name) != digest:
            raise ValueError("Domain inputs changed")
    input_audit_path = inputs.with_name(inputs.name + "_independent_audit.json")
    input_audit = read(input_audit_path)
    if not input_audit["passed"] or input_audit["input_completion_sha256"] != sha(
        inputs / "completion.json"
    ):
        raise ValueError("Domain input audit changed")
    parent_receipt = read(parent / "completion.json")
    if sha(parent / "summary.json") != parent_receipt["summary_sha256"]:
        raise ValueError("Parent reference changed")
    parent_summary = read(parent / "summary.json")
    if not parent_summary["audit"]["passed"]:
        raise ValueError("Parent reference audit missing")
    plan["inputs"] = str(inputs.resolve())
    plan["registered_at"] = datetime.now(UTC).isoformat()
    plan["mechanism_groups"] = {
        "trend_domain": "price_persistence",
        "relative_domain": "price_persistence",
        "breakout_domain": "price_persistence",
        "carry_domain": "funding_plus_momentum_unseparated",
        "price_reversion_domain": "price_reversion",
    }
    plan["implementation_clarifications"] = [
        "Missing full-family controls are generated only in diagnostic windows. Capacity comparisons use already frozen static trend and actual cash controls; missing own-family capacity is left unavailable.",
        "All high-cost2025module accounts verify real JSON continuation at first entry or a fixed flat boundary; selection uses only registered two-cost2025metrics.",
        "Trend, relative momentum and breakout are not automatically independent mechanisms. Funding-plus-momentum requires a same-opportunity funding-term ablation before claiming separate funding alpha.",
    ]
    paths = [
        Path(__file__),
        Path(plan["engine"]),
        Path(plan["gate"]),
        Path("research_tools/regime_program/account_audit.py"),
        Path("research_tools/regime_program/storage.py"),
        Path("research_tools/regime_program/reporting.py"),
        Path("research_tools/factor_lab/metrics.py"),
        Path("research_tools/factor_lab/io.py"),
        Path("research_tools/factor_lab/study.py"),
        Path("research_tools/github_factor_study/run.py"),
        Path(plan["protocol_snapshot"]),
        Path(plan["registry_snapshot"]),
        inputs / "completion.json",
        input_audit_path,
        data / "completion.json",
        parent / "completion.json",
    ]
    plan["source_hashes"] = {str(p.resolve()): sha(p) for p in paths}
    save(root / "plan.json", plan)
    save(
        root / "registration.json",
        {"plan_sha256": sha(root / "plan.json"), "before_domain_account_returns": True},
    )
    (root / "source_snapshot").mkdir()
    snapshot_sources(paths, root / "source_snapshot")
    for name, digest in read(data / "completion.json")["manifest"].items():
        if sha(data / name) != digest:
            raise ValueError("Official data changed")
    symbols = read(data / "prepare_plan.json")["symbols"]
    bars = {s: pd.read_parquet(data / (s + "_5m.parquet")) for s in symbols}
    rates = {s: pd.read_parquet(data / (s + "_funding.parquet")) for s in symbols}
    target = {m: pd.read_parquet(inputs / (m + "_targets.parquet")) for m in plan["screen_modules"]}
    families = {cards[m]["family"] for m in plan["screen_modules"]} | {"cash", "trend"}
    target.update(
        {
            f: pd.read_parquet(Path(plan["parent_inputs"]) / (f + "_targets.parquet"))
            for f in families
        }
    )
    engine = executor(plan["engine"])
    rows = []
    audits = []
    restarts = []
    seen = set()

    def reference(family, window, cost, phase, capital=10000):
        key = (family, phase, capital, cost, "control")
        if key in seen:
            return
        found = [
            r
            for r in parent_summary["account_rows"]
            if r["candidate"] == family
            and r["phase"] == phase
            and r["capital"] == capital
            and r["slippage_bps"] == cost
        ]
        if len(found) > 1:
            raise ValueError("Ambiguous frozen reference")
        if not found:
            return one(family, window, cost, phase, capital, kind="control")
        old = found[0]
        folder = parent / "accounts" / old["id"]
        if sha(folder / "result.json") != old["result_sha256"]:
            raise ValueError("Frozen reference changed")
        report = read(folder / "result.json")
        for name, digest in report["source_hashes"].items():
            if Path(name).name != name or sha(folder / name) != digest:
                raise ValueError("Reference ledger changed")
        if report["metrics"] != old["metrics"]:
            raise ValueError("Reference metric mismatch")
        rows.append(
            {
                **old,
                "id": "reference_" + old["id"],
                "kind": "control",
                "reused_reference": True,
                "source_account_id": old["id"],
                "source_run": str(parent.resolve()),
                "audit_passed": True,
            }
        )
        seen.add(key)

    def one(candidate, window, cost, phase, capital=10000, kind="module"):
        key = (candidate, phase, capital, cost, kind)
        if key in seen:
            return
        config = {**plan["account"], "capital": capital, "slippage_bps": cost}
        a = engine.replay(bars, rates, target[candidate], config, window["start"], window["end"])
        days = pd.date_range(window["start"], window["end"], freq="D", tz="UTC")
        benchmark = bars["BTCUSDT"].close.reindex(days - pd.Timedelta(minutes=5))
        benchmark.index = days
        a["metrics"] = portfolio_metrics(a["curve"], a["trades"], a["events"], capital, benchmark)
        a["metrics"]["mean_absolute_net_exposure"] = float(
            (a["curve"].net_exposure / a["curve"].equity).abs().mean()
        )
        identity = f"{kind}_{phase}_{candidate}_cap{capital}_slip{cost}"
        write_account(root / "accounts" / identity, a, config, window)
        uncertainty = cluster_uncertainty(a["trades"])
        row = {
            "id": identity,
            "candidate": candidate,
            "phase": phase,
            "capital": capital,
            "slippage_bps": cost,
            "start": window["start"],
            "end": window["end"],
            "kind": kind,
            "reused_reference": False,
            "source_run": str(root.resolve()),
            "source_account_id": identity,
            "metrics": a["metrics"],
            "capacity": a["capacity"],
            "risk_halted": a["risk_halted"],
            "lifecycle_uncertainty": uncertainty,
            "result_sha256": sha(root / "accounts" / identity / "result.json"),
        }
        audit = audit_account(root, row, bars, rates)
        row["audit_passed"] = audit["passed"]
        rows.append(row)
        audits.append(audit)
        seen.add(key)
        if kind == "module" and phase == "screen2025" and cost == 10:
            check = restart(engine, bars, rates, target[candidate], config, window, a)
            check["candidate"] = candidate
            restarts.append(check)
            save(root / (identity + "_restart.json"), check)
        print(
            {
                "new_accounts": len(audits),
                "kind": kind,
                "phase": phase,
                "candidate": candidate,
                "capital": capital,
                "slip": cost,
                "net_return": round(a["metrics"]["net_return"], 6),
                "closed": a["metrics"]["closed_trades"],
                "risk_halt": a["risk_halted"],
                "audit": True,
                "seconds": round(time.monotonic() - started, 1),
            },
            flush=True,
        )

    def assessment(candidate, phase):
        cases = [
            r
            for r in rows
            if r["kind"] == "module"
            and r["candidate"] == candidate
            and r["phase"] == phase
            and r["capital"] == 10000
        ]
        groups = min(
            max((g or {}).get("clusters", 0) for g in r["lifecycle_uncertainty"].values())
            for r in cases
        )
        return {
            "candidate": candidate,
            "phase": phase,
            **assess_module(
                cases,
                protocol["module_acceptance"],
                domain_registered=True,
                sleeve_capacity_passed=None,
                cluster_groups=groups,
            ),
        }

    for candidate in plan["screen_modules"]:
        for cost in plan["costs_slippage_bps"]:
            one(candidate, plan["screen"], cost, "screen2025")
            reference(cards[candidate]["family"], plan["screen"], cost, "screen2025")
    for cost in plan["costs_slippage_bps"]:
        reference("cash", plan["screen"], cost, "screen2025")
    followups = select_followups(rows, plan["screen_modules"])
    selection = {
        "research_followups": followups,
        "assessments": [assessment(c, "screen2025") for c in plan["screen_modules"]],
        "admitted_candidates": [],
        "later_returns_used": False,
        "best_static": parent_summary["selection"]["best_static"],
        "basis": plan["selection_before_later_diagnostics"],
        "followup_is_admission": False,
    }
    save(root / "selection.json", selection)
    for window in plan["diagnostics"] if followups else []:
        for candidate in followups:
            for cost in plan["costs_slippage_bps"]:
                one(candidate, window, cost, window["id"])
                reference(cards[candidate]["family"], window, cost, window["id"])
        for cost in plan["costs_slippage_bps"]:
            reference("trend", window, cost, window["id"])
            reference("cash", window, cost, window["id"])
    for candidate in followups:
        for capital in plan["capacity_capitals"]:
            for cost in plan["costs_slippage_bps"]:
                one(candidate, plan["diagnostics"][-1], cost, "capacity", capital)
    if followups:
        for capital in plan["capacity_capitals"]:
            for cost in plan["costs_slippage_bps"]:
                reference("trend", plan["diagnostics"][-1], cost, "capacity", capital)
                reference("cash", plan["diagnostics"][-1], cost, "capacity", capital)
    limits = plan["budgets"]
    new = [r for r in rows if not r["reused_reference"]]
    counts = {
        phase: sum(r["kind"] == "module" and r["phase"] == phase for r in new)
        for phase in ("screen2025", "diagnostic2026", "continuous2025_2026", "capacity")
    }
    if (
        counts["screen2025"] > limits["new_domain_screen_max"]
        or counts["diagnostic2026"] + counts["continuous2025_2026"]
        > limits["new_domain_diagnostics_max"]
        or counts["capacity"] > limits["new_domain_capacity_max"]
        or sum(r["kind"] == "control" for r in new) > limits["missing_frozen_family_controls_max"]
        or len(new) > limits["total_new_accounts_max"]
        or len({r["id"] for r in rows}) != len(rows)
    ):
        raise ValueError("Registered domain account budget exceeded")
    assessments = selection["assessments"] + [
        assessment(c, w["id"]) for c in followups for w in plan["diagnostics"]
    ]
    save(
        root / "summary.json",
        {
            "schema": "conditional-module-account-results.v1",
            "account_rows": rows,
            "selection": selection,
            "module_assessments": assessments,
            "audit": {
                "passed": all(a["passed"] for a in audits),
                "accounts": len(audits),
                "maximum_nav_error": max(a["max_nav_error"] for a in audits),
            },
            "audit_accounts": audits,
            "restart_checks": restarts,
            "new_accounts": len(new),
            "reused_references": sum(r["reused_reference"] for r in rows),
            "budget_counts": counts,
            "input_audit": input_audit,
            "profit_goal_met": False,
            "portfolio_built": False,
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
            "new_accounts": len(new),
            "audit_passed": True,
        },
    )
    print(
        {
            "finished": True,
            "new_accounts": len(new),
            "reused_references": sum(r["reused_reference"] for r in rows),
            "followups": followups,
        },
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    args = parser.parse_args()
    run(args.output.resolve(), args.inputs.resolve())
