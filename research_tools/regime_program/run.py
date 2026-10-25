"""Finite registered common-account evaluation; never fabricate profitable results."""

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
from research_tools.regime_program.storage import write_account


def select(rows, plan):
    assessments = []
    limits = plan["admission"]
    for candidate in plan["candidates"]:
        cases = [r for r in rows if r["candidate"] == candidate]
        if len(cases) != 2:
            raise ValueError("Incomplete two-cost screen")
        reasons = []
        for row in cases:
            m = row["metrics"]
            tests = {
                "cagr": m["cagr"] is not None and m["cagr"] >= limits["minimum_cagr"],
                "drawdown": m["max_drawdown"] <= limits["maximum_drawdown"],
                "sharpe": m["sharpe"] is not None and m["sharpe"] >= limits["minimum_sharpe"],
                "complete_lifecycles": m["closed_trades"] >= limits["minimum_complete_lifecycles"],
                "profit_factor": m["profit_factor"] is not None
                and m["profit_factor"] >= limits["minimum_closed_profit_factor"],
                "closed_pnl": m["closed_net_pnl"] > 0,
                "risk_halt": not row["risk_halted"],
            }
            reasons.extend(
                f"slip{row['slippage_bps']}:{name}" for name, passed in tests.items() if not passed
            )
        assessments.append(
            {
                "candidate": candidate,
                "passed": not reasons,
                "reasons": reasons,
                "minimum_two_cost_return": min(r["metrics"]["net_return"] for r in cases),
                "maximum_drawdown": max(r["metrics"]["max_drawdown"] for r in cases),
            }
        )
    ranked = sorted(
        assessments,
        key=lambda a: (-a["minimum_two_cost_return"], a["maximum_drawdown"], a["candidate"]),
    )
    research = [
        r["candidate"]
        for r in ranked
        if r["candidate"]
        in plan.get(
            "followup_candidates", [c for c in plan["candidates"] if c not in ("cash", "equal_mix")]
        )
    ][:3]
    best_static = next(r["candidate"] for r in ranked if r["candidate"] in plan["static_families"])
    return {
        "assessments": assessments,
        "research_followups": research,
        "best_static": best_static,
        "admitted_candidates": [a["candidate"] for a in ranked if a["passed"]],
        "followup_cohort": list(dict.fromkeys(research + ["cash", best_static, "equal_mix"])),
        "later_returns_used": False,
        "production_eligible": False,
    }


def restart_check(engine, bars, rates, targets, config, window, full):
    split = "2025-07-01"
    first = engine.replay(bars, rates, targets, config, window["start"], split)
    state = json.loads(json.dumps(first["resume_state"], allow_nan=False))
    second = engine.replay(bars, rates, targets, config, split, window["end"], state)
    joined = pd.concat([first["curve"], second["curve"].iloc[1:]], ignore_index=True)
    pd.testing.assert_frame_equal(joined, full["curve"])
    if (
        first["events"] + second["events"] != full["events"]
        or first["requests"] + second["requests"] != full["requests"]
        or first["trades"] + second["trades"] != full["trades"]
        or second["resume_state"] != full["resume_state"]
    ):
        raise ValueError("Restart changed continuous economics or account state")
    return {
        "passed": True,
        "split": split,
        "curve_rows": len(joined),
        "ledger_events": len(full["events"]),
        "original_high_water_preserved": True,
        "funding_and_pending_exits_preserved": True,
        "json_state_sha256_basis": "exact JSON roundtrip; source and output receipts sealed separately",
    }


def run(root, plan_path=None):
    started = time.monotonic()
    root.mkdir(parents=True, exist_ok=False)
    plan_path = plan_path or Path(__file__).with_name("execution_plan.json")
    plan = read(plan_path)
    inputs, data = Path(plan["inputs"]), Path(plan["data"])
    for name, digest in read(inputs / "completion.json")["manifest"].items():
        if Path(name).name != name or sha(inputs / name) != digest:
            raise ValueError("Prepared intent/forecast changed")
    audit = read(Path(plan["input_audit"]))
    if not audit["passed"] or audit["input_completion_sha256"] != sha(inputs / "completion.json"):
        raise ValueError("Input independent audit differs")
    paths = [
        Path(__file__),
        plan_path,
        Path(plan["engine"]),
        Path("research_tools/regime_program/account_audit.py"),
        Path("research_tools/regime_program/storage.py"),
        Path("research_tools/factor_lab/io.py"),
        Path("research_tools/factor_lab/metrics.py"),
        Path("research_tools/factor_lab/study.py"),
        Path("research_tools/github_factor_study/run.py"),
        inputs / "completion.json",
        data / "completion.json",
        Path(plan["input_audit"]),
    ]
    plan["registered_at"] = datetime.now(UTC).isoformat()
    plan["source_hashes"] = {str(p.resolve()): sha(p) for p in paths}
    save(root / "plan.json", plan)
    save(
        root / "registration.json",
        {"plan_sha256": sha(root / "plan.json"), "before_account_returns": True},
    )
    (root / "source_snapshot").mkdir()
    snapshot_sources(paths, root / "source_snapshot")
    for name, digest in read(data / "completion.json")["manifest"].items():
        if Path(name).name != name or sha(data / name) != digest:
            raise ValueError("Sealed official market source changed")
    symbols = read(data / "prepare_plan.json")["symbols"]
    bars = {s: pd.read_parquet(data / (s + "_5m.parquet")) for s in symbols}
    rates = {s: pd.read_parquet(data / (s + "_funding.parquet")) for s in symbols}
    targets = {c: pd.read_parquet(inputs / (c + "_targets.parquet")) for c in plan["candidates"]}
    engine = executor(plan["engine"])
    rows, audits, restarts = [], [], []

    def one(candidate, window, cost, phase, capital=10000):
        config = {
            **plan["account"],
            **plan.get("candidate_config", {}).get(candidate, {}),
            "capital": capital,
            "slippage_bps": cost,
        }
        a = engine.replay(bars, rates, targets[candidate], config, window["start"], window["end"])
        boundary = pd.date_range(window["start"], window["end"], freq="D", tz="UTC")
        benchmark = bars["BTCUSDT"].close.reindex(boundary - pd.Timedelta(minutes=5))
        benchmark.index = boundary
        a["metrics"] = portfolio_metrics(a["curve"], a["trades"], a["events"], capital, benchmark)
        a["metrics"]["mean_absolute_net_exposure"] = float(
            (a["curve"].net_exposure / a["curve"].equity).abs().mean()
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
            "metrics": a["metrics"],
            "capacity": a["capacity"],
            "risk_halted": a["risk_halted"],
            "result_sha256": sha(root / "accounts" / key / "result.json"),
        }
        rows.append(row)
        audits.append(audit_account(root, row, bars, rates))
        if (
            phase == "screen2025"
            and candidate == plan.get("restart_candidate", "event_router")
            and cost == 10
        ):
            restarts.append(
                restart_check(engine, bars, rates, targets[candidate], config, window, a)
            )
            save(root / "restart_verification.json", restarts[-1])
        print(
            {
                "phase": phase,
                "candidate": candidate,
                "slippage_bps": cost,
                "capital": capital,
                "net_return": round(a["metrics"]["net_return"], 6),
                "closed": a["metrics"]["closed_trades"],
                "audit": True,
                "accounts": len(rows),
                "seconds": round(time.monotonic() - started, 1),
            },
            flush=True,
        )

    for candidate in plan["candidates"]:
        for cost in plan["costs_slippage_bps"]:
            one(candidate, plan["screen"], cost, "screen2025")
    selection = select(rows, plan)
    save(root / "selection.json", selection)
    for window in plan["diagnostics"]:
        for candidate in selection["followup_cohort"]:
            for cost in plan["costs_slippage_bps"]:
                one(candidate, window, cost, window["id"])
    for candidate in selection["research_followups"]:
        for capital in plan["capacity_capitals"]:
            for cost in plan["costs_slippage_bps"]:
                one(candidate, plan["diagnostics"][-1], cost, "capacity", capital)
    if len(rows) > 60 or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Experiment budget or identity violated")
    summary = {
        "schema": "multi-regime-account-results.v1",
        "selection": selection,
        "account_rows": rows,
        "audit": {
            "passed": all(a["passed"] for a in audits),
            "accounts": len(audits),
            "maximum_nav_error": max(a["max_nav_error"] for a in audits),
        },
        "audit_accounts": audits,
        "restart_checks": restarts,
        "elapsed_seconds": time.monotonic() - started,
        "profit_goal_met": False,
        "independent_oos": False,
        "orders_enabled": False,
        "production_eligible": False,
    }
    save(root / "summary.json", summary)
    save(
        root / "completion.json",
        {
            "summary_sha256": sha(root / "summary.json"),
            "plan_sha256": sha(root / "plan.json"),
            "selection_sha256": sha(root / "selection.json"),
            "accounts": len(rows),
        },
    )
    print(
        {
            "finished": True,
            "accounts": len(rows),
            "admitted": selection["admitted_candidates"],
            "followups": selection["research_followups"],
            "seconds": time.monotonic() - started,
        },
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args()
    run(args.output.resolve(), args.plan.resolve() if args.plan else None)
