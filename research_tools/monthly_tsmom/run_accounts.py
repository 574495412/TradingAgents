"""Run the single monthly economic-return candidate and declared controls."""

import argparse
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.factor_lab.study import executor
from research_tools.max_effect.run_accounts import verify_manifest
from research_tools.trend_btc_role.run import calculate
from research_tools.trend_pool_audit.run import ENGINE, load
from research_tools.trend_stop_audit.run import paired_uncertainty


def run(inputs, output):
    output.mkdir(parents=True, exist_ok=False)
    verify_manifest(inputs)
    prepared = read(inputs / "plan.json")
    for name, digest in prepared["source_hashes"].items():
        assert sha(Path(name)) == digest, name
    paths = [
        Path(__file__),
        Path("research_tools/trend_btc_role/run.py"),
        Path("research_tools/trend_pool_audit/run.py"),
        Path("research_tools/trend_stop_audit/run.py"),
        Path("research_tools/trend_stop_audit/stop_audit.py"),
        Path("research_tools/sparse_market/account_audit.py"),
        Path("research_tools/factor_lab/metrics.py"),
        Path("research_tools/regime_program/storage.py"),
        ENGINE,
        ENGINE.with_name("sparse_sources_v2.py"),
        inputs / "completion.json",
    ]
    plan = {
        **prepared,
        "prepared_inputs": str(inputs),
        "account_source_hashes": {str(p): sha(p) for p in paths},
    }
    save(output / "plan.json", plan)
    equivalent = read(inputs / "summary.json")["source_target_equivalence"]
    save(
        output / "registration.json",
        {
            "plan_sha256": sha(output / "plan.json"),
            "before_new_financial_returns": True,
            "source_equivalence_before_returns": equivalent,
        },
    )
    _, bars, rates, lifecycle = load(inputs)
    sys.path.insert(0, str(ENGINE.parents[2]))
    engine = executor(str(ENGINE))
    dates = pd.date_range(plan["window"]["start"], plan["window"]["end"], freq="D", tz="UTC")
    benchmark = bars["BTCUSDT"].close.reindex(dates - pd.Timedelta(minutes=5))
    benchmark.index = dates
    tasks = [
        (f"daily_archive_{p}_cap10000_slip{c}", "daily_archive", p, c, 10000)
        for p in plan["profiles"]
        for c in plan["costs_slippage_bps"]
    ]
    assert len(tasks) <= plan["budget"]["maximum_actual_base_accounts"]
    rows, curves, failures, state = [], {}, [], {}
    started = time.monotonic()

    def perform(task):
        identity, variant, profile, cost, capital = task
        config = {**plan["account"], "slippage_bps": cost, "capital": capital}
        target_file = f"{variant}_{profile}_targets.parquet"
        row = {
            "id": identity,
            "variant": variant,
            "profile": profile,
            "capital": capital,
            "slippage_bps": cost,
            "reused_account": False,
            "financial_audit_passed": False,
            "production_eligible": False,
            "independent_oos": False,
        }
        try:
            account, receipt = calculate(
                output,
                identity,
                engine,
                bars,
                rates,
                pd.read_parquet(inputs / target_file),
                config,
                plan["window"],
                lifecycle,
                benchmark,
            )
            row.update(receipt)
            curves[identity] = account["curve"]
            if profile == "monthly_tsmom" and cost == 10 and capital == 10000:
                state["restart"] = (account, config, target_file)
        except Exception as error:
            row["failure"] = str(error)
            failures.append(
                {
                    "id": identity,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "economic_result_unknown": True,
                }
            )
        folder = output / "accounts" / identity
        save(folder / "row.json" if folder.exists() else output / (identity + "_failure.json"), row)
        rows.append(row)
        print(
            {
                "id": identity,
                "audited": row["financial_audit_passed"],
                "net_return": row.get("metrics", {}).get("net_return"),
                "seconds": round(time.monotonic() - started, 2),
            },
            flush=True,
        )

    for task in tasks:
        perform(task)
    comparisons = []
    for cost in plan["costs_slippage_bps"]:
        a = next(r for r in rows if r["profile"] == "monthly_tsmom" and r["slippage_bps"] == cost)
        b = next(r for r in rows if r["profile"] == "magnitude_long" and r["slippage_bps"] == cost)
        if a["financial_audit_passed"] and b["financial_audit_passed"]:
            comparisons.append(
                {
                    "slippage_bps": cost,
                    "trend_minus_long_net_pnl": a["metrics"]["net_pnl"] - b["metrics"]["net_pnl"],
                    "paired_uncertainty": paired_uncertainty(curves[b["id"]], curves[a["id"]]),
                }
            )
    principal = [r for r in rows if r["profile"] == "monthly_tsmom"]
    followup = len(comparisons) == 2 and all(c["trend_minus_long_net_pnl"] > 0 for c in comparisons)
    followup = followup and all(
        r["financial_audit_passed"]
        and r["metrics"]["net_pnl"] > 0
        and r["metrics"]["closed_net_pnl"] > 0
        and r["metrics"]["max_drawdown"] <= 0.15
        and not r["risk_halted"]
        for r in principal
    )
    capacity_tasks = (
        [
            (
                f"daily_archive_monthly_tsmom_cap{capital}_slip{cost}",
                "daily_archive",
                "monthly_tsmom",
                cost,
                capital,
            )
            for capital in plan["capacity_followup"]["capitals"]
            for cost in plan["costs_slippage_bps"]
        ]
        if followup
        else []
    )
    assert len(capacity_tasks) <= plan["capacity_followup"]["maximum_additional_actual_accounts"]
    for task in capacity_tasks:
        perform(task)
    restart = {"performed": False}
    if "restart" in state:
        account, config, target_file = state["restart"]
        entries = [e for e in account["events"] if e["type"] == "entry"]
        assert entries
        chosen = next((e for e in entries if e.get("addition")), entries[0])
        split = chosen["time"]
        prefix = engine.replay(
            bars,
            rates,
            pd.read_parquet(inputs / target_file),
            config,
            plan["window"]["start"],
            split,
            lifecycle=lifecycle,
        )
        assert prefix["positions"]
        save(output / "restart_checkpoint.json", prefix["resume_state"])
        save(
            output / "restart_registration.json",
            {
                "split": split,
                "config": config,
                "target_file": target_file,
                "engine_sha256": sha(ENGINE),
                "parent_pid": os.getpid(),
                "same_account_not_new_trial": True,
            },
        )
        subprocess.run(
            [sys.executable, "-m", "research_tools.trend_pool_audit.run", "--worker", str(output)],
            check=True,
        )
        child = read(output / "restart_worker.json")
        joined = pd.concat(
            [prefix["curve"].iloc[:-1], pd.read_parquet(output / "restart_worker_curve.parquet")],
            ignore_index=True,
        )
        pd.testing.assert_frame_equal(joined, account["curve"])
        assert prefix["events"] + child["events"] == account["events"]
        assert prefix["requests"] + child["requests"] == account["requests"]
        assert (
            child["positions"] == account["positions"]
            and child["resume_state"] == account["resume_state"]
        )
        assert child["pid"] != os.getpid()
        restart = {
            "performed": True,
            "passed": True,
            "split": split,
            "held_positions": len(prefix["positions"]),
            "addition_at_checkpoint": bool(chosen.get("addition")),
            "parent_pid": os.getpid(),
            "worker_pid": child["pid"],
            "full_curve_ledger_requests_inventory_and_risk_equal": True,
        }
    originals = list(rows)
    for row in originals:
        assert equivalent[row["profile"]]
        identity = row["id"].replace("daily_archive", "daily_with_complete_5m_reconciliation", 1)
        rows.append(
            {
                **row,
                "id": identity,
                "variant": "daily_with_complete_5m_reconciliation",
                "reused_account": True,
                "source_account_id": row["id"],
                "reuse_basis": "exact targets and identical execution/lifecycle before returns",
            }
        )
    summary = {
        "schema": "monthly-economic-tsmom-accounts.v1",
        "account_rows": rows,
        "comparisons": comparisons,
        "actual_new_financial_accounts": len(originals),
        "actual_base_accounts": len(tasks),
        "source_duplicate_cases_reused": len(originals),
        "actual_capacity_accounts": len(capacity_tasks),
        "logical_cases": len(rows),
        "capacity_triggered": followup,
        "capacity_completed": bool(capacity_tasks),
        "actual_new_process_restart": restart,
        "failures": failures,
        "external_review": False,
        "independent_unseen_evidence": False,
        "profit_goal_met": False,
        "production_eligible": False,
        "orders_enabled": False,
    }
    save(output / "summary.json", summary)
    save(output / "failures.json", failures)
    save(
        output / "completion.json",
        {
            "passed": not failures,
            "manifest": {
                str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()
            },
            "profit_goal_met": False,
        },
    )
    print(
        {
            "actual_new_accounts": len(originals),
            "capacity_triggered": followup,
            "failures": len(failures),
            "restart": restart,
        },
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.inputs, a.output)
