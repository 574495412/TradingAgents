"""Bounded frozen-family accounts; source/audit failures remain explicit unknowns."""

import argparse
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.factor_lab.metrics import portfolio_metrics
from research_tools.factor_lab.study import executor
from research_tools.max_effect.run_accounts import verify_manifest
from research_tools.regime_program.reporting import cluster_uncertainty
from research_tools.regime_program.storage import write_account
from research_tools.sparse_market.account_audit import audit_account


def run(inputs, output):
    output.mkdir(parents=True, exist_ok=False)
    verify_manifest(inputs)
    audited = inputs.with_name(inputs.name + "_independent_audit.json")
    if not read(audited)["passed"] or read(audited)["inputs_completion_sha256"] != sha(
        inputs / "completion.json"
    ):
        raise ValueError("Independent target audit missing or changed")
    prepared = read(inputs / "plan.json")
    for path, digest in prepared["source_hashes"].items():
        if sha(Path(path)) != digest:
            raise ValueError("Frozen target or engine implementation changed")
    recovery_path = Path(__file__).with_name("execution_recovery.json")
    recovery = read(recovery_path)
    if (
        sha(Path(recovery["failed_run"]) / "completion.json")
        != recovery["failed_completion_sha256"]
        or sha(inputs / "completion.json") != recovery["frozen_targets_completion_sha256"]
        or sha(Path(recovery["new_engine"])) != recovery["engine_sha256"]
        or sha(Path(recovery["new_helper"])) != recovery["helper_sha256"]
        or recovery["financial_parameters_changed"] is not False
    ):
        raise ValueError("Execution-only recovery identity changed")
    prepared = {**prepared, "engine": recovery["new_engine"], "execution_recovery": recovery}
    bundle, cohort = Path(prepared["bundle"]), Path(prepared["cohort"])
    verify_manifest(bundle)
    verify_manifest(cohort)
    bundle_audit = bundle.with_name(bundle.name + "_independent_audit.json")
    source_check = read(bundle_audit)
    if (
        not source_check["passed"]
        or source_check["bundle_completion_sha256"] != sha(bundle / "completion.json")
        or source_check["cohort_completion_sha256"] != sha(cohort / "completion.json")
    ):
        raise ValueError("Actual source-bundle audit missing or changed")
    paths = [
        Path(__file__),
        Path(__file__).with_name("account_audit.py"),
        inputs / "completion.json",
        audited,
        bundle_audit,
        recovery_path,
        Path(recovery["new_engine"]),
        Path(recovery["new_helper"]),
        Path(recovery["failed_run"]) / "completion.json",
        Path("research_tools/factor_lab/metrics.py"),
        Path("research_tools/factor_lab/io.py"),
        Path("research_tools/regime_program/storage.py"),
        Path("research_tools/regime_program/reporting.py"),
        Path("research_tools/factor_lab/study.py"),
        bundle / "completion.json",
        cohort / "completion.json",
    ]
    plan = {
        **prepared,
        "inputs": str(inputs),
        "capacity_priority": [c for c in prepared["candidates"] if c != "cash"],
        "maximum_initial_capacity_candidates": 2,
        "account_source_hashes": {str(p): sha(p) for p in paths},
        "before_any_account_returns": True,
        "orders_enabled": False,
        "production_eligible": False,
    }
    save(output / "plan.json", plan)
    save(
        output / "registration.json",
        {"plan_sha256": sha(output / "plan.json"), "before_account_returns": True},
    )
    save(output / "protocol.json", read(Path("docs/research/regime_program/targets.json")))
    symbols = read(cohort / "symbols.json")
    bars = {
        s: pd.read_parquet(
            bundle / "symbols" / (s + "_5m.parquet"),
            columns=[
                "open",
                "high",
                "low",
                "close",
                "quote_volume",
                "price_usable",
                "execution_liquidity_usable",
            ],
        )
        for s in symbols
    }
    rates = {s: pd.read_parquet(bundle / "funding" / (s + "_exact.parquet")) for s in symbols}
    lifecycle = read(inputs / "lifecycle_directives.json")
    sys.path.insert(0, str(Path(plan["engine"]).parents[2]))
    engine = executor(plan["engine"])
    engine.LifecyclePolicy(lifecycle, symbols)  # Full real directive schema before any accounts.
    rows, audits, failures = [], [], []
    journal = (output / "progress.jsonl").open("x")
    cases = []
    for variant in plan["source_variants"]:
        for window in plan["windows"]:
            for candidate in plan["candidates"]:
                if candidate != "cash":
                    cases.extend(
                        (variant, candidate, window, cost) for cost in plan["costs_slippage_bps"]
                    )
    if len(cases) > plan["budget"]["maximum_financial_base_accounts"]:
        raise ValueError("Base account budget exceeded")
    for window in plan["windows"]:
        cases.extend(("cash", "cash", window, cost) for cost in plan["costs_slippage_bps"])
    started = time.monotonic()
    for variant, candidate, window, cost in cases:
        identity = f"{variant}_{window['id']}_{candidate}_cap10000_slip{cost}"
        row = {
            "id": identity,
            "variant": variant,
            "candidate": candidate,
            "phase": window["id"],
            "slippage_bps": cost,
            "capital": 10000,
            "financial_audit_passed": False,
            "production_eligible": False,
            "independent_oos": False,
        }
        config = {**plan["account"], "slippage_bps": cost}
        path = inputs / (
            f"cash_{window['id']}_targets.parquet"
            if candidate == "cash"
            else f"{variant}_{window['id']}_{candidate}_targets.parquet"
        )
        scores = pd.read_parquet(path)
        try:
            a = engine.replay(
                bars, rates, scores, config, window["start"], window["end"], lifecycle=lifecycle
            )
            days = pd.date_range(window["start"], window["end"], freq="D", tz="UTC")
            benchmark = bars["BTCUSDT"].close.reindex(days - pd.Timedelta(minutes=5))
            benchmark.index = days
            m = portfolio_metrics(
                a["curve"], a["trades"], a["events"], config["capital"], benchmark
            )
            m["open_campaign_net_pnl"] = (
                sum(p["net_pnl"] for p in a["positions"].values()) + m["open_pnl_with_exit_reserve"]
            )
            if abs(m["closed_net_pnl"] + m["open_campaign_net_pnl"] - m["net_pnl"]) > 1e-6:
                raise ValueError("Complete/open lifecycle PnL does not reconcile")
            a["metrics"] = m
            write_account(output / "accounts" / identity, a, config, window)
            checked = audit_account(output, row, bars, rates, scores, lifecycle)
            audits.append(checked)
            row.update(
                metrics=m,
                capacity=a["capacity"],
                risk_halted=a["risk_halted"],
                financial_audit_passed=True,
                lifecycle_uncertainty=cluster_uncertainty(a["trades"]),
            )
            save(output / "accounts" / identity / "row.json", row)
        except Exception as exc:
            failed = output / "accounts" / identity
            failed.mkdir(parents=True, exist_ok=True)
            error = {
                "id": identity,
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "financial_metrics_unknown": True,
                "empty_or_zero_return_substitute": False,
            }
            save(failed / "failure.json", error)
            failures.append(error)
            row.update(metrics=None, error_type=type(exc).__name__, error=str(exc))
        rows.append(row)
        journal.write(__import__("json").dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        journal.flush()
        print(
            {
                "completed_cases": len(rows),
                "planned_cases": len(cases),
                "audited_accounts": len(audits),
                "failed_unknown_accounts": len(failures),
                "last": identity,
                "seconds": round(time.monotonic() - started, 1),
            },
            flush=True,
        )
    journal.close()
    triggers = []
    for candidate in plan["capacity_priority"]:
        cells = [
            r for r in rows if r["phase"] == "continuous2025_2026" and r["candidate"] == candidate
        ]
        if len(cells) == 4 and all(
            r["financial_audit_passed"]
            and not r["risk_halted"]
            and r["metrics"]["net_pnl"] > 0
            and r["metrics"]["closed_net_pnl"] > 0
            for r in cells
        ):
            triggers.append(candidate)
    summary = {
        "schema": "sparse-source-family-accounts.v1",
        "account_rows": rows,
        "planned_cases": len(cases),
        "financial_audit_passed_accounts": len(audits),
        "source_or_audit_unknown_accounts": len(failures),
        "capacity_followup_triggered_candidates": triggers,
        "capacity_followup_queue": triggers[:2],
        "capacity_completed": False,
        "actual_new_process_restoration_completed": False,
        "module_admission_completed": False,
        "total_account_actual": None,
        "profit_goal_met": False,
        "production_eligible": False,
        "orders_enabled": False,
    }
    save(output / "summary.json", summary)
    save(output / "financial_audits.json", audits)
    save(output / "failures.json", failures)
    save(
        output / "completion.json",
        {
            "manifest": {
                str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()
            },
            "profit_goal_met": False,
            "economic_goal_complete": False,
        },
    )
    print(
        {
            "terminal": True,
            "audited": len(audits),
            "unknown": len(failures),
            "capacity_queue": triggers[:2],
            "profit_goal_met": False,
        },
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.inputs, a.output)
