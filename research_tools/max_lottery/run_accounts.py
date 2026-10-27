"""Finite full-cost continuous accounts for one preregistered MAX7 mechanism."""

import argparse
import gc
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
    plan = read(inputs / "plan.json")
    for name, expected in plan["source_hashes"].items():
        if sha(Path(name)) != expected:
            raise ValueError("Frozen input changed: " + name)
    summary = read(inputs / "summary.json")
    if summary["raw_eight_date_book_checks"] != 352 or summary["future_perturbation_checks"] != 6:
        raise ValueError("Actual raw prefix or causality checks incomplete")
    bundle, cohort = Path(plan["bundle"]), Path(plan["cohort"])
    for folder in [bundle, cohort]:
        verify_manifest(folder)
    paths = [
        Path(__file__),
        inputs / "completion.json",
        Path(plan["engine"]),
        Path(plan["engine_helper"]),
        Path("research_tools/sparse_market/account_audit.py"),
        Path("research_tools/regime_program/storage.py"),
        Path("research_tools/regime_program/reporting.py"),
        Path("research_tools/factor_lab/metrics.py"),
        Path("research_tools/factor_lab/study.py"),
        bundle / "completion.json",
        cohort / "completion.json",
    ]
    # Source-equivalence grouping is fixed before any new account result.
    cases = []
    for candidate in plan["candidates"]:
        representatives = []
        for variant in plan["source_variants"]:
            path = inputs / f"{variant}_{candidate}_targets.parquet"
            frame = pd.read_parquet(path)
            group = next((r for r in representatives if frame.equals(r["frame"])), None)
            if group is None:
                group = {"candidate": candidate, "variant": variant, "frame": frame, "aliases": []}
                representatives.append(group)
            group["aliases"].append(variant)
        for representative in representatives:
            for cost in plan["costs_slippage_bps"]:
                cases.append(
                    {k: v for k, v in representative.items() if k != "frame"}
                    | {"slippage_bps": cost}
                )
    assert len(cases) <= plan["budget"]["maximum_actual_accounts"]
    save(
        output / "plan.json",
        {
            **plan,
            "inputs": str(inputs),
            "actual_cases": cases,
            "account_source_hashes": {str(p): sha(p) for p in paths},
        },
    )
    save(
        output / "registration.json",
        {
            "wall_ns": time.time_ns(),
            "before_new_account_results": True,
            "plan_sha256": sha(output / "plan.json"),
            "source_grouping_not_selected_by_returns": True,
        },
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
    directives = read(inputs / "lifecycle_directives.json")
    sys.path.insert(0, str(Path(plan["engine"]).parents[2]))
    engine = executor(plan["engine"])
    window = plan["window"]
    started = time.monotonic()
    rows = []
    failures = []
    journal = (output / "progress.jsonl").open("x")
    print(
        {"phase": "sources_loaded", "symbols": len(symbols), "actual_cases": len(cases)}, flush=True
    )
    for case in cases:
        identity = f"{case['variant']}_{case['candidate']}_slip{case['slippage_bps']}"
        row = {
            **case,
            "id": identity,
            "phase": window["id"],
            "capital": plan["account"]["capital"],
            "financial_audit_passed": False,
            "production_eligible": False,
            "independent_oos": False,
        }
        config = {**plan["account"], "slippage_bps": case["slippage_bps"]}
        scores = pd.read_parquet(inputs / f"{case['variant']}_{case['candidate']}_targets.parquet")
        print({"phase": "account_started", "id": identity}, flush=True)
        try:
            account = engine.replay(
                bars, rates, scores, config, window["start"], window["end"], lifecycle=directives
            )
            days = pd.date_range(window["start"], window["end"], freq="D", tz="UTC")
            benchmark = bars["BTCUSDT"].close.reindex(days - pd.Timedelta(minutes=5))
            benchmark.index = days
            metrics = portfolio_metrics(
                account["curve"], account["trades"], account["events"], config["capital"], benchmark
            )
            metrics["open_campaign_net_pnl"] = (
                sum(p["net_pnl"] for p in account["positions"].values())
                + metrics["open_pnl_with_exit_reserve"]
            )
            if (
                abs(
                    metrics["closed_net_pnl"]
                    + metrics["open_campaign_net_pnl"]
                    - metrics["net_pnl"]
                )
                > 1e-6
            ):
                raise ValueError("Closed/open continuous cash does not reconcile")
            metrics["mean_actual_gross_exposure"] = float(
                (account["curve"].gross_exposure / account["curve"].equity).mean()
            )
            metrics["mean_absolute_net_exposure"] = float(
                (account["curve"].net_exposure / account["curve"].equity).abs().mean()
            )
            account["metrics"] = metrics
            write_account(output / "accounts" / identity, account, config, window)
            audit = audit_account(output, row, bars, rates, scores, directives)
            save(output / "accounts" / identity / "internal_cash_audit.json", audit)
            row.update(
                metrics=metrics,
                capacity=account["capacity"],
                risk_halted=account["risk_halted"],
                financial_audit_passed=True,
                lifecycle_uncertainty=cluster_uncertainty(account["trades"]),
            )
            save(output / "accounts" / identity / "row.json", row)
            print(
                {
                    "phase": "account_verified",
                    "id": identity,
                    "net_return": metrics["net_return"],
                    "win_rate": metrics["win_rate"],
                    "drawdown": metrics["max_drawdown"],
                    "sharpe": metrics["sharpe"],
                },
                flush=True,
            )
            del account
            gc.collect()
        except Exception as error:
            folder = output / "accounts" / identity
            folder.mkdir(exist_ok=True, parents=True)
            failure = {
                "id": identity,
                "error_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
                "financial_metrics_unknown": True,
            }
            save(folder / "failure.json", failure)
            failures.append(failure)
            print({"phase": "account_failed", "id": identity, "error": str(error)}, flush=True)
        rows.append(row)
        journal.write(
            __import__("json").dumps(
                {
                    "id": identity,
                    "financial_audit_passed": row["financial_audit_passed"],
                    "elapsed_seconds": time.monotonic() - started,
                }
            )
            + "\n"
        )
        journal.flush()
    journal.close()
    save(
        output / "summary.json",
        {
            "rows": rows,
            "failures": failures,
            "actual_accounts": len(cases),
            "logical_source_policy_cost_cases": len(plan["candidates"])
            * len(plan["source_variants"])
            * len(plan["costs_slippage_bps"]),
            "new_economic_variants": 1,
            "third_party_review_performed": False,
            "independent_oos": False,
            "profit_goal_met": False,
        },
    )
    save(
        output / "completion.json",
        {
            "manifest": {
                str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()
            },
            "profit_goal_met": False,
            "orders_enabled": False,
        },
    )
    print(
        {
            "complete": str(output),
            "actual_accounts": len(cases),
            "failures": len(failures),
            "elapsed_seconds": time.monotonic() - started,
        },
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.inputs, args.output)
