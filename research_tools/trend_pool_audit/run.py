"""Matched-domain accounts; exact source-target duplicates reuse one economic path."""

import argparse
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.factor_lab.metrics import portfolio_metrics
from research_tools.factor_lab.study import executor
from research_tools.funding_pool_bridge.prepare import old_bars
from research_tools.max_effect.run_accounts import verify_manifest
from research_tools.regime_program.reporting import cluster_uncertainty
from research_tools.regime_program.storage import write_account
from research_tools.sparse_market.account_audit import audit_account
from research_tools.trend_stop_audit.run import paired_uncertainty
from research_tools.trend_stop_audit.stop_audit import audit_stops

ENGINE = Path("../freqtrade/research/crypto/sparse_calendar_v2.py").resolve()


def load(inputs):
    plan = read(inputs / "plan.json")
    begin = pd.Timestamp(plan["window"]["start"], tz="UTC") - pd.Timedelta(hours=1)
    end = pd.Timestamp(plan["window"]["end"], tz="UTC") - pd.Timedelta(minutes=5)
    bars, rates = {}, {}
    for s, binding in read(inputs / "execution_bindings.json").items():
        for key in ["bar", "funding"]:
            assert sha(Path(binding[key + "_path"])) == binding[key + "_sha256"]
        p = Path(binding["bar_path"])
        bars[s] = (
            pd.read_parquet(
                p,
                columns=[
                    "open",
                    "high",
                    "low",
                    "close",
                    "quote_volume",
                    "price_usable",
                    "execution_liquidity_usable",
                ],
            ).loc[begin:end]
            if binding["from_bundle"]
            else old_bars(p, begin, end)
        )
        rates[s] = pd.read_parquet(binding["funding_path"])
    return plan, bars, rates, read(inputs / "lifecycle.json")


def worker(output):
    p = read(output / "restart_registration.json")
    inputs = Path(read(output / "plan.json")["prepared_inputs"])
    plan, bars, rates, lifecycle = load(inputs)
    assert sha(ENGINE) == p["engine_sha256"]
    sys.path.insert(0, str(ENGINE.parents[2]))
    a = executor(str(ENGINE)).replay(
        bars,
        rates,
        pd.read_parquet(inputs / p["target_file"]),
        p["config"],
        p["split"],
        plan["window"]["end"],
        initial_state=read(output / "restart_checkpoint.json"),
        lifecycle=lifecycle,
    )
    a["curve"].to_parquet(output / "restart_worker_curve.parquet", index=False)
    save(
        output / "restart_worker.json",
        {
            "pid": os.getpid(),
            "events": a["events"],
            "requests": a["requests"],
            "positions": a["positions"],
            "resume_state": a["resume_state"],
        },
    )


def run(inputs, output):
    output.mkdir(parents=True, exist_ok=False)
    verify_manifest(inputs)
    checked = Path(str(inputs) + "_independent_audit.json")
    check = read(checked)
    assert check["passed"] and check["inputs_completion_sha256"] == sha(inputs / "completion.json")
    prepared = read(inputs / "plan.json")
    for path, digest in prepared["source_hashes"].items():
        assert sha(Path(path)) == digest, path
    paths = [
        Path(__file__),
        ENGINE,
        ENGINE.with_name("sparse_sources_v2.py"),
        Path("research_tools/trend_stop_audit/stop_audit.py"),
        Path("research_tools/trend_stop_audit/run.py"),
        Path("research_tools/sparse_market/account_audit.py"),
        Path("research_tools/factor_lab/metrics.py"),
        Path("research_tools/regime_program/storage.py"),
        checked,
        inputs / "completion.json",
    ]
    plan = {
        **prepared,
        "prepared_inputs": str(inputs),
        "account_source_hashes": {str(p): sha(p) for p in paths},
        "uncertainty_registration": {
            "paired_block_days": [32, 128],
            "seed": 17,
            "replications": 2000,
            "minimum_block_equivalents": 8,
        },
    }
    save(output / "plan.json", plan)
    save(
        output / "registration.json",
        {
            "plan_sha256": sha(output / "plan.json"),
            "before_any_new_account_returns": True,
            "source_target_equivalence_before_returns": check[
                "source_paired_targets_exactly_equal"
            ],
            "economic_variants": 0,
            "new_formula_parameters": 0,
        },
    )
    plan0, bars, rates, lifecycle = load(inputs)
    sys.path.insert(0, str(ENGINE.parents[2]))
    engine = executor(str(ENGINE))
    engine.LifecyclePolicy(lifecycle, plan["symbols"])
    aliases, tasks = {}, []
    for variant in plan["source_variants"]:
        for domain in ["fixed40", "pit"]:
            for cost in plan["costs_slippage_bps"]:
                identity = f"{variant}_{domain}_cap10000_slip{cost}"
                if (
                    variant != "daily_archive"
                    and check["source_paired_targets_exactly_equal"][domain]
                ):
                    aliases[identity] = f"daily_archive_{domain}_cap10000_slip{cost}"
                else:
                    tasks.append(
                        (identity, variant, domain, cost, f"{variant}_{domain}_targets.parquet")
                    )
    if read(inputs / "summary.json")["conditional_source_bridge_required"]:
        for cost in plan["costs_slippage_bps"]:
            tasks.insert(
                0,
                (
                    f"oldtarget_source_bridge_cap10000_slip{cost}",
                    "bridge",
                    "legacy",
                    cost,
                    "legacy_targets.parquet",
                ),
            )
    assert len(tasks) <= plan["budget"]["max_total_new_base_accounts"]
    dates = pd.date_range(plan["window"]["start"], plan["window"]["end"], freq="D", tz="UTC")
    benchmark = bars["BTCUSDT"].close.reindex(dates - pd.Timedelta(minutes=5))
    benchmark.index = dates
    rows, curves, failures, restart_case = [], {}, [], None
    started = time.monotonic()
    for identity, variant, domain, cost, target_file in tasks:
        row = {
            "id": identity,
            "variant": variant,
            "domain": domain,
            "slippage_bps": cost,
            "capital": 10000,
            "reused_account": False,
            "financial_audit_passed": False,
            "stop_audit_passed": False,
            "independent_oos": False,
            "production_eligible": False,
        }
        config = {**plan["account"], "slippage_bps": cost}
        scores = pd.read_parquet(inputs / target_file)
        try:
            a = engine.replay(
                bars,
                rates,
                scores,
                config,
                plan["window"]["start"],
                plan["window"]["end"],
                lifecycle=lifecycle,
            )
            m = portfolio_metrics(
                a["curve"], a["trades"], a["events"], config["capital"], benchmark
            )
            m["open_campaign_net_pnl"] = (
                sum(p["net_pnl"] for p in a["positions"].values()) + m["open_pnl_with_exit_reserve"]
            )
            m["mean_absolute_net_exposure"] = float(
                (a["curve"].net_exposure / a["curve"].equity).abs().mean()
            )
            assert abs(m["closed_net_pnl"] + m["open_campaign_net_pnl"] - m["net_pnl"]) < 1e-6
            a["metrics"] = m
            folder = output / "accounts" / identity
            write_account(folder, a, config, plan["window"])
            audit = audit_account(output, row, bars, rates, scores, lifecycle)
            save(folder / "internal_cash_audit.json", audit)
            stops = audit_stops(a["events"], a["positions"], bars, config["stop_loss"])
            save(folder / "internal_stop_audit.json", stops)
            if not stops["passed"]:
                raise ValueError(
                    "Stop activation referenced a preactivation bar; caseunknown pending timing repair"
                )
            row.update(
                {
                    "financial_audit_passed": True,
                    "stop_audit_passed": True,
                    "metrics": m,
                    "capacity": a["capacity"],
                    "risk_halted": a["risk_halted"],
                    "stop_trace": stops,
                    "lifecycle_uncertainty": cluster_uncertainty(a["trades"]),
                    "result_sha256": sha(folder / "result.json"),
                }
            )
            curves[identity] = a["curve"]
            if variant == "daily_archive" and domain == "pit" and cost == 10:
                restart_case = (a, config, target_file)
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
        rows.append(row)
        folder = output / "accounts" / identity
        save(folder / "row.json" if folder.exists() else output / (identity + "_failure.json"), row)
        print(
            {
                "id": identity,
                "audited": row["financial_audit_passed"],
                "net_return": row.get("metrics", {}).get("net_return"),
                "seconds": round(time.monotonic() - started, 2),
            },
            flush=True,
        )
    by_id = {r["id"]: r for r in rows}
    for identity, original in aliases.items():
        row = {
            **by_id[original],
            "id": identity,
            "variant": "daily_with_complete_5m_reconciliation",
            "reused_account": True,
            "source_account_id": original,
            "reuse_basis": "exact pre-return target/config/sharedsource/lifecycle equality",
        }
        rows.append(row)
        if original in curves:
            curves[identity] = curves[original]
    comparisons = []
    for variant in plan["source_variants"]:
        for cost in plan["costs_slippage_bps"]:
            old_id, pit_id = [f"{variant}_{d}_cap10000_slip{cost}" for d in ["fixed40", "pit"]]
            old, pit = [next(r for r in rows if r["id"] == i) for i in [old_id, pit_id]]
            if old["financial_audit_passed"] and pit["financial_audit_passed"]:
                comparisons.append(
                    {
                        "variant": variant,
                        "slippage_bps": cost,
                        "PIT_minus_matched40_net_pnl": pit["metrics"]["net_pnl"]
                        - old["metrics"]["net_pnl"],
                        "paired_uncertainty": paired_uncertainty(curves[old_id], curves[pit_id]),
                    }
                )
    restart = {"performed": False, "reason": "no valid principal PITstress account"}
    if restart_case:
        a, config, target_file = restart_case
        event = next(e for e in a["events"] if e["type"] == "entry" and e["addition"])
        split = event["time"]
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
        combined = pd.concat(
            [prefix["curve"].iloc[:-1], pd.read_parquet(output / "restart_worker_curve.parquet")],
            ignore_index=True,
        )
        pd.testing.assert_frame_equal(combined, a["curve"])
        assert (
            prefix["events"] + child["events"] == a["events"]
            and prefix["requests"] + child["requests"] == a["requests"]
        )
        assert child["positions"] == a["positions"] and child["resume_state"] == a["resume_state"]
        assert child["pid"] != os.getpid()
        restart = {
            "performed": True,
            "passed": True,
            "split": split,
            "parent_pid": os.getpid(),
            "worker_pid": child["pid"],
            "curve_cash_inventory_requests_full_state_equal": True,
        }
    pit_rows = [r for r in rows if r["domain"] == "pit"]
    provisional = len(pit_rows) == 4 and all(
        r["financial_audit_passed"]
        and r["metrics"]["net_pnl"] > 0
        and r["metrics"]["closed_net_pnl"] > 0
        and r["metrics"]["max_drawdown"] <= 0.15
        and not r["risk_halted"]
        for r in pit_rows
    )
    save(
        output / "summary.json",
        {
            "schema": "trend-pit-domain-accounts.v1",
            "account_rows": rows,
            "actual_new_base_accounts": len(tasks),
            "logical_source_domain_cost_cases": 8,
            "exact_source_duplicate_cases_reused": len(aliases),
            "new_economic_variants": 0,
            "failures": failures,
            "comparisons": comparisons,
            "actual_new_process_restart": restart,
            "capacity_triggered": provisional,
            "capacity_queue": [
                {"variant": v, "slippage_bps": c, "capital": capital, "domain": "pit"}
                for v in plan["source_variants"]
                for c in plan["costs_slippage_bps"]
                for capital in plan["capacity_followup"]["capitals"]
            ]
            if provisional
            else [],
            "capacity_completed": False,
            "profit_goal_met": False,
            "orders_enabled": False,
            "production_eligible": False,
            "external_review": False,
            "independent_unseen_evidence": False,
        },
    )
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
            "actual_new_accounts": len(tasks),
            "failures": len(failures),
            "capacity_triggered": provisional,
            "restart": restart,
        },
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    worker(args.worker) if args.worker else run(args.inputs, args.output)
