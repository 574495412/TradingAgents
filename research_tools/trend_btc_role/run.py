"""BTC eligibility and a magnitude-matched long control under frozen execution."""

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
from research_tools.max_effect.run_accounts import verify_manifest
from research_tools.regime_program.reporting import cluster_uncertainty
from research_tools.regime_program.storage import write_account
from research_tools.sparse_market.account_audit import audit_account
from research_tools.trend_pool_audit.run import ENGINE, load
from research_tools.trend_stop_audit.run import paired_uncertainty
from research_tools.trend_stop_audit.stop_audit import audit_stops


def calculate(output, identity, engine, bars, rates, scores, config, window, lifecycle, benchmark):
    a = engine.replay(
        bars, rates, scores, config, window["start"], window["end"], lifecycle=lifecycle
    )
    m = portfolio_metrics(a["curve"], a["trades"], a["events"], config["capital"], benchmark)
    m["open_campaign_net_pnl"] = (
        sum(p["net_pnl"] for p in a["positions"].values()) + m["open_pnl_with_exit_reserve"]
    )
    m["mean_absolute_net_exposure"] = float(
        (a["curve"].net_exposure / a["curve"].equity).abs().mean()
    )
    assert abs(m["closed_net_pnl"] + m["open_campaign_net_pnl"] - m["net_pnl"]) < 1e-6
    a["metrics"] = m
    folder = output / "accounts" / identity
    write_account(folder, a, config, window)
    audit = audit_account(output, {"id": identity}, bars, rates, scores, lifecycle)
    save(folder / "internal_cash_audit.json", audit)
    stops = audit_stops(a["events"], a["positions"], bars, config["stop_loss"])
    save(folder / "internal_stop_audit.json", stops)
    if not stops["passed"]:
        raise ValueError("Stop activation failure; preserve unknown economic result")
    return a, {
        "metrics": m,
        "capacity": a["capacity"],
        "risk_halted": a["risk_halted"],
        "financial_audit_passed": True,
        "stop_audit_passed": True,
        "result_sha256": sha(folder / "result.json"),
        "lifecycle_uncertainty": cluster_uncertainty(a["trades"]),
    }


def run(inputs, output):
    output.mkdir(parents=True, exist_ok=False)
    verify_manifest(inputs)
    checked = Path(str(inputs) + "_independent_audit.json")
    check = read(checked)
    assert check["passed"] and check["inputs_completion_sha256"] == sha(inputs / "completion.json")
    prepared = read(inputs / "plan.json")
    for path, digest in prepared["source_hashes"].items():
        assert sha(Path(path)) == digest
    source_paths = [
        Path(__file__),
        ENGINE,
        ENGINE.with_name("sparse_sources_v2.py"),
        Path("research_tools/trend_pool_audit/run.py"),
        Path("research_tools/trend_stop_audit/run.py"),
        Path("research_tools/trend_stop_audit/stop_audit.py"),
        Path("research_tools/sparse_market/account_audit.py"),
        Path("research_tools/factor_lab/metrics.py"),
        Path("research_tools/regime_program/storage.py"),
        checked,
        inputs / "completion.json",
    ]
    plan = {
        **prepared,
        "prepared_inputs": str(inputs),
        "account_source_hashes": {str(p): sha(p) for p in source_paths},
    }
    save(output / "plan.json", plan)
    save(
        output / "registration.json",
        {
            "plan_sha256": sha(output / "plan.json"),
            "before_new_account_returns": True,
            "eligibility_variants": 1,
            "new_economic_mechanisms": 0,
            "source_equivalence": check["source_target_equivalence"],
        },
    )
    _, bars, rates, lifecycle = load(inputs)
    sys.path.insert(0, str(ENGINE.parents[2]))
    engine = executor(str(ENGINE))
    engine.LifecyclePolicy(lifecycle, plan["symbols"])
    dates = pd.date_range(plan["window"]["start"], plan["window"]["end"], freq="D", tz="UTC")
    benchmark = bars["BTCUSDT"].close.reindex(dates - pd.Timedelta(minutes=5))
    benchmark.index = dates
    tasks, aliases = [], {}
    for variant in plan["variants"]:
        for profile in ["btc_tradable", "magnitude_long"]:
            for cost in plan["costs_slippage_bps"]:
                identity = f"{variant}_{profile}_cap10000_slip{cost}"
                if variant != "daily_archive" and check["source_target_equivalence"][profile]:
                    aliases[identity] = f"daily_archive_{profile}_cap10000_slip{cost}"
                else:
                    tasks.append((identity, variant, profile, cost))
    assert len(tasks) <= plan["budget"]["new_base_accounts_max"]
    rows, curves, failures = [], {}, []
    restart_case = None
    started = time.monotonic()
    for identity, variant, profile, cost in tasks:
        config = {**plan["account"], "slippage_bps": cost}
        row = {
            "id": identity,
            "variant": variant,
            "profile": profile,
            "capital": 10000,
            "slippage_bps": cost,
            "reused_account": False,
            "financial_audit_passed": False,
            "production_eligible": False,
            "independent_oos": False,
        }
        target_file = f"{variant}_{profile}_targets.parquet"
        try:
            a, receipt = calculate(
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
            curves[identity] = a["curve"]
            if variant == "daily_archive" and profile == "btc_tradable" and cost == 10:
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
    by_id = {r["id"]: r for r in rows}
    for identity, original in aliases.items():
        rows.append(
            {
                **by_id[original],
                "id": identity,
                "variant": "daily_with_complete_5m_reconciliation",
                "reused_account": True,
                "source_account_id": original,
                "reuse_basis": "exact pre-return input equivalence",
            }
        )
        if original in curves:
            curves[identity] = curves[original]
    parent = Path(plan["inputs"]["parent_accounts"])
    verify_manifest(parent)
    references = []
    for r in read(parent / "summary.json")["account_rows"]:
        if r["domain"] != "pit":
            continue
        original = r.get("source_account_id", r["id"])
        folder = parent / "accounts" / original
        assert sha(folder / "result.json") == r["result_sha256"]
        references.append(
            {
                **r,
                "profile": "btc_excluded",
                "source_run": str(parent),
                "source_account_id": original,
            }
        )
        curves[r["id"]] = pd.read_parquet(folder / "equity_5m.parquet")
    comparisons = []
    for variant in plan["variants"]:
        for cost in plan["costs_slippage_bps"]:
            a = next(
                r
                for r in rows
                if r["variant"] == variant
                and r["profile"] == "btc_tradable"
                and r["slippage_bps"] == cost
            )
            b = next(
                r
                for r in rows
                if r["variant"] == variant
                and r["profile"] == "magnitude_long"
                and r["slippage_bps"] == cost
            )
            c = next(r for r in references if r["variant"] == variant and r["slippage_bps"] == cost)
            if all(r["financial_audit_passed"] for r in [a, b, c]):
                comparisons.append(
                    {
                        "variant": variant,
                        "slippage_bps": cost,
                        "included_minus_excluded_net_pnl": a["metrics"]["net_pnl"]
                        - c["metrics"]["net_pnl"],
                        "trend_minus_magnitude_long_net_pnl": a["metrics"]["net_pnl"]
                        - b["metrics"]["net_pnl"],
                        "role_paired_uncertainty": paired_uncertainty(
                            curves[c["id"]], curves[a["id"]]
                        ),
                        "direction_paired_uncertainty": paired_uncertainty(
                            curves[b["id"]], curves[a["id"]]
                        ),
                    }
                )
    restart = {"performed": False, "reason": "no valid includedtrend stress account"}
    if restart_case:
        a, config, target_file = restart_case
        first = next(e for e in a["events"] if e["type"] == "entry" and e["addition"])
        split = first["time"]
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
        pd.testing.assert_frame_equal(joined, a["curve"])
        assert (
            prefix["events"] + child["events"] == a["events"]
            and prefix["requests"] + child["requests"] == a["requests"]
        )
        assert (
            child["resume_state"] == a["resume_state"]
            and child["positions"] == a["positions"]
            and child["pid"] != os.getpid()
        )
        restart = {
            "performed": True,
            "passed": True,
            "split": split,
            "parent_pid": os.getpid(),
            "worker_pid": child["pid"],
            "all_curve_ledger_inventory_state_equal": True,
        }
    principal = [r for r in rows if r["profile"] == "btc_tradable"]
    provisional = len(comparisons) == 4 and all(
        c["included_minus_excluded_net_pnl"] > 0 and c["trend_minus_magnitude_long_net_pnl"] > 0
        for c in comparisons
    )
    provisional = (
        provisional
        and len(principal) == 4
        and all(
            r["financial_audit_passed"]
            and r["metrics"]["net_pnl"] > 0
            and r["metrics"]["closed_net_pnl"] > 0
            and r["metrics"]["max_drawdown"] <= 0.15
            and not r["risk_halted"]
            for r in principal
        )
    )
    save(
        output / "summary.json",
        {
            "schema": "BTC-role-financial-diagnostic.v1",
            "account_rows": rows,
            "excluded_references": references,
            "comparisons": comparisons,
            "actual_new_financial_accounts": len(tasks),
            "source_duplicate_cases_reused": len(aliases),
            "logical_new_cases": 8,
            "old_reference_source_cost_cases": len(references),
            "failures": failures,
            "actual_new_process_restart": restart,
            "capacity_triggered": provisional,
            "capacity_queue": [
                {"variant": v, "slippage_bps": c, "capital": capital, "profile": "btc_tradable"}
                for v in plan["variants"]
                for c in plan["costs_slippage_bps"]
                for capital in plan["capacity_followup"]["capitals"]
            ]
            if provisional
            else [],
            "capacity_completed": False,
            "external_review": False,
            "independent_unseen_evidence": False,
            "profit_goal_met": False,
            "orders_enabled": False,
            "production_eligible": False,
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
            "new_accounts": len(tasks),
            "failures": len(failures),
            "capacity_triggered": provisional,
            "restart": restart,
        },
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.inputs, a.output)
