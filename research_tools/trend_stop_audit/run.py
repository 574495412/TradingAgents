"""Five frozen bridge/stop cases, no new signal or parameter search."""

import argparse
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.factor_lab.metrics import portfolio_metrics
from research_tools.factor_lab.study import executor
from research_tools.funding_pool_bridge.prepare import old_bars
from research_tools.max_effect.run_accounts import verify_manifest
from research_tools.regime_program.reporting import cluster_uncertainty, trade_stats
from research_tools.regime_program.storage import write_account
from research_tools.sparse_market.account_audit import audit_account
from research_tools.trend_stop_audit.stop_audit import audit_stops


def market(plan):
    a = pd.Timestamp(plan["window"]["start"], tz="UTC") - pd.Timedelta(hours=1)
    b = pd.Timestamp(plan["window"]["end"], tz="UTC")
    root = Path(plan["inputs"]["old_market"])
    bars = {s: old_bars(root / (s + "_5m.parquet"), a, b) for s in plan["symbols"]}
    return bars


def rates_for(plan, kind):
    root = Path(plan["inputs"]["exact_funding" if kind == "exact" else "old_market"])
    suffix = "_exact.parquet" if kind == "exact" else "_funding.parquet"
    return {s: pd.read_parquet(root / (s + suffix)) for s in plan["symbols"]}


def stop_stats(trades):
    by_reason = {}
    for reason in sorted({t["reason"] for t in trades}):
        subset = [t for t in trades if t["reason"] == reason]
        durations = [
            (pd.Timestamp(t["exit_time"]) - pd.Timestamp(t["entry_time"])).total_seconds() / 86400
            for t in subset
        ]
        by_reason[reason] = {
            **trade_stats(subset),
            "median_holding_days": float(np.median(durations)),
            "max_holding_days": float(max(durations)),
        }
    return by_reason


def worker(run):
    registration = read(run / "restart_registration.json")
    plan = read(run / "plan.json")
    inputs = Path(plan["prepared_inputs"])
    config = registration["config"]
    sys.path.insert(0, str(Path("../freqtrade").resolve()))
    engine_path = Path("../freqtrade/research/crypto/sparse_coststop_v1.py").resolve()
    assert sha(engine_path) == registration["engine_sha256"]
    account = executor(str(engine_path)).replay(
        market(plan),
        rates_for(plan, "exact"),
        pd.read_parquet(inputs / "targets.parquet"),
        config,
        registration["split"],
        plan["window"]["end"],
        initial_state=read(run / "restart_checkpoint.json"),
    )
    account["curve"].to_parquet(run / "restart_worker_curve.parquet", index=False)
    save(
        run / "restart_worker.json",
        {
            "pid": os.getpid(),
            "events": account["events"],
            "requests": account["requests"],
            "positions": account["positions"],
            "resume_state": account["resume_state"],
        },
    )


def paired_uncertainty(c, d):
    a, b = c.set_index("date").equity, d.set_index("date").equity
    a = a.loc[(a.index.hour == 0) & (a.index.minute == 0)].pct_change().dropna()
    b = b.loc[(b.index.hour == 0) & (b.index.minute == 0)].pct_change().dropna()
    diff = (b - a).to_numpy()
    intervals = []
    for block in [32, 128]:
        effective = len(diff) // block
        interval = None
        if effective >= 8:
            rng = np.random.default_rng(17)
            means = []
            for _ in range(2000):
                starts = rng.integers(0, len(diff), size=math.ceil(len(diff) / block))
                indexes = ((starts[:, None] + np.arange(block)) % len(diff)).ravel()[: len(diff)]
                means.append(float(np.mean(diff[indexes])))
            interval = np.quantile(means, [0.025, 0.975]).tolist()
        intervals.append(
            {
                "block_days": block,
                "full_block_equivalents": effective,
                "mean_daily_return_difference95": interval,
                "insufficient_if_less_than8blocks": effective < 8,
            }
        )
    return {
        "mean_daily_return_D_minus_C": float(diff.mean()),
        "paired_calendar_days": len(diff),
        "intervals": intervals,
        "scope": "Fixed32/128day circular blocks,seed17,2000replications. Knownhistory, haltedzero days retained; no search-bias correction, significance or future claim.",
    }


def run(inputs, output):
    output.mkdir(parents=True, exist_ok=False)
    verify_manifest(inputs)
    plan = read(inputs / "plan.json")
    for path, digest in plan["source_hashes"].items():
        if sha(Path(path)) != digest:
            raise ValueError("Frozen source changed: " + path)
    for binding in read(inputs / "market_bindings.json").values():
        for path, digest in binding.items():
            if sha(Path(path)) != digest:
                raise ValueError("Frozen market changed: " + path)
    source_paths = [
        Path(__file__),
        Path(__file__).with_name("stop_audit.py"),
        Path("research_tools/sparse_market/account_audit.py"),
        Path("research_tools/factor_lab/metrics.py"),
        Path("research_tools/regime_program/storage.py"),
        Path("research_tools/regime_program/reporting.py"),
        inputs / "completion.json",
    ]
    plan = {
        **plan,
        "prepared_inputs": str(inputs),
        "account_source_hashes": {str(p): sha(p) for p in source_paths},
        "uncertainty_preregistration": {
            "paired_daily_return_difference": True,
            "circular_block_days": [32, 128],
            "bootstrap_replications": 2000,
            "seed": 17,
            "minimum_full_block_equivalents": 8,
        },
    }
    save(output / "plan.json", plan)
    save(
        output / "registration.json",
        {
            "before_any_new_account_returns": True,
            "plan_sha256": sha(output / "plan.json"),
            "maximum_new_base_accounts": 5,
            "economic_variants": 1,
        },
    )
    sys.path.insert(0, str(Path("../freqtrade").resolve()))
    paths = {
        "original": Path("../freqtrade/research/crypto/sparse_calendar_v2.py").resolve(),
        "costanchor": Path("../freqtrade/research/crypto/sparse_coststop_v1.py").resolve(),
    }
    engines = {k: executor(str(v)) for k, v in paths.items()}
    bars, rates = market(plan), {k: rates_for(plan, k) for k in ["legacy", "exact"]}
    scores = pd.read_parquet(inputs / "targets.parquet")
    calendar_days = pd.date_range(
        plan["window"]["start"], plan["window"]["end"], freq="D", tz="UTC"
    )
    benchmark = bars["BTCUSDT"].close.reindex(calendar_days - pd.Timedelta(minutes=5))
    benchmark.index = calendar_days
    cases = [
        ("B_execution_stress", "original", "legacy", 10, False),
        ("C_exact_base", "original", "exact", 3, False),
        ("C_exact_stress", "original", "exact", 10, False),
        ("D_costanchor_base", "costanchor", "exact", 3, True),
        ("D_costanchor_stress", "costanchor", "exact", 10, True),
    ]
    rows, failures, curves, full_for_restart = [], [], {}, None
    journal = (output / "progress.jsonl").open("x")
    started = time.monotonic()
    for identity, engine_name, source, cost, weighted in cases:
        config = {**plan["account"], "slippage_bps": cost}
        if weighted:
            config["weighted_cost_stop"] = True
        row = {
            "id": identity,
            "candidate": "weighted_cost_anchor" if weighted else "fill_ratchet",
            "funding_source": source,
            "slippage_bps": cost,
            "capital": config["capital"],
            "financial_audit_passed": False,
            "stop_audit_passed": False,
            "production_eligible": False,
            "independent_oos": False,
        }
        try:
            a = engines[engine_name].replay(
                bars, rates[source], scores, config, plan["window"]["start"], plan["window"]["end"]
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
            if abs(m["closed_net_pnl"] + m["open_campaign_net_pnl"] - m["net_pnl"]) > 1e-6:
                raise ValueError("Complete/open lifecycle does not reconcile")
            a["metrics"] = m
            folder = output / "accounts" / identity
            write_account(folder, a, config, plan["window"])
            audit = audit_account(output, row, bars, rates[source], scores)
            save(folder / "internal_cash_audit.json", audit)
            stop = audit_stops(a["events"], a["positions"], bars, config["stop_loss"], weighted)
            save(folder / "internal_stop_audit.json", stop)
            if not stop["passed"]:
                raise ValueError(
                    "Actual newly activated stop referenced an earlier bar; economic case invalid pending timing repair"
                )
            row.update(
                {
                    "metrics": m,
                    "capacity": a["capacity"],
                    "risk_halted": a["risk_halted"],
                    "financial_audit_passed": True,
                    "stop_audit_passed": True,
                    "lifecycle_uncertainty": cluster_uncertainty(a["trades"]),
                    "by_exit_reason": stop_stats(a["trades"]),
                    "stop_trace": stop,
                    "result_sha256": sha(folder / "result.json"),
                }
            )
            curves[identity] = a["curve"]
            if identity == "D_costanchor_stress":
                full_for_restart = a
        except Exception as error:
            failures.append(
                {
                    "id": identity,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "economic_result_unknown": True,
                }
            )
            row["failure"] = str(error)
        rows.append(row)
        journal.write(
            __import__("json").dumps(
                {
                    "id": identity,
                    "audited": row["financial_audit_passed"],
                    "net_return": row.get("metrics", {}).get("net_return"),
                    "elapsed_seconds": round(time.monotonic() - started, 2),
                }
            )
            + "\n"
        )
        journal.flush()
        print(
            {
                "id": identity,
                "audited": row["financial_audit_passed"],
                "net_return": row.get("metrics", {}).get("net_return"),
                "seconds": round(time.monotonic() - started, 2),
            },
            flush=True,
        )
        save(
            output / "accounts" / identity / "row.json"
            if (output / "accounts" / identity).exists()
            else output / (identity + "_failure.json"),
            row,
        )
    journal.close()
    comparisons = []
    for label in ["base", "stress"]:
        c, d = (
            next(r for r in rows if r["id"] == "C_exact_" + label),
            next(r for r in rows if r["id"] == "D_costanchor_" + label),
        )
        if c["financial_audit_passed"] and d["financial_audit_passed"]:
            comparisons.append(
                {
                    "cost": label,
                    "D_minus_C_net_pnl": d["metrics"]["net_pnl"] - c["metrics"]["net_pnl"],
                    "D_minus_C_fees": d["metrics"]["fees"] - c["metrics"]["fees"],
                    "D_minus_C_funding": d["metrics"]["funding_net"] - c["metrics"]["funding_net"],
                    "paired_uncertainty": paired_uncertainty(curves[c["id"]], curves[d["id"]]),
                }
            )
    restart = {"performed": False, "reason": "no valid weighted stress account"}
    if full_for_restart:
        addition = next(
            e for e in full_for_restart["events"] if e["type"] == "entry" and e["addition"]
        )
        split = addition["time"]
        config = {**plan["account"], "slippage_bps": 10, "weighted_cost_stop": True}
        prefix = engines["costanchor"].replay(
            bars, rates["exact"], scores, config, plan["window"]["start"], split
        )
        assert prefix["positions"]
        save(output / "restart_checkpoint.json", prefix["resume_state"])
        save(
            output / "restart_registration.json",
            {
                "split": split,
                "config": config,
                "engine_sha256": sha(paths["costanchor"]),
                "parent_pid": os.getpid(),
                "same_account_not_new_trial": True,
            },
        )
        subprocess.run(
            [sys.executable, "-m", "research_tools.trend_stop_audit.run", "--worker", str(output)],
            check=True,
        )
        restored = read(output / "restart_worker.json")
        joined = pd.concat(
            [prefix["curve"].iloc[:-1], pd.read_parquet(output / "restart_worker_curve.parquet")],
            ignore_index=True,
        )
        pd.testing.assert_frame_equal(joined, full_for_restart["curve"])
        assert prefix["events"] + restored["events"] == full_for_restart["events"]
        assert prefix["requests"] + restored["requests"] == full_for_restart["requests"]
        assert (
            restored["positions"] == full_for_restart["positions"]
            and restored["resume_state"] == full_for_restart["resume_state"]
        )
        assert restored["pid"] != os.getpid()
        restart = {
            "performed": True,
            "passed": True,
            "holding_split": split,
            "parent_pid": os.getpid(),
            "worker_pid": restored["pid"],
            "full_curve_cash_ledger_inventory_state_equal": True,
        }
    eligible = len(comparisons) == 2 and all(c["D_minus_C_net_pnl"] > 0 for c in comparisons)
    ds = [r for r in rows if r["candidate"] == "weighted_cost_anchor"]
    eligible = eligible and all(
        r["financial_audit_passed"]
        and r["metrics"]["net_pnl"] > 0
        and r["metrics"]["closed_net_pnl"] > 0
        and r["metrics"]["max_drawdown"] <= 0.15
        and not r["risk_halted"]
        for r in ds
    )
    summary = {
        "schema": "trend-coststop-accounts.v1",
        "account_rows": rows,
        "planned_new_base_accounts": len(cases),
        "audited_accounts": sum(r["financial_audit_passed"] for r in rows),
        "failures": failures,
        "comparisons": comparisons,
        "actual_new_process_restart": restart,
        "capacity_triggered": eligible,
        "capacity_queue": [
            {"capital": cap, "slippage_bps": cost, "candidate": "weighted_cost_anchor"}
            for cap in plan["capacity_followup"]["capitals"]
            for cost in plan["costs_slippage_bps"]
        ]
        if eligible
        else [],
        "capacity_completed": False,
        "external_review": False,
        "independent_unseen_evidence": False,
        "fixed40_universe_blocks_alpha_admission": True,
        "profit_goal_met": False,
        "orders_enabled": False,
        "production_eligible": False,
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
            "financial_cases_attempted": len(cases),
        },
    )
    print(
        {
            "audited_accounts": summary["audited_accounts"],
            "failures": len(failures),
            "capacity_triggered": eligible,
            "restart": restart,
        },
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--worker", type=Path)
    a = p.parse_args()
    worker(a.worker) if a.worker else run(a.inputs, a.output)
