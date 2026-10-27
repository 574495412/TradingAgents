"""One explicit holding-policy counterfactual with exact legacy-engine bridges."""

import argparse
import copy
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
from research_tools.trend_pool_audit.run import load
from research_tools.trend_stop_audit.run import paired_uncertainty

ENGINE = Path("../freqtrade/research/crypto/monthly_hold_v1.py").resolve()


def worker(output):
    plan = read(output / "plan.json")
    for filename, digest in plan["source_hashes"].items():
        assert sha(Path(filename)) == digest, filename
    targets = Path(plan["targets"])
    _, bars, rates, lifecycle = load(targets)
    sys.path.insert(0, str(ENGINE.parents[2]))
    r = read(output / "restart_registration.json")
    account = executor(str(ENGINE)).replay(
        bars,
        rates,
        pd.read_parquet(targets / r["target_file"]),
        r["config"],
        r["split"],
        plan["window"]["end"],
        initial_state=read(output / "restart_checkpoint.json"),
        lifecycle=lifecycle,
    )
    account["curve"].to_parquet(output / "restart_worker_curve.parquet", index=False)
    save(
        output / "restart_worker.json",
        {
            "pid": os.getpid(),
            "events": account["events"],
            "requests": account["requests"],
            "positions": account["positions"],
            "resume_state": account["resume_state"],
        },
    )


def run(contract, output):
    output.mkdir(parents=True, exist_ok=False)
    verify_manifest(contract)
    assert read(contract / "summary.json")["passed"]
    for name, digest in read(contract / "registration.json")["source_hashes"].items():
        assert sha(Path(name)) == digest, name
    p = Path(__file__).with_name("plan.json")
    plan = read(p)
    targets, reference = Path(plan["targets"]), Path(plan["reference_run"])
    for root in [targets, reference]:
        verify_manifest(root)
    target_plan = read(targets / "plan.json")
    for name, digest in target_plan["source_hashes"].items():
        assert sha(Path(name)) == digest, name
    paths = [
        p,
        Path(__file__),
        ENGINE,
        ENGINE.with_name("sparse_sources_v2.py"),
        ENGINE.with_name("sparse_calendar_v2.py"),
        Path("research_tools/trend_btc_role/run.py"),
        Path("research_tools/trend_pool_audit/run.py"),
        Path("research_tools/trend_stop_audit/run.py"),
        Path("research_tools/trend_stop_audit/stop_audit.py"),
        Path("research_tools/sparse_market/account_audit.py"),
        Path("research_tools/factor_lab/metrics.py"),
        Path("research_tools/regime_program/storage.py"),
        contract / "completion.json",
        targets / "completion.json",
        reference / "completion.json",
    ]
    plan = {
        **plan,
        "contract": str(contract),
        "account": target_plan["account"],
        "source_hashes": {str(path): sha(path) for path in paths},
    }
    save(output / "plan.json", plan)
    save(
        output / "registration.json",
        {
            "plan_sha256": sha(output / "plan.json"),
            "before_new_policy_and_bridge_returns": True,
            "known_history_only": True,
            "new_policy_variants": 1,
        },
    )
    _, bars, rates, lifecycle = load(targets)
    sys.path.insert(0, str(ENGINE.parents[2]))
    engine = executor(str(ENGINE))
    dates = pd.date_range(plan["window"]["start"], plan["window"]["end"], freq="D", tz="UTC")
    benchmark = bars["BTCUSDT"].close.reindex(dates - pd.Timedelta(minutes=5))
    benchmark.index = dates
    rows, curves, bridge_checks, failures = [], {}, [], []
    restart_case = None
    started = time.monotonic()
    # Default-enabled controls must reproduce original account paths exactly.
    for cost in plan["costs_slippage_bps"]:
        original = f"daily_archive_monthly_tsmom_cap10000_slip{cost}"
        identity = f"bridge_enabled_monthly_tsmom_cap10000_slip{cost}"
        config = {**plan["account"], "slippage_bps": cost}
        f = pd.read_parquet(targets / "daily_archive_monthly_tsmom_targets.parquet")
        a, receipt = calculate(
            output, identity, engine, bars, rates, f, config, plan["window"], lifecycle, benchmark
        )
        old = reference / "accounts" / original
        pd.testing.assert_frame_equal(
            a["curve"], pd.read_parquet(old / "equity_5m.parquet"), check_exact=True
        )
        for key in ["events", "requests", "trades"]:
            assert a[key] == read(old / (key + ".json")), key
        prior = read(old / "resume_state.json")
        normalized = copy.deepcopy(a["resume_state"])
        normalized["engine_version"] = prior["engine_version"]
        assert normalized == prior
        assert a["metrics"] == read(old / "result.json")["metrics"]
        bridge_checks.append(
            {
                "id": identity,
                "original": original,
                "full_curve_ledger_requests_campaigns_inventory_risk_exact": True,
                "engine_identity_only_state_difference": True,
                "result_sha256": receipt["result_sha256"],
            }
        )
        print(
            {"bridge": identity, "exact": True, "seconds": round(time.monotonic() - started, 2)},
            flush=True,
        )

    def perform(profile, cost, capital):
        nonlocal restart_case
        identity = f"disabled_{profile}_cap{capital}_slip{cost}"
        config = {
            **plan["account"],
            "slippage_bps": cost,
            "capital": capital,
            "price_stop_enabled": False,
        }
        target_file = f"daily_archive_{profile}_targets.parquet"
        row = {
            "id": identity,
            "profile": profile,
            "slippage_bps": cost,
            "capital": capital,
            "variant": "daily_archive",
            "reused_account": False,
            "financial_audit_passed": False,
            "production_eligible": False,
            "independent_oos": False,
        }
        try:
            a, receipt = calculate(
                output,
                identity,
                engine,
                bars,
                rates,
                pd.read_parquet(targets / target_file),
                config,
                plan["window"],
                lifecycle,
                benchmark,
            )
            assert not any(e.get("reason") == "stop" for e in a["events"])
            row.update(receipt)
            row["price_stop_disabled_verified"] = True
            curves[identity] = a["curve"]
            if profile == "monthly_tsmom" and cost == 10 and capital == 10000:
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

    for profile in plan["profiles"]:
        for cost in plan["costs_slippage_bps"]:
            perform(profile, cost, 10000)
    old_rows = {r["id"]: r for r in read(reference / "summary.json")["account_rows"]}
    comparisons = []
    for cost in plan["costs_slippage_bps"]:
        a = next(r for r in rows if r["profile"] == "monthly_tsmom" and r["slippage_bps"] == cost)
        b = next(r for r in rows if r["profile"] == "magnitude_long" and r["slippage_bps"] == cost)
        original = f"daily_archive_monthly_tsmom_cap10000_slip{cost}"
        if a["financial_audit_passed"] and b["financial_audit_passed"]:
            comparisons.append(
                {
                    "slippage_bps": cost,
                    "disabled_trend_minus_long_net_pnl": a["metrics"]["net_pnl"]
                    - b["metrics"]["net_pnl"],
                    "disabled_minus_enabled_trend_net_pnl": a["metrics"]["net_pnl"]
                    - old_rows[original]["metrics"]["net_pnl"],
                    "direction_paired_uncertainty": paired_uncertainty(
                        curves[b["id"]], curves[a["id"]]
                    ),
                    "policy_paired_uncertainty": paired_uncertainty(
                        pd.read_parquet(reference / "accounts" / original / "equity_5m.parquet"),
                        curves[a["id"]],
                    ),
                }
            )
    principal = [r for r in rows if r["profile"] == "monthly_tsmom"]
    followup = len(comparisons) == 2 and all(
        c["disabled_trend_minus_long_net_pnl"] > 0 for c in comparisons
    )
    followup = followup and all(
        r["financial_audit_passed"]
        and r["metrics"]["net_pnl"] > 0
        and r["metrics"]["closed_net_pnl"] > 0
        and r["metrics"]["max_drawdown"] <= 0.15
        and not r["risk_halted"]
        for r in principal
    )
    if followup:
        for capital in plan["capacity"]["capitals"]:
            for cost in plan["costs_slippage_bps"]:
                perform("monthly_tsmom", cost, capital)
    case = plan["selected_case"]
    hours = pd.date_range(case["start"], case["end"], freq="h", tz="UTC", inclusive="left")
    f = pd.read_parquet(targets / "daily_archive_monthly_tsmom_targets.parquet").reindex(hours)[
        ["target_TONUSDT", "risk_scale", "rebalance"]
    ]
    config = {**plan["account"], "slippage_bps": case["slippage_bps"], "price_stop_enabled": False}
    w = {"id": "selected_TON_exit", "start": case["start"], "end": case["end"]}
    dates_case = pd.date_range(case["start"], case["end"], freq="D", tz="UTC")
    case_benchmark = bars["BTCUSDT"].close.reindex(dates_case - pd.Timedelta(minutes=5))
    case_benchmark.index = dates_case
    account, receipt = calculate(
        output,
        "selected_TON_exit",
        engine,
        {"TONUSDT": bars["TONUSDT"]},
        {"TONUSDT": rates["TONUSDT"]},
        f,
        config,
        w,
        {"TONUSDT": lifecycle["TONUSDT"]},
        case_benchmark,
    )
    exits = [e for e in account["events"] if e["type"] == "exit"]
    assert exits and all(e["reason"] == "contract_exit" for e in exits) and not account["positions"]
    assert all(pd.Timestamp(e["time"]) < pd.Timestamp("2026-06-23T09:00:00Z") for e in exits)
    save(
        output / "selected_case.json",
        {
            "case": case,
            "receipt": receipt,
            "exits": exits,
            "residual_positions": account["positions"],
            "selected_known_case_not_strategy_account": True,
            "terminal_settlement_invented": False,
        },
    )
    restart = {"performed": False}
    if restart_case:
        a, config, target_file = restart_case
        entries = [e for e in a["events"] if e["type"] == "entry"]
        chosen = next((e for e in entries if e.get("addition")), entries[0])
        split = chosen["time"]
        prefix = engine.replay(
            bars,
            rates,
            pd.read_parquet(targets / target_file),
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
                "parent_pid": os.getpid(),
            },
        )
        rejection = []
        for changed, initial, expected in [
            ({**config, "price_stop_enabled": True}, prefix["resume_state"], "configuration"),
            (config, {**prefix["resume_state"], "engine_version": "calendar-sparse.v2"}, "version"),
        ]:
            try:
                engine.replay(
                    bars,
                    rates,
                    pd.read_parquet(targets / target_file),
                    changed,
                    split,
                    plan["window"]["end"],
                    initial_state=initial,
                    lifecycle=lifecycle,
                )
            except ValueError as error:
                assert expected in str(error)
                rejection.append(str(error))
            else:
                raise AssertionError("Policy drift resume was not rejected")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "research_tools.monthly_hold_audit.run",
                "--worker",
                str(output),
            ],
            check=True,
        )
        child = read(output / "restart_worker.json")
        joined = pd.concat(
            [prefix["curve"].iloc[:-1], pd.read_parquet(output / "restart_worker_curve.parquet")],
            ignore_index=True,
        )
        pd.testing.assert_frame_equal(joined, a["curve"], check_exact=True)
        assert (
            prefix["events"] + child["events"] == a["events"]
            and prefix["requests"] + child["requests"] == a["requests"]
        )
        assert (
            child["positions"] == a["positions"]
            and child["resume_state"] == a["resume_state"]
            and child["pid"] != os.getpid()
        )
        restart = {
            "performed": True,
            "passed": True,
            "split": split,
            "parent_pid": os.getpid(),
            "worker_pid": child["pid"],
            "held_positions": len(prefix["positions"]),
            "full_curve_ledger_inventory_risk_equal": True,
            "wrong_flag_and_engine_state_rejected": rejection,
        }
    originals = list(rows)
    for row in originals:
        rows.append(
            {
                **row,
                "id": row["id"].replace("disabled_", "reconciled_disabled_", 1),
                "variant": "daily_with_complete_5m_reconciliation",
                "reused_account": True,
                "source_account_id": row["id"],
            }
        )
    save(
        output / "summary.json",
        {
            "schema": "monthly-hold-counterfactual.v1",
            "account_rows": rows,
            "new_policy_accounts": len(originals),
            "bridge_replays": bridge_checks,
            "selected_case_paths": 1,
            "actual_financial_computations": len(originals) + len(bridge_checks) + 1,
            "source_alias_cases": len(originals),
            "comparisons": comparisons,
            "capacity_triggered": followup,
            "actual_capacity_accounts": len(originals) - 4,
            "actual_process_restart": restart,
            "failures": failures,
            "profit_goal_met": False,
            "external_review": False,
            "independent_unseen_evidence": False,
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
                str(q.relative_to(output)): sha(q) for q in output.rglob("*") if q.is_file()
            },
            "profit_goal_met": False,
        },
    )
    print(
        {
            "new_policy_accounts": len(originals),
            "bridge_replays": len(bridge_checks),
            "capacity_triggered": followup,
            "selected_case_paths": 1,
            "failures": len(failures),
            "restart": restart,
        },
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--contract", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--worker", type=Path)
    a = p.parse_args()
    if a.worker:
        worker(a.worker)
    else:
        assert a.contract and a.output
        run(a.contract, a.output)
