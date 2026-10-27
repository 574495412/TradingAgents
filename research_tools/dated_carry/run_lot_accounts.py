"""Run four frozen conditional grid accounts and exact process-restoration checks."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from research_tools.dated_carry.account_audit_lots_v2 import audit_account
from research_tools.dated_carry.run_accounts import load, publish
from research_tools.factor_lab.io import read, save, sha
from research_tools.factor_lab.study import executor
from research_tools.max_effect.run_accounts import verify_manifest


def worker(output, name):
    plan = read(output / "plan.json")
    for path, digest in plan["run_source_hashes"].items():
        assert sha(Path(path)) == digest
    _, markets = load(Path(plan["base_prepared_inputs"]))
    reg = read(output / (name + "_registration.json"))
    account = executor(plan["engine"]).replay(
        *markets,
        reg["signals"],
        reg["config"],
        reg["split"],
        plan["window"]["end"],
        contracts=plan["contracts"],
        initial_state=read(output / (name + "_checkpoint.json")),
        final_end=plan["window"]["end"],
    )
    account["curve"].to_parquet(output / (name + "_child_curve.parquet"), index=False)
    save(
        output / (name + "_child.json"),
        {
            "pid": os.getpid(),
            **{k: account[k] for k in ["events", "requests", "trades", "resume_state"]},
        },
    )


def run(inputs, output):
    output.mkdir(parents=True, exist_ok=False)
    verify_manifest(inputs)
    plan = read(inputs / "plan.json")
    for path, digest in plan["source_hashes"].items():
        assert sha(Path(path)) == digest
    save(output / "protocol.json", read(Path("docs/research/regime_program/targets.json")))
    files = [
        Path(__file__),
        Path("research_tools/dated_carry/account_audit_lots_v2.py"),
        Path("research_tools/dated_carry/run_accounts.py"),
        Path(plan["engine"]),
        Path("research_tools/factor_lab/metrics.py"),
        Path("research_tools/factor_lab/io.py"),
        inputs / "completion.json",
        output / "protocol.json",
    ]
    plan["run_source_hashes"] = {str(p): sha(p) for p in files}
    plan["prepared_lot_inputs"] = str(inputs)
    save(output / "plan.json", plan)
    cases = [
        {"id": f"{profile}_slip{cost}", "profile": profile, "slippage_bps": cost}
        for profile in plan["profiles"]
        for cost in plan["costs_slippage_bps"]
    ]
    assert len(cases) == plan["budget"]["actual_strategy_accounts_max"]
    save(
        output / "registration.json",
        {
            "before_new_financial_returns": True,
            "cases": cases,
            "conditional_rules_only": True,
            "planned_restarts": plan["budget"]["planned_process_restarts"],
        },
    )
    _, markets = load(Path(plan["base_prepared_inputs"]))
    engine = executor(plan["engine"])
    rows, failures, full = [], [], None
    for case in cases:
        config = {**plan["account"], "slippage_bps": case["slippage_bps"]}
        signals = read(inputs / (case["profile"] + "_signals.json"))
        a = engine.replay(
            *markets,
            signals,
            config,
            plan["window"]["start"],
            plan["window"]["end"],
            contracts=plan["contracts"],
        )
        folder = output / "accounts" / case["id"]
        result = publish(folder, a, signals, config, plan, markets[0])
        row = {
            **case,
            **result,
            "result_sha256": sha(folder / "result.json"),
            "internal_cash_integer_audit_passed": False,
            "historical_venue_rules_certified": False,
            "module_admitted": False,
        }
        rows.append(row)
        if case["id"] == "hold_to_delivery_slip10":
            full = a
        print({"id": case["id"], "computed": True, "audit": "pending"}, flush=True)
    for row in rows:
        folder = output / "accounts" / row["id"]
        try:
            proof = audit_account(folder, *markets, plan["contracts"])
            save(folder / "internal_cash_integer_audit.json", proof)
            row["internal_cash_integer_audit_passed"] = True
        except Exception as e:
            failures.append(
                {
                    "stage": "cash_integer_audit",
                    "id": row["id"],
                    "error": str(e),
                    "type": type(e).__name__,
                }
            )
        save(folder / "row.json", row)
        print(
            {
                "id": row["id"],
                "audited": row["internal_cash_integer_audit_passed"],
                "net_pnl": row["metrics"]["net_pnl"],
                "closed_cycles": row["metrics"]["closed_trades"],
            },
            flush=True,
        )
    restarts = []
    if not failures and full is not None:
        config = {**plan["account"], "slippage_bps": 10}
        signals = read(inputs / "hold_to_delivery_signals.json")
        splits = [
            ("held_entry", next(e["time"] for e in full["events"] if e["type"] == "entry")),
            ("at_delivery", next(e["time"] for e in full["events"] if e["type"] == "settlement")),
            ("dust_only", max(p["dust_first_at"] for p in full["positions"].values())),
        ]
        for name, split in splits:
            try:
                prefix = engine.replay(
                    *markets,
                    signals,
                    config,
                    plan["window"]["start"],
                    split,
                    contracts=plan["contracts"],
                    final_end=plan["window"]["end"],
                )
                assert prefix["positions"]
                save(output / (name + "_checkpoint.json"), prefix["resume_state"])
                save(
                    output / (name + "_registration.json"),
                    {
                        "split": split,
                        "signals": signals,
                        "config": config,
                        "parent_pid": os.getpid(),
                    },
                )
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "research_tools.dated_carry.run_lot_accounts",
                        "--worker",
                        str(output),
                        "--name",
                        name,
                    ],
                    check=True,
                )
                child = read(output / (name + "_child.json"))
                curve = pd.read_parquet(output / (name + "_child_curve.parquet"))
                stitched = pd.concat([prefix["curve"], curve.iloc[1:]], ignore_index=True)
                pd.testing.assert_frame_equal(stitched, full["curve"], check_exact=True)
                pd.testing.assert_frame_equal(
                    curve.drop(columns=["margin_envelope_balance", "stress_nav_envelope"]),
                    full["curve"]
                    .loc[full["curve"].date >= pd.Timestamp(split)]
                    .reset_index(drop=True)
                    .drop(columns=["margin_envelope_balance", "stress_nav_envelope"]),
                    check_exact=True,
                )
                for k in ["events", "requests", "trades"]:
                    assert prefix[k] + child[k] == full[k], k
                assert child["resume_state"] == full["resume_state"] and child["pid"] != os.getpid()
                restarts.append(
                    {
                        "name": name,
                        "split": split,
                        "parent_pid": os.getpid(),
                        "child_pid": child["pid"],
                        "checkpoint_inventory": {
                            s: {k: p[k] for k in ["spot_atoms", "perp_atoms"]}
                            for s, p in prefix["positions"].items()
                        },
                        "whole_curve_money_ledger_requests_and_final_state_bit_exact": True,
                        "initial_interval_diagnostics_owned_by_prefix": True,
                    }
                )
            except Exception as e:
                failures.append(
                    {"stage": "restart", "name": name, "error": str(e), "type": type(e).__name__}
                )
    summary = {
        "schema": "dated-carry-conditional-lot-accounts.v1",
        "account_rows": rows,
        "failures": failures,
        "actual_process_restarts": restarts,
        "new_financial_accounts": 4,
        "new_predictive_models": 0,
        "new_economic_variants": 0,
        "original_fractional_accounts_changed": False,
        "sample_sufficient": False,
        "historical_venue_rules_certified": False,
        "original_3bps_price_counterexample_repaired": False,
        "independent_oos": False,
        "external_review_performed": False,
        "module_admitted": False,
        "capacity_accounts": 0,
        "profit_goal_met": False,
        "orders_enabled": False,
    }
    save(output / "summary.json", summary)
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
            "accounts": 4,
            "failures": failures,
            "actual_process_restarts": restarts,
            "module_admitted": False,
        },
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--worker", type=Path)
    p.add_argument("--name")
    a = p.parse_args()
    if a.worker:
        worker(a.worker, a.name)
    else:
        assert a.inputs and a.output
        run(a.inputs, a.output)
