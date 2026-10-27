"""Finalize preserved four pilot paths; repair restart boundary checking only.

No complete financial account is re-executed. A child's initial checkpoint row
has no preceding interval to diagnose. Its two interval diagnostics are unknown;
the actual prefix owns those values. All money, inventory, subsequent diagnostics,
events, requests and final state must remain bit exact, without numeric tolerance.
"""

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from research_tools.dated_carry.run_accounts import load
from research_tools.factor_lab.io import read, save, sha
from research_tools.factor_lab.study import executor
from research_tools.trend_stop_audit.run import paired_uncertainty


def run(original, output):
    assert not output.exists()
    plan = read(original / "plan.json")
    for path, digest in plan["account_source_hashes"].items():
        assert sha(Path(path)) == digest, path
    rows = [
        read(original / "accounts" / case["id"] / "row.json")
        for case in read(original / "registration.json")["cases"]
    ]
    assert len(rows) == 4 and all(r["internal_cash_audit_passed"] for r in rows)
    for row in rows:
        folder = original / "accounts" / row["id"]
        assert sha(folder / "result.json") == row["result_sha256"]
        for path, digest in row["source_hashes"].items():
            assert sha(folder / path) == digest, path
    output.mkdir(parents=True)
    save(output / "plan.json", plan)
    save(
        output / "repair_registration.json",
        {
            "original": str(original),
            "full_accounts_reexecuted": 0,
            "technical_scope": __doc__,
            "source_sha256": sha(Path(__file__)),
            "allowed_initial_unknown_columns": ["margin_envelope_balance", "stress_nav_envelope"],
            "all_other_checks": "bit exact; no relaxed tolerance",
        },
    )
    _, markets = load(Path(plan["prepared_inputs"]))
    engine = executor(plan["engine"])
    full_folder = original / "accounts/hold_to_delivery_slip10"
    full_curve = pd.read_parquet(full_folder / "equity_5m.parquet")
    full = {
        k: read(full_folder / (k + ".json"))
        for k in ["events", "requests", "trades", "resume_state"]
    }
    config = read(full_folder / "result.json")["config"]
    signals = read(full_folder / "signals.json")
    splits = [
        ("held_entry", next(e["time"] for e in full["events"] if e["type"] == "entry")),
        ("at_delivery", next(e["time"] for e in full["events"] if e["type"] == "settlement")),
    ]
    restarts = []
    for name, split in splits:
        prefix = engine.replay(
            *markets,
            signals,
            config,
            plan["window"]["start"],
            split,
            final_end=plan["window"]["end"],
            contracts=plan["contracts"],
        )
        assert prefix["positions"]
        pd.testing.assert_frame_equal(
            prefix["curve"],
            full_curve.loc[full_curve.date <= pd.Timestamp(split)].reset_index(drop=True),
            check_exact=True,
        )
        if name == "held_entry":
            assert prefix["resume_state"] == read(original / (name + "_checkpoint.json"))
            child = read(original / (name + "_child.json"))
            curve = pd.read_parquet(original / (name + "_child_curve.parquet"))
            reused = True
        else:
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
                    "research_tools.dated_carry.run_accounts",
                    "--worker",
                    str(output),
                    "--name",
                    name,
                ],
                check=True,
            )
            child = read(output / (name + "_child.json"))
            curve = pd.read_parquet(output / (name + "_child_curve.parquet"))
            reused = False
        expected = full_curve.loc[full_curve.date >= pd.Timestamp(split)].reset_index(drop=True)
        initial_unknown = ["margin_envelope_balance", "stress_nav_envelope"]
        assert curve.loc[0, initial_unknown].isna().all()
        pd.testing.assert_frame_equal(
            curve.drop(columns=initial_unknown),
            expected.drop(columns=initial_unknown),
            check_exact=True,
        )
        pd.testing.assert_frame_equal(curve.iloc[1:], expected.iloc[1:], check_exact=True)
        stitched = pd.concat([prefix["curve"], curve.iloc[1:]], ignore_index=True)
        pd.testing.assert_frame_equal(stitched, full_curve, check_exact=True)
        for key in ["events", "requests", "trades"]:
            assert prefix[key] + child[key] == full[key], key
        assert child["resume_state"] == full["resume_state"] and child["pid"] != os.getpid()
        restarts.append(
            {
                "name": name,
                "split": split,
                "positions": len(prefix["positions"]),
                "future_base_quantity": math.fsum(
                    p["perp_q"] for p in prefix["positions"].values()
                ),
                "spot_base_quantity": math.fsum(p["spot_q"] for p in prefix["positions"].values()),
                "parent_pid": os.getpid(),
                "child_pid": child["pid"],
                "reused_original_child": reused,
                "money_ledger_requests_final_state_bit_exact": True,
                "complete_prefix_plus_child_curve_bit_exact": True,
                "initial_interval_diagnostics_owned_by_prefix": initial_unknown,
            }
        )
    cash = Path(
        "results/research/sparse_family_accounts_20261026_v002/accounts/cash_continuous2025_2026_cash_cap10000_slip10"
    )
    cash_curve = (
        pd.read_parquet(cash / "equity_5m.parquet")
        .set_index("date")
        .loc[pd.Timestamp(plan["window"]["start"]) : pd.Timestamp(plan["window"]["end"])]
    )
    assert cash_curve.index.equals(
        pd.date_range(plan["window"]["start"], plan["window"]["end"], freq="5min")
    )
    assert cash_curve.equity.eq(config["capital"]).all()
    comparisons = []
    for cost in plan["costs_slippage_bps"]:
        a, b = [
            next(r for r in rows if r["profile"] == profile and r["slippage_bps"] == cost)
            for profile in plan["profiles"]
        ]
        ca, cb = [
            pd.read_parquet(original / "accounts" / r["id"] / "equity_5m.parquet") for r in [a, b]
        ]
        comparisons.append(
            {
                "slippage_bps": cost,
                "early_minus_delivery_net_pnl": b["metrics"]["net_pnl"] - a["metrics"]["net_pnl"],
                "paired_daily_uncertainty": paired_uncertainty(ca, cb),
                "same_realized_risk": False,
            }
        )
    save(
        output / "summary.json",
        {
            "schema": "dated-carry-pilot-accounts.v2",
            "original_accounts": str(original / "accounts"),
            "account_rows": rows,
            "actual_strategy_accounts": 4,
            "new_accounts_this_repair": 0,
            "financial_audit_failures": [],
            "actual_process_restarts": restarts,
            "comparisons": comparisons,
            "reused_cash_control": {
                "source": str(cash),
                "result_sha256": sha(cash / "result.json"),
                "calendar_days": 87,
                "net_pnl": 0,
            },
            "sample_sufficient": False,
            "module_admitted": False,
            "capacity_accounts": 0,
            "independent_oos": False,
            "third_party_review": False,
            "profit_goal_met": False,
            "orders_enabled": False,
            "production_eligible": False,
        },
    )
    save(
        original / "failure.json",
        {
            "stage": "restart checker, after all four independent financial audits passed",
            "reason": "Initial child checkpoint interval diagnostics are NaN; prefix owns preceding interval values. Money and final state unchanged.",
            "source_unchanged": True,
            "repair": str(output),
            "financial_paths_reexecuted": 0,
        },
    )
    save(
        original / "completion.json",
        {
            "passed": False,
            "manifest": {
                str(p.relative_to(original)): sha(p)
                for p in original.rglob("*")
                if p.is_file() and p.name != "completion.json"
            },
        },
    )
    save(
        output / "completion.json",
        {
            "passed": True,
            "original_completion_sha256": sha(original / "completion.json"),
            "manifest": {
                str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()
            },
            "profit_goal_met": False,
        },
    )
    print(
        {"accounts": 4, "new_accounts": 0, "restart_checks": restarts, "module_admitted": False},
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.original, a.output)
