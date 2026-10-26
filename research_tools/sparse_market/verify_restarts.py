"""Verify all completed high-cost continuous families without resetting risk."""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest


def verify(run, output):
    output.mkdir(parents=True, exist_ok=False)
    verify_manifest(run)
    cases, omitted = [], []
    for row in read(run / "summary.json")["account_rows"]:
        if (
            row["phase"] != "continuous2025_2026"
            or row["slippage_bps"] != 10
            or row["candidate"] == "cash"
        ):
            continue
        if not row["financial_audit_passed"]:
            omitted.append({"account": row["id"], "reason": "full_financial_account_unknown"})
            continue
        events = read(run / "accounts" / row["id"] / "events.json")
        first = next((e["time"] for e in events if e["type"] == "entry"), None)
        if first:
            cases.append({"account": row["id"], "split": first, "kind": "first_nonflat"})
        cases.append({"account": row["id"], "split": "2026-01-01", "kind": "cross_year"})
    plan = {
        "schema": "sparse-actual-process-restoration.v1",
        "cases": cases,
        "omitted": omitted,
        "run_completion_sha256": sha(run / "completion.json"),
        "source_hashes": {
            str(p): sha(p) for p in [Path(__file__), Path(__file__).with_name("restart_worker.py")]
        },
        "before_new_interpreters": True,
        "new_alpha_trials": 0,
    }
    save(output / "plan.json", plan)
    checks = []
    for i, case in enumerate(cases):
        first, second = output / f"case{i:02d}" / "first", output / f"case{i:02d}" / "restored"
        base = [
            sys.executable,
            "-m",
            "research_tools.sparse_market.restart_worker",
            "--run",
            str(run),
            "--account",
            case["account"],
            "--split",
            case["split"],
        ]
        subprocess.run(base + ["--output", str(first)], check=True)
        subprocess.run(
            base + ["--output", str(second), "--restore", str(first / "resume_state.json")],
            check=True,
        )
        full = run / "accounts" / case["account"]
        pd.testing.assert_frame_equal(
            pd.concat(
                [
                    pd.read_parquet(first / "curve.parquet"),
                    pd.read_parquet(second / "curve.parquet").iloc[1:],
                ],
                ignore_index=True,
            ),
            pd.read_parquet(full / "equity_5m.parquet"),
        )
        for key in ["events", "requests", "trades"]:
            if read(first / (key + ".json")) + read(second / (key + ".json")) != read(
                full / (key + ".json")
            ):
                raise ValueError("Actual process restoration changes " + key)
        if read(second / "resume_state.json") != read(full / "resume_state.json"):
            raise ValueError("Actual process restoration changes final risk/inventory state")
        a, b = read(first / "receipt.json"), read(second / "receipt.json")
        if (
            a["pid"] == b["pid"]
            or not b["restored_from_disk"]
            or b["checkpoint_sha256"] != sha(first / "resume_state.json")
        ):
            raise ValueError("Not a separate process and actual checkpoint")
        if case["kind"] == "first_nonflat" and not b["open_positions_at_start"]:
            raise ValueError("Nonflat checkpoint was flat")
        check = {
            **case,
            "passed": True,
            "first_pid": a["pid"],
            "restored_pid": b["pid"],
            "open_positions": b["open_positions_at_start"],
            "risk_halted": b["risk_halted_at_start"],
            "checkpoint_sha256": b["checkpoint_sha256"],
        }
        checks.append(check)
        print(
            {
                "verified_restores": len(checks),
                "planned_restores": len(cases),
                "kind": case["kind"],
                "open_positions": b["open_positions_at_start"],
            },
            flush=True,
        )
    result = {
        "schema": plan["schema"],
        "passed": True,
        "run_completion_sha256": plan["run_completion_sha256"],
        "checks": checks,
        "omitted_unknown_full_accounts": omitted,
        "new_alpha_trials": 0,
        "profit_goal_met": False,
    }
    save(output / "verification.json", result)
    save(
        output / "completion.json",
        {
            "manifest": {
                str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()
            },
            "profit_goal_met": False,
        },
    )
    print({"terminal": True, "restores": len(checks), "passed": True}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    verify(a.run, a.output)
