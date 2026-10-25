"""Check nonflat and cross-year process continuations without resetting risk state."""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from research_tools.factor_lab.io import read, save, sha


def verify(run, output):
    output.mkdir(parents=True, exist_ok=False)
    source = read(run / "summary.json")
    cases = []
    for row in source["account_rows"]:
        if (
            row["phase"] != "continuous2025_2026"
            or row["slippage_bps"] != 10
            or not row["candidate"].startswith("funding7")
        ):
            continue
        events = read(run / "accounts" / row["id"] / "events.json")
        first = next((e["time"] for e in events if e["type"] == "entry"), None)
        if first:
            cases.append((row["id"], first, "first_nonflat"))
        cases.append((row["id"], "2026-01-01", "cross_year"))
    save(
        output / "plan.json",
        {
            "cases": cases,
            "run_completion_sha256": sha(run / "completion.json"),
            "before_restart_replays": True,
        },
    )
    checks = []
    for i, (account, split, kind) in enumerate(cases):
        first, second = output / f"check{i:02d}" / "first", output / f"check{i:02d}" / "restored"
        command = [
            sys.executable,
            "-m",
            "research_tools.funding_cross_section.restart_worker",
            "--run",
            str(run),
            "--account",
            account,
            "--split",
            split,
        ]
        subprocess.run(command + ["--output", str(first)], check=True)
        subprocess.run(
            command + ["--output", str(second), "--restore", str(first / "resume_state.json")],
            check=True,
        )
        folder = run / "accounts" / account
        pd.testing.assert_frame_equal(
            pd.concat(
                [
                    pd.read_parquet(first / "curve.parquet"),
                    pd.read_parquet(second / "curve.parquet").iloc[1:],
                ],
                ignore_index=True,
            ),
            pd.read_parquet(folder / "equity_5m.parquet"),
        )
        for key in ["events", "requests", "trades"]:
            if read(first / (key + ".json")) + read(second / (key + ".json")) != read(
                folder / (key + ".json")
            ):
                raise ValueError("New process continuation changes " + key)
        if read(second / "resume_state.json") != read(folder / "resume_state.json"):
            raise ValueError("New process final state differs")
        a, b = read(first / "receipt.json"), read(second / "receipt.json")
        if (
            a["pid"] == b["pid"]
            or not b["restored_from_disk"]
            or b["checkpoint_sha256"] != sha(first / "resume_state.json")
        ):
            raise ValueError("Not a new interpreter with actual checkpoint")
        if kind == "first_nonflat" and not b["open_positions_at_start"]:
            raise ValueError("Nonflat checkpoint is flat")
        check = {
            "account": account,
            "split": split,
            "kind": kind,
            "passed": True,
            "first_pid": a["pid"],
            "restored_pid": b["pid"],
            "checkpoint_sha256": b["checkpoint_sha256"],
            "open_positions_at_split": b["open_positions_at_start"],
            "halted_at_split": b["halted_at_start"],
            "curve_and_all_ledgers_exact": True,
        }
        checks.append(check)
        print(check, flush=True)
    save(
        output / "audit.json",
        {
            "passed": True,
            "checks": checks,
            "run_completion_sha256": sha(run / "completion.json"),
            "controller_source_sha256": sha(Path(__file__)),
            "worker_source_sha256": sha(
                Path("research_tools/funding_cross_section/restart_worker.py")
            ),
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    verify(a.run, a.output)
