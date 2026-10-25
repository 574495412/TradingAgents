"""Verify four actual first-process/checkpoint/new-process account continuations."""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from research_tools.factor_lab.io import read, save, sha


def verify(run, output):
    output.mkdir(parents=True, exist_ok=False)
    checks = []
    for i, check in enumerate(read(run / "summary.json")["restart_checks"]):
        root = output / f"check{i:02d}"
        first, second = root / "first", root / "restored"
        command = [
            sys.executable,
            "-m",
            "research_tools.carry_feasibility.restart_worker",
            "--run",
            str(run),
            "--account",
            check["id"],
            "--split",
            check["split"],
        ]
        subprocess.run(command + ["--output", str(first)], check=True)
        subprocess.run(
            command + ["--output", str(second), "--restore", str(first / "resume_state.json")],
            check=True,
        )
        folder = run / "accounts" / check["id"]
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
        checks.append(
            {
                "account": check["id"],
                "split": check["split"],
                "passed": True,
                "first_pid": a["pid"],
                "restored_pid": b["pid"],
                "checkpoint_sha256": b["checkpoint_sha256"],
                "open_positions_at_split": b["open_positions_at_start"],
                "curve_and_all_ledgers_exact": True,
            }
        )
        print(
            {
                "verified_process_restart": i + 1,
                "of": 4,
                "account": check["id"],
                "nonflat": b["open_positions_at_start"],
            },
            flush=True,
        )
    save(
        output / "audit.json",
        {
            "passed": True,
            "checks": checks,
            "run_completion_sha256": sha(run / "completion.json"),
            "controller_source_sha256": sha(Path(__file__)),
            "worker_source_sha256": sha(Path("research_tools/carry_feasibility/restart_worker.py")),
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    verify(a.run, a.output)
