"""Supplemental real-data recovery at pending-cash boundaries, same accounts.

Does not add candidates, tune parameters, or change existing result files.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .continuous import replay
from .data import digest, write_json
from .layered import ROOT, inputs


def verify(output, child=False):
    registration = json.loads((output / "registration.json").read_text())
    for rel, sha in registration["source_hashes"].items():
        if digest(ROOT / rel) != sha:
            raise ValueError("Original account source identity changed")
    plan = json.loads((output / "protocol.json").read_text())
    evidence = output / "pending-cash-recovery"
    if not child:
        evidence.mkdir(exist_ok=False)
        jobs = []
        for market, capital in [("US", 10000), ("HK", 100000)]:
            name = f"{market}_static_{capital}_15bps"
            state = json.loads((output / name / "state.json").read_text())
            rows = state["equity_curve"]
            # Explicitly require an outstanding settlement queue in real history.
            idx = next(i for i, r in enumerate(rows[:-1]) if r.get("unsettled", 0) > 0)
            boundary = rows[idx + 1]["day"]
            args = inputs(registration["dataset"], plan, market, "static", capital)
            checkpoint = replay(
                *args,
                market=market,
                policy="static",
                start=plan["window"][0],
                end=boundary,
                slippage_bps=15,
                identity=registration["identity"],
            )
            write_json(evidence / f"{market}-checkpoint.json", checkpoint)
            jobs.append({"market": market, "capital": capital, "name": name, "boundary": boundary})
        write_json(evidence / "jobs.json", jobs)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tradingagents.research.equities.verify_boundaries",
                str(output),
                "--child",
            ],
            cwd=ROOT,
            check=True,
        )
        return
    receipts = []
    for job in json.loads((evidence / "jobs.json").read_text()):
        market = job["market"]
        args = inputs(registration["dataset"], plan, market, "static", job["capital"])
        file = evidence / f"{market}-checkpoint.json"
        checkpoint = json.loads(file.read_text())
        assert checkpoint["unsettled"]
        resumed = replay(
            *args,
            market=market,
            policy="static",
            start=job["boundary"],
            end=plan["window"][1],
            slippage_bps=15,
            identity=registration["identity"],
            state=checkpoint,
        )
        full = json.loads((output / job["name"] / "state.json").read_text())
        assert resumed == full
        receipts.append(
            {
                "passed": True,
                **job,
                "checkpoint_sha256": digest(file),
                "pending_entries": len(checkpoint["unsettled"]),
                "open_positions": len(checkpoint["positions"]),
                "high_water": checkpoint["peak"],
                "all_fields_exact": True,
                "method": "fresh interpreter; actual outstanding sale-cash queue",
            }
        )
    write_json(
        evidence / "verification.json",
        {"passed": True, "same_economic_accounts": True, "new_candidates": 0, "accounts": receipts},
    )
    print(json.dumps(receipts, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    verify(args.output, args.child)


if __name__ == "__main__":
    main()
