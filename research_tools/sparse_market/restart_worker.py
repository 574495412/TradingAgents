"""Replay one registered account segment in a separate interpreter from disk."""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.factor_lab.study import executor


def stage(run, account, output, split, restore):
    plan = read(run / "plan.json")
    for group in ["source_hashes", "account_source_hashes"]:
        for path, digest in plan[group].items():
            if sha(Path(path)) != digest:
                raise ValueError("Registered replay implementation changed")
    folder = run / "accounts" / account
    result = read(folder / "result.json")
    if not read(folder / "row.json")["financial_audit_passed"]:
        raise ValueError("Cannot restore an uncertified financial account")
    for name, digest in result["source_hashes"].items():
        if sha(folder / name) != digest:
            raise ValueError("Original financial account changed")
    cohort, bundle, inputs = Path(plan["cohort"]), Path(plan["bundle"]), Path(plan["inputs"])
    symbols = read(cohort / "symbols.json")
    bars = {
        s: pd.read_parquet(
            bundle / "symbols" / (s + "_5m.parquet"),
            columns=[
                "open",
                "high",
                "low",
                "close",
                "quote_volume",
                "price_usable",
                "execution_liquidity_usable",
            ],
        )
        for s in symbols
    }
    rates = {s: pd.read_parquet(bundle / "funding" / (s + "_exact.parquet")) for s in symbols}
    row = read(folder / "row.json")
    scores = pd.read_parquet(
        inputs / f"{row['variant']}_{row['phase']}_{row['candidate']}_targets.parquet"
    )
    state = read(restore) if restore else None
    sys.path.insert(0, str(Path(plan["engine"]).parents[2]))
    a = executor(plan["engine"]).replay(
        bars,
        rates,
        scores,
        result["config"],
        split if restore else result["window"]["start"],
        result["window"]["end"] if restore else split,
        initial_state=state,
        lifecycle=read(inputs / "lifecycle_directives.json"),
    )
    output.mkdir(parents=True, exist_ok=False)
    a["curve"].to_parquet(output / "curve.parquet", index=False)
    for key in ["events", "requests", "trades", "resume_state"]:
        save(output / (key + ".json"), a[key])
    save(
        output / "receipt.json",
        {
            "pid": os.getpid(),
            "restored_from_disk": bool(restore),
            "checkpoint_sha256": sha(restore) if restore else None,
            "open_positions_at_start": len(state["positions"]) if state else 0,
            "risk_halted_at_start": state["risk_halted"] if state else False,
            "source_sha256": sha(Path(__file__)),
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--account", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--restore", type=Path)
    a = p.parse_args()
    stage(a.run, a.account, a.output, a.split, a.restore)
