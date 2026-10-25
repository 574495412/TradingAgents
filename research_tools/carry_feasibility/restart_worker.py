"""New-interpreter replay stage reading a real on-disk account checkpoint."""

import argparse
import os
from pathlib import Path

import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.factor_lab.study import executor


def stage(run, account, output, split, restore=None):
    plan = read(run / "plan.json")
    folder = run / "accounts" / account
    for path, digest in plan["source_hashes"].items():
        if sha(Path(path)) != digest:
            raise ValueError("Frozen account source changed")
    result = read(folder / "result.json")
    for name, digest in result["source_hashes"].items():
        if sha(folder / name) != digest:
            raise ValueError("Frozen full account changed")
    symbols = plan["symbols"]
    spot = {s: pd.read_parquet(Path(plan["spot_data"]) / f"{s}_spot_5m.parquet") for s in symbols}
    perp = {s: pd.read_parquet(Path(plan["perpetual_data"]) / f"{s}_5m.parquet") for s in symbols}
    marks = {s: pd.read_parquet(Path(plan["mark_data"]) / f"{s}_mark_5m.parquet") for s in symbols}
    funding = {
        s: pd.read_parquet(Path(plan["spot_data"]) / f"{s}_funding_exact.parquet") for s in symbols
    }
    engine = executor(plan["engine"])
    window = result["window"]
    signals = read(folder / "signals.json")
    state = read(restore) if restore else None
    a = engine.replay(
        spot,
        perp,
        marks,
        funding,
        signals,
        result["config"],
        split if restore else window["start"],
        window["end"] if restore else split,
        initial_state=state,
        final_end=window["end"],
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
            "next_clock": a["resume_state"]["next_clock"],
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
