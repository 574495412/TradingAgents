"""Run one stage in a fresh interpreter, with an actual on-disk checkpoint."""

import argparse
import os
from pathlib import Path

import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.factor_lab.study import executor


def stage(run, account, output, split, restore=None):
    plan = read(run / "plan.json")
    for path, digest in plan["account_source_hashes"].items():
        if sha(Path(path)) != digest:
            raise ValueError("Frozen account source changed")
    folder = run / "accounts" / account
    result = read(folder / "result.json")
    for name, digest in result["source_hashes"].items():
        if Path(name).name != name or sha(folder / name) != digest:
            raise ValueError("Frozen full account changed")
    data, funding, inputs = Path(plan["data"]), Path(plan["funding_data"]), Path(plan["inputs"])
    symbols = read(data / "prepare_plan.json")["symbols"]
    bars = {s: pd.read_parquet(data / (s + "_5m.parquet")) for s in symbols}
    rates = {s: pd.read_parquet(funding / (s + "_exact.parquet")) for s in symbols}
    row = read(folder / "row.json")
    scores = pd.read_parquet(
        inputs / f"{result['window']['id']}_{row['candidate']}_targets.parquet"
    )
    state = read(restore) if restore else None
    a = executor(plan["engine"]).replay(
        bars,
        rates,
        scores,
        result["config"],
        split if restore else result["window"]["start"],
        result["window"]["end"] if restore else split,
        initial_state=state,
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
            "halted_at_start": state["risk_halted"] if state else False,
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
