"""One new interpreter for a frozen historical engineering regression, not alpha."""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from research_tools.expanded_market.ingest import flags
from research_tools.factor_lab.io import read, save, sha
from research_tools.factor_lab.study import executor
from research_tools.sparse_market.identity import directives


def inputs(plan, mode):
    before, end = (
        pd.Timestamp(plan["start"], tz="UTC") - pd.Timedelta(hours=1),
        pd.Timestamp(plan["end"], tz="UTC"),
    )
    query = [("date", ">=", before), ("date", "<", end)]
    market, funding = Path(plan["old_market"]), Path(plan["old_funding"])
    names = read(market / "prepare_plan.json")["symbols"]
    bars = {s: flags(pd.read_parquet(market / (s + "_5m.parquet"), filters=query)) for s in names}
    rates = {s: pd.read_parquet(funding / (s + "_exact.parquet")) for s in names}
    scores = pd.read_parquet(Path(plan["scores"]))
    if mode == "extended":
        for s in ["PROMUSDT", "RIFUSDT", "TACUSDT"]:
            bars[s] = pd.read_parquet(
                Path(plan["additional_market"]) / "symbols" / (s + "_5m.parquet"), filters=query
            )
            rates[s] = pd.read_parquet(Path(plan["bundle"]) / "funding" / (s + "_exact.parquet"))
            scores["target_" + s] = 0.0
    return bars, rates, scores


def stage(run, output, mode, start, end, restore):
    plan = read(run / "plan.json")
    for path, digest in plan["source_hashes"].items():
        if sha(Path(path)) != digest:
            raise ValueError("Registered engineering source changed")
    bars, rates, scores = inputs(plan, mode)
    sys.path.insert(0, str(Path(plan["sparse_engine"]).parents[2]))
    engine = executor(plan["dense_engine"] if mode == "dense" else plan["sparse_engine"])
    kwargs = (
        {}
        if mode == "dense"
        else {"lifecycle": directives(read(Path(plan["registry"])), sorted(bars))}
    )
    state = read(restore) if restore else None
    a = engine.replay(
        bars, rates, scores, plan["config"], start, end, initial_state=state, **kwargs
    )
    output.mkdir(parents=True, exist_ok=False)
    a["curve"].to_parquet(output / "curve.parquet", index=False)
    for key in ["events", "trades", "requests", "positions", "resume_state"]:
        save(output / (key + ".json"), a[key])
    save(
        output / "receipt.json",
        {
            "pid": os.getpid(),
            "mode": mode,
            "restored_from_disk": bool(restore),
            "checkpoint_sha256": sha(restore) if restore else None,
            "open_positions_at_start": len(state["positions"]) if state else 0,
            "historical_capability_certified": False,
            "alpha_admission": False,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=["dense", "sparse", "extended"], required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--restore", type=Path)
    a = p.parse_args()
    stage(a.run, a.output, a.mode, a.start, a.end, a.restore)
