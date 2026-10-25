"""Freeze causal membership clocks before archive discovery or return scoring."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from factors.common.max_effect import features
from research_tools.contract_lifecycles.segments import context_at
from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    p = Path(__file__).with_name("plan.json")
    plan = read(p)
    daily = Path(plan["daily_source"])
    segments = Path(plan["segment_inputs"])
    verify_manifest(daily)
    verify_manifest(segments)
    for path, digest in read(segments / "plan.json")["source_hashes"].items():
        if sha(Path(path)) != digest:
            raise ValueError("Sealed segment input changed")
    registry = read(segments / "registry.json")
    first, end = (
        pd.Timestamp(plan["start"], tz="UTC"),
        pd.Timestamp(plan["end_exclusive"], tz="UTC"),
    )
    clocks = sorted(
        set(
            [first, pd.Timestamp("2026-01-01", tz="UTC")]
            + list(pd.date_range(first, end, freq="W-MON", inclusive="left"))
        )
    )
    paths = [
        p,
        Path(__file__),
        Path("factors/common/max_effect.py"),
        Path("research_tools/contract_lifecycles/segments.py"),
        Path("research_tools/contract_lifecycles/timeline.py"),
        daily / "completion.json",
        segments / "completion.json",
        Path(plan["metadata_source"]),
    ]
    save(
        output / "registration.json",
        {
            "source_hashes": {str(p): sha(p) for p in paths},
            "before_membership_and_execution_acquisition": True,
            "new_financial_accounts": 0,
        },
    )
    frames = {}
    starts = {}
    validity = {}
    for f in (daily / "symbols").glob("*_1d.parquet"):
        s = f.name.removesuffix("_1d.parquet")
        raw = pd.read_parquet(f)
        cal = pd.date_range(raw.index[0], raw.index[-1] + pd.Timedelta(days=1), freq="D")
        q = raw.reindex(cal)
        valid = q.price_liquidity_feature_usable.eq(True)
        # Invalid source values stay archived; only the factor view uses missingness.
        frames[s] = q[["close", "quote_volume"]].where(valid, axis=0)
        validity[s] = valid
    base = features(frames)
    for launch in registry["launch_cards"]:
        symbol = launch["symbol"]
        at = pd.Timestamp(launch["launch_at"]).ceil("D")
        cut = frames[symbol].copy()
        cut.loc[cut.index < at, :] = np.nan
        local = features({"BTCUSDT": frames["BTCUSDT"], symbol: cut})[symbol]
        starts[launch["id"]] = local
    metadata = {r["symbol"]: r for r in read(Path(plan["metadata_source"]))["symbols"]}
    records = []
    chosen = {"BTCUSDT"}
    full = []
    for clock in clocks:
        eligible = []
        for s, f in base.items():
            if s == "BTCUSDT" or clock not in f.index:
                continue
            context = (
                context_at(registry, s, clock)
                if any(c["symbol"] == s for c in registry["cards"])
                else {"blocked": False, "feature_start": None}
            )
            if context["blocked"]:
                continue
            row = f.loc[clock]
            if context["feature_start"]:
                launch = next(
                    c
                    for c in registry["launch_cards"]
                    if c["symbol"] == s
                    and pd.Timestamp(c["launch_at"]).ceil("D")
                    == pd.Timestamp(context["feature_start"])
                )
                row = starts[launch["id"]].loc[clock]
            if (
                row.observations >= 180
                and row.quote28 >= 50_000_000
                and np.isfinite(row[["max28", "momentum28", "beta60"]].to_numpy(dtype=float)).all()
            ):
                eligible.append((s, float(row.quote28)))
        selected = [s for s, _ in sorted(eligible, key=lambda r: (-r[1], r[0]))[:30]]
        chosen.update(selected)
        records.append(
            {
                "decision": clock.isoformat(),
                "selected": selected,
                "eligible": len(eligible),
                "historical_tradability_certified": False,
            }
        )
        for s in selected:
            full.append({"decision": clock, "symbol": s, "quote28": dict(eligible)[s]})
    if len(chosen) > plan["budget"]["maximum_symbols"]:
        raise ValueError("Fixed symbol acquisition budget exceeded")
    unknown = []
    for s in sorted(chosen):
        m = metadata.get(s, {})
        if m.get("underlyingType") not in ["COIN", None] or m.get("contractType") not in [
            "PERPETUAL",
            None,
        ]:
            unknown.append(
                {
                    "symbol": s,
                    "capture_type": m.get("underlyingType"),
                    "capture_contract_type": m.get("contractType"),
                    "historical_type_calendar_certified": False,
                }
            )
    save(output / "cohort.json", records)
    save(output / "symbols.json", sorted(chosen))
    save(
        output / "summary.json",
        {
            "schema": "expanded-market-cohort.v1",
            "decisions": len(clocks),
            "symbols": len(chosen),
            "memberships": sum(len(r["selected"]) for r in records),
            "classifications_pending": unknown,
            "acquisition_union_is_not_membership_feature": True,
            "historical_versions_certified": False,
            "new_financial_accounts": 0,
            "profit_goal_met": False,
        },
    )
    save(output / "plan.json", plan)
    save(
        output / "completion.json",
        {
            "manifest": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
            "new_financial_accounts": 0,
        },
    )
    print(read(output / "summary.json"), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    prepare(p.parse_args().output)
