"""Independent prefix-only NumPy calculation of expanded cohort membership."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest


def audit(cohort, output):
    if output.exists():
        raise FileExistsError("Preserve independent cohort audit")
    verify_manifest(cohort)
    plan = read(cohort / "plan.json")
    registry = read(Path(plan["segment_inputs"]) / "registry.json")
    records = read(cohort / "cohort.json")
    raw = {
        p.name.removesuffix("_1d.parquet"): pd.read_parquet(p)
        for p in (Path(plan["daily_source"]) / "symbols").glob("*_1d.parquet")
    }
    passed = []
    for record in records:
        clock = pd.Timestamp(record["decision"])
        eligible = []
        visible = [c for c in registry["cards"] if pd.Timestamp(c["available_at_assumed"]) <= clock]
        replaced = {c["supersedes"] for c in visible if c["supersedes"]}
        active = [c for c in visible if c["id"] not in replaced]
        launches = {
            c["symbol"]: c
            for c in registry["launch_cards"]
            if pd.Timestamp(c["available_at_assumed"]) <= clock
            and pd.Timestamp(c["launch_at"]) <= clock
        }
        for s, frame in raw.items():
            if (
                s == "BTCUSDT"
                or clock > frame.index[-1] + pd.Timedelta(days=1)
                or clock < frame.index[0]
            ):
                continue
            events = [
                c
                for c in active
                if c["symbol"] == s
                and (s not in launches or c["episode"] == launches[s]["episode"])
            ]
            blocked = False
            for event in events:
                cut = event["entry_stop_at"]
                prior = event
                while cut is None and prior["supersedes"]:
                    prior = next(c for c in visible if c["id"] == prior["supersedes"])
                    cut = prior["entry_stop_at"]
                if clock >= pd.Timestamp(event["settlement_at"]) or (
                    cut and clock >= pd.Timestamp(cut)
                ):
                    blocked = True
            if blocked:
                continue
            start = (
                pd.Timestamp(launches[s]["launch_at"]).ceil("D")
                if s in launches
                else frame.index[0]
            )
            past = frame.loc[(frame.index >= start) & (frame.index < clock)]
            valid = past.price_liquidity_feature_usable.eq(True)
            if valid.sum() < 180:
                continue
            expected = pd.date_range(
                clock - pd.Timedelta(days=61), clock, freq="D", inclusive="left"
            )
            history = past.reindex(expected)
            close = history.close.where(history.price_liquidity_feature_usable.eq(True)).to_numpy(
                dtype=float
            )
            market = raw["BTCUSDT"].reindex(expected)
            btc = market.close.where(market.price_liquidity_feature_usable.eq(True)).to_numpy(
                dtype=float
            )
            if not np.isfinite(close).all() or not np.isfinite(btc).all():
                continue
            r = btc[1:] / btc[:-1] - 1
            if np.var(r, ddof=1) <= 1e-12:
                continue
            quote = history.quote_volume.to_numpy(dtype=float)[-28:]
            if (
                not np.isfinite(quote).all()
                or not history.price_liquidity_feature_usable.eq(True).iloc[-28:].all()
            ):
                continue
            q = float(np.mean(quote))
            if q >= 50_000_000:
                eligible.append((s, q))
        chosen = [s for s, _ in sorted(eligible, key=lambda r: (-r[1], r[0]))[:30]]
        if chosen != record["selected"] or len(eligible) != record["eligible"]:
            raise ValueError("Independent raw-prefix eligibility differs at " + record["decision"])
        passed.append(record["decision"])
    result = {
        "schema": "expanded-cohort-independent-audit.v1",
        "passed": True,
        "decisions": len(passed),
        "exact_selected_and_eligible_counts": True,
        "cohort_completion_sha256": sha(cohort / "completion.json"),
        "source_sha256": sha(Path(__file__)),
        "new_financial_accounts": 0,
        "historical_versions_certified": False,
    }
    save(output, result)
    print(result, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    audit(a.cohort, a.output)
