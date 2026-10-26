"""Independent raw-prefix membership checks, including reused ticker clocks."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest


def prepare_raw(plan, cohort):
    raw = {
        p.name.removesuffix("_1d.parquet"): pd.read_parquet(p)
        for p in (Path(plan["parent_daily"]) / "symbols").glob("*_1d.parquet")
    }
    alternative = {s: f.copy() for s, f in raw.items()}
    groups = {}
    for row in read(cohort / "reconciliation_days.json"):
        groups.setdefault(row["symbol"], []).append(pd.Timestamp(row["day"]))
    for s, days in groups.items():
        f = pd.read_parquet(
            Path(plan["parent_market"]) / "symbols" / (s + "_5m.parquet"),
            columns=["open", "high", "low", "close", "volume", "quote_volume", "price_usable"],
        )
        for day in days:
            bars = f.loc[(f.index >= day) & (f.index < day + pd.Timedelta(days=1))]
            if len(bars) != 288 or not bars.price_usable.all():
                raise ValueError("Alternative daily source is not full actual calendar")
            values = [
                bars.open.iloc[0],
                bars.high.max(),
                bars.low.min(),
                bars.close.iloc[-1],
                bars.volume.sum(),
                bars.quote_volume.sum(),
            ]
            o, h, low, close, vol, quote = values
            valid = (
                vol > 0
                and quote > 0
                and low > 0
                and h >= max(o, close)
                and low <= min(o, close)
                and low * vol - 0.01 - 1e-6 * abs(h * vol)
                <= quote
                <= h * vol + 0.01 + 1e-6 * abs(h * vol)
            )
            alternative[s].loc[day, ["open", "high", "low", "close", "volume", "quote_volume"]] = (
                values
            )
            alternative[s].at[day, "price_liquidity_feature_usable"] = valid
    return {"daily_archive": raw, "daily_with_complete_5m_reconciliation": alternative}


def audit(cohort, output):
    if output.exists():
        raise FileExistsError("Preserve independent cohort audit")
    verify_manifest(cohort)
    plan = read(cohort / "plan.json")
    registry = read(cohort / "registry.json")
    cases, source_cells = 0, 0
    all_selected = {"BTCUSDT"}
    for variant, raw in prepare_raw(plan, cohort).items():
        vectors = {}
        for s, frame in raw.items():
            cal = pd.date_range(frame.index[0], frame.index[-1] + pd.Timedelta(days=1), freq="D")
            f = frame.reindex(cal)
            valid = (
                f.price_liquidity_feature_usable.eq(True) & f.volume.gt(0) & f.quote_volume.gt(0)
            ).to_numpy(copy=True)
            valid[0] = False
            close = f.close.to_numpy(dtype=float).copy()
            close[~valid] = np.nan
            vectors[s] = (
                cal,
                close,
                f.quote_volume.to_numpy(dtype=float),
                np.r_[0, valid.cumsum()],
            )
        btc_dates, btc, _, _ = vectors["BTCUSDT"]
        for record in read(cohort / (variant + "_cohort.json")):
            now = pd.Timestamp(record["decision"])
            visible = [
                c for c in registry["cards"] if pd.Timestamp(c["available_at_assumed"]) <= now
            ]
            replaced = {c["supersedes"] for c in visible if c["supersedes"]}
            ends = [c for c in visible if c["id"] not in replaced]
            eligible = []
            for s, (dates, close, quotes, cumulative) in vectors.items():
                if s == "BTCUSDT" or now < dates[0] or now > dates[-1]:
                    continue
                if any(
                    c["symbol"] == s and pd.Timestamp(c["available_at_assumed"]) <= now
                    for c in registry["type_cards"]
                ):
                    continue
                launches = [
                    c
                    for c in registry["launch_cards"]
                    if c["symbol"] == s
                    and pd.Timestamp(c["available_at_assumed"]) <= now
                    and pd.Timestamp(c["launch_at"]) <= now
                ]
                launch = (
                    max(launches, key=lambda c: pd.Timestamp(c["launch_at"])) if launches else None
                )
                relevant = [
                    c
                    for c in ends
                    if c["symbol"] == s and (launch is None or c["episode"] == launch["episode"])
                ]
                blocked = False
                for c in relevant:
                    prior, cut = c, c["entry_stop_at"]
                    while cut is None and prior["supersedes"]:
                        prior = next(r for r in visible if r["id"] == prior["supersedes"])
                        cut = prior["entry_stop_at"]
                    blocked |= now >= pd.Timestamp(c["settlement_at"]) or (
                        cut is not None and now >= pd.Timestamp(cut)
                    )
                if blocked:
                    continue
                end = dates.searchsorted(now)
                start = (
                    dates.searchsorted(pd.Timestamp(launch["launch_at"]).ceil("D")) if launch else 0
                )
                if cumulative[end] - cumulative[start] < 180 or end - start < 61:
                    continue
                history = close[end - 61 : end]
                mi = btc_dates.searchsorted(now)
                market = btc[mi - 61 : mi]
                if (
                    not np.isfinite(history).all()
                    or len(market) != 61
                    or not np.isfinite(market).all()
                ):
                    continue
                returns = market[1:] / market[:-1] - 1
                if np.var(returns, ddof=1) <= 1e-12:
                    continue
                q = float(np.mean(quotes[end - 28 : end]))
                if q >= 50_000_000:
                    eligible.append((s, q))
            chosen = [s for s, _ in sorted(eligible, key=lambda r: (-r[1], r[0]))[:30]]
            if chosen != record["selected"] or len(eligible) != record["eligible"]:
                raise ValueError(
                    "Independent positive-liquidity/episode cohort differs: "
                    + variant
                    + record["decision"]
                )
            cases += 1
            source_cells += len(chosen)
            all_selected.update(chosen)
    if sorted(all_selected) != read(cohort / "symbols.json"):
        raise ValueError("Acquisition union differs")
    result = {
        "schema": "sparse-identity-cohort-independent-audit.v1",
        "passed": True,
        "cohort_completion_sha256": sha(cohort / "completion.json"),
        "source_sha256": sha(Path(__file__)),
        "clock_source_cases": cases,
        "selected_membership_cells": source_cells,
        "symbols": len(all_selected),
        "same_members_and_eligible_counts": True,
        "positive_liquidity_age_and_ticker_reuse_checked": True,
        "historical_original_versions_certified": False,
        "financial_accounts": 0,
    }
    save(output, result)
    print(result, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    audit(a.cohort, a.output)
