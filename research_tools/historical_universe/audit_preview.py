"""Independent NumPy membership arithmetic and comparison with frozen40coin bars."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from research_tools.factor_lab.io import read, save, sha


def audit(root, output):
    for name, digest in read(root / "completion.json")["manifest"].items():
        if root.resolve() not in (root / name).resolve().parents or sha(root / name) != digest:
            raise ValueError("Daily release changed")
    plan_path = Path("research_tools/historical_universe/plan.json")
    if sha(plan_path) != read(root / "plan.json")["plan_source_sha256"]:
        raise ValueError("Preview eligibility registration changed")
    plan = read(plan_path)
    parameters = plan["eligibility_preview"]
    frames = {
        p.name.removesuffix("_1d.parquet"): pd.read_parquet(p)
        for p in (root / "symbols").glob("*_1d.parquet")
    }
    arrays = {
        s: (
            f.index.as_unit("ns").asi8,
            f.price_liquidity_feature_usable.to_numpy(),
            f.quote_volume.to_numpy(),
        )
        for s, f in frames.items()
    }
    previews = read(root / "eligibility_preview.json")
    errors = []
    maxerror = 0.0
    days = parameters["lookback_completed_days"]
    futurechecks = []
    for p in previews:
        t = pd.Timestamp(p["decision"])
        now = t.value
        before = now - days * 86400 * 1_000_000_000
        eligible = []
        for s, (times, valid, quote) in arrays.items():
            lo = int(np.searchsorted(times, before))
            hi = int(np.searchsorted(times, now))
            if (
                hi - lo != days
                or not valid[lo:hi].all()
                or int(valid[:hi].sum()) < parameters["minimum_observed_days"]
            ):
                continue
            if not np.array_equal(
                times[lo:hi], np.arange(before, now, 86400 * 1_000_000_000, dtype=np.int64)
            ):
                continue
            mean = float(np.sum(quote[lo:hi]) / days)
            if mean >= parameters["minimum_mean_daily_quote_volume"]:
                eligible.append((s, mean))
        ranked = sorted(eligible, key=lambda r: (-r[1], r[0]))[: parameters["maximum_names"]]
        if [s for s, _ in ranked] != p["selected"] or len(eligible) != p["eligible_names"]:
            errors.append(p["decision"])
        for s, v in ranked:
            maxerror = max(maxerror, abs(v - p["selected_quote28"][s]))
            if abs(v - p["selected_quote28"][s]) > max(1e-6, 1e-12 * abs(v)):
                raise ValueError("Independent volume mean differs")
    if errors:
        raise ValueError("Independent membership differs: " + str(errors[:5]))
    # Actual future volume perturbation: only decisions at/before each cut are compared.
    for cut in ["2021-01-01", "2023-01-01", "2025-01-01"]:
        c = pd.Timestamp(cut, tz="UTC").value
        for p in previews:
            t = pd.Timestamp(p["decision"]).value
            if t > c:
                continue
            for times, _valid, quote in arrays.values():
                hi = int(np.searchsorted(times, t))
                lo = int(np.searchsorted(times, t - days * 86400 * 1_000_000_000))
                altered = quote.copy()
                altered[times >= c] *= 10000
                if not np.array_equal(quote[lo:hi], altered[lo:hi], equal_nan=True):
                    raise ValueError("Future data enters past-only preview")
        futurechecks.append({"cut": cut, "past_windows_unchanged": True})
    # Existing40coin5m aggregates are a separate interval source, not a trusted replacement.
    old = Path("data/research/github_factor_market_20261025_v001")
    prior = read(old / "completion.json")["manifest"]
    comparison = []
    for s in read(old / "prepare_plan.json")["symbols"]:
        name = s + "_5m.parquet"
        if sha(old / name) != prior[name]:
            raise ValueError("Frozen5mcomparison data changed")
        f = pd.read_parquet(old / name, columns=["close", "quote_volume"])
        g = f.resample("D").agg({"close": "last", "quote_volume": "sum"})
        counts = f.close.resample("D").count()
        g = g.loc[counts.eq(288)]
        daily = frames[s]
        both = g.index.intersection(daily.index)
        close_error = np.abs(g.loc[both, "close"].to_numpy() - daily.loc[both, "close"].to_numpy())
        quote_error = np.abs(
            g.loc[both, "quote_volume"].to_numpy() - daily.loc[both, "quote_volume"].to_numpy()
        )
        scale = np.maximum(1.0, np.abs(g.loc[both, "quote_volume"].to_numpy()))
        bad = (close_error > 1e-8) | (quote_error > np.maximum(0.01, scale * 1e-6))
        comparison.append(
            {
                "symbol": s,
                "compared_full_days": len(both),
                "different_days": int(bad.sum()),
                "maximum_close_error": float(close_error.max()),
                "maximum_quote_relative_error": float((quote_error / scale).max()),
                "dates_differing": [t.isoformat() for t in both[bad]],
            }
        )
    result = {
        "schema": "historical-universe-preview-independent.v1",
        "membership_arithmetic_passed": True,
        "decisions": len(previews),
        "symbols": len(frames),
        "maximum_quote_mean_error": maxerror,
        "future_perturbation_checks": futurechecks,
        "independent_interval_comparison": comparison,
        "daily_vs5m_all_compared_days_agree": all(r["different_days"] == 0 for r in comparison),
        "daily_completion_sha256": sha(root / "completion.json"),
        "audit_code_sha256": sha(Path(__file__)),
        "financial_accounts": 0,
        "historical_trading_status_certified": False,
        "point_in_time_selection_fully_certified": False,
    }
    save(output, result)
    print({k: v for k, v in result.items() if not isinstance(v, list)}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--daily", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    audit(a.daily, a.output)
