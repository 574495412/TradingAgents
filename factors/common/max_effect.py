"""MAX28 source direction and matched momentum/BTC-beta diagnostic mathematics.

Windows count actual daily observations supplied by the market adapter. Only
crypto scheduling/costs are validated by this experiment; stock accounts differ.
"""

import numpy as np
import pandas as pd


def features(frames, benchmark="BTCUSDT"):
    market = frames[benchmark].close.pct_change(fill_method=None)
    variance = market.rolling(60, min_periods=60).var(ddof=1)
    result = {}
    for symbol, frame in frames.items():
        returns = frame.close.pct_change(fill_method=None)
        result[symbol] = pd.DataFrame(
            {
                "max28": returns.rolling(28, min_periods=28).max(),
                "momentum28": frame.close / frame.close.shift(28) - 1,
                "beta60": returns.rolling(60, min_periods=60).cov(market)
                / variance.where(variance > 1e-12),
                "quote28": frame.quote_volume.rolling(28, min_periods=28).mean(),
                "observations": frame.close.notna().cumsum(),
            }
        ).shift(1)
    return result


def book(feature_rows, parameters, candidate, benchmark="BTCUSDT"):
    if candidate not in [
        "max28",
        "max28_btc_beta",
        "momentum28_control",
        "momentum28_btc_beta_control",
    ]:
        raise ValueError("Unknown fixed MAX comparison")
    eligible = [
        s
        for s, row in feature_rows.items()
        if s != benchmark
        and row.observations >= parameters["minimum_listing_observations"]
        and row.quote28 >= parameters["minimum_daily_quote_volume"]
        and np.isfinite(row[["max28", "momentum28", "beta60"]].to_numpy(dtype=float)).all()
    ]
    eligible = sorted(eligible, key=lambda s: (-float(feature_rows[s].quote28), s))[
        : parameters["maximum_eligible"]
    ]
    weights = pd.Series(0.0, index=sorted(feature_rows))
    if len(eligible) < parameters["minimum_eligible"]:
        return weights, {
            "eligible": len(eligible),
            "status": "insufficient_observable_universe",
            "target_beta": 0.0,
        }
    key = "max28" if candidate.startswith("max28") else "momentum28"
    ordered = sorted(eligible, key=lambda s: (-float(feature_rows[s][key]), s))
    long = ordered[: parameters["long_names"]]
    short = ordered[-parameters["short_names"] :]
    for s in long:
        weights[s] = 0.5 / len(long)
    for s in short:
        weights[s] = -0.5 / len(short)
    betas = pd.Series({s: float(r.beta60) for s, r in feature_rows.items()})
    betas[benchmark] = 1.0
    alpha_beta = float((weights * betas.fillna(0)).sum())
    if "btc_beta" in candidate:
        weights[benchmark] = -alpha_beta
    gross = float(weights.abs().sum())
    if gross > 1:
        weights /= gross
    return weights, {
        "eligible": len(eligible),
        "status": "formed",
        "long": long,
        "short": short,
        "unhedged_target_beta": alpha_beta,
        "target_beta": float((weights * betas.fillna(0)).sum()),
        "gross_before_downscale": gross,
        "gross_after_downscale": float(weights.abs().sum()),
    }
