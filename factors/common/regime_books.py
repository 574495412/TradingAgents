"""Five preregistered strategy intents from source-pinned daily mathematics.

These are portfolio intents, not profitable-strategy claims. Each row observes
only the prior completed session. Existing frozen factor panels are unchanged.
"""

import numpy as np
import pandas as pd

from .github_formulas import formulas

FAMILIES = ["trend", "relative_momentum", "carry_momentum", "mean_reversion", "breakout"]


def source_features(frame):
    out = formulas(frame)
    price = frame.close
    vol = out.relative_vol * price
    fast = (price.ewm(span=4).mean() - price.ewm(span=16).mean()) / vol.ffill()
    boundary = fast.rolling(5000, min_periods=3).std(ddof=1) * 3
    out["mr_wings4"] = -fast.mask(fast.abs() < boundary, 0.0)
    return out


def inverse_vol_intent(signal, volatility, threshold=0.0032, maximum=10):
    good = signal.loc[signal.abs() > threshold].dropna()
    names = sorted(good.index, key=lambda s: (-abs(good[s]), s))[:maximum]
    raw = pd.Series(
        {
            s: np.sign(good[s]) / max(float(volatility[s]), 0.001)
            for s in names
            if np.isfinite(volatility[s])
        }
    )
    size = raw.abs().sum()
    return raw / size if size > 0 else raw


def intents(frames, rates, eligible, normalization):
    features = {s: source_features(f) for s, f in sorted(frames.items())}
    index, symbols = eligible.index, eligible.columns
    out = {key: pd.DataFrame(0.0, index=index, columns=symbols) for key in FAMILIES}
    funding = (
        pd.DataFrame({s: rates[s].funding_rate.resample("D").sum(min_count=1) for s in symbols})
        .reindex(index)
        .rolling(5, min_periods=5)
        .mean()
        .shift(1)
    )
    known = {s: f.shift(1) for s, f in features.items()}
    for clock in index:
        names = [s for s in symbols if eligible.loc[clock, s]]
        volatility = pd.Series({s: known[s].loc[clock, "relative_vol"] for s in names})
        for key, factor in (
            ("trend", "pst_ewmac32"),
            ("breakout", "pst_breakout10"),
            ("mean_reversion", "mr_wings4"),
        ):
            scale = normalization[factor]
            if scale is None or scale <= 0:
                continue
            signal = pd.Series({s: known[s].loc[clock, factor] for s in names}, dtype=float)
            signal = (signal / scale).clip(-3, 3) * 0.01
            weights = inverse_vol_intent(signal, volatility)
            out[key].loc[clock, weights.index] = weights
        if len(names) < 10:
            continue
        momentum = pd.Series({s: known[s].loc[clock, "momentum21"] for s in names})
        if momentum.isna().any():
            continue
        for key, rank in (
            ("relative_momentum", momentum.rank(method="first")),
            (
                "carry_momentum",
                momentum.rank(method="first") + (-funding.loc[clock, names]).rank(method="first"),
            ),
        ):
            if rank.isna().any():
                continue
            ordered = rank.sort_values(ascending=False, kind="stable").index
            out[key].loc[clock, ordered[:5]] = 0.1
            out[key].loc[clock, ordered[-5:]] = -0.1
    return out, features
