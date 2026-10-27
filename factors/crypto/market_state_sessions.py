"""Existing BTC state math with an explicitly decision-timestamped PIT breadth domain."""

import numpy as np
import pandas as pd

from factors.common.market_states import classify_market


def classify_with_decision_domain(frames, parameters, members, benchmark="BTCUSDT"):
    """Preserve frozen state thresholds; replace only breadth/count eligibility.

    members maps an actual decision timestamp to symbols already qualified using
    information available strictly before that decision. Benchmark features from
    the common classifier have already been shifted from completed observations.
    """
    inherited, _ = classify_market(frames, parameters, benchmark)
    index = inherited.index
    eligible = pd.DataFrame(False, index=index, columns=sorted(frames))
    for clock, symbols in members.items():
        if clock not in index or len(symbols) != len(set(symbols)):
            raise ValueError("Unique members on a real decision timestamp required")
        if set(symbols) - set(frames):
            raise ValueError("State domain has unknown source symbols")
        eligible.loc[clock, symbols] = True
    relative = pd.DataFrame(
        {
            s: f.close.reindex(index) / f.close.reindex(index).shift(parameters["momentum_window"])
            - 1
            for s, f in sorted(frames.items())
        }
    ).shift(1)
    valid = eligible & relative.notna() & np.isfinite(relative)
    out = inherited.drop(columns="state").copy()
    out["eligible_count"] = valid.sum(axis=1).astype(float)
    out["breadth"] = relative.gt(0).astype(float).where(valid).sum(
        axis=1, min_count=1
    ) / out.eligible_count.replace(0, np.nan)
    close = frames[benchmark].close.reindex(index)
    known_close = close.shift(1)
    momentum = (close / close.shift(parameters["momentum_window"]) - 1).shift(1)
    slow = (
        close.rolling(parameters["slow_ma_window"], min_periods=parameters["slow_ma_window"])
        .mean()
        .shift(1)
    )
    strong = out.trend_z.abs().ge(parameters["trend_z_min"]) & out.efficiency.ge(
        parameters["efficiency_min"]
    )
    state = pd.Series("range", index=index, dtype=object)
    state.loc[strong & momentum.gt(0) & known_close.gt(slow)] = "trend_up"
    state.loc[strong & momentum.lt(0) & known_close.lt(slow)] = "trend_down"
    state.loc[out.volatility_ratio.gt(1)] = "volatility_breakout"
    state.loc[out.eligible_count.lt(parameters["minimum_eligible"])] = "liquidity_stress"
    complete = out.notna().all(axis=1) & np.isfinite(out).all(axis=1) & slow.notna()
    state.loc[~complete] = "unknown"
    out["state"] = state
    return out, valid
