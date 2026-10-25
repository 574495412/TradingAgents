"""Liulian micro-pulse absorption factor.

The factor scores short-lived OHLCV impulses near human/round-number anchors.
It is calibrated against leader labels, but the signal itself only uses
market data available at the bar close.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


DEFAULT_PARAMS = {
    "lookback_bars": 72.0,
    "max_anchor_distance_pct": 0.03,
    "min_vol_ratio_med30": 1.8,
    "min_vol_ratio10": 2.5,
    "min_bar_range_pct": 0.7,
}


def calculate_factor(df: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(0.0, index=getattr(df, "index", None))
    frame = df.copy().sort_index()
    close = pd.to_numeric(frame.get("close"), errors="coerce")
    high = pd.to_numeric(frame.get("high"), errors="coerce") if "high" in frame else close
    low = pd.to_numeric(frame.get("low"), errors="coerce") if "low" in frame else close
    open_ = pd.to_numeric(frame.get("open"), errors="coerce") if "open" in frame else close
    volume = (
        pd.to_numeric(frame.get("volume"), errors="coerce")
        if "volume" in frame
        else pd.Series(np.nan, index=frame.index)
    )

    lookback = max(12, int(_param(params, "lookback_bars")))
    recent_high = high.rolling(lookback, min_periods=min(12, lookback)).max()
    recent_low = low.rolling(lookback, min_periods=min(12, lookback)).min()
    width = (recent_high - recent_low).replace(0.0, np.nan)
    range_position = ((close - recent_low) / width).clip(0.0, 1.0)
    rebound = (close / recent_low.replace(0.0, np.nan) - 1.0) * 100.0
    pullback = (recent_high / close.replace(0.0, np.nan) - 1.0) * 100.0
    ret1 = close.pct_change(1) * 100.0
    ret3 = close.pct_change(3) * 100.0
    bar_range = (high - low) / close.replace(0.0, np.nan) * 100.0
    lower_wick = (np.minimum(close, open_) - low) / close.replace(0.0, np.nan) * 100.0
    vol_ratio30 = volume / volume.rolling(30, min_periods=5).median().replace(0.0, np.nan)
    vol_ratio10 = volume / volume.rolling(10, min_periods=3).median().replace(0.0, np.nan)

    volume_impulse = (
        vol_ratio30.ge(_param(params, "min_vol_ratio_med30"))
        | vol_ratio10.ge(_param(params, "min_vol_ratio10"))
        | bar_range.ge(_param(params, "min_bar_range_pct"))
    )
    short_impulse = (
        (range_position.ge(0.68) & rebound.ge(0.45) & ret3.ge(-0.45))
        | (ret3.le(-0.65) & pullback.ge(1.0))
        | (range_position.between(0.35, 0.70, inclusive="both") & ret1.ge(0.18))
    )
    long_impulse = (
        (range_position.le(0.35) & pullback.ge(0.60) & ret3.le(0.25))
        | (ret3.le(-0.55) & lower_wick.ge(0.05))
    )
    short_score = (
        0.45 * (range_position - 0.50).clip(lower=0.0)
        + 0.30 * (rebound / 3.0).clip(0.0, 1.0)
        + 0.25 * (bar_range / 1.5).clip(0.0, 1.0)
    )
    long_score = (
        0.45 * (0.50 - range_position).clip(lower=0.0)
        + 0.30 * (pullback / 3.0).clip(0.0, 1.0)
        + 0.25 * (bar_range / 1.5).clip(0.0, 1.0)
    )
    score = long_score.where(long_impulse, 0.0) - short_score.where(short_impulse, 0.0)
    return score.where(volume_impulse, 0.0).fillna(0.0)


def _param(params: dict[str, Any] | None, name: str) -> float:
    try:
        return float((params or {}).get(name, DEFAULT_PARAMS[name]))
    except Exception:
        return float(DEFAULT_PARAMS[name])
