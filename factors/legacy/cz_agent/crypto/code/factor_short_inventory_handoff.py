"""Liulian short-inventory handoff factor.

The factor looks for a BTC rebound that loses continuation after a liquidation
recovery.  It is an autonomous OHLCV score; leader records are labels only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


DEFAULT_PARAMS = {
    "lookback_bars": 72.0,
    "min_range_position": 0.80,
    "min_rebound_pct": 1.40,
    "max_pullback_pct": 0.35,
    "min_vol_ratio_med30": 3.0,
    "min_trend_bias_pct": 0.10,
    "min_ret3_pct": -0.10,
    "max_ret3_pct": 0.05,
}


def calculate_factor(df: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(0.0, index=getattr(df, "index", None))
    frame = df.copy().sort_index()
    close = pd.to_numeric(frame.get("close"), errors="coerce")
    high = pd.to_numeric(frame.get("high"), errors="coerce") if "high" in frame else close
    low = pd.to_numeric(frame.get("low"), errors="coerce") if "low" in frame else close
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
    ret3 = close.pct_change(3) * 100.0
    vol_ratio30 = volume / volume.rolling(30, min_periods=5).median().replace(0.0, np.nan)
    ema_fast = close.ewm(span=max(3, min(12, lookback // 4)), adjust=False, min_periods=3).mean()
    ema_slow = close.ewm(span=max(6, min(48, lookback // 2)), adjust=False, min_periods=6).mean()
    trend_bias = (ema_fast / ema_slow.replace(0.0, np.nan) - 1.0) * 100.0

    valid = (
        range_position.ge(_param(params, "min_range_position"))
        & rebound.ge(_param(params, "min_rebound_pct"))
        & pullback.le(_param(params, "max_pullback_pct"))
        & vol_ratio30.ge(_param(params, "min_vol_ratio_med30"))
        & trend_bias.ge(_param(params, "min_trend_bias_pct"))
        & ret3.ge(_param(params, "min_ret3_pct"))
        & ret3.le(_param(params, "max_ret3_pct"))
    )
    score = (
        0.35 * ((range_position - _param(params, "min_range_position")) / 0.20).clip(0.0, 1.0)
        + 0.25 * ((rebound - _param(params, "min_rebound_pct")) / 1.0).clip(0.0, 1.0)
        + 0.25 * (vol_ratio30 / max(_param(params, "min_vol_ratio_med30"), 1e-9)).clip(0.0, 2.0)
        + 0.15 * ((0.05 - ret3.abs()) / 0.05).clip(0.0, 1.0)
    )
    return (-score.where(valid, 0.0)).fillna(0.0)


def _param(params: dict[str, Any] | None, name: str) -> float:
    try:
        return float((params or {}).get(name, DEFAULT_PARAMS[name]))
    except Exception:
        return float(DEFAULT_PARAMS[name])
