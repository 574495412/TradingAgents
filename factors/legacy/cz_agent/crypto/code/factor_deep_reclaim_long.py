"""Deep reclaim long factor for Liulian BTC/ETH core legs.

The signal is fully OHLCV based.  It looks for BTC/ETH after a deep
multi-hour drawdown, where price is still in the lower part of its local range
but has enough volume to make a reclaim attempt worth holding longer than a
scalp.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


DEFAULT_PARAMS = {
    "lookback_bars": 96.0,
    "max_range_position": 0.35,
    "min_ret20_pct": -2.0,
    "max_ret20_pct": 0.5,
    "min_ret3_pct": -1.5,
    "max_ret3_pct": 0.2,
    "min_from_high240_drop_pct": 3.0,
    "max_trend_bias_pct": -0.2,
    "min_vol_ratio_med30": 1.5,
    "min_pullback_pct": 0.6,
}


def calculate_factor(df: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(0.0, index=getattr(df, "index", None))
    frame = df.copy().sort_index()
    close = pd.to_numeric(frame.get("close"), errors="coerce")
    high = pd.to_numeric(frame.get("high"), errors="coerce") if "high" in frame else close
    low = pd.to_numeric(frame.get("low"), errors="coerce") if "low" in frame else close
    volume = pd.to_numeric(frame.get("volume"), errors="coerce") if "volume" in frame else pd.Series(np.nan, index=frame.index)

    lookback = max(24, int(_param(params, "lookback_bars")))
    recent_high = high.rolling(lookback, min_periods=min(24, lookback)).max()
    recent_low = low.rolling(lookback, min_periods=min(24, lookback)).min()
    width = (recent_high - recent_low).replace(0.0, np.nan)
    range_position = ((close - recent_low) / width).clip(0.0, 1.0)
    pullback_pct = (recent_high / close.replace(0.0, np.nan) - 1.0) * 100.0
    ret3 = (close / close.shift(3).replace(0.0, np.nan) - 1.0) * 100.0
    ret20 = (close / close.shift(20).replace(0.0, np.nan) - 1.0) * 100.0
    high240 = high.rolling(240, min_periods=20).max()
    from_high240 = (close / high240.replace(0.0, np.nan) - 1.0) * 100.0
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=48, adjust=False).mean()
    trend_bias = (ema_fast / ema_slow.replace(0.0, np.nan) - 1.0) * 100.0
    vol_ratio = volume / volume.rolling(30, min_periods=5).median().replace(0.0, np.nan)

    valid = (
        range_position.le(_param(params, "max_range_position"))
        & ret20.ge(_param(params, "min_ret20_pct"))
        & ret20.le(_param(params, "max_ret20_pct"))
        & ret3.ge(_param(params, "min_ret3_pct"))
        & ret3.le(_param(params, "max_ret3_pct"))
        & from_high240.le(-abs(_param(params, "min_from_high240_drop_pct")))
        & trend_bias.le(_param(params, "max_trend_bias_pct"))
        & vol_ratio.ge(_param(params, "min_vol_ratio_med30"))
        & pullback_pct.ge(_param(params, "min_pullback_pct"))
    )
    score = (
        (1.0 - range_position).clip(0.0, 1.0) * 0.30
        + (pullback_pct / 4.0).clip(0.0, 1.0) * 0.25
        + ((-from_high240) / 8.0).clip(0.0, 1.0) * 0.25
        + (vol_ratio / 4.0).clip(0.0, 1.0) * 0.20
    )
    return score.where(valid, 0.0).fillna(0.0).astype(float)


def _param(params: dict[str, Any] | None, name: str) -> float:
    try:
        return float((params or {}).get(name, DEFAULT_PARAMS[name]))
    except Exception:
        return float(DEFAULT_PARAMS[name])
