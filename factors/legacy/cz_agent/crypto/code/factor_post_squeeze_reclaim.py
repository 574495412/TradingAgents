"""Post-squeeze reclaim runner factor for Liulian core BTC/ETH legs.

The signal is fully OHLCV based: after price has pulled back from the recent
range high, it must reclaim momentum from the local low without chasing the
extreme top of the range.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


DEFAULT_PARAMS = {
    "lookback_bars": 96.0,
    "min_rebound_pct": 1.20,
    "min_pullback_pct": 0.35,
    "min_ret3_pct": 0.05,
    "max_ret3_pct": 0.80,
    "min_range_position": 0.55,
    "max_range_position": 0.90,
    "min_from_high_drop_pct": 0.30,
    "min_vol_ratio_med30": 0.70,
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
    rebound_pct = (close / recent_low.replace(0.0, np.nan) - 1.0) * 100.0
    pullback_pct = (recent_high / close.replace(0.0, np.nan) - 1.0) * 100.0
    ret3 = (close / close.shift(3).replace(0.0, np.nan) - 1.0) * 100.0
    high240 = high.rolling(240, min_periods=20).max()
    from_high240 = (close / high240.replace(0.0, np.nan) - 1.0) * 100.0
    vol_ratio = volume / volume.rolling(30, min_periods=5).median().replace(0.0, np.nan)

    min_pos = _param(params, "min_range_position")
    max_pos = _param(params, "max_range_position")
    valid = (
        range_position.between(min_pos, max_pos, inclusive="both")
        & rebound_pct.ge(_param(params, "min_rebound_pct"))
        & pullback_pct.ge(_param(params, "min_pullback_pct"))
        & ret3.ge(_param(params, "min_ret3_pct"))
        & ret3.le(_param(params, "max_ret3_pct"))
        & from_high240.le(-abs(_param(params, "min_from_high_drop_pct")))
        & vol_ratio.ge(_param(params, "min_vol_ratio_med30"))
    )
    score = (
        (rebound_pct / max(_param(params, "min_rebound_pct"), 1e-6)).clip(0.0, 2.0)
        + (pullback_pct / max(_param(params, "min_pullback_pct"), 1e-6)).clip(0.0, 2.0)
        + (ret3 / max(_param(params, "min_ret3_pct"), 1e-6)).clip(0.0, 2.0)
    ) / 6.0
    return score.where(valid, 0.0).fillna(0.0).astype(float)


def _param(params: dict[str, Any] | None, name: str) -> float:
    try:
        return float((params or {}).get(name, DEFAULT_PARAMS[name]))
    except Exception:
        return float(DEFAULT_PARAMS[name])
