"""Liulian mainstream relative-strength rotation factor.

The live overlay computes the actual cross-symbol breadth signal from the full
mainstream basket.  This per-symbol factor keeps the portfolio registry
loadable and exposes an interpretable standalone momentum-pullback proxy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


DEFAULT_PARAMS = {
    "ret_fast_bars": 12.0,
    "ret_slow_bars": 36.0,
    "range_lookback_bars": 96.0,
    "min_range_position": 0.28,
    "max_range_position": 0.78,
}


def calculate_factor(df: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(0.0, index=getattr(df, "index", None))
    frame = df.copy().sort_index()
    close = pd.to_numeric(frame.get("close"), errors="coerce")
    if close.empty:
        return pd.Series(0.0, index=frame.index)
    ret_fast_bars = max(3, int(_param(params, "ret_fast_bars")))
    ret_slow_bars = max(ret_fast_bars + 1, int(_param(params, "ret_slow_bars")))
    range_lookback = max(12, int(_param(params, "range_lookback_bars")))
    fast_ret = (close / close.shift(ret_fast_bars).replace(0.0, np.nan) - 1.0) * 100.0
    slow_ret = (close / close.shift(ret_slow_bars).replace(0.0, np.nan) - 1.0) * 100.0
    recent_low = close.rolling(range_lookback, min_periods=min(12, range_lookback)).min()
    recent_high = close.rolling(range_lookback, min_periods=min(12, range_lookback)).max()
    width = (recent_high - recent_low).replace(0.0, np.nan)
    range_position = ((close - recent_low) / width).clip(0.0, 1.0)
    min_pos = _param(params, "min_range_position")
    max_pos = _param(params, "max_range_position")
    range_ok = range_position.between(min_pos, max_pos, inclusive="both")
    trend_score = (0.6 * fast_ret + 0.4 * slow_ret).clip(-3.0, 3.0) / 3.0
    return trend_score.where(range_ok, 0.0).fillna(0.0)


def _param(params: dict[str, Any] | None, name: str) -> float:
    try:
        return float((params or {}).get(name, DEFAULT_PARAMS[name]))
    except Exception:
        return float(DEFAULT_PARAMS[name])
