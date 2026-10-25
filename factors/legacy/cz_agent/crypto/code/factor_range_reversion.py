"""Liulian range high-sell/low-buy factor."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


DEFAULT_PARAMS = {
    "lookback_bars": 96.0,
    "min_width_pct": 0.80,
    "high_position": 0.78,
    "low_position": 0.22,
}


def calculate_factor(df: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(0.0, index=getattr(df, "index", None))
    frame = df.copy().sort_index()
    close = pd.to_numeric(frame.get("close"), errors="coerce")
    lookback = max(12, int(_param(params, "lookback_bars")))
    recent_low = close.rolling(lookback, min_periods=min(12, lookback)).min()
    recent_high = close.rolling(lookback, min_periods=min(12, lookback)).max()
    width = (recent_high - recent_low).replace(0.0, np.nan)
    width_pct = width / close.replace(0.0, np.nan) * 100.0
    range_position = ((close - recent_low) / width).clip(0.0, 1.0)
    high_pos = _param(params, "high_position")
    low_pos = _param(params, "low_position")
    high_score = ((range_position - high_pos) / max(1.0 - high_pos, 0.01)).clip(0.0, 1.0)
    low_score = ((low_pos - range_position) / max(low_pos, 0.01)).clip(0.0, 1.0)
    valid = width_pct.ge(_param(params, "min_width_pct"))
    return (low_score.where(valid, 0.0) - high_score.where(valid, 0.0)).fillna(0.0)


def _param(params: dict[str, Any] | None, name: str) -> float:
    try:
        return float((params or {}).get(name, DEFAULT_PARAMS[name]))
    except Exception:
        return float(DEFAULT_PARAMS[name])
