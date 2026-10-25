"""Liulian rebound-high short factor."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


DEFAULT_PARAMS = {
    "lookback_bars": 72.0,
    "min_range_position": 0.72,
    "min_rebound_pct": 0.60,
    "min_ret3_pct": 0.05,
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
    range_position = ((close - recent_low) / width).clip(0.0, 1.0)
    rebound = (close / recent_low.replace(0.0, np.nan) - 1.0) * 100.0
    ret3 = close.pct_change(3) * 100.0
    min_pos = _param(params, "min_range_position")
    min_rebound = _param(params, "min_rebound_pct")
    min_ret3 = _param(params, "min_ret3_pct")
    score = (
        0.45 * ((range_position - min_pos) / max(1.0 - min_pos, 0.01)).clip(0.0, 1.0)
        + 0.35 * ((rebound - min_rebound) / max(min_rebound * 2.0, 0.1)).clip(0.0, 1.0)
        + 0.20 * ((ret3 - min_ret3) / 0.60).clip(0.0, 1.0)
    )
    valid = range_position.ge(min_pos) & rebound.ge(min_rebound) & ret3.ge(min_ret3)
    return (-score.where(valid, 0.0)).fillna(0.0)


def _param(params: dict[str, Any] | None, name: str) -> float:
    try:
        return float((params or {}).get(name, DEFAULT_PARAMS[name]))
    except Exception:
        return float(DEFAULT_PARAMS[name])
