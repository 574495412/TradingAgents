"""Liulian trend roll continuation factor."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


DEFAULT_PARAMS = {
    "lookback_bars": 96.0,
    "min_bias_pct": 0.25,
    "min_move_pct": 0.70,
    "min_pullback_pct": 0.25,
}


def calculate_factor(df: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(0.0, index=getattr(df, "index", None))
    frame = df.copy().sort_index()
    close = pd.to_numeric(frame.get("close"), errors="coerce")
    lookback = max(12, int(_param(params, "lookback_bars")))
    recent_low = close.rolling(lookback, min_periods=min(12, lookback)).min()
    recent_high = close.rolling(lookback, min_periods=min(12, lookback)).max()
    pullback = (recent_high / close.replace(0.0, np.nan) - 1.0) * 100.0
    rebound = (close / recent_low.replace(0.0, np.nan) - 1.0) * 100.0
    ema_fast = close.ewm(span=max(3, min(12, lookback // 4)), adjust=False, min_periods=3).mean()
    ema_slow = close.ewm(span=max(6, min(48, lookback // 2)), adjust=False, min_periods=6).mean()
    bias = (ema_fast / ema_slow.replace(0.0, np.nan) - 1.0) * 100.0
    move = (close / close.shift(max(3, lookback // 2)).replace(0.0, np.nan) - 1.0) * 100.0
    min_bias = _param(params, "min_bias_pct")
    min_move = _param(params, "min_move_pct")
    min_pullback = _param(params, "min_pullback_pct")
    long_score = (
        0.45 * ((bias - min_bias) / max(min_bias * 2.0, 0.1)).clip(0.0, 1.0)
        + 0.35 * ((move - min_move) / max(min_move * 2.0, 0.1)).clip(0.0, 1.0)
        + 0.20 * ((pullback - min_pullback) / max(min_pullback * 2.0, 0.1)).clip(0.0, 1.0)
    )
    short_score = (
        0.45 * ((-bias - min_bias) / max(min_bias * 2.0, 0.1)).clip(0.0, 1.0)
        + 0.35 * ((-move - min_move) / max(min_move * 2.0, 0.1)).clip(0.0, 1.0)
        + 0.20 * ((rebound - min_pullback) / max(min_pullback * 2.0, 0.1)).clip(0.0, 1.0)
    )
    long_valid = close.ge(ema_slow) & bias.ge(min_bias) & move.ge(min_move) & pullback.ge(min_pullback)
    short_valid = close.le(ema_slow) & bias.le(-min_bias) & move.le(-min_move) & rebound.ge(min_pullback)
    return (long_score.where(long_valid, 0.0) - short_score.where(short_valid, 0.0)).fillna(0.0)


def _param(params: dict[str, Any] | None, name: str) -> float:
    try:
        return float((params or {}).get(name, DEFAULT_PARAMS[name]))
    except Exception:
        return float(DEFAULT_PARAMS[name])
