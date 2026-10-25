"""Liulian breakdown-continuation short factor."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


DEFAULT_PARAMS = {
    "lookback_bars": 96.0,
    "min_ret20_drop_pct": 0.90,
    "min_from_high240_drop_pct": 1.10,
    "min_vol_ratio_med30": 2.50,
}


def calculate_factor(df: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(0.0, index=getattr(df, "index", None))
    frame = df.copy().sort_index()
    close = pd.to_numeric(frame.get("close"), errors="coerce")
    high = pd.to_numeric(frame.get("high"), errors="coerce").fillna(close)
    volume = pd.to_numeric(frame.get("volume"), errors="coerce").fillna(1.0)
    ret20 = close.pct_change(20) * 100.0
    high240 = high.rolling(240, min_periods=20).max()
    from_high240 = (close / high240.replace(0.0, np.nan) - 1.0) * 100.0
    vol_ratio = volume / volume.rolling(30, min_periods=5).median().replace(0.0, np.nan)
    min_drop = _param(params, "min_ret20_drop_pct")
    min_high_drop = _param(params, "min_from_high240_drop_pct")
    min_vol = _param(params, "min_vol_ratio_med30")
    score = (
        0.40 * ((-ret20 - min_drop) / max(min_drop * 2.0, 0.1)).clip(0.0, 1.0)
        + 0.35 * ((-from_high240 - min_high_drop) / max(min_high_drop * 2.0, 0.1)).clip(0.0, 1.0)
        + 0.25 * ((vol_ratio - min_vol) / max(min_vol, 0.1)).clip(0.0, 1.0)
    )
    valid = ret20.le(-min_drop) & from_high240.le(-min_high_drop) & vol_ratio.ge(min_vol)
    return (-score.where(valid, 0.0)).fillna(0.0)


def _param(params: dict[str, Any] | None, name: str) -> float:
    try:
        return float((params or {}).get(name, DEFAULT_PARAMS[name]))
    except Exception:
        return float(DEFAULT_PARAMS[name])
