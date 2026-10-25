"""Liulian late-session trend-absorption long factor.

The factor is an autonomous OHLCV rule mined from Liulian-style labels but it
does not read leader events. Positive values indicate long continuation entries
in a strong evening trend after shallow pullback absorption.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


DEFAULT_PARAMS: dict[str, float] = {
    "lookback_bars": 96.0,
    "min_bias_pct": 0.32,
    "min_move_pct": 1.20,
    "min_ret3_pct": 0.05,
    "min_range_position": 0.80,
    "max_range_position": 0.97,
    "max_pullback_pct": 0.55,
    "min_vol_ratio_med30": 0.60,
    "min_open_hour": 20.0,
    "max_open_hour": 23.0,
    "min_output": 0.35,
}


def _param(params: dict[str, Any] | None, name: str) -> float:
    source = params or {}
    try:
        return float(source.get(name, DEFAULT_PARAMS[name]))
    except Exception:
        return float(DEFAULT_PARAMS[name])


def _numeric(frame: pd.DataFrame, name: str, default: float = np.nan) -> pd.Series:
    if frame is not None and name in frame:
        return pd.to_numeric(frame[name], errors="coerce")
    return pd.Series(default, index=getattr(frame, "index", pd.Index([])), dtype=float)


def _clip01(value: pd.Series) -> pd.Series:
    return value.clip(lower=0.0, upper=1.0)


def calculate_liulian_trend_absorption_frame(
    df: pd.DataFrame,
    params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(index=getattr(df, "index", None))
    frame = df.copy().sort_index()
    frame.index = pd.to_datetime(frame.index, errors="coerce")
    frame = frame[frame.index.notna()].copy()
    if frame.empty:
        return pd.DataFrame(index=frame.index)

    lookback = max(12, int(_param(params, "lookback_bars")))
    close = _numeric(frame, "close")
    high = _numeric(frame, "high").fillna(close)
    low = _numeric(frame, "low").fillna(close)
    volume = _numeric(frame, "volume", 1.0).fillna(1.0)

    min_periods = max(3, min(lookback, 12))
    recent_high = high.rolling(lookback, min_periods=min_periods).max()
    recent_low = low.rolling(lookback, min_periods=min_periods).min()
    range_width = (recent_high - recent_low).replace(0.0, np.nan)
    range_position = ((close - recent_low) / range_width).clip(lower=0.0, upper=1.0)
    rebound_pct = (close / recent_low.replace(0.0, np.nan) - 1.0) * 100.0
    pullback_pct = (recent_high / close.replace(0.0, np.nan) - 1.0) * 100.0
    ret3 = close.pct_change(3) * 100.0
    ret20 = close.pct_change(20) * 100.0
    high240 = high.rolling(240, min_periods=20).max()
    from_high240 = (close / high240.replace(0.0, np.nan) - 1.0) * 100.0
    vol_ratio_med30 = volume / volume.rolling(30, min_periods=5).median().replace(0.0, np.nan)
    ema_fast = close.ewm(span=max(3, min(12, lookback // 4)), adjust=False, min_periods=3).mean()
    ema_slow = close.ewm(span=max(6, min(48, lookback // 2)), adjust=False, min_periods=6).mean()
    trend_bias_pct = (ema_fast / ema_slow.replace(0.0, np.nan) - 1.0) * 100.0
    trend_move_pct = (close / close.shift(max(3, lookback // 2)).replace(0.0, np.nan) - 1.0) * 100.0
    hours = pd.Series(frame.index.hour, index=frame.index, dtype=float)

    min_bias = max(0.0, _param(params, "min_bias_pct"))
    min_move = max(0.0, _param(params, "min_move_pct"))
    min_ret3 = _param(params, "min_ret3_pct")
    min_range = max(0.0, min(1.0, _param(params, "min_range_position")))
    max_range = max(min_range, min(1.0, _param(params, "max_range_position")))
    max_pullback = max(0.0, _param(params, "max_pullback_pct"))
    min_vol = max(0.0, _param(params, "min_vol_ratio_med30"))
    min_hour = max(0.0, min(23.0, _param(params, "min_open_hour")))
    max_hour = max(min_hour + 1.0, min(24.0, _param(params, "max_open_hour")))
    min_output = max(0.0, min(1.0, _param(params, "min_output")))

    range_score = _clip01((range_position - min_range) / max(max_range - min_range, 0.01))
    trend_score = 0.55 * _clip01((trend_move_pct - min_move) / 1.6) + 0.45 * _clip01((trend_bias_pct - min_bias) / 0.45)
    absorption_score = (
        0.38 * _clip01((ret3 - min_ret3) / 0.45)
        + 0.30 * _clip01((max_pullback - pullback_pct) / max(max_pullback, 0.05))
        + 0.20 * _clip01((vol_ratio_med30 - min_vol) / 3.0)
        + 0.12 * (close >= ema_slow).astype(float)
    )
    overheat_guard = (1.0 - 0.45 * _clip01((range_position - max_range) / 0.08)).clip(lower=0.0, upper=1.0)
    score = (0.38 * trend_score + 0.34 * absorption_score + 0.28 * range_score) * overheat_guard
    valid = (
        close.gt(0.0)
        & ema_slow.gt(0.0)
        & close.ge(ema_slow)
        & trend_bias_pct.ge(min_bias)
        & trend_move_pct.ge(min_move)
        & ret3.ge(min_ret3)
        & range_position.between(min_range, max_range, inclusive="both")
        & pullback_pct.le(max_pullback)
        & vol_ratio_med30.ge(min_vol)
        & hours.ge(min_hour)
        & hours.lt(max_hour)
    )
    factor = score.clip(lower=0.0, upper=1.0).where(valid, 0.0).fillna(0.0)
    factor = factor.where(factor >= min_output, 0.0)
    return pd.DataFrame(
        {
            "liulian_trend_absorption": factor,
            "close": close,
            "recent_high": recent_high,
            "recent_low": recent_low,
            "range_position": range_position,
            "rebound_pct": rebound_pct,
            "pullback_pct": pullback_pct,
            "ret3": ret3,
            "ret20": ret20,
            "from_high240": from_high240,
            "vol_ratio_med30": vol_ratio_med30,
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "trend_bias_pct": trend_bias_pct,
            "trend_move_pct": trend_move_pct,
        },
        index=frame.index,
    )


def calculate_factor(df: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.Series:
    frame = calculate_liulian_trend_absorption_frame(df, params=params)
    if frame.empty or "liulian_trend_absorption" not in frame:
        return pd.Series(0.0, index=getattr(df, "index", None))
    return frame["liulian_trend_absorption"].fillna(0.0)


def calculate_last_factor(df: pd.DataFrame, params: dict[str, Any] | None = None) -> float:
    values = calculate_factor(df, params=params)
    if values.empty:
        return 0.0
    try:
        return float(values.iloc[-1])
    except Exception:
        return 0.0
