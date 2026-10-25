"""Event retest reclaim factor for meme continuation entries.

This factor targets a structure that the pure micro-burst route often misses:
after an anomaly impulse, price pauses or pulls back for a few minutes, then
reclaims the short EMA with controlled volume and wick structure.  It uses only
current and historical bars so live and backtest can share it safely.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _safe_series(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in df:
        return pd.to_numeric(df[name], errors="coerce").ffill().fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _clip01(value: pd.Series) -> pd.Series:
    return value.clip(lower=0.0, upper=1.0)


def calculate_factor(df: pd.DataFrame) -> pd.Series:
    if df is None or len(df) < 100:
        return pd.Series(0.0, index=getattr(df, "index", None))

    open_ = _safe_series(df, "open")
    high = _safe_series(df, "high")
    low = _safe_series(df, "low")
    close = _safe_series(df, "close")
    volume = _safe_series(df, "volume", 1.0)
    oi = _safe_series(df, "open_interest")
    funding = _safe_series(df, "funding_rate")

    ret1 = close.pct_change(1) * 100.0
    ret3 = close.pct_change(3) * 100.0
    ret5 = close.pct_change(5) * 100.0
    ret15 = close.pct_change(15) * 100.0
    ret30 = close.pct_change(30) * 100.0
    ret60 = close.pct_change(60) * 100.0
    prev_ret5 = close.shift(5).pct_change(5) * 100.0
    acceleration = ret5 - prev_ret5

    ema8 = close.ewm(span=8, adjust=False, min_periods=5).mean()
    ema21 = close.ewm(span=21, adjust=False, min_periods=10).mean()
    high20 = high.rolling(20, min_periods=10).max()
    low20 = low.rolling(20, min_periods=10).min()
    high60 = high.rolling(60, min_periods=24).max()
    low60 = low.rolling(60, min_periods=24).min()
    from_high20 = (close / high20.replace(0.0, np.nan) - 1.0) * 100.0
    from_low20 = (close / low20.replace(0.0, np.nan) - 1.0) * 100.0
    from_high60 = (close / high60.replace(0.0, np.nan) - 1.0) * 100.0
    from_low60 = (close / low60.replace(0.0, np.nan) - 1.0) * 100.0

    bar_range = (high - low).replace(0.0, np.nan)
    upper_wick = (high - pd.concat([open_, close], axis=1).max(axis=1)) / bar_range
    lower_wick = (pd.concat([open_, close], axis=1).min(axis=1) - low) / bar_range
    close_pos = (close - low) / bar_range

    vol_base = volume.rolling(30, min_periods=12).median().shift(1).replace(0.0, np.nan)
    vol_ratio = (volume / vol_base).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    oi_roc5 = oi.pct_change(5).replace([np.inf, -np.inf], np.nan).fillna(0.0) * 100.0
    oi_roc20 = oi.pct_change(20).replace([np.inf, -np.inf], np.nan).fillna(0.0) * 100.0

    min_output = _env_float("MEME_EVENT_RETEST_RECLAIM_MIN_OUTPUT", 0.24)
    max_output = _env_float("MEME_EVENT_RETEST_RECLAIM_MAX_OUTPUT", 0.76)
    max_ret5_long = _env_float("MEME_EVENT_RETEST_RECLAIM_MAX_RET5_LONG", 8.0)
    max_ret5_short = _env_float("MEME_EVENT_RETEST_RECLAIM_MAX_RET5_SHORT", 7.0)
    max_bounce_long = _env_float("MEME_EVENT_RETEST_RECLAIM_MAX_BOUNCE_LONG", 34.0)
    max_drop_short = _env_float("MEME_EVENT_RETEST_RECLAIM_MAX_DROP_SHORT", 30.0)

    trend_long = _clip01((ret30.fillna(0.0) + ret60.fillna(0.0) * 0.45 - 3.0) / 24.0)
    trend_short = _clip01((-ret30.fillna(0.0) - ret60.fillna(0.0) * 0.45 - 3.0) / 24.0)

    pullback_long = _clip01((-from_high20.fillna(0.0) - 0.25) / 5.5) * (1.0 - _clip01((-from_high20.fillna(0.0) - 9.0) / 10.0))
    pullback_short = _clip01((from_low20.fillna(0.0) - 0.25) / 5.5) * (1.0 - _clip01((from_low20.fillna(0.0) - 9.0) / 10.0))

    reclaim_long = (
        (close >= ema8 * 0.998)
        & (ema8 >= ema21 * 0.995)
        & (ret3.fillna(0.0) >= -0.20)
        & (ret1.fillna(0.0) >= -0.10)
    ).astype(float)
    reclaim_short = (
        (close <= ema8 * 1.002)
        & (ema8 <= ema21 * 1.005)
        & (ret3.fillna(0.0) <= 0.20)
        & (ret1.fillna(0.0) <= 0.10)
    ).astype(float)

    controlled_volume = _clip01((vol_ratio - 0.85) / 3.2) * (1.0 - _clip01((vol_ratio - 8.5) / 12.0))
    long_close_quality = _clip01((close_pos.fillna(0.5) - 0.45) / 0.42) * (1.0 - _clip01(upper_wick.fillna(0.0) / 0.55))
    short_close_quality = _clip01((0.55 - close_pos.fillna(0.5)) / 0.42) * (1.0 - _clip01(lower_wick.fillna(0.0) / 0.55))
    long_oi = _clip01((oi_roc5.fillna(0.0) + oi_roc20.fillna(0.0) * 0.35) / 14.0)
    short_oi = _clip01((oi_roc5.fillna(0.0) + oi_roc20.fillna(0.0) * 0.35) / 16.0)
    long_accel = _clip01((acceleration.fillna(0.0) + ret5.fillna(0.0) * 0.35) / 5.5)
    short_accel = _clip01((-acceleration.fillna(0.0) - ret5.fillna(0.0) * 0.35) / 5.5)

    long_score = (
        0.20 * trend_long
        + 0.18 * pullback_long
        + 0.18 * reclaim_long
        + 0.14 * controlled_volume
        + 0.12 * long_close_quality
        + 0.08 * long_accel
        + 0.06 * long_oi
        + 0.04 * _clip01(-funding.fillna(0.0) / 0.25)
    )
    short_score = (
        0.20 * trend_short
        + 0.18 * pullback_short
        + 0.18 * reclaim_short
        + 0.14 * controlled_volume
        + 0.12 * short_close_quality
        + 0.08 * short_accel
        + 0.06 * short_oi
        + 0.04 * _clip01(funding.fillna(0.0) / 0.25)
    )

    long_valid = (
        (trend_long > 0)
        & (pullback_long > 0)
        & (reclaim_long > 0)
        & (ret5.fillna(0.0) <= max_ret5_long)
        & (from_low60.fillna(0.0) <= max_bounce_long)
        & (ret15.fillna(0.0) > -3.0)
    )
    short_valid = (
        (trend_short > 0)
        & (pullback_short > 0)
        & (reclaim_short > 0)
        & (-ret5.fillna(0.0) <= max_ret5_short)
        & (-from_high60.fillna(0.0) <= max_drop_short)
        & (ret15.fillna(0.0) < 3.0)
    )

    raw = pd.Series(0.0, index=df.index)
    raw = raw.mask(long_valid, long_score)
    raw = raw.mask(short_valid & (short_score > raw.abs()), -short_score)
    raw = raw.clip(lower=-max_output, upper=max_output)
    raw = raw.where(raw.abs() >= min_output, 0.0)
    return raw.fillna(0.0)
