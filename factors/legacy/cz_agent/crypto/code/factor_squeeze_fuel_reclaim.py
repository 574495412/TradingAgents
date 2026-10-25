"""Squeeze fuel reclaim factor.

This is a stricter companion to ``funding_rate_squeeze``.  It only fires when
extreme funding has fresh OI fuel and price has started to reclaim after a
controlled pullback.  The goal is to catch squeeze continuation earlier than a
late momentum chase while avoiding symbols where the squeeze is already spent.
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
    if df is None or len(df) < 90:
        return pd.Series(0.0, index=getattr(df, "index", None))
    if "funding_rate" not in df.columns or "open_interest" not in df.columns:
        return pd.Series(0.0, index=df.index)

    open_ = _safe_series(df, "open")
    high = _safe_series(df, "high")
    low = _safe_series(df, "low")
    close = _safe_series(df, "close")
    volume = _safe_series(df, "volume", 1.0)
    oi = _safe_series(df, "open_interest")
    funding = _safe_series(df, "funding_rate")

    ret3 = close.pct_change(3) * 100.0
    ret5 = close.pct_change(5) * 100.0
    ret10 = close.pct_change(10) * 100.0
    ret20 = close.pct_change(20) * 100.0
    ret60 = close.pct_change(60) * 100.0
    oi_roc5 = oi.pct_change(5).replace([np.inf, -np.inf], np.nan).fillna(0.0) * 100.0
    oi_roc20 = oi.pct_change(20).replace([np.inf, -np.inf], np.nan).fillna(0.0) * 100.0

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

    min_output = _env_float("MEME_SQUEEZE_FUEL_RECLAIM_MIN_OUTPUT", 0.25)
    max_output = _env_float("MEME_SQUEEZE_FUEL_RECLAIM_MAX_OUTPUT", 0.82)
    funding_abs_min = _env_float("MEME_SQUEEZE_FUEL_RECLAIM_MIN_FUNDING_ABS", 0.035)
    max_ret60_long = _env_float("MEME_SQUEEZE_FUEL_RECLAIM_MAX_RET60_LONG", 18.0)
    max_ret60_short = _env_float("MEME_SQUEEZE_FUEL_RECLAIM_MAX_RET60_SHORT", 18.0)

    funding_long = _clip01((-funding - funding_abs_min) / 0.35)
    funding_short = _clip01((funding - funding_abs_min) / 0.35)
    oi_fuel = _clip01((oi_roc5.fillna(0.0) + oi_roc20.fillna(0.0) * 0.35 - 0.8) / 10.0)
    vol_quality = _clip01((vol_ratio - 0.75) / 3.2) * (1.0 - _clip01((vol_ratio - 12.0) / 18.0))

    long_reclaim = (
        (close >= ema8 * 0.997)
        & (ema8 >= ema21 * 0.992)
        & (ret3.fillna(0.0) > -0.15)
        & (ret10.fillna(0.0) > 0.4)
    ).astype(float)
    short_reclaim = (
        (close <= ema8 * 1.003)
        & (ema8 <= ema21 * 1.008)
        & (ret3.fillna(0.0) < 0.15)
        & (ret10.fillna(0.0) < -0.4)
    ).astype(float)

    long_location = (1.0 - _clip01((-from_high20.fillna(0.0) - 6.0) / 12.0)) * (1.0 - _clip01((from_low60.fillna(0.0) - max_ret60_long) / 24.0))
    short_location = (1.0 - _clip01((from_low20.fillna(0.0) - 6.0) / 12.0)) * (1.0 - _clip01((-from_high60.fillna(0.0) - max_ret60_short) / 24.0))
    long_close = _clip01((close_pos.fillna(0.5) - 0.48) / 0.42) * (1.0 - _clip01(upper_wick.fillna(0.0) / 0.50))
    short_close = _clip01((0.52 - close_pos.fillna(0.5)) / 0.42) * (1.0 - _clip01(lower_wick.fillna(0.0) / 0.50))
    long_momentum = _clip01((ret5.fillna(0.0) + ret20.fillna(0.0) * 0.25) / 7.0)
    short_momentum = _clip01((-ret5.fillna(0.0) - ret20.fillna(0.0) * 0.25) / 7.0)

    long_score = (
        0.24 * funding_long
        + 0.20 * oi_fuel
        + 0.16 * long_reclaim
        + 0.14 * long_location
        + 0.10 * vol_quality
        + 0.10 * long_close
        + 0.06 * long_momentum
    )
    short_score = (
        0.24 * funding_short
        + 0.20 * oi_fuel
        + 0.16 * short_reclaim
        + 0.14 * short_location
        + 0.10 * vol_quality
        + 0.10 * short_close
        + 0.06 * short_momentum
    )

    long_valid = (
        (funding <= -funding_abs_min)
        & (oi_fuel > 0)
        & (long_reclaim > 0)
        & (ret60.fillna(0.0) <= max_ret60_long)
        & (ret5.fillna(0.0) <= 9.0)
    )
    short_valid = (
        (funding >= funding_abs_min)
        & (oi_fuel > 0)
        & (short_reclaim > 0)
        & (-ret60.fillna(0.0) <= max_ret60_short)
        & (-ret5.fillna(0.0) <= 8.0)
    )

    raw = pd.Series(0.0, index=df.index)
    raw = raw.mask(long_valid, long_score)
    raw = raw.mask(short_valid & (short_score > raw.abs()), -short_score)
    raw = raw.clip(lower=-max_output, upper=max_output)
    raw = raw.where(raw.abs() >= min_output, 0.0)
    return raw.fillna(0.0)
