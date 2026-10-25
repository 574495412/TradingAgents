"""Fast runner quality factor.

The existing runner_quality factor is intentionally strict and often fires
only after a move is already obvious.  This companion factor looks for earlier
runner structure: multi-minute directional expansion, close near the impulse
edge, volume expansion, and either OI fuel or clean trend alignment.
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
    if df is None or len(df) < 80:
        return pd.Series(0.0, index=getattr(df, "index", None))

    open_ = _safe_series(df, "open")
    high = _safe_series(df, "high")
    low = _safe_series(df, "low")
    close = _safe_series(df, "close")
    volume = _safe_series(df, "volume", 1.0)
    oi = _safe_series(df, "open_interest")

    ret5 = close.pct_change(5) * 100.0
    ret10 = close.pct_change(10) * 100.0
    ret20 = close.pct_change(20) * 100.0
    ret60 = close.pct_change(60) * 100.0
    prev_ret5 = close.shift(5).pct_change(5) * 100.0
    accel = ret5 - prev_ret5

    ema8 = close.ewm(span=8, adjust=False, min_periods=5).mean()
    ema21 = close.ewm(span=21, adjust=False, min_periods=10).mean()
    ema_bias = (ema8 / ema21.replace(0.0, np.nan) - 1.0) * 100.0

    bar_range = (high - low).replace(0.0, np.nan)
    close_pos = (close - low) / bar_range
    upper_wick = (high - pd.concat([open_, close], axis=1).max(axis=1)) / bar_range
    lower_wick = (pd.concat([open_, close], axis=1).min(axis=1) - low) / bar_range
    high15 = high.rolling(15, min_periods=8).max()
    low15 = low.rolling(15, min_periods=8).min()
    pullback_from_15_high = (close / high15.replace(0.0, np.nan) - 1.0) * 100.0
    bounce_from_15_low = (close / low15.replace(0.0, np.nan) - 1.0) * 100.0

    vol_base = volume.rolling(24, min_periods=10).median().shift(1).replace(0.0, np.nan)
    vol_ratio = (volume / vol_base).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    oi_roc5 = oi.pct_change(5).replace([np.inf, -np.inf], np.nan).fillna(0.0) * 100.0
    oi_roc20 = oi.pct_change(20).replace([np.inf, -np.inf], np.nan).fillna(0.0) * 100.0

    min_output = _env_float("MEME_RUNNER_QUALITY_FAST_MIN_OUTPUT", 0.32)
    max_output = _env_float("MEME_RUNNER_QUALITY_FAST_MAX_OUTPUT", 0.86)
    ret10_min = _env_float("MEME_RUNNER_QUALITY_FAST_RET10_PCT", 7.0)
    ret20_min = _env_float("MEME_RUNNER_QUALITY_FAST_RET20_PCT", 13.0)
    ret60_min = _env_float("MEME_RUNNER_QUALITY_FAST_RET60_PCT", 24.0)
    max_ret5_long = _env_float("MEME_RUNNER_QUALITY_FAST_MAX_RET5_LONG", 18.0)
    max_ret5_short = _env_float("MEME_RUNNER_QUALITY_FAST_MAX_RET5_SHORT", 16.0)
    max_pullback_long = _env_float("MEME_RUNNER_QUALITY_FAST_MAX_PULLBACK_LONG", 5.5)
    max_bounce_short = _env_float("MEME_RUNNER_QUALITY_FAST_MAX_BOUNCE_SHORT", 5.5)

    long_trend = pd.concat(
        [
            (ret10.fillna(0.0) - ret10_min) / max(ret10_min, 1.0),
            (ret20.fillna(0.0) - ret20_min) / max(ret20_min, 1.0),
            (ret60.fillna(0.0) - ret60_min) / max(ret60_min, 1.0),
        ],
        axis=1,
    ).max(axis=1)
    short_trend = pd.concat(
        [
            (-ret10.fillna(0.0) - ret10_min) / max(ret10_min, 1.0),
            (-ret20.fillna(0.0) - ret20_min) / max(ret20_min, 1.0),
            (-ret60.fillna(0.0) - ret60_min) / max(ret60_min, 1.0),
        ],
        axis=1,
    ).max(axis=1)
    long_trend = _clip01(long_trend)
    short_trend = _clip01(short_trend)

    long_structure = (
        _clip01((close_pos.fillna(0.5) - 0.56) / 0.36)
        * (1.0 - _clip01(upper_wick.fillna(0.0) / 0.50))
        * (1.0 - _clip01((-pullback_from_15_high.fillna(0.0) - max_pullback_long) / 10.0))
    )
    short_structure = (
        _clip01((0.44 - close_pos.fillna(0.5)) / 0.36)
        * (1.0 - _clip01(lower_wick.fillna(0.0) / 0.50))
        * (1.0 - _clip01((bounce_from_15_low.fillna(0.0) - max_bounce_short) / 10.0))
    )
    vol_score = _clip01((vol_ratio - 1.1) / 4.0) * (1.0 - _clip01((vol_ratio - 16.0) / 22.0))
    oi_score = _clip01((oi_roc5.fillna(0.0) + oi_roc20.fillna(0.0) * 0.35) / 14.0)
    long_alignment = _clip01((ema_bias.fillna(0.0) + 0.08) / 1.2) * _clip01((accel.fillna(0.0) + 0.4) / 5.0)
    short_alignment = _clip01((-ema_bias.fillna(0.0) + 0.08) / 1.2) * _clip01((-accel.fillna(0.0) + 0.4) / 5.0)

    long_score = 0.30 * long_trend + 0.22 * long_structure + 0.16 * vol_score + 0.14 * oi_score + 0.18 * long_alignment
    short_score = 0.30 * short_trend + 0.22 * short_structure + 0.16 * vol_score + 0.14 * oi_score + 0.18 * short_alignment

    long_valid = (
        (long_trend > 0)
        & (ret5.fillna(0.0) <= max_ret5_long)
        & (pullback_from_15_high.fillna(0.0) >= -max_pullback_long)
        & ((oi_score > 0) | (vol_score > 0.20))
    )
    short_valid = (
        (short_trend > 0)
        & (-ret5.fillna(0.0) <= max_ret5_short)
        & (bounce_from_15_low.fillna(0.0) <= max_bounce_short)
        & ((oi_score > 0) | (vol_score > 0.20))
    )

    raw = pd.Series(0.0, index=df.index)
    raw = raw.mask(long_valid, long_score)
    raw = raw.mask(short_valid & (short_score > raw.abs()), -short_score)
    raw = raw.clip(lower=-max_output, upper=max_output)
    raw = raw.where(raw.abs() >= min_output, 0.0)
    return raw.fillna(0.0)
