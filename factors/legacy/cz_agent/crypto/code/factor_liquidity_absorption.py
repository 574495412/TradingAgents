"""
Liquidity absorption and distribution factor for meme coins.

The factor looks for high-volume rejection bars after a fast move:
- positive score: sell pressure is absorbed, price closes away from the low
- negative score: buy pressure is distributed, price closes away from the high

It is intentionally conservative. Most bars return zero so the factor confirms
strong setups or dampens weak ones instead of becoming a standalone churn engine.
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


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _safe_series(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").ffill().fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _clip01(series: pd.Series) -> pd.Series:
    return series.clip(lower=0.0, upper=1.0)


def calculate_factor(df: pd.DataFrame) -> pd.Series:
    if df is None or len(df) < 80:
        return pd.Series(0.0, index=getattr(df, "index", None))
    if not _env_bool("MEME_LIQUIDITY_ABSORPTION_ENABLED", False):
        return pd.Series(0.0, index=df.index)

    open_ = _safe_series(df, "open")
    high = _safe_series(df, "high")
    low = _safe_series(df, "low")
    close = _safe_series(df, "close")
    volume = _safe_series(df, "volume", 1.0)
    oi = _safe_series(df, "open_interest")
    funding = _safe_series(df, "funding_rate")

    bar_range = (high - low).replace(0.0, np.nan)
    upper_wick = (high - pd.concat([open_, close], axis=1).max(axis=1)) / bar_range
    lower_wick = (pd.concat([open_, close], axis=1).min(axis=1) - low) / bar_range
    close_pos = (close - low) / bar_range

    vol_base = volume.rolling(30, min_periods=12).median().replace(0.0, np.nan)
    vol_ratio = (volume / vol_base).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    ret1 = close.pct_change(1) * 100.0
    ret5 = close.pct_change(5) * 100.0
    ret20 = close.pct_change(20) * 100.0
    oi_roc5 = oi.pct_change(5) * 100.0

    ema8 = close.ewm(span=8, adjust=False, min_periods=5).mean()
    ema21 = close.ewm(span=21, adjust=False, min_periods=10).mean()
    trend_bias = ((ema8 / ema21.replace(0.0, np.nan)) - 1.0) * 100.0

    min_volume_ratio = _env_float("MEME_LIQUIDITY_ABSORPTION_MIN_VOLUME_RATIO", 1.25)
    min_wick_ratio = _env_float("MEME_LIQUIDITY_ABSORPTION_MIN_WICK_RATIO", 0.34)
    min_move_5 = _env_float("MEME_LIQUIDITY_ABSORPTION_MIN_MOVE_5_PCT", 1.0)
    min_move_20 = _env_float("MEME_LIQUIDITY_ABSORPTION_MIN_MOVE_20_PCT", 2.5)
    min_output = _env_float("MEME_LIQUIDITY_ABSORPTION_MIN_OUTPUT", 0.28)
    max_output = _env_float("MEME_LIQUIDITY_ABSORPTION_MAX_OUTPUT", 0.62)

    vol_score = _clip01((vol_ratio - min_volume_ratio) / 2.25)
    long_wick_score = _clip01((lower_wick.fillna(0.0) - min_wick_ratio) / 0.42)
    short_wick_score = _clip01((upper_wick.fillna(0.0) - min_wick_ratio) / 0.42)
    long_reclaim_score = _clip01((close_pos.fillna(0.5) - 0.48) / 0.42)
    short_reject_score = _clip01((0.52 - close_pos.fillna(0.5)) / 0.42)

    sell_pressure = pd.concat(
        [
            (-ret5.fillna(0.0) - min_move_5) / max(min_move_5 * 3.0, 1.0),
            (-ret20.fillna(0.0) - min_move_20) / max(min_move_20 * 3.0, 1.0),
        ],
        axis=1,
    ).max(axis=1)
    buy_pressure = pd.concat(
        [
            (ret5.fillna(0.0) - min_move_5) / max(min_move_5 * 3.0, 1.0),
            (ret20.fillna(0.0) - min_move_20) / max(min_move_20 * 3.0, 1.0),
        ],
        axis=1,
    ).max(axis=1)

    sell_pressure = _clip01(sell_pressure)
    buy_pressure = _clip01(buy_pressure)
    oi_relief = _clip01((-oi_roc5.fillna(0.0)) / 8.0)
    oi_distribution = _clip01(oi_roc5.fillna(0.0) / 10.0)
    funding_long = _clip01((-funding.fillna(0.0)) / 0.20)
    funding_short = _clip01(funding.fillna(0.0) / 0.20)

    long_score = (
        0.24 * long_wick_score
        + 0.20 * vol_score
        + 0.20 * long_reclaim_score
        + 0.18 * sell_pressure
        + 0.10 * oi_relief
        + 0.08 * funding_long
    )
    short_score = (
        0.24 * short_wick_score
        + 0.20 * vol_score
        + 0.20 * short_reject_score
        + 0.18 * buy_pressure
        + 0.10 * oi_distribution
        + 0.08 * funding_short
    )

    long_valid = (
        (vol_ratio >= min_volume_ratio)
        & (lower_wick >= min_wick_ratio)
        & (close_pos >= 0.50)
        & (ret1.fillna(0.0) >= -0.35)
        & ((ret5 <= -min_move_5) | (ret20 <= -min_move_20))
    )
    short_valid = (
        (vol_ratio >= min_volume_ratio)
        & (upper_wick >= min_wick_ratio)
        & (close_pos <= 0.50)
        & (ret1.fillna(0.0) <= 0.35)
        & ((ret5 >= min_move_5) | (ret20 >= min_move_20))
    )

    # Avoid fighting a clean trend unless the rejection bar is very strong.
    long_score = long_score.where((trend_bias.fillna(0.0) > -1.8) | (long_score >= 0.50), long_score * 0.65)
    short_score = short_score.where((trend_bias.fillna(0.0) < 1.8) | (short_score >= 0.50), short_score * 0.65)

    raw = pd.Series(0.0, index=df.index)
    raw = raw.mask(long_valid, long_score)
    raw = raw.mask(short_valid & (short_score > raw.abs()), -short_score)
    raw = raw.clip(lower=-max_output, upper=max_output)
    raw = raw.where(raw.abs() >= min_output, 0.0)
    return raw.fillna(0.0)
