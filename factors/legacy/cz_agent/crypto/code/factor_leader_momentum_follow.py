"""Leader-inspired momentum follow factor for meme bursts.

This research factor targets the missing structure from recent leader samples:
after an anomaly is discovered, follow the first confirmed continuation rather
than waiting for exhaustion.  It is deliberately symmetric:

* long: short-term momentum is accelerating, price holds above fast EMA, volume
  expands without a blow-off rejection candle;
* short: short-term momentum is accelerating lower, price holds below fast EMA,
  volume expands without a capitulation rebound candle.

Only as-of OHLCV/OI/funding fields are used.
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
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").ffill().fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _clip01(value: pd.Series) -> pd.Series:
    return value.clip(lower=0.0, upper=1.0)


def calculate_factor(df: pd.DataFrame) -> pd.Series:
    if df is None or len(df) < 90:
        return pd.Series(0.0, index=getattr(df, "index", None))

    open_ = _safe_series(df, "open")
    high = _safe_series(df, "high")
    low = _safe_series(df, "low")
    close = _safe_series(df, "close")
    volume = _safe_series(df, "volume", 1.0)
    oi = _safe_series(df, "open_interest")
    min_output = _env_float("MEME_LEADER_MOMENTUM_FOLLOW_MIN_OUTPUT", 0.30)
    max_output = _env_float("MEME_LEADER_MOMENTUM_FOLLOW_MAX_OUTPUT", 0.92)
    min_vol_ratio = _env_float("MEME_LEADER_MOMENTUM_FOLLOW_MIN_VOL_RATIO", 1.6)
    max_vol_ratio = _env_float("MEME_LEADER_MOMENTUM_FOLLOW_MAX_VOL_RATIO", 35.0)
    min_ret3 = _env_float("MEME_LEADER_MOMENTUM_FOLLOW_MIN_RET3_PCT", 1.0)
    min_ret5 = _env_float("MEME_LEADER_MOMENTUM_FOLLOW_MIN_RET5_PCT", 2.0)
    min_ret15 = _env_float("MEME_LEADER_MOMENTUM_FOLLOW_MIN_RET15_PCT", 3.4)
    max_ret5 = _env_float("MEME_LEADER_MOMENTUM_FOLLOW_MAX_RET5_PCT", 18.0)
    max_pullback = _env_float("MEME_LEADER_MOMENTUM_FOLLOW_MAX_PULLBACK_PCT", 5.0)
    max_bounce = _env_float("MEME_LEADER_MOMENTUM_FOLLOW_MAX_BOUNCE_PCT", 5.0)
    max_bar_range = _env_float("MEME_LEADER_MOMENTUM_FOLLOW_MAX_BAR_RANGE_PCT", 12.0)
    max_rejection_wick = _env_float("MEME_LEADER_MOMENTUM_FOLLOW_MAX_REJECTION_WICK", 0.46)
    min_close_pos_long = _env_float("MEME_LEADER_MOMENTUM_FOLLOW_MIN_CLOSE_POS_LONG", 0.46)
    max_close_pos_short = _env_float("MEME_LEADER_MOMENTUM_FOLLOW_MAX_CLOSE_POS_SHORT", 0.54)

    ret1 = close.pct_change(1) * 100.0
    ret3 = close.pct_change(3) * 100.0
    ret5 = close.pct_change(5) * 100.0
    ret15 = close.pct_change(15) * 100.0
    ret30 = close.pct_change(30) * 100.0

    ema5 = close.ewm(span=5, min_periods=3, adjust=False).mean()
    ema13 = close.ewm(span=13, min_periods=6, adjust=False).mean()
    ema34 = close.ewm(span=34, min_periods=12, adjust=False).mean()
    ema_fast_bias = (ema5 / ema13.replace(0.0, np.nan) - 1.0) * 100.0
    ema_trend_bias = (ema13 / ema34.replace(0.0, np.nan) - 1.0) * 100.0

    high20 = high.rolling(20, min_periods=8).max()
    low20 = low.rolling(20, min_periods=8).min()
    from_high20 = (close / high20.replace(0.0, np.nan) - 1.0) * 100.0
    from_low20 = (close / low20.replace(0.0, np.nan) - 1.0) * 100.0

    vol_base = volume.rolling(30, min_periods=12).median().shift(1).replace(0.0, np.nan)
    vol_ratio = (volume / vol_base).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    vol_recent = vol_ratio.rolling(5, min_periods=2).max().fillna(vol_ratio)
    oi_roc5 = oi.pct_change(5).replace([np.inf, -np.inf], np.nan).fillna(0.0) * 100.0

    bar_range = (high - low).replace(0.0, np.nan)
    upper_wick = (high - pd.concat([open_, close], axis=1).max(axis=1)) / bar_range
    lower_wick = (pd.concat([open_, close], axis=1).min(axis=1) - low) / bar_range
    close_pos = (close - low) / bar_range
    bar_range_pct = (high / low.replace(0.0, np.nan) - 1.0) * 100.0

    long_momentum = pd.concat(
        [
            (ret3.fillna(0.0) - min_ret3) / max(min_ret3 * 5.0, 1.0),
            (ret5.fillna(0.0) - min_ret5) / max(min_ret5 * 4.0, 1.0),
            (ret15.fillna(0.0) - min_ret15) / max(min_ret15 * 3.5, 1.0),
        ],
        axis=1,
    ).max(axis=1)
    short_momentum = pd.concat(
        [
            ((-ret3.fillna(0.0)) - min_ret3) / max(min_ret3 * 5.0, 1.0),
            ((-ret5.fillna(0.0)) - min_ret5) / max(min_ret5 * 4.0, 1.0),
            ((-ret15.fillna(0.0)) - min_ret15) / max(min_ret15 * 3.5, 1.0),
        ],
        axis=1,
    ).max(axis=1)

    volume_fuel = _clip01((vol_recent - min_vol_ratio) / 7.0)
    oi_fuel = _clip01(oi_roc5.abs() / 12.0)
    long_trend = 0.55 * _clip01(ema_fast_bias / 1.2) + 0.45 * _clip01(ema_trend_bias / 1.8)
    short_trend = 0.55 * _clip01((-ema_fast_bias) / 1.2) + 0.45 * _clip01((-ema_trend_bias) / 1.8)
    long_accept = 0.60 * _clip01((close_pos.fillna(0.5) - min_close_pos_long) / 0.44) + 0.40 * (
        upper_wick.fillna(0.0) <= max_rejection_wick
    ).astype(float)
    short_accept = 0.60 * _clip01((max_close_pos_short - close_pos.fillna(0.5)) / 0.44) + 0.40 * (
        lower_wick.fillna(0.0) <= max_rejection_wick
    ).astype(float)
    long_room = 1.0 - _clip01((-from_high20.fillna(0.0) - max_pullback) / max(max_pullback, 1.0))
    short_room = 1.0 - _clip01((from_low20.fillna(0.0) - max_bounce) / max(max_bounce, 1.0))

    long_score = (
        0.34 * _clip01(long_momentum)
        + 0.20 * long_trend
        + 0.16 * volume_fuel
        + 0.14 * long_accept
        + 0.10 * long_room
        + 0.06 * oi_fuel
    )
    short_score = (
        0.34 * _clip01(short_momentum)
        + 0.20 * short_trend
        + 0.16 * volume_fuel
        + 0.14 * short_accept
        + 0.10 * short_room
        + 0.06 * oi_fuel
    )

    long_valid = (
        (vol_recent >= min_vol_ratio)
        & (vol_ratio <= max_vol_ratio)
        & (ret3 >= min_ret3)
        & (ret5 >= min_ret5)
        & (ret15 >= min_ret15)
        & (ret5 <= max_ret5)
        & (ret1 >= -1.0)
        & (ema5 >= ema13)
        & (close >= ema5)
        & (from_high20 >= -abs(max_pullback))
        & (bar_range_pct <= max_bar_range)
        & (upper_wick.fillna(0.0) <= max_rejection_wick)
        & (close_pos.fillna(0.5) >= min_close_pos_long)
    )
    short_valid = (
        (vol_recent >= min_vol_ratio)
        & (vol_ratio <= max_vol_ratio)
        & (ret3 <= -min_ret3)
        & (ret5 <= -min_ret5)
        & (ret15 <= -min_ret15)
        & (ret5 >= -max_ret5)
        & (ret1 <= 1.0)
        & (ema5 <= ema13)
        & (close <= ema5)
        & (from_low20 <= abs(max_bounce))
        & (bar_range_pct <= max_bar_range)
        & (lower_wick.fillna(0.0) <= max_rejection_wick)
        & (close_pos.fillna(0.5) <= max_close_pos_short)
        & (ret30 <= 6.0)
    )

    out = pd.Series(0.0, index=df.index)
    out = out.mask(long_valid & (long_score >= short_score), long_score.clip(lower=0.0, upper=max_output))
    out = out.mask(short_valid & (short_score > long_score), -short_score.clip(lower=0.0, upper=max_output))
    return out.where(out.abs() >= min_output, 0.0).fillna(0.0)
