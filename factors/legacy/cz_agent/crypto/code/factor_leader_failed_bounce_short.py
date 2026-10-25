"""Leader-style failed-bounce short factor.

High-yield meme lead-trader samples are often not pure bottom-chasing shorts.
They enter after a larger pump has started to roll over, then short the first
failed bounce while price is still close enough to the recent high to leave
downside room.  This factor models that structure using only as-of OHLCV data.
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
    if df is None or len(df) < 260:
        return pd.Series(0.0, index=getattr(df, "index", None))

    open_ = _safe_series(df, "open")
    high = _safe_series(df, "high")
    low = _safe_series(df, "low")
    close = _safe_series(df, "close")
    volume = _safe_series(df, "volume", 1.0)
    oi = _safe_series(df, "open_interest")
    funding = _safe_series(df, "funding_rate")

    min_output = _env_float("MEME_LEADER_FAILED_BOUNCE_MIN_OUTPUT", 0.26)
    max_output = _env_float("MEME_LEADER_FAILED_BOUNCE_MAX_OUTPUT", 0.86)
    min_ret240 = _env_float("MEME_LEADER_FAILED_BOUNCE_MIN_RET240_PCT", 2.5)
    min_ret60 = _env_float("MEME_LEADER_FAILED_BOUNCE_MIN_RET60_PCT", -2.0)
    min_from_high60 = _env_float("MEME_LEADER_FAILED_BOUNCE_MIN_FROM_HIGH60_PCT", -4.5)
    min_from_low240 = _env_float("MEME_LEADER_FAILED_BOUNCE_MIN_FROM_LOW240_PCT", 5.0)
    min_bounce20 = _env_float("MEME_LEADER_FAILED_BOUNCE_MIN_BOUNCE20_PCT", 0.6)
    max_bounce20 = _env_float("MEME_LEADER_FAILED_BOUNCE_MAX_BOUNCE20_PCT", 12.0)
    min_ret15 = _env_float("MEME_LEADER_FAILED_BOUNCE_MIN_RET15_PCT", -1.8)
    max_ret15 = _env_float("MEME_LEADER_FAILED_BOUNCE_MAX_RET15_PCT", 8.0)
    min_ret5 = _env_float("MEME_LEADER_FAILED_BOUNCE_MIN_RET5_PCT", -2.6)
    max_ret5 = _env_float("MEME_LEADER_FAILED_BOUNCE_MAX_RET5_PCT", 5.5)
    max_ret1 = _env_float("MEME_LEADER_FAILED_BOUNCE_MAX_RET1_PCT", 0.45)
    max_bar_range = _env_float("MEME_LEADER_FAILED_BOUNCE_MAX_BAR_RANGE_PCT", 8.0)
    min_recent_vol = _env_float("MEME_LEADER_FAILED_BOUNCE_MIN_RECENT_VOL_RATIO", 0.55)
    max_current_vol = _env_float("MEME_LEADER_FAILED_BOUNCE_MAX_CURRENT_VOL_RATIO", 18.0)

    ret1 = close.pct_change(1) * 100.0
    ret3 = close.pct_change(3) * 100.0
    ret5 = close.pct_change(5) * 100.0
    ret15 = close.pct_change(15) * 100.0
    ret60 = close.pct_change(60) * 100.0
    ret240 = close.pct_change(240) * 100.0

    high20 = high.rolling(20, min_periods=8).max()
    high60 = high.rolling(60, min_periods=20).max()
    low20 = low.rolling(20, min_periods=8).min()
    low60 = low.rolling(60, min_periods=20).min()
    low240 = low.rolling(240, min_periods=80).min()
    from_high20 = (close / high20.replace(0.0, np.nan) - 1.0) * 100.0
    from_high60 = (close / high60.replace(0.0, np.nan) - 1.0) * 100.0
    from_low20 = (close / low20.replace(0.0, np.nan) - 1.0) * 100.0
    from_low60 = (close / low60.replace(0.0, np.nan) - 1.0) * 100.0
    from_low240 = (close / low240.replace(0.0, np.nan) - 1.0) * 100.0

    ema8 = close.ewm(span=8, min_periods=5, adjust=False).mean()
    ema21 = close.ewm(span=21, min_periods=10, adjust=False).mean()

    vol_base = volume.rolling(30, min_periods=12).median().shift(1).replace(0.0, np.nan)
    vol_ratio = (volume / vol_base).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    vol_recent = vol_ratio.rolling(8, min_periods=3).max().fillna(vol_ratio)

    bar_range = (high - low).replace(0.0, np.nan)
    upper_wick = (high - pd.concat([open_, close], axis=1).max(axis=1)) / bar_range
    close_pos = (close - low) / bar_range
    bar_range_pct = (high / low.replace(0.0, np.nan) - 1.0) * 100.0

    oi_roc5 = oi.pct_change(5).replace([np.inf, -np.inf], np.nan).fillna(0.0) * 100.0
    funding_pos = funding.fillna(0.0).clip(lower=0.0)

    failed_bar = (
        (ret1 <= max_ret1)
        | (close <= open_)
        | ((close_pos.fillna(0.5) <= 0.58) & (upper_wick.fillna(0.0) >= 0.12))
    )
    pump_context = pd.concat(
        [
            (ret240.fillna(0.0) - min_ret240) / 14.0,
            (ret60.fillna(0.0) - min_ret60) / 10.0,
            (from_low240.fillna(0.0) - min_from_low240) / 18.0,
        ],
        axis=1,
    ).max(axis=1)
    not_oversold = (
        0.32 * _clip01((ret5.fillna(0.0) - min_ret5) / 4.0)
        + 0.22 * _clip01((ret15.fillna(0.0) - min_ret15) / 6.0)
        + 0.24 * _clip01((from_high60.fillna(-999.0) - min_from_high60) / 5.0)
        + 0.22 * (from_low20.fillna(0.0) >= min_bounce20).astype(float)
    )
    rejection = (
        0.38 * _clip01((upper_wick.fillna(0.0) - 0.10) / 0.45)
        + 0.34 * _clip01((0.66 - close_pos.fillna(0.5)) / 0.56)
        + 0.28 * failed_bar.astype(float)
    )
    trend_fail = (
        0.34 * (close <= ema8 * 1.002).astype(float)
        + 0.28 * (ema8 <= ema21 * 1.006).astype(float)
        + 0.20 * _clip01((-ret3.fillna(0.0) + 0.35) / 3.5)
        + 0.18 * _clip01((-from_high20.fillna(0.0)) / 4.0)
    )
    activity = _clip01((vol_recent - min_recent_vol) / 5.0) * (
        1.0 - _clip01((vol_ratio - max_current_vol) / max(max_current_vol, 1.0))
    )
    crowding = 0.55 * _clip01(oi_roc5 / 12.0) + 0.45 * _clip01(funding_pos / 0.20)

    score = (
        0.26 * _clip01(pump_context)
        + 0.24 * not_oversold
        + 0.22 * trend_fail
        + 0.16 * rejection
        + 0.08 * activity
        + 0.04 * crowding
    )

    valid = (
        ((ret240 >= min_ret240) | (ret60 >= min_ret60 + 3.0) | (from_low240 >= min_from_low240))
        & (ret60 >= min_ret60)
        & (ret15 >= min_ret15)
        & (ret15 <= max_ret15)
        & (ret5 >= min_ret5)
        & (ret5 <= max_ret5)
        & (from_high60 >= min_from_high60)
        & (from_low20 >= min_bounce20)
        & (from_low20 <= max_bounce20)
        & (from_low60 <= max(max_bounce20 * 1.8, 14.0))
        & (bar_range_pct <= max_bar_range)
        & (vol_recent >= min_recent_vol)
        & (vol_ratio <= max_current_vol)
        & failed_bar
    )

    out = pd.Series(0.0, index=df.index)
    out = out.mask(valid, -score.clip(lower=0.0, upper=max_output))
    return out.where(out.abs() >= min_output, 0.0).fillna(0.0)
