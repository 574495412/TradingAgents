"""Leader parabolic-fade short factor.

Recent closed leader labels with the largest short-side contribution clustered
around a narrower pattern than the older pump-fade factors: a very fast
3-5 minute extension, price far above the local 20 minute low, and a first
stall/weak-close bar near the local high.  This factor uses only as-of public
OHLCV/OI/funding fields.
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
    quote_volume = _safe_series(df, "quote_volume")
    taker_buy_quote = _safe_series(df, "taker_buy_quote")
    oi = _safe_series(df, "open_interest")
    funding = _safe_series(df, "funding_rate")

    min_output = _env_float("MEME_LEADER_PARABOLIC_FADE_MIN_OUTPUT", 0.30)
    max_output = _env_float("MEME_LEADER_PARABOLIC_FADE_MAX_OUTPUT", 0.92)
    min_ret3 = _env_float("MEME_LEADER_PARABOLIC_FADE_MIN_RET3_PCT", 5.8)
    min_ret5 = _env_float("MEME_LEADER_PARABOLIC_FADE_MIN_RET5_PCT", 8.0)
    min_from_low20 = _env_float("MEME_LEADER_PARABOLIC_FADE_MIN_FROM_LOW20_PCT", 18.0)
    min_from_low60 = _env_float("MEME_LEADER_PARABOLIC_FADE_MIN_FROM_LOW60_PCT", 14.0)
    max_pullback_high20 = abs(_env_float("MEME_LEADER_PARABOLIC_FADE_MAX_PULLBACK_HIGH20_PCT", 8.0))
    max_ret1 = _env_float("MEME_LEADER_PARABOLIC_FADE_MAX_RET1_PCT", 4.8)
    max_close_pos = _env_float("MEME_LEADER_PARABOLIC_FADE_MAX_CLOSE_POS", 0.76)
    min_upper_wick = _env_float("MEME_LEADER_PARABOLIC_FADE_MIN_UPPER_WICK", 0.05)
    min_vol_ratio = _env_float("MEME_LEADER_PARABOLIC_FADE_MIN_VOL_RATIO", 0.70)
    max_vol_ratio = _env_float("MEME_LEADER_PARABOLIC_FADE_MAX_VOL_RATIO", 120.0)
    max_taker_buy_ratio = _env_float("MEME_LEADER_PARABOLIC_FADE_MAX_TAKER_BUY_RATIO", 0.78)
    max_bar_range = _env_float("MEME_LEADER_PARABOLIC_FADE_MAX_BAR_RANGE_PCT", 34.0)

    ret1 = close.pct_change(1) * 100.0
    ret3 = close.pct_change(3) * 100.0
    ret5 = close.pct_change(5) * 100.0
    ret15 = close.pct_change(15) * 100.0
    high20 = high.rolling(20, min_periods=8).max()
    high60 = high.rolling(60, min_periods=20).max()
    low20 = low.rolling(20, min_periods=8).min()
    low60 = low.rolling(60, min_periods=20).min()
    from_high20 = (close / high20.replace(0.0, np.nan) - 1.0) * 100.0
    from_high60 = (close / high60.replace(0.0, np.nan) - 1.0) * 100.0
    from_low20 = (close / low20.replace(0.0, np.nan) - 1.0) * 100.0
    from_low60 = (close / low60.replace(0.0, np.nan) - 1.0) * 100.0

    vol_base = volume.rolling(30, min_periods=12).median().shift(1).replace(0.0, np.nan)
    vol_ratio = (volume / vol_base).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    vol_recent = vol_ratio.rolling(5, min_periods=2).max().fillna(vol_ratio)
    taker_buy_ratio = (
        taker_buy_quote / quote_volume.replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.50)
    oi_roc5 = oi.pct_change(5).replace([np.inf, -np.inf], np.nan).fillna(0.0) * 100.0

    bar_range = (high - low).replace(0.0, np.nan)
    close_pos = (close - low) / bar_range
    upper_wick = (high - pd.concat([open_, close], axis=1).max(axis=1)) / bar_range
    bar_range_pct = (high / low.replace(0.0, np.nan) - 1.0) * 100.0
    weak_close = (
        (close <= open_)
        | (ret1 <= 0.6)
        | (close_pos.fillna(0.5) <= max_close_pos)
        | (upper_wick.fillna(0.0) >= min_upper_wick)
    )

    extension = pd.concat(
        [
            _clip01((ret3.fillna(0.0) - min_ret3) / 10.0),
            _clip01((ret5.fillna(0.0) - min_ret5) / 14.0),
            _clip01((ret15.fillna(0.0) - 10.0) / 30.0),
            _clip01((from_low20.fillna(0.0) - min_from_low20) / 30.0),
        ],
        axis=1,
    ).max(axis=1)
    local_excess = (
        0.60 * _clip01((from_low20.fillna(0.0) - min_from_low20) / 36.0)
        + 0.40 * _clip01((from_low60.fillna(0.0) - min_from_low60) / 50.0)
    )
    near_high = (
        0.65 * (1.0 - _clip01((-from_high20.fillna(-999.0)) / max(max_pullback_high20, 0.5)))
        + 0.35 * (1.0 - _clip01((-from_high60.fillna(-999.0)) / 14.0))
    )
    stall = (
        0.32 * _clip01((max_ret1 - ret1.fillna(0.0)) / 7.0)
        + 0.24 * _clip01((max_close_pos - close_pos.fillna(0.5)) / max(max_close_pos, 0.1))
        + 0.22 * _clip01((upper_wick.fillna(0.0) - min_upper_wick) / 0.40)
        + 0.22 * (close <= open_ * 1.001).astype(float)
    )
    activity = _clip01((vol_recent - min_vol_ratio) / 5.0) * (
        1.0 - _clip01((vol_ratio - max_vol_ratio) / 80.0)
    )
    sell_pressure = _clip01((max_taker_buy_ratio - taker_buy_ratio) / max(max_taker_buy_ratio - 0.40, 0.08))
    crowding = 0.55 * _clip01(oi_roc5 / 18.0) + 0.45 * _clip01(funding.fillna(0.0).clip(lower=0.0) / 0.25)

    score = (
        0.30 * extension
        + 0.20 * local_excess
        + 0.18 * near_high
        + 0.16 * stall
        + 0.08 * activity
        + 0.05 * sell_pressure
        + 0.03 * crowding
    )
    valid = (
        ((ret3 >= min_ret3) | (ret5 >= min_ret5) | (from_low20 >= min_from_low20))
        & (ret5 >= 4.5)
        & (from_low20 >= min_from_low20)
        & (from_low60 >= min_from_low60)
        & (from_high20 >= -max_pullback_high20)
        & (ret1 <= max_ret1)
        & weak_close
        & (vol_recent >= min_vol_ratio)
        & (vol_ratio <= max_vol_ratio)
        & (taker_buy_ratio <= max_taker_buy_ratio)
        & (bar_range_pct <= max_bar_range)
    )

    out = pd.Series(0.0, index=df.index)
    out = out.mask(valid, -score.clip(lower=0.0, upper=max_output))
    return out.where(out.abs() >= min_output, 0.0).fillna(0.0)


def calculate_last_factor(df: pd.DataFrame) -> float:
    if df is None or len(df) < 90:
        return 0.0
    values = calculate_factor(df.tail(260))
    if values.empty:
        return 0.0
    try:
        return float(values.iloc[-1])
    except Exception:
        return 0.0
