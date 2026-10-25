"""High-win-rate leader pump-fade short factor.

Recent frequent 90%+ win-rate leader samples are not only extreme blow-offs.
Many entries are short scalps after a mild 15m/60m rebound stalls near a local
high.  This factor models that structure from as-of OHLCV/OI/funding only; it
does not replay leader trades or live logs.
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

    min_output = _env_float("MEME_LEADER_HIGH_WIN_PUMP_FADE_MIN_OUTPUT", 0.24)
    max_output = _env_float("MEME_LEADER_HIGH_WIN_PUMP_FADE_MAX_OUTPUT", 0.88)
    min_ret15 = _env_float("MEME_LEADER_HIGH_WIN_PUMP_FADE_MIN_RET15_PCT", 0.35)
    min_ret60 = _env_float("MEME_LEADER_HIGH_WIN_PUMP_FADE_MIN_RET60_PCT", 2.0)
    min_from_low20 = _env_float("MEME_LEADER_HIGH_WIN_PUMP_FADE_MIN_FROM_LOW20_PCT", 1.2)
    min_from_low60 = _env_float("MEME_LEADER_HIGH_WIN_PUMP_FADE_MIN_FROM_LOW60_PCT", 2.0)
    max_pullback_high60 = abs(_env_float("MEME_LEADER_HIGH_WIN_PUMP_FADE_MAX_PULLBACK_HIGH60_PCT", 5.5))
    min_vol_ratio = _env_float("MEME_LEADER_HIGH_WIN_PUMP_FADE_MIN_VOL_RATIO", 0.55)
    max_vol_ratio = _env_float("MEME_LEADER_HIGH_WIN_PUMP_FADE_MAX_VOL_RATIO", 18.0)
    min_ret5 = _env_float("MEME_LEADER_HIGH_WIN_PUMP_FADE_MIN_RET5_PCT", -999.0)
    max_ret5 = _env_float("MEME_LEADER_HIGH_WIN_PUMP_FADE_MAX_RET5_PCT", 8.0)
    max_ret1 = _env_float("MEME_LEADER_HIGH_WIN_PUMP_FADE_MAX_RET1_PCT", 2.4)
    max_close_pos = _env_float("MEME_LEADER_HIGH_WIN_PUMP_FADE_MAX_CLOSE_POS", 0.82)
    min_upper_wick = _env_float("MEME_LEADER_HIGH_WIN_PUMP_FADE_MIN_UPPER_WICK", 0.05)
    max_taker_buy_ratio = _env_float("MEME_LEADER_HIGH_WIN_PUMP_FADE_MAX_TAKER_BUY_RATIO", 0.74)
    extreme_min_ret15 = _env_float("MEME_LEADER_HIGH_WIN_PUMP_FADE_EXTREME_MIN_RET15_PCT", 10.0)
    extreme_min_from_low20 = _env_float("MEME_LEADER_HIGH_WIN_PUMP_FADE_EXTREME_MIN_FROM_LOW20_PCT", 8.0)
    extreme_max_ret5 = _env_float("MEME_LEADER_HIGH_WIN_PUMP_FADE_EXTREME_MAX_RET5_PCT", 12.0)
    extreme_min_vol_ratio = _env_float("MEME_LEADER_HIGH_WIN_PUMP_FADE_EXTREME_MIN_VOL_RATIO", 0.80)

    ret1 = close.pct_change(1) * 100.0
    ret5 = close.pct_change(5) * 100.0
    ret15 = close.pct_change(15) * 100.0
    ret60 = close.pct_change(60) * 100.0
    ret240 = close.pct_change(240) * 100.0

    high20 = high.rolling(20, min_periods=8).max()
    high60 = high.rolling(60, min_periods=20).max()
    low20 = low.rolling(20, min_periods=8).min()
    low60 = low.rolling(60, min_periods=20).min()
    low240 = low.rolling(240, min_periods=80).min().fillna(low60)
    from_high20 = (close / high20.replace(0.0, np.nan) - 1.0) * 100.0
    from_high60 = (close / high60.replace(0.0, np.nan) - 1.0) * 100.0
    from_low20 = (close / low20.replace(0.0, np.nan) - 1.0) * 100.0
    from_low60 = (close / low60.replace(0.0, np.nan) - 1.0) * 100.0
    from_low240 = (close / low240.replace(0.0, np.nan) - 1.0) * 100.0

    vol_base = volume.rolling(30, min_periods=12).median().shift(1).replace(0.0, np.nan)
    vol_ratio = (volume / vol_base).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    vol_recent = vol_ratio.rolling(8, min_periods=2).max().fillna(vol_ratio)
    taker_buy_ratio = (
        taker_buy_quote / quote_volume.replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.50)
    oi_roc20 = oi.pct_change(20).replace([np.inf, -np.inf], np.nan).fillna(0.0) * 100.0

    bar_range = (high - low).replace(0.0, np.nan)
    open_close_max = pd.concat([open_, close], axis=1).max(axis=1)
    upper_wick = (high - open_close_max) / bar_range
    close_pos = (close - low) / bar_range
    weak_bar = (close <= open_) | (ret1 <= max_ret1) | (close_pos <= max_close_pos) | (upper_wick >= min_upper_wick)

    ema8 = close.ewm(span=8, min_periods=5, adjust=False).mean()
    ema21 = close.ewm(span=21, min_periods=10, adjust=False).mean()
    ema_bias = (ema8 / ema21.replace(0.0, np.nan) - 1.0) * 100.0

    extension = pd.concat(
        [
            _clip01((ret15.fillna(0.0) - min_ret15) / 10.0),
            _clip01((ret60.fillna(0.0) - min_ret60) / 18.0),
            _clip01((ret240.fillna(0.0) + 4.0) / 28.0),
            _clip01((from_low240.fillna(0.0) - 2.0) / 24.0),
        ],
        axis=1,
    ).max(axis=1)
    near_high = 1.0 - _clip01((-from_high60.fillna(-999.0)) / max(max_pullback_high60, 0.5))
    downside_room = _clip01((from_low20.fillna(0.0) - min_from_low20) / 12.0)
    broader_room = _clip01((from_low60.fillna(0.0) - min_from_low60) / 18.0)
    stall = (
        0.30 * _clip01((max_ret5 - ret5.fillna(0.0)) / 10.0)
        + 0.25 * _clip01((max_ret1 - ret1.fillna(0.0)) / 5.0)
        + 0.20 * _clip01((max_close_pos - close_pos.fillna(0.5)) / max(max_close_pos, 0.1))
        + 0.15 * _clip01((upper_wick.fillna(0.0) - min_upper_wick) / 0.35)
        + 0.10 * (close <= ema8 * 1.002).astype(float)
    )
    activity = _clip01((vol_recent - min_vol_ratio) / 3.0) * (1.0 - _clip01((vol_ratio - max_vol_ratio) / 8.0))
    sell_pressure = _clip01((max_taker_buy_ratio - taker_buy_ratio) / max(max_taker_buy_ratio - 0.40, 0.08))
    crowding = 0.65 * _clip01(oi_roc20.abs() / 14.0) + 0.35 * _clip01(funding.fillna(0.0).clip(lower=0.0) / 0.25)

    soft_score = (
        0.26 * extension
        + 0.22 * near_high
        + 0.16 * downside_room
        + 0.10 * broader_room
        + 0.14 * stall
        + 0.08 * activity
        + 0.04 * (0.65 * sell_pressure + 0.35 * crowding)
    )
    extreme_score = (
        0.30 * _clip01((ret15.fillna(0.0) - extreme_min_ret15) / 22.0)
        + 0.22 * _clip01((from_low20.fillna(0.0) - extreme_min_from_low20) / 24.0)
        + 0.18 * near_high
        + 0.12 * stall
        + 0.10 * activity
        + 0.08 * (0.55 * sell_pressure + 0.45 * crowding)
    )

    soft_valid = (
        (ret15 >= min_ret15)
        & (ret60 >= min_ret60)
        & (ret5 >= min_ret5)
        & (ret5 <= max_ret5)
        & (ret1 <= max_ret1)
        & (from_high60 >= -max_pullback_high60)
        & (from_low20 >= min_from_low20)
        & (from_low60 >= min_from_low60)
        & (vol_ratio >= min_vol_ratio)
        & (vol_ratio <= max_vol_ratio)
        & (ema_bias >= -1.0)
        & weak_bar
        & (taker_buy_ratio <= max_taker_buy_ratio)
    )
    extreme_valid = (
        (ret15 >= extreme_min_ret15)
        & (ret5 >= min_ret5)
        & (ret5 <= extreme_max_ret5)
        & (from_low20 >= extreme_min_from_low20)
        & (from_high20 >= -7.0)
        & (from_high60 >= -9.0)
        & (vol_ratio >= extreme_min_vol_ratio)
        & (vol_ratio <= max_vol_ratio)
        & (weak_bar | (ret1 <= max_ret1 * 2.0))
        & (taker_buy_ratio <= max(max_taker_buy_ratio, 0.80))
    )

    score = pd.concat(
        [
            soft_score.where(soft_valid, 0.0),
            extreme_score.where(extreme_valid, 0.0),
        ],
        axis=1,
    ).max(axis=1)
    out = -score.clip(lower=0.0, upper=max_output)
    return out.where(out.abs() >= min_output, 0.0).fillna(0.0)


def calculate_last_factor(df: pd.DataFrame) -> float:
    if df is None or len(df) < 90:
        return 0.0
    # The factor uses at most 240 bars.  Slicing keeps live/backtest last-row
    # evaluation fast while preserving exact values needed by rolling windows.
    window = df.tail(260)
    values = calculate_factor(window)
    if values.empty:
        return 0.0
    try:
        return float(values.iloc[-1])
    except Exception:
        return 0.0
