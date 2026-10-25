"""Leader-learned crash-reclaim long factor.

High-yield lead-trader long samples are often not fresh breakouts.  A separate
cluster buys after a deep 60m flush has already reclaimed from the local low,
with current buy acceptance and volume returning.  This factor targets that
"crash then reclaim" structure without using live orders or future data.
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
    quote_volume = _safe_series(df, "quote_volume")
    taker_buy_quote = _safe_series(df, "taker_buy_quote")
    oi = _safe_series(df, "open_interest")
    funding = _safe_series(df, "funding_rate")

    min_output = _env_float("MEME_LEADER_CRASH_RECLAIM_LONG_MIN_OUTPUT", 0.28)
    max_output = _env_float("MEME_LEADER_CRASH_RECLAIM_LONG_MAX_OUTPUT", 0.86)
    max_ret60 = _env_float("MEME_LEADER_CRASH_RECLAIM_LONG_MAX_RET60_PCT", -7.5)
    max_from_high60 = _env_float("MEME_LEADER_CRASH_RECLAIM_LONG_MAX_FROM_HIGH60_PCT", -10.0)
    min_from_low20 = _env_float("MEME_LEADER_CRASH_RECLAIM_LONG_MIN_FROM_LOW20_PCT", 3.8)
    max_from_low20 = _env_float("MEME_LEADER_CRASH_RECLAIM_LONG_MAX_FROM_LOW20_PCT", 16.0)
    min_from_low60 = _env_float("MEME_LEADER_CRASH_RECLAIM_LONG_MIN_FROM_LOW60_PCT", 5.0)
    max_from_low60 = _env_float("MEME_LEADER_CRASH_RECLAIM_LONG_MAX_FROM_LOW60_PCT", 24.0)
    min_ret5 = _env_float("MEME_LEADER_CRASH_RECLAIM_LONG_MIN_RET5_PCT", -2.5)
    max_ret5 = _env_float("MEME_LEADER_CRASH_RECLAIM_LONG_MAX_RET5_PCT", 5.8)
    min_ret1 = _env_float("MEME_LEADER_CRASH_RECLAIM_LONG_MIN_RET1_PCT", -1.6)
    max_ret1 = _env_float("MEME_LEADER_CRASH_RECLAIM_LONG_MAX_RET1_PCT", 3.2)
    min_vol_ratio = _env_float("MEME_LEADER_CRASH_RECLAIM_LONG_MIN_VOL_RATIO", 0.85)
    max_vol_ratio = _env_float("MEME_LEADER_CRASH_RECLAIM_LONG_MAX_VOL_RATIO", 18.0)
    min_taker_buy_ratio = _env_float("MEME_LEADER_CRASH_RECLAIM_LONG_MIN_TAKER_BUY_RATIO", 0.50)
    min_close_pos = _env_float("MEME_LEADER_CRASH_RECLAIM_LONG_MIN_CLOSE_POS", 0.48)
    max_upper_wick = _env_float("MEME_LEADER_CRASH_RECLAIM_LONG_MAX_UPPER_WICK", 0.50)
    max_bar_range = _env_float("MEME_LEADER_CRASH_RECLAIM_LONG_MAX_BAR_RANGE_PCT", 18.0)

    ret1 = close.pct_change(1) * 100.0
    ret3 = close.pct_change(3) * 100.0
    ret5 = close.pct_change(5) * 100.0
    ret15 = close.pct_change(15) * 100.0
    ret60 = close.pct_change(60) * 100.0
    ret240 = close.pct_change(240) * 100.0

    high20 = high.rolling(20, min_periods=8).max()
    low20 = low.rolling(20, min_periods=8).min()
    high60 = high.rolling(60, min_periods=20).max()
    low60 = low.rolling(60, min_periods=20).min()
    high240 = high.rolling(240, min_periods=80).max()
    low240 = low.rolling(240, min_periods=80).min()
    from_high20 = (close / high20.replace(0.0, np.nan) - 1.0) * 100.0
    from_high60 = (close / high60.replace(0.0, np.nan) - 1.0) * 100.0
    from_high240 = (close / high240.replace(0.0, np.nan) - 1.0) * 100.0
    from_low20 = (close / low20.replace(0.0, np.nan) - 1.0) * 100.0
    from_low60 = (close / low60.replace(0.0, np.nan) - 1.0) * 100.0
    from_low240 = (close / low240.replace(0.0, np.nan) - 1.0) * 100.0

    vol_base = volume.rolling(30, min_periods=12).median().shift(1).replace(0.0, np.nan)
    vol_ratio = (volume / vol_base).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    taker_buy_ratio = (
        taker_buy_quote / quote_volume.replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.52)
    oi_roc20 = oi.pct_change(20).replace([np.inf, -np.inf], np.nan).fillna(0.0) * 100.0

    bar_range = (high - low).replace(0.0, np.nan)
    close_pos = (close - low) / bar_range
    upper_wick = (high - pd.concat([open_, close], axis=1).max(axis=1)) / bar_range
    bar_range_pct = (high / low.replace(0.0, np.nan) - 1.0) * 100.0

    crash_depth = _clip01((max_ret60 - ret60.fillna(0.0)) / 18.0)
    high_flush = _clip01((max_from_high60 - from_high60.fillna(0.0)) / 26.0)
    reclaim20 = _clip01((from_low20.fillna(0.0) - min_from_low20) / max(max_from_low20 - min_from_low20, 1.0))
    reclaim60 = _clip01((from_low60.fillna(0.0) - min_from_low60) / max(max_from_low60 - min_from_low60, 1.0))
    not_late = (
        1.0 - _clip01((from_low20.fillna(999.0) - max_from_low20) / 16.0)
    ) * (1.0 - _clip01((ret5.fillna(0.0) - max_ret5) / 8.0))
    acceptance = (
        0.40 * _clip01((close_pos.fillna(0.5) - min_close_pos) / 0.42)
        + 0.25 * _clip01((taker_buy_ratio - min_taker_buy_ratio) / 0.22)
        + 0.20 * (upper_wick.fillna(0.0) <= max_upper_wick).astype(float)
        + 0.15 * _clip01((ret3.fillna(0.0) + 1.0) / 5.0)
    )
    volume_quality = _clip01((vol_ratio - min_vol_ratio) / 4.0) * (
        1.0 - _clip01((vol_ratio - max_vol_ratio) / max(max_vol_ratio, 1.0))
    )
    fuel = (
        0.42 * volume_quality
        + 0.18 * _clip01(oi_roc20.abs() / 16.0)
        + 0.14 * _clip01((-funding.fillna(0.0)) / 0.20)
        + 0.14 * _clip01((from_low240.fillna(0.0) - 2.0) / 18.0)
        + 0.12 * _clip01((-from_high240.fillna(0.0) - 4.0) / 28.0)
    )

    score = (
        0.24 * crash_depth
        + 0.18 * high_flush
        + 0.16 * reclaim20
        + 0.12 * reclaim60
        + 0.14 * not_late
        + 0.10 * acceptance
        + 0.06 * fuel
    )

    valid = (
        (ret60 <= max_ret60)
        & (from_high60 <= max_from_high60)
        & (from_low20 >= min_from_low20)
        & (from_low20 <= max_from_low20)
        & (from_low60 >= min_from_low60)
        & (from_low60 <= max_from_low60)
        & (ret5 >= min_ret5)
        & (ret5 <= max_ret5)
        & (ret1 >= min_ret1)
        & (ret1 <= max_ret1)
        & (ret15 >= -18.0)
        & (ret240 >= -30.0)
        & (from_high20 <= 1.0)
        & (vol_ratio >= min_vol_ratio)
        & (vol_ratio <= max_vol_ratio)
        & (taker_buy_ratio >= min_taker_buy_ratio)
        & (close_pos.fillna(0.5) >= min_close_pos)
        & (upper_wick.fillna(0.0) <= max_upper_wick)
        & (bar_range_pct <= max_bar_range)
    )

    out = score.clip(lower=0.0, upper=max_output).where(valid, 0.0)
    return out.where(out.abs() >= min_output, 0.0).fillna(0.0)
