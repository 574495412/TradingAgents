"""Leader-labeled crash-rebound long factor.

High-yield meme lead-trader samples show that the profitable long side is often
not breakout chasing.  The stronger cluster is a very short hold after a fast
flush: price is far below the local high, near a fresh low, but the current bar
starts showing absorption through a lower wick or a controlled close.

Only as-of OHLCV/OI/funding data is used at runtime.
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
    funding = _safe_series(df, "funding_rate")

    min_output = _env_float("MEME_LEADER_CRASH_REBOUND_MIN_OUTPUT", 0.30)
    max_output = _env_float("MEME_LEADER_CRASH_REBOUND_MAX_OUTPUT", 0.88)
    min_abs_ret3 = _env_float("MEME_LEADER_CRASH_REBOUND_MIN_ABS_RET3_PCT", 2.4)
    min_abs_ret5 = _env_float("MEME_LEADER_CRASH_REBOUND_MIN_ABS_RET5_PCT", 3.2)
    min_abs_ret15 = _env_float("MEME_LEADER_CRASH_REBOUND_MIN_ABS_RET15_PCT", 4.8)
    min_drop_high20 = _env_float("MEME_LEADER_CRASH_REBOUND_MIN_DROP_HIGH20_PCT", 5.0)
    max_bounce_low20 = _env_float("MEME_LEADER_CRASH_REBOUND_MAX_BOUNCE20_PCT", 9.0)
    min_lower_wick = _env_float("MEME_LEADER_CRASH_REBOUND_MIN_LOWER_WICK", 0.18)
    min_close_pos = _env_float("MEME_LEADER_CRASH_REBOUND_MIN_CLOSE_POS", 0.22)
    max_close_pos = _env_float("MEME_LEADER_CRASH_REBOUND_MAX_CLOSE_POS", 0.92)
    min_vol_ratio = _env_float("MEME_LEADER_CRASH_REBOUND_MIN_VOL_RATIO", 1.2)
    max_vol_ratio = _env_float("MEME_LEADER_CRASH_REBOUND_MAX_VOL_RATIO", 28.0)
    max_ret1 = _env_float("MEME_LEADER_CRASH_REBOUND_MAX_RET1_PCT", 1.2)
    min_ret1 = _env_float("MEME_LEADER_CRASH_REBOUND_MIN_RET1_PCT", -14.0)
    max_bar_range = _env_float("MEME_LEADER_CRASH_REBOUND_MAX_BAR_RANGE_PCT", 18.0)

    ret1 = close.pct_change(1) * 100.0
    ret3 = close.pct_change(3) * 100.0
    ret5 = close.pct_change(5) * 100.0
    ret15 = close.pct_change(15) * 100.0
    ret60 = close.pct_change(60) * 100.0

    high20 = high.rolling(20, min_periods=8).max()
    low20 = low.rolling(20, min_periods=8).min()
    high60 = high.rolling(60, min_periods=20).max()
    from_high20 = (close / high20.replace(0.0, np.nan) - 1.0) * 100.0
    from_high60 = (close / high60.replace(0.0, np.nan) - 1.0) * 100.0
    from_low20 = (close / low20.replace(0.0, np.nan) - 1.0) * 100.0

    vol_base = volume.rolling(30, min_periods=12).median().shift(1).replace(0.0, np.nan)
    vol_ratio = (volume / vol_base).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    vol_recent = vol_ratio.rolling(5, min_periods=2).max().fillna(vol_ratio)

    bar_range = (high - low).replace(0.0, np.nan)
    lower_wick = (pd.concat([open_, close], axis=1).min(axis=1) - low) / bar_range
    close_pos = (close - low) / bar_range
    bar_range_pct = (high / low.replace(0.0, np.nan) - 1.0) * 100.0

    oi_roc5 = oi.pct_change(5).replace([np.inf, -np.inf], np.nan).fillna(0.0) * 100.0
    funding_neg = (-funding.fillna(0.0)).clip(lower=0.0)

    flush_score = pd.concat(
        [
            ((-ret3.fillna(0.0)) - min_abs_ret3) / max(min_abs_ret3 * 4.0, 1.0),
            ((-ret5.fillna(0.0)) - min_abs_ret5) / max(min_abs_ret5 * 4.0, 1.0),
            ((-ret15.fillna(0.0)) - min_abs_ret15) / max(min_abs_ret15 * 4.0, 1.0),
        ],
        axis=1,
    ).max(axis=1)
    drop_score = _clip01((-from_high20.fillna(0.0) - min_drop_high20) / 20.0)
    high60_drop_score = _clip01((-from_high60.fillna(0.0) - min_drop_high20) / 35.0)
    near_low_score = 1.0 - _clip01(from_low20.fillna(999.0) / max(max_bounce_low20, 1.0))
    absorption = (
        0.42 * _clip01((lower_wick.fillna(0.0) - min_lower_wick) / 0.55)
        + 0.28 * _clip01((close_pos.fillna(0.5) - min_close_pos) / 0.55)
        + 0.18 * (ret1.fillna(0.0) >= -4.0).astype(float)
        + 0.12 * (close >= open_).astype(float)
    )
    fuel = (
        0.62 * _clip01((vol_recent - min_vol_ratio) / 8.0)
        + 0.22 * _clip01(oi_roc5.abs() / 12.0)
        + 0.16 * _clip01(funding_neg / 0.20)
    )

    score = (
        0.30 * _clip01(flush_score)
        + 0.20 * drop_score
        + 0.12 * high60_drop_score
        + 0.16 * near_low_score
        + 0.14 * absorption
        + 0.08 * fuel
    )
    valid = (
        (vol_ratio >= min_vol_ratio)
        & (vol_ratio <= max_vol_ratio)
        & ((ret3 <= -min_abs_ret3) | (ret5 <= -min_abs_ret5) | (ret15 <= -min_abs_ret15))
        & (from_high20 <= -abs(min_drop_high20))
        & (from_low20 <= max_bounce_low20)
        & (lower_wick >= min_lower_wick)
        & (close_pos >= min_close_pos)
        & (close_pos <= max_close_pos)
        & (ret1 <= max_ret1)
        & (ret1 >= min_ret1)
        & (bar_range_pct <= max_bar_range)
    )

    out = score.clip(lower=0.0, upper=max_output).where(valid, 0.0)
    return out.where(out.abs() >= min_output, 0.0).fillna(0.0)
