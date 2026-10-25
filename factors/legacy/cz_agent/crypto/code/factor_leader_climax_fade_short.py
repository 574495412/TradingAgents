"""Leader-inspired climax fade short factor.

This research factor models a common structure in recent high-yield meme
leader trades: after a fast pump or a high-volume blow-off bar, the leader
shorts the first weak close instead of following momentum.  It uses only
as-of OHLCV/OI/funding values available at the signal bar.
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

    min_output = _env_float("MEME_LEADER_CLIMAX_FADE_MIN_OUTPUT", 0.30)
    max_output = _env_float("MEME_LEADER_CLIMAX_FADE_MAX_OUTPUT", 0.92)
    min_vol_ratio = _env_float("MEME_LEADER_CLIMAX_FADE_MIN_VOL_RATIO", 3.0)
    min_recent_vol_ratio = _env_float("MEME_LEADER_CLIMAX_FADE_MIN_RECENT_VOL_RATIO", 3.0)
    max_current_vol_ratio = _env_float("MEME_LEADER_CLIMAX_FADE_MAX_CURRENT_VOL_RATIO", 80.0)
    min_ret15 = _env_float("MEME_LEADER_CLIMAX_FADE_MIN_RET15_PCT", 4.0)
    min_ret60 = _env_float("MEME_LEADER_CLIMAX_FADE_MIN_RET60_PCT", 1.0)
    max_pullback_high20 = _env_float("MEME_LEADER_CLIMAX_FADE_MAX_PULLBACK_HIGH20_PCT", 12.0)
    min_bounce_low20 = _env_float("MEME_LEADER_CLIMAX_FADE_MIN_BOUNCE20_PCT", 4.0)
    max_bar_range = _env_float("MEME_LEADER_CLIMAX_FADE_MAX_BAR_RANGE_PCT", 26.0)
    min_bar_range = _env_float("MEME_LEADER_CLIMAX_FADE_MIN_BAR_RANGE_PCT", 0.4)
    max_close_pos = _env_float("MEME_LEADER_CLIMAX_FADE_MAX_CLOSE_POS", 0.68)
    min_upper_wick = _env_float("MEME_LEADER_CLIMAX_FADE_MIN_UPPER_WICK", 0.12)
    pullback_max_ret1 = _env_float("MEME_LEADER_CLIMAX_FADE_PULLBACK_MAX_RET1_PCT", 0.25)

    ret1 = close.pct_change(1) * 100.0
    ret2 = close.pct_change(2) * 100.0
    ret3 = close.pct_change(3) * 100.0
    ret5 = close.pct_change(5) * 100.0
    ret15 = close.pct_change(15) * 100.0
    ret60 = close.pct_change(60) * 100.0

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
    close_pos = (close - low) / bar_range
    bar_range_pct = (high / low.replace(0.0, np.nan) - 1.0) * 100.0

    weak_close = (
        0.45 * _clip01((max_close_pos - close_pos.fillna(0.5)) / max(max_close_pos, 0.1))
        + 0.35 * _clip01((upper_wick.fillna(0.0) - min_upper_wick) / 0.55)
        + 0.20 * (ret1.fillna(0.0) <= pullback_max_ret1).astype(float)
    )
    extension = pd.concat(
        [
            _clip01((ret1.fillna(0.0) - 2.0) / 8.0),
            _clip01((ret3.fillna(0.0) - 3.0) / 10.0),
            _clip01((ret5.fillna(0.0) - 3.5) / 12.0),
            _clip01((ret15.fillna(0.0) - min_ret15) / 18.0),
        ],
        axis=1,
    ).max(axis=1)
    near_high = 1.0 - _clip01((-from_high20.fillna(-999.0)) / max(max_pullback_high20, 0.5))
    bounce = _clip01((from_low20.fillna(0.0) - min_bounce_low20) / 18.0)
    volume_fuel = _clip01((vol_recent - min_recent_vol_ratio) / 12.0)
    crowding = 0.55 * _clip01(oi_roc5 / 14.0) + 0.45 * _clip01(funding.fillna(0.0).clip(lower=0.0) / 0.25)

    blowoff_score = (
        0.28 * extension
        + 0.20 * weak_close
        + 0.18 * volume_fuel
        + 0.16 * near_high
        + 0.12 * bounce
        + 0.06 * crowding
    )
    pullback_score = (
        0.25 * _clip01((ret15.fillna(0.0) - min_ret15) / 18.0)
        + 0.23 * weak_close
        + 0.18 * near_high
        + 0.16 * bounce
        + 0.12 * volume_fuel
        + 0.06 * crowding
    )

    blowoff_valid = (
        (vol_ratio >= min_vol_ratio)
        & (vol_ratio <= max_current_vol_ratio)
        & (ret15 >= min_ret15)
        & (ret60 >= min_ret60)
        & (
            (ret1 >= 4.0)
            | (ret3 >= 5.0)
            | ((ret5 >= 4.0) & (upper_wick.fillna(0.0) >= min_upper_wick))
        )
        & (from_high20 >= -abs(max_pullback_high20))
        & (from_low20 >= min_bounce_low20)
        & (bar_range_pct >= min_bar_range)
        & (bar_range_pct <= max_bar_range)
        & (close_pos.fillna(0.5) <= max_close_pos)
        & (upper_wick.fillna(0.0) >= min_upper_wick)
    )
    pullback_valid = (
        (vol_recent >= min_recent_vol_ratio)
        & (ret15 >= min_ret15)
        & (ret60 >= min_ret60)
        & (ret1 <= pullback_max_ret1)
        & (ret2 <= 1.8)
        & (from_high20 >= -3.5)
        & (from_low20 >= min_bounce_low20)
        & (bar_range_pct >= min_bar_range)
        & (bar_range_pct <= 8.0)
        & (close_pos.fillna(0.5) <= 0.58)
    )

    score = pd.concat(
        [
            blowoff_score.where(blowoff_valid, 0.0),
            pullback_score.where(pullback_valid, 0.0),
        ],
        axis=1,
    ).max(axis=1)
    out = -score.clip(lower=0.0, upper=max_output)
    return out.where(out.abs() >= min_output, 0.0).fillna(0.0)
