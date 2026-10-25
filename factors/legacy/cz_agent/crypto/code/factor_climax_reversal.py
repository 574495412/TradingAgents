"""
Climax-volume reversal factor for meme coins.

This factor targets the failure mode seen in high-leverage meme trading:
an explosive pump prints an extreme volume bar, then price cannot hold the
local high. It emits a short-side confirmation only after the rejection starts,
so it can be used as a research confirmation for post-climax shorts or as a
guard against late long entries.
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

    min_recent_vol_ratio = _env_float("MEME_CLIMAX_REVERSAL_MIN_RECENT_VOL_RATIO", 80.0)
    min_ret15 = _env_float("MEME_CLIMAX_REVERSAL_MIN_RET15_PCT", 5.0)
    min_ret60 = _env_float("MEME_CLIMAX_REVERSAL_MIN_RET60_PCT", 10.0)
    min_pullback = _env_float("MEME_CLIMAX_REVERSAL_MIN_PULLBACK_FROM_HIGH_PCT", 1.2)
    max_pullback = _env_float("MEME_CLIMAX_REVERSAL_MAX_PULLBACK_FROM_HIGH_PCT", 8.0)
    min_output = _env_float("MEME_CLIMAX_REVERSAL_MIN_OUTPUT", 0.25)
    max_output = _env_float("MEME_CLIMAX_REVERSAL_MAX_OUTPUT", 0.85)

    vol_base = volume.rolling(30, min_periods=12).median().shift(1).replace(0.0, np.nan)
    vol_ratio = (volume / vol_base).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    recent_vol_ratio = vol_ratio.rolling(15, min_periods=3).max()
    recent_high = high.rolling(15, min_periods=3).max()
    pullback = (close / recent_high.replace(0.0, np.nan) - 1.0) * 100.0

    ret1 = close.pct_change(1) * 100.0
    ret3 = close.pct_change(3) * 100.0
    ret15 = close.pct_change(15) * 100.0
    ret60 = close.pct_change(60) * 100.0
    recent_ret15 = ret15.rolling(15, min_periods=3).max()
    recent_ret60 = ret60.rolling(15, min_periods=3).max()

    bar_range = (high - low).replace(0.0, np.nan)
    upper_wick = (high - pd.concat([open_, close], axis=1).max(axis=1)) / bar_range
    close_pos = (close - low) / bar_range
    red_body = close < open_

    ema8 = close.ewm(span=8, min_periods=5, adjust=False).mean()
    ema21 = close.ewm(span=21, min_periods=10, adjust=False).mean()
    oi_roc5 = oi.pct_change(5) * 100.0

    vol_score = _clip01((recent_vol_ratio - min_recent_vol_ratio) / max(min_recent_vol_ratio * 3.0, 1.0))
    extension_score = pd.concat(
        [
            (recent_ret15.fillna(0.0) - min_ret15) / max(min_ret15 * 3.0, 1.0),
            (recent_ret60.fillna(0.0) - min_ret60) / max(min_ret60 * 2.5, 1.0),
        ],
        axis=1,
    ).max(axis=1)
    extension_score = _clip01(extension_score)

    pullback_abs = -pullback.fillna(0.0)
    pullback_score = _clip01((pullback_abs - min_pullback) / max(max_pullback - min_pullback, 1.0))
    wick_score = _clip01((upper_wick.fillna(0.0) - 0.28) / 0.45)
    weak_close_score = _clip01((0.52 - close_pos.fillna(0.5)) / 0.45)
    momentum_fail_score = _clip01(((-ret3.fillna(0.0)) - 0.3) / 3.0)
    oi_press_score = _clip01(oi_roc5.fillna(0.0) / 10.0)
    funding_score = _clip01(funding.fillna(0.0) / 0.20)

    score = (
        0.24 * vol_score
        + 0.20 * extension_score
        + 0.20 * pullback_score
        + 0.12 * wick_score
        + 0.10 * weak_close_score
        + 0.08 * momentum_fail_score
        + 0.04 * oi_press_score
        + 0.02 * funding_score
    )

    valid = (
        (recent_vol_ratio >= min_recent_vol_ratio)
        & ((recent_ret15 >= min_ret15) | (recent_ret60 >= min_ret60))
        & (pullback_abs >= min_pullback)
        & (pullback_abs <= max_pullback)
        & ((ret1 <= 0.0) | (ret3 <= -0.25) | red_body | (close < ema8))
        & (close <= ema21 * 1.04)
    )

    raw = pd.Series(0.0, index=df.index)
    raw = raw.mask(valid, -score.clip(lower=0.0, upper=max_output))
    raw = raw.where(raw.abs() >= min_output, 0.0)
    return raw.fillna(0.0)
