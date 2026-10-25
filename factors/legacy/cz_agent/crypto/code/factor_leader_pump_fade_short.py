"""Leader-inspired early pump-fade short factor for meme coins.

High-yield meme copy traders in the latest samples often short before a full
downtrend is confirmed: price has extended quickly, then leaves the local high
by a small amount while volume is still active. This research factor captures
that early pump-fade structure without using leader trades at runtime.
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

    min_ret60 = _env_float("MEME_LEADER_PUMP_FADE_MIN_RET60_PCT", 4.0)
    min_ret240 = _env_float("MEME_LEADER_PUMP_FADE_MIN_RET240_PCT", 8.0)
    min_pullback = _env_float("MEME_LEADER_PUMP_FADE_MIN_PULLBACK_PCT", 0.45)
    max_pullback = _env_float("MEME_LEADER_PUMP_FADE_MAX_PULLBACK_PCT", 5.0)
    max_ret5 = _env_float("MEME_LEADER_PUMP_FADE_MAX_RET5_PCT", 18.0)
    min_output = _env_float("MEME_LEADER_PUMP_FADE_MIN_OUTPUT", 0.22)
    max_output = _env_float("MEME_LEADER_PUMP_FADE_MAX_OUTPUT", 0.78)

    ret1 = close.pct_change(1) * 100.0
    ret5 = close.pct_change(5) * 100.0
    ret15 = close.pct_change(15) * 100.0
    ret60 = close.pct_change(60) * 100.0
    ret240 = close.pct_change(240) * 100.0

    high20 = high.rolling(20, min_periods=8).max()
    high60 = high.rolling(60, min_periods=20).max()
    low20 = low.rolling(20, min_periods=8).min()
    pull20 = (close / high20.replace(0.0, np.nan) - 1.0) * 100.0
    pull60 = (close / high60.replace(0.0, np.nan) - 1.0) * 100.0
    pullback = pd.concat([-pull20, -pull60 * 0.75], axis=1).max(axis=1)
    bounce20 = (close / low20.replace(0.0, np.nan) - 1.0) * 100.0

    vol_base = volume.rolling(30, min_periods=12).median().shift(1).replace(0.0, np.nan)
    vol_ratio = (volume / vol_base).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    vol_recent_max = vol_ratio.rolling(10, min_periods=3).max()
    vol_fade = volume / volume.rolling(8, min_periods=4).max().replace(0.0, np.nan)

    bar_range = (high - low).replace(0.0, np.nan)
    upper_wick = (high - pd.concat([open_, close], axis=1).max(axis=1)) / bar_range
    close_pos = (close - low) / bar_range
    weak_body = (close <= open_) | (ret1 <= 0.2)

    ema8 = close.ewm(span=8, min_periods=5, adjust=False).mean()
    ema21 = close.ewm(span=21, min_periods=10, adjust=False).mean()
    oi_roc5 = oi.pct_change(5) * 100.0

    extension_score = pd.concat(
        [
            (ret60.fillna(0.0) - min_ret60) / max(min_ret60 * 3.0, 1.0),
            (ret240.fillna(0.0) - min_ret240) / max(min_ret240 * 2.0, 1.0),
            (ret15.fillna(0.0) - 1.0) / 18.0,
        ],
        axis=1,
    ).max(axis=1)
    pullback_score = _clip01((pullback.fillna(0.0) - min_pullback) / max(max_pullback - min_pullback, 1.0))
    early_fail_score = _clip01((pullback.fillna(0.0) - 0.25) / 2.5)
    wick_score = _clip01((upper_wick.fillna(0.0) - 0.18) / 0.45)
    weak_close_score = _clip01((0.58 - close_pos.fillna(0.5)) / 0.55)
    vol_score = _clip01((vol_recent_max.fillna(1.0) - 1.2) / 5.0) * 0.7 + _clip01((1.0 - vol_fade.fillna(1.0)) / 0.65) * 0.3
    oi_score = _clip01(oi_roc5.fillna(0.0) / 8.0)
    funding_score = _clip01(funding.fillna(0.0) / 0.20)

    score = (
        0.25 * _clip01(extension_score)
        + 0.22 * pullback_score
        + 0.16 * early_fail_score
        + 0.12 * wick_score
        + 0.10 * weak_close_score
        + 0.08 * vol_score
        + 0.04 * oi_score
        + 0.03 * funding_score
    )

    valid = (
        ((ret60 >= min_ret60) | (ret15 >= 6.0))
        & (pullback >= min_pullback)
        & (pullback <= max_pullback)
        & (ret5 <= max_ret5)
        & (bounce20 >= 1.0)
        & ((upper_wick >= 0.18) | weak_body | (close < ema8 * 1.002))
        & (close <= ema21 * 1.12)
    )

    out = pd.Series(0.0, index=df.index)
    out = out.mask(valid, -score.clip(lower=0.0, upper=max_output))
    out = out.where(out.abs() >= min_output, 0.0)
    return out.fillna(0.0)
