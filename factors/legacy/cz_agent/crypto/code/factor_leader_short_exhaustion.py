"""Leader-inspired post-pump short exhaustion factor for meme coins.

External high-yield meme leaders in the latest samples are strongly skewed to
shorting long-tail coins after an impulse has started to fail. This research
factor captures that structure without using leader trades as labels at
runtime: large recent extension, local-high rejection, and fading short-term
momentum.
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


def _clip01(series: pd.Series) -> pd.Series:
    return series.clip(lower=0.0, upper=1.0)


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

    min_ret60 = _env_float("MEME_LEADER_SHORT_MIN_RET60_PCT", 8.0)
    min_ret240 = _env_float("MEME_LEADER_SHORT_MIN_RET240_PCT", 18.0)
    min_pullback = _env_float("MEME_LEADER_SHORT_MIN_PULLBACK_PCT", 0.6)
    max_pullback = _env_float("MEME_LEADER_SHORT_MAX_PULLBACK_PCT", 7.0)
    min_output = _env_float("MEME_LEADER_SHORT_MIN_OUTPUT", 0.24)
    max_output = _env_float("MEME_LEADER_SHORT_MAX_OUTPUT", 0.82)

    ret3 = close.pct_change(3) * 100.0
    ret5 = close.pct_change(5) * 100.0
    ret15 = close.pct_change(15) * 100.0
    ret60 = close.pct_change(60) * 100.0
    ret240 = close.pct_change(240) * 100.0

    high20 = high.rolling(20, min_periods=8).max()
    high60 = high.rolling(60, min_periods=20).max()
    pullback20 = (high20 / close.replace(0.0, np.nan) - 1.0) * 100.0
    pullback60 = (high60 / close.replace(0.0, np.nan) - 1.0) * 100.0
    pullback = pd.concat([pullback20, pullback60 * 0.7], axis=1).max(axis=1)

    ema8 = close.ewm(span=8, min_periods=5, adjust=False).mean()
    ema21 = close.ewm(span=21, min_periods=10, adjust=False).mean()
    vol_base = volume.rolling(30, min_periods=12).median().shift(1).replace(0.0, np.nan)
    vol_ratio = (volume / vol_base).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    vol_fade = volume / volume.rolling(8, min_periods=4).max().replace(0.0, np.nan)

    bar_range = (high - low).replace(0.0, np.nan)
    upper_wick = (high - pd.concat([open_, close], axis=1).max(axis=1)) / bar_range
    close_pos = (close - low) / bar_range
    oi_roc5 = oi.pct_change(5) * 100.0

    extension_score = pd.concat(
        [
            (ret60.fillna(0.0) - min_ret60) / max(min_ret60 * 2.5, 1.0),
            (ret240.fillna(0.0) - min_ret240) / max(min_ret240 * 1.8, 1.0),
            (ret15.fillna(0.0) - 4.0) / 16.0,
        ],
        axis=1,
    ).max(axis=1)
    extension_score = _clip01(extension_score)
    pullback_score = _clip01((pullback.fillna(0.0) - min_pullback) / max(max_pullback - min_pullback, 1.0))
    fail_score = _clip01(((-ret3.fillna(0.0)) + 0.2) / 3.0)
    ema_fail_score = ((close < ema8) | ((ema8 < ema21) & (ret5 < 0))).astype(float)
    wick_score = _clip01((upper_wick.fillna(0.0) - 0.22) / 0.5)
    weak_close_score = _clip01((0.55 - close_pos.fillna(0.5)) / 0.5)
    vol_score = _clip01((vol_ratio - 1.0) / 3.5) * 0.6 + _clip01((1.0 - vol_fade.fillna(1.0)) / 0.65) * 0.4
    oi_score = _clip01(oi_roc5.fillna(0.0) / 8.0)
    funding_score = _clip01(funding.fillna(0.0) / 0.18)

    score = (
        0.23 * extension_score
        + 0.20 * pullback_score
        + 0.16 * fail_score
        + 0.12 * ema_fail_score
        + 0.10 * wick_score
        + 0.08 * weak_close_score
        + 0.06 * vol_score
        + 0.03 * oi_score
        + 0.02 * funding_score
    )

    valid = (
        ((ret60 >= min_ret60) | (ret240 >= min_ret240))
        & (pullback >= min_pullback)
        & (pullback <= max_pullback)
        & ((ret3 <= 0.0) | (close < ema8) | (upper_wick >= 0.25))
        & (close <= ema21 * 1.08)
    )

    out = pd.Series(0.0, index=df.index)
    out = out.mask(valid, -score.clip(lower=0.0, upper=max_output))
    out = out.where(out.abs() >= min_output, 0.0)
    return out.fillna(0.0)
