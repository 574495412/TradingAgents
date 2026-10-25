"""Leader-inspired short re-entry factor for meme downtrends.

Recent high-yield meme leaders often shorted after a move had already turned
down, not only after a fresh pump. This factor looks for a controlled bounce
inside a short-term downtrend that starts failing again. It uses only data
available before the signal bar.
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
    if df is None or len(df) < 120:
        return pd.Series(0.0, index=getattr(df, "index", None))

    open_ = _safe_series(df, "open")
    high = _safe_series(df, "high")
    low = _safe_series(df, "low")
    close = _safe_series(df, "close")
    volume = _safe_series(df, "volume", 1.0)
    oi = _safe_series(df, "open_interest")
    funding = _safe_series(df, "funding_rate")

    min_ret60 = _env_float("MEME_LEADER_DOWN_REENTRY_MIN_RET60_PCT", 3.0)
    min_ret240 = _env_float("MEME_LEADER_DOWN_REENTRY_MIN_RET240_PCT", 6.0)
    min_bounce = _env_float("MEME_LEADER_DOWN_REENTRY_MIN_BOUNCE20_PCT", 0.45)
    max_bounce = _env_float("MEME_LEADER_DOWN_REENTRY_MAX_BOUNCE20_PCT", 8.0)
    max_ret5_drop = _env_float("MEME_LEADER_DOWN_REENTRY_MAX_RET5_DROP_PCT", 10.0)
    min_output = _env_float("MEME_LEADER_DOWN_REENTRY_MIN_OUTPUT", 0.20)
    max_output = _env_float("MEME_LEADER_DOWN_REENTRY_MAX_OUTPUT", 0.78)

    ret1 = close.pct_change(1) * 100.0
    ret3 = close.pct_change(3) * 100.0
    ret5 = close.pct_change(5) * 100.0
    ret15 = close.pct_change(15) * 100.0
    ret60 = close.pct_change(60) * 100.0
    ret240 = close.pct_change(240) * 100.0

    low20 = low.rolling(20, min_periods=8).min()
    low60 = low.rolling(60, min_periods=20).min()
    high20 = high.rolling(20, min_periods=8).max()
    high60 = high.rolling(60, min_periods=20).max()
    bounce20 = (close / low20.replace(0.0, np.nan) - 1.0) * 100.0
    bounce60 = (close / low60.replace(0.0, np.nan) - 1.0) * 100.0
    from_high20 = (close / high20.replace(0.0, np.nan) - 1.0) * 100.0
    from_high60 = (close / high60.replace(0.0, np.nan) - 1.0) * 100.0

    ema8 = close.ewm(span=8, min_periods=5, adjust=False).mean()
    ema21 = close.ewm(span=21, min_periods=10, adjust=False).mean()
    ema55 = close.ewm(span=55, min_periods=20, adjust=False).mean()

    vol_base = volume.rolling(30, min_periods=12).median().shift(1).replace(0.0, np.nan)
    vol_ratio = (volume / vol_base).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    bar_range = (high - low).replace(0.0, np.nan)
    upper_wick = (high - pd.concat([open_, close], axis=1).max(axis=1)) / bar_range
    close_pos = (close - low) / bar_range
    oi_roc5 = oi.pct_change(5).replace([np.inf, -np.inf], np.nan).fillna(0.0) * 100.0

    trend_score = pd.concat(
        [
            ((-ret60.fillna(0.0)) - min_ret60) / max(min_ret60 * 3.0, 1.0),
            ((-ret240.fillna(0.0)) - min_ret240) / max(min_ret240 * 2.2, 1.0),
            ((-ret15.fillna(0.0)) - 1.0) / 12.0,
            (-from_high60.fillna(0.0) - 4.0) / 24.0,
        ],
        axis=1,
    ).max(axis=1)
    bounce_score = _clip01((bounce20.fillna(0.0) - min_bounce) / max(max_bounce - min_bounce, 1.0))
    bounce_score *= 1.0 - _clip01((bounce20.fillna(0.0) - max_bounce) / max(max_bounce, 1.0))
    fail_score = (
        0.36 * _clip01((-ret3.fillna(0.0) + 0.15) / 3.5)
        + 0.26 * (close < ema8).astype(float)
        + 0.20 * (ema8 <= ema21 * 1.002).astype(float)
        + 0.10 * _clip01((upper_wick.fillna(0.0) - 0.16) / 0.40)
        + 0.08 * _clip01((0.56 - close_pos.fillna(0.5)) / 0.50)
    )
    volume_score = _clip01((vol_ratio - 0.9) / 4.0) * (1.0 - _clip01((vol_ratio - 16.0) / 30.0))
    oi_score = _clip01(oi_roc5 / 10.0)
    funding_score = _clip01(funding.fillna(0.0) / 0.20)

    score = (
        0.28 * _clip01(trend_score)
        + 0.22 * bounce_score
        + 0.24 * fail_score
        + 0.10 * (close <= ema21 * 1.006).astype(float)
        + 0.08 * volume_score
        + 0.05 * oi_score
        + 0.03 * funding_score
    )

    valid = (
        ((ret60 <= -min_ret60) | (ret15 <= -3.0) | (from_high60 <= -8.0))
        & (bounce20 >= min_bounce)
        & (bounce20 <= max_bounce)
        & (bounce60 <= max(max_bounce * 1.8, max_bounce + 2.0))
        & (from_high20 <= -0.4)
        & ((ret1 <= 0.25) | (ret3 <= 0.0) | (upper_wick >= 0.18) | (close < ema8))
        & (close <= ema55 * 1.02)
        & (ret5 >= -abs(max_ret5_drop))
    )

    out = pd.Series(0.0, index=df.index)
    out = out.mask(valid, -score.clip(lower=0.0, upper=max_output))
    out = out.where(out.abs() >= min_output, 0.0)
    return out.fillna(0.0)
