"""Breakout acceptance factor for meme continuation candidates.

Scores whether a breakout is being accepted by the market instead of fading
back into the prior range.  The factor is intentionally simple and lag-safe:
it only uses current and historical bars, and promotion must go through IC
audit plus portfolio backtests.
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


def _pct(cur: float, prev: float) -> float:
    if prev <= 0:
        return 0.0
    return (cur - prev) / prev * 100.0


def calculate_factor(df: pd.DataFrame) -> pd.Series:
    n = len(df)
    out = pd.Series(0.0, index=df.index)
    if n < 80:
        return out

    close = df["close"].astype(float).values
    open_ = df["open"].astype(float).values if "open" in df else np.roll(close, 1)
    open_[0] = close[0]
    high = df["high"].astype(float).values if "high" in df else close
    low = df["low"].astype(float).values if "low" in df else close
    volume = df["volume"].astype(float).values if "volume" in df else np.ones(n)

    close_s = pd.Series(close)
    vol_med30 = pd.Series(volume).rolling(30, min_periods=15).median().fillna(0.0).values
    hh20 = pd.Series(high).rolling(20, min_periods=10).max().fillna(0.0).values
    ll20 = pd.Series(low).rolling(20, min_periods=10).min().fillna(0.0).values
    ema8 = close_s.ewm(span=8, min_periods=4).mean().values
    ema21 = close_s.ewm(span=21, min_periods=10).mean().values
    min_output = _env_float("MEME_BREAKOUT_ACCEPTANCE_MIN_OUTPUT", 0.24)
    max_pullback_long = _env_float("MEME_BREAKOUT_ACCEPTANCE_MAX_PULLBACK_LONG_PCT", 1.6)
    max_bounce_short = _env_float("MEME_BREAKOUT_ACCEPTANCE_MAX_BOUNCE_SHORT_PCT", 1.6)

    raw = np.zeros(n)
    for i in range(60, n):
        ret3 = _pct(close[i], close[i - 3])
        ret10 = _pct(close[i], close[i - 10])
        ret30 = _pct(close[i], close[i - 30])
        vol_ratio = volume[i] / vol_med30[i] if vol_med30[i] > 0 else 1.0
        range20 = max(1e-12, hh20[i] - ll20[i])
        close_pos = (close[i] - ll20[i]) / range20
        pullback_from_high = (close[i] / max(hh20[i], 1e-12) - 1.0) * 100.0
        bounce_from_low = (close[i] / max(ll20[i], 1e-12) - 1.0) * 100.0
        bar_range = max(1e-12, high[i] - low[i])
        close_in_bar = (close[i] - low[i]) / bar_range
        upper_wick = (high[i] - max(open_[i], close[i])) / bar_range
        lower_wick = (min(open_[i], close[i]) - low[i]) / bar_range
        ema_bias = _pct(ema8[i], ema21[i]) if ema21[i] > 0 else 0.0

        long_score = 0.0
        if ret10 > 2.5 and ret30 > 5.0:
            long_score += min(0.24, ret30 / 45.0 * 0.24)
        if close_pos > 0.78 and close_in_bar > 0.62:
            long_score += min(0.22, (close_pos - 0.60) * 0.42)
        if vol_ratio > 1.8:
            long_score += min(0.22, (vol_ratio - 1.0) / 6.0 * 0.22)
        if ret3 > 0.2 and ema_bias > 0:
            long_score += min(0.18, (ret3 + ema_bias) / 8.0 * 0.18)
        if pullback_from_high < -abs(max_pullback_long) or upper_wick > 0.42:
            long_score *= 0.35

        short_score = 0.0
        if ret10 < -2.5 and ret30 < -5.0:
            short_score += min(0.24, abs(ret30) / 45.0 * 0.24)
        if close_pos < 0.22 and close_in_bar < 0.38:
            short_score += min(0.22, (0.40 - close_pos) * 0.42)
        if vol_ratio > 1.8:
            short_score += min(0.22, (vol_ratio - 1.0) / 6.0 * 0.22)
        if ret3 < -0.2 and ema_bias < 0:
            short_score += min(0.18, (abs(ret3) + abs(ema_bias)) / 8.0 * 0.18)
        if bounce_from_low > abs(max_bounce_short) or lower_wick > 0.42:
            short_score *= 0.35

        if long_score >= min_output and long_score >= short_score:
            raw[i] = min(1.0, long_score * 1.7)
        elif short_score >= min_output:
            raw[i] = -min(1.0, short_score * 1.7)

    smooth = pd.Series(raw).ewm(span=3, min_periods=2).mean().values
    for i in range(60, n):
        value = float(np.clip(smooth[i], -1.0, 1.0))
        if abs(value) >= min_output:
            out.iloc[i] = value
    return out
