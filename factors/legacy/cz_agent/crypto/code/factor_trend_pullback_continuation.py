"""Trend pullback continuation factor for meme leaders.

The factor targets the middle of a strong meme move after a controlled
pullback, where pure spike factors often go quiet but trend followers still
hold or add.  It stays lag-safe by using only current and historical bars.
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


def _safe_ratio(cur: float, base: float, default: float = 1.0) -> float:
    if base <= 0 or not np.isfinite(base):
        return default
    return cur / base


def calculate_factor(df: pd.DataFrame) -> pd.Series:
    n = len(df)
    out = pd.Series(0.0, index=df.index)
    if n < 180:
        return out

    close = df["close"].astype(float).values
    open_ = df["open"].astype(float).values if "open" in df else np.roll(close, 1)
    open_[0] = close[0]
    high = df["high"].astype(float).values if "high" in df else close
    low = df["low"].astype(float).values if "low" in df else close
    volume = df["volume"].astype(float).values if "volume" in df else np.ones(n)
    oi = df["open_interest"].ffill().astype(float).values if "open_interest" in df else np.zeros(n)

    close_s = pd.Series(close)
    ema12 = close_s.ewm(span=12, min_periods=6).mean().values
    ema48 = close_s.ewm(span=48, min_periods=24).mean().values
    ema120 = close_s.ewm(span=120, min_periods=60).mean().values
    vol_med60 = pd.Series(volume).rolling(60, min_periods=30).median().fillna(0.0).values
    hh60 = pd.Series(high).rolling(60, min_periods=30).max().fillna(0.0).values
    ll60 = pd.Series(low).rolling(60, min_periods=30).min().fillna(0.0).values

    min_output = _env_float("MEME_TREND_PULLBACK_MIN_OUTPUT", 0.24)
    max_ret5_long = _env_float("MEME_TREND_PULLBACK_MAX_RET5_LONG", 9.0)
    max_ret5_short = _env_float("MEME_TREND_PULLBACK_MAX_RET5_SHORT", 8.0)
    max_pullback_long = _env_float("MEME_TREND_PULLBACK_MAX_PULLBACK_LONG_PCT", 9.0)
    max_bounce_short = _env_float("MEME_TREND_PULLBACK_MAX_BOUNCE_SHORT_PCT", 9.0)

    raw = np.zeros(n)
    for i in range(150, n):
        ret5 = _pct(close[i], close[i - 5])
        ret15 = _pct(close[i], close[i - 15])
        ret30 = _pct(close[i], close[i - 30])
        ret60 = _pct(close[i], close[i - 60])
        ret120 = _pct(close[i], close[i - 120])
        vol_ratio = _safe_ratio(volume[i], vol_med60[i])
        oi_roc15 = _pct(oi[i], oi[i - 15]) if oi[i - 15] > 0 else 0.0

        bar_range = max(1e-12, high[i] - low[i])
        close_in_bar = (close[i] - low[i]) / bar_range
        upper_wick = (high[i] - max(open_[i], close[i])) / bar_range
        lower_wick = (min(open_[i], close[i]) - low[i]) / bar_range
        pullback_from_high = (close[i] / max(hh60[i], 1e-12) - 1.0) * 100.0
        bounce_from_low = (close[i] / max(ll60[i], 1e-12) - 1.0) * 100.0

        long_score = 0.0
        long_evidence = 0
        if ret60 > 6.0 and ret120 > 8.0:
            long_score += min(0.24, ret120 / 80.0 * 0.24)
            long_evidence += 1
        if ema12[i] > ema48[i] > ema120[i]:
            long_score += 0.18
            long_evidence += 1
        if -max_pullback_long <= pullback_from_high <= -0.2 and ret5 > 0 and ret15 > -1.5:
            long_score += min(0.22, (max_pullback_long + pullback_from_high) / max_pullback_long * 0.22)
            long_evidence += 1
        elif pullback_from_high > -0.8 and ret15 > 1.0 and close_in_bar > 0.58:
            long_score += 0.14
            long_evidence += 1
        if close[i] > ema12[i] and close_in_bar > 0.55:
            long_score += 0.12
            long_evidence += 1
        if vol_ratio > 1.15:
            long_score += min(0.14, (vol_ratio - 1.0) / 3.0 * 0.14)
            long_evidence += 1
        if oi_roc15 > 1.0:
            long_score += min(0.10, oi_roc15 / 8.0 * 0.10)
            long_evidence += 1
        if ret30 < -2.0 or ret5 > max_ret5_long or upper_wick > 0.48:
            long_score *= 0.45

        short_score = 0.0
        short_evidence = 0
        if ret60 < -6.0 and ret120 < -8.0:
            short_score += min(0.24, abs(ret120) / 80.0 * 0.24)
            short_evidence += 1
        if ema12[i] < ema48[i] < ema120[i]:
            short_score += 0.18
            short_evidence += 1
        if 0.2 <= bounce_from_low <= max_bounce_short and ret5 < 0 and ret15 < 1.5:
            short_score += min(0.22, (max_bounce_short - bounce_from_low) / max_bounce_short * 0.22)
            short_evidence += 1
        elif bounce_from_low < 0.8 and ret15 < -1.0 and close_in_bar < 0.42:
            short_score += 0.14
            short_evidence += 1
        if close[i] < ema12[i] and close_in_bar < 0.45:
            short_score += 0.12
            short_evidence += 1
        if vol_ratio > 1.15:
            short_score += min(0.14, (vol_ratio - 1.0) / 3.0 * 0.14)
            short_evidence += 1
        if abs(oi_roc15) > 1.0:
            short_score += min(0.10, abs(oi_roc15) / 8.0 * 0.10)
            short_evidence += 1
        if ret30 > 2.0 or abs(ret5) > max_ret5_short or lower_wick > 0.48:
            short_score *= 0.45

        if long_evidence >= 4 and long_score >= min_output and long_score >= short_score:
            raw[i] = min(1.0, long_score * 1.45)
        elif short_evidence >= 4 and short_score >= min_output:
            raw[i] = -min(1.0, short_score * 1.45)

    smooth = pd.Series(raw).ewm(span=5, min_periods=3).mean().values
    for i in range(150, n):
        value = float(np.clip(smooth[i], -1.0, 1.0))
        if abs(value) >= min_output:
            out.iloc[i] = value
    return out
