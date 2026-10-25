"""Squeeze breakout continuation factor for meme profit-max mode.

The factor scores directional continuation after a volume-backed breakout.
It deliberately avoids future data and can work even when funding/OI fields are
missing, while using them as extra confirmation when available.
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
    if n < 90:
        return out

    close = df["close"].astype(float).values
    open_ = df["open"].astype(float).values if "open" in df else np.roll(close, 1)
    open_[0] = close[0]
    high = df["high"].astype(float).values if "high" in df else close
    low = df["low"].astype(float).values if "low" in df else close
    volume = df["volume"].astype(float).values if "volume" in df else np.ones(n)
    oi = df["open_interest"].ffill().astype(float).values if "open_interest" in df else np.zeros(n)
    funding = df["funding_rate"].fillna(0).astype(float).values if "funding_rate" in df else np.zeros(n)

    close_s = pd.Series(close)
    ema9 = close_s.ewm(span=9, min_periods=5).mean().values
    ema30 = close_s.ewm(span=30, min_periods=15).mean().values
    vol_med30 = pd.Series(volume).rolling(30, min_periods=15).median().fillna(0.0).values

    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr12 = pd.Series(tr).rolling(12, min_periods=6).mean().fillna(0.0).values
    atr60 = pd.Series(tr).rolling(60, min_periods=30).mean().fillna(0.0).values

    hh30 = pd.Series(high).rolling(30, min_periods=15).max().fillna(0.0).values
    ll30 = pd.Series(low).rolling(30, min_periods=15).min().fillna(0.0).values
    min_output = _env_float("MEME_SQUEEZE_BREAKOUT_MIN_OUTPUT", 0.26)
    max_ret5_long = _env_float("MEME_SQUEEZE_BREAKOUT_MAX_RET5_LONG", 20.0)
    max_ret5_short = _env_float("MEME_SQUEEZE_BREAKOUT_MAX_RET5_SHORT", 16.0)

    raw = np.zeros(n)
    for i in range(60, n):
        ret3 = _pct(close[i], close[i - 3])
        ret5 = _pct(close[i], close[i - 5])
        ret15 = _pct(close[i], close[i - 15])
        ret30 = _pct(close[i], close[i - 30])
        prev_ret5 = _pct(close[i - 5], close[i - 10])
        accel = ret5 - prev_ret5
        vol_ratio = volume[i] / vol_med30[i] if vol_med30[i] > 0 else 1.0
        atr_ratio = atr12[i] / atr60[i] if atr60[i] > 0 else 1.0
        oi_roc5 = _pct(oi[i], oi[i - 5]) if oi[i - 5] > 0 else 0.0
        ema_bias = _pct(ema9[i], ema30[i]) if ema30[i] > 0 else 0.0

        bar_range = max(1e-12, high[i] - low[i])
        close_in_bar = (close[i] - low[i]) / bar_range
        upper_wick = (high[i] - max(open_[i], close[i])) / bar_range
        lower_wick = (min(open_[i], close[i]) - low[i]) / bar_range

        roll_range = max(1e-12, hh30[i] - ll30[i])
        close_pos30 = (close[i] - ll30[i]) / roll_range

        long_score = 0.0
        long_evidence = 0
        if ret5 > 1.8 and ret15 > 4.0 and accel > 0.2:
            long_score += min(0.25, ret15 / 22.0 * 0.25)
            long_evidence += 1
        if close_pos30 > 0.72 and close_in_bar > 0.60:
            long_score += min(0.18, (close_pos30 - 0.55) * 0.36)
            long_evidence += 1
        if vol_ratio > 1.6 and atr_ratio > 0.85:
            long_score += min(0.18, (vol_ratio - 1.0) / 5.0 * 0.18)
            long_evidence += 1
        if ema_bias > 0.04 and ret30 > 5.0:
            long_score += min(0.14, ret30 / 45.0 * 0.14)
            long_evidence += 1
        if oi_roc5 > 0.5:
            long_score += min(0.13, oi_roc5 / 10.0 * 0.13)
            long_evidence += 1
        if funding[i] < -0.03 and ret3 > 0:
            long_score += 0.08
            long_evidence += 1
        if upper_wick > 0.48 or ret5 > max_ret5_long:
            long_score *= 0.55

        short_score = 0.0
        short_evidence = 0
        if ret5 < -1.8 and ret15 < -4.0 and accel < -0.2:
            short_score += min(0.25, abs(ret15) / 22.0 * 0.25)
            short_evidence += 1
        if close_pos30 < 0.28 and close_in_bar < 0.40:
            short_score += min(0.18, (0.45 - close_pos30) * 0.36)
            short_evidence += 1
        if vol_ratio > 1.6 and atr_ratio > 0.85:
            short_score += min(0.18, (vol_ratio - 1.0) / 5.0 * 0.18)
            short_evidence += 1
        if ema_bias < -0.04 and ret30 < -5.0:
            short_score += min(0.14, abs(ret30) / 45.0 * 0.14)
            short_evidence += 1
        if abs(oi_roc5) > 0.8:
            short_score += min(0.13, abs(oi_roc5) / 10.0 * 0.13)
            short_evidence += 1
        if funding[i] > 0.03 and ret3 < 0:
            short_score += 0.08
            short_evidence += 1
        if lower_wick > 0.48 or abs(ret5) > max_ret5_short:
            short_score *= 0.55

        if long_evidence >= 3 and long_score >= min_output and long_score >= short_score:
            raw[i] = min(1.0, long_score * 1.55)
        elif short_evidence >= 3 and short_score >= min_output:
            raw[i] = -min(1.0, short_score * 1.55)

    smooth = pd.Series(raw).ewm(span=3, min_periods=2).mean().values
    for i in range(60, n):
        value = float(np.clip(smooth[i], -1.0, 1.0))
        if abs(value) >= min_output:
            out.iloc[i] = value
    return out
