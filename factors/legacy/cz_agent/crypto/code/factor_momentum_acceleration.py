"""Momentum acceleration factor for high-risk meme trading.

This factor targets the earliest tradable part of a meme move: price
acceleration plus volume expansion plus acceptable wick structure. It is
intentionally aggressive and should be used through a high-risk portfolio.
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
    if n < 45:
        return out

    close = df["close"].astype(float).values
    open_ = df["open"].astype(float).values if "open" in df else np.roll(close, 1)
    open_[0] = close[0]
    high = df["high"].astype(float).values if "high" in df else close
    low = df["low"].astype(float).values if "low" in df else close
    volume = df["volume"].astype(float).values if "volume" in df else np.ones(n)
    oi = df["open_interest"].ffill().astype(float).values if "open_interest" in df else np.zeros(n)
    funding = df["funding_rate"].fillna(0).astype(float).values if "funding_rate" in df else np.zeros(n)

    vol_ma = pd.Series(volume).rolling(20, min_periods=10).median().fillna(0.0).values
    close_s = pd.Series(close)
    ema8 = close_s.ewm(span=8, min_periods=5).mean().values
    ema21 = close_s.ewm(span=21, min_periods=10).mean().values
    min_output = _env_float("MEME_MOMENTUM_ACCELERATION_MIN_OUTPUT", 0.24)
    max_ret5_long = _env_float("MEME_MOMENTUM_ACCELERATION_MAX_RET5_LONG", 18.0)
    max_ret5_short = _env_float("MEME_MOMENTUM_ACCELERATION_MAX_RET5_SHORT", 14.0)

    raw = np.zeros(n)
    for i in range(30, n):
        ret1 = _pct(close[i], close[i - 1])
        ret3 = _pct(close[i], close[i - 3])
        ret5 = _pct(close[i], close[i - 5])
        ret15 = _pct(close[i], close[i - 15])
        ret30 = _pct(close[i], close[i - 30])
        accel = ret3 - (_pct(close[i - 3], close[i - 6]) if i >= 36 else 0.0)
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 1.0
        oi_roc3 = _pct(oi[i], oi[i - 3]) if oi[i - 3] > 0 else 0.0

        bar_range = max(1e-12, high[i] - low[i])
        upper_wick = (high[i] - max(open_[i], close[i])) / bar_range
        lower_wick = (min(open_[i], close[i]) - low[i]) / bar_range
        close_pos = (close[i] - low[i]) / bar_range
        ema_bias = _pct(ema8[i], ema21[i]) if ema21[i] > 0 else 0.0

        long_score = 0.0
        short_score = 0.0
        long_evidence = 0
        short_evidence = 0

        if ret1 > 0.25 and ret3 > 0.8 and accel > 0.15:
            long_score += min(0.28, ret3 / 7.0 * 0.28)
            long_evidence += 1
        if ret5 > 1.6 and ret15 > 2.8:
            long_score += min(0.22, ret15 / 18.0 * 0.22)
            long_evidence += 1
        if vol_ratio > 1.8:
            long_score += min(0.18, (vol_ratio - 1.0) / 5.0 * 0.18)
            long_evidence += 1
        if oi_roc3 > 0.8:
            long_score += min(0.16, oi_roc3 / 8.0 * 0.16)
            long_evidence += 1
        if ema_bias > 0.06 and close_pos >= 0.58:
            long_score += 0.10
            long_evidence += 1
        if funding[i] < -0.04 and ret5 > 0:
            long_score += 0.06
            long_evidence += 1
        if upper_wick > 0.50 or ret5 > max_ret5_long:
            long_score *= 0.55
        if ret30 > 35 and oi_roc3 <= 0:
            long_score *= 0.70

        if ret1 < -0.25 and ret3 < -0.8 and accel < -0.15:
            short_score += min(0.28, abs(ret3) / 7.0 * 0.28)
            short_evidence += 1
        if ret5 < -1.6 and ret15 < -2.8:
            short_score += min(0.22, abs(ret15) / 18.0 * 0.22)
            short_evidence += 1
        if vol_ratio > 1.8:
            short_score += min(0.16, (vol_ratio - 1.0) / 5.0 * 0.16)
            short_evidence += 1
        if abs(oi_roc3) > 1.0:
            short_score += min(0.16, abs(oi_roc3) / 8.0 * 0.16)
            short_evidence += 1
        if ema_bias < -0.06 and close_pos <= 0.42:
            short_score += 0.10
            short_evidence += 1
        if funding[i] > 0.04 and ret5 < 0:
            short_score += 0.06
            short_evidence += 1
        if lower_wick > 0.50 or abs(ret5) > max_ret5_short:
            short_score *= 0.55
        if ret30 < -35 and oi_roc3 >= 0:
            short_score *= 0.70

        if long_evidence >= 3 and long_score >= min_output and long_score >= short_score:
            raw[i] = min(1.0, long_score * 1.45)
        elif short_evidence >= 3 and short_score >= min_output:
            raw[i] = -min(1.0, short_score * 1.45)

    smooth = pd.Series(raw).ewm(span=3, min_periods=2).mean().values
    for i in range(30, n):
        value = float(np.clip(smooth[i], -1.0, 1.0))
        if abs(value) >= min_output:
            out.iloc[i] = value
    return out
