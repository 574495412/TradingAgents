"""Micro burst scalp factor for 1m meme event trading.

The existing momentum factors wait for broader 5-30 minute structure. This
factor targets the first tradable 1-3 minute impulse after an anomaly enters
the candidate set. It stays lag-safe: only current and historical bars are
used, and output is capped when the move is already too extended.
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
    if n < 24:
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
    ema5 = close_s.ewm(span=5, min_periods=3).mean().values
    ema13 = close_s.ewm(span=13, min_periods=6).mean().values
    vol_med12 = pd.Series(volume).rolling(12, min_periods=6).median().fillna(0.0).values
    hh12 = pd.Series(high).rolling(12, min_periods=6).max().fillna(0.0).values
    ll12 = pd.Series(low).rolling(12, min_periods=6).min().fillna(0.0).values

    min_output = _env_float("MEME_MICRO_BURST_MIN_OUTPUT", 0.30)
    min_ret1 = _env_float("MEME_MICRO_BURST_MIN_RET1_PCT", 0.35)
    min_ret3 = _env_float("MEME_MICRO_BURST_MIN_RET3_PCT", 0.95)
    min_vol_ratio = _env_float("MEME_MICRO_BURST_MIN_VOLUME_RATIO", 2.2)
    max_ret5_long = _env_float("MEME_MICRO_BURST_MAX_RET5_LONG", 10.0)
    max_ret5_short = _env_float("MEME_MICRO_BURST_MAX_RET5_SHORT", 8.0)
    max_wick = _env_float("MEME_MICRO_BURST_MAX_COUNTER_WICK", 0.46)

    raw = np.zeros(n)
    for i in range(18, n):
        ret1 = _pct(close[i], close[i - 1])
        ret2 = _pct(close[i], close[i - 2])
        ret3 = _pct(close[i], close[i - 3])
        ret5 = _pct(close[i], close[i - 5])
        prev_ret3 = _pct(close[i - 3], close[i - 6])
        accel = ret3 - prev_ret3
        vol_ratio = volume[i] / vol_med12[i] if vol_med12[i] > 0 else 1.0
        oi_roc3 = _pct(oi[i], oi[i - 3]) if oi[i - 3] > 0 else 0.0
        ema_bias = _pct(ema5[i], ema13[i]) if ema13[i] > 0 else 0.0

        bar_range = max(1e-12, high[i] - low[i])
        close_in_bar = (close[i] - low[i]) / bar_range
        upper_wick = (high[i] - max(open_[i], close[i])) / bar_range
        lower_wick = (min(open_[i], close[i]) - low[i]) / bar_range
        range12 = max(1e-12, hh12[i] - ll12[i])
        close_pos12 = (close[i] - ll12[i]) / range12

        long_score = 0.0
        long_evidence = 0
        if ret1 >= min_ret1 and ret3 >= min_ret3 and accel > 0:
            long_score += min(0.34, ret3 / 4.5 * 0.34)
            long_evidence += 1
        if ret2 > 0.55 and close_in_bar >= 0.62 and close_pos12 >= 0.62:
            long_score += 0.18
            long_evidence += 1
        if vol_ratio >= min_vol_ratio:
            long_score += min(0.20, (vol_ratio - 1.0) / 6.0 * 0.20)
            long_evidence += 1
        if oi_roc3 > 0.4:
            long_score += min(0.12, oi_roc3 / 6.0 * 0.12)
            long_evidence += 1
        if ema_bias > 0.02:
            long_score += min(0.08, ema_bias / 1.2 * 0.08)
            long_evidence += 1
        if funding[i] < -0.03 and ret3 > 0:
            long_score += 0.06
            long_evidence += 1
        if upper_wick > max_wick or ret5 > max_ret5_long:
            long_score *= 0.45

        short_score = 0.0
        short_evidence = 0
        if ret1 <= -min_ret1 and ret3 <= -min_ret3 and accel < 0:
            short_score += min(0.34, abs(ret3) / 4.5 * 0.34)
            short_evidence += 1
        if ret2 < -0.55 and close_in_bar <= 0.38 and close_pos12 <= 0.38:
            short_score += 0.18
            short_evidence += 1
        if vol_ratio >= min_vol_ratio:
            short_score += min(0.20, (vol_ratio - 1.0) / 6.0 * 0.20)
            short_evidence += 1
        if abs(oi_roc3) > 0.6:
            short_score += min(0.12, abs(oi_roc3) / 6.0 * 0.12)
            short_evidence += 1
        if ema_bias < -0.02:
            short_score += min(0.08, abs(ema_bias) / 1.2 * 0.08)
            short_evidence += 1
        if funding[i] > 0.03 and ret3 < 0:
            short_score += 0.06
            short_evidence += 1
        if lower_wick > max_wick or abs(ret5) > max_ret5_short:
            short_score *= 0.45

        if long_evidence >= 3 and long_score >= min_output and long_score >= short_score:
            raw[i] = min(1.0, long_score * 1.55)
        elif short_evidence >= 3 and short_score >= min_output:
            raw[i] = -min(1.0, short_score * 1.55)

    smooth = pd.Series(raw).ewm(span=2, min_periods=1).mean().values
    for i in range(18, n):
        value = float(np.clip(smooth[i], -1.0, 1.0))
        if abs(value) >= min_output:
            out.iloc[i] = value
    return out


def _ewm_adjusted_last(values: np.ndarray, span: int, min_periods: int) -> float:
    alpha = 2.0 / (span + 1.0)
    decay = 1.0 - alpha
    numerator = 0.0
    denominator = 0.0
    count = 0
    result = float("nan")
    for raw in values:
        value = 0.0 if not np.isfinite(raw) else float(raw)
        numerator = value + decay * numerator
        denominator = 1.0 + decay * denominator
        count += 1
        if count >= min_periods:
            result = numerator / denominator
    return result


def calculate_last_factor(df: pd.DataFrame) -> float:
    if df is None:
        return 0.0
    n = len(df)
    if n < 24:
        return 0.0

    close = df["close"].astype(float).values
    open_ = df["open"].astype(float).values if "open" in df else np.roll(close, 1)
    open_[0] = close[0]
    high = df["high"].astype(float).values if "high" in df else close
    low = df["low"].astype(float).values if "low" in df else close
    volume = df["volume"].astype(float).values if "volume" in df else np.ones(n)
    oi = df["open_interest"].ffill().astype(float).values if "open_interest" in df else np.zeros(n)
    funding = df["funding_rate"].fillna(0).astype(float).values if "funding_rate" in df else np.zeros(n)

    close_s = pd.Series(close)
    ema5 = close_s.ewm(span=5, min_periods=3).mean().values
    ema13 = close_s.ewm(span=13, min_periods=6).mean().values
    vol_med12 = pd.Series(volume).rolling(12, min_periods=6).median().fillna(0.0).values
    hh12 = pd.Series(high).rolling(12, min_periods=6).max().fillna(0.0).values
    ll12 = pd.Series(low).rolling(12, min_periods=6).min().fillna(0.0).values

    min_output = _env_float("MEME_MICRO_BURST_MIN_OUTPUT", 0.30)
    min_ret1 = _env_float("MEME_MICRO_BURST_MIN_RET1_PCT", 0.35)
    min_ret3 = _env_float("MEME_MICRO_BURST_MIN_RET3_PCT", 0.95)
    min_vol_ratio = _env_float("MEME_MICRO_BURST_MIN_VOLUME_RATIO", 2.2)
    max_ret5_long = _env_float("MEME_MICRO_BURST_MAX_RET5_LONG", 10.0)
    max_ret5_short = _env_float("MEME_MICRO_BURST_MAX_RET5_SHORT", 8.0)
    max_wick = _env_float("MEME_MICRO_BURST_MAX_COUNTER_WICK", 0.46)

    raw = np.zeros(n)
    for i in range(18, n):
        ret1 = _pct(close[i], close[i - 1])
        ret2 = _pct(close[i], close[i - 2])
        ret3 = _pct(close[i], close[i - 3])
        ret5 = _pct(close[i], close[i - 5])
        prev_ret3 = _pct(close[i - 3], close[i - 6])
        accel = ret3 - prev_ret3
        vol_ratio = volume[i] / vol_med12[i] if vol_med12[i] > 0 else 1.0
        oi_roc3 = _pct(oi[i], oi[i - 3]) if oi[i - 3] > 0 else 0.0
        ema_bias = _pct(ema5[i], ema13[i]) if ema13[i] > 0 else 0.0

        bar_range = max(1e-12, high[i] - low[i])
        close_in_bar = (close[i] - low[i]) / bar_range
        upper_wick = (high[i] - max(open_[i], close[i])) / bar_range
        lower_wick = (min(open_[i], close[i]) - low[i]) / bar_range
        range12 = max(1e-12, hh12[i] - ll12[i])
        close_pos12 = (close[i] - ll12[i]) / range12

        long_score = 0.0
        long_evidence = 0
        if ret1 >= min_ret1 and ret3 >= min_ret3 and accel > 0:
            long_score += min(0.34, ret3 / 4.5 * 0.34)
            long_evidence += 1
        if ret2 > 0.55 and close_in_bar >= 0.62 and close_pos12 >= 0.62:
            long_score += 0.18
            long_evidence += 1
        if vol_ratio >= min_vol_ratio:
            long_score += min(0.20, (vol_ratio - 1.0) / 6.0 * 0.20)
            long_evidence += 1
        if oi_roc3 > 0.4:
            long_score += min(0.12, oi_roc3 / 6.0 * 0.12)
            long_evidence += 1
        if ema_bias > 0.02:
            long_score += min(0.08, ema_bias / 1.2 * 0.08)
            long_evidence += 1
        if funding[i] < -0.03 and ret3 > 0:
            long_score += 0.06
            long_evidence += 1
        if upper_wick > max_wick or ret5 > max_ret5_long:
            long_score *= 0.45

        short_score = 0.0
        short_evidence = 0
        if ret1 <= -min_ret1 and ret3 <= -min_ret3 and accel < 0:
            short_score += min(0.34, abs(ret3) / 4.5 * 0.34)
            short_evidence += 1
        if ret2 < -0.55 and close_in_bar <= 0.38 and close_pos12 <= 0.38:
            short_score += 0.18
            short_evidence += 1
        if vol_ratio >= min_vol_ratio:
            short_score += min(0.20, (vol_ratio - 1.0) / 6.0 * 0.20)
            short_evidence += 1
        if abs(oi_roc3) > 0.6:
            short_score += min(0.12, abs(oi_roc3) / 6.0 * 0.12)
            short_evidence += 1
        if ema_bias < -0.02:
            short_score += min(0.08, abs(ema_bias) / 1.2 * 0.08)
            short_evidence += 1
        if funding[i] > 0.03 and ret3 < 0:
            short_score += 0.06
            short_evidence += 1
        if lower_wick > max_wick or abs(ret5) > max_ret5_short:
            short_score *= 0.45

        if long_evidence >= 3 and long_score >= min_output and long_score >= short_score:
            raw[i] = min(1.0, long_score * 1.55)
        elif short_evidence >= 3 and short_score >= min_output:
            raw[i] = -min(1.0, short_score * 1.55)

    value = float(np.clip(_ewm_adjusted_last(raw, span=2, min_periods=1), -1.0, 1.0))
    return value if abs(value) >= min_output else 0.0
