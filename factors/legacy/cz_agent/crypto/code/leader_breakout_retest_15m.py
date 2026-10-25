"""Leader-style 15m breakout retest factor.

This candidate covers a separate entry structure from range-edge reversal:
after a recent accepted breakout or breakdown, price retests the breakout
level and closes back in the breakout direction.  It is intentionally stricter
than a raw breakout factor because the learned leader entries favored improved
entry price rather than chasing the first expansion bar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    return pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _recent_level(mask: pd.Series, level: pd.Series, window: int) -> pd.Series:
    return level.where(mask).ffill(limit=max(1, int(window)))


def calculate_factor(df: pd.DataFrame, parameters: dict | None = None) -> pd.Series:
    params = parameters or {}
    range_lookback = int(params.get("range_lookback", 48))
    breakout_window = int(params.get("breakout_window", 12))
    breakout_min_pct = float(params.get("breakout_min_pct", 0.28))
    retest_tolerance_pct = float(params.get("retest_tolerance_pct", 0.42))
    reclaim_buffer_pct = float(params.get("reclaim_buffer_pct", 0.08))
    ema_fast_period = int(params.get("ema_fast", 20))
    ema_slow_period = int(params.get("ema_slow", 96))
    atr_period = int(params.get("atr_period", 14))
    atr_min_pct = float(params.get("atr_min_pct", 0.16))
    atr_max_pct = float(params.get("atr_max_pct", 1.65))
    max_retest_distance_pct = float(params.get("max_retest_distance_pct", 1.20))
    min_body_ratio = float(params.get("min_body_ratio", 0.24))
    min_close_pos_long = float(params.get("min_close_pos_long", 0.55))
    max_close_pos_short = float(params.get("max_close_pos_short", 0.45))
    volume_ratio_min = float(params.get("volume_ratio_min", 0.80))
    cooldown_bars = int(params.get("signal_cooldown_bars", 14))
    signal_value = float(params.get("signal_value", 0.7))

    close = df["close"].astype("float64")
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")
    open_ = df["open"].astype("float64")
    volume = df.get("volume", pd.Series(0.0, index=df.index)).astype("float64")
    min_bars = max(range_lookback, ema_slow_period, atr_period) + breakout_window + 10
    if len(df) < min_bars:
        return pd.Series(0.0, index=df.index, dtype="float64")

    range_high = high.shift(1).rolling(range_lookback, min_periods=range_lookback).max()
    range_low = low.shift(1).rolling(range_lookback, min_periods=range_lookback).min()

    prev_close = close.shift(1)
    prev_range_high = range_high.shift(1)
    prev_range_low = range_low.shift(1)
    accepted_breakout = prev_close >= prev_range_high * (1.0 + breakout_min_pct / 100.0)
    accepted_breakdown = prev_close <= prev_range_low * (1.0 - breakout_min_pct / 100.0)
    breakout_level = _recent_level(accepted_breakout, prev_range_high, breakout_window)
    breakdown_level = _recent_level(accepted_breakdown, prev_range_low, breakout_window)

    ema_fast = close.ewm(span=ema_fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=ema_slow_period, adjust=False).mean()
    trend_up = (ema_fast >= ema_slow * 0.998) & (close >= ema_fast * 0.996)
    trend_down = (ema_fast <= ema_slow * 1.002) & (close <= ema_fast * 1.004)

    tr = _true_range(high, low, close)
    atr_pct = tr.rolling(atr_period, min_periods=atr_period).mean() / (close + 1e-10) * 100.0
    volume_ratio = volume / (volume.shift(1).rolling(20, min_periods=10).mean() + 1e-10)
    body_ratio = (close - open_).abs() / ((high - low).abs() + 1e-10)
    close_pos = (close - low) / ((high - low) + 1e-10)

    long_retest_distance = (close / (breakout_level + 1e-10) - 1.0) * 100.0
    short_retest_distance = (close / (breakdown_level + 1e-10) - 1.0) * 100.0
    volatility_ok = (atr_pct >= atr_min_pct) & (atr_pct <= atr_max_pct)
    liquidity_ok = volume_ratio >= volume_ratio_min
    candle_body_ok = body_ratio >= min_body_ratio

    long_signal = (
        breakout_level.notna()
        & trend_up
        & volatility_ok
        & liquidity_ok
        & candle_body_ok
        & (low <= breakout_level * (1.0 + retest_tolerance_pct / 100.0))
        & (close >= breakout_level * (1.0 + reclaim_buffer_pct / 100.0))
        & (long_retest_distance <= max_retest_distance_pct)
        & (close > open_)
        & (close_pos >= min_close_pos_long)
    )
    short_signal = (
        breakdown_level.notna()
        & trend_down
        & volatility_ok
        & liquidity_ok
        & candle_body_ok
        & (high >= breakdown_level * (1.0 - retest_tolerance_pct / 100.0))
        & (close <= breakdown_level * (1.0 - reclaim_buffer_pct / 100.0))
        & (short_retest_distance >= -max_retest_distance_pct)
        & (close < open_)
        & (close_pos <= max_close_pos_short)
    )

    result = pd.Series(0.0, index=df.index, dtype="float64")
    result[long_signal] = signal_value
    result[short_signal] = -signal_value

    if cooldown_bars > 0:
        cleaned = result.copy()
        last_long = -10**9
        last_short = -10**9
        for i, value in enumerate(cleaned.to_numpy()):
            if value > 0:
                if i - last_long < cooldown_bars:
                    cleaned.iloc[i] = 0.0
                else:
                    last_long = i
            elif value < 0:
                if i - last_short < cooldown_bars:
                    cleaned.iloc[i] = 0.0
                else:
                    last_short = i
        result = cleaned

    return result.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype("float64")
