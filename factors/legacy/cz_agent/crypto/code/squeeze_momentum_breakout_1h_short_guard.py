"""Squeeze momentum breakout with a late-short exhaustion guard.

This research variant keeps the base squeeze breakout logic, but blocks short
signals after an already extreme multi-day selloff.  The guard is intended for
leader-learned BNB short routing where the losing sample came from chasing the
tail of a deep cascade rather than from the first expansion leg.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    return pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)


def calculate_factor(df: pd.DataFrame, parameters: dict | None = None) -> pd.Series:
    params = parameters or {}
    bb_period = int(params.get("bb_period", 40))
    bb_std = float(params.get("bb_std", 2.0))
    squeeze_lookback = int(params.get("squeeze_lookback", 120))
    squeeze_quantile = float(params.get("squeeze_quantile", 0.25))
    breakout_lookback = int(params.get("breakout_lookback", 24))
    atr_period = int(params.get("atr_period", 14))
    range_expand_ratio = float(params.get("range_expand_ratio", 1.2))
    volume_ma_period = int(params.get("volume_ma_period", 30))
    volume_ratio_min = float(params.get("volume_ratio_min", 1.25))
    ema_fast_period = int(params.get("ema_fast", 20))
    ema_slow_period = int(params.get("ema_slow", 80))
    body_ratio_min = float(params.get("body_ratio_min", 0.42))
    max_extension_pct = float(params.get("max_extension_pct", 3.0)) / 100.0
    cooldown_bars = int(params.get("signal_cooldown_bars", 4))
    signal_value = float(params.get("signal_value", 0.7))
    short_guard_lookback = int(params.get("short_guard_lookback_bars", 96))
    short_guard_min_return = float(params.get("short_guard_min_return_pct", -10.0)) / 100.0

    close = df["close"].astype("float64")
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")
    open_ = df["open"].astype("float64")
    volume = df["volume"].astype("float64")
    min_bars = max(squeeze_lookback, ema_slow_period, breakout_lookback, short_guard_lookback) + 5
    if len(df) < min_bars:
        return pd.Series(0.0, index=df.index, dtype="float64")

    ma = close.rolling(bb_period, min_periods=bb_period).mean()
    std = close.rolling(bb_period, min_periods=bb_period).std()
    upper = ma + bb_std * std
    lower = ma - bb_std * std
    band_width = (upper - lower) / (ma.abs() + 1e-10)
    squeeze_line = band_width.rolling(squeeze_lookback, min_periods=squeeze_lookback // 2).quantile(squeeze_quantile)
    recent_squeeze = (band_width <= squeeze_line).shift(1).rolling(8, min_periods=1).max().fillna(0).astype(bool)

    prev_high = high.shift(1).rolling(breakout_lookback, min_periods=breakout_lookback).max()
    prev_low = low.shift(1).rolling(breakout_lookback, min_periods=breakout_lookback).min()
    break_up = close > prev_high
    break_down = close < prev_low
    extension_up = (close - prev_high) / (close + 1e-10)
    extension_down = (prev_low - close) / (close + 1e-10)
    extension_ok_up = extension_up.between(0, max_extension_pct)
    extension_ok_down = extension_down.between(0, max_extension_pct)

    tr = _true_range(high, low, close)
    atr = tr.rolling(atr_period, min_periods=atr_period).mean()
    range_expanding = tr > atr * range_expand_ratio
    vol_ratio = volume / (volume.rolling(volume_ma_period, min_periods=10).mean() + 1e-10)
    volume_ok = vol_ratio >= volume_ratio_min

    ema_fast = close.ewm(span=ema_fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=ema_slow_period, adjust=False).mean()
    ema_slope = (ema_fast - ema_fast.shift(5)) / (ema_fast.shift(5) + 1e-10)
    trend_up = (ema_fast > ema_slow) & (ema_slope > 0)
    trend_down = (ema_fast < ema_slow) & (ema_slope < 0)

    candle_range = high - low + 1e-10
    body = close - open_
    body_ratio = body.abs() / candle_range
    bull_body = (body > 0) & (body_ratio >= body_ratio_min)
    bear_body = (body < 0) & (body_ratio >= body_ratio_min)

    short_return = close / (close.shift(short_guard_lookback) + 1e-10) - 1.0
    short_not_late_chase = short_return >= short_guard_min_return

    result = pd.Series(0.0, index=df.index, dtype="float64")
    result[recent_squeeze & break_up & extension_ok_up & range_expanding & volume_ok & trend_up & bull_body] = signal_value
    result[
        recent_squeeze
        & break_down
        & extension_ok_down
        & range_expanding
        & volume_ok
        & trend_down
        & bear_body
        & short_not_late_chase
    ] = -signal_value

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
