"""
Donchian 波动率扩张突破因子 v1.0

用途:
  补齐主策略缺失的“横盘收缩后趋势启动”入口，和 EMA 4h 回踩因子低相关。
  设计参考成熟趋势跟踪/CTA 常见的 Donchian channel breakout 与波动率过滤。

信号:
  多头: 收盘价向上突破过去 N 根高点，且此前波动率处于收缩状态、当前 ATR 扩张、
        成交量放大、长周期 EMA 方向向上。
  空头: 对称逻辑。
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


def calculate_factor(df: pd.DataFrame, parameters: dict | None = None) -> pd.Series:
    params = parameters or {}
    lookback = int(params.get("donchian_lookback", 20))
    compression_lookback = int(params.get("compression_lookback", 60))
    compression_recent_bars = int(params.get("compression_recent_bars", 8))
    compression_quantile = float(params.get("compression_quantile", 0.35))
    atr_period = int(params.get("atr_period", 14))
    atr_expand_ratio = float(params.get("atr_expand_ratio", 1.08))
    volume_ma_period = int(params.get("volume_ma_period", 20))
    volume_ratio_min = float(params.get("volume_ratio_min", 1.15))
    ema_fast_period = int(params.get("ema_fast", 20))
    ema_slow_period = int(params.get("ema_slow", 80))
    max_breakout_distance_pct = float(params.get("max_breakout_distance_pct", 2.8)) / 100.0
    cooldown_bars = int(params.get("signal_cooldown_bars", 3))
    signal_value = float(params.get("signal_value", 0.7))

    close = df["close"].astype("float64")
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")
    open_ = df["open"].astype("float64")
    volume = df["volume"].astype("float64")

    min_bars = max(lookback + compression_lookback, ema_slow_period, volume_ma_period) + 5
    if len(df) < min_bars:
        return pd.Series(0.0, index=df.index, dtype="float64")

    prev_high = high.shift(1).rolling(lookback, min_periods=lookback).max()
    prev_low = low.shift(1).rolling(lookback, min_periods=lookback).min()
    break_up = close > prev_high
    break_down = close < prev_low

    breakout_distance_up = (close - prev_high) / (close + 1e-10)
    breakout_distance_down = (prev_low - close) / (close + 1e-10)
    not_too_extended_up = breakout_distance_up.between(0, max_breakout_distance_pct)
    not_too_extended_down = breakout_distance_down.between(0, max_breakout_distance_pct)

    tr = _true_range(high, low, close)
    atr = tr.rolling(atr_period, min_periods=atr_period).mean()
    atr_pct = atr / (close + 1e-10)
    atr_q = atr_pct.rolling(compression_lookback, min_periods=compression_lookback // 2).quantile(compression_quantile)
    compressed_now = atr_pct <= atr_q
    was_compressed = compressed_now.shift(1).rolling(compression_recent_bars, min_periods=1).max().fillna(0).astype(bool)
    atr_expanding = atr > atr.rolling(10, min_periods=5).mean() * atr_expand_ratio

    vol_ratio = volume / (volume.rolling(volume_ma_period, min_periods=5).mean() + 1e-10)
    volume_confirm = vol_ratio >= volume_ratio_min

    ema_fast = close.ewm(span=ema_fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=ema_slow_period, adjust=False).mean()
    ema_fast_slope = (ema_fast - ema_fast.shift(3)) / (ema_fast.shift(3) + 1e-10)
    trend_up = (ema_fast > ema_slow) | (ema_fast_slope > 0.0005)
    trend_down = (ema_fast < ema_slow) | (ema_fast_slope < -0.0005)

    body = close - open_
    candle_range = high - low + 1e-10
    body_ratio = body.abs() / candle_range
    bullish_body = (body > 0) & (body_ratio >= 0.35)
    bearish_body = (body < 0) & (body_ratio >= 0.35)

    result = pd.Series(0.0, index=df.index, dtype="float64")
    bull = (
        break_up
        & not_too_extended_up
        & was_compressed
        & atr_expanding
        & volume_confirm
        & trend_up
        & bullish_body
    )
    bear = (
        break_down
        & not_too_extended_down
        & was_compressed
        & atr_expanding
        & volume_confirm
        & trend_down
        & bearish_body
    )
    result[bull] = signal_value
    result[bear] = -signal_value

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
