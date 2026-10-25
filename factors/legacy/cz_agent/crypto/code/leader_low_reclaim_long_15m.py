"""Leader-style 15m low-edge reclaim long factor.

This candidate isolates one structure seen in external main-strategy leader
samples: price probes the recent range low, refuses to continue lower, then
reclaims with a clean close.  It is long-only by design so IC and execution can
be judged separately from short rejection behavior.
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


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100.0 - 100.0 / (1.0 + rs)


def calculate_factor(df: pd.DataFrame, parameters: dict | None = None) -> pd.Series:
    params = parameters or {}
    range_lookback = int(params.get("range_lookback", 20))
    ema_fast_period = int(params.get("ema_fast", 20))
    ema_slow_period = int(params.get("ema_slow", 96))
    atr_period = int(params.get("atr_period", 14))
    atr_min_pct = float(params.get("atr_min_pct", 0.12))
    atr_max_pct = float(params.get("atr_max_pct", 1.45))
    edge_zone_pct = float(params.get("edge_zone_pct", 1.35))
    reclaim_buffer_pct = float(params.get("reclaim_buffer_pct", 0.10))
    max_pierce_pct = float(params.get("max_pierce_pct", 0.65))
    min_close_pos = float(params.get("min_close_pos", 0.58))
    min_body_ratio = float(params.get("min_body_ratio", 0.28))
    ret5_min_pct = float(params.get("ret5_min_pct", -1.90))
    ret20_min_pct = float(params.get("ret20_min_pct", -5.20))
    ret20_max_pct = float(params.get("ret20_max_pct", 4.80))
    volume_ratio_min = float(params.get("volume_ratio_min", 0.75))
    rsi_period = int(params.get("rsi_period", 14))
    rsi_min = float(params.get("rsi_min", 28.0))
    rsi_max = float(params.get("rsi_max", 62.0))
    cooldown_bars = int(params.get("signal_cooldown_bars", 12))
    signal_value = float(params.get("signal_value", 0.7))

    close = df["close"].astype("float64")
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")
    open_ = df["open"].astype("float64")
    volume = df.get("volume", pd.Series(0.0, index=df.index)).astype("float64")
    min_bars = max(range_lookback, ema_slow_period, atr_period, rsi_period) + 10
    if len(df) < min_bars:
        return pd.Series(0.0, index=df.index, dtype="float64")

    ema_fast = close.ewm(span=ema_fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=ema_slow_period, adjust=False).mean()
    trend_not_bearish = (ema_fast >= ema_slow * 0.992) | (close >= ema_slow * 0.996)

    prev_low = low.shift(1).rolling(range_lookback, min_periods=range_lookback).min()
    from_low_close = (close / (prev_low + 1e-10) - 1.0) * 100.0
    from_low_low = (low / (prev_low + 1e-10) - 1.0) * 100.0

    tr = _true_range(high, low, close)
    atr_pct = tr.rolling(atr_period, min_periods=atr_period).mean() / (close + 1e-10) * 100.0
    rsi = _rsi(close, rsi_period)
    ret5 = close.pct_change(5) * 100.0
    ret20 = close.pct_change(20) * 100.0
    body_ratio = (close - open_).abs() / ((high - low).abs() + 1e-10)
    close_pos = (close - low) / ((high - low) + 1e-10)
    volume_ratio = volume / (volume.shift(1).rolling(20, min_periods=10).mean() + 1e-10)

    low_probe = (from_low_low >= -max_pierce_pct) & (from_low_close <= edge_zone_pct)
    reclaimed = (close >= prev_low * (1.0 + reclaim_buffer_pct / 100.0)) & (close > open_)
    not_falling_knife = (ret5 >= ret5_min_pct) & (ret20 >= ret20_min_pct) & (ret20 <= ret20_max_pct)
    volatility_ok = (atr_pct >= atr_min_pct) & (atr_pct <= atr_max_pct)
    candle_ok = (close_pos >= min_close_pos) & (body_ratio >= min_body_ratio)
    rsi_ok = (rsi >= rsi_min) & (rsi <= rsi_max)
    liquidity_ok = volume_ratio >= volume_ratio_min

    long_signal = (
        low_probe
        & reclaimed
        & trend_not_bearish
        & not_falling_knife
        & volatility_ok
        & candle_ok
        & rsi_ok
        & liquidity_ok
    )

    result = pd.Series(0.0, index=df.index, dtype="float64")
    result[long_signal] = signal_value

    if cooldown_bars > 0:
        cleaned = result.copy()
        last_idx = -10**9
        for i, value in enumerate(cleaned.to_numpy()):
            if value > 0:
                if i - last_idx < cooldown_bars:
                    cleaned.iloc[i] = 0.0
                else:
                    last_idx = i
        result = cleaned

    return result.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype("float64")
