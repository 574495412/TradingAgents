"""Leader-style 15m trend continuation pulse factor.

This research candidate targets a structure seen in high-return main-strategy
leader labels: after a multi-hour directional move is accepted, price continues
in the same direction without waiting for a large 4h pullback.  It uses only
public OHLCV features available at the bar close.
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
    ema_fast_period = int(params.get("ema_fast", 20))
    ema_slow_period = int(params.get("ema_slow", 96))
    trend_slope_bars = int(params.get("trend_slope_bars", 8))
    range_lookback = int(params.get("range_lookback", 20))
    atr_period = int(params.get("atr_period", 14))
    atr_min_pct = float(params.get("atr_min_pct", 0.12))
    atr_max_pct = float(params.get("atr_max_pct", 1.35))
    volume_window = int(params.get("volume_window", 30))
    volume_ratio_min = float(params.get("volume_ratio_min", 0.65))
    min_body_ratio = float(params.get("min_body_ratio", 0.20))
    min_close_pos_long = float(params.get("min_close_pos_long", 0.54))
    max_close_pos_short = float(params.get("max_close_pos_short", 0.46))
    long_ret5_min_pct = float(params.get("long_ret5_min_pct", -0.20))
    long_ret20_min_pct = float(params.get("long_ret20_min_pct", -0.60))
    long_ret20_max_pct = float(params.get("long_ret20_max_pct", 5.60))
    short_ret5_max_pct = float(params.get("short_ret5_max_pct", 0.20))
    short_ret20_max_pct = float(params.get("short_ret20_max_pct", -0.60))
    short_ret20_min_pct = float(params.get("short_ret20_min_pct", -7.20))
    short_ret192_min_pct = float(params.get("short_ret192_min_pct", -7.00))
    edge_acceptance_pct = float(params.get("edge_acceptance_pct", 0.65))
    rsi_period = int(params.get("rsi_period", 14))
    rsi_long_max = float(params.get("rsi_long_max", 74.0))
    rsi_short_min = float(params.get("rsi_short_min", 26.0))
    cooldown_bars = int(params.get("signal_cooldown_bars", 10))
    signal_value = float(params.get("signal_value", 0.7))

    close = df["close"].astype("float64")
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")
    open_ = df["open"].astype("float64")
    volume = df.get("volume", pd.Series(0.0, index=df.index)).astype("float64")
    min_bars = max(ema_slow_period, range_lookback, atr_period, rsi_period, volume_window) + trend_slope_bars + 10
    if len(df) < min_bars:
        return pd.Series(0.0, index=df.index, dtype="float64")

    ema_fast = close.ewm(span=ema_fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=ema_slow_period, adjust=False).mean()
    ema_slope = (ema_fast / (ema_fast.shift(trend_slope_bars) + 1e-10) - 1.0) * 100.0
    ret5 = close.pct_change(5) * 100.0
    ret20 = close.pct_change(20) * 100.0
    ret60 = close.pct_change(60) * 100.0
    ret192 = close.pct_change(192) * 100.0

    tr = _true_range(high, low, close)
    atr_pct = tr.rolling(atr_period, min_periods=atr_period).mean() / (close + 1e-10) * 100.0
    volume_ratio = volume / (volume.shift(1).rolling(volume_window, min_periods=10).mean() + 1e-10)
    body_ratio = (close - open_).abs() / ((high - low).abs() + 1e-10)
    close_pos = (close - low) / ((high - low) + 1e-10)
    rsi = _rsi(close, rsi_period)

    prev_high = high.shift(1).rolling(range_lookback, min_periods=range_lookback).max()
    prev_low = low.shift(1).rolling(range_lookback, min_periods=range_lookback).min()
    accepted_high = close >= prev_high * (1.0 - edge_acceptance_pct / 100.0)
    accepted_low = close <= prev_low * (1.0 + edge_acceptance_pct / 100.0)

    volatility_ok = (atr_pct >= atr_min_pct) & (atr_pct <= atr_max_pct)
    liquidity_ok = volume_ratio >= volume_ratio_min
    candle_long = ((close > open_) | (close_pos >= min_close_pos_long)) & (body_ratio >= min_body_ratio)
    candle_short = ((close < open_) | (close_pos <= max_close_pos_short)) & (body_ratio >= min_body_ratio)

    trend_up = ((ema_fast >= ema_slow * 0.996) & (ema_slope >= -0.04)) | (ret60 >= 1.20)
    trend_down = ((ema_fast <= ema_slow * 1.004) & (ema_slope <= 0.04)) | (ret60 <= -1.20)

    long_acceptance = (
        (ret5 >= long_ret5_min_pct)
        & (ret20 >= long_ret20_min_pct)
        & (ret20 <= long_ret20_max_pct)
        & (accepted_high | (close >= ema_fast * 0.996))
    )
    short_acceptance = (
        (ret5 <= short_ret5_max_pct)
        & (ret20 <= short_ret20_max_pct)
        & (ret20 >= short_ret20_min_pct)
        & (ret192 >= short_ret192_min_pct)
        & (accepted_low | (close <= ema_fast * 1.004))
    )

    long_signal = volatility_ok & liquidity_ok & trend_up & long_acceptance & candle_long & (rsi <= rsi_long_max)
    short_signal = volatility_ok & liquidity_ok & trend_down & short_acceptance & candle_short & (rsi >= rsi_short_min)

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
