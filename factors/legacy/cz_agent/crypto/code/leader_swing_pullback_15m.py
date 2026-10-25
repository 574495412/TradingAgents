"""Leader-style 15m swing pullback factor.

This candidate is derived from external main-strategy leader samples.  Their
entries often happen near the 20-bar range edge after a controlled pullback or
bounce, while the existing production signal remains neutral.  The factor emits
only a pulse after price rejects the range edge and starts to reclaim.
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
    atr_max_pct = float(params.get("atr_max_pct", 1.2))
    edge_zone_pct = float(params.get("edge_zone_pct", 1.6))
    min_reclaim_body = float(params.get("min_reclaim_body", 0.35))
    ret20_abs_max_pct = float(params.get("ret20_abs_max_pct", 4.5))
    rsi_period = int(params.get("rsi_period", 14))
    rsi_long_max = float(params.get("rsi_long_max", 58))
    rsi_short_min = float(params.get("rsi_short_min", 42))
    cooldown_bars = int(params.get("signal_cooldown_bars", 10))
    signal_value = float(params.get("signal_value", 0.7))

    close = df["close"].astype("float64")
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")
    open_ = df["open"].astype("float64")
    if len(df) < max(ema_slow_period, range_lookback, atr_period) + 10:
        return pd.Series(0.0, index=df.index, dtype="float64")

    ema_fast = close.ewm(span=ema_fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=ema_slow_period, adjust=False).mean()
    ema_slope = (ema_fast - ema_fast.shift(8)) / (ema_fast.shift(8) + 1e-10) * 100.0
    trend_up = (ema_fast >= ema_slow * 0.995) & (ema_slope > -0.08)
    trend_down = (ema_fast <= ema_slow * 1.005) & (ema_slope < 0.08)

    prev_high = high.shift(1).rolling(range_lookback, min_periods=range_lookback).max()
    prev_low = low.shift(1).rolling(range_lookback, min_periods=range_lookback).min()
    from_high = (close / (prev_high + 1e-10) - 1.0) * 100.0
    from_low = (close / (prev_low + 1e-10) - 1.0) * 100.0
    ret5 = close.pct_change(5) * 100.0
    ret20 = close.pct_change(20) * 100.0

    tr = _true_range(high, low, close)
    atr_pct = tr.rolling(atr_period, min_periods=atr_period).mean() / (close + 1e-10) * 100.0
    rsi = _rsi(close, rsi_period)
    body_ratio = (close - open_).abs() / ((high - low).abs() + 1e-10)
    close_pos = (close - low) / ((high - low) + 1e-10)

    near_low_edge = (from_low >= -0.20) & (from_low <= edge_zone_pct)
    near_high_edge = (from_high <= 0.20) & (from_high >= -edge_zone_pct)
    calm_enough = (atr_pct <= atr_max_pct) & (ret20.abs() <= ret20_abs_max_pct)

    reclaim_long = (
        near_low_edge
        & calm_enough
        & trend_up
        & (ret5 > -0.25)
        & (close > open_)
        & (body_ratio >= min_reclaim_body)
        & (close_pos >= 0.55)
        & (rsi <= rsi_long_max)
    )
    reject_short = (
        near_high_edge
        & calm_enough
        & trend_down
        & (ret5 < 0.25)
        & (close < open_)
        & (body_ratio >= min_reclaim_body)
        & (close_pos <= 0.45)
        & (rsi >= rsi_short_min)
    )

    result = pd.Series(0.0, index=df.index, dtype="float64")
    result[reclaim_long] = signal_value
    result[reject_short] = -signal_value

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
