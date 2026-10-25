"""Bollinger chop mean-reversion factor.

This candidate is designed for low-trend, range-bound regimes. It fades
temporary band extensions only when ADX and EMA spread indicate chop.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100.0 - 100.0 / (1.0 + rs)


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=period).mean()
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=close.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=close.index)
    plus_di = 100.0 * plus_dm.rolling(period, min_periods=period).mean() / (atr + 1e-10)
    minus_di = 100.0 * minus_dm.rolling(period, min_periods=period).mean() / (atr + 1e-10)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    return dx.rolling(period, min_periods=period).mean()


def calculate_factor(df: pd.DataFrame, parameters: dict | None = None) -> pd.Series:
    params = parameters or {}
    bb_period = int(params.get("bb_period", 40))
    bb_std = float(params.get("bb_std", 2.15))
    rsi_period = int(params.get("rsi_period", 14))
    rsi_long_max = float(params.get("rsi_long_max", 32))
    rsi_short_min = float(params.get("rsi_short_min", 68))
    adx_period = int(params.get("adx_period", 14))
    adx_max = float(params.get("adx_max", 22))
    ema_fast_period = int(params.get("ema_fast", 20))
    ema_slow_period = int(params.get("ema_slow", 80))
    ema_spread_max_pct = float(params.get("ema_spread_max_pct", 0.9)) / 100.0
    wick_ratio_min = float(params.get("wick_ratio_min", 0.35))
    cooldown_bars = int(params.get("signal_cooldown_bars", 8))
    signal_value = float(params.get("signal_value", 0.7))

    close = df["close"].astype("float64")
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")
    open_ = df["open"].astype("float64")
    if len(df) < max(bb_period, ema_slow_period, adx_period * 2) + 5:
        return pd.Series(0.0, index=df.index, dtype="float64")

    mid = close.rolling(bb_period, min_periods=bb_period).mean()
    std = close.rolling(bb_period, min_periods=bb_period).std()
    upper = mid + bb_std * std
    lower = mid - bb_std * std
    band_width = (upper - lower) / (mid.abs() + 1e-10)
    width_ok = band_width < band_width.rolling(160, min_periods=80).quantile(0.75)

    rsi = _rsi(close, rsi_period)
    adx = _adx(high, low, close, adx_period)
    ema_fast = close.ewm(span=ema_fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=ema_slow_period, adjust=False).mean()
    ema_spread = (ema_fast - ema_slow).abs() / (close + 1e-10)
    chop = (adx <= adx_max) & (ema_spread <= ema_spread_max_pct) & width_ok

    candle_range = high - low + 1e-10
    lower_wick = (np.minimum(open_, close) - low) / candle_range
    upper_wick = (high - np.maximum(open_, close)) / candle_range
    bullish_reject = (low < lower) & (close > lower) & (close > open_) & (lower_wick >= wick_ratio_min)
    bearish_reject = (high > upper) & (close < upper) & (close < open_) & (upper_wick >= wick_ratio_min)

    result = pd.Series(0.0, index=df.index, dtype="float64")
    result[chop & bullish_reject & (rsi <= rsi_long_max)] = signal_value
    result[chop & bearish_reject & (rsi >= rsi_short_min)] = -signal_value

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
