"""Volume climax reversal factor.

This candidate looks for short-lived exhaustion after an extended move. It only
fires when price is overextended, volume is extreme, and the candle rejects the
move with a strong wick/body reversal.
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


def calculate_factor(df: pd.DataFrame, parameters: dict | None = None) -> pd.Series:
    params = parameters or {}
    ret_lookback = int(params.get("ret_lookback", 16))
    min_abs_ret_pct = float(params.get("min_abs_ret_pct", 3.2)) / 100.0
    volume_ma_period = int(params.get("volume_ma_period", 48))
    volume_z_min = float(params.get("volume_z_min", 2.2))
    rsi_period = int(params.get("rsi_period", 14))
    rsi_long_max = float(params.get("rsi_long_max", 26))
    rsi_short_min = float(params.get("rsi_short_min", 74))
    wick_ratio_min = float(params.get("wick_ratio_min", 0.42))
    close_reclaim_pct = float(params.get("close_reclaim_pct", 0.25))
    cooldown_bars = int(params.get("signal_cooldown_bars", 12))
    signal_value = float(params.get("signal_value", 0.7))

    close = df["close"].astype("float64")
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")
    open_ = df["open"].astype("float64")
    volume = df["volume"].astype("float64")
    if len(df) < max(ret_lookback, volume_ma_period, rsi_period) + 10:
        return pd.Series(0.0, index=df.index, dtype="float64")

    ret = close.pct_change(ret_lookback)
    vol_mean = volume.rolling(volume_ma_period, min_periods=volume_ma_period // 2).mean()
    vol_std = volume.rolling(volume_ma_period, min_periods=volume_ma_period // 2).std()
    vol_z = (volume - vol_mean) / (vol_std + 1e-10)
    extreme_volume = vol_z >= volume_z_min

    rsi = _rsi(close, rsi_period)
    candle_range = high - low + 1e-10
    lower_wick = (np.minimum(open_, close) - low) / candle_range
    upper_wick = (high - np.maximum(open_, close)) / candle_range
    close_location = (close - low) / candle_range

    capitulation_down = ret <= -min_abs_ret_pct
    blowoff_up = ret >= min_abs_ret_pct
    bullish_reclaim = (close > open_) & (lower_wick >= wick_ratio_min) & (close_location >= close_reclaim_pct)
    bearish_reject = (close < open_) & (upper_wick >= wick_ratio_min) & (close_location <= 1.0 - close_reclaim_pct)

    result = pd.Series(0.0, index=df.index, dtype="float64")
    result[capitulation_down & extreme_volume & bullish_reclaim & (rsi <= rsi_long_max)] = signal_value
    result[blowoff_up & extreme_volume & bearish_reject & (rsi >= rsi_short_min)] = -signal_value

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
