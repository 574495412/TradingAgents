"""TradFi perpetual 15m range-expansion factor.

The crypto factor set is too conservative for Binance TradFi perpetuals. This
factor targets a simpler equity-style structure: a fresh 15m range break with
volume participation, directional candle quality, and moderate trend support.
It emits pulses only, so the live engine cannot repeatedly reuse an old signal.
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
    breakout_lookback = int(params.get("breakout_lookback", 28))
    atr_period = int(params.get("atr_period", 20))
    atr_min_pct = float(params.get("atr_min_pct", 0.08)) / 100.0
    atr_max_pct = float(params.get("atr_max_pct", 2.80)) / 100.0
    range_expand_ratio = float(params.get("range_expand_ratio", 0.95))
    volume_window = int(params.get("volume_window", 48))
    volume_ratio_min = float(params.get("volume_ratio_min", 1.05))
    ema_fast_period = int(params.get("ema_fast", 24))
    ema_slow_period = int(params.get("ema_slow", 96))
    trend_slope_bars = int(params.get("trend_slope_bars", 8))
    trend_slope_min = float(params.get("trend_slope_min", 0.02)) / 100.0
    ema_spread_min_long = float(params.get("ema_spread_min_long_pct", 0.45)) / 100.0
    ema_spread_min_short = float(params.get("ema_spread_min_short_pct", 0.25)) / 100.0
    min_body_ratio = float(params.get("min_body_ratio", 0.30))
    min_close_pos_long = float(params.get("min_close_pos_long", 0.58))
    max_close_pos_short = float(params.get("max_close_pos_short", 0.42))
    max_extension_atr = float(params.get("max_extension_atr", 1.25))
    max_ret5_pct = float(params.get("max_ret5_pct", 4.80)) / 100.0
    max_ret20_pct = float(params.get("max_ret20_pct", 10.50)) / 100.0
    signal_cooldown_bars = int(params.get("signal_cooldown_bars", 10))
    signal_value = float(params.get("signal_value", 0.68))

    close = df["close"].astype("float64")
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")
    open_ = df["open"].astype("float64")
    volume = df["volume"].astype("float64")

    min_bars = max(breakout_lookback, volume_window, ema_slow_period, atr_period) + trend_slope_bars + 5
    if len(df) < min_bars:
        return pd.Series(0.0, index=df.index, dtype="float64")

    tr = _true_range(high, low, close)
    atr = tr.rolling(atr_period, min_periods=atr_period).mean()
    atr_pct = atr / (close.abs() + 1e-10)
    atr_ok = atr_pct.between(atr_min_pct, atr_max_pct)
    range_expanding = tr >= atr * range_expand_ratio

    prev_high = high.shift(1).rolling(breakout_lookback, min_periods=breakout_lookback).max()
    prev_low = low.shift(1).rolling(breakout_lookback, min_periods=breakout_lookback).min()
    break_up = close > prev_high
    break_down = close < prev_low
    extension_up_atr = (close - prev_high) / (atr + 1e-10)
    extension_down_atr = (prev_low - close) / (atr + 1e-10)
    extension_ok_up = extension_up_atr.between(0, max_extension_atr)
    extension_ok_down = extension_down_atr.between(0, max_extension_atr)

    vol_base = volume.shift(1).rolling(volume_window, min_periods=max(10, volume_window // 3)).median()
    vol_ratio = volume / (vol_base + 1e-10)
    volume_ok = vol_ratio >= volume_ratio_min

    ema_fast = close.ewm(span=ema_fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=ema_slow_period, adjust=False).mean()
    ema_slope = (ema_fast - ema_fast.shift(trend_slope_bars)) / (ema_fast.shift(trend_slope_bars).abs() + 1e-10)
    ema_spread = (ema_fast - ema_slow) / (close.abs() + 1e-10)
    trend_up = (ema_fast >= ema_slow) & (ema_slope >= trend_slope_min) & (ema_spread >= ema_spread_min_long)
    trend_down = (ema_fast <= ema_slow) & (ema_slope <= -trend_slope_min) & (ema_spread <= -ema_spread_min_short)

    candle_range = (high - low).abs() + 1e-10
    body = close - open_
    body_ratio = body.abs() / candle_range
    close_pos = (close - low) / candle_range
    bull_body = (body > 0) & (body_ratio >= min_body_ratio) & (close_pos >= min_close_pos_long)
    bear_body = (body < 0) & (body_ratio >= min_body_ratio) & (close_pos <= max_close_pos_short)

    ret5 = close.pct_change(5)
    ret20 = close.pct_change(20)
    not_chasing_long = (ret5 <= max_ret5_pct) & (ret20 <= max_ret20_pct)
    not_chasing_short = (ret5 >= -max_ret5_pct) & (ret20 >= -max_ret20_pct)

    result = pd.Series(0.0, index=df.index, dtype="float64")
    long_signal = (
        break_up
        & extension_ok_up
        & range_expanding
        & volume_ok
        & atr_ok
        & trend_up
        & bull_body
        & not_chasing_long
    )
    short_signal = (
        break_down
        & extension_ok_down
        & range_expanding
        & volume_ok
        & atr_ok
        & trend_down
        & bear_body
        & not_chasing_short
    )
    result[long_signal] = signal_value
    result[short_signal] = -signal_value

    if signal_cooldown_bars > 0:
        cleaned = result.copy()
        last_signal = -10**9
        for i, value in enumerate(cleaned.to_numpy()):
            if value == 0:
                continue
            if i - last_signal < signal_cooldown_bars:
                cleaned.iloc[i] = 0.0
            else:
                last_signal = i
        result = cleaned

    return result.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype("float64")
