"""Leader-style rebound reclaim / flip factor for ETH timing research.

This research candidate targets the missing part of the ETH lead-trader timing
audit: after a short cascade, a controlled rebound can produce either a long
reclaim entry or a later bearish flip once that rebound fails.  Only closed-bar
OHLCV features are used.
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
    atr_period = int(params.get("atr_period", 14))
    atr_min_pct = float(params.get("atr_min_pct", 0.18))
    atr_max_pct = float(params.get("atr_max_pct", 1.10))
    volume_window = int(params.get("volume_window", 30))
    volume_ratio_min = float(params.get("volume_ratio_min", 0.55))
    body_min = float(params.get("body_min", 0.34))
    long_ret48_min_pct = float(params.get("long_ret48_min_pct", 2.40))
    long_ret96_min_pct = float(params.get("long_ret96_min_pct", 1.00))
    long_ret192_max_pct = float(params.get("long_ret192_max_pct", 1.00))
    long_ema_fast_dev_min_pct = float(params.get("long_ema_fast_dev_min_pct", -0.75))
    long_ema_fast_dev_max_pct = float(params.get("long_ema_fast_dev_max_pct", 0.35))
    long_ema_slow_dev_min_pct = float(params.get("long_ema_slow_dev_min_pct", 0.55))
    long_from_high48_max_pct = float(params.get("long_from_high48_max_pct", -1.15))
    long_rsi_min = float(params.get("long_rsi_min", 42.0))
    long_rsi_max = float(params.get("long_rsi_max", 58.0))
    long_close_pos_min = float(params.get("long_close_pos_min", 0.55))
    short_ret48_min_pct = float(params.get("short_ret48_min_pct", 1.50))
    short_ret96_min_pct = float(params.get("short_ret96_min_pct", 2.50))
    short_ret192_min_pct = float(params.get("short_ret192_min_pct", 3.00))
    short_from_low96_min_pct = float(params.get("short_from_low96_min_pct", 4.00))
    short_ema_fast_dev_min_pct = float(params.get("short_ema_fast_dev_min_pct", -0.20))
    short_rsi_min = float(params.get("short_rsi_min", 50.0))
    short_rsi_max = float(params.get("short_rsi_max", 68.0))
    short_close_pos_max = float(params.get("short_close_pos_max", 0.45))
    signal_cooldown_bars = int(params.get("signal_cooldown_bars", 10))
    signal_value = float(params.get("signal_value", 0.7))

    close = df["close"].astype("float64")
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")
    open_ = df["open"].astype("float64")
    volume = df.get("volume", pd.Series(0.0, index=df.index)).astype("float64")
    min_bars = max(ema_slow_period, atr_period, volume_window, 192) + 10
    if len(df) < min_bars:
        return pd.Series(0.0, index=df.index, dtype="float64")

    ema_fast = close.ewm(span=ema_fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=ema_slow_period, adjust=False).mean()
    ema_fast_dev = (close / (ema_fast + 1e-10) - 1.0) * 100.0
    ema_slow_dev = (close / (ema_slow + 1e-10) - 1.0) * 100.0
    ret5 = close.pct_change(5) * 100.0
    ret48 = close.pct_change(48) * 100.0
    ret96 = close.pct_change(96) * 100.0
    ret192 = close.pct_change(192) * 100.0

    range_ = high - low + 1e-10
    body_ratio = (close - open_).abs() / range_
    close_pos = (close - low) / range_
    atr_pct = _true_range(high, low, close).rolling(atr_period, min_periods=atr_period).mean() / (close + 1e-10) * 100.0
    volume_ratio = volume / (volume.shift(1).rolling(volume_window, min_periods=10).mean() + 1e-10)
    rsi = _rsi(close, int(params.get("rsi_period", 14)))

    from_high48 = (close / (high.shift(1).rolling(48, min_periods=48).max() + 1e-10) - 1.0) * 100.0
    from_low96 = (close / (low.shift(1).rolling(96, min_periods=96).min() + 1e-10) - 1.0) * 100.0
    volatility_ok = (atr_pct >= atr_min_pct) & (atr_pct <= atr_max_pct)
    liquidity_ok = volume_ratio >= volume_ratio_min

    long_reclaim = (
        volatility_ok
        & liquidity_ok
        & (ret48 >= long_ret48_min_pct)
        & (ret96 >= long_ret96_min_pct)
        & (ret192 <= long_ret192_max_pct)
        & (ema_fast_dev >= long_ema_fast_dev_min_pct)
        & (ema_fast_dev <= long_ema_fast_dev_max_pct)
        & (ema_slow_dev >= long_ema_slow_dev_min_pct)
        & (from_high48 <= long_from_high48_max_pct)
        & (body_ratio >= body_min)
        & (close_pos >= long_close_pos_min)
        & (rsi >= long_rsi_min)
        & (rsi <= long_rsi_max)
        & (ret5 >= -1.20)
    )
    rebound_flip_short = (
        volatility_ok
        & liquidity_ok
        & (ret48 >= short_ret48_min_pct)
        & (ret96 >= short_ret96_min_pct)
        & (ret192 >= short_ret192_min_pct)
        & (from_low96 >= short_from_low96_min_pct)
        & (ema_fast_dev >= short_ema_fast_dev_min_pct)
        & (body_ratio >= body_min)
        & (close_pos <= short_close_pos_max)
        & (rsi >= short_rsi_min)
        & (rsi <= short_rsi_max)
        & (ret5 <= 0.35)
    )

    result = pd.Series(0.0, index=df.index, dtype="float64")
    result[long_reclaim] = signal_value
    result[rebound_flip_short] = -signal_value

    if signal_cooldown_bars > 0:
        cleaned = result.copy()
        last_long = -10**9
        last_short = -10**9
        for i, value in enumerate(cleaned.to_numpy()):
            if value > 0:
                if i - last_long < signal_cooldown_bars:
                    cleaned.iloc[i] = 0.0
                else:
                    last_long = i
            elif value < 0:
                if i - last_short < signal_cooldown_bars:
                    cleaned.iloc[i] = 0.0
                else:
                    last_short = i
        result = cleaned

    return result.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype("float64")
