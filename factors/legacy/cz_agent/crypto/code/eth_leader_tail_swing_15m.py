"""ETH leader-tail swing factor for main-trend timing research.

This research candidate targets the right-tail trades observed in ETH main
copytrading leaders.  It uses only closed-bar OHLCV state and splits the setup
into three interpretable structures:

1. accepted panic breakdown continuation shorts;
2. rebound-failure shorts after a multi-hour squeeze upward;
3. deep-cascade reclaim longs after a large drawdown.
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


def _apply_directional_cooldown(values: pd.Series, cooldown_bars: int) -> pd.Series:
    if cooldown_bars <= 0:
        return values
    cleaned = values.copy()
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
    return cleaned


def calculate_factor(df: pd.DataFrame, parameters: dict | None = None) -> pd.Series:
    params = parameters or {}
    ema_fast_period = int(params.get("ema_fast", 20))
    ema_mid_period = int(params.get("ema_mid", 96))
    ema_slow_period = int(params.get("ema_slow", 192))
    atr_period = int(params.get("atr_period", 14))
    atr_min_pct = float(params.get("atr_min_pct", 0.13))
    atr_max_pct = float(params.get("atr_max_pct", 1.85))
    volume_window = int(params.get("volume_window", 30))
    volume_long_window = int(params.get("volume_long_window", 160))
    volume_ratio_min = float(params.get("volume_ratio_min", 0.55))
    volume_long_ratio_min = float(params.get("volume_long_ratio_min", 0.70))
    body_min = float(params.get("body_min", 0.18))
    signal_cooldown_bars = int(params.get("signal_cooldown_bars", 12))
    signal_value = float(params.get("signal_value", 0.78))
    weak_signal_value = float(params.get("weak_signal_value", 0.35))

    panic_ret20_max_pct = float(params.get("panic_ret20_max_pct", -0.45))
    panic_ret80_max_pct = float(params.get("panic_ret80_max_pct", -1.90))
    panic_from_high80_max_pct = float(params.get("panic_from_high80_max_pct", -1.30))
    panic_close_pos_max = float(params.get("panic_close_pos_max", 0.56))
    panic_ema_mid_dev_max_pct = float(params.get("panic_ema_mid_dev_max_pct", 0.70))
    panic_break_edge_pct = float(params.get("panic_break_edge_pct", 0.45))
    panic_ret320_min_pct = float(params.get("panic_ret320_min_pct", -14.0))
    panic_ret320_max_pct = float(params.get("panic_ret320_max_pct", 1.50))

    failed_ret80_min_pct = float(params.get("failed_ret80_min_pct", 2.10))
    failed_ret160_min_pct = float(params.get("failed_ret160_min_pct", 5.50))
    failed_from_low160_min_pct = float(params.get("failed_from_low160_min_pct", 3.80))
    failed_from_low160_max_pct = float(params.get("failed_from_low160_max_pct", 8.20))
    failed_from_high80_max_pct = float(params.get("failed_from_high80_max_pct", -2.00))
    failed_ema_mid_dev_max_pct = float(params.get("failed_ema_mid_dev_max_pct", 1.60))
    failed_close_pos_max = float(params.get("failed_close_pos_max", 0.34))
    failed_ret5_max_pct = float(params.get("failed_ret5_max_pct", 0.50))
    failed_rsi_min = float(params.get("failed_rsi_min", 42.0))
    failed_rsi_max = float(params.get("failed_rsi_max", 64.0))

    reclaim_ret320_max_pct = float(params.get("reclaim_ret320_max_pct", -3.80))
    reclaim_ret40_min_pct = float(params.get("reclaim_ret40_min_pct", 1.15))
    reclaim_ret80_min_pct = float(params.get("reclaim_ret80_min_pct", 2.50))
    reclaim_ret160_max_pct = float(params.get("reclaim_ret160_max_pct", 4.80))
    reclaim_from_low80_min_pct = float(params.get("reclaim_from_low80_min_pct", 3.20))
    reclaim_from_high80_max_pct = float(params.get("reclaim_from_high80_max_pct", -0.65))
    reclaim_close_pos_min = float(params.get("reclaim_close_pos_min", 0.58))
    reclaim_ema_fast_dev_min_pct = float(params.get("reclaim_ema_fast_dev_min_pct", -0.75))
    reclaim_rsi_min = float(params.get("reclaim_rsi_min", 42.0))
    reclaim_rsi_max = float(params.get("reclaim_rsi_max", 70.0))
    high_short_ret80_max_pct = float(params.get("high_short_ret80_max_pct", -2.0))
    high_short_ret80_close_pos_max = float(params.get("high_short_ret80_close_pos_max", 0.35))
    high_short_ret160_max_pct = float(params.get("high_short_ret160_max_pct", -4.5))
    high_short_ret160_close_pos_max = float(params.get("high_short_ret160_close_pos_max", 0.32))
    high_short_vol_long_ratio_min = float(params.get("high_short_vol_long_ratio_min", 1.5))
    high_short_vol_close_pos_max = float(params.get("high_short_vol_close_pos_max", 0.35))
    high_rebound_ret160_min_pct = float(params.get("high_rebound_ret160_min_pct", 5.5))
    high_rebound_from_high80_max_pct = float(params.get("high_rebound_from_high80_max_pct", -2.0))
    high_rebound_close_pos_max = float(params.get("high_rebound_close_pos_max", 0.30))
    high_rebound_body_min = float(params.get("high_rebound_body_min", 0.22))
    high_long_ret320_max_pct = float(params.get("high_long_ret320_max_pct", -3.8))
    high_long_ret80_min_pct = float(params.get("high_long_ret80_min_pct", 2.0))
    high_long_ret160_max_pct = float(params.get("high_long_ret160_max_pct", 5.2))
    high_long_close_pos_min = float(params.get("high_long_close_pos_min", 0.55))

    close = df["close"].astype("float64")
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")
    open_ = df["open"].astype("float64")
    volume = df.get("volume", pd.Series(0.0, index=df.index)).astype("float64")
    min_bars = max(ema_slow_period, atr_period, volume_long_window, 320) + 10
    if len(df) < min_bars:
        return pd.Series(0.0, index=df.index, dtype="float64")

    ema_fast = close.ewm(span=ema_fast_period, adjust=False).mean()
    ema_mid = close.ewm(span=ema_mid_period, adjust=False).mean()
    ema_slow = close.ewm(span=ema_slow_period, adjust=False).mean()
    ema_fast_dev = (close / (ema_fast + 1e-10) - 1.0) * 100.0
    ema_mid_dev = (close / (ema_mid + 1e-10) - 1.0) * 100.0
    ema_slow_dev = (close / (ema_slow + 1e-10) - 1.0) * 100.0

    ret5 = close.pct_change(5) * 100.0
    ret20 = close.pct_change(20) * 100.0
    ret40 = close.pct_change(40) * 100.0
    ret80 = close.pct_change(80) * 100.0
    ret160 = close.pct_change(160) * 100.0
    ret320 = close.pct_change(320) * 100.0

    range_ = high - low + 1e-10
    body_ratio = (close - open_).abs() / range_
    close_pos = (close - low) / range_
    red = close < open_
    green = close > open_

    atr_pct = _true_range(high, low, close).rolling(atr_period, min_periods=atr_period).mean() / (close + 1e-10) * 100.0
    volume_ratio = volume / (volume.shift(1).rolling(volume_window, min_periods=10).median() + 1e-10)
    volume_long_ratio = volume / (volume.shift(1).rolling(volume_long_window, min_periods=40).median() + 1e-10)
    rsi = _rsi(close, int(params.get("rsi_period", 14)))

    prev_low20 = low.shift(1).rolling(20, min_periods=20).min()
    prev_low80 = low.shift(1).rolling(80, min_periods=80).min()
    prev_high80 = high.shift(1).rolling(80, min_periods=80).max()
    prev_low160 = low.shift(1).rolling(160, min_periods=160).min()

    from_high80 = (close / (prev_high80 + 1e-10) - 1.0) * 100.0
    from_low80 = (close / (prev_low80 + 1e-10) - 1.0) * 100.0
    from_low160 = (close / (prev_low160 + 1e-10) - 1.0) * 100.0

    volatility_ok = (atr_pct >= atr_min_pct) & (atr_pct <= atr_max_pct)
    liquidity_ok = (volume_ratio >= volume_ratio_min) | (volume_long_ratio >= volume_long_ratio_min)
    candle_short = ((red & (body_ratio >= body_min)) | (close_pos <= 0.18))
    candle_long = ((green & (body_ratio >= body_min)) | (close_pos >= 0.82))

    accepted_break = (close <= prev_low20 * (1.0 + panic_break_edge_pct / 100.0)) | (
        close <= prev_low80 * (1.0 + panic_break_edge_pct / 100.0)
    )
    panic_breakdown_short = (
        volatility_ok
        & liquidity_ok
        & ((ret20 <= panic_ret20_max_pct) | (ret40 <= panic_ret80_max_pct * 0.72))
        & ((ret80 <= panic_ret80_max_pct) | (from_high80 <= panic_from_high80_max_pct) | accepted_break)
        & (ret320 >= panic_ret320_min_pct)
        & (ret320 <= panic_ret320_max_pct)
        & (ema_mid_dev <= panic_ema_mid_dev_max_pct)
        & (close_pos <= panic_close_pos_max)
        & candle_short
    )

    rebound_failure_short = (
        volatility_ok
        & liquidity_ok
        & ((ret80 >= failed_ret80_min_pct) | (ret160 >= failed_ret160_min_pct))
        & (ret160 >= failed_ret160_min_pct)
        & (from_low160 >= failed_from_low160_min_pct)
        & (from_low160 <= failed_from_low160_max_pct)
        & (from_high80 <= failed_from_high80_max_pct)
        & (ema_mid_dev <= failed_ema_mid_dev_max_pct)
        & (ret5 <= failed_ret5_max_pct)
        & (close_pos <= failed_close_pos_max)
        & (body_ratio >= body_min)
        & red
        & (rsi >= failed_rsi_min)
        & (rsi <= failed_rsi_max)
        & (close >= ema_mid * 0.982)
    )

    late_cascade_short = (
        volatility_ok
        & liquidity_ok
        & (ret160 <= -3.80)
        & (ret20 <= -0.35)
        & (ret320 >= panic_ret320_min_pct)
        & (ret320 <= panic_ret320_max_pct)
        & (close_pos <= panic_close_pos_max)
        & candle_short
        & ((ema_fast <= ema_mid * 1.006) | (ema_slow_dev <= 0.20))
    )

    deep_reclaim_long = (
        volatility_ok
        & liquidity_ok
        & ((ret320 <= reclaim_ret320_max_pct) | (ret160 <= -3.40))
        & (ret80 >= reclaim_ret80_min_pct)
        & (ret160 <= reclaim_ret160_max_pct)
        & (ret40 >= reclaim_ret40_min_pct)
        & (from_low80 >= reclaim_from_low80_min_pct)
        & (from_high80 <= reclaim_from_high80_max_pct)
        & (ema_fast_dev >= reclaim_ema_fast_dev_min_pct)
        & (close_pos >= reclaim_close_pos_min)
        & candle_long
        & (rsi >= reclaim_rsi_min)
        & (rsi <= reclaim_rsi_max)
    )

    short_signal = panic_breakdown_short | rebound_failure_short | late_cascade_short
    long_signal = deep_reclaim_long
    high_confidence_short = (
        ((ret80 <= high_short_ret80_max_pct) & (close_pos <= high_short_ret80_close_pos_max))
        | ((ret160 <= high_short_ret160_max_pct) & (close_pos <= high_short_ret160_close_pos_max))
        | ((volume_long_ratio >= high_short_vol_long_ratio_min) & (close_pos <= high_short_vol_close_pos_max))
        | (
            (ret160 >= high_rebound_ret160_min_pct)
            & (from_high80 <= high_rebound_from_high80_max_pct)
            & (close_pos <= high_rebound_close_pos_max)
            & (body_ratio >= high_rebound_body_min)
            & red
        )
    )
    high_confidence_long = (
        (ret320 <= high_long_ret320_max_pct)
        & (ret80 >= high_long_ret80_min_pct)
        & (ret160 <= high_long_ret160_max_pct)
        & (close_pos >= high_long_close_pos_min)
        & green
    )

    result = pd.Series(0.0, index=df.index, dtype="float64")
    result[short_signal] = -weak_signal_value
    result[long_signal] = weak_signal_value
    result[short_signal & high_confidence_short] = -signal_value
    result[long_signal & high_confidence_long] = signal_value
    result[short_signal & long_signal] = 0.0
    result = _apply_directional_cooldown(result, signal_cooldown_bars)
    return result.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype("float64")
