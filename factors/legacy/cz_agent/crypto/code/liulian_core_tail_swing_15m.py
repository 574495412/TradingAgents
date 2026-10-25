"""Liulian core BTC/ETH tail-swing factor for leader replication research.

The factor targets the core BTC/ETH right-tail trades observed in a high-ROI
main-strategy leader.  It intentionally uses only closed-bar OHLCV features:

1. downside acceptance after a multi-hour rollover;
2. rebound/retest failure below recent highs;
3. conservative reclaim longs after a controlled dip.

Leader fills are labels for validation only and are not read by this factor.
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
    ema_mid_period = int(params.get("ema_mid", 64))
    ema_slow_period = int(params.get("ema_slow", 128))
    atr_period = int(params.get("atr_period", 14))
    atr_min_pct = float(params.get("atr_min_pct", 0.12))
    atr_max_pct = float(params.get("atr_max_pct", 1.70))
    volume_window = int(params.get("volume_window", 30))
    volume_long_window = int(params.get("volume_long_window", 120))
    volume_ratio_min = float(params.get("volume_ratio_min", 0.55))
    volume_long_ratio_min = float(params.get("volume_long_ratio_min", 0.65))
    volume_spike_ratio = float(params.get("volume_spike_ratio", 1.45))
    volume_long_spike_ratio = float(params.get("volume_long_spike_ratio", 1.35))
    signal_value = float(params.get("signal_value", 0.76))
    weak_signal_value = float(params.get("weak_signal_value", 0.35))
    long_signal_value = float(params.get("long_signal_value", weak_signal_value))
    signal_cooldown_bars = int(params.get("signal_cooldown_bars", 12))

    close = df["close"].astype("float64")
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")
    open_ = df["open"].astype("float64")
    volume = df.get("volume", pd.Series(0.0, index=df.index)).astype("float64")
    min_bars = max(ema_slow_period, volume_long_window, 160) + 20
    if len(df) < min_bars:
        return pd.Series(0.0, index=df.index, dtype="float64")

    ema_fast = close.ewm(span=ema_fast_period, adjust=False).mean()
    ema_mid = close.ewm(span=ema_mid_period, adjust=False).mean()
    ema_slow = close.ewm(span=ema_slow_period, adjust=False).mean()

    ret5 = close.pct_change(5) * 100.0
    ret15 = close.pct_change(15) * 100.0
    ret20 = close.pct_change(20) * 100.0
    ret40 = close.pct_change(40) * 100.0
    ret80 = close.pct_change(80) * 100.0
    ret160 = close.pct_change(160) * 100.0

    prev_high20 = high.shift(1).rolling(20, min_periods=20).max()
    prev_high60 = high.shift(1).rolling(60, min_periods=40).max()
    prev_high80 = high.shift(1).rolling(80, min_periods=50).max()
    prev_low60 = low.shift(1).rolling(60, min_periods=40).min()
    prev_low80 = low.shift(1).rolling(80, min_periods=50).min()

    from_high20 = (close / (prev_high20 + 1e-10) - 1.0) * 100.0
    from_high60 = (close / (prev_high60 + 1e-10) - 1.0) * 100.0
    from_high80 = (close / (prev_high80 + 1e-10) - 1.0) * 100.0
    from_low60 = (close / (prev_low60 + 1e-10) - 1.0) * 100.0
    from_low80 = (close / (prev_low80 + 1e-10) - 1.0) * 100.0

    bar_range = high - low + 1e-10
    close_pos = (close - low) / bar_range
    body_ratio = (close - open_).abs() / bar_range
    red = close < open_

    atr_pct = _true_range(high, low, close).rolling(atr_period, min_periods=8).mean() / (close + 1e-10) * 100.0
    volume_ratio = volume / (volume.shift(1).rolling(volume_window, min_periods=10).median() + 1e-10)
    volume_long_ratio = volume / (volume.shift(1).rolling(volume_long_window, min_periods=40).median() + 1e-10)
    rsi = _rsi(close, int(params.get("rsi_period", 14)))

    volatility_ok = (atr_pct >= atr_min_pct) & (atr_pct <= atr_max_pct)
    liquidity_ok = (volume_ratio >= volume_ratio_min) | (volume_long_ratio >= volume_long_ratio_min)
    volume_spike = (volume_ratio >= volume_spike_ratio) | (volume_long_ratio >= volume_long_spike_ratio)

    downside_acceptance_short = (
        volatility_ok
        & liquidity_ok
        & ((ret5 <= -0.25) | (ret20 <= -0.55))
        & (from_high60 <= -0.45)
        & (close_pos <= 0.70)
        & (ret160 >= -7.50)
        & (ret160 <= 3.80)
        & ((red & (body_ratio >= 0.12)) | (close_pos <= 0.35) | volume_spike)
    )

    retest_failure_short = (
        volatility_ok
        & liquidity_ok
        & ((ret20 >= 0.25) | (ret40 >= 0.45))
        & (from_high20 <= -0.05)
        & (from_high60 <= -0.05)
        & (from_low60 >= 0.65)
        & (close_pos <= 0.62)
        & (ret160 >= -6.50)
        & (ret160 <= 4.00)
        & ((red & (body_ratio >= 0.10)) | (volume_spike & (close_pos <= 0.72)))
    )

    late_rollover_short = (
        volatility_ok
        & liquidity_ok
        & (ret80 <= -1.20)
        & (from_high80 <= -1.50)
        & (from_low80 >= 0.35)
        & (ret5 <= 0.75)
        & (close_pos <= 0.72)
        & (close <= ema_fast * 1.012)
        & (ema_fast <= ema_mid * 1.012)
    )

    short_regime_ok = (
        (ret160 <= 2.20)
        & ((close <= ema_slow * 1.018) | (ret80 <= -1.00) | (from_high80 <= -1.00))
    )
    vertical_reclaim_block = (
        (ret40 >= 3.20)
        & (close >= ema_fast * 1.010)
        & (ema_fast >= ema_mid * 1.006)
        & (close_pos >= 0.45)
    )

    short_signal = (downside_acceptance_short | retest_failure_short | late_rollover_short) & short_regime_ok & ~vertical_reclaim_block
    high_confidence_short = (
        ((ret20 <= -0.90) & (from_high60 <= -1.00) & (close_pos <= 0.55))
        | (
            (ret40 >= 0.70)
            & (ret40 <= 3.20)
            & (from_high20 <= -0.15)
            & (from_high60 <= -0.25)
            & (from_low60 >= 1.00)
            & (close_pos <= 0.50)
            & (red | volume_spike)
        )
        | ((ret80 <= -1.80) & (from_high80 <= -2.00) & (close_pos <= 0.62) & (ret5 <= 0.30))
        | (volume_spike & (close_pos <= 0.30) & ((ret20 <= -0.40) | (from_high60 <= -0.80)))
    )

    controlled_reclaim_long = (
        volatility_ok
        & liquidity_ok
        & (ret5 >= 0.15)
        & (ret15 >= 0.55)
        & (from_low60 >= 1.00)
        & (from_high80 <= -0.25)
        & (from_high80 >= -2.60)
        & (ret80 >= -3.00)
        & (ret80 <= 1.80)
        & (close >= ema_fast * 0.992)
        & (rsi >= 38.0)
        & (rsi <= 64.0)
    )
    high_confidence_long = (
        controlled_reclaim_long
        & (ret15 >= 0.80)
        & (from_low60 >= 1.15)
        & (from_high80 <= -0.45)
        & (close >= ema_slow * 0.985)
    )

    result = pd.Series(0.0, index=df.index, dtype="float64")
    result[short_signal] = -weak_signal_value
    result[controlled_reclaim_long] = weak_signal_value
    result[short_signal & high_confidence_short] = -signal_value
    result[high_confidence_long] = long_signal_value
    result[short_signal & controlled_reclaim_long] = 0.0
    result = _apply_directional_cooldown(result, signal_cooldown_bars)
    return result.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype("float64")
