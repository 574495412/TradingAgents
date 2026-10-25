"""
Short cascade continuation factor.

Meme crashes often have two profitable short structures:
- OI rises while price falls: fresh shorts are pressing the move.
- OI drops while price falls: long liquidation is already cascading.

Older factors focused mostly on squeeze/funding setups or post-pump crash
warnings. This factor is deliberately one-sided and detects tradable downside
continuation after the break has started.
"""

import os

import numpy as np
import pandas as pd


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _pct(cur: float, prev: float) -> float:
    if prev <= 0:
        return 0.0
    return (cur - prev) / prev * 100.0


def calculate_factor(df: pd.DataFrame) -> pd.Series:
    """Return a negative signal in [-1, 0] when downside continuation has fuel."""
    n = len(df)
    signal = pd.Series(0.0, index=df.index)
    if n < 70:
        return signal

    close = df["close"].astype(float).values
    open_ = df["open"].astype(float).values if "open" in df else np.roll(close, 1)
    open_[0] = close[0]
    high = df["high"].astype(float).values if "high" in df else close
    low = df["low"].astype(float).values if "low" in df else close
    volume = df["volume"].astype(float).values if "volume" in df else np.ones(n)
    oi = df["open_interest"].ffill().astype(float).values if "open_interest" in df else np.zeros(n)
    funding = df["funding_rate"].fillna(0).astype(float).values if "funding_rate" in df else np.zeros(n)

    ret5_min = _env_float("MEME_SHORT_CASCADE_RET5_PCT", -1.5)
    ret20_min = _env_float("MEME_SHORT_CASCADE_RET20_PCT", -5.0)
    ret60_min = _env_float("MEME_SHORT_CASCADE_RET60_PCT", -10.0)
    oi_rise_min = _env_float("MEME_SHORT_CASCADE_OI_RISE_MIN", 2.0)
    oi_drop_min = _env_float("MEME_SHORT_CASCADE_OI_DROP_MIN", 3.0)
    vol_ratio_min = _env_float("MEME_SHORT_CASCADE_VOL_RATIO_MIN", 1.3)
    max_bounce_pct = _env_float("MEME_SHORT_CASCADE_MAX_BOUNCE_PCT", 4.5)
    overheat_ret5 = _env_float("MEME_SHORT_CASCADE_OVERHEAT_RET5_PCT", -16.0)
    min_output = _env_float("MEME_SHORT_CASCADE_MIN_OUTPUT", 0.30)

    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=8, min_periods=5).mean().values
    ema_slow = close_s.ewm(span=21, min_periods=10).mean().values
    vol_ma = pd.Series(volume).rolling(20, min_periods=10).mean().values

    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_fast = pd.Series(tr).rolling(5, min_periods=3).mean().values
    atr_slow = pd.Series(tr).rolling(20, min_periods=10).mean().values

    raw = np.zeros(n)
    for i in range(60, n):
        ret5 = _pct(close[i], close[i - 5])
        ret20 = _pct(close[i], close[i - 20])
        ret60 = _pct(close[i], close[i - 60])
        if ret5 > ret5_min and ret20 > ret20_min and ret60 > ret60_min:
            continue

        ema_diff = _pct(ema_fast[i], ema_slow[i]) if ema_slow[i] > 0 else 0.0
        if ema_diff > 0.05 and ret20 > ret20_min:
            continue

        recent_low = float(np.min(low[max(0, i - 14): i + 1]))
        bounce = _pct(close[i], recent_low) if recent_low > 0 else 0.0
        if bounce > max_bounce_pct:
            continue

        oi_roc3 = _pct(oi[i], oi[i - 3]) if oi[i - 3] > 0 else 0.0
        oi_roc20 = _pct(oi[i], oi[i - 20]) if oi[i - 20] > 0 else 0.0
        oi_press = max(0.0, oi_roc3, oi_roc20 * 0.6)
        liquidation = max(0.0, -oi_roc3, -oi_roc20 * 0.7)

        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 1.0
        atr_ratio = atr_fast[i] / atr_slow[i] if atr_slow[i] > 0 else 1.0

        bar_range = max(1e-12, high[i] - low[i])
        close_near_low = (close[i] - low[i]) / bar_range
        red_body = close[i] < open_[i]

        evidence = 0
        score = 0.0

        if ret5 <= ret5_min:
            score += min(0.18, abs(ret5) / max(abs(ret5_min) * 4.0, 1.0) * 0.18)
            evidence += 1
        if ret20 <= ret20_min:
            score += min(0.24, abs(ret20) / max(abs(ret20_min) * 3.0, 1.0) * 0.24)
            evidence += 1
        if ret60 <= ret60_min:
            score += min(0.18, abs(ret60) / max(abs(ret60_min) * 2.5, 1.0) * 0.18)
            evidence += 1
        if oi_press >= oi_rise_min:
            score += min(0.18, oi_press / max(oi_rise_min * 5.0, 1.0) * 0.18)
            evidence += 1
        if liquidation >= oi_drop_min:
            score += min(0.20, liquidation / max(oi_drop_min * 4.0, 1.0) * 0.20)
            evidence += 1
        if vol_ratio >= vol_ratio_min:
            score += min(0.10, (vol_ratio - 1.0) / 4.0 * 0.10)
            evidence += 1
        if atr_ratio >= 1.25:
            score += min(0.08, (atr_ratio - 1.0) / 2.5 * 0.08)
            evidence += 1
        if funding[i] > 0.05:
            score += min(0.08, funding[i] / 1.0 * 0.08)
            evidence += 1
        if red_body or close_near_low <= 0.35:
            score += 0.08
            evidence += 1

        if evidence < 3:
            continue
        if ret5 <= overheat_ret5 and close_near_low > 0.45 and liquidation < oi_drop_min:
            score *= 0.65
        if bounce > max_bounce_pct * 0.55:
            score *= 0.75

        if score >= min_output:
            raw[i] = -min(1.0, score * 1.25)

    smooth = pd.Series(raw).ewm(span=3, min_periods=2).mean().values
    for i in range(60, n):
        if smooth[i] <= -min_output:
            signal.iloc[i] = float(np.clip(smooth[i] * 1.2, -1.0, 0.0))

    return signal
