"""
Strong runner quality factor.

This factor turns the old runner heuristics into a normal meme factor so live
trading and backtests can share one signal dictionary. It only fires when a
fast meme move has both OI fuel and trend confirmation; ordinary short bursts
stay unconfirmed and are handled as small-size trades by sizing policy.
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
    """
    Return a signed quality score in [-1, 1].

    Positive values confirm long runners; negative values confirm short
    runners. A zero value means the move is only a generic anomaly, not a
    strong runner.
    """
    n = len(df)
    signal = pd.Series(0.0, index=df.index)
    if n < 70:
        return signal

    close = df["close"].astype(float).values
    high = df["high"].astype(float).values if "high" in df else close
    low = df["low"].astype(float).values if "low" in df else close
    volume = df["volume"].astype(float).values if "volume" in df else np.ones(n)
    oi = df["open_interest"].ffill().astype(float).values if "open_interest" in df else np.zeros(n)

    ret20_min = _env_float("MEME_RUNNER_QUALITY_RET20_PCT", _env_float("MEME_RUNNER_RET20_PCT", 22.0))
    ret60_min = _env_float("MEME_RUNNER_QUALITY_RET60_PCT", _env_float("MEME_RUNNER_RET60_PCT", 35.0))
    oi_roc_min = _env_float("MEME_RUNNER_QUALITY_OI_ROC_MIN", 2.5)
    pullback_max = _env_float("MEME_RUNNER_QUALITY_MAX_PULLBACK_PCT", 3.0)
    bounce_max = _env_float("MEME_RUNNER_QUALITY_MAX_BOUNCE_PCT", 3.0)
    ret5_overheat = _env_float("MEME_RUNNER_QUALITY_RET5_OVERHEAT_PCT", 12.0)

    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=8, min_periods=5).mean().values
    ema_slow = close_s.ewm(span=21, min_periods=10).mean().values
    vol_ma = pd.Series(volume).rolling(20, min_periods=10).mean().values

    for i in range(60, n):
        ret20 = _pct(close[i], close[i - 20])
        ret60 = _pct(close[i], close[i - 60])
        if abs(ret20) < ret20_min and abs(ret60) < ret60_min:
            continue

        ret5 = _pct(close[i], close[i - 5])
        oi_roc = _pct(oi[i], oi[i - 3]) if oi[i - 3] > 0 else 0.0
        oi_roc20 = _pct(oi[i], oi[i - 20]) if oi[i - 20] > 0 else 0.0
        oi_evidence = max(abs(oi_roc), abs(oi_roc20) * 0.6)
        if oi_evidence < oi_roc_min:
            continue

        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 1.0
        ema_diff = _pct(ema_fast[i], ema_slow[i]) if ema_slow[i] > 0 else 0.0
        recent_high = float(np.max(high[max(0, i - 14): i + 1]))
        recent_low = float(np.min(low[max(0, i - 14): i + 1]))
        pullback = _pct(close[i], recent_high) if recent_high > 0 else 0.0
        bounce = _pct(close[i], recent_low) if recent_low > 0 else 0.0

        direction = 0
        if (ret20 >= ret20_min or ret60 >= ret60_min) and oi_roc > 0 and ema_diff > 0.03:
            if pullback < -pullback_max:
                continue
            direction = 1
        elif (
            (ret20 <= -ret20_min or ret60 <= -ret60_min)
            and (oi_roc > 0 or oi_roc20 <= -oi_roc_min)
            and ema_diff < -0.03
        ):
            if bounce > bounce_max:
                continue
            direction = -1
        if direction == 0:
            continue

        trend_score = min(0.30, max(abs(ret20) / max(ret20_min * 2.0, 1.0), abs(ret60) / max(ret60_min * 2.0, 1.0)) * 0.30)
        oi_score = min(0.30, oi_evidence / max(oi_roc_min * 4.0, 1.0) * 0.30)
        vol_score = min(0.15, max(0.0, vol_ratio - 1.0) / 4.0 * 0.15)
        structure_score = 0.25
        if direction > 0:
            structure_score *= max(0.0, 1.0 + pullback / max(pullback_max, 1e-9))
        else:
            structure_score *= max(0.0, 1.0 - bounce / max(bounce_max, 1e-9))

        quality = trend_score + oi_score + vol_score + structure_score
        if direction > 0 and ret5 > ret5_overheat:
            quality *= 0.70
        elif direction < 0 and ret5 < -ret5_overheat:
            quality *= 0.70
        if direction > 0 and pullback < -pullback_max * 0.5:
            quality *= 0.75
        elif direction < 0 and bounce > bounce_max * 0.5:
            quality *= 0.75
        if quality >= _env_float("MEME_RUNNER_QUALITY_MIN_OUTPUT", 0.35):
            signal.iloc[i] = float(np.clip(direction * quality, -1.0, 1.0))

    return signal
