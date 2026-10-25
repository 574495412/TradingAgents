"""
Short rebound trap factor for meme coins.

Core idea:
- Only short inside an established downtrend.
- Wait for a rebound (dead-cat bounce) to reduce chasing risk.
- Enter when rebound quality is weak (volume not expanding, momentum fading).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


def _safe_series(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def calculate_factor(df: pd.DataFrame) -> pd.Series:
    if df is None or len(df) < 80:
        return pd.Series(0.0, index=getattr(df, "index", None))

    close = _safe_series(df, "close")
    volume = _safe_series(df, "volume")
    oi = _safe_series(df, "open_interest")
    funding = _safe_series(df, "funding_rate")

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema60 = close.ewm(span=60, adjust=False).mean()
    ret60 = close.pct_change(60) * 100.0

    low30 = close.rolling(30, min_periods=20).min()
    bounce_pct = (close / low30 - 1.0) * 100.0
    rsi14 = _rsi(close, 14)
    vol_ratio = volume / volume.rolling(20, min_periods=10).mean().replace(0.0, np.nan)
    oi_roc5 = oi.pct_change(5) * 100.0

    out = np.zeros(len(df), dtype=float)
    min_output = float(os.getenv("MEME_SHORT_REBOUND_TRAP_MIN_OUTPUT", "0.25"))

    for i in range(80, len(df)):
        if any(np.isnan(x) for x in (ret60.iloc[i], ema20.iloc[i], ema60.iloc[i], bounce_pct.iloc[i], rsi14.iloc[i])):
            continue
        if ret60.iloc[i] > -6.0:
            continue
        if ema20.iloc[i] >= ema60.iloc[i]:
            continue

        score = 0.0
        b = float(bounce_pct.iloc[i])
        if 2.0 <= b <= 12.0:
            score += min(0.45, (b - 2.0) / 10.0 * 0.45)
        elif b > 12.0:
            score += 0.35

        r = float(rsi14.iloc[i])
        if 42.0 <= r <= 62.0:
            score += 0.20
        elif r > 62.0:
            score += 0.10

        vr = float(vol_ratio.iloc[i]) if not np.isnan(vol_ratio.iloc[i]) else 1.0
        if vr < 1.10:
            score += 0.15

        oi5 = float(oi_roc5.iloc[i]) if not np.isnan(oi_roc5.iloc[i]) else 0.0
        if oi5 > 0.5:
            score += 0.10

        fr = float(funding.iloc[i]) if not np.isnan(funding.iloc[i]) else 0.0
        if fr > 0.05:
            score += 0.10

        if close.iloc[i] < ema20.iloc[i]:
            score += 0.10

        signal = -min(1.0, score * 1.35) if score > 0.35 else 0.0
        if abs(signal) < min_output:
            signal = 0.0
        out[i] = signal

    return pd.Series(out, index=df.index)

