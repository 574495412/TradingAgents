"""Leader-labeled quant rejection short factor.

External high-performing meme lead-trader samples cluster around a repeatable
short-side structure: fast upward extension, active volume, then a weak close or
upper-wick rejection near the local high.  This factor encodes that market
structure directly with as-of OHLCV/OI/funding data, without reading leader
trades at runtime.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _safe_series(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").ffill().fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _clip01(value: pd.Series) -> pd.Series:
    return value.clip(lower=0.0, upper=1.0)


def calculate_factor(df: pd.DataFrame) -> pd.Series:
    if df is None or len(df) < 90:
        return pd.Series(0.0, index=getattr(df, "index", None))

    open_ = _safe_series(df, "open")
    high = _safe_series(df, "high")
    low = _safe_series(df, "low")
    close = _safe_series(df, "close")
    volume = _safe_series(df, "volume", 1.0)
    oi = _safe_series(df, "open_interest")
    funding = _safe_series(df, "funding_rate")

    min_output = _env_float("MEME_LEADER_QUANT_REJECTION_MIN_OUTPUT", 0.26)
    max_output = _env_float("MEME_LEADER_QUANT_REJECTION_MAX_OUTPUT", 0.88)
    min_vol_ratio = _env_float("MEME_LEADER_QUANT_REJECTION_MIN_VOL_RATIO", 0.9)
    max_close_pos = _env_float("MEME_LEADER_QUANT_REJECTION_MAX_CLOSE_POS", 0.68)
    min_close_pos = _env_float("MEME_LEADER_QUANT_REJECTION_MIN_CLOSE_POS", 0.42)
    min_upper_wick = _env_float("MEME_LEADER_QUANT_REJECTION_MIN_UPPER_WICK", 0.14)
    max_pullback_high20 = _env_float("MEME_LEADER_QUANT_REJECTION_MAX_PULLBACK_HIGH20_PCT", 5.0)
    min_ret15 = _env_float("MEME_LEADER_QUANT_REJECTION_MIN_RET15_PCT", -0.5)
    min_bounce20 = _env_float("MEME_LEADER_QUANT_REJECTION_MIN_BOUNCE20_PCT", 1.2)
    max_bounce20 = _env_float("MEME_LEADER_QUANT_REJECTION_MAX_BOUNCE20_PCT", 28.0)
    max_ret5 = _env_float("MEME_LEADER_QUANT_REJECTION_MAX_RET5_PCT", 18.0)

    ret1 = close.pct_change(1) * 100.0
    ret3 = close.pct_change(3) * 100.0
    ret5 = close.pct_change(5) * 100.0
    ret15 = close.pct_change(15) * 100.0
    ret60 = close.pct_change(60) * 100.0

    high20 = high.rolling(20, min_periods=8).max()
    low20 = low.rolling(20, min_periods=8).min()
    high60 = high.rolling(60, min_periods=20).max()
    from_high20 = (close / high20.replace(0.0, np.nan) - 1.0) * 100.0
    from_high60 = (close / high60.replace(0.0, np.nan) - 1.0) * 100.0
    from_low20 = (close / low20.replace(0.0, np.nan) - 1.0) * 100.0

    vol_base = volume.rolling(30, min_periods=12).median().shift(1).replace(0.0, np.nan)
    vol_ratio = (volume / vol_base).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    vol_recent = vol_ratio.rolling(8, min_periods=3).max().fillna(vol_ratio)

    bar_range = (high - low).replace(0.0, np.nan)
    upper_wick = (high - pd.concat([open_, close], axis=1).max(axis=1)) / bar_range
    close_pos = (close - low) / bar_range

    ema5 = close.ewm(span=5, min_periods=3, adjust=False).mean()
    ema13 = close.ewm(span=13, min_periods=6, adjust=False).mean()
    oi_roc5 = oi.pct_change(5).replace([np.inf, -np.inf], np.nan).fillna(0.0) * 100.0

    extension_score = pd.concat(
        [
            (ret3.fillna(0.0) - 0.4) / 4.0,
            (ret5.fillna(0.0) - 0.8) / 6.0,
            (ret15.fillna(0.0) - 1.2) / 12.0,
            (ret60.fillna(0.0) - 1.0) / 25.0,
            (from_low20.fillna(0.0) - 1.0) / 12.0,
        ],
        axis=1,
    ).max(axis=1)
    near_high_score = 1.0 - _clip01(
        pd.concat([-from_high20.fillna(-999.0), -from_high60.fillna(-999.0) * 0.65], axis=1).min(axis=1)
        / max(max_pullback_high20, 0.5)
    )
    rejection_score = (
        0.42 * _clip01((upper_wick.fillna(0.0) - min_upper_wick) / 0.45)
        + 0.34 * _clip01((max_close_pos - close_pos.fillna(0.5)) / max(max_close_pos, 0.1))
        + 0.24 * (ret1.fillna(0.0) <= 0.10).astype(float)
    )
    volume_score = _clip01((vol_recent - min_vol_ratio) / 5.0) * (1.0 - _clip01((vol_recent - 60.0) / 90.0))
    crowding_score = 0.65 * _clip01(oi_roc5 / 12.0) + 0.35 * _clip01(funding.fillna(0.0) / 0.20)
    ema_failure = ((close <= ema5 * 1.004) | (ema5 <= ema13 * 1.012)).astype(float)

    score = (
        0.30 * _clip01(extension_score)
        + 0.22 * near_high_score
        + 0.24 * rejection_score
        + 0.13 * volume_score
        + 0.07 * crowding_score
        + 0.04 * ema_failure
    )

    rejection = (
        (upper_wick.fillna(0.0) >= min_upper_wick)
        | (close_pos.fillna(0.5) <= max_close_pos)
        | (ret1.fillna(0.0) <= 0.10)
    )
    valid = (
        (vol_recent >= min_vol_ratio)
        & rejection
        & ((ret5 >= 0.4) | (ret15 >= 1.0) | (from_low20 >= min_bounce20))
        & (ret5 <= max_ret5)
        & (ret15 >= min_ret15)
        & (from_low20 >= min_bounce20)
        & (from_low20 <= max_bounce20)
        & (from_high20 >= -abs(max_pullback_high20))
        & (close_pos.fillna(0.5) >= min_close_pos)
        & (close >= ema13 * 0.985)
    )

    out = pd.Series(0.0, index=df.index)
    out = out.mask(valid, -score.clip(lower=0.0, upper=max_output))
    return out.where(out.abs() >= min_output, 0.0).fillna(0.0)
