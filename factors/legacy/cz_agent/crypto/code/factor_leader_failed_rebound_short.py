"""Leader-audit failed-rebound short factor.

This is the executable version of the ``反抽失败做空`` closed-position
structure mined from external lead-trader labels.  The family definition is:
short after a small rebound where 3-minute return is positive but 5-minute
follow-through is capped.  All inputs are as-of market features only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cz.trading.crypto.meme.profile_config import profile_float


def _env_float(name: str, default: float) -> float:
    return profile_float(name, default)


def _safe_series(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").ffill().fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _clip01(value: pd.Series) -> pd.Series:
    return value.clip(lower=0.0, upper=1.0)


def calculate_factor(df: pd.DataFrame) -> pd.Series:
    min_bars = max(80, int(_env_float("MEME_LEADER_FAILED_REBOUND_SHORT_MIN_BARS", 80)))
    if df is None or len(df) < min_bars:
        return pd.Series(0.0, index=getattr(df, "index", None))

    open_ = _safe_series(df, "open")
    high = _safe_series(df, "high")
    low = _safe_series(df, "low")
    close = _safe_series(df, "close")
    volume = _safe_series(df, "volume", 1.0)
    quote_volume = _safe_series(df, "quote_volume")
    taker_buy_quote = _safe_series(df, "taker_buy_quote")

    min_output = _env_float("MEME_LEADER_FAILED_REBOUND_SHORT_MIN_OUTPUT", 0.25)
    max_output = _env_float("MEME_LEADER_FAILED_REBOUND_SHORT_MAX_OUTPUT", 0.84)
    min_ret3 = _env_float("MEME_LEADER_FAILED_REBOUND_SHORT_MIN_RET3_PCT", 0.50)
    max_ret5 = _env_float("MEME_LEADER_FAILED_REBOUND_SHORT_MAX_RET5_PCT", 1.00)
    min_ret5 = _env_float("MEME_LEADER_FAILED_REBOUND_SHORT_MIN_RET5_PCT", -3.80)
    max_ret1 = _env_float("MEME_LEADER_FAILED_REBOUND_SHORT_MAX_RET1_PCT", 1.00)
    min_from_low20 = _env_float("MEME_LEADER_FAILED_REBOUND_SHORT_MIN_FROM_LOW20_PCT", 0.40)
    max_from_low20 = _env_float("MEME_LEADER_FAILED_REBOUND_SHORT_MAX_FROM_LOW20_PCT", 8.50)
    max_from_high20 = _env_float("MEME_LEADER_FAILED_REBOUND_SHORT_MAX_FROM_HIGH20_PCT", -0.15)
    min_vol_ratio = _env_float("MEME_LEADER_FAILED_REBOUND_SHORT_MIN_VOL_RATIO", 0.45)
    max_vol_ratio = _env_float("MEME_LEADER_FAILED_REBOUND_SHORT_MAX_VOL_RATIO", 8.00)
    max_taker_buy_ratio = _env_float("MEME_LEADER_FAILED_REBOUND_SHORT_MAX_TAKER_BUY_RATIO", 0.72)
    max_bar_range = _env_float("MEME_LEADER_FAILED_REBOUND_SHORT_MAX_BAR_RANGE_PCT", 8.00)

    ret1 = close.pct_change(1) * 100.0
    ret3 = close.pct_change(3) * 100.0
    ret5 = close.pct_change(5) * 100.0
    ret60 = close.pct_change(60) * 100.0

    high20 = high.rolling(20, min_periods=8).max()
    low20 = low.rolling(20, min_periods=8).min()
    high60 = high.rolling(60, min_periods=20).max()
    from_high20 = (close / high20.replace(0.0, np.nan) - 1.0) * 100.0
    from_high60 = (close / high60.replace(0.0, np.nan) - 1.0) * 100.0
    from_low20 = (close / low20.replace(0.0, np.nan) - 1.0) * 100.0

    vol_base = volume.rolling(30, min_periods=12).median().shift(1).replace(0.0, np.nan)
    vol_ratio = (volume / vol_base).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    taker_buy_ratio = (
        taker_buy_quote / quote_volume.replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.50)

    bar_range = (high - low).replace(0.0, np.nan)
    close_pos = (close - low) / bar_range
    upper_wick = (high - pd.concat([open_, close], axis=1).max(axis=1)) / bar_range
    bar_range_pct = (high / low.replace(0.0, np.nan) - 1.0) * 100.0

    failed_rebound = (
        0.34 * _clip01((ret3.fillna(0.0) - min_ret3) / 2.8)
        + 0.26 * _clip01((max_ret5 - ret5.fillna(0.0)) / max(max_ret5 - min_ret5, 0.1))
        + 0.18 * _clip01((0.62 - close_pos.fillna(0.5)) / 0.50)
        + 0.14 * _clip01((upper_wick.fillna(0.0) - 0.08) / 0.35)
        + 0.08 * _clip01((max_taker_buy_ratio - taker_buy_ratio) / 0.28)
    )
    location_quality = (
        0.42 * _clip01((from_low20.fillna(0.0) - min_from_low20) / 4.0)
        + 0.30 * _clip01((max_from_high20 - from_high20.fillna(0.0)) / 5.0)
        + 0.18 * _clip01((from_high60.fillna(0.0) + 8.0) / 8.0)
        + 0.10 * _clip01((ret60.fillna(0.0) + 4.0) / 12.0)
    )
    volume_ok = _clip01((vol_ratio - min_vol_ratio) / 1.6) * (
        1.0 - _clip01((vol_ratio - max_vol_ratio) / max(max_vol_ratio, 1.0))
    )

    score = 0.54 * failed_rebound + 0.34 * location_quality + 0.12 * volume_ok
    valid = (
        (ret3 >= min_ret3)
        & (ret5 <= max_ret5)
        & (ret5 >= min_ret5)
        & (ret1 <= max_ret1)
        & (from_low20 >= min_from_low20)
        & (from_low20 <= max_from_low20)
        & (from_high20 <= max_from_high20)
        & (vol_ratio >= min_vol_ratio)
        & (vol_ratio <= max_vol_ratio)
        & (taker_buy_ratio <= max_taker_buy_ratio)
        & (bar_range_pct <= max_bar_range)
    )

    out = -score.clip(lower=0.0, upper=max_output).where(valid, 0.0)
    return out.where(out.abs() >= min_output, 0.0).fillna(0.0)
