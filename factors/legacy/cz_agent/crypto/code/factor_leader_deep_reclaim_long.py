"""Leader-audit deep-reclaim long factor.

Executable version of the ``深跌反包做多`` family mined from external
lead-trader labels.  The core label shape is a large 60-minute selloff followed
by a reclaim from the local 20-minute low, with current acceptance and volume
confirming the bounce.
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
    min_bars = max(90, int(_env_float("MEME_LEADER_DEEP_RECLAIM_LONG_MIN_BARS", 90)))
    if df is None or len(df) < min_bars:
        return pd.Series(0.0, index=getattr(df, "index", None))

    open_ = _safe_series(df, "open")
    high = _safe_series(df, "high")
    low = _safe_series(df, "low")
    close = _safe_series(df, "close")
    volume = _safe_series(df, "volume", 1.0)
    quote_volume = _safe_series(df, "quote_volume")
    taker_buy_quote = _safe_series(df, "taker_buy_quote")
    oi = _safe_series(df, "open_interest")
    funding = _safe_series(df, "funding_rate")

    min_output = _env_float("MEME_LEADER_DEEP_RECLAIM_LONG_MIN_OUTPUT", 0.28)
    max_output = _env_float("MEME_LEADER_DEEP_RECLAIM_LONG_MAX_OUTPUT", 0.88)
    max_ret60 = _env_float("MEME_LEADER_DEEP_RECLAIM_LONG_MAX_RET60_PCT", -12.0)
    panic_max_ret60 = _env_float("MEME_LEADER_DEEP_RECLAIM_LONG_PANIC_MAX_RET60_PCT", -28.0)
    min_from_low20 = _env_float("MEME_LEADER_DEEP_RECLAIM_LONG_MIN_FROM_LOW20_PCT", 6.0)
    max_from_low20 = _env_float("MEME_LEADER_DEEP_RECLAIM_LONG_MAX_FROM_LOW20_PCT", 18.0)
    max_from_high60 = _env_float("MEME_LEADER_DEEP_RECLAIM_LONG_MAX_FROM_HIGH60_PCT", -7.0)
    panic_max_from_high60 = _env_float("MEME_LEADER_DEEP_RECLAIM_LONG_PANIC_MAX_FROM_HIGH60_PCT", -12.0)
    min_ret1 = _env_float("MEME_LEADER_DEEP_RECLAIM_LONG_MIN_RET1_PCT", -1.4)
    max_ret1 = _env_float("MEME_LEADER_DEEP_RECLAIM_LONG_MAX_RET1_PCT", 4.5)
    min_ret5 = _env_float("MEME_LEADER_DEEP_RECLAIM_LONG_MIN_RET5_PCT", -2.2)
    max_ret5 = _env_float("MEME_LEADER_DEEP_RECLAIM_LONG_MAX_RET5_PCT", 9.0)
    min_ret240 = _env_float("MEME_LEADER_DEEP_RECLAIM_LONG_MIN_RET240_PCT", -45.0)
    min_vol_ratio = _env_float("MEME_LEADER_DEEP_RECLAIM_LONG_MIN_VOL_RATIO", 0.75)
    max_vol_ratio = _env_float("MEME_LEADER_DEEP_RECLAIM_LONG_MAX_VOL_RATIO", 16.0)
    min_taker_buy_ratio = _env_float("MEME_LEADER_DEEP_RECLAIM_LONG_MIN_TAKER_BUY_RATIO", 0.45)
    min_close_pos = _env_float("MEME_LEADER_DEEP_RECLAIM_LONG_MIN_CLOSE_POS", 0.44)
    max_bar_range = _env_float("MEME_LEADER_DEEP_RECLAIM_LONG_MAX_BAR_RANGE_PCT", 18.0)

    ret1 = close.pct_change(1) * 100.0
    ret3 = close.pct_change(3) * 100.0
    ret5 = close.pct_change(5) * 100.0
    ret15 = close.pct_change(15) * 100.0
    ret60 = close.pct_change(60) * 100.0
    ret240 = close.pct_change(240) * 100.0

    high20 = high.rolling(20, min_periods=8).max()
    low20 = low.rolling(20, min_periods=8).min()
    high60 = high.rolling(60, min_periods=20).max()
    low60 = low.rolling(60, min_periods=20).min()
    from_high20 = (close / high20.replace(0.0, np.nan) - 1.0) * 100.0
    from_high60 = (close / high60.replace(0.0, np.nan) - 1.0) * 100.0
    from_low20 = (close / low20.replace(0.0, np.nan) - 1.0) * 100.0
    from_low60 = (close / low60.replace(0.0, np.nan) - 1.0) * 100.0

    vol_base = volume.rolling(30, min_periods=12).median().shift(1).replace(0.0, np.nan)
    vol_ratio = (volume / vol_base).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    taker_buy_ratio = (
        taker_buy_quote / quote_volume.replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.50)
    oi_roc20 = oi.pct_change(20).replace([np.inf, -np.inf], np.nan).fillna(0.0) * 100.0

    bar_range = (high - low).replace(0.0, np.nan)
    close_pos = (close - low) / bar_range
    upper_wick = (high - pd.concat([open_, close], axis=1).max(axis=1)) / bar_range
    bar_range_pct = (high / low.replace(0.0, np.nan) - 1.0) * 100.0

    crash_depth = _clip01((max_ret60 - ret60.fillna(0.0)) / 20.0)
    reclaim = _clip01((from_low20.fillna(0.0) - min_from_low20) / max(max_from_low20 - min_from_low20, 1.0))
    not_late = 1.0 - _clip01((from_low20.fillna(999.0) - max_from_low20) / 12.0)
    acceptance = (
        0.34 * _clip01((close_pos.fillna(0.5) - min_close_pos) / 0.42)
        + 0.28 * _clip01((taker_buy_ratio - min_taker_buy_ratio) / 0.22)
        + 0.18 * _clip01((ret3.fillna(0.0) + 1.0) / 5.0)
        + 0.20 * (upper_wick.fillna(0.0) <= 0.55).astype(float)
    )
    volume_quality = _clip01((vol_ratio - min_vol_ratio) / 3.8) * (
        1.0 - _clip01((vol_ratio - max_vol_ratio) / max(max_vol_ratio, 1.0))
    )
    fuel = 0.35 * volume_quality + 0.25 * _clip01(oi_roc20.abs() / 18.0) + 0.20 * _clip01((-funding) / 0.20) + 0.20 * _clip01((ret15 + 12.0) / 16.0)

    score = 0.30 * crash_depth + 0.26 * reclaim + 0.16 * not_late + 0.18 * acceptance + 0.10 * fuel
    controlled_reclaim = (
        (ret60 <= max_ret60)
        & (from_low20 >= min_from_low20)
        & (from_low20 <= max_from_low20)
        & (from_high60 <= max_from_high60)
        & (from_low60 >= min_from_low20)
        & (ret1 >= min_ret1)
        & (ret1 <= max_ret1)
        & (ret5 >= min_ret5)
        & (ret5 <= max_ret5)
        & (ret240.isna() | (ret240 >= min_ret240))
        & (from_high20 <= 1.5)
        & (vol_ratio >= min_vol_ratio)
        & (vol_ratio <= max_vol_ratio)
        & (taker_buy_ratio >= min_taker_buy_ratio)
        & (close_pos.fillna(0.5) >= min_close_pos)
        & (bar_range_pct <= max_bar_range)
    )
    panic_reclaim = (
        (ret60 <= panic_max_ret60)
        & (from_low20 >= min_from_low20)
        & (from_low20 <= max_from_low20)
        & (from_high60 <= panic_max_from_high60)
        & (from_low60 >= min_from_low20)
        & (ret1 >= min_ret1)
        & (ret1 <= max_ret1)
        & (ret5 >= min_ret5)
        & (ret5 <= max_ret5)
        & (from_high20 <= 1.5)
        & (vol_ratio >= min_vol_ratio)
        & (vol_ratio <= max_vol_ratio)
        & (taker_buy_ratio >= min_taker_buy_ratio)
        & (close_pos.fillna(0.5) >= min_close_pos)
        & (bar_range_pct <= max_bar_range)
    )
    valid = controlled_reclaim | panic_reclaim

    out = score.clip(lower=0.0, upper=max_output).where(valid, 0.0)
    return out.where(out.abs() >= min_output, 0.0).fillna(0.0)
