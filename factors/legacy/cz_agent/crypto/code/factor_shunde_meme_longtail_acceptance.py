"""Shunde-style autonomous meme longtail acceptance factor.

The visible Shunde Fund labels are dominated by longtail meme longs with a
smaller failed-squeeze short sleeve.  This factor deliberately does not read
leader/copytrading events at runtime; it converts that behavior into as-of
market-structure templates: launch acceptance, retest reclaim, and failed
squeeze rejection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cz.trading.crypto.meme.profile_config import profile_bool, profile_float


def _env_float(name: str, default: float) -> float:
    return profile_float(name, default)


def _env_bool(name: str, default: bool = False) -> bool:
    return profile_bool(name, default)


def _safe_series(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").ffill().fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _clip01(value: pd.Series) -> pd.Series:
    return value.clip(lower=0.0, upper=1.0)


def calculate_factor(df: pd.DataFrame) -> pd.Series:
    min_bars = max(100, int(_env_float("MEME_SHUNDE_LONGTAIL_ACCEPTANCE_MIN_BARS", 100)))
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

    min_output = _env_float("MEME_SHUNDE_LONGTAIL_ACCEPTANCE_MIN_OUTPUT", 0.24)
    max_long_output = _env_float("MEME_SHUNDE_LONGTAIL_ACCEPTANCE_MAX_LONG_OUTPUT", 0.88)
    max_short_output = _env_float("MEME_SHUNDE_LONGTAIL_ACCEPTANCE_MAX_SHORT_OUTPUT", 0.68)
    max_bar_range = _env_float("MEME_SHUNDE_LONGTAIL_ACCEPTANCE_MAX_BAR_RANGE_PCT", 18.0)
    max_vol_ratio = _env_float("MEME_SHUNDE_LONGTAIL_ACCEPTANCE_MAX_VOL_RATIO", 16.0)
    min_taker_buy = _env_float("MEME_SHUNDE_LONGTAIL_ACCEPTANCE_MIN_TAKER_BUY_RATIO", 0.43)

    ret1 = close.pct_change(1) * 100.0
    ret3 = close.pct_change(3) * 100.0
    ret5 = close.pct_change(5) * 100.0
    ret15 = close.pct_change(15) * 100.0
    ret30 = close.pct_change(30) * 100.0
    ret60 = close.pct_change(60) * 100.0
    ret240 = close.pct_change(240) * 100.0

    high20 = high.rolling(20, min_periods=8).max()
    low20 = low.rolling(20, min_periods=8).min()
    high60 = high.rolling(60, min_periods=20).max()
    low60 = low.rolling(60, min_periods=20).min()
    high240 = high.rolling(240, min_periods=80).max()
    low240 = low.rolling(240, min_periods=80).min()
    from_high20 = (close / high20.replace(0.0, np.nan) - 1.0) * 100.0
    from_high60 = (close / high60.replace(0.0, np.nan) - 1.0) * 100.0
    from_high240 = (close / high240.replace(0.0, np.nan) - 1.0) * 100.0
    from_low20 = (close / low20.replace(0.0, np.nan) - 1.0) * 100.0
    from_low60 = (close / low60.replace(0.0, np.nan) - 1.0) * 100.0
    from_low240 = (close / low240.replace(0.0, np.nan) - 1.0) * 100.0

    ema8 = close.ewm(span=8, min_periods=5, adjust=False).mean()
    ema21 = close.ewm(span=21, min_periods=10, adjust=False).mean()
    ema55 = close.ewm(span=55, min_periods=25, adjust=False).mean()

    vol_med_base = volume.rolling(30, min_periods=12).median().shift(1).replace(0.0, np.nan)
    vol_mean_base = volume.rolling(30, min_periods=12).mean().shift(1).replace(0.0, np.nan)
    quote_med_base = quote_volume.rolling(45, min_periods=18).median().shift(1).replace(0.0, np.nan)
    vol_ratio_med = (volume / vol_med_base).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    vol_ratio_mean = (volume / vol_mean_base).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    quote_ratio = (quote_volume / quote_med_base).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    taker_buy_ratio = (
        taker_buy_quote / quote_volume.replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.50)
    oi_roc20 = oi.pct_change(20).replace([np.inf, -np.inf], np.nan).fillna(0.0) * 100.0
    oi_roc60 = oi.pct_change(60).replace([np.inf, -np.inf], np.nan).fillna(0.0) * 100.0

    bar_range = (high - low).replace(0.0, np.nan)
    close_pos = (close - low) / bar_range
    upper_wick = (high - pd.concat([open_, close], axis=1).max(axis=1)) / bar_range
    lower_wick = (pd.concat([open_, close], axis=1).min(axis=1) - low) / bar_range
    bar_range_pct = (high / low.replace(0.0, np.nan) - 1.0) * 100.0

    liquidity_quality = (
        0.42 * _clip01((vol_ratio_med - 0.78) / 3.2)
        + 0.28 * _clip01((vol_ratio_mean - 0.70) / 3.0)
        + 0.20 * _clip01((quote_ratio - 0.85) / 3.5)
        + 0.10 * _clip01(quote_volume / 750_000.0)
    ) * (1.0 - _clip01((vol_ratio_med - max_vol_ratio) / max(max_vol_ratio, 1.0)))
    buy_acceptance = (
        0.36 * _clip01((taker_buy_ratio - min_taker_buy) / 0.24)
        + 0.30 * _clip01((close_pos.fillna(0.5) - 0.45) / 0.42)
        + 0.18 * (upper_wick.fillna(0.0) <= 0.54).astype(float)
        + 0.10 * (lower_wick.fillna(0.0) <= 0.62).astype(float)
        + 0.06 * _clip01(oi_roc20.abs() / 18.0)
    )
    broad_context = (
        0.28 * _clip01((ret60.fillna(0.0) + 8.0) / 34.0)
        + 0.24 * _clip01((from_high60.fillna(-999.0) + 12.0) / 12.0)
        + 0.20 * _clip01((from_low60.fillna(0.0) - 1.0) / 18.0)
        + 0.16 * _clip01((from_high240.fillna(-999.0) + 28.0) / 28.0)
        + 0.12 * _clip01((from_low240.fillna(0.0) + 4.0) / 28.0)
    )

    launch_momentum = (
        0.42 * _clip01((ret15.fillna(0.0) - 3.2) / 15.0)
        + 0.24 * _clip01((ret5.fillna(0.0) - 0.15) / 9.5)
        + 0.18 * _clip01((ret30.fillna(0.0) - 4.0) / 24.0)
        + 0.16 * _clip01((from_high20.fillna(-999.0) + 2.8) / 2.8)
    )
    launch_health = (
        0.34 * (close >= ema8 * 0.994).astype(float)
        + 0.28 * (ema8 >= ema21 * 0.988).astype(float)
        + 0.20 * _clip01((from_low20.fillna(0.0) - 1.8) / 8.5)
        + 0.18 * _clip01((oi_roc20.fillna(0.0) + 3.0) / 18.0)
    )
    launch_score = (
        0.34 * launch_momentum
        + 0.22 * liquidity_quality
        + 0.18 * buy_acceptance
        + 0.16 * launch_health
        + 0.10 * broad_context
    )
    launch_valid = (
        (ret15 >= 3.2)
        & (ret15 <= 24.0)
        & (ret5 >= 0.15)
        & (ret5 <= 12.0)
        & (ret1 >= -1.25)
        & (ret1 <= 5.2)
        & (ret60 >= -8.0)
        & (ret240.isna() | (ret240 >= -34.0))
        & (from_high20 >= -2.8)
        & (from_high60 >= -9.5)
        & (from_low20 >= 1.8)
        & (from_low60 >= 2.5)
        & (vol_ratio_med >= 0.78)
        & (vol_ratio_med <= max_vol_ratio)
        & (vol_ratio_mean <= 12.0)
        & (taker_buy_ratio >= min_taker_buy)
        & (close_pos.fillna(0.5) >= 0.45)
        & (upper_wick.fillna(0.0) <= 0.58)
        & (bar_range_pct <= max_bar_range)
        & (close >= ema21 * 0.975)
    )

    retest_quality = (
        0.26 * _clip01((-from_high20.fillna(0.0) - 0.35) / 7.5)
        + 0.22 * _clip01((from_low20.fillna(0.0) - 0.8) / 7.0)
        + 0.22 * _clip01((ret3.fillna(0.0) + 0.10) / 3.6)
        + 0.16 * _clip01((ret5.fillna(0.0) + 1.2) / 6.0)
        + 0.14 * _clip01((close / ema8.replace(0.0, np.nan) - 0.992) / 0.020)
    )
    retest_fuel = (
        0.34 * liquidity_quality
        + 0.24 * buy_acceptance
        + 0.16 * _clip01((oi_roc60.fillna(0.0) + 4.0) / 22.0)
        + 0.14 * _clip01((-funding.fillna(0.0)) / 0.22)
        + 0.12 * broad_context
    )
    reclaim_fresh = (
        ((close >= ema8 * 0.996) & (close.shift(1) <= ema8.shift(1) * 1.006))
        | ((ret1.fillna(0.0) >= -0.10) & (ret3.shift(1).fillna(0.0) <= 0.25))
        | ((vol_ratio_med >= vol_ratio_med.shift(1).fillna(1.0) * 1.12) & (close_pos.fillna(0.5) >= 0.50))
    )
    retest_score = 0.58 * retest_quality + 0.42 * retest_fuel
    retest_valid = (
        (from_high20 >= -8.8)
        & (from_high20 <= -0.30)
        & (from_high60 >= -13.5)
        & (from_low20 >= 0.75)
        & (from_low20 <= 18.0)
        & (from_low60 <= 28.0)
        & (ret1 >= -0.85)
        & (ret1 <= 3.2)
        & (ret3 >= -0.25)
        & (ret5 >= -1.35)
        & (ret5 <= 7.5)
        & (ret15 >= -4.5)
        & (ret15 <= 12.0)
        & (ret60 >= -12.0)
        & (ret240.isna() | (ret240 >= -40.0))
        & reclaim_fresh
        & (vol_ratio_med >= 0.68)
        & (vol_ratio_med <= 10.5)
        & (taker_buy_ratio >= 0.41)
        & (close_pos.fillna(0.5) >= 0.44)
        & (upper_wick.fillna(0.0) <= 0.60)
        & (bar_range_pct <= 12.5)
        & (close >= ema21 * 0.965)
    )

    overheat_guard = (
        1.0
        - 0.30 * _clip01((ret15.fillna(0.0) - 24.0) / 16.0)
        - 0.28 * _clip01((ret5.fillna(0.0) - 12.0) / 8.0)
        - 0.22 * _clip01((vol_ratio_med - max_vol_ratio) / max(max_vol_ratio, 1.0))
        - 0.20 * (upper_wick.fillna(0.0) > 0.66).astype(float)
    ).clip(lower=0.0, upper=1.0)
    downtrend_guard = (
        1.0
        - 0.35 * ((ret240.fillna(0.0) < -30.0) & (ret60.fillna(0.0) < -12.0)).astype(float)
        - 0.25 * (close < ema55 * 0.945).astype(float)
    ).clip(lower=0.25, upper=1.0)
    long_score = pd.concat(
        [
            launch_score.where(launch_valid, 0.0),
            retest_score.where(retest_valid, 0.0),
        ],
        axis=1,
    ).max(axis=1)
    long_score = (long_score * overheat_guard * downtrend_guard).clip(lower=0.0, upper=max_long_output)
    if _env_bool("MEME_SHUNDE_LONGTAIL_ACCEPTANCE_STRICT_QUALITY", False):
        strict_quality = (
            (from_high20 >= _env_float("MEME_SHUNDE_LONGTAIL_ACCEPTANCE_STRICT_MIN_FROM_HIGH20_PCT", -2.2))
            & (from_low20 <= _env_float("MEME_SHUNDE_LONGTAIL_ACCEPTANCE_STRICT_MAX_FROM_LOW20_PCT", 8.5))
            & (vol_ratio_med <= _env_float("MEME_SHUNDE_LONGTAIL_ACCEPTANCE_STRICT_MAX_VOL_RATIO", 3.5))
            & (oi_roc20 >= _env_float("MEME_SHUNDE_LONGTAIL_ACCEPTANCE_STRICT_MIN_OI20_PCT", 1.0))
            & (ret1 >= _env_float("MEME_SHUNDE_LONGTAIL_ACCEPTANCE_STRICT_MIN_RET1_PCT", -1.2))
        )
        long_score = long_score.where(strict_quality, 0.0)

    failed_squeeze = (
        0.28 * _clip01((ret60.fillna(0.0) - 7.0) / 28.0)
        + 0.24 * _clip01((ret15.fillna(0.0) - 2.5) / 12.0)
        + 0.18 * _clip01((-from_high20.fillna(0.0) - 0.20) / 7.0)
        + 0.16 * _clip01((0.54 - close_pos.fillna(0.5)) / 0.45)
        + 0.14 * _clip01((upper_wick.fillna(0.0) - 0.18) / 0.42)
    )
    supply_confirm = (
        0.30 * _clip01((0.57 - taker_buy_ratio) / 0.22)
        + 0.24 * _clip01((ret1.fillna(0.0) + 0.40) / 2.2)
        + 0.18 * _clip01((vol_ratio_med - 0.70) / 3.0)
        + 0.16 * _clip01((close / ema8.replace(0.0, np.nan) - 0.985) / 0.035)
        + 0.12 * _clip01((ema8 / ema21.replace(0.0, np.nan) - 0.995) / 0.050)
    )
    short_score = (0.62 * failed_squeeze + 0.38 * supply_confirm).clip(lower=0.0, upper=max_short_output)
    short_valid = (
        ((ret60 >= 7.0) | (ret15 >= 2.5))
        & (from_high20 <= -0.20)
        & (from_high20 >= -9.5)
        & (from_low20 >= 0.60)
        & (ret1 <= 0.85)
        & (ret3 <= 1.8)
        & (ret5 >= -5.0)
        & (ret15 <= 28.0)
        & (vol_ratio_med >= 0.65)
        & (vol_ratio_med <= 12.0)
        & (taker_buy_ratio <= 0.57)
        & ((close_pos.fillna(0.5) <= 0.48) | (upper_wick.fillna(0.0) >= 0.32))
        & (upper_wick.fillna(0.0) <= 0.82)
        & (bar_range_pct <= 16.0)
        & (close >= ema21 * 0.940)
    )
    short_score = short_score.where(short_valid, 0.0)

    out = pd.Series(0.0, index=df.index)
    out = out.mask(long_score >= min_output, long_score)
    if _env_bool("MEME_SHUNDE_LONGTAIL_ACCEPTANCE_ENABLE_SHORT", True):
        out = out.mask((short_score >= min_output) & (long_score < min_output + 0.08), -short_score)
    return out.where(out.abs() >= min_output, 0.0).fillna(0.0)
