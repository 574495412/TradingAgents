"""Public leader open-structure short factor.

High-win public lead-trader samples are useful as labels, but their public
trade stream is not a dependable live entry feed.  This factor models the
market structure repeatedly seen before those public short opens using only
as-of OHLCV/OI/funding fields that live and backtest share.
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
    if df is None or len(df) < 90:
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

    min_output = _env_float("MEME_LEADER_PUBLIC_OPEN_STRUCTURE_MIN_OUTPUT", 0.24)
    max_output = _env_float("MEME_LEADER_PUBLIC_OPEN_STRUCTURE_MAX_OUTPUT", 0.86)
    min_activity = _env_float("MEME_LEADER_PUBLIC_OPEN_STRUCTURE_MIN_VOL_RATIO", 0.65)
    max_activity = _env_float("MEME_LEADER_PUBLIC_OPEN_STRUCTURE_MAX_VOL_RATIO", 28.0)
    max_close_pos = _env_float("MEME_LEADER_PUBLIC_OPEN_STRUCTURE_MAX_CLOSE_POS", 0.72)
    min_upper_wick = _env_float("MEME_LEADER_PUBLIC_OPEN_STRUCTURE_MIN_UPPER_WICK", 0.08)
    max_ret5 = _env_float("MEME_LEADER_PUBLIC_OPEN_STRUCTURE_MAX_RET5_PCT", 9.0)
    max_ret1 = _env_float("MEME_LEADER_PUBLIC_OPEN_STRUCTURE_MAX_RET1_PCT", 2.0)
    min_from_low20 = _env_float("MEME_LEADER_PUBLIC_OPEN_STRUCTURE_MIN_FROM_LOW20_PCT", 0.7)
    max_from_low20 = _env_float("MEME_LEADER_PUBLIC_OPEN_STRUCTURE_MAX_FROM_LOW20_PCT", 999.0)
    max_from_high60 = _env_float("MEME_LEADER_PUBLIC_OPEN_STRUCTURE_MAX_PULLBACK_HIGH60_PCT", 8.0)
    max_taker_buy_ratio = _env_float("MEME_LEADER_PUBLIC_OPEN_STRUCTURE_MAX_TAKER_BUY_RATIO", 0.72)
    min_downtrend_bounce = _env_float("MEME_LEADER_PUBLIC_OPEN_STRUCTURE_MIN_DOWNTREND_BOUNCE_PCT", 1.0)
    max_downtrend_ret60 = _env_float("MEME_LEADER_PUBLIC_OPEN_STRUCTURE_MAX_DOWNTREND_RET60_PCT", 3.0)
    max_failed_bounce_ret5 = _env_float("MEME_LEADER_PUBLIC_OPEN_STRUCTURE_MAX_FAILED_BOUNCE_RET5_PCT", 0.0)
    max_failed_bounce_ret120 = _env_float("MEME_LEADER_PUBLIC_OPEN_STRUCTURE_MAX_FAILED_BOUNCE_RET120_PCT", 5.0)
    min_pump_ret15 = _env_float("MEME_LEADER_PUBLIC_OPEN_STRUCTURE_MIN_PUMP_RET15_PCT", 1.2)
    min_pump_ret60 = _env_float("MEME_LEADER_PUBLIC_OPEN_STRUCTURE_MIN_PUMP_RET60_PCT", 2.5)
    max_confirmed_decline_ret60 = _env_float(
        "MEME_LEADER_PUBLIC_OPEN_STRUCTURE_MAX_CONFIRMED_DECLINE_RET60_PCT",
        999.0,
    )
    max_confirmed_decline_from_high60 = _env_float(
        "MEME_LEADER_PUBLIC_OPEN_STRUCTURE_MAX_CONFIRMED_DECLINE_FROM_HIGH60_PCT",
        999.0,
    )
    min_confirmed_decline_vol_mean = _env_float(
        "MEME_LEADER_PUBLIC_OPEN_STRUCTURE_MIN_CONFIRMED_DECLINE_VOL_MEAN_RATIO",
        0.0,
    )

    ret1 = close.pct_change(1) * 100.0
    ret5 = close.pct_change(5) * 100.0
    ret15 = close.pct_change(15) * 100.0
    ret60 = close.pct_change(60) * 100.0
    ret120 = close.pct_change(120) * 100.0
    window_ret = (close / float(close.iloc[0]) - 1.0) * 100.0 if len(close) > 0 and close.iloc[0] else ret120
    medium_ret = ret120.where(ret120.notna(), window_ret)
    ret240 = close.pct_change(240) * 100.0

    high20 = high.rolling(20, min_periods=8).max()
    high60 = high.rolling(60, min_periods=20).max()
    low20 = low.rolling(20, min_periods=8).min()
    low60 = low.rolling(60, min_periods=20).min()
    from_high20 = (close / high20.replace(0.0, np.nan) - 1.0) * 100.0
    from_high60 = (close / high60.replace(0.0, np.nan) - 1.0) * 100.0
    from_low20 = (close / low20.replace(0.0, np.nan) - 1.0) * 100.0
    from_low60 = (close / low60.replace(0.0, np.nan) - 1.0) * 100.0

    vol_base = volume.rolling(30, min_periods=12).median().shift(1).replace(0.0, np.nan)
    vol_ratio = (volume / vol_base).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    vol_mean_base = volume.rolling(30, min_periods=12).mean().shift(1).replace(0.0, np.nan)
    vol_ratio_mean = (volume / vol_mean_base).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    vol_recent = vol_ratio.rolling(5, min_periods=2).max().fillna(vol_ratio)

    taker_buy_ratio = (
        taker_buy_quote / quote_volume.replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.50)
    oi_roc8 = oi.pct_change(8).replace([np.inf, -np.inf], np.nan).fillna(0.0) * 100.0

    bar_range = (high - low).replace(0.0, np.nan)
    close_pos = ((close - low) / bar_range).fillna(0.5)
    upper_wick = ((high - pd.concat([open_, close], axis=1).max(axis=1)) / bar_range).fillna(0.0)
    body_pct = ((close - open_) / open_.replace(0.0, np.nan) * 100.0).fillna(0.0)

    ema8 = close.ewm(span=8, min_periods=5, adjust=False).mean()
    ema21 = close.ewm(span=21, min_periods=10, adjust=False).mean()
    ema55 = close.ewm(span=55, min_periods=25, adjust=False).mean()

    weak_close = (
        (close_pos <= max_close_pos)
        | (upper_wick >= min_upper_wick)
        | (body_pct <= 0.08)
        | (ret1 <= max_ret1)
    )
    activity_ok = (vol_recent >= min_activity) & (vol_recent <= max_activity)
    buyer_not_dominant = taker_buy_ratio <= max_taker_buy_ratio
    not_late_dump = from_low20 >= min_from_low20
    not_late_bounce = from_low20 <= max_from_low20
    near_recent_high = from_high60 >= -abs(max_from_high60)

    failed_bounce = (
        (from_low20 >= min_downtrend_bounce)
        & (from_low60 >= min_downtrend_bounce)
        & (ret60 <= max_downtrend_ret60)
        & (ret5 <= max_failed_bounce_ret5)
        & (medium_ret.fillna(0.0) <= max_failed_bounce_ret120)
        & (close <= ema8 * 1.004)
        & (ema8 <= ema21 * 1.012)
        & (close <= ema55 * 1.025)
    )
    confirmed_decline = pd.Series(True, index=df.index)
    if max_confirmed_decline_ret60 < 999.0:
        confirmed_decline &= ret60 <= max_confirmed_decline_ret60
    if max_confirmed_decline_from_high60 < 999.0:
        confirmed_decline &= from_high60 <= max_confirmed_decline_from_high60
    if min_confirmed_decline_vol_mean > 0.0:
        confirmed_decline &= vol_ratio_mean >= min_confirmed_decline_vol_mean
    pump_stall = (
        (ret15 >= min_pump_ret15)
        & (ret60 >= min_pump_ret60)
        & (ret5 <= max_ret5)
        & near_recent_high
        & (close <= high20 * 0.998)
    )

    rejection_score = (
        0.38 * _clip01((max_close_pos - close_pos) / max(max_close_pos, 0.1))
        + 0.32 * _clip01((upper_wick - min_upper_wick) / 0.45)
        + 0.18 * _clip01((max_ret1 - ret1.fillna(0.0)) / 5.0)
        + 0.12 * (close <= ema8 * 1.002).astype(float)
    )
    extension_score = pd.concat(
        [
            _clip01((from_low20.fillna(0.0) - min_from_low20) / 12.0),
            _clip01((from_low60.fillna(0.0) - min_downtrend_bounce) / 18.0),
            _clip01((ret15.fillna(0.0) + 1.5) / 12.0),
            _clip01((ret240.fillna(0.0) + 6.0) / 36.0),
        ],
        axis=1,
    ).max(axis=1)
    activity_score = _clip01((vol_recent - min_activity) / 4.0) * (1.0 - _clip01((vol_recent - max_activity) / 18.0))
    pressure_score = (
        0.55 * _clip01((max_taker_buy_ratio - taker_buy_ratio) / max(max_taker_buy_ratio - 0.42, 0.08))
        + 0.25 * _clip01(oi_roc8.abs() / 12.0)
        + 0.20 * _clip01(funding.fillna(0.0).clip(lower=0.0) / 0.18)
    )
    context_score = (
        0.58 * failed_bounce.astype(float)
        + 0.42 * pump_stall.astype(float)
    ).clip(upper=1.0)

    score = (
        0.30 * context_score
        + 0.24 * rejection_score
        + 0.20 * extension_score
        + 0.15 * activity_score
        + 0.11 * pressure_score
    )
    valid = (
        (failed_bounce | pump_stall)
        & confirmed_decline
        & weak_close
        & activity_ok
        & buyer_not_dominant
        & not_late_dump
        & not_late_bounce
        & (ret5 <= max_ret5)
        & (ret1 <= max_ret1)
        & (from_high20 >= -max(max_from_high60, 2.0))
    )

    out = pd.Series(0.0, index=df.index)
    out = out.mask(valid, -score.clip(lower=0.0, upper=max_output))
    return out.where(out.abs() >= min_output, 0.0).fillna(0.0)


def calculate_last_factor(df: pd.DataFrame) -> float:
    if df is None or len(df) < 90:
        return 0.0
    values = calculate_factor(df.tail(260))
    if values.empty:
        return 0.0
    try:
        return float(values.iloc[-1])
    except Exception:
        return 0.0
