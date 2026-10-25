"""Leader-inspired high-leverage ultra-short scalp factor.

High-leverage copy-trader samples with very short holding periods split into
two mirror structures:

* short: fast pump with volume expansion, then scalp the local exhaustion;
* long: fast dump with volume expansion, then scalp the local rebound.

The factor is intentionally research-only by default.  It uses only historical
OHLCV/OI/funding available at the signal bar.
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
    if name in df:
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

    min_output = _env_float("MEME_LEADER_ULTRA_SCALP_MIN_OUTPUT", 0.34)
    max_output = _env_float("MEME_LEADER_ULTRA_SCALP_MAX_OUTPUT", 0.90)
    min_vol_ratio = _env_float("MEME_LEADER_ULTRA_SCALP_MIN_VOL_RATIO", 2.0)

    short_min_ret3 = _env_float("MEME_LEADER_ULTRA_SCALP_SHORT_MIN_RET3_PCT", 2.0)
    short_min_ret5 = _env_float("MEME_LEADER_ULTRA_SCALP_SHORT_MIN_RET5_PCT", 3.0)
    short_min_ret15 = _env_float("MEME_LEADER_ULTRA_SCALP_SHORT_MIN_RET15_PCT", 5.6)
    short_min_ret60 = _env_float("MEME_LEADER_ULTRA_SCALP_SHORT_MIN_RET60_PCT", 4.0)
    short_max_pullback = _env_float("MEME_LEADER_ULTRA_SCALP_SHORT_MAX_PULLBACK_HIGH20_PCT", 8.0)
    short_min_bounce = _env_float("MEME_LEADER_ULTRA_SCALP_SHORT_MIN_BOUNCE20_PCT", 2.0)
    short_max_bounce = _env_float("MEME_LEADER_ULTRA_SCALP_SHORT_MAX_BOUNCE20_PCT", 10.0)
    short_high_vol_ratio = _env_float("MEME_LEADER_ULTRA_SCALP_SHORT_HIGH_VOL_RATIO", 6.0)
    short_max_close_pos_high_vol = _env_float("MEME_LEADER_ULTRA_SCALP_SHORT_MAX_CLOSE_POS_HIGH_VOL", 0.62)

    long_min_abs_ret3 = _env_float("MEME_LEADER_ULTRA_SCALP_LONG_MIN_ABS_RET3_PCT", 2.0)
    long_min_abs_ret5 = _env_float("MEME_LEADER_ULTRA_SCALP_LONG_MIN_ABS_RET5_PCT", 2.0)
    long_min_abs_ret15 = _env_float("MEME_LEADER_ULTRA_SCALP_LONG_MIN_ABS_RET15_PCT", 3.0)
    long_min_drop_from_high = _env_float("MEME_LEADER_ULTRA_SCALP_LONG_MIN_DROP_HIGH20_PCT", 2.0)
    long_max_bounce = _env_float("MEME_LEADER_ULTRA_SCALP_LONG_MAX_BOUNCE20_PCT", 12.0)

    ret1 = close.pct_change(1) * 100.0
    ret3 = close.pct_change(3) * 100.0
    ret5 = close.pct_change(5) * 100.0
    ret15 = close.pct_change(15) * 100.0
    ret60 = close.pct_change(60) * 100.0

    high20 = high.rolling(20, min_periods=8).max()
    low20 = low.rolling(20, min_periods=8).min()
    from_high20 = (close / high20.replace(0.0, np.nan) - 1.0) * 100.0
    from_low20 = (close / low20.replace(0.0, np.nan) - 1.0) * 100.0

    vol_base = volume.rolling(30, min_periods=12).median().shift(1).replace(0.0, np.nan)
    vol_ratio = (volume / vol_base).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    oi_roc3 = oi.pct_change(3).replace([np.inf, -np.inf], np.nan).fillna(0.0) * 100.0

    bar_range = (high - low).replace(0.0, np.nan)
    open_np = open_.to_numpy(dtype=float)
    close_np = close.to_numpy(dtype=float)
    open_close_max = pd.Series(np.maximum(open_np, close_np), index=df.index)
    open_close_min = pd.Series(np.minimum(open_np, close_np), index=df.index)
    upper_wick = (high - open_close_max) / bar_range
    lower_wick = (open_close_min - low) / bar_range
    close_pos = (close - low) / bar_range

    ema5 = close.ewm(span=5, min_periods=3, adjust=False).mean()
    ema13 = close.ewm(span=13, min_periods=6, adjust=False).mean()
    ema_bias = (ema5 / ema13.replace(0.0, np.nan) - 1.0) * 100.0

    vol_score = _clip01((vol_ratio - min_vol_ratio) / 6.0)
    oi_score = _clip01(oi_roc3.abs() / 10.0)

    short_extension = pd.Series(
        np.maximum.reduce(
            [
                ((ret3.fillna(0.0) - short_min_ret3) / max(short_min_ret3 * 4.0, 1.0)).to_numpy(dtype=float),
                ((ret5.fillna(0.0) - short_min_ret5) / max(short_min_ret5 * 4.0, 1.0)).to_numpy(dtype=float),
                ((ret15.fillna(0.0) - short_min_ret15) / max(short_min_ret15 * 4.0, 1.0)).to_numpy(dtype=float),
            ]
        ),
        index=df.index,
    )
    short_near_high = 1.0 - _clip01((-from_high20.fillna(-999.0) - 0.2) / max(short_max_pullback - 0.2, 0.5))
    short_bounce = _clip01((from_low20.fillna(0.0) - short_min_bounce) / 12.0)
    short_rejection = (
        0.45 * _clip01((upper_wick.fillna(0.0) - 0.08) / 0.52)
        + 0.35 * _clip01((0.78 - close_pos.fillna(0.5)) / 0.78)
        + 0.20 * (ret1.fillna(0.0) <= 0.35).astype(float)
    )
    short_fuel = 0.70 * vol_score + 0.20 * oi_score + 0.10 * _clip01(funding.fillna(0.0) / 0.20)
    short_score = (
        0.32 * _clip01(short_extension)
        + 0.22 * short_near_high
        + 0.16 * short_bounce
        + 0.18 * short_rejection
        + 0.12 * short_fuel
    )
    short_valid = (
        (vol_ratio >= min_vol_ratio)
        & (ret5 >= short_min_ret5)
        & ((ret3 >= short_min_ret3) | (ret15 >= short_min_ret15))
        & (from_high20 >= -abs(short_max_pullback))
        & (from_low20 >= short_min_bounce)
        & (from_low20 <= abs(short_max_bounce))
        & (ret60 >= short_min_ret60)
        & (ema_bias >= -0.25)
        & ~((vol_ratio >= short_high_vol_ratio) & (close_pos.fillna(0.5) >= short_max_close_pos_high_vol))
    )

    long_extension = pd.Series(
        np.maximum.reduce(
            [
                (((-ret3.fillna(0.0)) - long_min_abs_ret3) / max(long_min_abs_ret3 * 4.0, 1.0)).to_numpy(dtype=float),
                (((-ret5.fillna(0.0)) - long_min_abs_ret5) / max(long_min_abs_ret5 * 4.0, 1.0)).to_numpy(dtype=float),
                (((-ret15.fillna(0.0)) - long_min_abs_ret15) / max(long_min_abs_ret15 * 4.0, 1.0)).to_numpy(dtype=float),
            ]
        ),
        index=df.index,
    )
    long_drop = _clip01((-from_high20.fillna(0.0) - long_min_drop_from_high) / 12.0)
    long_rebound_room = 1.0 - _clip01((from_low20.fillna(0.0) - long_max_bounce) / max(long_max_bounce, 1.0))
    long_acceptance = (
        0.44 * _clip01((lower_wick.fillna(0.0) - 0.08) / 0.50)
        + 0.34 * _clip01((0.78 - close_pos.fillna(0.5)) / 0.78)
        + 0.22 * (ret1.fillna(0.0) >= -3.0).astype(float)
    )
    long_fuel = 0.75 * vol_score + 0.15 * oi_score + 0.10 * _clip01((-funding.fillna(0.0)) / 0.20)
    long_score = (
        0.32 * _clip01(long_extension)
        + 0.22 * long_drop
        + 0.14 * long_rebound_room
        + 0.20 * long_acceptance
        + 0.12 * long_fuel
    )
    long_valid = (
        (vol_ratio >= min_vol_ratio)
        & (
            (ret3 <= -long_min_abs_ret3)
            | (ret5 <= -long_min_abs_ret5)
            | (ret15 <= -long_min_abs_ret15)
        )
        & (from_high20 <= -abs(long_min_drop_from_high))
        & (from_low20 <= long_max_bounce)
        & ((lower_wick >= 0.10) | (close_pos <= 0.75))
        & (ret60 <= 8.0)
    )

    out = pd.Series(0.0, index=df.index)
    out = out.mask(long_valid & (long_score >= short_score), long_score.clip(lower=0.0, upper=max_output))
    out = out.mask(short_valid & (short_score > long_score), -short_score.clip(lower=0.0, upper=max_output))
    return out.where(out.abs() >= min_output, 0.0).fillna(0.0)


def _pct_last(values: np.ndarray, periods: int) -> float:
    if len(values) <= periods:
        return float("nan")
    prev = float(values[-periods - 1])
    if prev <= 0:
        return float("nan")
    return (float(values[-1]) - prev) / prev * 100.0


def calculate_last_factor(df: pd.DataFrame) -> float:
    """Return the exact last-row factor value without building full output."""

    if df is None or len(df) < 90:
        return 0.0

    open_ = _safe_series(df, "open")
    high = _safe_series(df, "high")
    low = _safe_series(df, "low")
    close = _safe_series(df, "close")
    volume = _safe_series(df, "volume", 1.0)
    oi = _safe_series(df, "open_interest")
    funding = _safe_series(df, "funding_rate")

    min_output = _env_float("MEME_LEADER_ULTRA_SCALP_MIN_OUTPUT", 0.34)
    max_output = _env_float("MEME_LEADER_ULTRA_SCALP_MAX_OUTPUT", 0.90)
    min_vol_ratio = _env_float("MEME_LEADER_ULTRA_SCALP_MIN_VOL_RATIO", 2.0)

    short_min_ret3 = _env_float("MEME_LEADER_ULTRA_SCALP_SHORT_MIN_RET3_PCT", 2.0)
    short_min_ret5 = _env_float("MEME_LEADER_ULTRA_SCALP_SHORT_MIN_RET5_PCT", 3.0)
    short_min_ret15 = _env_float("MEME_LEADER_ULTRA_SCALP_SHORT_MIN_RET15_PCT", 5.6)
    short_min_ret60 = _env_float("MEME_LEADER_ULTRA_SCALP_SHORT_MIN_RET60_PCT", 4.0)
    short_max_pullback = _env_float("MEME_LEADER_ULTRA_SCALP_SHORT_MAX_PULLBACK_HIGH20_PCT", 8.0)
    short_min_bounce = _env_float("MEME_LEADER_ULTRA_SCALP_SHORT_MIN_BOUNCE20_PCT", 2.0)
    short_max_bounce = _env_float("MEME_LEADER_ULTRA_SCALP_SHORT_MAX_BOUNCE20_PCT", 10.0)
    short_high_vol_ratio = _env_float("MEME_LEADER_ULTRA_SCALP_SHORT_HIGH_VOL_RATIO", 6.0)
    short_max_close_pos_high_vol = _env_float("MEME_LEADER_ULTRA_SCALP_SHORT_MAX_CLOSE_POS_HIGH_VOL", 0.62)

    long_min_abs_ret3 = _env_float("MEME_LEADER_ULTRA_SCALP_LONG_MIN_ABS_RET3_PCT", 2.0)
    long_min_abs_ret5 = _env_float("MEME_LEADER_ULTRA_SCALP_LONG_MIN_ABS_RET5_PCT", 2.0)
    long_min_abs_ret15 = _env_float("MEME_LEADER_ULTRA_SCALP_LONG_MIN_ABS_RET15_PCT", 3.0)
    long_min_drop_from_high = _env_float("MEME_LEADER_ULTRA_SCALP_LONG_MIN_DROP_HIGH20_PCT", 2.0)
    long_max_bounce = _env_float("MEME_LEADER_ULTRA_SCALP_LONG_MAX_BOUNCE20_PCT", 12.0)

    close_np = close.to_numpy(dtype=float)
    open_last = float(open_.iloc[-1])
    high_last = float(high.iloc[-1])
    low_last = float(low.iloc[-1])
    close_last = float(close.iloc[-1])
    volume_last = float(volume.iloc[-1])
    funding_last = float(funding.iloc[-1])

    ret1 = _pct_last(close_np, 1)
    ret3 = _pct_last(close_np, 3)
    ret5 = _pct_last(close_np, 5)
    ret15 = _pct_last(close_np, 15)
    ret60 = _pct_last(close_np, 60)

    high20 = float(high.iloc[-20:].max())
    low20 = float(low.iloc[-20:].min())
    from_high20 = (close_last / high20 - 1.0) * 100.0 if high20 > 0 else float("nan")
    from_low20 = (close_last / low20 - 1.0) * 100.0 if low20 > 0 else float("nan")

    vol_base = float(volume.iloc[-31:-1].median()) if len(volume) >= 31 else float("nan")
    vol_ratio = volume_last / vol_base if vol_base > 0 else 1.0
    oi_prev = float(oi.iloc[-4])
    oi_roc3 = (float(oi.iloc[-1]) - oi_prev) / oi_prev * 100.0 if oi_prev > 0 else 0.0

    bar_range = high_last - low_last
    if not np.isfinite(bar_range) or bar_range == 0:
        return 0.0
    upper_wick = (high_last - max(open_last, close_last)) / bar_range
    lower_wick = (min(open_last, close_last) - low_last) / bar_range
    close_pos = (close_last - low_last) / bar_range

    ema5 = float(close.ewm(span=5, min_periods=3, adjust=False).mean().iloc[-1])
    ema13 = float(close.ewm(span=13, min_periods=6, adjust=False).mean().iloc[-1])
    ema_bias = (ema5 / ema13 - 1.0) * 100.0 if ema13 > 0 else float("nan")

    def clip01(value: float) -> float:
        if not np.isfinite(value):
            return 0.0
        return max(0.0, min(1.0, float(value)))

    vol_score = clip01((vol_ratio - min_vol_ratio) / 6.0)
    oi_score = clip01(abs(oi_roc3) / 10.0)

    short_extension = max(
        ((0.0 if not np.isfinite(ret3) else ret3) - short_min_ret3) / max(short_min_ret3 * 4.0, 1.0),
        ((0.0 if not np.isfinite(ret5) else ret5) - short_min_ret5) / max(short_min_ret5 * 4.0, 1.0),
        ((0.0 if not np.isfinite(ret15) else ret15) - short_min_ret15) / max(short_min_ret15 * 4.0, 1.0),
    )
    short_near_high = 1.0 - clip01((-(from_high20 if np.isfinite(from_high20) else -999.0) - 0.2) / max(short_max_pullback - 0.2, 0.5))
    short_bounce = clip01(((from_low20 if np.isfinite(from_low20) else 0.0) - short_min_bounce) / 12.0)
    short_rejection = (
        0.45 * clip01((upper_wick - 0.08) / 0.52)
        + 0.35 * clip01((0.78 - (close_pos if np.isfinite(close_pos) else 0.5)) / 0.78)
        + 0.20 * float((0.0 if not np.isfinite(ret1) else ret1) <= 0.35)
    )
    short_fuel = 0.70 * vol_score + 0.20 * oi_score + 0.10 * clip01(funding_last / 0.20)
    short_score = (
        0.32 * clip01(short_extension)
        + 0.22 * short_near_high
        + 0.16 * short_bounce
        + 0.18 * short_rejection
        + 0.12 * short_fuel
    )
    short_valid = (
        vol_ratio >= min_vol_ratio
        and ret5 >= short_min_ret5
        and (ret3 >= short_min_ret3 or ret15 >= short_min_ret15)
        and from_high20 >= -abs(short_max_pullback)
        and from_low20 >= short_min_bounce
        and from_low20 <= abs(short_max_bounce)
        and ret60 >= short_min_ret60
        and ema_bias >= -0.25
        and not (vol_ratio >= short_high_vol_ratio and (close_pos if np.isfinite(close_pos) else 0.5) >= short_max_close_pos_high_vol)
    )

    long_extension = max(
        ((-(0.0 if not np.isfinite(ret3) else ret3)) - long_min_abs_ret3) / max(long_min_abs_ret3 * 4.0, 1.0),
        ((-(0.0 if not np.isfinite(ret5) else ret5)) - long_min_abs_ret5) / max(long_min_abs_ret5 * 4.0, 1.0),
        ((-(0.0 if not np.isfinite(ret15) else ret15)) - long_min_abs_ret15) / max(long_min_abs_ret15 * 4.0, 1.0),
    )
    long_drop = clip01((-(from_high20 if np.isfinite(from_high20) else 0.0) - long_min_drop_from_high) / 12.0)
    long_rebound_room = 1.0 - clip01(((from_low20 if np.isfinite(from_low20) else 0.0) - long_max_bounce) / max(long_max_bounce, 1.0))
    long_acceptance = (
        0.44 * clip01((lower_wick - 0.08) / 0.50)
        + 0.34 * clip01((0.78 - (close_pos if np.isfinite(close_pos) else 0.5)) / 0.78)
        + 0.22 * float((0.0 if not np.isfinite(ret1) else ret1) >= -3.0)
    )
    long_fuel = 0.75 * vol_score + 0.15 * oi_score + 0.10 * clip01((-funding_last) / 0.20)
    long_score = (
        0.32 * clip01(long_extension)
        + 0.22 * long_drop
        + 0.14 * long_rebound_room
        + 0.20 * long_acceptance
        + 0.12 * long_fuel
    )
    long_valid = (
        vol_ratio >= min_vol_ratio
        and (ret3 <= -long_min_abs_ret3 or ret5 <= -long_min_abs_ret5 or ret15 <= -long_min_abs_ret15)
        and from_high20 <= -abs(long_min_drop_from_high)
        and from_low20 <= long_max_bounce
        and (lower_wick >= 0.10 or close_pos <= 0.75)
        and ret60 <= 8.0
    )

    value = 0.0
    if long_valid and long_score >= short_score:
        value = min(max_output, max(0.0, long_score))
    if short_valid and short_score > long_score:
        value = -min(max_output, max(0.0, short_score))
    return value if abs(value) >= min_output else 0.0
