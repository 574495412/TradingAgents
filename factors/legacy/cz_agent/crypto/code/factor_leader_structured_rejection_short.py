"""Leader-labeled structured rejection short factor.

This research factor turns recent high-yield meme lead-trader labels into two
separate, executable short structures:

* S1 micro exhaustion: fast extension near the local high, active volume, weak
  close or upper-wick rejection. It targets very short scalp exits.
* S4 volume-event fade: larger volume event, still near the high, with enough
  rejection to avoid blindly shorting every strong candle.
* S5 pullback confirmation: a weaker, already-rolled-over short burst near the
  local high. This keeps the leader-style "wait for the first failed push"
  structure without shorting a still-closing-strong candle.

Only as-of OHLCV/OI/funding data is used at runtime.
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

    min_output = _env_float("MEME_LEADER_STRUCTURED_REJECTION_MIN_OUTPUT", 0.28)
    max_output = _env_float("MEME_LEADER_STRUCTURED_REJECTION_MAX_OUTPUT", 0.92)
    micro_min_ret5 = _env_float("MEME_LEADER_STRUCTURED_REJECTION_MICRO_MIN_RET5_PCT", 2.6)
    micro_min_ret15 = _env_float("MEME_LEADER_STRUCTURED_REJECTION_MICRO_MIN_RET15_PCT", 4.8)
    micro_max_pullback_high20 = _env_float("MEME_LEADER_STRUCTURED_REJECTION_MICRO_MAX_PULLBACK_HIGH20_PCT", 3.2)
    micro_min_vol = _env_float("MEME_LEADER_STRUCTURED_REJECTION_MICRO_MIN_VOL_RATIO", 1.7)
    event_min_ret15 = _env_float("MEME_LEADER_STRUCTURED_REJECTION_EVENT_MIN_RET15_PCT", 6.5)
    event_min_vol = _env_float("MEME_LEADER_STRUCTURED_REJECTION_EVENT_MIN_VOL_RATIO", 4.0)
    event_max_pullback_high20 = _env_float("MEME_LEADER_STRUCTURED_REJECTION_EVENT_MAX_PULLBACK_HIGH20_PCT", 2.6)
    pullback_min_ret5 = _env_float("MEME_LEADER_STRUCTURED_REJECTION_PULLBACK_MIN_RET5_PCT", 0.8)
    pullback_min_ret15 = _env_float("MEME_LEADER_STRUCTURED_REJECTION_PULLBACK_MIN_RET15_PCT", 3.6)
    pullback_max_ret1 = _env_float("MEME_LEADER_STRUCTURED_REJECTION_PULLBACK_MAX_RET1_PCT", -0.15)
    pullback_max_close_pos = _env_float("MEME_LEADER_STRUCTURED_REJECTION_PULLBACK_MAX_CLOSE_POS", 0.52)
    pullback_min_vol = _env_float("MEME_LEADER_STRUCTURED_REJECTION_PULLBACK_MIN_VOL_RATIO", 3.0)
    max_current_vol = _env_float("MEME_LEADER_STRUCTURED_REJECTION_MAX_CURRENT_VOL_RATIO", 12.5)
    max_ret1 = _env_float("MEME_LEADER_STRUCTURED_REJECTION_MAX_RET1_PCT", 1.0)
    max_ret2 = _env_float("MEME_LEADER_STRUCTURED_REJECTION_MAX_RET2_PCT", 1.2)
    max_ret5 = _env_float("MEME_LEADER_STRUCTURED_REJECTION_MAX_RET5_PCT", 5.5)
    max_bounce20 = _env_float("MEME_LEADER_STRUCTURED_REJECTION_MAX_BOUNCE20_PCT", 7.0)
    max_bar_range = _env_float("MEME_LEADER_STRUCTURED_REJECTION_MAX_BAR_RANGE_PCT", 2.5)

    ret1 = close.pct_change(1) * 100.0
    ret2 = close.pct_change(2) * 100.0
    ret5 = close.pct_change(5) * 100.0
    ret15 = close.pct_change(15) * 100.0
    ret60 = close.pct_change(60) * 100.0

    high20 = high.rolling(20, min_periods=8).max()
    high60 = high.rolling(60, min_periods=20).max()
    low20 = low.rolling(20, min_periods=8).min()
    from_high20 = (close / high20.replace(0.0, np.nan) - 1.0) * 100.0
    from_high60 = (close / high60.replace(0.0, np.nan) - 1.0) * 100.0
    from_low20 = (close / low20.replace(0.0, np.nan) - 1.0) * 100.0

    vol_base = volume.rolling(30, min_periods=12).median().shift(1).replace(0.0, np.nan)
    vol_ratio = (volume / vol_base).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    vol_recent = vol_ratio.rolling(8, min_periods=3).max().fillna(vol_ratio)

    bar_range = (high - low).replace(0.0, np.nan)
    open_close_max = pd.Series(
        np.maximum(open_.to_numpy(dtype=float), close.to_numpy(dtype=float)),
        index=df.index,
    )
    upper_wick = (high - open_close_max) / bar_range
    close_pos = (close - low) / bar_range
    bar_range_pct = (high / low.replace(0.0, np.nan) - 1.0) * 100.0
    weak_close = ((close <= open_) | (ret1 <= 0.05) | (close_pos <= 0.55)).astype(float)
    failed_current_bar = (ret1 <= 0.15) | (close <= open_) | ((close_pos <= 0.58) & (upper_wick >= 0.12))

    oi_roc5 = oi.pct_change(5).replace([np.inf, -np.inf], np.nan).fillna(0.0) * 100.0
    funding_pos = funding.fillna(0.0).clip(lower=0.0)

    near_high_micro = 1.0 - _clip01((-from_high20.fillna(-999.0)) / max(micro_max_pullback_high20, 0.5))
    near_high_event = 1.0 - _clip01((-from_high20.fillna(-999.0)) / max(event_max_pullback_high20, 0.5))
    high60_support = 1.0 - _clip01((-from_high60.fillna(-999.0)) / 6.0)
    rejection = (
        0.38 * _clip01((upper_wick.fillna(0.0) - 0.18) / 0.42)
        + 0.34 * _clip01((0.68 - close_pos.fillna(0.5)) / 0.58)
        + 0.28 * weak_close
    )
    crowding = 0.55 * _clip01(oi_roc5 / 12.0) + 0.45 * _clip01(funding_pos / 0.20)

    micro_score = (
        0.28 * _clip01((ret5.fillna(0.0) - micro_min_ret5) / 8.0)
        + 0.20 * _clip01((ret15.fillna(0.0) - micro_min_ret15) / 12.0)
        + 0.20 * near_high_micro
        + 0.18 * rejection
        + 0.10 * _clip01((vol_recent - micro_min_vol) / 5.0)
        + 0.04 * crowding
    )
    event_score = (
        0.22 * _clip01((ret15.fillna(0.0) - event_min_ret15) / 14.0)
        + 0.16 * _clip01((ret60.fillna(0.0) - 2.0) / 22.0)
        + 0.18 * near_high_event
        + 0.16 * high60_support
        + 0.18 * _clip01((vol_recent - event_min_vol) / 12.0)
        + 0.10 * rejection
    )
    pullback_score = (
        0.22 * _clip01((ret5.fillna(0.0) - pullback_min_ret5) / 5.0)
        + 0.18 * _clip01((ret15.fillna(0.0) - pullback_min_ret15) / 8.0)
        + 0.20 * near_high_micro
        + 0.22 * _clip01((pullback_max_close_pos - close_pos.fillna(0.5)) / 0.42)
        + 0.12 * _clip01((vol_recent - pullback_min_vol) / 8.0)
        + 0.06 * crowding
    )

    micro_valid = (
        (ret5 >= micro_min_ret5)
        & (ret15 >= micro_min_ret15)
        & (ret1 <= max_ret1)
        & (ret2 <= max_ret2)
        & (ret5 <= max_ret5)
        & (from_high20 >= -abs(micro_max_pullback_high20))
        & (from_low20 <= max_bounce20)
        & (bar_range_pct <= max_bar_range)
        & (vol_ratio <= max_current_vol)
        & (vol_recent >= micro_min_vol)
        & failed_current_bar
    )
    event_valid = (
        (ret15 >= event_min_ret15)
        & (ret1 <= max_ret1)
        & (ret2 <= max_ret2)
        & (ret5 <= max_ret5)
        & (from_high20 >= -abs(event_max_pullback_high20))
        & (from_low20 <= max_bounce20)
        & (bar_range_pct <= max_bar_range)
        & (vol_ratio <= max_current_vol)
        & (vol_recent >= event_min_vol)
        & failed_current_bar
    )
    pullback_valid = (
        (ret5 >= pullback_min_ret5)
        & (ret15 >= pullback_min_ret15)
        & (ret1 <= pullback_max_ret1)
        & (ret2 <= max_ret2)
        & (ret5 <= max_ret5)
        & (from_high20 >= -abs(micro_max_pullback_high20))
        & (from_low20 <= max_bounce20)
        & (bar_range_pct <= max_bar_range)
        & (vol_ratio <= max_current_vol)
        & (vol_recent >= pullback_min_vol)
        & (close_pos <= pullback_max_close_pos)
    )

    score = pd.Series(
        np.maximum.reduce(
            [
                micro_score.where(micro_valid, 0.0).to_numpy(dtype=float),
                event_score.where(event_valid, 0.0).to_numpy(dtype=float),
                pullback_score.where(pullback_valid, 0.0).to_numpy(dtype=float),
            ]
        ),
        index=df.index,
    )
    out = -score.clip(lower=0.0, upper=max_output)
    out = out.where(out.abs() >= min_output, 0.0)
    return out.fillna(0.0)


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

    min_output = _env_float("MEME_LEADER_STRUCTURED_REJECTION_MIN_OUTPUT", 0.28)
    max_output = _env_float("MEME_LEADER_STRUCTURED_REJECTION_MAX_OUTPUT", 0.92)
    micro_min_ret5 = _env_float("MEME_LEADER_STRUCTURED_REJECTION_MICRO_MIN_RET5_PCT", 2.6)
    micro_min_ret15 = _env_float("MEME_LEADER_STRUCTURED_REJECTION_MICRO_MIN_RET15_PCT", 4.8)
    micro_max_pullback_high20 = _env_float("MEME_LEADER_STRUCTURED_REJECTION_MICRO_MAX_PULLBACK_HIGH20_PCT", 3.2)
    micro_min_vol = _env_float("MEME_LEADER_STRUCTURED_REJECTION_MICRO_MIN_VOL_RATIO", 1.7)
    event_min_ret15 = _env_float("MEME_LEADER_STRUCTURED_REJECTION_EVENT_MIN_RET15_PCT", 6.5)
    event_min_vol = _env_float("MEME_LEADER_STRUCTURED_REJECTION_EVENT_MIN_VOL_RATIO", 4.0)
    event_max_pullback_high20 = _env_float("MEME_LEADER_STRUCTURED_REJECTION_EVENT_MAX_PULLBACK_HIGH20_PCT", 2.6)
    pullback_min_ret5 = _env_float("MEME_LEADER_STRUCTURED_REJECTION_PULLBACK_MIN_RET5_PCT", 0.8)
    pullback_min_ret15 = _env_float("MEME_LEADER_STRUCTURED_REJECTION_PULLBACK_MIN_RET15_PCT", 3.6)
    pullback_max_ret1 = _env_float("MEME_LEADER_STRUCTURED_REJECTION_PULLBACK_MAX_RET1_PCT", -0.15)
    pullback_max_close_pos = _env_float("MEME_LEADER_STRUCTURED_REJECTION_PULLBACK_MAX_CLOSE_POS", 0.52)
    pullback_min_vol = _env_float("MEME_LEADER_STRUCTURED_REJECTION_PULLBACK_MIN_VOL_RATIO", 3.0)
    max_current_vol = _env_float("MEME_LEADER_STRUCTURED_REJECTION_MAX_CURRENT_VOL_RATIO", 12.5)
    max_ret1 = _env_float("MEME_LEADER_STRUCTURED_REJECTION_MAX_RET1_PCT", 1.0)
    max_ret2 = _env_float("MEME_LEADER_STRUCTURED_REJECTION_MAX_RET2_PCT", 1.2)
    max_ret5 = _env_float("MEME_LEADER_STRUCTURED_REJECTION_MAX_RET5_PCT", 5.5)
    max_bounce20 = _env_float("MEME_LEADER_STRUCTURED_REJECTION_MAX_BOUNCE20_PCT", 7.0)
    max_bar_range = _env_float("MEME_LEADER_STRUCTURED_REJECTION_MAX_BAR_RANGE_PCT", 2.5)

    close_np = close.to_numpy(dtype=float)
    open_last = float(open_.iloc[-1])
    high_last = float(high.iloc[-1])
    low_last = float(low.iloc[-1])
    close_last = float(close.iloc[-1])
    funding_last = float(funding.iloc[-1])

    ret1 = _pct_last(close_np, 1)
    ret2 = _pct_last(close_np, 2)
    ret5 = _pct_last(close_np, 5)
    ret15 = _pct_last(close_np, 15)
    ret60 = _pct_last(close_np, 60)

    high20 = float(high.iloc[-20:].max())
    high60 = float(high.iloc[-60:].max())
    low20 = float(low.iloc[-20:].min())
    from_high20 = (close_last / high20 - 1.0) * 100.0 if high20 > 0 else float("nan")
    from_high60 = (close_last / high60 - 1.0) * 100.0 if high60 > 0 else float("nan")
    from_low20 = (close_last / low20 - 1.0) * 100.0 if low20 > 0 else float("nan")

    vol_base = volume.rolling(30, min_periods=12).median().shift(1).replace(0.0, np.nan)
    vol_ratio_series = (volume / vol_base).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    vol_ratio = float(vol_ratio_series.iloc[-1])
    vol_recent = float(vol_ratio_series.rolling(8, min_periods=3).max().fillna(vol_ratio_series).iloc[-1])

    bar_range = high_last - low_last
    if not np.isfinite(bar_range) or bar_range == 0:
        return 0.0
    upper_wick = (high_last - max(open_last, close_last)) / bar_range
    close_pos = (close_last - low_last) / bar_range
    bar_range_pct = (high_last / low_last - 1.0) * 100.0 if low_last > 0 else float("nan")
    weak_close = float(close_last <= open_last or ret1 <= 0.05 or close_pos <= 0.55)
    failed_current_bar = ret1 <= 0.15 or close_last <= open_last or (close_pos <= 0.58 and upper_wick >= 0.12)

    oi_prev = float(oi.iloc[-6])
    oi_roc5 = (float(oi.iloc[-1]) - oi_prev) / oi_prev * 100.0 if oi_prev > 0 else 0.0
    funding_pos = max(0.0, funding_last) if np.isfinite(funding_last) else 0.0

    def clip01(value: float) -> float:
        if not np.isfinite(value):
            return 0.0
        return max(0.0, min(1.0, float(value)))

    near_high_micro = 1.0 - clip01((-(from_high20 if np.isfinite(from_high20) else -999.0)) / max(micro_max_pullback_high20, 0.5))
    near_high_event = 1.0 - clip01((-(from_high20 if np.isfinite(from_high20) else -999.0)) / max(event_max_pullback_high20, 0.5))
    high60_support = 1.0 - clip01((-(from_high60 if np.isfinite(from_high60) else -999.0)) / 6.0)
    rejection = (
        0.38 * clip01((upper_wick - 0.18) / 0.42)
        + 0.34 * clip01((0.68 - (close_pos if np.isfinite(close_pos) else 0.5)) / 0.58)
        + 0.28 * weak_close
    )
    crowding = 0.55 * clip01(oi_roc5 / 12.0) + 0.45 * clip01(funding_pos / 0.20)

    ret5_fill = 0.0 if not np.isfinite(ret5) else ret5
    ret15_fill = 0.0 if not np.isfinite(ret15) else ret15
    ret60_fill = 0.0 if not np.isfinite(ret60) else ret60

    micro_score = (
        0.28 * clip01((ret5_fill - micro_min_ret5) / 8.0)
        + 0.20 * clip01((ret15_fill - micro_min_ret15) / 12.0)
        + 0.20 * near_high_micro
        + 0.18 * rejection
        + 0.10 * clip01((vol_recent - micro_min_vol) / 5.0)
        + 0.04 * crowding
    )
    event_score = (
        0.22 * clip01((ret15_fill - event_min_ret15) / 14.0)
        + 0.16 * clip01((ret60_fill - 2.0) / 22.0)
        + 0.18 * near_high_event
        + 0.16 * high60_support
        + 0.18 * clip01((vol_recent - event_min_vol) / 12.0)
        + 0.10 * rejection
    )
    pullback_score = (
        0.22 * clip01((ret5_fill - pullback_min_ret5) / 5.0)
        + 0.18 * clip01((ret15_fill - pullback_min_ret15) / 8.0)
        + 0.20 * near_high_micro
        + 0.22 * clip01((pullback_max_close_pos - (close_pos if np.isfinite(close_pos) else 0.5)) / 0.42)
        + 0.12 * clip01((vol_recent - pullback_min_vol) / 8.0)
        + 0.06 * crowding
    )

    micro_valid = (
        ret5 >= micro_min_ret5
        and ret15 >= micro_min_ret15
        and ret1 <= max_ret1
        and ret2 <= max_ret2
        and ret5 <= max_ret5
        and from_high20 >= -abs(micro_max_pullback_high20)
        and from_low20 <= max_bounce20
        and bar_range_pct <= max_bar_range
        and vol_ratio <= max_current_vol
        and vol_recent >= micro_min_vol
        and failed_current_bar
    )
    event_valid = (
        ret15 >= event_min_ret15
        and ret1 <= max_ret1
        and ret2 <= max_ret2
        and ret5 <= max_ret5
        and from_high20 >= -abs(event_max_pullback_high20)
        and from_low20 <= max_bounce20
        and bar_range_pct <= max_bar_range
        and vol_ratio <= max_current_vol
        and vol_recent >= event_min_vol
        and failed_current_bar
    )
    pullback_valid = (
        ret5 >= pullback_min_ret5
        and ret15 >= pullback_min_ret15
        and ret1 <= pullback_max_ret1
        and ret2 <= max_ret2
        and ret5 <= max_ret5
        and from_high20 >= -abs(micro_max_pullback_high20)
        and from_low20 <= max_bounce20
        and bar_range_pct <= max_bar_range
        and vol_ratio <= max_current_vol
        and vol_recent >= pullback_min_vol
        and close_pos <= pullback_max_close_pos
    )

    score = max(
        micro_score if micro_valid else 0.0,
        event_score if event_valid else 0.0,
        pullback_score if pullback_valid else 0.0,
    )
    value = -min(max_output, max(0.0, score))
    return value if abs(value) >= min_output else 0.0
