"""High-win leader volume-pullback long factor.

High-frequency, high-win leader samples show a long structure that is neither
late breakout chasing nor deep knife catching: volume stays accepted, price has
pulled back from the local high, and the latest bars have stopped deteriorating.
This factor only uses as-of OHLCV/order-flow fields.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cz.trading.crypto.meme.profile_config import profile_bool, profile_float, profile_get


def _env_float(name: str, default: float) -> float:
    return profile_float(name, default)


def _safe_series(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").ffill().fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _clip01(value: pd.Series) -> pd.Series:
    return value.clip(lower=0.0, upper=1.0)


def _frame_symbol(df: pd.DataFrame) -> str:
    raw = df.attrs.get("symbol") or df.attrs.get("market_symbol") or ""
    return str(raw or "").strip().upper().replace("/", "")


def _leader_long_scores(symbol: str, index: pd.Index) -> pd.Series:
    if not profile_bool("MEME_LEADER_HIGH_WIN_VOLUME_PULLBACK_LONG_REQUIRE_EDGE", False):
        return pd.Series(100.0, index=index, dtype=float)
    if not symbol:
        return pd.Series(0.0, index=index, dtype=float)
    try:
        from cz.trading.crypto.meme.leader_edge import leader_symbol_edge, leader_symbol_edge_from_file
    except Exception:
        return pd.Series(0.0, index=index, dtype=float)

    edge = None
    edge_file = str(profile_get("MEME_LEADER_HIGH_WIN_VOLUME_PULLBACK_LONG_EDGE_FILE", "") or "").strip()
    min_score = _env_float("MEME_LEADER_HIGH_WIN_VOLUME_PULLBACK_LONG_MIN_EDGE_SCORE", 76.0)
    min_samples = int(_env_float("MEME_LEADER_HIGH_WIN_VOLUME_PULLBACK_LONG_MIN_EDGE_SAMPLES", 2.0))
    min_win_rate = _env_float("MEME_LEADER_HIGH_WIN_VOLUME_PULLBACK_LONG_MIN_EDGE_WIN_RATE", 90.0)
    if edge_file:
        try:
            edge = leader_symbol_edge_from_file(
                symbol,
                edge_file,
                min_score=min_score,
                min_samples=min_samples,
                min_win_rate=min_win_rate,
                direction_mode="label",
            )
        except Exception:
            edge = None
    else:
        try:
            edge = leader_symbol_edge(symbol)
        except Exception:
            edge = None
    if edge is None or getattr(edge, "direction", "") != "LONG":
        return pd.Series(0.0, index=index, dtype=float)
    return pd.Series(float(getattr(edge, "score", 0.0)), index=index, dtype=float)


def calculate_factor(df: pd.DataFrame) -> pd.Series:
    if df is None or len(df) < 90:
        return pd.Series(0.0, index=getattr(df, "index", None))

    symbol = _frame_symbol(df)
    leader_score = _leader_long_scores(symbol, df.index)
    min_edge_score = _env_float("MEME_LEADER_HIGH_WIN_VOLUME_PULLBACK_LONG_MIN_EDGE_SCORE", 76.0)
    edge_valid = leader_score >= min_edge_score
    if not bool(edge_valid.any()):
        return pd.Series(0.0, index=df.index)

    open_ = _safe_series(df, "open")
    high = _safe_series(df, "high")
    low = _safe_series(df, "low")
    close = _safe_series(df, "close")
    volume = _safe_series(df, "volume", 1.0)
    quote_volume = _safe_series(df, "quote_volume")
    taker_buy_quote = _safe_series(df, "taker_buy_quote")
    oi = _safe_series(df, "open_interest")
    funding = _safe_series(df, "funding_rate")

    min_output = _env_float("MEME_LEADER_HIGH_WIN_VOLUME_PULLBACK_LONG_MIN_OUTPUT", 0.24)
    max_output = _env_float("MEME_LEADER_HIGH_WIN_VOLUME_PULLBACK_LONG_MAX_OUTPUT", 0.82)
    min_mean_vol_ratio = _env_float("MEME_LEADER_HIGH_WIN_VOLUME_PULLBACK_LONG_MIN_MEAN_VOL_RATIO", 0.84)
    min_med_vol_ratio = _env_float("MEME_LEADER_HIGH_WIN_VOLUME_PULLBACK_LONG_MIN_MED_VOL_RATIO", 0.72)
    max_vol_ratio = _env_float("MEME_LEADER_HIGH_WIN_VOLUME_PULLBACK_LONG_MAX_VOL_RATIO", 7.5)
    min_from_high20 = _env_float("MEME_LEADER_HIGH_WIN_VOLUME_PULLBACK_LONG_MIN_FROM_HIGH20_PCT", -8.0)
    max_from_high20 = _env_float("MEME_LEADER_HIGH_WIN_VOLUME_PULLBACK_LONG_MAX_FROM_HIGH20_PCT", -0.49)
    min_from_low20 = _env_float("MEME_LEADER_HIGH_WIN_VOLUME_PULLBACK_LONG_MIN_FROM_LOW20_PCT", 0.48)
    max_from_low20 = _env_float("MEME_LEADER_HIGH_WIN_VOLUME_PULLBACK_LONG_MAX_FROM_LOW20_PCT", 999.0)
    min_ret1 = _env_float("MEME_LEADER_HIGH_WIN_VOLUME_PULLBACK_LONG_MIN_RET1_PCT", -0.50)
    min_ret5 = _env_float("MEME_LEADER_HIGH_WIN_VOLUME_PULLBACK_LONG_MIN_RET5_PCT", -0.77)
    max_ret5 = _env_float("MEME_LEADER_HIGH_WIN_VOLUME_PULLBACK_LONG_MAX_RET5_PCT", 5.5)
    min_ret15 = _env_float("MEME_LEADER_HIGH_WIN_VOLUME_PULLBACK_LONG_MIN_RET15_PCT", -6.5)
    max_ret15 = _env_float("MEME_LEADER_HIGH_WIN_VOLUME_PULLBACK_LONG_MAX_RET15_PCT", 8.0)
    min_ret60 = _env_float("MEME_LEADER_HIGH_WIN_VOLUME_PULLBACK_LONG_MIN_RET60_PCT", -10.0)
    min_taker_buy_ratio = _env_float("MEME_LEADER_HIGH_WIN_VOLUME_PULLBACK_LONG_MIN_TAKER_BUY_RATIO", 0.43)
    min_close_pos = _env_float("MEME_LEADER_HIGH_WIN_VOLUME_PULLBACK_LONG_MIN_CLOSE_POS", 0.46)
    max_upper_wick = _env_float("MEME_LEADER_HIGH_WIN_VOLUME_PULLBACK_LONG_MAX_UPPER_WICK", 0.56)
    max_bar_range = _env_float("MEME_LEADER_HIGH_WIN_VOLUME_PULLBACK_LONG_MAX_BAR_RANGE_PCT", 9.0)

    ret1 = close.pct_change(1) * 100.0
    ret3 = close.pct_change(3) * 100.0
    ret5 = close.pct_change(5) * 100.0
    ret15 = close.pct_change(15) * 100.0
    ret60 = close.pct_change(60) * 100.0

    high20 = high.rolling(20, min_periods=8).max()
    low20 = low.rolling(20, min_periods=8).min()
    high60 = high.rolling(60, min_periods=20).max()
    low60 = low.rolling(60, min_periods=20).min()
    from_high20 = (close / high20.replace(0.0, np.nan) - 1.0) * 100.0
    from_high60 = (close / high60.replace(0.0, np.nan) - 1.0) * 100.0
    from_low20 = (close / low20.replace(0.0, np.nan) - 1.0) * 100.0
    from_low60 = (close / low60.replace(0.0, np.nan) - 1.0) * 100.0

    vol_mean_base = volume.rolling(30, min_periods=12).mean().shift(1).replace(0.0, np.nan)
    vol_med_base = volume.rolling(30, min_periods=12).median().shift(1).replace(0.0, np.nan)
    vol_ratio_mean = (volume / vol_mean_base).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    vol_ratio_med = (volume / vol_med_base).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    taker_buy_ratio = (
        taker_buy_quote / quote_volume.replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.50)
    oi_roc20 = oi.pct_change(20).replace([np.inf, -np.inf], np.nan).fillna(0.0) * 100.0

    bar_range = (high - low).replace(0.0, np.nan)
    close_pos = (close - low) / bar_range
    upper_wick = (high - pd.concat([open_, close], axis=1).max(axis=1)) / bar_range
    lower_wick = (pd.concat([open_, close], axis=1).min(axis=1) - low) / bar_range
    bar_range_pct = (high / low.replace(0.0, np.nan) - 1.0) * 100.0

    ema8 = close.ewm(span=8, min_periods=5, adjust=False).mean()
    ema21 = close.ewm(span=21, min_periods=10, adjust=False).mean()

    pullback_quality = _clip01((max_from_high20 - from_high20.fillna(0.0)) / 4.5) * (
        1.0 - _clip01((min_from_high20 - from_high20.fillna(0.0)) / 7.0)
    )
    reclaim_quality = (
        0.52 * _clip01((from_low20.fillna(0.0) - min_from_low20) / 4.5)
        + 0.30 * _clip01((from_low60.fillna(0.0) - 0.50) / 8.0)
        + 0.18 * _clip01((from_high60.fillna(-999.0) + 12.0) / 12.0)
    )
    volume_quality = (
        0.58 * _clip01((vol_ratio_mean - min_mean_vol_ratio) / 2.6)
        + 0.42 * _clip01((vol_ratio_med - min_med_vol_ratio) / 2.8)
    ) * (
        1.0 - _clip01((pd.concat([vol_ratio_mean, vol_ratio_med], axis=1).max(axis=1) - max_vol_ratio) / 5.0)
    )
    stability = (
        0.34 * _clip01((ret1.fillna(0.0) - min_ret1) / 1.8)
        + 0.30 * _clip01((ret5.fillna(0.0) - min_ret5) / 4.5)
        + 0.36 * _clip01((ret60.fillna(0.0) - min_ret60) / 16.0)
    )
    bar_acceptance = (
        0.34 * _clip01((taker_buy_ratio - min_taker_buy_ratio) / 0.22)
        + 0.28 * _clip01((close_pos.fillna(0.5) - min_close_pos) / 0.36)
        + 0.18 * (upper_wick.fillna(0.0) <= max_upper_wick).astype(float)
        + 0.12 * (lower_wick.fillna(0.0) <= 0.58).astype(float)
        + 0.08 * _clip01(oi_roc20.abs() / 12.0)
    )
    trend_context = (
        0.28 * (close >= ema8 * 0.996).astype(float)
        + 0.24 * (ema8 >= ema21 * 0.990).astype(float)
        + 0.38 * _clip01((from_high60.fillna(-999.0) + 14.0) / 13.0)
        + 0.10 * _clip01((-funding.fillna(0.0)) / 0.18)
    )
    symbol_quality = _clip01((leader_score - min_edge_score) / max(95.0 - min_edge_score, 1.0))

    score = (
        0.17 * pullback_quality
        + 0.16 * reclaim_quality
        + 0.19 * volume_quality
        + 0.16 * stability
        + 0.15 * bar_acceptance
        + 0.08 * trend_context
        + 0.09 * symbol_quality
    )

    valid = (
        edge_valid
        & (leader_score >= min_edge_score)
        &
        (vol_ratio_mean >= min_mean_vol_ratio)
        & (vol_ratio_med >= min_med_vol_ratio)
        & (vol_ratio_mean <= max_vol_ratio)
        & (vol_ratio_med <= max_vol_ratio)
        & (from_high20 >= min_from_high20)
        & (from_high20 <= max_from_high20)
        & (from_high60 >= -14.0)
        & (from_low20 >= min_from_low20)
        & (from_low20 <= max_from_low20)
        & (from_low60 <= 16.0)
        & (ret1 >= min_ret1)
        & (ret1 <= 2.8)
        & (ret3 >= -1.4)
        & (ret5 >= min_ret5)
        & (ret5 <= max_ret5)
        & (ret15 >= min_ret15)
        & (ret15 <= max_ret15)
        & (ret60 >= min_ret60)
        & (ret60 <= 16.0)
        & (taker_buy_ratio >= min_taker_buy_ratio)
        & (close_pos.fillna(0.5) >= min_close_pos)
        & (upper_wick.fillna(0.0) <= max_upper_wick)
        & (bar_range_pct <= max_bar_range)
        & (close >= ema21 * 0.970)
    )

    out = score.clip(lower=0.0, upper=max_output).where(valid, 0.0)
    return out.where(out.abs() >= min_output, 0.0).fillna(0.0)


def calculate_last_factor(df: pd.DataFrame) -> float:
    if df is None or len(df) < 90:
        return 0.0
    values = calculate_factor(df.tail(120))
    if values.empty:
        return 0.0
    try:
        return float(values.iloc[-1])
    except Exception:
        return 0.0
