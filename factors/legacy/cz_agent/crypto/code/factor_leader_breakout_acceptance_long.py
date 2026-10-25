"""Recent leader breakout-acceptance long factor.

This factor is intentionally narrow: it only goes long when recent lead-trader
labels for the same symbol are positive as of the current bar, and public
market data shows a breakout being accepted rather than fading immediately.
It is a research factor for increasing high-quality meme long opportunities.
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


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


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
    if not symbol:
        return pd.Series(0.0, index=index, dtype=float)
    try:
        from cz.trading.crypto.meme.leader_edge import leader_symbol_edge, recent_leader_symbol_edges
    except Exception:
        return pd.Series(0.0, index=index, dtype=float)

    if _env_bool("MEME_LEADER_BREAKOUT_LONG_USE_RECENT_ASOF", True):
        scores = pd.Series(0.0, index=index, dtype=float)
        try:
            edges = recent_leader_symbol_edges(symbol, index, require_enabled=False)
        except Exception:
            edges = ()
        for ts, edge in zip(index, edges):
            if edge is not None and getattr(edge, "direction", "") == "LONG":
                scores.loc[ts] = float(getattr(edge, "score", 0.0))
        return scores

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
    min_leader_score = _env_float("MEME_LEADER_BREAKOUT_LONG_MIN_SCORE", 58.0)
    leader_valid = leader_score >= min_leader_score
    if not bool(leader_valid.any()):
        return pd.Series(0.0, index=df.index)

    open_ = _safe_series(df, "open")
    high = _safe_series(df, "high")
    low = _safe_series(df, "low")
    close = _safe_series(df, "close")
    volume = _safe_series(df, "volume", 1.0)
    quote_volume = _safe_series(df, "quote_volume")
    taker_buy_quote = _safe_series(df, "taker_buy_quote")
    oi = _safe_series(df, "open_interest")

    min_output = _env_float("MEME_LEADER_BREAKOUT_LONG_MIN_OUTPUT", 0.24)
    max_output = _env_float("MEME_LEADER_BREAKOUT_LONG_MAX_OUTPUT", 0.86)
    min_ret5 = _env_float("MEME_LEADER_BREAKOUT_LONG_MIN_RET5_PCT", -1.2)
    min_ret15 = _env_float("MEME_LEADER_BREAKOUT_LONG_MIN_RET15_PCT", 4.0)
    min_ret60 = _env_float("MEME_LEADER_BREAKOUT_LONG_MIN_RET60_PCT", -2.0)
    max_ret5 = _env_float("MEME_LEADER_BREAKOUT_LONG_MAX_RET5_PCT", 10.0)
    max_pullback_high20 = _env_float("MEME_LEADER_BREAKOUT_LONG_MAX_PULLBACK_HIGH20_PCT", 3.0)
    min_from_low60 = _env_float("MEME_LEADER_BREAKOUT_LONG_MIN_FROM_LOW60_PCT", 4.0)
    min_vol_ratio = _env_float("MEME_LEADER_BREAKOUT_LONG_MIN_VOL_RATIO", 1.05)
    max_vol_ratio = _env_float("MEME_LEADER_BREAKOUT_LONG_MAX_VOL_RATIO", 18.0)
    min_taker_buy_ratio = _env_float("MEME_LEADER_BREAKOUT_LONG_MIN_TAKER_BUY_RATIO", 0.45)

    ret1 = close.pct_change(1) * 100.0
    ret3 = close.pct_change(3) * 100.0
    ret5 = close.pct_change(5) * 100.0
    ret15 = close.pct_change(15) * 100.0
    ret60 = close.pct_change(60) * 100.0

    high20 = high.rolling(20, min_periods=8).max()
    low20 = low.rolling(20, min_periods=8).min()
    low60 = low.rolling(60, min_periods=20).min()
    from_high20 = (close / high20.replace(0.0, np.nan) - 1.0) * 100.0
    from_low20 = (close / low20.replace(0.0, np.nan) - 1.0) * 100.0
    from_low60 = (close / low60.replace(0.0, np.nan) - 1.0) * 100.0

    vol_base = volume.rolling(30, min_periods=12).median().shift(1).replace(0.0, np.nan)
    vol_ratio = (volume / vol_base).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    taker_buy_ratio = (
        taker_buy_quote / quote_volume.replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.52)
    oi_roc20 = oi.pct_change(20).replace([np.inf, -np.inf], np.nan).fillna(0.0) * 100.0

    bar_range = (high - low).replace(0.0, np.nan)
    upper_wick = (high - pd.concat([open_, close], axis=1).max(axis=1)) / bar_range
    close_pos = (close - low) / bar_range

    symbol_strength = (leader_score / 86.0).clip(lower=0.0, upper=1.0)
    momentum = (
        0.44 * _clip01((ret15.fillna(0.0) - min_ret15) / 10.0)
        + 0.22 * _clip01((ret5.fillna(0.0) - min_ret5) / max(max_ret5 - min_ret5, 1.0))
        + 0.14 * _clip01((ret3.fillna(0.0) + 0.5) / 4.0)
        + 0.20 * _clip01((ret60.fillna(0.0) - min_ret60) / 14.0)
    )
    accepted_near_high = 1.0 - _clip01(
        (-from_high20.fillna(-999.0)) / max(max_pullback_high20, 0.5)
    )
    room_from_low = _clip01((from_low60.fillna(0.0) - min_from_low60) / 12.0)
    volume_quality = _clip01((vol_ratio - min_vol_ratio) / 3.5) * (
        1.0 - _clip01((vol_ratio - max_vol_ratio) / max(max_vol_ratio, 1.0))
    )
    buy_acceptance = (
        0.55 * _clip01((close_pos.fillna(0.5) - 0.48) / 0.42)
        + 0.25 * _clip01((taker_buy_ratio - min_taker_buy_ratio) / 0.25)
        + 0.20 * (upper_wick.fillna(0.0) <= 0.42).astype(float)
    )
    crowding_fuel = _clip01(oi_roc20 / 18.0)

    score = symbol_strength * (
        0.34 * momentum
        + 0.22 * accepted_near_high
        + 0.16 * room_from_low
        + 0.14 * volume_quality
        + 0.10 * buy_acceptance
        + 0.04 * crowding_fuel
    )

    valid = (
        leader_valid
        & (ret5 >= min_ret5)
        & (ret5 <= max_ret5)
        & (ret15 >= min_ret15)
        & (ret60 >= min_ret60)
        & (ret1 >= -1.6)
        & (from_high20 >= -abs(max_pullback_high20))
        & (from_low20 >= 1.8)
        & (from_low60 >= min_from_low60)
        & (vol_ratio >= min_vol_ratio)
        & (vol_ratio <= max_vol_ratio)
        & (taker_buy_ratio >= min_taker_buy_ratio)
        & (close_pos.fillna(0.5) >= 0.48)
        & (upper_wick.fillna(0.0) <= 0.48)
    )

    out = pd.Series(0.0, index=df.index)
    out = out.mask(valid, score.clip(lower=0.0, upper=max_output))
    return out.where(out.abs() >= min_output, 0.0).fillna(0.0)
