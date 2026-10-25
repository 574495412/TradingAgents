"""Leader long-tail short factor.

This factor targets the structure that dominated recent high-yield lead-trader
samples: a learned short-side symbol prior, price still far enough above the
multi-hour low to leave downside room, and current momentum stalling instead of
an obvious bottom chase.  It only uses as-of market data plus the versioned
leader symbol report shared by live and backtest.
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


def _leader_short_scores(symbol: str, index: pd.Index) -> pd.Series:
    if not symbol:
        return pd.Series(0.0, index=index, dtype=float)
    try:
        from cz.trading.crypto.meme.leader_edge import leader_symbol_edge, leader_symbol_edges
    except Exception:
        return pd.Series(0.0, index=index, dtype=float)

    if _env_bool("MEME_LEADER_RECENT_DECAY_ENABLED", False):
        scores = pd.Series(0.0, index=index, dtype=float)
        try:
            edges = leader_symbol_edges(symbol, index)
        except Exception:
            edges = ()
        for ts, edge in zip(index, edges):
            if edge is not None and getattr(edge, "direction", "") == "SHORT":
                scores.loc[ts] = float(getattr(edge, "score", 0.0))
        return scores

    try:
        edge = leader_symbol_edge(symbol)
    except Exception:
        edge = None
    if edge is None or getattr(edge, "direction", "") != "SHORT":
        return pd.Series(0.0, index=index, dtype=float)
    return pd.Series(float(getattr(edge, "score", 0.0)), index=index, dtype=float)


def calculate_factor(df: pd.DataFrame) -> pd.Series:
    if df is None or len(df) < 260:
        return pd.Series(0.0, index=getattr(df, "index", None))

    symbol = _frame_symbol(df)
    short_score = _leader_short_scores(symbol, df.index)
    min_short_score = _env_float("MEME_LEADER_LONGTAIL_SHORT_MIN_SCORE", 58.0)
    edge_valid = short_score >= min_short_score
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

    min_output = _env_float("MEME_LEADER_LONGTAIL_SHORT_MIN_OUTPUT", 0.24)
    max_output = _env_float("MEME_LEADER_LONGTAIL_SHORT_MAX_OUTPUT", 0.88)
    min_from_low240 = _env_float("MEME_LEADER_LONGTAIL_SHORT_MIN_FROM_LOW240_PCT", 1.2)
    max_pullback_high60 = _env_float("MEME_LEADER_LONGTAIL_SHORT_MAX_PULLBACK_HIGH60_PCT", 16.0)
    min_ret240 = _env_float("MEME_LEADER_LONGTAIL_SHORT_MIN_RET240_PCT", -8.0)
    min_ret60 = _env_float("MEME_LEADER_LONGTAIL_SHORT_MIN_RET60_PCT", -26.0)
    max_ret60 = _env_float("MEME_LEADER_LONGTAIL_SHORT_MAX_RET60_PCT", 35.0)
    min_ret15 = _env_float("MEME_LEADER_LONGTAIL_SHORT_MIN_RET15_PCT", -12.0)
    max_ret15 = _env_float("MEME_LEADER_LONGTAIL_SHORT_MAX_RET15_PCT", 12.0)
    min_ret5 = _env_float("MEME_LEADER_LONGTAIL_SHORT_MIN_RET5_PCT", -8.0)
    max_ret5 = _env_float("MEME_LEADER_LONGTAIL_SHORT_MAX_RET5_PCT", 5.0)
    max_ret1 = _env_float("MEME_LEADER_LONGTAIL_SHORT_MAX_RET1_PCT", 1.2)
    min_vol_ratio = _env_float("MEME_LEADER_LONGTAIL_SHORT_MIN_VOL_RATIO", 0.35)
    max_vol_ratio = _env_float("MEME_LEADER_LONGTAIL_SHORT_MAX_VOL_RATIO", 12.0)
    max_taker_buy_ratio = _env_float("MEME_LEADER_LONGTAIL_SHORT_MAX_TAKER_BUY_RATIO", 0.62)

    ret1 = close.pct_change(1) * 100.0
    ret5 = close.pct_change(5) * 100.0
    ret15 = close.pct_change(15) * 100.0
    ret60 = close.pct_change(60) * 100.0
    ret240 = close.pct_change(240) * 100.0

    high60 = high.rolling(60, min_periods=20).max()
    low60 = low.rolling(60, min_periods=20).min()
    low240 = low.rolling(240, min_periods=80).min()
    from_high60 = (close / high60.replace(0.0, np.nan) - 1.0) * 100.0
    from_low60 = (close / low60.replace(0.0, np.nan) - 1.0) * 100.0
    from_low240 = (close / low240.replace(0.0, np.nan) - 1.0) * 100.0

    vol_base = volume.rolling(30, min_periods=12).median().shift(1).replace(0.0, np.nan)
    vol_ratio = (volume / vol_base).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    taker_buy_ratio = (
        taker_buy_quote / quote_volume.replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.50)

    bar_range = (high - low).replace(0.0, np.nan)
    upper_wick = (high - pd.concat([open_, close], axis=1).max(axis=1)) / bar_range
    close_pos = (close - low) / bar_range
    ema8 = close.ewm(span=8, min_periods=5, adjust=False).mean()
    ema21 = close.ewm(span=21, min_periods=10, adjust=False).mean()
    oi_roc20 = oi.pct_change(20).replace([np.inf, -np.inf], np.nan).fillna(0.0) * 100.0

    symbol_strength = (short_score / 86.0).clip(lower=0.0, upper=1.0)
    downside_room = _clip01((from_low240.fillna(0.0) - min_from_low240) / 18.0)
    pump_memory = _clip01((ret240.fillna(0.0) - min_ret240) / 34.0)
    not_too_far_from_high = 1.0 - _clip01((-from_high60.fillna(-999.0) - 1.0) / max(max_pullback_high60 - 1.0, 1.0))
    stall = (
        0.32 * _clip01((max_ret5 - ret5.fillna(0.0)) / max(max_ret5 - min_ret5, 1.0))
        + 0.24 * _clip01((max_ret15 - ret15.fillna(0.0)) / max(max_ret15 - min_ret15, 1.0))
        + 0.18 * (close <= ema8 * 1.004).astype(float)
        + 0.14 * (ema8 <= ema21 * 1.010).astype(float)
        + 0.12 * _clip01((0.62 - close_pos.fillna(0.5)) / 0.50)
    )
    sell_pressure = (
        0.56 * _clip01((max_taker_buy_ratio - taker_buy_ratio) / max(max_taker_buy_ratio - 0.34, 0.08))
        + 0.24 * _clip01(upper_wick.fillna(0.0) / 0.42)
        + 0.12 * _clip01(funding.fillna(0.0) / 0.20)
        + 0.08 * _clip01(oi_roc20 / 18.0)
    )
    activity = _clip01((vol_ratio - min_vol_ratio) / 2.2) * (1.0 - _clip01((vol_ratio - max_vol_ratio) / 12.0))

    score = symbol_strength * (
        0.24 * downside_room
        + 0.20 * pump_memory
        + 0.18 * not_too_far_from_high
        + 0.18 * stall
        + 0.14 * sell_pressure
        + 0.06 * activity
    )

    valid = (
        edge_valid
        & (from_low240 >= min_from_low240)
        & (from_low60 >= 0.4)
        & (from_high60 >= -abs(max_pullback_high60))
        & (ret240 >= min_ret240)
        & (ret60 >= min_ret60)
        & (ret60 <= max_ret60)
        & (ret15 >= min_ret15)
        & (ret15 <= max_ret15)
        & (ret5 >= min_ret5)
        & (ret5 <= max_ret5)
        & (ret1 <= max_ret1)
        & (vol_ratio >= min_vol_ratio)
        & (vol_ratio <= max_vol_ratio)
        & (taker_buy_ratio <= max_taker_buy_ratio)
    )

    out = pd.Series(0.0, index=df.index)
    out = out.mask(valid, -score.clip(lower=0.0, upper=max_output))
    return out.where(out.abs() >= min_output, 0.0).fillna(0.0)
