"""Leader-learned pullback continuation short factor.

High-yield leader samples often short after the first 5m breakdown, not only at
the exact blow-off top.  The public-market structure is: learned short-side
symbol prior, a fresh 5m drop, enough multi-hour downside room, and buy volume
still present into the drop.  That last piece models absorption of late buyers
rather than a fully exhausted move.
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


def _leader_short_score_last(symbol: str, asof: object) -> float:
    if not symbol:
        return 0.0
    try:
        from cz.trading.crypto.meme.leader_edge import leader_symbol_edge
    except Exception:
        return 0.0
    try:
        if _env_bool("MEME_LEADER_RECENT_DECAY_ENABLED", False):
            edge = leader_symbol_edge(symbol, asof=asof)
        else:
            edge = leader_symbol_edge(symbol)
    except Exception:
        return 0.0
    if edge is None or getattr(edge, "direction", "") != "SHORT":
        return 0.0
    try:
        return float(getattr(edge, "score", 0.0))
    except Exception:
        return 0.0


def calculate_factor(df: pd.DataFrame) -> pd.Series:
    if df is None or len(df) < 260:
        return pd.Series(0.0, index=getattr(df, "index", None))

    symbol = _frame_symbol(df)
    short_score = _leader_short_scores(symbol, df.index)
    min_short_score = _env_float("MEME_LEADER_PULLBACK_SHORT_MIN_SCORE", 58.0)
    edge_valid = short_score >= min_short_score
    if not bool(edge_valid.any()):
        return pd.Series(0.0, index=df.index)

    high = _safe_series(df, "high")
    low = _safe_series(df, "low")
    close = _safe_series(df, "close")
    volume = _safe_series(df, "volume", 1.0)
    quote_volume = _safe_series(df, "quote_volume")
    taker_buy_quote = _safe_series(df, "taker_buy_quote")
    oi = _safe_series(df, "open_interest")
    funding = _safe_series(df, "funding_rate")

    min_output = _env_float("MEME_LEADER_PULLBACK_SHORT_MIN_OUTPUT", 0.26)
    max_output = _env_float("MEME_LEADER_PULLBACK_SHORT_MAX_OUTPUT", 0.88)
    min_ret5 = _env_float("MEME_LEADER_PULLBACK_SHORT_MIN_RET5_PCT", -8.0)
    max_ret5 = _env_float("MEME_LEADER_PULLBACK_SHORT_MAX_RET5_PCT", -2.0)
    min_ret60 = _env_float("MEME_LEADER_PULLBACK_SHORT_MIN_RET60_PCT", -18.0)
    max_ret60 = _env_float("MEME_LEADER_PULLBACK_SHORT_MAX_RET60_PCT", 12.0)
    min_ret240 = _env_float("MEME_LEADER_PULLBACK_SHORT_MIN_RET240_PCT", 0.0)
    min_from_low240 = _env_float("MEME_LEADER_PULLBACK_SHORT_MIN_FROM_LOW240_PCT", 4.0)
    max_pullback_high60 = _env_float("MEME_LEADER_PULLBACK_SHORT_MAX_PULLBACK_HIGH60_PCT", 14.0)
    min_taker_buy_ratio = _env_float("MEME_LEADER_PULLBACK_SHORT_MIN_TAKER_BUY_RATIO", 0.48)
    max_taker_buy_ratio = _env_float("MEME_LEADER_PULLBACK_SHORT_MAX_TAKER_BUY_RATIO", 0.74)
    min_vol_ratio = _env_float("MEME_LEADER_PULLBACK_SHORT_MIN_VOL_RATIO", 0.4)
    max_vol_ratio = _env_float("MEME_LEADER_PULLBACK_SHORT_MAX_VOL_RATIO", 4.0)

    ret1 = close.pct_change(1) * 100.0
    ret3 = close.pct_change(3) * 100.0
    ret5 = close.pct_change(5) * 100.0
    ret15 = close.pct_change(15) * 100.0
    ret60 = close.pct_change(60) * 100.0
    ret240 = close.pct_change(240) * 100.0

    high60 = high.rolling(60, min_periods=20).max()
    low240 = low.rolling(240, min_periods=80).min()
    from_high60 = (close / high60.replace(0.0, np.nan) - 1.0) * 100.0
    from_low240 = (close / low240.replace(0.0, np.nan) - 1.0) * 100.0

    vol_base = volume.rolling(30, min_periods=12).median().shift(1).replace(0.0, np.nan)
    vol_ratio = (volume / vol_base).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    taker_buy_ratio = (
        taker_buy_quote / quote_volume.replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.50)
    oi_roc20 = oi.pct_change(20).replace([np.inf, -np.inf], np.nan).fillna(0.0) * 100.0

    symbol_strength = (short_score / 86.0).clip(lower=0.0, upper=1.0)
    fresh_break = _clip01((max_ret5 - ret5.fillna(0.0)) / max(max_ret5 - min_ret5, 1.0))
    no_late_crash = 1.0 - _clip01((min_ret5 - ret5.fillna(0.0)) / 8.0)
    pullback_room = _clip01((from_low240.fillna(0.0) - min_from_low240) / 18.0)
    pump_memory = _clip01((ret240.fillna(0.0) - min_ret240) / 28.0)
    not_too_far_from_high = 1.0 - _clip01((-from_high60.fillna(-999.0) - 2.0) / max(max_pullback_high60 - 2.0, 1.0))
    buy_absorption = _clip01((taker_buy_ratio - min_taker_buy_ratio) / max(max_taker_buy_ratio - min_taker_buy_ratio, 0.08))
    controlled_volume = _clip01((vol_ratio - min_vol_ratio) / 1.8) * (1.0 - _clip01((vol_ratio - max_vol_ratio) / 8.0))
    continuation = (
        0.36 * fresh_break
        + 0.16 * no_late_crash
        + 0.16 * pullback_room
        + 0.12 * pump_memory
        + 0.10 * not_too_far_from_high
        + 0.07 * buy_absorption
        + 0.03 * _clip01(oi_roc20 / 16.0)
    )
    score = symbol_strength * (0.88 * continuation + 0.12 * controlled_volume)

    valid = (
        edge_valid
        & (ret5 >= min_ret5)
        & (ret5 <= max_ret5)
        & (ret3 <= 1.0)
        & (ret1 <= 1.0)
        & (ret15 >= -10.0)
        & (ret60 >= min_ret60)
        & (ret60 <= max_ret60)
        & (ret240 >= min_ret240)
        & (from_low240 >= min_from_low240)
        & (from_high60 >= -abs(max_pullback_high60))
        & (taker_buy_ratio >= min_taker_buy_ratio)
        & (taker_buy_ratio <= max_taker_buy_ratio)
        & (vol_ratio >= min_vol_ratio)
        & (vol_ratio <= max_vol_ratio)
        & (funding.fillna(0.0) > -1.5)
    )

    out = pd.Series(0.0, index=df.index)
    out = out.mask(valid, -score.clip(lower=0.0, upper=max_output))
    return out.where(out.abs() >= min_output, 0.0).fillna(0.0)


def _pct_last(values: np.ndarray, periods: int) -> float:
    if len(values) <= periods:
        return float("nan")
    prev = float(values[-periods - 1])
    if prev <= 0:
        return float("nan")
    return (float(values[-1]) - prev) / prev * 100.0


def _clip01_scalar(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def calculate_last_factor(df: pd.DataFrame) -> float:
    """Return the exact last-row value without constructing the full output."""

    if df is None or len(df) < 260:
        return 0.0

    symbol = _frame_symbol(df)
    asof = df.index[-1] if len(df.index) else None
    short_score = _leader_short_score_last(symbol, asof)
    min_short_score = _env_float("MEME_LEADER_PULLBACK_SHORT_MIN_SCORE", 58.0)
    if short_score < min_short_score:
        return 0.0

    high = _safe_series(df, "high")
    low = _safe_series(df, "low")
    close = _safe_series(df, "close")
    volume = _safe_series(df, "volume", 1.0)
    quote_volume = _safe_series(df, "quote_volume")
    taker_buy_quote = _safe_series(df, "taker_buy_quote")
    oi = _safe_series(df, "open_interest")
    funding = _safe_series(df, "funding_rate")

    min_output = _env_float("MEME_LEADER_PULLBACK_SHORT_MIN_OUTPUT", 0.26)
    max_output = _env_float("MEME_LEADER_PULLBACK_SHORT_MAX_OUTPUT", 0.88)
    min_ret5 = _env_float("MEME_LEADER_PULLBACK_SHORT_MIN_RET5_PCT", -8.0)
    max_ret5 = _env_float("MEME_LEADER_PULLBACK_SHORT_MAX_RET5_PCT", -2.0)
    min_ret60 = _env_float("MEME_LEADER_PULLBACK_SHORT_MIN_RET60_PCT", -18.0)
    max_ret60 = _env_float("MEME_LEADER_PULLBACK_SHORT_MAX_RET60_PCT", 12.0)
    min_ret240 = _env_float("MEME_LEADER_PULLBACK_SHORT_MIN_RET240_PCT", 0.0)
    min_from_low240 = _env_float("MEME_LEADER_PULLBACK_SHORT_MIN_FROM_LOW240_PCT", 4.0)
    max_pullback_high60 = _env_float("MEME_LEADER_PULLBACK_SHORT_MAX_PULLBACK_HIGH60_PCT", 14.0)
    min_taker_buy_ratio = _env_float("MEME_LEADER_PULLBACK_SHORT_MIN_TAKER_BUY_RATIO", 0.48)
    max_taker_buy_ratio = _env_float("MEME_LEADER_PULLBACK_SHORT_MAX_TAKER_BUY_RATIO", 0.74)
    min_vol_ratio = _env_float("MEME_LEADER_PULLBACK_SHORT_MIN_VOL_RATIO", 0.4)
    max_vol_ratio = _env_float("MEME_LEADER_PULLBACK_SHORT_MAX_VOL_RATIO", 4.0)

    close_np = close.to_numpy(dtype=float)
    ret1 = _pct_last(close_np, 1)
    ret3 = _pct_last(close_np, 3)
    ret5 = _pct_last(close_np, 5)
    ret15 = _pct_last(close_np, 15)
    ret60 = _pct_last(close_np, 60)
    ret240 = _pct_last(close_np, 240)

    close_last = float(close.iloc[-1])
    high60 = float(high.iloc[-60:].max())
    low240 = float(low.iloc[-240:].min())
    from_high60 = (close_last / high60 - 1.0) * 100.0 if high60 > 0 else float("nan")
    from_low240 = (close_last / low240 - 1.0) * 100.0 if low240 > 0 else float("nan")

    vol_base = float(volume.iloc[-31:-1].median())
    vol_ratio = float(volume.iloc[-1]) / vol_base if vol_base > 0 else float("nan")
    if not np.isfinite(vol_ratio):
        vol_ratio = 1.0
    quote_last = float(quote_volume.iloc[-1])
    taker_buy_ratio = float(taker_buy_quote.iloc[-1]) / quote_last if quote_last > 0 else float("nan")
    if not np.isfinite(taker_buy_ratio):
        taker_buy_ratio = 0.50

    oi_prev = float(oi.iloc[-21])
    oi_roc20 = (float(oi.iloc[-1]) - oi_prev) / oi_prev * 100.0 if oi_prev != 0 else 0.0
    if not np.isfinite(oi_roc20):
        oi_roc20 = 0.0

    ret5_fill = 0.0 if not np.isfinite(ret5) else ret5
    ret240_fill = 0.0 if not np.isfinite(ret240) else ret240
    from_high60_fill = -999.0 if not np.isfinite(from_high60) else from_high60
    from_low240_fill = 0.0 if not np.isfinite(from_low240) else from_low240

    symbol_strength = max(0.0, min(1.0, short_score / 86.0))
    fresh_break = _clip01_scalar((max_ret5 - ret5_fill) / max(max_ret5 - min_ret5, 1.0))
    no_late_crash = 1.0 - _clip01_scalar((min_ret5 - ret5_fill) / 8.0)
    pullback_room = _clip01_scalar((from_low240_fill - min_from_low240) / 18.0)
    pump_memory = _clip01_scalar((ret240_fill - min_ret240) / 28.0)
    not_too_far_from_high = 1.0 - _clip01_scalar(
        (-from_high60_fill - 2.0) / max(max_pullback_high60 - 2.0, 1.0)
    )
    buy_absorption = _clip01_scalar(
        (taker_buy_ratio - min_taker_buy_ratio) / max(max_taker_buy_ratio - min_taker_buy_ratio, 0.08)
    )
    controlled_volume = _clip01_scalar((vol_ratio - min_vol_ratio) / 1.8) * (
        1.0 - _clip01_scalar((vol_ratio - max_vol_ratio) / 8.0)
    )
    continuation = (
        0.36 * fresh_break
        + 0.16 * no_late_crash
        + 0.16 * pullback_room
        + 0.12 * pump_memory
        + 0.10 * not_too_far_from_high
        + 0.07 * buy_absorption
        + 0.03 * _clip01_scalar(oi_roc20 / 16.0)
    )
    score = symbol_strength * (0.88 * continuation + 0.12 * controlled_volume)

    valid = (
        ret5 >= min_ret5
        and ret5 <= max_ret5
        and ret3 <= 1.0
        and ret1 <= 1.0
        and ret15 >= -10.0
        and ret60 >= min_ret60
        and ret60 <= max_ret60
        and ret240 >= min_ret240
        and from_low240 >= min_from_low240
        and from_high60 >= -abs(max_pullback_high60)
        and taker_buy_ratio >= min_taker_buy_ratio
        and taker_buy_ratio <= max_taker_buy_ratio
        and vol_ratio >= min_vol_ratio
        and vol_ratio <= max_vol_ratio
        and float(funding.iloc[-1]) > -1.5
    )
    if not valid:
        return 0.0
    value = -max(0.0, min(max_output, score))
    return value if abs(value) >= min_output else 0.0
