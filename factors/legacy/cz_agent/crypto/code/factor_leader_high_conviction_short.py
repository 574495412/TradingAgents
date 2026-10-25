"""Leader high-conviction short continuation factor.

This research factor targets a structure repeatedly seen in high-yield leader
samples: a learned short-side symbol prior, a sharp one-hour selloff, enough
rebound room from the intrabar low to avoid pure bottom-chasing, and renewed
volume during the second leg.  It is intentionally narrow and must be promoted
only through portfolio backtests.
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


def _clip01_scalar(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _frame_symbol(df: pd.DataFrame) -> str:
    raw = df.attrs.get("symbol") or df.attrs.get("market_symbol") or ""
    return str(raw or "").strip().upper().replace("/", "")


def _leader_short_scores(symbol: str, index: pd.Index) -> pd.Series:
    if not symbol:
        return pd.Series(0.0, index=index, dtype=float)
    explicit_file = os.getenv("MEME_LEADER_HIGH_CONVICTION_SHORT_EDGE_FILE", "").strip()
    if explicit_file:
        try:
            from cz.trading.crypto.meme.leader_edge import leader_symbol_edge_from_file
        except Exception:
            return pd.Series(0.0, index=index, dtype=float)
        edge = leader_symbol_edge_from_file(
            symbol,
            explicit_file,
            min_score=_env_float("MEME_LEADER_HIGH_CONVICTION_SHORT_MIN_SCORE", 60.0),
            min_samples=int(_env_float("MEME_LEADER_HIGH_CONVICTION_SHORT_MIN_SAMPLES", 1.0)),
            min_win_rate=_env_float("MEME_LEADER_HIGH_CONVICTION_SHORT_MIN_WIN_RATE", 0.0),
        )
        if edge is None or getattr(edge, "direction", "") != "SHORT":
            return pd.Series(0.0, index=index, dtype=float)
        return pd.Series(float(getattr(edge, "score", 0.0)), index=index, dtype=float)
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
    explicit_file = os.getenv("MEME_LEADER_HIGH_CONVICTION_SHORT_EDGE_FILE", "").strip()
    if explicit_file:
        try:
            from cz.trading.crypto.meme.leader_edge import leader_symbol_edge_from_file
        except Exception:
            return 0.0
        edge = leader_symbol_edge_from_file(
            symbol,
            explicit_file,
            min_score=_env_float("MEME_LEADER_HIGH_CONVICTION_SHORT_MIN_SCORE", 60.0),
            min_samples=int(_env_float("MEME_LEADER_HIGH_CONVICTION_SHORT_MIN_SAMPLES", 1.0)),
            min_win_rate=_env_float("MEME_LEADER_HIGH_CONVICTION_SHORT_MIN_WIN_RATE", 0.0),
        )
        if edge is None or getattr(edge, "direction", "") != "SHORT":
            return 0.0
        try:
            return float(getattr(edge, "score", 0.0))
        except Exception:
            return 0.0
    try:
        from cz.trading.crypto.meme.leader_edge import leader_symbol_edge
    except Exception:
        return 0.0
    try:
        edge = leader_symbol_edge(symbol, asof=asof) if _env_bool("MEME_LEADER_RECENT_DECAY_ENABLED", False) else leader_symbol_edge(symbol)
    except Exception:
        return 0.0
    if edge is None or getattr(edge, "direction", "") != "SHORT":
        return 0.0
    try:
        return float(getattr(edge, "score", 0.0))
    except Exception:
        return 0.0


def _pct_last(values: np.ndarray, periods: int) -> float:
    if len(values) <= periods:
        return float("nan")
    prev = float(values[-periods - 1])
    if prev <= 0:
        return float("nan")
    return (float(values[-1]) - prev) / prev * 100.0


def calculate_factor(df: pd.DataFrame) -> pd.Series:
    if df is None or len(df) < 260:
        return pd.Series(0.0, index=getattr(df, "index", None))

    symbol = _frame_symbol(df)
    short_score = _leader_short_scores(symbol, df.index)
    min_short_score = _env_float("MEME_LEADER_HIGH_CONVICTION_SHORT_MIN_SCORE", 60.0)
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

    min_output = _env_float("MEME_LEADER_HIGH_CONVICTION_SHORT_MIN_OUTPUT", 0.28)
    max_output = _env_float("MEME_LEADER_HIGH_CONVICTION_SHORT_MAX_OUTPUT", 0.90)
    min_ret60 = _env_float("MEME_LEADER_HIGH_CONVICTION_SHORT_MAX_RET60_PCT", -4.0)
    min_ret5 = _env_float("MEME_LEADER_HIGH_CONVICTION_SHORT_MIN_RET5_PCT", -14.5)
    max_ret5 = _env_float("MEME_LEADER_HIGH_CONVICTION_SHORT_MAX_RET5_PCT", 3.0)
    max_ret1 = _env_float("MEME_LEADER_HIGH_CONVICTION_SHORT_MAX_RET1_PCT", 5.0)
    min_from_low15 = _env_float("MEME_LEADER_HIGH_CONVICTION_SHORT_MIN_FROM_LOW15_PCT", 0.55)
    max_from_low15 = _env_float("MEME_LEADER_HIGH_CONVICTION_SHORT_MAX_FROM_LOW15_PCT", 7.0)
    max_from_high60 = _env_float("MEME_LEADER_HIGH_CONVICTION_SHORT_MAX_FROM_HIGH60_PCT", -5.0)
    min_vol_ratio = _env_float("MEME_LEADER_HIGH_CONVICTION_SHORT_MIN_VOL_RATIO", 1.45)
    max_vol_ratio = _env_float("MEME_LEADER_HIGH_CONVICTION_SHORT_MAX_VOL_RATIO", 14.0)
    max_taker_buy_ratio = _env_float("MEME_LEADER_HIGH_CONVICTION_SHORT_MAX_TAKER_BUY_RATIO", 0.68)

    ret1 = close.pct_change(1) * 100.0
    ret5 = close.pct_change(5) * 100.0
    ret15 = close.pct_change(15) * 100.0
    ret60 = close.pct_change(60) * 100.0
    ret240 = close.pct_change(240) * 100.0

    high60 = high.rolling(60, min_periods=20).max()
    low15 = low.rolling(15, min_periods=5).min()
    low60 = low.rolling(60, min_periods=20).min()
    low240 = low.rolling(240, min_periods=80).min()
    from_high60 = (close / high60.replace(0.0, np.nan) - 1.0) * 100.0
    from_low15 = (close / low15.replace(0.0, np.nan) - 1.0) * 100.0
    from_low60 = (close / low60.replace(0.0, np.nan) - 1.0) * 100.0
    from_low240 = (close / low240.replace(0.0, np.nan) - 1.0) * 100.0

    vol_base = volume.rolling(30, min_periods=12).median().shift(1).replace(0.0, np.nan)
    vol_ratio = (volume / vol_base).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    taker_buy_ratio = (
        taker_buy_quote / quote_volume.replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.50)
    oi_roc20 = oi.pct_change(20).replace([np.inf, -np.inf], np.nan).fillna(0.0) * 100.0

    bar_range = (high - low).replace(0.0, np.nan)
    close_pos = (close - low) / bar_range
    upper_wick = (high - pd.concat([open_, close], axis=1).max(axis=1)) / bar_range

    symbol_strength = (short_score / 88.0).clip(lower=0.0, upper=1.0)
    cascade = _clip01((-ret60.fillna(0.0) - abs(min_ret60)) / 14.0)
    fresh_break = _clip01((max_ret5 - ret5.fillna(0.0)) / max(max_ret5 - min_ret5, 1.0))
    rebound_room = _clip01((from_low15.fillna(0.0) - min_from_low15) / max(max_from_low15 - min_from_low15, 1.0))
    downside_room = _clip01((from_low240.fillna(0.0) - 0.3) / 8.0)
    high_break = _clip01((-from_high60.fillna(0.0) - abs(max_from_high60)) / 12.0)
    activity = _clip01((vol_ratio - min_vol_ratio) / 5.0) * (1.0 - _clip01((vol_ratio - max_vol_ratio) / 10.0))
    sell_pressure = (
        0.48 * _clip01((max_taker_buy_ratio - taker_buy_ratio) / max(max_taker_buy_ratio - 0.38, 0.08))
        + 0.26 * _clip01((0.82 - close_pos.fillna(0.5)) / 0.62)
        + 0.16 * _clip01(upper_wick.fillna(0.0) / 0.35)
        + 0.10 * _clip01((funding.fillna(0.0).clip(lower=0.0)) / 0.18)
    )
    flow = _clip01((oi_roc20.abs()) / 8.0)

    score = symbol_strength * (
        0.30 * cascade
        + 0.16 * fresh_break
        + 0.18 * rebound_room
        + 0.12 * downside_room
        + 0.16 * high_break
        + 0.06 * activity
        + 0.02 * sell_pressure
        + 0.00 * flow
    )

    valid = (
        edge_valid
        & (ret60 <= min_ret60)
        & (ret5 >= min_ret5)
        & (ret5 <= max_ret5)
        & (ret1 <= max_ret1)
        & (from_low15 >= min_from_low15)
        & (from_low15 <= max_from_low15)
        & (from_low60 >= 0.25)
        & (from_high60 <= max_from_high60)
        & (vol_ratio >= min_vol_ratio)
        & (vol_ratio <= max_vol_ratio)
        & (taker_buy_ratio <= max_taker_buy_ratio)
        & (ret15 <= 1.5)
        & (ret240 <= 16.0)
    )

    out = pd.Series(0.0, index=df.index)
    out = out.mask(valid, -score.clip(lower=0.0, upper=max_output))
    return out.where(out.abs() >= min_output, 0.0).fillna(0.0)


def calculate_last_factor(df: pd.DataFrame) -> float:
    if df is None or len(df) < 260:
        return 0.0

    symbol = _frame_symbol(df)
    asof = df.index[-1] if len(df.index) else None
    short_score = _leader_short_score_last(symbol, asof)
    min_short_score = _env_float("MEME_LEADER_HIGH_CONVICTION_SHORT_MIN_SCORE", 60.0)
    if short_score < min_short_score:
        return 0.0

    open_ = _safe_series(df, "open")
    high = _safe_series(df, "high")
    low = _safe_series(df, "low")
    close = _safe_series(df, "close")
    volume = _safe_series(df, "volume", 1.0)
    quote_volume = _safe_series(df, "quote_volume")
    taker_buy_quote = _safe_series(df, "taker_buy_quote")
    oi = _safe_series(df, "open_interest")
    funding = _safe_series(df, "funding_rate")

    min_output = _env_float("MEME_LEADER_HIGH_CONVICTION_SHORT_MIN_OUTPUT", 0.28)
    max_output = _env_float("MEME_LEADER_HIGH_CONVICTION_SHORT_MAX_OUTPUT", 0.90)
    min_ret60 = _env_float("MEME_LEADER_HIGH_CONVICTION_SHORT_MAX_RET60_PCT", -4.0)
    min_ret5 = _env_float("MEME_LEADER_HIGH_CONVICTION_SHORT_MIN_RET5_PCT", -14.5)
    max_ret5 = _env_float("MEME_LEADER_HIGH_CONVICTION_SHORT_MAX_RET5_PCT", 3.0)
    max_ret1 = _env_float("MEME_LEADER_HIGH_CONVICTION_SHORT_MAX_RET1_PCT", 5.0)
    min_from_low15 = _env_float("MEME_LEADER_HIGH_CONVICTION_SHORT_MIN_FROM_LOW15_PCT", 0.55)
    max_from_low15 = _env_float("MEME_LEADER_HIGH_CONVICTION_SHORT_MAX_FROM_LOW15_PCT", 7.0)
    max_from_high60 = _env_float("MEME_LEADER_HIGH_CONVICTION_SHORT_MAX_FROM_HIGH60_PCT", -5.0)
    min_vol_ratio = _env_float("MEME_LEADER_HIGH_CONVICTION_SHORT_MIN_VOL_RATIO", 1.45)
    max_vol_ratio = _env_float("MEME_LEADER_HIGH_CONVICTION_SHORT_MAX_VOL_RATIO", 14.0)
    max_taker_buy_ratio = _env_float("MEME_LEADER_HIGH_CONVICTION_SHORT_MAX_TAKER_BUY_RATIO", 0.68)

    close_np = close.to_numpy(dtype=float)
    ret1 = _pct_last(close_np, 1)
    ret5 = _pct_last(close_np, 5)
    ret15 = _pct_last(close_np, 15)
    ret60 = _pct_last(close_np, 60)
    ret240 = _pct_last(close_np, 240)

    close_last = float(close.iloc[-1])
    high60 = float(high.iloc[-60:].max())
    low15 = float(low.iloc[-15:].min())
    low60 = float(low.iloc[-60:].min())
    low240 = float(low.iloc[-240:].min())
    from_high60 = (close_last / high60 - 1.0) * 100.0 if high60 > 0 else float("nan")
    from_low15 = (close_last / low15 - 1.0) * 100.0 if low15 > 0 else float("nan")
    from_low60 = (close_last / low60 - 1.0) * 100.0 if low60 > 0 else float("nan")
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

    bar_range = float(high.iloc[-1] - low.iloc[-1])
    close_pos = (close_last - float(low.iloc[-1])) / bar_range if bar_range > 0 else 0.5
    upper_wick = (float(high.iloc[-1]) - max(float(open_.iloc[-1]), close_last)) / bar_range if bar_range > 0 else 0.0

    fills = {
        "ret1": 0.0 if not np.isfinite(ret1) else ret1,
        "ret5": 0.0 if not np.isfinite(ret5) else ret5,
        "ret15": 0.0 if not np.isfinite(ret15) else ret15,
        "ret60": 0.0 if not np.isfinite(ret60) else ret60,
        "ret240": 0.0 if not np.isfinite(ret240) else ret240,
        "from_high60": 0.0 if not np.isfinite(from_high60) else from_high60,
        "from_low15": 0.0 if not np.isfinite(from_low15) else from_low15,
        "from_low60": 0.0 if not np.isfinite(from_low60) else from_low60,
        "from_low240": 0.0 if not np.isfinite(from_low240) else from_low240,
    }

    valid = (
        fills["ret60"] <= min_ret60
        and fills["ret5"] >= min_ret5
        and fills["ret5"] <= max_ret5
        and fills["ret1"] <= max_ret1
        and fills["from_low15"] >= min_from_low15
        and fills["from_low15"] <= max_from_low15
        and fills["from_low60"] >= 0.25
        and fills["from_high60"] <= max_from_high60
        and vol_ratio >= min_vol_ratio
        and vol_ratio <= max_vol_ratio
        and taker_buy_ratio <= max_taker_buy_ratio
        and fills["ret15"] <= 1.5
        and fills["ret240"] <= 16.0
    )
    if not valid:
        return 0.0

    symbol_strength = _clip01_scalar(short_score / 88.0)
    cascade = _clip01_scalar((-fills["ret60"] - abs(min_ret60)) / 14.0)
    fresh_break = _clip01_scalar((max_ret5 - fills["ret5"]) / max(max_ret5 - min_ret5, 1.0))
    rebound_room = _clip01_scalar((fills["from_low15"] - min_from_low15) / max(max_from_low15 - min_from_low15, 1.0))
    downside_room = _clip01_scalar((fills["from_low240"] - 0.3) / 8.0)
    high_break = _clip01_scalar((-fills["from_high60"] - abs(max_from_high60)) / 12.0)
    activity = _clip01_scalar((vol_ratio - min_vol_ratio) / 5.0) * (
        1.0 - _clip01_scalar((vol_ratio - max_vol_ratio) / 10.0)
    )
    sell_pressure = (
        0.48 * _clip01_scalar((max_taker_buy_ratio - taker_buy_ratio) / max(max_taker_buy_ratio - 0.38, 0.08))
        + 0.26 * _clip01_scalar((0.82 - close_pos) / 0.62)
        + 0.16 * _clip01_scalar(upper_wick / 0.35)
        + 0.10 * _clip01_scalar(max(0.0, float(funding.iloc[-1])) / 0.18)
    )
    flow = _clip01_scalar(abs(oi_roc20) / 8.0)
    score = symbol_strength * (
        0.30 * cascade
        + 0.16 * fresh_break
        + 0.18 * rebound_room
        + 0.12 * downside_room
        + 0.16 * high_break
        + 0.06 * activity
        + 0.02 * sell_pressure
        + 0.00 * flow
    )
    value = -max(0.0, min(max_output, score))
    return value if abs(value) >= min_output else 0.0
