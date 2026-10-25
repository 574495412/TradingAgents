"""Leader range-scalp short factor.

This research factor targets the high-frequency leader pattern found in the
Bihaijianjin closed-position labels: short-only, local-range entries near the
20/60 minute high, tiny expected price edge, and fast exits.  It does not
replay leader orders; the learned file is only a symbol/direction prior.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


_FALLBACK_SHORT_EDGES: dict[str, tuple[float, int, float]] = {
    "WLDUSDT": (95.0, 81, 100.0),
    "ZECUSDT": (85.9738, 52, 98.0769),
    "NEARUSDT": (81.8763, 38, 100.0),
    "BABYUSDT": (79.5858, 34, 100.0),
    "TRUMPUSDT": (78.7321, 31, 100.0),
    "TAOUSDT": (76.5579, 29, 100.0),
    "OPGUSDT": (75.8675, 20, 100.0),
    "XLMUSDT": (73.2547, 20, 100.0),
    "ALLOUSDT": (75.8865, 16, 100.0),
    "XPLUSDT": (73.0069, 16, 100.0),
    "UNIUSDT": (72.2555, 16, 100.0),
    "HMSTRUSDT": (78.3211, 15, 100.0),
    "ENAUSDT": (72.3532, 14, 100.0),
    "JTOUSDT": (68.3754, 8, 100.0),
    "HOMEUSDT": (68.9275, 7, 100.0),
    "MEGAUSDT": (66.8444, 5, 100.0),
    "ETHUSDT": (64.9330, 3, 100.0),
    "SOLUSDT": (64.0377, 3, 100.0),
    "STGUSDT": (67.5615, 2, 100.0),
    "SAHARAUSDT": (65.9507, 2, 100.0),
}


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


def _frame_symbol(df: pd.DataFrame) -> str:
    raw = df.attrs.get("symbol") or df.attrs.get("market_symbol") or ""
    return str(raw or "").strip().upper().replace("/", "")


def _leader_short_score(symbol: str) -> float:
    min_score = _env_float("MEME_LEADER_RANGE_SCALP_SHORT_MIN_SCORE", 64.0)
    min_samples = int(_env_float("MEME_LEADER_RANGE_SCALP_SHORT_MIN_SAMPLES", 2.0))
    min_win_rate = _env_float("MEME_LEADER_RANGE_SCALP_SHORT_MIN_WIN_RATE", 95.0)
    edge_file = os.getenv(
        "MEME_LEADER_RANGE_SCALP_SHORT_EDGE_FILE",
        "factors/crypto/trading_feedback/leader_learning/meme_bihaijianjin_range_scalp_short_20260617.symbols.csv",
    ).strip()
    if not symbol:
        return 0.0
    if edge_file:
        try:
            from cz.trading.crypto.meme.leader_edge import leader_symbol_edge_from_file

            edge = leader_symbol_edge_from_file(
                symbol,
                edge_file,
                min_score=min_score,
                min_samples=min_samples,
                min_win_rate=min_win_rate,
            )
            if edge is not None and getattr(edge, "direction", "") == "SHORT":
                return float(getattr(edge, "score", 0.0))
        except Exception:
            pass
    score, samples, win_rate = _FALLBACK_SHORT_EDGES.get(symbol, (0.0, 0, 0.0))
    if score < min_score or samples < min_samples or win_rate < min_win_rate:
        return 0.0
    return score


def calculate_factor(df: pd.DataFrame) -> pd.Series:
    if df is None or len(df) < 90:
        return pd.Series(0.0, index=getattr(df, "index", None))

    symbol_score = _leader_short_score(_frame_symbol(df))
    min_score = _env_float("MEME_LEADER_RANGE_SCALP_SHORT_MIN_SCORE", 64.0)
    if symbol_score < min_score:
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

    min_output = _env_float("MEME_LEADER_RANGE_SCALP_SHORT_MIN_OUTPUT", 0.24)
    max_output = _env_float("MEME_LEADER_RANGE_SCALP_SHORT_MAX_OUTPUT", 0.78)
    min_ret1 = _env_float("MEME_LEADER_RANGE_SCALP_SHORT_MIN_RET1_PCT", -0.95)
    max_ret1 = _env_float("MEME_LEADER_RANGE_SCALP_SHORT_MAX_RET1_PCT", 0.72)
    min_ret3 = _env_float("MEME_LEADER_RANGE_SCALP_SHORT_MIN_RET3_PCT", -1.10)
    max_ret3 = _env_float("MEME_LEADER_RANGE_SCALP_SHORT_MAX_RET3_PCT", 1.10)
    min_ret5 = _env_float("MEME_LEADER_RANGE_SCALP_SHORT_MIN_RET5_PCT", -1.35)
    max_ret5 = _env_float("MEME_LEADER_RANGE_SCALP_SHORT_MAX_RET5_PCT", 1.75)
    min_ret15 = _env_float("MEME_LEADER_RANGE_SCALP_SHORT_MIN_RET15_PCT", -2.70)
    max_ret15 = _env_float("MEME_LEADER_RANGE_SCALP_SHORT_MAX_RET15_PCT", 2.35)
    min_ret60 = _env_float("MEME_LEADER_RANGE_SCALP_SHORT_MIN_RET60_PCT", -7.20)
    max_ret60 = _env_float("MEME_LEADER_RANGE_SCALP_SHORT_MAX_RET60_PCT", 5.40)
    min_from_high20 = _env_float("MEME_LEADER_RANGE_SCALP_SHORT_MIN_FROM_HIGH20_PCT", -4.50)
    min_from_high60 = _env_float("MEME_LEADER_RANGE_SCALP_SHORT_MIN_FROM_HIGH60_PCT", -7.00)
    min_from_low20 = _env_float("MEME_LEADER_RANGE_SCALP_SHORT_MIN_FROM_LOW20_PCT", 0.12)
    min_from_low60 = _env_float("MEME_LEADER_RANGE_SCALP_SHORT_MIN_FROM_LOW60_PCT", 0.45)
    min_vol_ratio = _env_float("MEME_LEADER_RANGE_SCALP_SHORT_MIN_VOL_RATIO", 0.22)
    max_vol_ratio = _env_float("MEME_LEADER_RANGE_SCALP_SHORT_MAX_VOL_RATIO", 4.20)
    max_taker_buy_ratio = _env_float("MEME_LEADER_RANGE_SCALP_SHORT_MAX_TAKER_BUY_RATIO", 0.84)
    min_quote_volume = _env_float("MEME_LEADER_RANGE_SCALP_SHORT_MIN_QUOTE_VOLUME", 5000.0)

    ret1 = close.pct_change(1) * 100.0
    ret3 = close.pct_change(3) * 100.0
    ret5 = close.pct_change(5) * 100.0
    ret15 = close.pct_change(15) * 100.0
    ret60 = close.pct_change(60) * 100.0

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
    vol_recent = vol_ratio.rolling(5, min_periods=2).max().fillna(vol_ratio)
    taker_buy_ratio = (
        taker_buy_quote / quote_volume.replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.50)
    oi_roc10 = oi.pct_change(10).replace([np.inf, -np.inf], np.nan).fillna(0.0) * 100.0

    bar_range = (high - low).replace(0.0, np.nan)
    close_pos = (close - low) / bar_range
    upper_wick = (high - pd.concat([open_, close], axis=1).max(axis=1)) / bar_range

    symbol_strength = min(1.0, max(0.0, symbol_score / 95.0))
    local_high = (
        0.62 * (1.0 - _clip01((-from_high20.fillna(-999.0)) / 4.5))
        + 0.38 * (1.0 - _clip01((-from_high60.fillna(-999.0)) / 7.0))
    )
    range_center = (
        0.40 * (1.0 - _clip01(ret60.fillna(0.0).abs() / 7.0))
        + 0.35 * (1.0 - _clip01(ret15.fillna(0.0).abs() / 3.0))
        + 0.25 * (1.0 - _clip01(ret5.fillna(0.0).abs() / 2.2))
    )
    scalp_room = (
        0.58 * _clip01((from_low20.fillna(0.0) - min_from_low20) / 3.2)
        + 0.42 * _clip01((from_low60.fillna(0.0) - min_from_low60) / 5.6)
    )
    stall = (
        0.28 * _clip01((max_ret1 - ret1.fillna(0.0)) / 1.7)
        + 0.22 * _clip01((max_ret5 - ret5.fillna(0.0)) / 3.0)
        + 0.22 * _clip01((0.84 - close_pos.fillna(0.5)) / 0.70)
        + 0.18 * _clip01(upper_wick.fillna(0.0) / 0.32)
        + 0.10 * (close <= open_ * 1.0025).astype(float)
    )
    activity = _clip01((vol_recent - min_vol_ratio) / 1.8) * (
        1.0 - _clip01((vol_ratio - max_vol_ratio) / 5.0)
    )
    sell_pressure = _clip01((max_taker_buy_ratio - taker_buy_ratio) / max(max_taker_buy_ratio - 0.32, 0.08))
    crowding = 0.65 * _clip01(oi_roc10.abs() / 10.0) + 0.35 * _clip01(funding.fillna(0.0).clip(lower=0.0) / 0.20)

    score = symbol_strength * (
        0.26 * local_high
        + 0.22 * range_center
        + 0.18 * scalp_room
        + 0.16 * stall
        + 0.10 * activity
        + 0.05 * sell_pressure
        + 0.03 * crowding
    )
    valid = (
        (ret1 >= min_ret1)
        & (ret1 <= max_ret1)
        & (ret3 >= min_ret3)
        & (ret3 <= max_ret3)
        & (ret5 >= min_ret5)
        & (ret5 <= max_ret5)
        & (ret15 >= min_ret15)
        & (ret15 <= max_ret15)
        & (ret60 >= min_ret60)
        & (ret60 <= max_ret60)
        & (from_high20 >= min_from_high20)
        & (from_high60 >= min_from_high60)
        & (from_low20 >= min_from_low20)
        & (from_low60 >= min_from_low60)
        & (vol_ratio >= min_vol_ratio)
        & (vol_ratio <= max_vol_ratio)
        & (taker_buy_ratio <= max_taker_buy_ratio)
        & (quote_volume.fillna(0.0) >= min_quote_volume)
        & ((close_pos.fillna(0.5) <= 0.88) | (upper_wick.fillna(0.0) >= 0.04) | (ret1 <= 0.28))
    )

    out = pd.Series(0.0, index=df.index)
    out = out.mask(valid, -score.clip(lower=0.0, upper=max_output))
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
