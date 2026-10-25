"""High-win leader reversal scalp factor.

This research factor turns copy-trader closed-position labels into a tradable
structure instead of replaying their orders.  It uses only as-of OHLCV/order
flow plus a leader symbol/direction prior:

* SHORT: local high stall after a mild 15m/60m lift, with room back to the
  recent low and no obvious buyer dominance.
* LONG: local low reclaim after a mild dump, with volume acceptance and no
  continued breakdown.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from cz.trading.crypto.meme.profile_config import profile_bool, profile_float, profile_get, profile_int


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


@dataclass(frozen=True)
class _LeaderTrade:
    symbol: str
    direction: str
    opened_at: datetime
    closed_at: datetime
    return_pct: float


def _parse_dt(raw: object) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("T", " "))
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _parse_direction(raw: object) -> str:
    text = str(raw or "").strip().upper()
    if text in {"LONG", "BUY", "多", "做多"}:
        return "LONG"
    if text in {"SHORT", "SELL", "空", "做空"}:
        return "SHORT"
    return ""


def _to_float(raw: object, default: float = 0.0) -> float:
    try:
        return float(str(raw).strip().replace("%", ""))
    except (TypeError, ValueError):
        return default


@lru_cache(maxsize=16)
def _load_recent_trades(path_text: str, mtime_ns: int) -> tuple[_LeaderTrade, ...]:
    del mtime_ns
    path = Path(path_text).expanduser()
    if not path.exists():
        return ()
    trades: list[_LeaderTrade] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as fp:
            for row in csv.DictReader(fp):
                symbol = str(row.get("标的") or row.get("symbol") or "").upper().replace("/", "").strip()
                direction = _parse_direction(row.get("方向") or row.get("side") or "")
                opened_at = _parse_dt(row.get("开仓时间") or row.get("opened_at") or "")
                closed_at = _parse_dt(row.get("平仓时间") or row.get("closed_at") or "")
                if not symbol or not direction or opened_at is None or closed_at is None:
                    continue
                ret = _to_float(row.get("收益率估算") or row.get("return_pct") or row.get("ROI") or row.get("roi"), 0.0)
                trades.append(_LeaderTrade(symbol, direction, opened_at, closed_at, ret))
    except Exception:
        return ()
    return tuple(sorted(trades, key=lambda item: item.closed_at))


def _recent_prior(symbol: str, index: pd.Index) -> tuple[pd.Series, pd.Series, pd.Series]:
    path_text = str(profile_get("MEME_LEADER_HIGH_WIN_REVERSAL_RECENT_TRADE_FILE", "") or "").strip()
    if not symbol or not path_text:
        empty = pd.Series(0.0, index=index, dtype=float)
        return empty, empty.copy(), empty.copy()
    path = Path(path_text).expanduser()
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        empty = pd.Series(0.0, index=index, dtype=float)
        return empty, empty.copy(), empty.copy()

    trades = tuple(trade for trade in _load_recent_trades(str(path), mtime_ns) if trade.symbol == symbol)
    if not trades:
        empty = pd.Series(0.0, index=index, dtype=float)
        return empty, empty.copy(), empty.copy()

    lookback_hours = profile_int("MEME_LEADER_HIGH_WIN_REVERSAL_RECENT_LOOKBACK_HOURS", 96)
    min_samples = profile_int("MEME_LEADER_HIGH_WIN_REVERSAL_RECENT_MIN_SAMPLES", 2)
    min_win_rate = _env_float("MEME_LEADER_HIGH_WIN_REVERSAL_RECENT_MIN_WIN_RATE", 75.0)
    min_avg_return = _env_float("MEME_LEADER_HIGH_WIN_REVERSAL_RECENT_MIN_AVG_RETURN", 0.2)
    base_score = _env_float("MEME_LEADER_HIGH_WIN_REVERSAL_RECENT_BASE_SCORE", 48.0)

    sign_values: list[float] = []
    score_values: list[float] = []
    sample_values: list[float] = []
    for raw_asof in index:
        asof = pd.Timestamp(raw_asof).to_pydatetime()
        cutoff = asof - timedelta(hours=lookback_hours)
        stats: dict[str, list[float]] = {}
        for trade in trades:
            if trade.closed_at > asof:
                break
            if trade.opened_at < cutoff:
                continue
            stats.setdefault(trade.direction, []).append(trade.return_pct)
        best_direction = ""
        best_tuple = (-10**9, -10**9, -10**9)
        best_score = 0.0
        best_samples = 0
        for direction, returns in stats.items():
            count = len(returns)
            if count < min_samples:
                continue
            wins = sum(1 for value in returns if value > 0)
            win_rate = wins / count * 100.0
            avg_return = float(sum(returns)) / count
            total_return = float(sum(returns))
            if win_rate < min_win_rate or avg_return < min_avg_return:
                continue
            rank_tuple = (total_return, win_rate, count)
            if rank_tuple > best_tuple:
                best_tuple = rank_tuple
                best_direction = direction
                best_score = min(95.0, base_score + win_rate * 0.16 + max(0.0, avg_return) * 1.2)
                best_samples = count
        sign_values.append(1.0 if best_direction == "LONG" else -1.0 if best_direction == "SHORT" else 0.0)
        score_values.append(best_score)
        sample_values.append(float(best_samples))
    return (
        pd.Series(sign_values, index=index, dtype=float),
        pd.Series(score_values, index=index, dtype=float),
        pd.Series(sample_values, index=index, dtype=float),
    )


def _leader_prior(symbol: str, index: pd.Index) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return direction sign, score and sample count at each timestamp.

    The recent-trade path is as-of aware.  Static files are allowed only as a
    fallback for live forward use or explicit research; they are still gated by
    score/sample/win-rate thresholds.
    """
    empty_sign = pd.Series(0.0, index=index, dtype=float)
    empty_score = pd.Series(0.0, index=index, dtype=float)
    empty_samples = pd.Series(0.0, index=index, dtype=float)
    if not symbol:
        return empty_sign, empty_score, empty_samples

    min_score = _env_float("MEME_LEADER_HIGH_WIN_REVERSAL_MIN_EDGE_SCORE", 48.0)
    min_samples = profile_int("MEME_LEADER_HIGH_WIN_REVERSAL_MIN_EDGE_SAMPLES", 2)
    min_win_rate = _env_float("MEME_LEADER_HIGH_WIN_REVERSAL_MIN_EDGE_WIN_RATE", 80.0)
    direction_mode = str(profile_get("MEME_LEADER_HIGH_WIN_REVERSAL_DIRECTION_MODE", "label") or "label")

    signs = empty_sign.copy()
    scores = empty_score.copy()
    samples = empty_samples.copy()

    if profile_bool("MEME_LEADER_HIGH_WIN_REVERSAL_USE_RECENT_ASOF", True):
        signs, scores, samples = _recent_prior(symbol, index)

    if bool((signs != 0.0).any()) or not profile_bool("MEME_LEADER_HIGH_WIN_REVERSAL_ALLOW_STATIC_EDGE", True):
        return signs, scores, samples

    edge_file = str(profile_get("MEME_LEADER_HIGH_WIN_REVERSAL_EDGE_FILE", "") or "").strip()
    edge = None
    try:
        from cz.trading.crypto.meme.leader_edge import leader_symbol_edge, leader_symbol_edge_from_file

        if edge_file:
            edge = leader_symbol_edge_from_file(
                symbol,
                edge_file,
                min_score=min_score,
                min_samples=min_samples,
                min_win_rate=min_win_rate,
                direction_mode=direction_mode,
            )
        else:
            edge = leader_symbol_edge(symbol)
    except Exception:
        edge = None
    if edge is None:
        return signs, scores, samples
    direction = getattr(edge, "direction", "")
    sign = 1.0 if direction == "LONG" else -1.0 if direction == "SHORT" else 0.0
    if sign == 0.0:
        return signs, scores, samples
    score = float(getattr(edge, "score", 0.0) or 0.0)
    count = float(getattr(edge, "samples", 0) or 0)
    if score < min_score or count < min_samples:
        return signs, scores, samples
    return (
        pd.Series(sign, index=index, dtype=float),
        pd.Series(score, index=index, dtype=float),
        pd.Series(count, index=index, dtype=float),
    )


def calculate_factor(df: pd.DataFrame) -> pd.Series:
    if df is None or len(df) < 90:
        return pd.Series(0.0, index=getattr(df, "index", None))

    symbol = _frame_symbol(df)
    prior_sign, prior_score, prior_samples = _leader_prior(symbol, df.index)
    if not bool((prior_sign != 0.0).any()):
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

    min_output = _env_float("MEME_LEADER_HIGH_WIN_REVERSAL_MIN_OUTPUT", 0.24)
    max_output = _env_float("MEME_LEADER_HIGH_WIN_REVERSAL_MAX_OUTPUT", 0.78)

    short_min_ret15 = _env_float("MEME_LEADER_HIGH_WIN_REVERSAL_SHORT_MIN_RET15_PCT", 0.2)
    short_max_ret15 = _env_float("MEME_LEADER_HIGH_WIN_REVERSAL_SHORT_MAX_RET15_PCT", 8.5)
    short_min_ret60 = _env_float("MEME_LEADER_HIGH_WIN_REVERSAL_SHORT_MIN_RET60_PCT", 0.4)
    short_max_ret60 = _env_float("MEME_LEADER_HIGH_WIN_REVERSAL_SHORT_MAX_RET60_PCT", 18.0)
    short_min_from_low20 = _env_float("MEME_LEADER_HIGH_WIN_REVERSAL_SHORT_MIN_FROM_LOW20_PCT", 1.8)
    short_max_from_low20 = _env_float("MEME_LEADER_HIGH_WIN_REVERSAL_SHORT_MAX_FROM_LOW20_PCT", 13.5)
    short_min_from_high20 = _env_float("MEME_LEADER_HIGH_WIN_REVERSAL_SHORT_MIN_FROM_HIGH20_PCT", -3.8)
    short_max_taker_buy = _env_float("MEME_LEADER_HIGH_WIN_REVERSAL_SHORT_MAX_TAKER_BUY_RATIO", 0.58)
    short_max_close_pos = _env_float("MEME_LEADER_HIGH_WIN_REVERSAL_SHORT_MAX_CLOSE_POS", 0.82)

    long_min_ret15 = _env_float("MEME_LEADER_HIGH_WIN_REVERSAL_LONG_MIN_RET15_PCT", -9.5)
    long_max_ret15 = _env_float("MEME_LEADER_HIGH_WIN_REVERSAL_LONG_MAX_RET15_PCT", 1.2)
    long_min_ret60 = _env_float("MEME_LEADER_HIGH_WIN_REVERSAL_LONG_MIN_RET60_PCT", -18.0)
    long_max_ret60 = _env_float("MEME_LEADER_HIGH_WIN_REVERSAL_LONG_MAX_RET60_PCT", 2.5)
    long_max_from_high20 = _env_float("MEME_LEADER_HIGH_WIN_REVERSAL_LONG_MAX_FROM_HIGH20_PCT", -1.0)
    long_min_from_low20 = _env_float("MEME_LEADER_HIGH_WIN_REVERSAL_LONG_MIN_FROM_LOW20_PCT", 0.2)
    long_max_from_low20 = _env_float("MEME_LEADER_HIGH_WIN_REVERSAL_LONG_MAX_FROM_LOW20_PCT", 6.5)
    long_min_taker_buy = _env_float("MEME_LEADER_HIGH_WIN_REVERSAL_LONG_MIN_TAKER_BUY_RATIO", 0.40)
    long_min_close_pos = _env_float("MEME_LEADER_HIGH_WIN_REVERSAL_LONG_MIN_CLOSE_POS", 0.22)

    min_vol_ratio = _env_float("MEME_LEADER_HIGH_WIN_REVERSAL_MIN_VOL_RATIO", 0.75)
    max_vol_ratio = _env_float("MEME_LEADER_HIGH_WIN_REVERSAL_MAX_VOL_RATIO", 6.0)
    max_bar_range = _env_float("MEME_LEADER_HIGH_WIN_REVERSAL_MAX_BAR_RANGE_PCT", 10.0)

    ret1 = close.pct_change(1) * 100.0
    ret5 = close.pct_change(5) * 100.0
    ret15 = close.pct_change(15) * 100.0
    ret60 = close.pct_change(60) * 100.0

    high20 = high.rolling(20, min_periods=8).max()
    low20 = low.rolling(20, min_periods=8).min()
    high60 = high.rolling(60, min_periods=20).max()
    low60 = low.rolling(60, min_periods=20).min()
    from_high20 = (close / high20.replace(0.0, np.nan) - 1.0) * 100.0
    from_low20 = (close / low20.replace(0.0, np.nan) - 1.0) * 100.0
    from_high60 = (close / high60.replace(0.0, np.nan) - 1.0) * 100.0
    from_low60 = (close / low60.replace(0.0, np.nan) - 1.0) * 100.0

    vol_base = volume.rolling(30, min_periods=12).median().shift(1).replace(0.0, np.nan)
    vol_ratio = (volume / vol_base).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    taker_buy_ratio = (
        taker_buy_quote / quote_volume.replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.50)
    oi_roc10 = oi.pct_change(10).replace([np.inf, -np.inf], np.nan).fillna(0.0) * 100.0

    bar_range = (high - low).replace(0.0, np.nan)
    close_pos = (close - low) / bar_range
    upper_wick = (high - pd.concat([open_, close], axis=1).max(axis=1)) / bar_range
    lower_wick = (pd.concat([open_, close], axis=1).min(axis=1) - low) / bar_range
    bar_range_pct = (high / low.replace(0.0, np.nan) - 1.0) * 100.0

    score_prior = _clip01((prior_score - _env_float("MEME_LEADER_HIGH_WIN_REVERSAL_MIN_EDGE_SCORE", 48.0)) / 35.0)
    sample_prior = _clip01((prior_samples - 1.0) / 8.0)
    activity = _clip01((vol_ratio - min_vol_ratio) / 2.8) * (1.0 - _clip01((vol_ratio - max_vol_ratio) / 5.0))
    oi_activity = _clip01(oi_roc10.abs() / 12.0)

    short_context = (
        0.24 * _clip01((ret15.fillna(0.0) - short_min_ret15) / max(short_max_ret15 - short_min_ret15, 1.0))
        + 0.20 * _clip01((ret60.fillna(0.0) - short_min_ret60) / max(short_max_ret60 - short_min_ret60, 1.0))
        + 0.18 * (1.0 - _clip01((-from_high20.fillna(-999.0)) / 4.5))
        + 0.16 * _clip01((from_low20.fillna(0.0) - short_min_from_low20) / 9.0)
        + 0.10 * _clip01((short_max_close_pos - close_pos.fillna(0.5)) / max(short_max_close_pos, 0.1))
        + 0.06 * _clip01(upper_wick.fillna(0.0) / 0.42)
        + 0.06 * activity
    )
    short_valid = (
        (prior_sign < 0)
        & (ret15 >= short_min_ret15)
        & (ret15 <= short_max_ret15)
        & (ret60 >= short_min_ret60)
        & (ret60 <= short_max_ret60)
        & (ret5 <= 7.5)
        & (ret1 <= 2.0)
        & (from_high20 >= short_min_from_high20)
        & (from_low20 >= short_min_from_low20)
        & (from_low20 <= short_max_from_low20)
        & (from_low60 <= 22.0)
        & (vol_ratio >= min_vol_ratio)
        & (vol_ratio <= max_vol_ratio)
        & (taker_buy_ratio <= short_max_taker_buy)
        & ((close_pos.fillna(0.5) <= short_max_close_pos) | (upper_wick.fillna(0.0) >= 0.08))
        & (bar_range_pct <= max_bar_range)
        & (funding.fillna(0.0) <= 0.35)
    )

    long_context = (
        0.24 * _clip01((long_max_ret15 - ret15.fillna(0.0)) / max(long_max_ret15 - long_min_ret15, 1.0))
        + 0.20 * _clip01((long_max_ret60 - ret60.fillna(0.0)) / max(long_max_ret60 - long_min_ret60, 1.0))
        + 0.18 * _clip01((-from_high20.fillna(0.0) - abs(long_max_from_high20)) / 7.0)
        + 0.16 * (1.0 - _clip01((from_low20.fillna(0.0) - long_max_from_low20) / 8.0))
        + 0.10 * _clip01((close_pos.fillna(0.5) - long_min_close_pos) / 0.55)
        + 0.06 * _clip01(lower_wick.fillna(0.0) / 0.45)
        + 0.06 * activity
    )
    long_valid = (
        (prior_sign > 0)
        & (ret15 >= long_min_ret15)
        & (ret15 <= long_max_ret15)
        & (ret60 >= long_min_ret60)
        & (ret60 <= long_max_ret60)
        & (ret5 >= -7.5)
        & (ret1 >= -2.8)
        & (from_high20 <= long_max_from_high20)
        & (from_high60 >= -18.0)
        & (from_low20 >= long_min_from_low20)
        & (from_low20 <= long_max_from_low20)
        & (from_low60 <= 18.0)
        & (vol_ratio >= min_vol_ratio)
        & (vol_ratio <= max_vol_ratio)
        & (taker_buy_ratio >= long_min_taker_buy)
        & ((close_pos.fillna(0.5) >= long_min_close_pos) | (lower_wick.fillna(0.0) >= 0.08))
        & (bar_range_pct <= max_bar_range)
        & (funding.fillna(0.0) >= -0.35)
    )

    common_quality = 0.58 * score_prior + 0.18 * sample_prior + 0.16 * activity + 0.08 * oi_activity
    short_score = (0.66 * short_context + 0.34 * common_quality).clip(lower=0.0, upper=max_output)
    long_score = (0.66 * long_context + 0.34 * common_quality).clip(lower=0.0, upper=max_output)

    out = pd.Series(0.0, index=df.index)
    out = out.mask(long_valid & (long_score >= min_output), long_score)
    out = out.mask(short_valid & (short_score >= min_output), -short_score)
    return out.where(out.abs() >= min_output, 0.0).fillna(0.0)


def calculate_last_factor(df: pd.DataFrame) -> float:
    if df is None or len(df) < 90:
        return 0.0

    window = df.tail(260).copy()
    symbol = _frame_symbol(window)
    prior_index = pd.Index([window.index[-1]])
    prior_sign, prior_score, prior_samples = _leader_prior(symbol, prior_index)
    if prior_sign.empty:
        return 0.0
    sign = float(prior_sign.iloc[-1])
    if abs(sign) <= 1e-12:
        return 0.0

    open_ = _safe_series(window, "open")
    high = _safe_series(window, "high")
    low = _safe_series(window, "low")
    close = _safe_series(window, "close")
    volume = _safe_series(window, "volume", 1.0)
    quote_volume = _safe_series(window, "quote_volume")
    taker_buy_quote = _safe_series(window, "taker_buy_quote")
    oi = _safe_series(window, "open_interest")
    funding = _safe_series(window, "funding_rate")

    min_output = _env_float("MEME_LEADER_HIGH_WIN_REVERSAL_MIN_OUTPUT", 0.24)
    max_output = _env_float("MEME_LEADER_HIGH_WIN_REVERSAL_MAX_OUTPUT", 0.78)

    short_min_ret15 = _env_float("MEME_LEADER_HIGH_WIN_REVERSAL_SHORT_MIN_RET15_PCT", 0.2)
    short_max_ret15 = _env_float("MEME_LEADER_HIGH_WIN_REVERSAL_SHORT_MAX_RET15_PCT", 8.5)
    short_min_ret60 = _env_float("MEME_LEADER_HIGH_WIN_REVERSAL_SHORT_MIN_RET60_PCT", 0.4)
    short_max_ret60 = _env_float("MEME_LEADER_HIGH_WIN_REVERSAL_SHORT_MAX_RET60_PCT", 18.0)
    short_min_from_low20 = _env_float("MEME_LEADER_HIGH_WIN_REVERSAL_SHORT_MIN_FROM_LOW20_PCT", 1.8)
    short_max_from_low20 = _env_float("MEME_LEADER_HIGH_WIN_REVERSAL_SHORT_MAX_FROM_LOW20_PCT", 13.5)
    short_min_from_high20 = _env_float("MEME_LEADER_HIGH_WIN_REVERSAL_SHORT_MIN_FROM_HIGH20_PCT", -3.8)
    short_max_taker_buy = _env_float("MEME_LEADER_HIGH_WIN_REVERSAL_SHORT_MAX_TAKER_BUY_RATIO", 0.58)
    short_max_close_pos = _env_float("MEME_LEADER_HIGH_WIN_REVERSAL_SHORT_MAX_CLOSE_POS", 0.82)

    long_min_ret15 = _env_float("MEME_LEADER_HIGH_WIN_REVERSAL_LONG_MIN_RET15_PCT", -9.5)
    long_max_ret15 = _env_float("MEME_LEADER_HIGH_WIN_REVERSAL_LONG_MAX_RET15_PCT", 1.2)
    long_min_ret60 = _env_float("MEME_LEADER_HIGH_WIN_REVERSAL_LONG_MIN_RET60_PCT", -18.0)
    long_max_ret60 = _env_float("MEME_LEADER_HIGH_WIN_REVERSAL_LONG_MAX_RET60_PCT", 2.5)
    long_max_from_high20 = _env_float("MEME_LEADER_HIGH_WIN_REVERSAL_LONG_MAX_FROM_HIGH20_PCT", -1.0)
    long_min_from_low20 = _env_float("MEME_LEADER_HIGH_WIN_REVERSAL_LONG_MIN_FROM_LOW20_PCT", 0.2)
    long_max_from_low20 = _env_float("MEME_LEADER_HIGH_WIN_REVERSAL_LONG_MAX_FROM_LOW20_PCT", 6.5)
    long_min_taker_buy = _env_float("MEME_LEADER_HIGH_WIN_REVERSAL_LONG_MIN_TAKER_BUY_RATIO", 0.40)
    long_min_close_pos = _env_float("MEME_LEADER_HIGH_WIN_REVERSAL_LONG_MIN_CLOSE_POS", 0.22)

    min_vol_ratio = _env_float("MEME_LEADER_HIGH_WIN_REVERSAL_MIN_VOL_RATIO", 0.75)
    max_vol_ratio = _env_float("MEME_LEADER_HIGH_WIN_REVERSAL_MAX_VOL_RATIO", 6.0)
    max_bar_range = _env_float("MEME_LEADER_HIGH_WIN_REVERSAL_MAX_BAR_RANGE_PCT", 10.0)

    ret1_s = close.pct_change(1) * 100.0
    ret5_s = close.pct_change(5) * 100.0
    ret15_s = close.pct_change(15) * 100.0
    ret60_s = close.pct_change(60) * 100.0

    high20 = high.rolling(20, min_periods=8).max()
    low20 = low.rolling(20, min_periods=8).min()
    high60 = high.rolling(60, min_periods=20).max()
    low60 = low.rolling(60, min_periods=20).min()
    from_high20_s = (close / high20.replace(0.0, np.nan) - 1.0) * 100.0
    from_low20_s = (close / low20.replace(0.0, np.nan) - 1.0) * 100.0
    from_high60_s = (close / high60.replace(0.0, np.nan) - 1.0) * 100.0
    from_low60_s = (close / low60.replace(0.0, np.nan) - 1.0) * 100.0

    vol_base = volume.rolling(30, min_periods=12).median().shift(1).replace(0.0, np.nan)
    vol_ratio_s = (volume / vol_base).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    taker_buy_ratio_s = (
        taker_buy_quote / quote_volume.replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.50)
    oi_roc10_s = oi.pct_change(10).replace([np.inf, -np.inf], np.nan).fillna(0.0) * 100.0

    bar_range = (high - low).replace(0.0, np.nan)
    close_pos_s = (close - low) / bar_range
    upper_wick_s = (high - pd.concat([open_, close], axis=1).max(axis=1)) / bar_range
    lower_wick_s = (pd.concat([open_, close], axis=1).min(axis=1) - low) / bar_range
    bar_range_pct_s = (high / low.replace(0.0, np.nan) - 1.0) * 100.0

    def _last(series: pd.Series, default: float = 0.0) -> float:
        try:
            value = float(series.iloc[-1])
        except Exception:
            return default
        return value if np.isfinite(value) else default

    ret1 = _last(ret1_s)
    ret5 = _last(ret5_s)
    ret15 = _last(ret15_s)
    ret60 = _last(ret60_s)
    from_high20 = _last(from_high20_s, -999.0)
    from_high60 = _last(from_high60_s, -999.0)
    from_low20 = _last(from_low20_s)
    from_low60 = _last(from_low60_s)
    vol_ratio = _last(vol_ratio_s, 1.0)
    taker_buy_ratio = _last(taker_buy_ratio_s, 0.50)
    oi_roc10 = _last(oi_roc10_s)
    close_pos = _last(close_pos_s, 0.5)
    upper_wick = _last(upper_wick_s)
    lower_wick = _last(lower_wick_s)
    bar_range_pct = _last(bar_range_pct_s)
    funding_value = _last(funding)
    score_value = float(prior_score.iloc[-1])
    sample_value = float(prior_samples.iloc[-1])

    score_prior = float(np.clip((score_value - _env_float("MEME_LEADER_HIGH_WIN_REVERSAL_MIN_EDGE_SCORE", 48.0)) / 35.0, 0.0, 1.0))
    sample_prior = float(np.clip((sample_value - 1.0) / 8.0, 0.0, 1.0))
    activity = float(np.clip((vol_ratio - min_vol_ratio) / 2.8, 0.0, 1.0)) * (
        1.0 - float(np.clip((vol_ratio - max_vol_ratio) / 5.0, 0.0, 1.0))
    )
    oi_activity = float(np.clip(abs(oi_roc10) / 12.0, 0.0, 1.0))

    common_quality = 0.58 * score_prior + 0.18 * sample_prior + 0.16 * activity + 0.08 * oi_activity

    if sign < 0:
        short_context = (
            0.24 * float(np.clip((ret15 - short_min_ret15) / max(short_max_ret15 - short_min_ret15, 1.0), 0.0, 1.0))
            + 0.20 * float(np.clip((ret60 - short_min_ret60) / max(short_max_ret60 - short_min_ret60, 1.0), 0.0, 1.0))
            + 0.18 * (1.0 - float(np.clip((-from_high20) / 4.5, 0.0, 1.0)))
            + 0.16 * float(np.clip((from_low20 - short_min_from_low20) / 9.0, 0.0, 1.0))
            + 0.10 * float(np.clip((short_max_close_pos - close_pos) / max(short_max_close_pos, 0.1), 0.0, 1.0))
            + 0.06 * float(np.clip(upper_wick / 0.42, 0.0, 1.0))
            + 0.06 * activity
        )
        valid = (
            ret15 >= short_min_ret15
            and ret15 <= short_max_ret15
            and ret60 >= short_min_ret60
            and ret60 <= short_max_ret60
            and ret5 <= 7.5
            and ret1 <= 2.0
            and from_high20 >= short_min_from_high20
            and from_low20 >= short_min_from_low20
            and from_low20 <= short_max_from_low20
            and from_low60 <= 22.0
            and vol_ratio >= min_vol_ratio
            and vol_ratio <= max_vol_ratio
            and taker_buy_ratio <= short_max_taker_buy
            and (close_pos <= short_max_close_pos or upper_wick >= 0.08)
            and bar_range_pct <= max_bar_range
            and funding_value <= 0.35
        )
        if valid:
            value = -float(np.clip(0.66 * short_context + 0.34 * common_quality, 0.0, max_output))
            return value if abs(value) >= min_output else 0.0
        return 0.0

    long_context = (
        0.24 * float(np.clip((long_max_ret15 - ret15) / max(long_max_ret15 - long_min_ret15, 1.0), 0.0, 1.0))
        + 0.20 * float(np.clip((long_max_ret60 - ret60) / max(long_max_ret60 - long_min_ret60, 1.0), 0.0, 1.0))
        + 0.18 * float(np.clip((-from_high20 - abs(long_max_from_high20)) / 7.0, 0.0, 1.0))
        + 0.16 * (1.0 - float(np.clip((from_low20 - long_max_from_low20) / 8.0, 0.0, 1.0)))
        + 0.10 * float(np.clip((close_pos - long_min_close_pos) / 0.55, 0.0, 1.0))
        + 0.06 * float(np.clip(lower_wick / 0.45, 0.0, 1.0))
        + 0.06 * activity
    )
    valid = (
        ret15 >= long_min_ret15
        and ret15 <= long_max_ret15
        and ret60 >= long_min_ret60
        and ret60 <= long_max_ret60
        and ret5 >= -7.5
        and ret1 >= -2.8
        and from_high20 <= long_max_from_high20
        and from_high60 >= -18.0
        and from_low20 >= long_min_from_low20
        and from_low20 <= long_max_from_low20
        and from_low60 <= 18.0
        and vol_ratio >= min_vol_ratio
        and vol_ratio <= max_vol_ratio
        and taker_buy_ratio >= long_min_taker_buy
        and (close_pos >= long_min_close_pos or lower_wick >= 0.08)
        and bar_range_pct <= max_bar_range
        and funding_value >= -0.35
    )
    if not valid:
        return 0.0
    try:
        value = float(np.clip(0.66 * long_context + 0.34 * common_quality, 0.0, max_output))
        return value if abs(value) >= min_output else 0.0
    except Exception:
        return 0.0
