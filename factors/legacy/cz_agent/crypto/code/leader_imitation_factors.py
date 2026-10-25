"""Runtime leader-imitation factor helpers.

These helpers approximate structures observed in external high-performing
lead-trader samples without reading future trades at runtime.
"""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
import os
from pathlib import Path

import numpy as np
import pandas as pd


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def safe_series(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in df:
        return pd.to_numeric(df[name], errors="coerce").ffill().fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def clip01(value: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    if isinstance(value, pd.Series):
        return value.clip(lower=0.0, upper=1.0)
    return np.clip(value, 0.0, 1.0)


def base_features(df: pd.DataFrame) -> dict[str, pd.Series]:
    close = safe_series(df, "close")
    open_ = safe_series(df, "open")
    high = safe_series(df, "high")
    low = safe_series(df, "low")
    volume = safe_series(df, "volume", 1.0)
    oi = safe_series(df, "open_interest")
    funding = safe_series(df, "funding_rate")

    high20 = high.rolling(20, min_periods=8).max()
    low20 = low.rolling(20, min_periods=8).min()
    high60 = high.rolling(60, min_periods=20).max()
    low60 = low.rolling(60, min_periods=20).min()
    bar_range = (high - low).replace(0.0, np.nan)
    upper_wick = (high - pd.concat([open_, close], axis=1).max(axis=1)) / bar_range
    lower_wick = (pd.concat([open_, close], axis=1).min(axis=1) - low) / bar_range
    close_pos = (close - low) / bar_range
    vol_base = volume.rolling(30, min_periods=12).median().shift(1).replace(0.0, np.nan)

    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=7).mean()
    atr_pct = atr / close.replace(0.0, np.nan) * 100.0

    return {
        "close": close,
        "high": high,
        "low": low,
        "ret3": close.pct_change(3) * 100.0,
        "ret5": close.pct_change(5) * 100.0,
        "ret15": close.pct_change(15) * 100.0,
        "ret60": close.pct_change(60) * 100.0,
        "from_high20": (close / high20.replace(0.0, np.nan) - 1.0) * 100.0,
        "from_low20": (close / low20.replace(0.0, np.nan) - 1.0) * 100.0,
        "from_high60": (close / high60.replace(0.0, np.nan) - 1.0) * 100.0,
        "from_low60": (close / low60.replace(0.0, np.nan) - 1.0) * 100.0,
        "upper_wick": upper_wick.fillna(0.0),
        "lower_wick": lower_wick.fillna(0.0),
        "close_pos": close_pos.fillna(0.5),
        "vol_ratio": (volume / vol_base).replace([np.inf, -np.inf], np.nan).fillna(1.0),
        "oi_roc5": oi.pct_change(5).replace([np.inf, -np.inf], np.nan).fillna(0.0) * 100.0,
        "oi_roc20": oi.pct_change(20).replace([np.inf, -np.inf], np.nan).fillna(0.0) * 100.0,
        "funding": funding.fillna(0.0),
        "atr_pct": atr_pct.fillna(0.0),
        "ema8": close.ewm(span=8, min_periods=5, adjust=False).mean(),
        "ema21": close.ewm(span=21, min_periods=10, adjust=False).mean(),
    }


def leader_entry_quality_signal(df: pd.DataFrame) -> pd.Series:
    if df is None or len(df) < 90:
        return pd.Series(0.0, index=getattr(df, "index", None))
    f = base_features(df)
    min_output = env_float("MEME_LEADER_ENTRY_QUALITY_MIN_OUTPUT", 0.24)
    max_output = env_float("MEME_LEADER_ENTRY_QUALITY_MAX_OUTPUT", 0.82)

    trend_long = clip01((f["ret15"].fillna(0.0) + f["ret60"].fillna(0.0) * 0.45) / 22.0)
    trend_short = clip01((-f["ret15"].fillna(0.0) - f["ret60"].fillna(0.0) * 0.45) / 22.0)
    reclaim_long = ((f["close"] > f["ema8"]) & (f["ema8"] >= f["ema21"] * 0.997)).astype(float)
    reclaim_short = ((f["close"] < f["ema8"]) & (f["ema8"] <= f["ema21"] * 1.003)).astype(float)
    controlled_long = 1.0 - clip01((f["ret5"].fillna(0.0) - 7.0) / 18.0)
    controlled_short = 1.0 - clip01((-f["ret5"].fillna(0.0) - 7.0) / 18.0)
    pullback_long = clip01((-f["from_high20"].fillna(0.0) - 0.3) / 5.0) * (1.0 - clip01((f["from_low20"].fillna(0.0) - 18.0) / 24.0))
    pullback_short = clip01((f["from_low20"].fillna(0.0) - 0.3) / 5.0) * (1.0 - clip01((-f["from_high20"].fillna(0.0) - 18.0) / 24.0))
    vol_quality = clip01((f["vol_ratio"] - 0.9) / 2.8) * (1.0 - clip01((f["vol_ratio"] - 10.0) / 20.0))
    oi_long = clip01(f["oi_roc5"].fillna(0.0) / 10.0)
    oi_short = clip01(f["oi_roc5"].fillna(0.0) / 12.0)
    close_long = clip01((f["close_pos"] - 0.45) / 0.45)
    close_short = clip01((0.55 - f["close_pos"]) / 0.45)

    long_score = (
        0.20 * trend_long
        + 0.16 * reclaim_long
        + 0.15 * controlled_long
        + 0.14 * pullback_long
        + 0.12 * vol_quality
        + 0.08 * oi_long
        + 0.10 * close_long
        + 0.05 * clip01(-f["funding"].fillna(0.0) / 0.30)
    )
    short_score = (
        0.20 * trend_short
        + 0.16 * reclaim_short
        + 0.15 * controlled_short
        + 0.14 * pullback_short
        + 0.12 * vol_quality
        + 0.08 * oi_short
        + 0.10 * close_short
        + 0.05 * clip01(f["funding"].fillna(0.0) / 0.30)
    )
    raw = pd.Series(0.0, index=df.index)
    raw = raw.mask(long_score >= short_score, long_score)
    raw = raw.mask(short_score > long_score, -short_score)
    raw = raw.clip(lower=-max_output, upper=max_output)
    return raw.where(raw.abs() >= min_output, 0.0).fillna(0.0)


def leader_no_chase_guard_signal(df: pd.DataFrame) -> pd.Series:
    if df is None or len(df) < 80:
        return pd.Series(0.0, index=getattr(df, "index", None))
    f = base_features(df)
    min_output = env_float("MEME_LEADER_NO_CHASE_MIN_OUTPUT", 0.12)
    max_output = env_float("MEME_LEADER_NO_CHASE_MAX_OUTPUT", 0.72)

    long_chase = (
        0.34 * clip01((f["ret5"].fillna(0.0) - 4.0) / 14.0)
        + 0.22 * clip01((f["ret15"].fillna(0.0) - 9.0) / 26.0)
        + 0.14 * clip01((f["from_low20"].fillna(0.0) - 18.0) / 35.0)
        + 0.12 * clip01((f["from_high20"].fillna(0.0) + 0.5) / 4.0)
        + 0.10 * clip01((f["vol_ratio"] - 6.0) / 18.0)
        + 0.08 * clip01(f["upper_wick"] / 0.45)
    )
    short_chase = (
        0.34 * clip01((-f["ret5"].fillna(0.0) - 4.0) / 14.0)
        + 0.22 * clip01((-f["ret15"].fillna(0.0) - 9.0) / 26.0)
        + 0.14 * clip01((-f["from_high20"].fillna(0.0) - 18.0) / 35.0)
        + 0.12 * clip01((1.0 - f["from_low20"].fillna(0.0)) / 4.0)
        + 0.10 * clip01((f["vol_ratio"] - 6.0) / 18.0)
        + 0.08 * clip01(f["lower_wick"] / 0.45)
    )
    raw = pd.Series(0.0, index=df.index)
    raw = raw.mask(long_chase >= short_chase, -long_chase)
    raw = raw.mask(short_chase > long_chase, short_chase)
    raw = raw.clip(lower=-max_output, upper=max_output)
    return raw.where(raw.abs() >= min_output, 0.0).fillna(0.0)


def leader_dynamic_risk_tier_signal(df: pd.DataFrame) -> pd.Series:
    if df is None or len(df) < 90:
        return pd.Series(0.0, index=getattr(df, "index", None))
    entry = leader_entry_quality_signal(df)
    guard = leader_no_chase_guard_signal(df)
    f = base_features(df)
    min_output = env_float("MEME_LEADER_RISK_TIER_MIN_OUTPUT", 0.22)
    low_noise = 1.0 - clip01((f["atr_pct"].fillna(0.0) - 1.2) / 5.0)
    clean_close = pd.Series(np.where(entry >= 0, f["close_pos"], 1.0 - f["close_pos"]), index=df.index).clip(0.0, 1.0)
    risk_quality = entry.abs() * (0.62 + 0.22 * low_noise + 0.16 * clean_close) * (1.0 - 0.40 * guard.abs())
    raw = np.sign(entry) * risk_quality
    out = pd.Series(raw, index=df.index).clip(-0.8, 0.8)
    return out.where(out.abs() >= min_output, 0.0).fillna(0.0)


def leader_exit_timing_signal(df: pd.DataFrame) -> pd.Series:
    if df is None or len(df) < 90:
        return pd.Series(0.0, index=getattr(df, "index", None))
    f = base_features(df)
    min_output = env_float("MEME_LEADER_EXIT_TIMING_MIN_OUTPUT", 0.22)
    max_output = env_float("MEME_LEADER_EXIT_TIMING_MAX_OUTPUT", 0.70)

    long_giveback_risk = (
        0.25 * clip01((f["ret60"].fillna(0.0) - 25.0) / 50.0)
        + 0.20 * clip01(f["upper_wick"] / 0.45)
        + 0.18 * clip01((-f["ret3"].fillna(0.0)) / 4.0)
        + 0.15 * (f["close"] < f["ema8"]).astype(float)
        + 0.12 * clip01((f["vol_ratio"] - 5.0) / 15.0)
        + 0.10 * clip01((f["atr_pct"].fillna(0.0) - 2.5) / 6.0)
    )
    short_giveback_risk = (
        0.25 * clip01((-f["ret60"].fillna(0.0) - 25.0) / 50.0)
        + 0.20 * clip01(f["lower_wick"] / 0.45)
        + 0.18 * clip01(f["ret3"].fillna(0.0) / 4.0)
        + 0.15 * (f["close"] > f["ema8"]).astype(float)
        + 0.12 * clip01((f["vol_ratio"] - 5.0) / 15.0)
        + 0.10 * clip01((f["atr_pct"].fillna(0.0) - 2.5) / 6.0)
    )
    raw = pd.Series(0.0, index=df.index)
    trend = f["ret60"].fillna(0.0)
    raw = raw.mask(trend >= 0, -long_giveback_risk)
    raw = raw.mask(trend < 0, short_giveback_risk)
    raw = raw.clip(lower=-max_output, upper=max_output)
    return raw.where(raw.abs() >= min_output, 0.0).fillna(0.0)


def leader_meme_symbol_score_signal(df: pd.DataFrame) -> pd.Series:
    if df is None or len(df) == 0:
        return pd.Series(0.0, index=getattr(df, "index", None))
    symbol = _frame_symbol(df)
    if not symbol:
        return pd.Series(0.0, index=df.index)
    scores = _leader_symbol_scores(os.getenv("MEME_LEADER_IMITATION_SYMBOL_FILE", ""))
    if not scores:
        return pd.Series(0.0, index=df.index)
    long_score = scores.get((symbol, "做多"), 0.0)
    short_score = scores.get((symbol, "做空"), 0.0)
    value = 0.0
    if long_score or short_score:
        value = (long_score - short_score) / 100.0
    else:
        value = scores.get((symbol, ""), 0.0) / 100.0
    value = float(np.clip(value, -0.65, 0.65))
    min_output = env_float("MEME_LEADER_SYMBOL_SCORE_MIN_OUTPUT", 0.12)
    if abs(value) < min_output:
        value = 0.0
    return pd.Series(value, index=df.index)


def leader_symbol_fade_short_signal(df: pd.DataFrame) -> pd.Series:
    if df is None or len(df) < 90:
        return pd.Series(0.0, index=getattr(df, "index", None))
    symbol = _frame_symbol(df)
    if not symbol:
        return pd.Series(0.0, index=df.index)

    source_mode = str(os.getenv("MEME_LEADER_SYMBOL_FADE_SOURCE_MODE", "auto") or "auto").strip().lower()
    explicit_score_files = [
        os.getenv("MEME_LEADER_SYMBOL_FADE_FILE", ""),
        os.getenv("MEME_LEADER_IMITATION_SYMBOL_FILE", ""),
    ]
    explicit_score_file = ",".join(path for path in explicit_score_files if str(path or "").strip())
    edge_modes = {"edge", "leader_edge", "shared_edge", "legacy_edge"}
    merge_modes = {"merge", "merged", "all"}
    primary_modes = {"primary", "symbol_fade", "symbol"}
    imitation_modes = {"imitation", "imitate"}
    edge_csv_modes = {"edge_csv", "edge_file"}
    if source_mode in edge_modes:
        use_shared_edge = bool(os.getenv("MEME_LEADER_EDGE_FILE") or env_bool("MEME_LEADER_RECENT_DECAY_ENABLED", False))
    elif source_mode in merge_modes | primary_modes | imitation_modes | edge_csv_modes:
        use_shared_edge = False
    else:
        use_shared_edge = bool(
            not explicit_score_file
            and (os.getenv("MEME_LEADER_EDGE_FILE") or env_bool("MEME_LEADER_RECENT_DECAY_ENABLED", False))
        )
    if use_shared_edge:
        try:
            from cz.trading.crypto.meme.leader_edge import leader_symbol_edge, leader_symbol_edges
        except Exception:
            leader_symbol_edge = None
            leader_symbol_edges = None
        if leader_symbol_edge is None:
            return pd.Series(0.0, index=df.index)
        if env_bool("MEME_LEADER_RECENT_DECAY_ENABLED", False) and leader_symbol_edges is not None:
            short_score = pd.Series(0.0, index=df.index, dtype=float)
            long_score = pd.Series(0.0, index=df.index, dtype=float)
            try:
                edges = leader_symbol_edges(symbol, df.index)
            except Exception:
                edges = ()
            for ts, edge in zip(df.index, edges):
                if edge is not None and edge.direction == "SHORT":
                    short_score.loc[ts] = _age_adjusted_leader_score(float(edge.score), ts, getattr(edge, "latest_open", None))
                elif edge is not None and edge.direction == "LONG":
                    long_score.loc[ts] = _age_adjusted_leader_score(float(edge.score), ts, getattr(edge, "latest_open", None))
        else:
            try:
                edge = leader_symbol_edge(symbol)
            except Exception:
                edge = None
            if edge is None or edge.direction != "SHORT":
                return pd.Series(0.0, index=df.index)
            short_score = pd.Series(float(edge.score), index=df.index, dtype=float)
            long_score = pd.Series(0.0, index=df.index, dtype=float)
    else:
        scores: dict[tuple[str, str], float] = {}
        if source_mode in primary_modes:
            score_files = [
                os.getenv("MEME_LEADER_SYMBOL_FADE_FILE", "")
                or os.getenv("MEME_LEADER_IMITATION_SYMBOL_FILE", "")
                or os.getenv("MEME_LEADER_EDGE_FILE", "")
            ]
        elif source_mode in imitation_modes:
            score_files = [os.getenv("MEME_LEADER_IMITATION_SYMBOL_FILE", "")]
        elif source_mode in edge_csv_modes:
            score_files = [os.getenv("MEME_LEADER_EDGE_FILE", "")]
        else:
            score_files = explicit_score_files + [os.getenv("MEME_LEADER_EDGE_FILE", "")]
        for path_text in score_files:
            for key, value in _leader_symbol_scores(path_text).items():
                scores[key] = max(scores.get(key, 0.0), value)
        short_score = pd.Series(float(scores.get((symbol, "做空"), 0.0)), index=df.index, dtype=float)
        long_score = pd.Series(float(scores.get((symbol, "做多"), 0.0)), index=df.index, dtype=float)
    min_score = env_float("MEME_LEADER_SYMBOL_FADE_MIN_SHORT_SCORE", 45.0)
    edge_valid = (short_score >= min_score) & (short_score > long_score)
    if not bool(edge_valid.any()):
        return pd.Series(0.0, index=df.index)

    f = base_features(df)
    min_output = env_float("MEME_LEADER_SYMBOL_FADE_MIN_OUTPUT", 0.18)
    max_output = env_float("MEME_LEADER_SYMBOL_FADE_MAX_OUTPUT", 0.82)
    max_ret5 = env_float("MEME_LEADER_SYMBOL_FADE_MAX_RET5_PCT", 18.0)
    max_bounce20 = env_float("MEME_LEADER_SYMBOL_FADE_MAX_BOUNCE20_PCT", 35.0)
    max_pullback_high20 = env_float("MEME_LEADER_SYMBOL_FADE_MAX_PULLBACK_HIGH20_PCT", 6.0)
    min_ret5 = env_float("MEME_LEADER_SYMBOL_FADE_MIN_RET5_PCT", -2.8)
    min_ret15 = env_float("MEME_LEADER_SYMBOL_FADE_MIN_RET15_PCT", -4.8)
    min_vol_ratio = env_float("MEME_LEADER_SYMBOL_FADE_MIN_VOL_RATIO", 0.0)
    max_close_pos = env_float("MEME_LEADER_SYMBOL_FADE_MAX_CLOSE_POS", 0.72)
    min_upper_wick = env_float("MEME_LEADER_SYMBOL_FADE_MIN_UPPER_WICK", 0.08)
    max_ret1 = env_float("MEME_LEADER_SYMBOL_FADE_MAX_RET1_PCT", 0.10)
    require_rejection = env_bool("MEME_LEADER_SYMBOL_FADE_REQUIRE_REJECTION", False)

    symbol_strength = (short_score / 70.0).clip(lower=0.0, upper=1.0)
    ret5 = f["ret5"].fillna(0.0)
    ret15 = f["ret15"].fillna(0.0)
    from_low20 = f["from_low20"].fillna(0.0)
    from_high20 = f["from_high20"].fillna(-999.0)
    ret1 = f["close"].pct_change(1).fillna(0.0) * 100.0

    near_high_score = 1.0 - clip01((-from_high20 - 0.2) / max(abs(max_pullback_high20) - 0.2, 0.5))
    micro_bounce_score = (
        0.18 * clip01((ret5 + 0.4) / 4.0)
        + 0.12 * clip01((ret15 + 0.8) / 8.0)
        + 0.10 * clip01((from_low20 - 0.1) / 4.0)
    )
    fail_score = (
        0.16 * (ret1 <= 0.10).astype(float)
        + 0.16 * clip01((f["upper_wick"] - 0.08) / 0.52)
        + 0.14 * clip01((0.72 - f["close_pos"]) / 0.72)
        + 0.08 * (f["close"] <= f["ema8"] * 1.006).astype(float)
    )
    vol_quality = clip01((f["vol_ratio"] - 0.25) / 3.5) * (1.0 - clip01((f["vol_ratio"] - 9.0) / 12.0))
    crowding_score = 0.04 * clip01(f["funding"].fillna(0.0) / 0.20) + 0.03 * clip01(f["oi_roc5"].fillna(0.0) / 10.0)
    score = symbol_strength * (0.36 * near_high_score + micro_bounce_score + 0.12 * vol_quality + fail_score + crowding_score)

    rejection = (ret1 <= max_ret1) | (f["upper_wick"] >= min_upper_wick) | (f["close_pos"] <= max_close_pos)
    close_structure = rejection if require_rejection else (rejection | (f["close"] <= f["ema8"] * 1.006))

    valid = (
        edge_valid
        & ((ret5 >= -0.4) | (ret15 >= -0.8) | (from_low20 >= 0.1))
        & (ret5 >= min_ret5)
        & (ret15 >= min_ret15)
        & (ret5 <= max_ret5)
        & (from_low20 <= max_bounce20)
        & (from_high20 >= -abs(max_pullback_high20))
        & (f["vol_ratio"].fillna(0.0) >= min_vol_ratio)
        & close_structure
    )
    raw = pd.Series(0.0, index=df.index)
    raw = raw.mask(valid, -pd.Series(score, index=df.index).clip(lower=0.0, upper=max_output))
    return raw.where(raw.abs() >= min_output, 0.0).fillna(0.0)


def _age_adjusted_leader_score(score: float, asof: object, latest_open: object | None) -> float:
    if not env_bool("MEME_LEADER_SYMBOL_FADE_AGE_DECAY_ENABLED", False):
        return score
    opened_at = _coerce_naive_datetime(latest_open)
    asof_dt = _coerce_naive_datetime(asof)
    if opened_at is None or asof_dt is None:
        return score
    age_hours = (asof_dt - opened_at).total_seconds() / 3600.0
    if age_hours < 0:
        return 0.0

    full_score_hours = max(0.0, env_float("MEME_LEADER_SYMBOL_FADE_FULL_SCORE_HOURS", 2.0))
    decay_hours = max(full_score_hours, env_float("MEME_LEADER_SYMBOL_FADE_DECAY_HOURS", 24.0))
    min_mult = float(np.clip(env_float("MEME_LEADER_SYMBOL_FADE_MIN_AGE_MULT", 0.35), 0.0, 1.0))
    if age_hours <= full_score_hours or decay_hours <= full_score_hours:
        return score
    if age_hours >= decay_hours:
        return score * min_mult
    progress = (age_hours - full_score_hours) / max(decay_hours - full_score_hours, 1e-9)
    return score * (1.0 - (1.0 - min_mult) * progress)


def _coerce_naive_datetime(value: object | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    return ts.to_pydatetime().replace(tzinfo=None)


def _frame_symbol(df: pd.DataFrame) -> str:
    raw = df.attrs.get("symbol") or df.attrs.get("market_symbol") or ""
    symbol = str(raw).strip().upper().replace("/", "")
    return symbol


@lru_cache(maxsize=8)
def _leader_symbol_scores(path_text: str) -> dict[tuple[str, str], float]:
    if not str(path_text or "").strip():
        return {}
    path = Path(str(path_text)).expanduser()
    if not path.exists():
        return {}
    try:
        frame = pd.read_csv(path)
    except Exception:
        return {}
    if frame.empty or "标的" not in frame:
        return {}
    score_col = "标签_选币评分" if "标签_选币评分" in frame else "选择评分" if "选择评分" in frame else ""
    if not score_col:
        return {}
    side_col = "方向" if "方向" in frame else "优势方向" if "优势方向" in frame else ""
    scores: dict[tuple[str, str], float] = {}
    for _, row in frame.iterrows():
        symbol = str(row.get("标的", "")).strip().upper().replace("/", "")
        if not symbol:
            continue
        side = str(row.get(side_col, "")).strip() if side_col else ""
        try:
            score = float(row.get(score_col, 0.0))
        except (TypeError, ValueError):
            continue
        scores[(symbol, side)] = max(scores.get((symbol, side), 0.0), score)
    return scores
