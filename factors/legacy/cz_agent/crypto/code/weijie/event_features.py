"""Combo-event and leg-ranking factors for Weijie Quant."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from factors.crypto.code.weijie.common import (
    direction_from_action,
    finite_float,
    weijie_asset_group,
    weijie_market_prior_group,
)
from factors.crypto.code.weijie.positioning import POSITIONING_COLUMNS, enrich_signal_positioning
from factors.crypto.code.weijie.quality_features import build_signal_quality_feature_frame

EVENT_NUMERIC_FEATURES = (
    "signal_count",
    "unique_symbols",
    "max_score",
    "mean_score",
    "event_pressure",
    "short_momentum",
    "mid_momentum",
    "long_momentum",
    "mean_abs_momentum",
    "mean_don",
    "mean_taker",
    "mean_seller",
    "max_qvr",
    "mean_qvr",
    "mean_abs_oi_change12",
    "mean_abs_oi_value_change12",
    "mean_oi_value_ratio96",
    "max_abs_funding_rate",
    "mean_funding_rate",
    "mean_rsi",
    "max_adx",
    "market_down",
    "market_up",
    "mean_atr_pct",
    "mean_cluster_age_min",
    "max_cluster_age_min",
    "mean_cluster_count",
    "cluster_mature_fraction",
    "session_volatility_state",
    "session_liquidity_state",
)
EVENT_CATEGORICAL_FEATURES = ("action",)
SIGNAL_EVENT_NUMERIC_COLUMNS = (
    "ret20",
    "ret60",
    "ret144",
    "ret288",
    "don",
    "taker",
    "seller",
    "qvr",
    "rsi",
    "adx",
    "atr_pct",
    "down",
    "up",
    *POSITIONING_COLUMNS,
)


def build_signal_event_feature_frame(
    signals: pd.DataFrame,
    *,
    bucket_minutes: int,
    enrich_positioning: bool,
) -> pd.DataFrame:
    frame = build_signal_quality_feature_frame(signals)
    if frame.empty:
        return frame
    frame["signal_time"] = pd.to_datetime(frame["signal_time"], errors="coerce")
    frame = frame.dropna(subset=["signal_time", "symbol", "action"]).copy()
    frame["event_bucket"] = frame["signal_time"].dt.floor(f"{max(1, int(bucket_minutes))}min")
    for col in SIGNAL_EVENT_NUMERIC_COLUMNS:
        if col not in frame:
            frame[col] = 0.0
        frame[col] = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if enrich_positioning:
        frame = enrich_signal_positioning(frame)
    for col in POSITIONING_COLUMNS:
        if col not in frame:
            frame[col] = 0.0
        frame[col] = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return frame.sort_values(["event_bucket", "action", "symbol", "signal_time", "score"]).reset_index(drop=True)


def build_event_metrics(group: pd.DataFrame, *, bucket: pd.Timestamp) -> dict[str, float]:
    def numeric(name: str, default: float = 0.0) -> pd.Series:
        values = group.get(name, pd.Series(default, index=group.index, dtype=float))
        return pd.to_numeric(values, errors="coerce").fillna(default)

    score = numeric("score")
    ret20 = numeric("ret20")
    ret60 = numeric("ret60")
    ret144 = numeric("ret144")
    ret288 = numeric("ret288")
    don = numeric("don")
    taker = numeric("taker")
    seller = numeric("seller")
    qvr = numeric("qvr")
    oi_change12 = numeric("oi_change12")
    oi_value_change12 = numeric("oi_value_change12")
    oi_value_ratio96 = numeric("oi_value_ratio96")
    funding_rate = numeric("funding_rate")
    rsi = numeric("rsi")
    adx = numeric("adx")
    atr_pct = numeric("atr_pct")
    down = numeric("down")
    up = numeric("up")
    cluster_age = pd.to_numeric(
        group.get("signal_cluster_age_min", pd.Series(0.0, index=group.index)),
        errors="coerce",
    ).fillna(0.0)
    cluster_count = pd.to_numeric(
        group.get("signal_cluster_count", pd.Series(1.0, index=group.index)),
        errors="coerce",
    ).fillna(1.0)
    previous_gap = pd.to_numeric(
        group.get("signal_prev_gap_min", pd.Series(np.nan, index=group.index)),
        errors="coerce",
    )
    reason = group.get("reason_family", pd.Series(dtype=str)).astype(str)
    symbols = group.get("symbol", pd.Series("", index=group.index, dtype=str)).astype(str)
    asset_groups = symbols.map(weijie_asset_group)
    asset_denominator = float(max(1, len(asset_groups)))
    eth_count = float(asset_groups.eq("eth").sum())
    stock_gold_count = float(asset_groups.eq("stock_gold").sum())
    meme_count = float(asset_groups.eq("meme").sum())
    unique_symbols = int(group["symbol"].astype(str).nunique()) if "symbol" in group else 0
    event_pressure = (
        float(np.sqrt(max(1, len(group))) * (score.max() + qvr.max() + max(abs(ret20).max(), abs(ret60).max()) / 2.0))
        + unique_symbols / 3.0
        + max(float(up.max()), float(down.max()))
    )
    short_momentum = float(np.clip((0.65 * ret20.mean() + 0.35 * ret60.mean()) / 3.0, -1.0, 1.0))
    mid_momentum = float(np.clip((0.45 * ret60.mean() + 0.55 * ret144.mean()) / 5.0, -1.0, 1.0))
    long_momentum = float(np.clip((0.40 * ret144.mean() + 0.60 * ret288.mean()) / 8.0, -1.0, 1.0))
    mean_atr_pct = float(atr_pct.mean())
    session_volatility_state = float(np.clip(mean_atr_pct / 2.0, 0.0, 1.0))
    session_liquidity_state = float(np.clip(np.log1p(max(float(qvr.mean()), 0.0)) / np.log(6.0), 0.0, 1.0))
    minute_of_day = int(bucket.hour * 60 + bucket.minute)
    return {
        "signal_count": float(len(group)),
        "unique_symbols": float(unique_symbols),
        "max_score": float(score.max()),
        "mean_score": float(score.mean()),
        "score_std": float(score.std(ddof=0)) if len(score) > 1 else 0.0,
        "max_ret20": float(ret20.max()),
        "min_ret20": float(ret20.min()),
        "mean_abs_ret20": float(ret20.abs().mean()),
        "max_abs_ret20": float(ret20.abs().max()),
        "max_ret60": float(ret60.max()),
        "min_ret60": float(ret60.min()),
        "mean_abs_ret60": float(ret60.abs().mean()),
        "max_abs_ret60": float(ret60.abs().max()),
        "mean_ret144": float(ret144.mean()),
        "mean_ret288": float(ret288.mean()),
        "mean_don": float(don.mean()),
        "max_don": float(don.max()),
        "min_don": float(don.min()),
        "mean_taker": float(taker.mean()),
        "mean_seller": float(seller.mean()),
        "max_qvr": float(qvr.max()),
        "mean_qvr": float(qvr.mean()),
        "max_oi_change12": float(oi_change12.max()),
        "min_oi_change12": float(oi_change12.min()),
        "mean_abs_oi_change12": float(oi_change12.abs().mean()),
        "max_oi_value_change12": float(oi_value_change12.max()),
        "min_oi_value_change12": float(oi_value_change12.min()),
        "mean_abs_oi_value_change12": float(oi_value_change12.abs().mean()),
        "mean_oi_value_ratio96": float(oi_value_ratio96.mean()),
        "max_abs_funding_rate": float(funding_rate.abs().max()),
        "mean_funding_rate": float(funding_rate.mean()),
        "mean_rsi": float(rsi.mean()),
        "max_adx": float(adx.max()),
        "market_down": float(down.max()),
        "market_up": float(up.max()),
        "has_eth": float(symbols.str.upper().eq("ETHUSDT").any()),
        "has_paxg": float(symbols.str.upper().eq("PAXGUSDT").any()),
        "eth_count": eth_count,
        "stock_gold_count": stock_gold_count,
        "meme_count": meme_count,
        "eth_fraction": eth_count / asset_denominator,
        "stock_gold_fraction": stock_gold_count / asset_denominator,
        "meme_fraction": meme_count / asset_denominator,
        "trend_long_count": float(reason.eq("core_trend_long").sum()),
        "downtrend_short_count": float(reason.eq("core_downtrend_short").sum()),
        "anchor_count": float(reason.str.contains("anchor", regex=False).sum()),
        "session_count": float(reason.str.contains("session", regex=False).sum()),
        "event_pressure": float(event_pressure),
        "short_momentum": short_momentum,
        "mid_momentum": mid_momentum,
        "long_momentum": long_momentum,
        "mean_abs_momentum": float(np.mean(np.abs([short_momentum, mid_momentum, long_momentum]))),
        "mean_atr_pct": mean_atr_pct,
        "mean_cluster_age_min": float(cluster_age.mean()),
        "max_cluster_age_min": float(cluster_age.max()),
        "mean_cluster_count": float(cluster_count.mean()),
        "max_cluster_count": float(cluster_count.max()),
        "cluster_onset_fraction": float(cluster_age.le(15.0).mean()),
        "cluster_mature_fraction": float(cluster_age.between(60.0, 240.0, inclusive="both").mean()),
        "session_volatility_state": session_volatility_state,
        "session_liquidity_state": session_liquidity_state,
        "mean_prev_gap_min": float(previous_gap.mean()) if previous_gap.notna().any() else 0.0,
        "hour": float(bucket.hour),
        "dow": float(bucket.dayofweek),
        "minutes_from_midnight": float(minute_of_day),
        "dist_to_session_min": float(distance_to_session_minute(minute_of_day)),
    }


def score_event_leg(row: pd.Series, *, prior: Any, active: dict[tuple[str, str], dict[str, Any]], for_close: bool) -> float:
    symbol = str(row.get("symbol") or "")
    action = str(row.get("action") or "")
    family = str(row.get("reason_family") or "")
    base = finite_float(row.get("score"))
    qvr = finite_float(row.get("qvr"))
    ret20 = abs(finite_float(row.get("ret20")))
    ret60 = abs(finite_float(row.get("ret60")))
    oi_move = abs(finite_float(row.get("oi_value_change12")))
    funding_abs = abs(finite_float(row.get("funding_rate"))) * 10000.0
    asset_group = weijie_market_prior_group(symbol)
    asset_group_prior = prior.asset_group_action.get((asset_group, action), 0.0)
    family_prior = prior.family_action.get((family, action), 0.0)
    direction = direction_from_action(action)
    state_key = (symbol, direction)
    state_bonus = 1.5 if for_close and state_key in active else 0.0
    return float(
        base
        + 0.35 * qvr
        + 0.08 * ret20
        + 0.05 * ret60
        + 0.04 * oi_move
        + 0.02 * funding_abs
        + 3.0 * asset_group_prior
        + 2.0 * family_prior
        + state_bonus,
    )


def event_session_family(ts: pd.Timestamp) -> str:
    minute = int(ts.hour * 60 + ts.minute)
    if minute <= 10:
        return "midnight"
    if 6 * 60 - 15 <= minute <= 6 * 60 + 45:
        return "asia_morning"
    if 10 * 60 + 15 <= minute <= 10 * 60 + 45:
        return "late_morning_cleanup"
    if 17 * 60 - 15 <= minute <= 17 * 60 + 15:
        return "evening_cleanup"
    if 20 * 60 <= minute <= 23 * 60:
        return "night_trend"
    return "regular"


def distance_to_session_minute(minute: int) -> int:
    anchors = (0, 5, 6 * 60, 6 * 60 + 30, 10 * 60 + 30, 17 * 60, 20 * 60, 23 * 60)
    return int(min(min(abs(minute - anchor), 1440 - abs(minute - anchor)) for anchor in anchors))
