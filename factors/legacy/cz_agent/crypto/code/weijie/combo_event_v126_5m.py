"""Archived V126 OHLCV combo-event factor.

The adapter has no account, exchange, deployment, or leader-data dependency.
It builds the retained K-line features, runs the template action families,
then applies combo-event compression before emitting an open/close pulse.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from cz.trading.crypto.factors.action_codec import DEFAULT_ACTION_SIGNAL_VALUES, signal_value_for_action
from factors.crypto.code.weijie.combo_event_strategy import (
    apply_close_session_gate,
    apply_recall_compression_event_gate,
    apply_sparse_production_event_gate,
    build_event_feature_frame,
    production_open_factor_quality_allowed,
    production_open_factor_quality_floor,
    production_open_factor_quality_score,
)
from factors.crypto.code.weijie.common import weijie_asset_group
from factors.crypto.code.weijie.event_features import build_signal_event_feature_frame
from factors.crypto.code.weijie.market_regime import (
    market_regime_open_allowed,
    score_market_regime,
)
from factors.crypto.code.weijie.risk_config import resolve_dynamic_risk_sizing
from factors.crypto.code.weijie.signal_templates import build_close_factor_signals, build_open_factor_signals

DEFAULT_PARAMETERS: dict[str, Any] = {
    "kline_interval": "5m",
    # The factor needs 1008 feature/event warmup bars plus the current closed
    # decision endpoint.
    "warmup_bars": 1009,
    "signal_type": "pulse",
    "signal_value": 0.82,
    "close_signal_value": 0.62,
    "action_signal_values": dict(DEFAULT_ACTION_SIGNAL_VALUES),
    "open_bucket_minutes": 30,
    "close_bucket_minutes": 30,
    "open_cooldown_minutes": 0.0,
    "close_cooldown_minutes": 0.0,
    "enable_reversal_open_factors": True,
    "enable_sparse_gate": True,
    "close_session_gate_only": False,
    "close_session_min_unique_symbols": 1,
    "close_priority_bonus": 0.18,
    "factor_risk_sizing": True,
    "risk_budget_pct": 0.35,
    "atr_stop_multiplier": 2.20,
    "min_stop_distance_pct": 0.80,
    "max_stop_distance_pct": 4.00,
    "max_single_margin_pct": 3.00,
    "eth_leverage": 12.0,
    "stock_gold_leverage": 10.0,
    "meme_leverage": 8.0,
    "quality_weighted_sizing": True,
    "quality_multiplier_min": 0.75,
    "quality_multiplier_max": 1.50,
}


@dataclass(frozen=True)
class _TemplateConfig:
    open_cooldown_minutes: float = 0.0
    close_cooldown_minutes: float = 0.0
    enable_reversal_open_factors: bool = True


def calculate_factor(df: pd.DataFrame, parameters: dict | None = None) -> pd.Series:
    params = dict(DEFAULT_PARAMETERS)
    params.update(parameters or {})
    result = pd.Series(0.0, index=df.index if df is not None else None, dtype="float64")
    input_df = df
    warmup_bars = max(120, int(float(params.get("warmup_bars") or DEFAULT_PARAMETERS["warmup_bars"])))
    if input_df is not None and len(input_df) > warmup_bars:
        input_df = input_df.tail(warmup_bars)

    if input_df is None or input_df.empty:
        return result
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(set(input_df.columns)):
        return result

    symbol = _clean_symbol(input_df.attrs.get("symbol") or params.get("symbol") or "")
    if not symbol:
        return result

    bars = _normalise_live_bars(input_df, symbol)
    if bars.empty:
        return result

    features = _build_live_feature_rows(bars)
    if features.empty:
        return result

    template_cfg = _TemplateConfig(
        open_cooldown_minutes=float(params.get("open_cooldown_minutes") or 0.0),
        close_cooldown_minutes=float(params.get("close_cooldown_minutes") or 0.0),
        enable_reversal_open_factors=bool(params.get("enable_reversal_open_factors", True)),
    )
    open_signals = build_open_factor_signals(features, config=template_cfg)
    close_signals = build_close_factor_signals(features, config=template_cfg)
    selected_signals = _select_v126_event_signals(open_signals, close_signals, params)
    if selected_signals.empty:
        result.attrs["factor_actions"] = pd.Series("", index=result.index, dtype="object")
        result.attrs["factor_latest_risk"] = {}
        return result

    selected_signals = selected_signals.copy()
    selected_signals["signal_time"] = pd.to_datetime(selected_signals["signal_time"], errors="coerce")
    selected_signals = selected_signals.dropna(subset=["signal_time", "action"])
    if selected_signals.empty:
        result.attrs["factor_actions"] = pd.Series("", index=result.index, dtype="object")
        return result

    signal_values = dict(params.get("action_signal_values") or DEFAULT_ACTION_SIGNAL_VALUES)
    close_bonus = float(params.get("close_priority_bonus") or 0.0)
    selected_signals["_effective_score"] = pd.to_numeric(selected_signals.get("score"), errors="coerce").fillna(0.0)
    selected_signals.loc[selected_signals["action"].astype(str).str.startswith("CLOSE_"), "_effective_score"] += close_bonus
    selected_signals = selected_signals.sort_values(["signal_time", "_effective_score", "score"], ascending=[True, False, False])
    selected_signals = selected_signals.drop_duplicates(["signal_time"], keep="first")

    time_to_index = {
        pd.Timestamp(row["bar_time"]): row["_source_index"]
        for _, row in bars.dropna(subset=["bar_time", "_source_index"]).iterrows()
    }
    actions = pd.Series("", index=result.index, dtype="object")
    risk_by_index: dict[str, dict[str, Any]] = {}
    for _, row in selected_signals.iterrows():
        ts = pd.Timestamp(row["signal_time"])
        idx = time_to_index.get(ts)
        if idx is None:
            continue
        action = str(row.get("action") or "").upper()
        value = signal_value_for_action(
            action,
            signal_value=float(params.get("signal_value") or 0.82),
            close_signal_value=float(params.get("close_signal_value") or 0.62),
            params={"action_signal_values": signal_values},
        )
        if np.isfinite(value) and value != 0.0:
            result.loc[idx] = float(value)
            actions.loc[idx] = action
            if action in {"OPEN_LONG", "OPEN_SHORT"} and bool(params.get("factor_risk_sizing", True)):
                risk = _resolve_live_factor_risk(row, params)
                if risk:
                    risk_by_index[str(idx)] = risk
    cleaned = result.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype("float64")
    cleaned.attrs["factor_actions"] = actions
    cleaned.attrs["factor_risk_by_index"] = risk_by_index
    cleaned.attrs["factor_latest_risk"] = risk_by_index.get(str(cleaned.index[-1]), {}) if len(cleaned) else {}
    return cleaned


def _resolve_live_factor_risk(
    row: pd.Series,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    action = str(row.get("action") or "").upper()
    symbol = str(row.get("symbol") or "").upper()
    if action not in {"OPEN_LONG", "OPEN_SHORT"} or not symbol:
        return {}

    asset_group = weijie_asset_group(symbol)
    leverage_keys = {
        "eth": "eth_leverage",
        "stock_gold": "stock_gold_leverage",
        "meme": "meme_leverage",
    }
    leverage = _finite_float(params.get(leverage_keys[asset_group]), 6.0)
    atr_pct = _finite_float(row.get("atr_pct"), 0.0)
    if leverage <= 0.0 or atr_pct <= 0.0:
        return {}

    multiplier = 1.0
    if bool(params.get("quality_weighted_sizing", True)):
        quality = production_open_factor_quality_score(row)
        floor = production_open_factor_quality_floor(row)
        if np.isfinite(quality) and np.isfinite(floor) and floor > 0.0:
            multiplier = float(
                np.clip(
                    quality / floor,
                    float(params.get("quality_multiplier_min") or 0.75),
                    float(params.get("quality_multiplier_max") or 1.50),
                ),
            )
    decision = resolve_dynamic_risk_sizing(
        symbol=symbol,
        action=action,
        account_equity_usdt=100.0,
        price=max(1.0, _finite_float(row.get("close"), 1.0)),
        atr_pct=atr_pct,
        position_lot_count=0,
        quality_multiplier=multiplier,
        risk_budget_pct=_finite_float(params.get("risk_budget_pct"), 0.35),
        atr_stop_multiplier=_finite_float(params.get("atr_stop_multiplier"), 2.20),
        min_stop_distance_pct=_finite_float(params.get("min_stop_distance_pct"), 0.80),
        max_stop_distance_pct=_finite_float(params.get("max_stop_distance_pct"), 4.00),
        max_single_margin_pct=_finite_float(params.get("max_single_margin_pct"), 3.00),
        max_leverage=leverage,
    )
    if decision.margin_pct <= 0.0:
        return {}
    return {
        "action": action,
        "symbol": symbol,
        "asset_group": asset_group,
        "leverage": float(decision.leverage),
        "margin_pct": float(decision.margin_pct),
        "risk_basis_usdt": 0.0,
        "risk_budget_pct": float(decision.risk_budget_pct),
        "stop_distance_pct": float(decision.stop_distance_pct),
        "atr_pct": float(decision.atr_pct),
        "quality_multiplier": multiplier,
        "sizing_reason": decision.sizing_reason,
    }


def _select_v126_event_signals(open_signals: pd.DataFrame, close_signals: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    selected_frames: list[pd.DataFrame] = []
    open_bucket = int(float(params.get("open_bucket_minutes") or 30))
    close_bucket = int(float(params.get("close_bucket_minutes") or 30))
    if open_signals is not None and not open_signals.empty:
        open_event_signals = build_signal_event_feature_frame(
            open_signals,
            bucket_minutes=open_bucket,
            enrich_positioning=False,
        )
        open_events = build_event_feature_frame(open_event_signals, matches=None, bucket_minutes=open_bucket, family="open")
        open_events = _seed_live_event_gate_columns(open_events)
        open_events = apply_recall_compression_event_gate(open_events, family="open")
        if bool(params.get("enable_sparse_gate", True)):
            open_events = apply_sparse_production_event_gate(open_events, family="open")
        selected_frames.append(_filter_live_open_quality(_signals_for_selected_events(open_event_signals, open_events)))

    if close_signals is not None and not close_signals.empty:
        close_event_signals = build_signal_event_feature_frame(
            close_signals,
            bucket_minutes=close_bucket,
            enrich_positioning=False,
        )
        close_events = build_event_feature_frame(close_event_signals, matches=None, bucket_minutes=close_bucket, family="close")
        close_events = _seed_live_event_gate_columns(close_events)
        close_events = apply_close_session_gate(
            close_events,
            session_gate_only=bool(params.get("close_session_gate_only", False)),
            min_unique_symbols=int(float(params.get("close_session_min_unique_symbols") or 1)),
        )
        close_events = apply_recall_compression_event_gate(close_events, family="close")
        if bool(params.get("enable_sparse_gate", True)):
            close_events = apply_sparse_production_event_gate(close_events, family="close")
        selected_frames.append(_signals_for_selected_events(close_event_signals, close_events))

    selected_frames = [frame for frame in selected_frames if frame is not None and not frame.empty]
    if not selected_frames:
        return pd.DataFrame(columns=["signal_id", "symbol", "action", "signal_time", "score", "reason"])
    return pd.concat(selected_frames, ignore_index=True, sort=False)


def _seed_live_event_gate_columns(events: pd.DataFrame) -> pd.DataFrame:
    if events is None or events.empty:
        return events
    out = events.copy()
    for col in ("model_gate_selected", "pressure_gate_selected", "close_session_gate_selected", "selected_event"):
        if col not in out:
            out[col] = 0
    if "event_score" not in out:
        out["event_score"] = 0.0
    out["event_score"] = pd.to_numeric(out["event_score"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _signals_for_selected_events(signals: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    if signals is None or events is None or signals.empty or events.empty:
        return pd.DataFrame(columns=signals.columns if signals is not None else [])
    selected = events[pd.to_numeric(events.get("selected_event"), errors="coerce").fillna(0).astype(int).eq(1)]
    if selected.empty:
        return pd.DataFrame(columns=signals.columns)
    ids: set[str] = set()
    for raw in selected.get("signal_ids", pd.Series(dtype=str)).astype(str):
        ids.update(item for item in raw.split(",") if item)
    if not ids:
        return pd.DataFrame(columns=signals.columns)
    return signals[signals["signal_id"].astype(str).isin(ids)].copy()


def _filter_live_open_quality(signals: pd.DataFrame) -> pd.DataFrame:
    if signals is None or signals.empty:
        return signals
    out = signals.copy()
    out["factor_quality_score"] = [
        production_open_factor_quality_score(row) for _, row in out.iterrows()
    ]
    out["factor_quality_floor"] = [
        production_open_factor_quality_floor(row) for _, row in out.iterrows()
    ]
    regimes = [score_market_regime(row) for _, row in out.iterrows()]
    for field in (
        "trend_strength",
        "liquidity_condition",
        "crowding_score",
        "reversal_probability",
        "market_regime_score",
    ):
        out[field] = [getattr(regime, field) for regime in regimes]
    keep = [_live_open_quality_allowed(row) for _, row in out.iterrows()]
    return out.loc[keep].copy()


def _live_open_quality_allowed(row: pd.Series) -> bool:
    if not production_open_factor_quality_allowed(row):
        return False
    symbol = str(row.get("symbol") or "")
    action = str(row.get("action") or "")
    family = str(row.get("reason_family") or "")
    signal_time = pd.Timestamp(row.get("signal_time")) if row.get("signal_time") is not None else pd.NaT
    hour = int(signal_time.hour) if pd.notna(signal_time) else -1
    ret20 = _finite_float(row.get("ret20"), 0.0)
    ret60 = _finite_float(row.get("ret60"), 0.0)
    ret144 = _finite_float(row.get("ret144"), 0.0)
    ret288 = _finite_float(row.get("ret288"), 0.0)
    don = _finite_float(row.get("don"), 0.5)
    taker = _finite_float(row.get("taker"), 0.5)
    seller = _finite_float(row.get("seller"), 0.5)
    qvr = _finite_float(row.get("qvr"), 0.0)
    rsi = _finite_float(row.get("rsi"), 50.0)
    down = _finite_float(row.get("down"), 0.0)
    if action == "OPEN_LONG" and family == "core_trend_long":
        return market_regime_open_allowed(row)
    if action == "OPEN_LONG" and family == "capitulation_absorption_long":
        eth_shallow_tail_absorption = (
            symbol == "ETHUSDT"
            and ret20 > -0.80
            and ret60 > -1.80
            and ret144 > -1.00
            and ret288 > -3.00
            and qvr < 2.0
            and rsi > 30.0
        )
        ltc_session_absorption_trap = (
            symbol == "LTCUSDT"
            and hour in {8, 13, 16, 21}
            and ret60 <= -1.00
            and don <= 0.30
            and rsi <= 38.0
        )
        return not (eth_shallow_tail_absorption or ltc_session_absorption_trap)
    if action == "OPEN_SHORT" and family == "eth_anchor_exhaustion_short":
        eth_late_overextended_anchor = (
            symbol == "ETHUSDT"
            and ret60 >= 2.50
            and don >= 0.90
            and qvr >= 5.0
            and rsi >= 75.0
        )
        return not eth_late_overextended_anchor
    if action == "OPEN_SHORT" and family == "core_downtrend_short":
        eth_weak_breakdown_chase = (
            symbol == "ETHUSDT"
            and ret60 <= -3.00
            and ret144 <= -2.50
            and don <= 0.30
            and rsi <= 48.0
            and qvr < 1.20
            and down >= 0.50
        )
        cfx_evening_crowded_short_chase = (
            symbol == "CFXUSDT"
            and 19 <= hour <= 21
            and qvr >= 2.0
            and 27.0 <= rsi <= 42.0
            and down >= 0.85
            and ret20 > -1.50
            and not (seller >= 0.80 and ret144 > -0.50)
        )
        cfx_session_false_breakdown = (
            symbol == "CFXUSDT"
            and (
                (hour == 22 and ret144 <= -1.20 and qvr <= 5.0 and rsi <= 33.0)
                or (hour == 6 and ret60 <= -3.00 and qvr < 3.0)
                or (hour == 6 and ret144 >= 0.50 and ret288 >= 1.00 and qvr <= 1.50)
                or (hour == 16 and ret144 <= -1.00 and seller <= 0.55 and qvr < 3.0)
            )
        )
        return not (eth_weak_breakdown_chase or cfx_evening_crowded_short_chase or cfx_session_false_breakdown)
    return True


def _finite_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _normalise_live_bars(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["_source_index"] = list(df.index)
    out["bar_time"] = pd.to_datetime(df.index, errors="coerce")
    out["symbol"] = symbol
    for col in ("open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_quote"):
        if col in df:
            out[col] = pd.to_numeric(df[col], errors="coerce")
    if "quote_volume" not in out:
        out["quote_volume"] = pd.to_numeric(out.get("volume"), errors="coerce") * pd.to_numeric(out.get("close"), errors="coerce")
    if "taker_buy_quote" not in out:
        out["taker_buy_quote"] = np.nan
    if "trades" not in out:
        out["trades"] = np.nan
    out = out.dropna(subset=["bar_time", "open", "high", "low", "close", "volume"]).copy()
    return out.reset_index(drop=True)


def _build_live_feature_rows(bars: pd.DataFrame) -> pd.DataFrame:
    out = bars.sort_values("bar_time").copy()
    close = pd.to_numeric(out["close"], errors="coerce").astype("float64")
    high = pd.to_numeric(out["high"], errors="coerce").astype("float64")
    low = pd.to_numeric(out["low"], errors="coerce").astype("float64")
    quote_volume = pd.to_numeric(out.get("quote_volume"), errors="coerce").astype("float64")
    for window in (1, 3, 5, 12, 20, 36, 60, 72, 144, 240, 288):
        out[f"ret{window}"] = _pct_change(close, close.shift(window))
    low60 = low.rolling(60, min_periods=20).min()
    high60 = high.rolling(60, min_periods=20).max()
    out["donchian60_pos"] = ((close - low60) / (high60 - low60).replace(0.0, np.nan)).clip(0.0, 1.0)
    qv_base = quote_volume.shift(1).rolling(30, min_periods=5).median()
    out["quote_volume_ratio_med30"] = quote_volume / qv_base.replace(0.0, np.nan)
    taker_buy_quote = pd.to_numeric(out.get("taker_buy_quote"), errors="coerce").astype("float64")
    out["taker_buy_ratio"] = (taker_buy_quote / quote_volume.replace(0.0, np.nan)).clip(0.0, 1.0).fillna(0.5)
    out["seller_pressure"] = 1.0 - out["taker_buy_ratio"]
    out["rsi14"] = _rsi(close, 14)
    out["adx14"] = _adx(high, low, close, 14)
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            (high - low).abs(),
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr14 = true_range.ewm(alpha=1.0 / 14.0, adjust=False, min_periods=7).mean()
    out["atr14_pct"] = atr14 / close.replace(0.0, np.nan) * 100.0
    out["market_down_breadth20"] = pd.to_numeric(out["ret20"], errors="coerce").lt(0.0).astype(float)
    out["market_up_breadth20"] = pd.to_numeric(out["ret20"], errors="coerce").gt(0.0).astype(float)
    feature_cols = [
        "ret20",
        "ret60",
        "ret144",
        "ret288",
        "donchian60_pos",
        "taker_buy_ratio",
        "seller_pressure",
        "quote_volume_ratio_med30",
        "rsi14",
        "adx14",
        "atr14_pct",
        "market_down_breadth20",
        "market_up_breadth20",
    ]
    for col in feature_cols:
        out[col] = pd.to_numeric(out.get(col), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["taker_buy_ratio"] = out["taker_buy_ratio"].replace(0.0, 0.5)
    out["seller_pressure"] = out["seller_pressure"].replace(0.0, 0.5)
    out["quote_volume_ratio_med30"] = out["quote_volume_ratio_med30"].replace(0.0, 1.0)
    out["rsi14"] = out["rsi14"].replace(0.0, 50.0)
    out["adx14"] = out["adx14"].replace(0.0, 20.0)
    return out.reset_index(drop=True)


def _pct_change(current: pd.Series, previous: pd.Series) -> pd.Series:
    return ((current / previous.replace(0.0, np.nan)) - 1.0) * 100.0


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / window, adjust=False, min_periods=max(2, window // 2)).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / window, adjust=False, min_periods=max(2, window // 2)).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)
    plus_dm = (high - prev_high).where((high - prev_high) > (prev_low - low), 0.0).clip(lower=0.0)
    minus_dm = (prev_low - low).where((prev_low - low) > (high - prev_high), 0.0).clip(lower=0.0)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / window, adjust=False, min_periods=max(2, window // 2)).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / window, adjust=False, min_periods=max(2, window // 2)).mean() / atr.replace(0.0, np.nan)
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / window, adjust=False, min_periods=max(2, window // 2)).mean() / atr.replace(0.0, np.nan)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan) * 100.0
    return dx.ewm(alpha=1.0 / window, adjust=False, min_periods=max(2, window // 2)).mean()


def _clean_symbol(value: object) -> str:
    return str(value or "").upper().replace("/", "").replace("-", "").strip()
