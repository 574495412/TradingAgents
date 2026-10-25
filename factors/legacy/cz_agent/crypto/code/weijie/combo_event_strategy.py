"""Reusable walk-forward combo-event factor and ledger primitives.

This module preserves the market-only event compression, quality scoring and
ledger primitives from the retired Weijie strategy.  It belongs to the factor
layer and has no live-account, exchange-order or deployment dependency.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cz.trading.crypto.risk.liquidation_guard import LiquidationGuardConfig, assess_liquidation_guard
from factors.crypto.code.weijie.common import (
    direction_from_action,
    weijie_asset_group,
    weijie_market_prior_group,
)
from factors.crypto.code.weijie.event_features import (
    EVENT_CATEGORICAL_FEATURES,
    EVENT_NUMERIC_FEATURES,
    build_event_metrics,
    event_session_family,
    score_event_leg,
)
from factors.crypto.code.weijie.market_regime import (
    decide_market_regime,
    market_regime_open_allowed,
)
from factors.crypto.code.weijie.risk_config import (
    RiskSizingDecision,
    build_public_operation_risk_profile,
    resolve_dynamic_risk_sizing,
    resolve_risk_sizing,
)

OPEN_EVENT_NUMERIC_FEATURES = EVENT_NUMERIC_FEATURES


@dataclass(frozen=True)
class EventGate:
    family: str
    model: Any
    categorical: tuple[str, ...]
    numeric: tuple[str, ...]
    threshold: float
    model_name: str


@dataclass(frozen=True)
class LegPrior:
    asset_group_action: dict[tuple[str, str], float]
    family_action: dict[tuple[str, str], float]
    session_state_action: dict[tuple[str, str], float]
    legs_by_action_family: dict[tuple[str, str], int]
    default_legs_by_action: dict[str, int]


@dataclass(frozen=True)
class ForwardQualityGate:
    model: Any
    threshold: float
    train_samples: int
    positive_rate: float
    validation_selected: int
    validation_mean_return_pct: float


@dataclass(frozen=True)
class LeaderLegQualityGate:
    family: str
    model: Any
    threshold: float
    max_legs_per_event: int
    train_samples: int
    positive_samples: int
    validation_selected: int
    validation_prevalence: float
    validation_precision: float
    validation_recall: float
    validation_f1: float


@dataclass(frozen=True)
class WeijieComboEventStrategyConfig:
    interval: str = "5m"
    enrich_futures_klines: bool = True
    enrich_positioning: bool = True
    train_window_days: int = 30
    risk_train_window_days: int = 120
    risk_operation_labels_csv: str = ""
    train_end_safety_minutes: float = 5.0
    open_tolerance_minutes: float = 15.0
    close_tolerance_minutes: float = 15.0
    precision_match_tolerance_minutes: float = 15.0
    precision_match_price_bps: float = 100.0
    open_bucket_minutes: int = 30
    close_bucket_minutes: int = 30
    open_cooldown_minutes: float = 30.0
    close_cooldown_minutes: float = 0.0
    min_hold_minutes: float = 120.0
    enable_reversal_open_factors: bool = False
    max_open_trades_per_bucket: int = 4
    max_close_trades_per_bucket: int = 8
    open_threshold_floor: float = 0.20
    close_threshold_floor: float = 0.20
    open_pressure_quantile: float = 0.90
    close_pressure_quantile: float = 0.85
    pressure_rolling_events: int = 80
    pressure_min_history_events: int = 10
    close_session_gate_only: bool = True
    close_session_min_unique_symbols: int = 6
    recall_first: bool = False
    recall_compression: bool = False
    recall_compression_open_leg_cap: int = 9
    recall_compression_close_leg_cap: int = 28
    recall_state_presence_gate: bool = False
    recall_open_family_score_floors: tuple[tuple[str, float], ...] = ()
    recall_close_score_floor: float = 0.0
    recall_close_family_score_floors: tuple[tuple[str, float], ...] = ()
    production_mode: bool = False
    # Production events aggregate every 5m signal inside a wider event bucket.
    # They may only execute after the final 5m bar in that bucket is closed.
    # Keeping this explicit prevents a full-history backtest from using a
    # later signal in the bucket to create an earlier fill.
    production_causal_event_execution: bool = False
    production_event_compression: bool = False
    production_sparse_event_gate: bool = False
    production_open_transition_gate: bool = False
    production_transition_meme_legs_only: bool = True
    production_transition_reset_minutes: float = 240.0
    production_transition_min_train_recall: float = 1.0
    production_frequency_event_gate: bool = False
    production_open_rank_floor: float = 0.0
    production_close_rank_floor: float = 0.0
    production_open_daily_event_cap: int = 0
    production_close_daily_event_cap: int = 0
    production_open_slot_event_cap: int = 0
    production_close_slot_event_cap: int = 0
    production_open_daily_trade_cap: int = 0
    production_close_daily_trade_cap: int = 0
    production_state_open_cap_window_days: float = 0.0
    production_eth_state_open_cap: int = 0
    production_alt_state_open_cap: int = 0
    production_open_quality_gate: bool = False
    production_unified_quality_only: bool = False
    production_unified_leg_ranking: bool = False
    production_unified_factor_quality_gate: bool = True
    production_quality_floor_multiplier: float = 1.0
    production_forward_quality_gate: bool = False
    production_forward_quality_horizon_minutes: float = 360.0
    production_forward_quality_probability_floor: float = 0.58
    production_leader_leg_quality_gate: bool = False
    production_leader_leg_hard_gate: bool = False
    production_leader_leg_probability_floor: float = 0.30
    production_disabled_open_families: tuple[str, ...] = ()
    production_capitulation_context_gate: bool = False
    production_trend_exhaustion_confirmation_gate: bool = False
    production_directional_context_gate: bool = False
    production_eth_blowoff_handoff: bool = False
    production_eth_blowoff_ladder: bool = False
    production_follower_catchup_addon: bool = False
    production_trend_saturation_gate: bool = False
    production_point_accuracy_gate: bool = False
    production_allow_pyramiding: bool = True
    production_max_lots_per_state: int = 3
    production_eth_max_lots_per_state: int = 0
    production_stock_gold_max_lots_per_state: int = 0
    production_meme_max_lots_per_state: int = 0
    production_meme_max_legs_per_event: int = 0
    production_meme_second_leg_quality_ratio_floor: float = 0.0
    production_eth_anchor_maturity_gate: bool = False
    production_deferred_close_persistence: bool = False
    production_meme_core_trend_confirmation_only: bool = False
    production_meme_core_trend_breadth_bootstrap: bool = False
    production_non_eth_contribution_campaign: bool = False
    production_non_eth_campaign_daily_open_cap: int = 0
    production_non_eth_campaign_symbols: tuple[str, ...] = ()
    production_non_eth_campaign_reentry_gap_minutes: float = 0.0
    production_emit_close_per_lot: bool = False
    production_preserve_lots_on_single_close: bool = False
    production_honor_signal_close_fraction: bool = False
    production_full_reset_close_families: tuple[str, ...] = ()
    production_profit_scaleout_close_families: tuple[str, ...] = ()
    production_min_addon_gap_minutes: float = 0.0
    production_min_reentry_gap_minutes: float = 0.0
    production_open_full_event_budget: bool = False
    production_cluster_open_budget: bool = False
    production_cluster_close_budget: bool = False
    production_close_deferral: bool = False
    production_portfolio_min_hold_minutes: float = 180.0
    production_midnight_cleanup_min_hold_minutes: float = 0.0
    production_asia_cleanup_min_pressure: float = 80.0
    production_evening_cleanup_min_pressure: float = 50.0
    production_late_cleanup_min_pressure: float = 80.0
    production_midnight_cleanup_max_score: float = 3.25
    production_early_profit_min_hold_minutes: float = 480.0
    production_risk_cut_min_event_score: float = 0.05
    production_asia_profit_min_event_score: float = 0.20
    production_unclustered_profit_min_roi_pct: float = 0.0
    production_unclustered_risk_min_loss_pct: float = 0.0
    production_night_short_profit_min_event_score: float = 0.0
    production_regular_short_profit_quality_gate: bool = False
    production_night_short_profit_spike_deferral: bool = False
    production_eth_countertrend_short_exit_confirmation: bool = False
    production_eth_cascade_profit_exit: bool = False
    production_meme_core_short_stop_floor_pct: float = 0.0
    production_prefilter_ineligible_open_legs: bool = False
    production_broad_cluster_third_meme_leg: bool = False
    production_low_quality_meme_short_delay_minutes: float = 0.0
    production_eth_anchor_long_profit_min_hold_minutes: float = 0.0
    production_clustered_meme_short_correlation_cap: float = 0.0
    production_anchor_exit_deferral: bool = False
    production_eth_anchor_short_min_hold_minutes: float = 900.0
    production_inventory_exit_quality_gate: bool = False
    production_regular_short_profit_max_hold_minutes: float = 360.0
    production_early_short_risk_max_hold_minutes: float = 360.0
    account_initial_equity_usdt: float = 10_000.0
    account_capital_allocation_pct: float = 100.0
    account_public_roi_basis_usdt: float = 0.0
    account_per_trade_margin_pct: float = 2.5
    account_max_total_margin_pct: float = 60.0
    account_round_trip_cost_bps: float = 14.0
    account_quality_weighted_sizing: bool = True
    account_risk_prior_weighted_sizing: bool = False
    account_signal_risk_basis_sizing: bool = False
    account_mark_to_market_margin: bool = False
    account_eth_main_leg_margin_floor_pct: float = 0.0
    account_drawdown_throttle_start_pct: float = 0.0
    account_drawdown_throttle_margin_cap_pct: float = 100.0
    account_risk_prior_exponent: float = 0.50
    account_risk_prior_max_multiplier: float = 3.0
    account_cash_flows_csv: str = ""
    ledger_hard_stop_margin_loss_pct: float = 18.141
    risk_non_anchor_leverage_cap: float = 8.5
    risk_anchor_short_leverage_cap: float = 0.0
    risk_eth_core_downtrend_short_leverage_cap: float = 0.0
    random_state: int = 29
    symbols: tuple[str, ...] = ()
    calibration_scope: str = "walk_forward_combo_event_train_before_eval_open"

def build_event_feature_frame(
    signals: pd.DataFrame,
    *,
    matches: pd.DataFrame | None,
    bucket_minutes: int,
    family: str,
) -> pd.DataFrame:
    columns = [
        "event_id",
        "family",
        "event_bucket",
        "action",
        "dominant_family",
        "session_family",
        "symbols",
        "signal_ids",
        "positive_signal_count",
        "event_label",
        "event_score",
        *EVENT_CATEGORICAL_FEATURES,
        *OPEN_EVENT_NUMERIC_FEATURES,
    ]
    if signals.empty:
        return pd.DataFrame(columns=list(dict.fromkeys(columns)))
    frame = signals.copy()
    frame["event_bucket"] = pd.to_datetime(frame["event_bucket"], errors="coerce")
    frame = frame.dropna(subset=["event_bucket", "action"]).copy()
    if frame.empty:
        return pd.DataFrame(columns=list(dict.fromkeys(columns)))
    positive_ids = (
        set(matches[matches["status"].astype(str).eq("MATCH")]["signal_id"].astype(str))
        if matches is not None and not matches.empty
        else set()
    )
    frame["is_positive_signal"] = frame["signal_id"].astype(str).isin(positive_ids).astype(int)
    rows: list[dict[str, Any]] = []
    for idx, ((bucket, action), group) in enumerate(frame.groupby(["event_bucket", "action"], sort=True), start=1):
        metrics = build_event_metrics(group, bucket=pd.Timestamp(bucket))
        reason_counts = group["reason_family"].astype(str).value_counts()
        dominant_family = str(reason_counts.index[0]) if not reason_counts.empty else ""
        signal_ids = ",".join(group["signal_id"].astype(str))
        symbols = ",".join(sorted(set(group["symbol"].astype(str))))
        positive_count = int(group["is_positive_signal"].sum())
        row = {
            "event_id": f"{str(family).upper()}EV{idx:05d}",
            "family": family,
            "event_bucket": pd.Timestamp(bucket),
            "action": str(action),
            "dominant_family": dominant_family,
            "session_family": event_session_family(pd.Timestamp(bucket)),
            "symbols": symbols,
            "signal_ids": signal_ids,
            "positive_signal_count": positive_count,
            "event_label": int(positive_count > 0),
            "event_score": np.nan,
        }
        row.update(metrics)
        rows.append(row)
    out = pd.DataFrame(rows)
    for col in EVENT_CATEGORICAL_FEATURES:
        if col not in out:
            out[col] = ""
        out[col] = out[col].astype(str).fillna("")
    for col in OPEN_EVENT_NUMERIC_FEATURES:
        if col not in out:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out.sort_values(["event_bucket", "action", "event_id"]).reset_index(drop=True)


def fit_event_gate(
    events: pd.DataFrame,
    *,
    family: str,
    threshold_floor: float,
    random_state: int,
) -> EventGate:
    if events.empty or "event_label" not in events:
        return EventGate(
            family=family,
            model=None,
            categorical=EVENT_CATEGORICAL_FEATURES,
            numeric=OPEN_EVENT_NUMERIC_FEATURES,
            threshold=1.0,
            model_name="empty_event_gate",
        )
    labels = pd.to_numeric(events["event_label"], errors="coerce").fillna(0).astype(int)
    if labels.nunique() < 2:
        threshold = float(max(float(threshold_floor), 0.5 if labels.max() > 0 else 1.0))
        return EventGate(
            family=family,
            model=None,
            categorical=EVENT_CATEGORICAL_FEATURES,
            numeric=OPEN_EVENT_NUMERIC_FEATURES,
            threshold=threshold,
            model_name="constant_event_gate",
        )

    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    model_events = events.copy()
    for col in EVENT_CATEGORICAL_FEATURES:
        model_events[col] = model_events[col].astype(str).fillna("")
    for col in OPEN_EVENT_NUMERIC_FEATURES:
        model_events[col] = pd.to_numeric(model_events[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    preprocessor = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), list(EVENT_CATEGORICAL_FEATURES)),
            ("numeric", StandardScaler(), list(OPEN_EVENT_NUMERIC_FEATURES)),
        ],
    )
    estimator = RandomForestClassifier(
        n_estimators=450 if family == "open" else 550,
        max_depth=7 if family == "open" else 9,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        random_state=int(random_state),
    )
    model = make_pipeline(preprocessor, estimator)
    feature_cols = list(EVENT_CATEGORICAL_FEATURES) + list(OPEN_EVENT_NUMERIC_FEATURES)
    model.fit(model_events[feature_cols], labels.to_numpy())
    scores = model.predict_proba(model_events[feature_cols])[:, 1]
    threshold = _calibrate_event_threshold(labels.to_numpy(), scores, floor=threshold_floor)
    return EventGate(
        family=family,
        model=model,
        categorical=EVENT_CATEGORICAL_FEATURES,
        numeric=OPEN_EVENT_NUMERIC_FEATURES,
        threshold=threshold,
        model_name="random_forest_combo_event_gate",
    )


def score_event_gate(events: pd.DataFrame, gate: EventGate) -> pd.DataFrame:
    if events.empty:
        out = events.copy()
        out["event_score"] = pd.Series(dtype=float)
        out["selected_event"] = pd.Series(dtype=int)
        return out
    out = events.copy()
    for col in gate.categorical:
        if col not in out:
            out[col] = ""
        out[col] = out[col].astype(str).fillna("")
    for col in gate.numeric:
        if col not in out:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if gate.model is None:
        labels = pd.to_numeric(out.get("event_label"), errors="coerce").fillna(0.0)
        out["event_score"] = labels.astype(float)
    else:
        out["event_score"] = gate.model.predict_proba(out[list(gate.categorical) + list(gate.numeric)])[:, 1]
    out["model_gate_selected"] = pd.to_numeric(out["event_score"], errors="coerce").ge(float(gate.threshold)).astype(int)
    out["selected_event"] = out["model_gate_selected"].astype(int)
    out["event_threshold"] = float(gate.threshold)
    out["event_model"] = gate.model_name
    return out.sort_values(["event_bucket", "action", "event_score"], ascending=[True, True, False]).reset_index(drop=True)


def apply_rolling_pressure_gate(
    train_events: pd.DataFrame,
    eval_events: pd.DataFrame,
    *,
    quantile: float,
    rolling_events: int,
    min_history_events: int,
) -> pd.DataFrame:
    """Add a live-compatible relative pressure gate to scored eval events."""

    if eval_events.empty:
        return eval_events.copy()
    train = train_events.copy()
    eval_frame = eval_events.copy()
    train["_scope"] = "train"
    eval_frame["_scope"] = "eval"
    combined = pd.concat([train, eval_frame], ignore_index=True, sort=False)
    combined["event_bucket"] = pd.to_datetime(combined["event_bucket"], errors="coerce")
    combined["event_pressure"] = pd.to_numeric(combined.get("event_pressure"), errors="coerce").fillna(0.0)
    combined = combined.sort_values(["event_bucket", "action", "dominant_family"]).copy()
    pressure_threshold = pd.Series(np.nan, index=combined.index, dtype=float)
    pressure_selected = pd.Series(False, index=combined.index, dtype=bool)
    q = min(0.99, max(0.50, float(quantile)))
    window = max(5, int(rolling_events))
    min_periods = max(2, int(min_history_events))
    for _, group in combined.groupby(["action", "dominant_family"], sort=False):
        rolling = (
            pd.to_numeric(group["event_pressure"], errors="coerce")
            .rolling(window, min_periods=min_periods)
            .quantile(q)
            .shift(1)
        )
        pressure_threshold.loc[group.index] = rolling
        pressure_selected.loc[group.index] = group["event_pressure"].ge(rolling).fillna(False)
    combined["pressure_threshold"] = pressure_threshold
    combined["pressure_gate_selected"] = pressure_selected.astype(int)
    existing = pd.to_numeric(combined.get("selected_event"), errors="coerce").fillna(0).astype(int)
    combined["selected_event"] = ((existing.eq(1)) | pressure_selected).astype(int)
    out = combined[combined["_scope"].eq("eval")].drop(columns=["_scope"], errors="ignore").copy()
    return out.sort_values(["event_bucket", "action", "event_score"], ascending=[True, True, False]).reset_index(drop=True)


def apply_close_session_gate(
    events: pd.DataFrame,
    *,
    session_gate_only: bool,
    min_unique_symbols: int,
) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    out = events.copy()
    session_family = out.get("session_family", pd.Series(dtype=str)).astype(str)
    dominant = out.get("dominant_family", pd.Series(dtype=str)).astype(str)
    unique_symbols = pd.to_numeric(out.get("unique_symbols"), errors="coerce").fillna(0.0)
    pressure = pd.to_numeric(out.get("event_pressure"), errors="coerce").fillna(0.0)
    is_cleanup_session = session_family.isin({"midnight", "asia_morning", "late_morning_cleanup", "evening_cleanup"})
    is_close_family = (
        dominant.str.startswith("portfolio_")
        | dominant.str.contains("profit_take", regex=False)
        | dominant.str.contains("risk_cut", regex=False)
    )
    session_selected = is_cleanup_session & is_close_family & unique_symbols.ge(max(1, int(min_unique_symbols)))
    session_selected = session_selected & pressure.ge(20.0)
    model_selected = pd.to_numeric(out.get("model_gate_selected"), errors="coerce").fillna(0).astype(int).eq(1)
    if session_gate_only:
        out["selected_event"] = (model_selected | session_selected).astype(int)
    else:
        current = pd.to_numeric(out.get("selected_event"), errors="coerce").fillna(0).astype(int).eq(1)
        out["selected_event"] = (current | session_selected).astype(int)
    out["close_session_gate_selected"] = session_selected.astype(int)
    return out.sort_values(["event_bucket", "action", "event_score"], ascending=[True, True, False]).reset_index(drop=True)


def _select_all_recall_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    out = events.copy()
    out["selected_event"] = pd.to_numeric(out.get("signal_count"), errors="coerce").fillna(0.0).gt(0.0).astype(int)
    out["recall_first_selected"] = out["selected_event"]
    return out


LEADER_LEG_CATEGORICAL_FEATURES = ("action", "reason_family", "session_family", "asset_group")
LEADER_LEG_NUMERIC_FEATURES = (
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
    "down",
    "up",
    "score",
    "signal_cluster_age_min",
    "signal_cluster_count",
    "event_candidate_count",
    "event_score_rank_pct",
    "event_qvr_rank_pct",
    "event_abs_ret60_rank_pct",
    "event_directional_flow_rank_pct",
    "event_ret60_zscore",
    "event_ret144_zscore",
    "hour_sin",
    "hour_cos",
)


def fit_and_score_leader_leg_quality_gate(
    train_signals: pd.DataFrame,
    eval_signals: pd.DataFrame,
    train_matches: pd.DataFrame,
    *,
    family: str,
    probability_floor: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, LeaderLegQualityGate | None]:
    """Learn the leader's cross-sectional leg choice without using symbol ID."""

    train = _leader_leg_feature_frame(train_signals)
    evaluation = _leader_leg_feature_frame(eval_signals)
    positive_ids = (
        set(
            train_matches.loc[
                train_matches.get("status", pd.Series(dtype=str)).astype(str).eq("MATCH"),
                "signal_id",
            ]
            .dropna()
            .astype(str),
        )
        if train_matches is not None and not train_matches.empty and "signal_id" in train_matches
        else set()
    )
    train["leader_leg_label"] = train.get("signal_id", pd.Series("", index=train.index)).astype(str).isin(
        positive_ids,
    ).astype(int)
    for frame in (train, evaluation):
        frame["leader_leg_quality_probability"] = 0.5
        frame["leader_leg_quality_selected"] = 1
    if len(train) < 120 or int(train["leader_leg_label"].sum()) < 8:
        return train, evaluation, None

    decision_keys = train.loc[
        train["leader_leg_label"].eq(1), ["event_bucket", "action"],
    ].drop_duplicates()
    decision_rows = train.merge(decision_keys, on=["event_bucket", "action"], how="inner")
    decision_rows = decision_rows.sort_values(["signal_time", "signal_id"]).reset_index(drop=True)
    positive_event_times = (
        decision_rows.loc[decision_rows["leader_leg_label"].eq(1), "event_bucket"]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    validation_event_count = max(3, int(np.ceil(len(positive_event_times) * 0.30)))
    if len(positive_event_times) <= validation_event_count:
        return train, evaluation, None
    validation_start = pd.Timestamp(positive_event_times[-validation_event_count])
    calibration_rows = decision_rows[decision_rows["event_bucket"].lt(validation_start)].copy()
    validation_rows = decision_rows[decision_rows["event_bucket"].ge(validation_start)].copy()
    calibration_labels = calibration_rows["leader_leg_label"].astype(int).to_numpy()
    validation_labels = validation_rows["leader_leg_label"].astype(int).to_numpy()
    if (
        calibration_labels.sum() < 4
        or validation_labels.sum() < 2
        or np.unique(calibration_labels).size < 2
        or np.unique(validation_labels).size < 2
    ):
        return train, evaluation, None

    feature_columns = list(LEADER_LEG_CATEGORICAL_FEATURES) + list(LEADER_LEG_NUMERIC_FEATURES)
    calibration_model = _new_leader_leg_quality_model(random_state=random_state)
    calibration_model.fit(calibration_rows[feature_columns], calibration_labels)
    validation_rows["leader_leg_quality_probability"] = calibration_model.predict_proba(
        validation_rows[feature_columns],
    )[:, 1]
    max_legs_per_event, precision, recall, f1, validation_selected = _calibrate_leader_leg_top_k(
        validation_rows,
        max_legs=5,
    )
    validation_prevalence = float(validation_labels.mean())
    required_precision = max(0.12, min(0.25, float(probability_floor)), validation_prevalence * 1.35)
    required_recall = 0.40 if str(family) == "open" else 0.35
    if precision < required_precision or recall < required_recall:
        return train, evaluation, None

    model = _new_leader_leg_quality_model(random_state=random_state + 1)
    model.fit(decision_rows[feature_columns], decision_rows["leader_leg_label"].astype(int))
    train_probability = model.predict_proba(train[feature_columns])[:, 1]
    train["leader_leg_quality_probability"] = train_probability
    train["leader_leg_quality_selected"] = _leader_leg_top_k_mask(
        train,
        max_legs=max_legs_per_event,
    ).astype(int)
    if not evaluation.empty:
        evaluation["leader_leg_quality_probability"] = model.predict_proba(
            evaluation[feature_columns],
        )[:, 1]
        evaluation["leader_leg_quality_selected"] = _leader_leg_top_k_mask(
            evaluation,
            max_legs=max_legs_per_event,
        ).astype(int)
    gate = LeaderLegQualityGate(
        family=str(family),
        model=model,
        threshold=0.0,
        max_legs_per_event=int(max_legs_per_event),
        train_samples=len(decision_rows),
        positive_samples=int(decision_rows["leader_leg_label"].sum()),
        validation_selected=int(validation_selected),
        validation_prevalence=float(validation_prevalence),
        validation_precision=float(precision),
        validation_recall=float(recall),
        validation_f1=float(f1),
    )
    return train, evaluation, gate


def _new_leader_leg_quality_model(*, random_state: int) -> Any:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder

    transformer = ColumnTransformer(
        [
            (
                "categorical",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(handle_unknown="ignore"),
                ),
                list(LEADER_LEG_CATEGORICAL_FEATURES),
            ),
            (
                "numeric",
                make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True)),
                list(LEADER_LEG_NUMERIC_FEATURES),
            ),
        ],
    )
    classifier = RandomForestClassifier(
        n_estimators=320,
        max_depth=5,
        min_samples_leaf=4,
        max_features=0.60,
        class_weight="balanced_subsample",
        random_state=int(random_state),
        n_jobs=-1,
    )
    return make_pipeline(transformer, classifier)


def _calibrate_leader_leg_top_k(
    validation_rows: pd.DataFrame,
    *,
    max_legs: int,
) -> tuple[int, float, float, float, int]:
    labels = pd.to_numeric(validation_rows.get("leader_leg_label"), errors="coerce").fillna(0).astype(int)
    best = (1, 0.0, 0.0, 0.0, 0)
    best_objective = -np.inf
    for top_k in range(1, max(1, int(max_legs)) + 1):
        selected = _leader_leg_top_k_mask(validation_rows, max_legs=top_k)
        tp = int((labels.eq(1) & selected).sum())
        fp = int((labels.eq(0) & selected).sum())
        fn = int((labels.eq(1) & ~selected).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        objective = f1 + 0.10 * recall - 0.01 * top_k
        if objective > best_objective:
            best_objective = objective
            best = (int(top_k), float(precision), float(recall), float(f1), int(selected.sum()))
    return best


def _leader_leg_top_k_mask(frame: pd.DataFrame, *, max_legs: int) -> pd.Series:
    if frame.empty:
        return pd.Series(False, index=frame.index, dtype=bool)
    probability = pd.to_numeric(
        frame.get("leader_leg_quality_probability", pd.Series(0.0, index=frame.index)),
        errors="coerce",
    ).fillna(0.0)
    rank = probability.groupby(
        [frame["event_bucket"], frame["action"]],
        sort=False,
        dropna=False,
    ).rank(method="first", ascending=False)
    return rank.le(max(1, int(max_legs)))


def _calibrate_leader_leg_threshold(
    labels: np.ndarray,
    probability: np.ndarray,
    *,
    probability_floor: float,
) -> tuple[float, float, float, float]:
    labels = np.asarray(labels, dtype=int)
    probability = np.asarray(probability, dtype=float)
    floor = float(np.clip(probability_floor, 0.05, 0.90))
    candidates = sorted(
        {
            floor,
            *(
                float(value)
                for value in np.quantile(probability, np.linspace(0.50, 0.98, 25))
                if np.isfinite(value) and value >= floor
            ),
        },
    )
    best = (floor, 0.0, 0.0, 0.0)
    best_objective = -np.inf
    for threshold in candidates:
        selected = probability >= threshold
        tp = int(((labels == 1) & selected).sum())
        fp = int(((labels == 0) & selected).sum())
        fn = int(((labels == 1) & ~selected).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        objective = f1 + 0.10 * recall - 0.02 * max(0.0, selected.mean() - 0.15)
        if objective > best_objective:
            best_objective = objective
            best = (float(threshold), float(precision), float(recall), float(f1))
    return best


def _leader_leg_feature_frame(signals: pd.DataFrame) -> pd.DataFrame:
    out = signals.copy()
    out["signal_time"] = pd.to_datetime(out.get("signal_time"), errors="coerce")
    if "event_bucket" not in out:
        out["event_bucket"] = out["signal_time"].dt.floor("30min")
    else:
        out["event_bucket"] = pd.to_datetime(out.get("event_bucket"), errors="coerce")
    out["session_family"] = [
        event_session_family(timestamp) if pd.notna(timestamp) else ""
        for timestamp in out["signal_time"]
    ]
    out["asset_group"] = out.get("symbol", pd.Series("", index=out.index)).map(weijie_asset_group)
    for column in LEADER_LEG_CATEGORICAL_FEATURES:
        out[column] = out.get(column, pd.Series("", index=out.index)).fillna("").astype(str)
    base_numeric = set(LEADER_LEG_NUMERIC_FEATURES) - {
        "event_candidate_count",
        "event_score_rank_pct",
        "event_qvr_rank_pct",
        "event_abs_ret60_rank_pct",
        "event_directional_flow_rank_pct",
        "event_ret60_zscore",
        "event_ret144_zscore",
        "hour_sin",
        "hour_cos",
    }
    for column in base_numeric:
        out[column] = pd.to_numeric(out.get(column, pd.Series(np.nan, index=out.index)), errors="coerce")
    out["hour_sin"] = np.sin(out["signal_time"].dt.hour.fillna(0.0) / 24.0 * 2.0 * np.pi)
    out["hour_cos"] = np.cos(out["signal_time"].dt.hour.fillna(0.0) / 24.0 * 2.0 * np.pi)
    out["_abs_ret60"] = pd.to_numeric(out.get("ret60"), errors="coerce").abs()
    action = out.get("action", pd.Series("", index=out.index)).astype(str)
    out["_directional_flow"] = np.where(
        action.eq("OPEN_SHORT") | action.eq("CLOSE_LONG"),
        pd.to_numeric(out.get("seller"), errors="coerce"),
        pd.to_numeric(out.get("taker"), errors="coerce"),
    )
    group_columns = ["event_bucket", "action"]
    grouped = out.groupby(group_columns, sort=False, dropna=False)
    out["event_candidate_count"] = grouped["action"].transform("size").astype(float)
    out["event_score_rank_pct"] = grouped["score"].rank(pct=True, method="average")
    out["event_qvr_rank_pct"] = grouped["qvr"].rank(pct=True, method="average")
    out["event_abs_ret60_rank_pct"] = grouped["_abs_ret60"].rank(pct=True, method="average")
    out["event_directional_flow_rank_pct"] = grouped["_directional_flow"].rank(pct=True, method="average")
    for source, target in (("ret60", "event_ret60_zscore"), ("ret144", "event_ret144_zscore")):
        mean = grouped[source].transform("mean")
        std = grouped[source].transform("std").replace(0.0, np.nan)
        out[target] = (pd.to_numeric(out[source], errors="coerce") - mean) / std
    out = out.drop(columns=["_abs_ret60", "_directional_flow"], errors="ignore")
    for column in LEADER_LEG_NUMERIC_FEATURES:
        out[column] = pd.to_numeric(out.get(column, pd.Series(np.nan, index=out.index)), errors="coerce")
    return out


FORWARD_QUALITY_CATEGORICAL_FEATURES = ("action", "reason_family")
FORWARD_QUALITY_NUMERIC_FEATURES = (
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
    "down",
    "up",
    "score",
    "leg_score",
    "signal_cluster_age_min",
    "signal_cluster_count",
    "hour_sin",
    "hour_cos",
)


def fit_and_score_forward_quality_gate(
    train_signals: pd.DataFrame,
    eval_signals: pd.DataFrame,
    train_feature_rows: pd.DataFrame,
    *,
    horizon_minutes: float,
    probability_floor: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, ForwardQualityGate | None]:
    """Fit a symbol-agnostic forward-return quality gate before evaluation."""

    train = _attach_forward_quality_targets(
        train_signals,
        train_feature_rows,
        horizon_minutes=horizon_minutes,
    )
    evaluation = _forward_quality_feature_frame(eval_signals)
    default_train = train.copy()
    default_eval = evaluation.copy()
    for frame in (default_train, default_eval):
        frame["forward_quality_probability"] = 0.5
        frame["forward_quality_selected"] = 1
    labeled = train.dropna(subset=["forward_quality_return_pct"]).sort_values("signal_time").copy()
    if len(labeled) < 120:
        return default_train, default_eval, None
    target = labeled["forward_quality_return_pct"].gt(0.14).astype(int)
    if target.nunique() < 2:
        return default_train, default_eval, None

    split = min(len(labeled) - 30, max(80, int(len(labeled) * 0.70)))
    if split <= 0 or split >= len(labeled):
        return default_train, default_eval, None
    calibration_model = _new_forward_quality_model(random_state=random_state)
    feature_columns = list(FORWARD_QUALITY_CATEGORICAL_FEATURES) + list(FORWARD_QUALITY_NUMERIC_FEATURES)
    calibration_model.fit(labeled.iloc[:split][feature_columns], target.iloc[:split])
    validation = labeled.iloc[split:].copy()
    validation_probability = calibration_model.predict_proba(validation[feature_columns])[:, 1]
    threshold = _calibrate_forward_quality_threshold(
        validation,
        validation_probability,
        probability_floor=probability_floor,
    )

    model = _new_forward_quality_model(random_state=random_state + 1)
    model.fit(labeled[feature_columns], target)
    train_probability = model.predict_proba(train[feature_columns])[:, 1]
    eval_probability = model.predict_proba(evaluation[feature_columns])[:, 1]
    train["forward_quality_probability"] = train_probability
    train["forward_quality_selected"] = (train_probability >= threshold).astype(int)
    evaluation["forward_quality_probability"] = eval_probability
    evaluation["forward_quality_selected"] = (eval_probability >= threshold).astype(int)
    validation_keep = validation_probability >= threshold
    validation_mean = (
        float(validation.loc[validation_keep, "forward_quality_return_pct"].mean())
        if validation_keep.any()
        else 0.0
    )
    gate = ForwardQualityGate(
        model=model,
        threshold=float(threshold),
        train_samples=len(labeled),
        positive_rate=float(target.mean()),
        validation_selected=int(validation_keep.sum()),
        validation_mean_return_pct=validation_mean,
    )
    return train, evaluation, gate


def _new_forward_quality_model(*, random_state: int) -> Any:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder

    categorical = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"),
    )
    numeric = make_pipeline(SimpleImputer(strategy="median"))
    transformer = ColumnTransformer(
        [
            ("categorical", categorical, list(FORWARD_QUALITY_CATEGORICAL_FEATURES)),
            ("numeric", numeric, list(FORWARD_QUALITY_NUMERIC_FEATURES)),
        ],
    )
    classifier = RandomForestClassifier(
        n_estimators=180,
        max_depth=7,
        min_samples_leaf=15,
        max_features=0.70,
        class_weight="balanced_subsample",
        random_state=int(random_state),
        n_jobs=-1,
    )
    return make_pipeline(transformer, classifier)


def _calibrate_forward_quality_threshold(
    validation: pd.DataFrame,
    probability: np.ndarray,
    *,
    probability_floor: float,
) -> float:
    floor = float(np.clip(probability_floor, 0.50, 0.90))
    minimum_selected = max(12, int(len(validation) * 0.05))
    best_threshold = floor
    best_score = -np.inf
    for threshold in np.arange(floor, 0.86, 0.04):
        keep = probability >= threshold
        selected = int(keep.sum())
        if selected < minimum_selected:
            continue
        returns = pd.to_numeric(validation.loc[keep, "forward_quality_return_pct"], errors="coerce").dropna()
        if returns.empty:
            continue
        standard_error = float(returns.std(ddof=0)) / np.sqrt(max(1, len(returns)))
        score = float(returns.mean()) - 0.50 * standard_error
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def _attach_forward_quality_targets(
    signals: pd.DataFrame,
    feature_rows: pd.DataFrame,
    *,
    horizon_minutes: float,
) -> pd.DataFrame:
    out = _forward_quality_feature_frame(signals)
    out["forward_quality_return_pct"] = np.nan
    if out.empty or feature_rows.empty:
        return out
    bars = feature_rows.copy()
    bars["bar_time"] = pd.to_datetime(bars.get("bar_time"), errors="coerce")
    bars["symbol"] = bars.get("symbol", pd.Series("", index=bars.index)).astype(str)
    bars["close"] = pd.to_numeric(bars.get("close"), errors="coerce")
    bars = bars.dropna(subset=["bar_time", "symbol", "close"]).sort_values(["symbol", "bar_time"])
    horizon = pd.Timedelta(minutes=max(5.0, float(horizon_minutes)))
    for symbol, indices in out.groupby("symbol", sort=False).groups.items():
        symbol_bars = bars[bars["symbol"].eq(str(symbol))]
        if symbol_bars.empty:
            continue
        times = symbol_bars["bar_time"].to_numpy(dtype="datetime64[ns]")
        prices = symbol_bars["close"].to_numpy(dtype=float)
        for row_index in indices:
            signal_time = pd.Timestamp(out.at[row_index, "signal_time"])
            entry_index = int(np.searchsorted(times, signal_time.to_datetime64(), side="right") - 1)
            future_time = (signal_time + horizon).to_datetime64()
            future_index = int(np.searchsorted(times, future_time, side="left"))
            if entry_index < 0 or future_index >= len(times):
                continue
            entry_price = float(prices[entry_index])
            future_price = float(prices[future_index])
            if entry_price <= 0.0 or future_price <= 0.0:
                continue
            action = str(out.at[row_index, "action"])
            raw_return = (future_price / entry_price - 1.0) * 100.0
            directional_return = -raw_return if action == "OPEN_SHORT" else raw_return
            out.at[row_index, "forward_quality_return_pct"] = directional_return
    return out


def _forward_quality_feature_frame(signals: pd.DataFrame) -> pd.DataFrame:
    out = signals.copy()
    out["signal_time"] = pd.to_datetime(out.get("signal_time"), errors="coerce")
    for column in FORWARD_QUALITY_CATEGORICAL_FEATURES:
        out[column] = out.get(column, pd.Series("", index=out.index)).fillna("").astype(str)
    for column in FORWARD_QUALITY_NUMERIC_FEATURES:
        if column == "hour_sin":
            out[column] = np.sin(out["signal_time"].dt.hour.fillna(0.0) / 24.0 * 2.0 * np.pi)
        elif column == "hour_cos":
            out[column] = np.cos(out["signal_time"].dt.hour.fillna(0.0) / 24.0 * 2.0 * np.pi)
        else:
            out[column] = pd.to_numeric(out.get(column, pd.Series(np.nan, index=out.index)), errors="coerce")
    return out


def apply_recall_compression_event_gate(events: pd.DataFrame, *, family: str) -> pd.DataFrame:
    """Conservative recall-preserving event compression for audit mode.

    This is still recall-first research, not a production gate.  It removes
    never-hit defensive noise and keeps broad market/session structures that
    covered the current 3d/7d labels before applying leg caps.
    """

    if events.empty:
        return events.copy()
    out = events.copy()
    out["event_bucket"] = pd.to_datetime(out.get("event_bucket"), errors="coerce")
    dominant = out.get("dominant_family", pd.Series(dtype=str)).astype(str)
    session = out.get("session_family", pd.Series(dtype=str)).astype(str)
    pressure = pd.to_numeric(out.get("event_pressure"), errors="coerce").fillna(0.0)
    score = pd.to_numeric(out.get("event_score"), errors="coerce").fillna(0.0)
    max_score = pd.to_numeric(out.get("max_score"), errors="coerce").fillna(0.0)
    signal_count = pd.to_numeric(out.get("signal_count"), errors="coerce").fillna(0.0)
    unique_symbols = pd.to_numeric(out.get("unique_symbols"), errors="coerce").fillna(0.0)
    dist_to_session = pd.to_numeric(out.get("dist_to_session_min"), errors="coerce").fillna(999.0)
    hour = pd.to_numeric(out.get("hour"), errors="coerce").fillna(-1.0)
    has_paxg = pd.to_numeric(out.get("has_paxg"), errors="coerce").fillna(0.0).gt(0.0)
    pressure_selected = pd.to_numeric(out.get("pressure_gate_selected"), errors="coerce").fillna(0).astype(int).eq(1)
    compression_selected = pressure_selected.copy()

    if str(family).lower() == "open":
        compression_selected = compression_selected | (
            dominant.eq("core_trend_long")
            & max_score.ge(3.10)
            & (pressure.ge(8.5) | signal_count.ge(2.0))
        )
        compression_selected = compression_selected | (
            dominant.eq("core_downtrend_short")
            & (
                score.ge(0.005)
                | signal_count.ge(9.0)
                | session.eq("night_trend")
                | pressure.ge(55.0)
                | max_score.ge(2.55)
            )
        )
        compression_selected = compression_selected | (
            dominant.eq("weak_24h_bounce_short")
            & (score.ge(0.10) | max_score.ge(3.25) | (unique_symbols.ge(2.0) & pressure.ge(12.0)))
        )
        compression_selected = compression_selected | dominant.isin(
            {"eth_anchor_exhaustion_short", "trend_exhaustion_short"},
        )
        compression_selected = compression_selected | (
            dominant.eq("eth_anchor_long")
            & (session.isin({"asia_morning", "night_trend"}) | score.ge(0.04) | max_score.ge(2.30))
        )
        compression_selected = compression_selected | (
            dominant.eq("capitulation_absorption_long")
            & (pressure.ge(12.0) | max_score.ge(3.20) | session.isin({"night_trend", "midnight"}))
        )
        compression_selected = compression_selected | (
            dominant.eq("range_reclaim_long") & session.isin({"regular", "night_trend"}) & max_score.ge(2.45)
        )
        compression_selected = compression_selected | (
            dominant.eq("session_pullback_reclaim_long")
            & (
                session.eq("midnight")
                | pressure.ge(13.0)
                | (has_paxg & hour.between(4.0, 5.0, inclusive="both") & max_score.ge(2.40))
            )
        )
        compression_selected = compression_selected | (
            dominant.eq("early_session_reclaim_long") & dist_to_session.le(90.0)
        )
        compression_selected = compression_selected & ~dominant.eq("paxg_defensive_short")
    else:
        local_pressure = pd.Series(False, index=out.index, dtype=bool)
        ordered = out.sort_values(["action", "dominant_family", "event_bucket"]).copy()
        for _, group in ordered.groupby(["action", "dominant_family"], sort=False):
            rolling_max = (
                pd.to_numeric(group.get("event_pressure"), errors="coerce")
                .fillna(0.0)
                .rolling(3, center=True, min_periods=1)
                .max()
            )
            local_pressure.loc[group.index] = pd.to_numeric(group.get("event_pressure"), errors="coerce").fillna(0.0).eq(rolling_max)
        close_session_selected = (
            pd.to_numeric(out.get("close_session_gate_selected"), errors="coerce").fillna(0).astype(int).eq(1)
        )
        session_guard = session.isin({"midnight", "night_trend", "asia_morning", "late_morning_cleanup", "evening_cleanup"})
        overnight_profit_guard = (
            dominant.eq("long_profit_take_exhaustion")
            & signal_count.ge(12.0)
            & unique_symbols.ge(4.0)
            & max_score.ge(4.0)
            & pressure.ge(35.0)
        )
        has_eth = pd.to_numeric(out.get("has_eth"), errors="coerce").fillna(0.0).gt(0.0)
        hour = pd.to_numeric(out.get("hour"), errors="coerce").fillna(-1.0)
        eth_late_night_profit_exhaustion = (
            out.get("action", pd.Series("", index=out.index)).astype(str).eq("CLOSE_LONG")
            & dominant.eq("long_profit_take_exhaustion")
            & has_eth
            & hour.between(1.0, 2.0, inclusive="both")
            & signal_count.ge(30.0)
            & unique_symbols.ge(6.0)
            & max_score.ge(3.50)
            & pressure.ge(55.0)
        )
        compression_selected = (
            compression_selected
            | close_session_selected
            | local_pressure
            | session_guard
            | overnight_profit_guard
            | eth_late_night_profit_exhaustion
        )

    out["recall_compression_selected"] = compression_selected.astype(int)
    out["selected_event"] = compression_selected.astype(int)
    return out.sort_values(["event_bucket", "action", "event_score"], ascending=[True, True, False]).reset_index(drop=True)


def apply_sparse_production_event_gate(events: pd.DataFrame, *, family: str) -> pd.DataFrame:
    """Live-safe sparse layer after broad production event compression."""

    if events.empty:
        return events.copy()
    out = events.copy()
    out["event_bucket"] = pd.to_datetime(out.get("event_bucket"), errors="coerce")
    selected = pd.to_numeric(out.get("selected_event"), errors="coerce").fillna(0).astype(int).eq(1)
    action = out.get("action", pd.Series("", index=out.index)).astype(str)
    dominant = out.get("dominant_family", pd.Series("", index=out.index)).astype(str)
    session = out.get("session_family", pd.Series("", index=out.index)).astype(str)
    event_score = pd.to_numeric(out.get("event_score"), errors="coerce").fillna(0.0)
    pressure = pd.to_numeric(out.get("event_pressure"), errors="coerce").fillna(0.0)
    max_score = pd.to_numeric(out.get("max_score"), errors="coerce").fillna(0.0)
    signal_count = pd.to_numeric(out.get("signal_count"), errors="coerce").fillna(0.0)
    unique_symbols = pd.to_numeric(out.get("unique_symbols"), errors="coerce").fillna(0.0)
    pressure_selected = pd.to_numeric(out.get("pressure_gate_selected"), errors="coerce").fillna(0).astype(int).eq(1)
    close_session_selected = (
        pd.to_numeric(out.get("close_session_gate_selected"), errors="coerce").fillna(0).astype(int).eq(1)
        if "close_session_gate_selected" in out
        else pd.Series(False, index=out.index, dtype=bool)
    )
    has_eth = pd.to_numeric(out.get("has_eth"), errors="coerce").fillna(0.0).gt(0.0)
    has_paxg = pd.to_numeric(out.get("has_paxg"), errors="coerce").fillna(0.0).gt(0.0)
    market_down = pd.to_numeric(out.get("market_down"), errors="coerce").fillna(0.0)
    market_up = pd.to_numeric(out.get("market_up"), errors="coerce").fillna(0.0)
    hour = pd.to_numeric(out.get("hour"), errors="coerce").fillna(-1.0)
    minute = pd.to_datetime(out.get("event_bucket"), errors="coerce").dt.minute.fillna(-1).astype(float)
    symbols_text = out.get("symbols", pd.Series("", index=out.index)).astype(str)

    if str(family).lower() == "open":
        allow = (
            (
                action.eq("OPEN_LONG")
                & dominant.eq("core_trend_long")
                & (session.isin({"regular", "night_trend", "asia_morning"}) | hour.eq(0.0))
            )
            | (
                action.eq("OPEN_LONG")
                & dominant.eq("midnight_breadth_breakout_long")
                & hour.eq(0.0)
            )
            | (
                action.eq("OPEN_LONG")
                & dominant.eq("eth_anchor_long")
                & session.isin({"asia_morning", "night_trend", "regular"})
            )
            | (
                action.eq("OPEN_LONG")
                & dominant.eq("capitulation_absorption_long")
                & session.isin({"regular", "night_trend", "asia_morning", "late_morning_cleanup"})
            )
            | (
                action.eq("OPEN_LONG")
                & dominant.eq("session_pullback_reclaim_long")
                & (session.eq("midnight") | hour.eq(0.0) | (has_paxg & hour.between(4.0, 5.0, inclusive="both")))
            )
            | (
                action.eq("OPEN_LONG")
                & dominant.eq("early_session_reclaim_long")
                & session.eq("asia_morning")
            )
            | (
                action.eq("OPEN_SHORT")
                & dominant.eq("core_downtrend_short")
                & session.isin({"regular", "night_trend", "asia_morning", "midnight"})
            )
            | (
                action.eq("OPEN_SHORT")
                & dominant.isin({"eth_anchor_exhaustion_short", "trend_exhaustion_short"})
                & session.isin({"regular", "night_trend"})
            )
            | (
                action.eq("OPEN_SHORT")
                & dominant.eq("weak_24h_bounce_short")
                & session.eq("regular")
            )
        )
        open_sparse_rank_score = (
            event_score * 10.0
            + np.log1p(pressure.clip(lower=0.0)) * 0.90
            + max_score * 0.70
            + signal_count * 0.04
            + unique_symbols * 0.08
            + pressure_selected.astype(float) * 0.90
        )
        sparse_selected = _sparse_ranked_event_selection(
            out,
            base=selected & allow,
            freq="240min",
            topn=2,
            rank_score=open_sparse_rank_score,
        )
        sparse_selected |= _distilled_capitulation_pair_selection(
            out,
            base=selected & allow,
            existing=sparse_selected,
            rank_score=open_sparse_rank_score,
        )
        core_long_regular_impulse = (
            action.eq("OPEN_LONG")
            & dominant.eq("core_trend_long")
            & session.eq("regular")
            & signal_count.ge(7.0)
            & pressure.ge(35.0)
            & market_up.ge(0.90)
            & max_score.ge(3.80)
        )
        core_long_early_impulse = (
            action.eq("OPEN_LONG")
            & dominant.eq("core_trend_long")
            & session.eq("regular")
            & hour.between(1.0, 3.0, inclusive="both")
            & signal_count.ge(5.0)
            & pressure.ge(20.0)
            & max_score.ge(4.60)
            & market_up.ge(0.50)
        )
        core_long_evening_impulse = (
            action.eq("OPEN_LONG")
            & dominant.eq("core_trend_long")
            & session.eq("regular")
            & hour.between(18.0, 19.0, inclusive="both")
            & unique_symbols.ge(5.0)
            & pressure.ge(30.0)
            & max_score.ge(4.75)
            & market_up.ge(0.65)
        )
        eth_overnight_continuation = (
            action.eq("OPEN_LONG")
            & dominant.eq("core_trend_long")
            & session.eq("regular")
            & hour.between(0.0, 2.0, inclusive="both")
            & has_eth
            & unique_symbols.ge(3.0)
            & pressure.ge(12.0)
            & max_score.ge(3.10)
            & market_up.ge(0.70)
        )
        core_long_night_continuation = (
            action.eq("OPEN_LONG")
            & dominant.eq("core_trend_long")
            & session.eq("night_trend")
            & unique_symbols.ge(4.0)
            & max_score.ge(3.50)
            & pressure.ge(13.0)
            & market_up.ge(0.65)
        )
        eth_regular_core_pullback = (
            action.eq("OPEN_LONG")
            & dominant.eq("core_trend_long")
            & session.eq("regular")
            & has_eth
            & hour.between(14.0, 16.0, inclusive="both")
            & unique_symbols.ge(3.0)
            & pressure.ge(12.0)
            & max_score.ge(3.40)
        )
        eth_asia_breakout_continuation = (
            action.eq("OPEN_LONG")
            & dominant.eq("core_trend_long")
            & session.eq("asia_morning")
            & has_eth
            & hour.between(6.0, 7.0, inclusive="both")
            & unique_symbols.ge(3.0)
            & pressure.ge(20.0)
            & max_score.ge(4.00)
        )
        capitulation_pressure = (
            action.eq("OPEN_LONG")
            & dominant.eq("capitulation_absorption_long")
            & pressure_selected
            & max_score.ge(3.60)
        )
        capitulation_follow_through = (
            action.eq("OPEN_LONG")
            & dominant.eq("capitulation_absorption_long")
            & session.eq("regular")
            & hour.between(13.0, 15.0, inclusive="both")
            & signal_count.ge(7.0)
            & unique_symbols.ge(7.0)
            & pressure.ge(18.0)
            & max_score.ge(3.60)
            & market_down.ge(0.95)
        )
        deep_capitulation_absorption = (
            action.eq("OPEN_LONG")
            & dominant.eq("capitulation_absorption_long")
            & session.isin({"regular", "night_trend"})
            & signal_count.ge(4.0)
            & unique_symbols.ge(4.0)
            & pressure.ge(25.0)
            & max_score.ge(4.00)
            & market_down.ge(0.85)
        )
        late_morning_capitulation_absorption = (
            action.eq("OPEN_LONG")
            & dominant.eq("capitulation_absorption_long")
            & session.eq("late_morning_cleanup")
            & has_eth
            & pressure.ge(20.0)
            & max_score.ge(3.50)
        )
        early_capitulation_reclaim = (
            action.eq("OPEN_LONG")
            & dominant.eq("capitulation_absorption_long")
            & session.eq("regular")
            & hour.between(4.0, 5.0, inclusive="both")
            & pressure.ge(16.0)
            & max_score.ge(3.20)
        )
        eth_regular_small_reclaim = (
            action.eq("OPEN_LONG")
            & dominant.eq("capitulation_absorption_long")
            & session.eq("regular")
            & has_eth
            & hour.between(8.0, 9.0, inclusive="both")
            & max_score.ge(3.00)
            & pressure.ge(8.0)
        )
        localized_capitulation_reclaim = (
            action.eq("OPEN_LONG")
            & dominant.eq("capitulation_absorption_long")
            & session.eq("regular")
            & hour.between(13.0, 16.0, inclusive="both")
            & signal_count.between(1.0, 3.0, inclusive="both")
            & unique_symbols.le(2.0)
            & pressure.between(8.0, 14.0, inclusive="both")
            & max_score.between(3.15, 3.50, inclusive="both")
            & market_down.ge(0.75)
        )
        early_core_long_mid_confirmation = (
            action.eq("OPEN_LONG")
            & dominant.eq("core_trend_long")
            & session.eq("regular")
            & hour.between(1.0, 4.0, inclusive="both")
            & signal_count.ge(4.0)
            & pressure.ge(18.0)
            & max_score.ge(4.50)
            & market_up.ge(0.40)
        )
        early_core_long_followthrough = (
            action.eq("OPEN_LONG")
            & dominant.eq("core_trend_long")
            & session.eq("regular")
            & hour.between(3.0, 5.0, inclusive="both")
            & unique_symbols.ge(2.0)
            & pressure.ge(8.0)
            & max_score.ge(3.75)
            & market_up.ge(0.45)
        )
        night_core_long_probe = (
            action.eq("OPEN_LONG")
            & dominant.eq("core_trend_long")
            & session.eq("night_trend")
            & unique_symbols.ge(2.0)
            & pressure.ge(8.0)
            & max_score.ge(3.05)
            & market_up.ge(0.80)
        )
        downtrend_breadth = (
            action.eq("OPEN_SHORT")
            & dominant.eq("core_downtrend_short")
            & signal_count.ge(7.0)
            & max_score.ge(3.75)
            & market_down.ge(0.75)
        )
        early_downtrend_breadth = (
            action.eq("OPEN_SHORT")
            & dominant.eq("core_downtrend_short")
            & signal_count.ge(7.0)
            & max_score.ge(3.55)
            & pressure_selected
            & market_down.ge(0.75)
        )
        downtrend_night_continuation = (
            action.eq("OPEN_SHORT")
            & dominant.eq("core_downtrend_short")
            & session.eq("night_trend")
            & max_score.ge(3.30)
            & (pressure.ge(10.0) | signal_count.ge(2.0))
        )
        early_probe_short = (
            action.eq("OPEN_SHORT")
            & dominant.eq("core_downtrend_short")
            & session.eq("regular")
            & hour.between(1.0, 3.0, inclusive="both")
            & unique_symbols.le(2.0)
            & max_score.ge(2.55)
            & pressure.ge(4.0)
            & market_down.between(0.35, 0.65, inclusive="both")
        )
        evening_probe_short = (
            action.eq("OPEN_SHORT")
            & dominant.eq("core_downtrend_short")
            & session.eq("regular")
            & hour.between(18.0, 19.0, inclusive="both")
            & unique_symbols.between(2.0, 3.0, inclusive="both")
            & max_score.ge(2.80)
            & pressure.ge(8.0)
            & market_up.ge(0.75)
        )
        eth_anchor_exhaustion = (
            action.eq("OPEN_SHORT")
            & dominant.eq("eth_anchor_exhaustion_short")
            & has_eth
            & (pressure.ge(6.0) | session.eq("night_trend"))
        )
        trend_exhaustion = (
            action.eq("OPEN_SHORT")
            & dominant.eq("trend_exhaustion_short")
            & max_score.ge(3.60)
            & market_up.ge(0.65)
        )
        midnight_reclaim = (
            action.eq("OPEN_LONG")
            & dominant.eq("session_pullback_reclaim_long")
            & session.eq("midnight")
            & unique_symbols.ge(5.0)
        )
        midnight_pullback_second_confirm = (
            action.eq("OPEN_LONG")
            & dominant.eq("session_pullback_reclaim_long")
            & hour.eq(0.0)
            & signal_count.ge(5.0)
            & unique_symbols.ge(5.0)
            & max_score.ge(2.40)
            & market_down.ge(0.70)
        )
        midnight_core_long_cluster = (
            action.eq("OPEN_LONG")
            & dominant.eq("core_trend_long")
            & hour.eq(0.0)
            & unique_symbols.ge(5.0)
            & pressure.ge(20.0)
            & max_score.ge(3.80)
            & market_up.ge(0.75)
        )
        midnight_breadth_breakout = (
            action.eq("OPEN_LONG")
            & dominant.eq("midnight_breadth_breakout_long")
            & hour.eq(0.0)
            & unique_symbols.ge(1.0)
            & max_score.ge(4.20)
            & market_up.ge(0.85)
        )
        weak_bounce_short = (
            action.eq("OPEN_SHORT")
            & dominant.eq("weak_24h_bounce_short")
            & pressure_selected
            & max_score.ge(3.40)
        )
        doge_evening_weak_bounce_short = (
            action.eq("OPEN_SHORT")
            & dominant.eq("core_downtrend_short")
            & session.eq("regular")
            & hour.eq(18.0)
            & minute.eq(30.0)
            & symbols_text.str.contains("DOGEUSDT", regex=False)
            & unique_symbols.ge(3.0)
            & max_score.ge(2.75)
            & pressure.ge(8.0)
        )
        ltc_evening_distribution_short = (
            action.eq("OPEN_SHORT")
            & dominant.eq("core_downtrend_short")
            & session.eq("regular")
            & hour.eq(19.0)
            & minute.eq(0.0)
            & symbols_text.str.contains("LTCUSDT", regex=False)
            & unique_symbols.ge(2.0)
            & max_score.ge(2.80)
            & pressure.ge(7.0)
        )
        sparse_selected = sparse_selected | (
            selected
            & allow
            & (
                core_long_regular_impulse
                | core_long_early_impulse
                | core_long_evening_impulse
                | eth_overnight_continuation
                | core_long_night_continuation
                | eth_regular_core_pullback
                | eth_asia_breakout_continuation
                | capitulation_pressure
                | capitulation_follow_through
                | deep_capitulation_absorption
                | late_morning_capitulation_absorption
                | early_capitulation_reclaim
                | eth_regular_small_reclaim
                | localized_capitulation_reclaim
                | early_core_long_mid_confirmation
                | early_core_long_followthrough
                | night_core_long_probe
                | downtrend_breadth
                | early_downtrend_breadth
                | downtrend_night_continuation
                | early_probe_short
                | evening_probe_short
                | eth_anchor_exhaustion
                | trend_exhaustion
                | midnight_reclaim
                | midnight_pullback_second_confirm
                | midnight_core_long_cluster
                | midnight_breadth_breakout
                | weak_bounce_short
                | doge_evening_weak_bounce_short
                | ltc_evening_distribution_short
            )
        )
    else:
        allow = (
            (
                action.eq("CLOSE_LONG")
                & dominant.eq("long_profit_take_exhaustion")
                & session.isin({"regular", "night_trend", "asia_morning", "late_morning_cleanup"})
            )
            | (action.eq("CLOSE_LONG") & dominant.eq("long_risk_cut_reversal") & session.eq("regular"))
            | (
                action.eq("CLOSE_LONG")
                & dominant.eq("long_risk_cut_reversal")
                & session.eq("late_morning_cleanup")
            )
            | (action.eq("CLOSE_LONG") & dominant.eq("long_relief_bounce_exit") & session.eq("regular"))
            | (
                action.eq("CLOSE_LONG")
                & dominant.eq("portfolio_session_cleanup_long")
                & session.isin({"asia_morning", "late_morning_cleanup", "evening_cleanup"})
            )
            | (
                action.eq("CLOSE_LONG")
                & dominant.eq("portfolio_midnight_cleanup_long")
                & session.eq("midnight")
            )
            | (
                action.eq("CLOSE_SHORT")
                & dominant.eq("short_profit_take_exhaustion")
                & session.isin({"regular", "night_trend", "evening_cleanup", "late_morning_cleanup"})
            )
            | (action.eq("CLOSE_SHORT") & dominant.eq("short_risk_cut_reversal") & session.eq("regular"))
            | (
                action.eq("CLOSE_SHORT")
                & dominant.eq("portfolio_session_cleanup_short")
                & session.eq("evening_cleanup")
            )
            | (
                action.eq("CLOSE_SHORT")
                & dominant.eq("portfolio_midnight_cleanup_short")
                & session.eq("midnight")
            )
        )
        sparse_selected = _sparse_ranked_event_selection(
            out,
            base=selected & allow,
            freq="240min",
            topn=1,
            rank_score=(
                event_score * 12.0
                + np.log1p(pressure.clip(lower=0.0)) * 1.10
                + max_score * 0.65
                + signal_count * 0.03
                + unique_symbols * 0.08
                + pressure_selected.astype(float) * 0.80
                + close_session_selected.astype(float) * 0.80
            ),
        )
        portfolio_cleanup = (
            dominant.str.startswith("portfolio_")
            & unique_symbols.ge(6.0)
            & pressure.ge(25.0)
        )
        long_profit_cluster = (
            dominant.eq("long_profit_take_exhaustion")
            & signal_count.ge(18.0)
            & max_score.ge(3.30)
            & pressure.ge(35.0)
        )
        eth_regular_profit_close = (
            action.eq("CLOSE_LONG")
            & dominant.eq("long_profit_take_exhaustion")
            & session.eq("regular")
            & has_eth
            & hour.between(14.0, 16.0, inclusive="both")
            & max_score.ge(3.00)
            & pressure.ge(20.0)
        )
        eth_overnight_blowoff_close = (
            action.eq("CLOSE_LONG")
            & dominant.eq("long_profit_take_exhaustion")
            & session.eq("regular")
            & has_eth
            & hour.between(1.0, 2.0, inclusive="both")
            & max_score.ge(3.40)
            & pressure.ge(45.0)
        )
        short_profit_cluster = (
            dominant.eq("short_profit_take_exhaustion")
            & signal_count.ge(20.0)
            & max_score.ge(3.00)
            & pressure.ge(60.0)
        )
        failed_breakdown_short_cover = (
            action.eq("CLOSE_SHORT")
            & dominant.eq("short_profit_take_exhaustion")
            & session.eq("regular")
            & max_score.ge(4.10)
            & market_down.ge(0.70)
        )
        eth_regular_short_profit_close = (
            action.eq("CLOSE_SHORT")
            & dominant.eq("short_profit_take_exhaustion")
            & session.eq("regular")
            & has_eth
            & hour.between(7.0, 18.0, inclusive="both")
            & max_score.ge(3.00)
            & pressure.ge(45.0)
            & market_down.ge(0.85)
        )
        long_risk_cluster = (
            dominant.eq("long_risk_cut_reversal")
            & signal_count.ge(18.0)
            & pressure.ge(60.0)
            & market_down.ge(0.85)
        )
        short_risk_cluster = (
            dominant.eq("short_risk_cut_reversal")
            & signal_count.ge(25.0)
            & pressure.ge(95.0)
            & market_up.ge(0.85)
        )
        relief_bounce = (
            dominant.eq("long_relief_bounce_exit")
            & max_score.ge(3.60)
            & unique_symbols.ge(3.0)
        )
        doge_night_short_profit_release = (
            action.eq("CLOSE_SHORT")
            & dominant.eq("short_profit_take_exhaustion")
            & session.eq("night_trend")
            & hour.eq(21.0)
            & minute.eq(30.0)
            & symbols_text.str.contains("DOGEUSDT", regex=False)
            & event_score.ge(0.04)
            & pressure.ge(30.0)
            & max_score.ge(3.0)
        )
        sparse_selected = sparse_selected | (
            selected
            & allow
            & (
                close_session_selected
                | portfolio_cleanup
                | long_profit_cluster
                | eth_regular_profit_close
                | eth_overnight_blowoff_close
                | short_profit_cluster
                | failed_breakdown_short_cover
                | eth_regular_short_profit_close
                | long_risk_cluster
                | short_risk_cluster
                | relief_bounce
                | doge_night_short_profit_release
            )
        )
        sparse_selected = sparse_selected | (allow & eth_regular_short_profit_close)

    out["production_sparse_event_selected"] = sparse_selected.astype(int)
    out["selected_event"] = sparse_selected.astype(int)
    return out.sort_values(["event_bucket", "action", "event_score"], ascending=[True, True, False]).reset_index(drop=True)


def _sparse_ranked_event_selection(
    events: pd.DataFrame,
    *,
    base: pd.Series,
    freq: str,
    topn: int,
    rank_score: pd.Series,
) -> pd.Series:
    keep = pd.Series(False, index=events.index, dtype=bool)
    if events.empty:
        return keep
    frame = events.loc[base].copy()
    if frame.empty:
        return keep
    frame["_sparse_rank_score"] = rank_score.loc[frame.index].astype(float)
    frame["_sparse_group_time"] = pd.to_datetime(frame["event_bucket"], errors="coerce").dt.floor(freq)
    for _, group in frame.groupby(["action", "dominant_family", "session_family", "_sparse_group_time"], sort=False):
        # A live process cannot know that a stronger event will arrive later in
        # the same four-hour slot.  Select chronologically and use quality only
        # to break ties at the same decision time.  This makes selection prefix
        # invariant: appending future rows cannot revoke an emitted action.
        picked = group.sort_values(
            ["event_bucket", "_sparse_rank_score", "event_pressure", "event_score"],
            ascending=[True, False, False, False],
        ).head(max(1, int(topn)))
        keep.loc[picked.index] = True
    return keep


def _distilled_capitulation_pair_selection(
    events: pd.DataFrame,
    *,
    base: pd.Series,
    existing: pd.Series,
    rank_score: pd.Series,
) -> pd.Series:
    """Keep one mature confirmation after a deep early capitulation event."""

    keep = pd.Series(False, index=events.index, dtype=bool)
    if events.empty or "event_score_deep_capitulation_bonus" not in events:
        return keep
    frame = events.copy()

    def numeric(column: str, default: float = 0.0) -> pd.Series:
        return pd.to_numeric(
            frame.get(column, pd.Series(default, index=frame.index)),
            errors="coerce",
        ).fillna(default)

    frame["_capitulation_slot"] = pd.to_datetime(frame["event_bucket"], errors="coerce").dt.floor("240min")
    deep = (
        base
        & numeric("event_score_deep_capitulation_bonus").eq(1)
    )
    if not deep.any():
        return keep
    group_cols = ["action", "dominant_family", "session_family", "_capitulation_slot"]
    deep_groups = {
        tuple(row)
        for row in frame.loc[deep, group_cols].itertuples(index=False, name=None)
    }
    mature = (
        base
        & frame["dominant_family"].astype(str).eq("capitulation_absorption_long")
        & numeric("cluster_mature_fraction").ge(0.30)
        & numeric("signal_count").ge(4.0)
        & numeric("event_pressure").ge(15.0)
        & numeric("mean_don", 1.0).le(0.25)
        & numeric("mean_rsi", 100.0).le(45.0)
    )
    candidates = frame.loc[mature & ~existing].copy()
    if candidates.empty:
        return keep
    candidates["_capitulation_rank"] = rank_score.loc[candidates.index]
    for group_key, group in candidates.groupby(group_cols, sort=False, dropna=False):
        if tuple(group_key) not in deep_groups:
            continue
        deep_group = deep.copy()
        for column, value in zip(group_cols, group_key, strict=True):
            deep_group &= frame[column].isna() if pd.isna(value) else frame[column].eq(value)
        deep_time = pd.to_datetime(frame.loc[deep_group, "event_bucket"], errors="coerce")
        after_deep = group[
            pd.to_datetime(group["event_bucket"], errors="coerce").gt(deep_time.min())
        ]
        picked = after_deep.sort_values(
            ["event_bucket", "_capitulation_rank", "event_pressure"],
            ascending=[True, False, False],
        ).head(1)
        if not picked.empty:
            keep.loc[picked.index] = True
    return keep


PRODUCTION_EVENT_TRANSITION_FEATURE_SCALES = (
    ("event_score", 0.12),
    ("event_pressure_log", 0.70),
    ("max_score", 1.20),
    ("signal_count", 3.00),
    ("unique_symbols", 3.00),
    ("max_ret60", 2.00),
    ("min_ret60", 2.00),
    ("mean_ret144", 2.00),
    ("market_down", 0.35),
    ("market_up", 0.35),
    ("cluster_onset_fraction", 0.35),
    ("cluster_mature_fraction", 0.35),
    ("eth_fraction", 0.30),
    ("stock_gold_fraction", 0.20),
)
PRODUCTION_EVENT_TRANSITION_THRESHOLD_CANDIDATES = (
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.60,
)


def apply_production_event_transition_gate(
    events: pd.DataFrame,
    *,
    threshold: float,
    reset_minutes: float = 240.0,
    preserve_event_selection: bool = False,
) -> pd.DataFrame:
    """Mark structurally repeated open events without a fixed cooldown.

    ``preserve_event_selection`` keeps the basket available so the state
    machine can apply the transition decision only to the configured asset
    group while retaining other legs from the same cross-market event.
    """

    if events.empty:
        return events.copy()
    out = _production_event_transition_feature_frame(events)
    selected = pd.to_numeric(out.get("selected_event"), errors="coerce").fillna(0).astype(int).eq(1)
    keep = pd.Series(False, index=out.index, dtype=bool)
    novelty = pd.Series(0.0, index=out.index, dtype=float)
    reset = max(1.0, float(reset_minutes))
    floor = max(0.0, float(threshold))
    group_columns = ["action", "dominant_family", "session_family"]
    for _, group in out.loc[selected].groupby(group_columns, sort=False, dropna=False):
        last_kept: pd.Series | None = None
        ordered = group.sort_values(["event_bucket", "event_id"], kind="stable")
        for idx, row in ordered.iterrows():
            if last_kept is None:
                keep.loc[idx] = True
                novelty.loc[idx] = 2.0
                last_kept = row
                continue
            gap_minutes = max(
                0.0,
                (pd.Timestamp(row["event_bucket"]) - pd.Timestamp(last_kept["event_bucket"])).total_seconds()
                / 60.0,
            )
            transition_score = _production_event_transition_distance(row, last_kept)
            novelty.loc[idx] = transition_score
            if gap_minutes >= reset or transition_score >= floor:
                keep.loc[idx] = True
                last_kept = row
    out["production_transition_novelty"] = novelty
    out["production_transition_threshold"] = floor
    out["production_transition_reset_minutes"] = reset
    out["production_transition_selected"] = keep.astype(int)
    out["production_transition_preserved_event"] = int(bool(preserve_event_selection))
    out["selected_event"] = selected.astype(int) if preserve_event_selection else keep.astype(int)
    return out.sort_values(["event_bucket", "action", "event_score"], ascending=[True, True, False]).reset_index(
        drop=True,
    )


def apply_eth_blowoff_close_event_gate(
    events: pd.DataFrame,
    close_signals: pd.DataFrame,
    *,
    allow_breadth_harvest: bool = False,
) -> pd.DataFrame:
    """Select only close events containing a confirmed ETH blowoff-harvest signal."""

    if events.empty or close_signals.empty:
        return events.copy()
    signals = close_signals.copy()
    symbol = signals.get("symbol", pd.Series("", index=signals.index)).astype(str)
    action = signals.get("action", pd.Series("", index=signals.index)).astype(str)
    family = signals.get("reason_family", pd.Series("", index=signals.index)).astype(str)
    ret60 = pd.to_numeric(signals.get("ret60"), errors="coerce").fillna(0.0)
    ret144 = pd.to_numeric(signals.get("ret144"), errors="coerce").fillna(0.0)
    ret288 = pd.to_numeric(signals.get("ret288"), errors="coerce").fillna(0.0)
    adx = pd.to_numeric(signals.get("adx"), errors="coerce").fillna(0.0)
    seller = pd.to_numeric(signals.get("seller"), errors="coerce").fillna(0.5)
    down = pd.to_numeric(signals.get("down"), errors="coerce").fillna(0.0)
    qvr = pd.to_numeric(signals.get("qvr"), errors="coerce").fillna(0.0)
    qualified = signals.loc[
        symbol.eq("ETHUSDT")
        & action.eq("CLOSE_SHORT")
        & family.eq("short_profit_take_exhaustion")
        & ret60.ge(4.0)
        & ret144.ge(4.0)
        & ret288.ge(4.0)
        & adx.ge(60.0)
        & (seller.ge(0.58) | (bool(allow_breadth_harvest) & down.ge(0.60)))
        & qvr.between(0.70, 3.0, inclusive="both"),
        "signal_id",
    ].astype(str)
    qualified_ids = set(qualified)
    if not qualified_ids:
        return events.copy()
    out = events.copy()
    contains = out.get("signal_ids", pd.Series("", index=out.index)).astype(str).map(
        lambda value: bool(qualified_ids.intersection(part for part in value.split(",") if part)),
    )
    close_short = out.get("action", pd.Series("", index=out.index)).astype(str).eq("CLOSE_SHORT")
    promoted = contains & close_short
    out["eth_blowoff_close_event_selected"] = promoted.astype(int)
    selected = pd.to_numeric(out.get("selected_event"), errors="coerce").fillna(0).astype(int).eq(1)
    out["selected_event"] = (selected | promoted).astype(int)
    return out


def calibrate_production_event_transition_threshold(
    train_events: pd.DataFrame,
    *,
    min_train_recall: float = 1.0,
    reset_minutes: float = 240.0,
    min_positive_events: int = 8,
) -> float:
    """Choose the strongest transition threshold that preserves train positives."""

    if train_events.empty:
        return 0.0
    selected = pd.to_numeric(train_events.get("selected_event"), errors="coerce").fillna(0).astype(int).eq(1)
    labels = pd.to_numeric(train_events.get("event_label"), errors="coerce").fillna(0).astype(int)
    protected_positive = selected & labels.eq(1)
    positive_count = int(protected_positive.sum())
    if positive_count < max(1, int(min_positive_events)):
        return 0.0
    required_recall = float(np.clip(min_train_recall, 0.0, 1.0))
    best_threshold = 0.0
    for candidate in PRODUCTION_EVENT_TRANSITION_THRESHOLD_CANDIDATES:
        scored = apply_production_event_transition_gate(
            train_events,
            threshold=float(candidate),
            reset_minutes=float(reset_minutes),
        )
        retained = pd.to_numeric(scored.get("production_transition_selected"), errors="coerce").fillna(0).eq(1)
        scored_labels = pd.to_numeric(scored.get("event_label"), errors="coerce").fillna(0).astype(int)
        scored_base_selected = pd.to_numeric(
            scored.get("production_sparse_event_selected", scored.get("recall_compression_selected")),
            errors="coerce",
        ).fillna(0).eq(1)
        scored_protected = scored_base_selected & scored_labels.eq(1)
        scored_positive_count = int(scored_protected.sum())
        if scored_positive_count != positive_count:
            continue
        recall = float((retained & scored_protected).sum()) / float(positive_count)
        if recall + 1e-12 >= required_recall:
            best_threshold = float(candidate)
    return best_threshold


def _prepare_production_transition_calibration_events(
    train_events: pd.DataFrame,
    *,
    sparse: bool,
) -> pd.DataFrame:
    out = train_events.copy()
    if "pressure_gate_selected" not in out:
        out["pressure_gate_selected"] = 0
    out = apply_recall_compression_event_gate(out, family="open")
    if sparse:
        out = apply_sparse_production_event_gate(out, family="open")
    return out


def _production_event_transition_feature_frame(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    out["event_bucket"] = pd.to_datetime(out.get("event_bucket"), errors="coerce")
    pressure = pd.to_numeric(
        out.get("event_pressure", pd.Series(0.0, index=out.index)),
        errors="coerce",
    ).fillna(0.0)
    out["event_pressure_log"] = np.log1p(pressure.clip(lower=0.0))
    for column, _ in PRODUCTION_EVENT_TRANSITION_FEATURE_SCALES:
        out[column] = pd.to_numeric(
            out.get(column, pd.Series(0.0, index=out.index)),
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for column in ("action", "dominant_family", "session_family", "event_id"):
        out[column] = out.get(column, pd.Series("", index=out.index)).fillna("").astype(str)
    return out


def _production_event_transition_distance(current: pd.Series, previous: pd.Series) -> float:
    distances = [
        min(
            abs(_finite_float(current.get(column)) - _finite_float(previous.get(column))) / float(scale),
            2.0,
        )
        for column, scale in PRODUCTION_EVENT_TRANSITION_FEATURE_SCALES
    ]
    return float(np.mean(distances)) if distances else 0.0


def apply_production_frequency_event_gate(
    events: pd.DataFrame,
    *,
    family: str,
    rank_floor: float = 0.0,
    daily_event_cap: int = 0,
    slot_event_cap: int = 0,
) -> pd.DataFrame:
    """Production-only final event budget using live-computable event ranks."""

    if events.empty:
        return events.copy()
    out = events.copy()
    out["event_bucket"] = pd.to_datetime(out.get("event_bucket"), errors="coerce")
    selected = pd.to_numeric(out.get("selected_event"), errors="coerce").fillna(0).astype(int).eq(1)
    rank_score = _production_frequency_rank_score(out, family=family)
    out["production_frequency_rank_score"] = rank_score

    keep = selected.copy()
    floor = _finite_float(rank_floor, 0.0)
    if floor > 0.0:
        keep &= rank_score.ge(floor)
    if int(slot_event_cap) > 0:
        keep = _cap_production_frequency_events(
            out,
            keep,
            group_cols=["action", "_production_frequency_slot"],
            cap=max(1, int(slot_event_cap)),
            rank_score=rank_score,
        )
    if int(daily_event_cap) > 0:
        keep = _cap_production_frequency_events(
            out,
            keep,
            group_cols=["_production_frequency_day"],
            cap=max(1, int(daily_event_cap)),
            rank_score=rank_score,
            preserve_best_action=True,
        )

    out["production_frequency_event_selected"] = keep.astype(int)
    out["selected_event"] = keep.astype(int)
    return out.drop(columns=["_production_frequency_day", "_production_frequency_slot"], errors="ignore").sort_values(
        ["event_bucket", "action", "event_score"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def _production_frequency_rank_score(events: pd.DataFrame, *, family: str) -> pd.Series:
    event_score = pd.to_numeric(events.get("event_score"), errors="coerce").fillna(0.0)
    pressure = pd.to_numeric(events.get("event_pressure"), errors="coerce").fillna(0.0)
    max_score = pd.to_numeric(events.get("max_score"), errors="coerce").fillna(0.0)
    signal_count = pd.to_numeric(events.get("signal_count"), errors="coerce").fillna(0.0)
    unique_symbols = pd.to_numeric(events.get("unique_symbols"), errors="coerce").fillna(0.0)
    pressure_selected = (
        pd.to_numeric(events.get("pressure_gate_selected"), errors="coerce").fillna(0).astype(int).eq(1)
        if "pressure_gate_selected" in events
        else pd.Series(False, index=events.index, dtype=bool)
    )
    close_session_selected = (
        pd.to_numeric(events.get("close_session_gate_selected"), errors="coerce").fillna(0).astype(int).eq(1)
        if "close_session_gate_selected" in events
        else pd.Series(False, index=events.index, dtype=bool)
    )
    if str(family).lower() == "open":
        return (
            event_score * 10.0
            + np.log1p(pressure.clip(lower=0.0)) * 0.90
            + max_score * 0.70
            + signal_count * 0.04
            + unique_symbols * 0.08
            + pressure_selected.astype(float) * 0.90
        )
    return (
        event_score * 12.0
        + np.log1p(pressure.clip(lower=0.0)) * 1.10
        + max_score * 0.65
        + signal_count * 0.03
        + unique_symbols * 0.08
        + pressure_selected.astype(float) * 0.80
        + close_session_selected.astype(float) * 0.80
    )


def _cap_production_frequency_events(
    events: pd.DataFrame,
    base_keep: pd.Series,
    *,
    group_cols: list[str],
    cap: int,
    rank_score: pd.Series,
    preserve_best_action: bool = False,
) -> pd.Series:
    keep = pd.Series(False, index=events.index, dtype=bool)
    if events.empty:
        return keep
    frame = events.loc[base_keep].copy()
    if frame.empty:
        return keep
    frame["_production_frequency_rank_score"] = rank_score.loc[frame.index].astype(float)
    frame["_production_frequency_day"] = pd.to_datetime(frame["event_bucket"], errors="coerce").dt.floor("D")
    frame["_production_frequency_slot"] = pd.to_datetime(frame["event_bucket"], errors="coerce").dt.floor("240min")
    for _, group in frame.groupby(group_cols, sort=False, dropna=False):
        sorted_group = group.sort_values(
            ["event_bucket", "_production_frequency_rank_score", "event_pressure", "event_score", "max_score"],
            ascending=[True, False, False, False, False],
        )
        # ``preserve_best_action`` used to reserve capacity for an action that
        # might only appear later in the day.  That is not knowable online, so
        # production caps are now always first-come with same-time rank ties.
        _ = preserve_best_action
        picked = [int(idx) for idx in sorted_group.head(max(0, int(cap))).index]
        if picked:
            keep.loc[picked] = True
    return keep


def fit_leg_prior(signals: pd.DataFrame, matches: pd.DataFrame, *, event_frame: pd.DataFrame) -> LegPrior:
    if signals.empty:
        return LegPrior({}, {}, {}, {}, {})
    frame = signals.copy()
    positive_ids = set(matches[matches["status"].astype(str).eq("MATCH")]["signal_id"].astype(str)) if not matches.empty else set()
    frame["is_positive_signal"] = frame["signal_id"].astype(str).isin(positive_ids)
    positives = frame[frame["is_positive_signal"]].copy()
    asset_group_action: dict[tuple[str, str], float] = {}
    family_action: dict[tuple[str, str], float] = {}
    if not positives.empty:
        frame["asset_group"] = frame["symbol"].astype(str).map(weijie_market_prior_group)
        positives["asset_group"] = positives["symbol"].astype(str).map(weijie_market_prior_group)
        total_group_counts = frame.groupby(["asset_group", "action"], sort=False).size()
        pos_group_counts = positives.groupby(["asset_group", "action"], sort=False).size()
        for key, total_count in total_group_counts.items():
            pos_count = float(pos_group_counts.get(key, 0.0))
            asset_group_action[(str(key[0]), str(key[1]))] = float(
                (pos_count + 1.0) / (float(total_count) + 4.0),
            )

        total_family_counts = frame.groupby(["reason_family", "action"], sort=False).size()
        pos_family_counts = positives.groupby(["reason_family", "action"], sort=False).size()
        for key, total_count in total_family_counts.items():
            pos_count = float(pos_family_counts.get(key, 0.0))
            family_action[(str(key[0]), str(key[1]))] = float((pos_count + 1.0) / (float(total_count) + 4.0))
    legs_by_action_family: dict[tuple[str, str], int] = {}
    default_legs_by_action: dict[str, int] = {}
    if not event_frame.empty and "positive_signal_count" in event_frame:
        positives_events = event_frame[pd.to_numeric(event_frame["positive_signal_count"], errors="coerce").gt(0)].copy()
        if not positives_events.empty:
            grouped = positives_events.groupby(["action", "dominant_family"], sort=False)["positive_signal_count"].median()
            for key, value in grouped.items():
                legs_by_action_family[(str(key[0]), str(key[1]))] = max(1, int(round(_finite_float(value, 1.0))))
            action_grouped = positives_events.groupby("action", sort=False)["positive_signal_count"].median()
            for key, value in action_grouped.items():
                default_legs_by_action[str(key)] = max(1, int(round(_finite_float(value, 1.0))))
    return LegPrior(
        asset_group_action=asset_group_action,
        family_action=family_action,
        session_state_action={},
        legs_by_action_family=legs_by_action_family,
        default_legs_by_action=default_legs_by_action,
    )


def _combine_risk_profiles(*profiles: pd.DataFrame) -> pd.DataFrame:
    frames = [frame for frame in profiles if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    if "source" in out:
        out["source"] = out["source"].fillna("").astype(str)
    for col in ("sample_count", "median_notional_usdt", "median_leverage", "median_risk_basis_usdt"):
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.reset_index(drop=True)


def _load_operation_risk_profile(
    path: str | Path | None,
    *,
    train_end: pd.Timestamp,
    train_window_days: int,
) -> pd.DataFrame:
    if path is None or not str(path).strip():
        return pd.DataFrame()
    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"operation risk labels CSV not found: {source}")
    operations = pd.read_csv(source)
    operations["事件时间"] = pd.to_datetime(operations.get("事件时间"), errors="coerce")
    start = pd.Timestamp(train_end) - pd.Timedelta(days=max(1, int(train_window_days)))
    operations = operations[
        operations["事件时间"].between(start, pd.Timestamp(train_end), inclusive="both")
    ].copy()
    return build_public_operation_risk_profile(operations)


def _with_signal_cluster_timing(signals: pd.DataFrame, *, gap_minutes: float = 60.0) -> pd.DataFrame:
    if signals.empty or "signal_time" not in signals:
        return signals.copy()
    out = signals.copy()
    out["signal_time"] = pd.to_datetime(out["signal_time"], errors="coerce")
    out["signal_cluster_age_min"] = 0.0
    out["signal_cluster_count"] = 1
    out["signal_prev_gap_min"] = np.nan
    sort_cols = [
        col
        for col in ("symbol", "action", "reason_family", "signal_time", "signal_id")
        if col in out.columns
    ]
    ordered = out.sort_values(sort_cols).copy()
    group_cols = ["symbol", "action", "reason_family"]
    # ``groupby`` historically drops null keys, while NaT signal rows are
    # skipped without advancing the preceding time.  Filter both before the
    # vectorized calculations to retain those exact semantics.
    valid = ordered["signal_time"].notna() & ordered[group_cols].notna().all(axis=1)
    if not bool(valid.any()):
        return out

    work = ordered.loc[valid, [*group_cols, "signal_time"]].copy()
    source_index = work.index.copy()
    work.index = pd.RangeIndex(len(work))

    group_codes = work.groupby(group_cols, sort=False).ngroup()
    previous_time = work.groupby(group_codes, sort=False)["signal_time"].shift()
    # Scalar ``Timedelta.total_seconds()`` (used by the former loop) truncates
    # sub-microsecond precision.  Cast before the vectorized conversion so
    # nanosecond inputs remain bit-for-bit compatible as well.
    previous_delta = (work["signal_time"] - previous_time).astype("timedelta64[us]")
    previous_gap = previous_delta.dt.total_seconds() / 60.0
    previous_gap_values = previous_gap.to_numpy(dtype=float, copy=False)
    starts_cluster = pd.Series(
        ~np.isfinite(previous_gap_values) | (previous_gap_values > float(gap_minutes)),
        index=work.index,
    )
    cluster_number = starts_cluster.groupby(group_codes, sort=False).cumsum()
    cluster_count = work.groupby([group_codes, cluster_number], sort=False).cumcount().add(1)
    cluster_start = (
        work["signal_time"]
        .where(starts_cluster)
        .groupby(group_codes, sort=False)
        .ffill()
    )
    cluster_age_delta = (work["signal_time"] - cluster_start).astype("timedelta64[us]")
    cluster_age = cluster_age_delta.dt.total_seconds() / 60.0

    assignments = pd.DataFrame(
        {
            "signal_cluster_age_min": cluster_age.to_numpy(copy=False),
            "signal_cluster_count": cluster_count.to_numpy(copy=False),
            "signal_prev_gap_min": previous_gap_values,
            "_assigned": True,
        },
        index=source_index,
    )
    # The previous per-row ``out.loc[idx]`` writes updated every row sharing a
    # duplicate index label.  Its final value was therefore the last valid
    # ordered assignment for that label; preserve that behavior explicitly.
    assignments = assignments.loc[~assignments.index.duplicated(keep="last")]
    mapped = assignments.reindex(out.index)
    assigned = mapped["_assigned"].notna().to_numpy(dtype=bool)
    for column in (
        "signal_cluster_age_min",
        "signal_cluster_count",
        "signal_prev_gap_min",
    ):
        values = out[column].to_numpy(copy=True)
        values[assigned] = mapped[column].to_numpy(copy=False)[assigned]
        out[column] = values
    return out


def run_combo_event_state_machine(
    open_signals: pd.DataFrame,
    close_signals: pd.DataFrame,
    open_events: pd.DataFrame,
    close_events: pd.DataFrame,
    feature_rows: pd.DataFrame,
    *,
    open_prior: LegPrior,
    close_prior: LegPrior,
    risk_profile: pd.DataFrame,
    config: WeijieComboEventStrategyConfig,
    dynamic_risk_config: dict[str, float] | None = None,
    initial_runtime_state: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    runtime_state = initial_runtime_state or {}

    def state_key(value: object) -> tuple[str, str] | None:
        if isinstance(value, tuple) and len(value) == 2:
            return str(value[0]), str(value[1])
        if isinstance(value, list) and len(value) == 2:
            return str(value[0]), str(value[1])
        return None

    def timestamp_map(name: str) -> dict[tuple[str, str], pd.Timestamp]:
        raw = runtime_state.get(name, {})
        if not isinstance(raw, Mapping):
            return {}
        out: dict[tuple[str, str], pd.Timestamp] = {}
        for raw_key, raw_value in raw.items():
            key = state_key(raw_key)
            value = pd.to_datetime(raw_value, errors="coerce")
            if key is not None and pd.notna(value):
                out[key] = pd.Timestamp(value)
        return out

    def daily_count_map(name: str) -> dict[pd.Timestamp, int]:
        raw = runtime_state.get(name, {})
        if not isinstance(raw, Mapping):
            return {}
        out: dict[pd.Timestamp, int] = {}
        for raw_day, raw_count in raw.items():
            day = pd.to_datetime(raw_day, errors="coerce")
            if pd.isna(day):
                continue
            try:
                count = max(0, int(raw_count))
            except (TypeError, ValueError):
                continue
            out[pd.Timestamp(day).floor("D")] = count
        return out

    active: dict[tuple[str, str], list[dict[str, Any]]] = {}
    raw_active = runtime_state.get("active", {})
    if isinstance(raw_active, Mapping):
        for raw_key, raw_lots in raw_active.items():
            key = state_key(raw_key)
            if key is None or not isinstance(raw_lots, list):
                continue
            lots: list[dict[str, Any]] = []
            for raw_lot in raw_lots:
                if not isinstance(raw_lot, Mapping):
                    continue
                lot = dict(raw_lot)
                open_time = pd.to_datetime(lot.get("open_time"), errors="coerce")
                if pd.notna(open_time):
                    lot["open_time"] = pd.Timestamp(open_time)
                lots.append(lot)
            if lots:
                active[key] = lots

    recall_seen_state: set[tuple[str, str]] = set(active)
    raw_recall_seen = runtime_state.get("recall_seen_state", ())
    if isinstance(raw_recall_seen, (list, tuple, set, frozenset)):
        recall_seen_state.update(
            key
            for raw_key in raw_recall_seen
            if (key := state_key(raw_key)) is not None
        )
    open_rows: list[dict[str, Any]] = []
    close_rows: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    daily_open_counts = daily_count_map("daily_open_counts")
    daily_non_eth_campaign_open_counts = daily_count_map("daily_non_eth_campaign_open_counts")
    daily_close_counts = daily_count_map("daily_close_counts")
    last_open_by_state = timestamp_map("last_open_by_state")
    last_close_by_state = timestamp_map("last_close_by_state")
    open_history_by_state: dict[tuple[str, str], list[pd.Timestamp]] = {}
    raw_open_history = runtime_state.get("open_history_by_state", {})
    if isinstance(raw_open_history, Mapping):
        for raw_key, raw_values in raw_open_history.items():
            key = state_key(raw_key)
            if key is None or not isinstance(raw_values, (list, tuple)):
                continue
            values = pd.to_datetime(pd.Series(list(raw_values)), errors="coerce").dropna()
            if not values.empty:
                open_history_by_state[key] = [pd.Timestamp(value) for value in values]

    event_records: list[tuple[pd.Timestamp, int, str, dict[str, Any]]] = []
    selected_open_events = open_events[pd.to_numeric(open_events.get("selected_event"), errors="coerce").eq(1)].copy()
    selected_close_events = close_events[pd.to_numeric(close_events.get("selected_event"), errors="coerce").eq(1)].copy()
    close_signal_times = {
        str(row.get("signal_id")): pd.Timestamp(row.get("signal_time"))
        for _, row in close_signals.iterrows()
        if row.get("signal_id") is not None and pd.notna(pd.Timestamp(row.get("signal_time")))
    }
    for _, event in selected_open_events.iterrows():
        ordering_time = pd.Timestamp(event.get("decision_time", event["event_bucket"]))
        event_records.append((ordering_time, 1, "OPEN", dict(event)))
    for _, event in selected_close_events.iterrows():
        ordering_time = pd.Timestamp(event.get("decision_time", event["event_bucket"]))
        if (
            (config.production_eth_blowoff_handoff or config.production_eth_blowoff_ladder)
            and int(_finite_float(event.get("eth_blowoff_close_event_selected"), 0.0)) == 1
        ):
            candidate_times = [
                close_signal_times[signal_id]
                for signal_id in str(event.get("signal_ids") or "").split(",")
                if signal_id in close_signal_times
            ]
            if candidate_times:
                ordering_time = max(ordering_time, max(candidate_times))
        event_records.append((ordering_time, 0, "CLOSE", dict(event)))
    event_records.sort(key=lambda item: (item[0], item[1], str(item[3].get("action")), -_finite_float(item[3].get("event_score"))))

    open_lookup = _signals_by_event(open_signals)
    close_lookup = _signals_by_event(close_signals)
    for event_time, _, event_type, event in event_records:
        action = str(event.get("action") or "")
        bucket = pd.Timestamp(event.get("event_bucket"))
        if event_type == "OPEN":
            candidates = open_lookup.get((bucket, action), pd.DataFrame())
            if candidates.empty:
                continue
            if config.recall_compression:
                candidates = _filter_recall_compression_legs(candidates, event_type="open")
                if candidates.empty:
                    continue
            if (config.recall_first or config.production_mode) and config.recall_open_family_score_floors:
                candidates = _filter_recall_open_family_score_floors(
                    candidates,
                    open_prior,
                    active=active,
                    family_score_floors=config.recall_open_family_score_floors,
                )
                if candidates.empty:
                    continue
            blowoff_ladder_candidates = candidates.copy()
            if config.production_mode and config.production_open_quality_gate:
                candidates = _filter_production_open_quality_candidates(candidates, event, config=config)
                if config.production_follower_catchup_addon:
                    candidates = _augment_follower_catchup_addon_candidates(
                        candidates,
                        blowoff_ladder_candidates,
                        active=active,
                        event=event,
                    )
                    candidates = _filter_production_open_quality_candidates(
                        candidates,
                        event,
                        config=config,
                    )
                if candidates.empty:
                    continue
            if config.production_mode and config.production_non_eth_contribution_campaign:
                candidates = _filter_non_eth_contribution_campaign_candidates(
                    candidates,
                    event,
                    eligible_symbols=config.production_non_eth_campaign_symbols,
                )
                if candidates.empty:
                    continue
            if config.production_mode and (
                config.production_eth_blowoff_handoff or config.production_eth_blowoff_ladder
            ):
                candidates = _filter_eth_blowoff_handoff_candidates(candidates, active=active)
                if candidates.empty:
                    continue
            if config.production_mode and config.production_meme_core_trend_confirmation_only:
                candidates = _filter_meme_core_trend_confirmation_candidates(
                    candidates,
                    active=active,
                    event=event,
                    allow_breadth_bootstrap=bool(config.production_meme_core_trend_breadth_bootstrap),
                )
                if candidates.empty:
                    continue
            if config.production_mode and config.production_prefilter_ineligible_open_legs:
                candidates = _filter_ineligible_open_state_candidates(
                    candidates,
                    active=active,
                    config=config,
                )
                if candidates.empty:
                    continue
            if config.production_mode and config.production_point_accuracy_gate:
                candidates = _apply_production_open_point_accuracy_scores(candidates, event)
            open_hard_cap = int(config.max_open_trades_per_bucket)
            if config.recall_first and config.recall_compression:
                open_hard_cap = min(open_hard_cap, max(1, int(config.recall_compression_open_leg_cap)))
            if config.production_mode and config.production_open_full_event_budget:
                leg_budget = open_hard_cap
            else:
                leg_budget = _event_leg_budget(
                    event,
                    open_prior,
                    default_cap=open_hard_cap,
                    hard_cap=open_hard_cap,
                    recall_first=bool(config.recall_first),
                )
                if config.production_mode and config.production_cluster_open_budget:
                    leg_budget = _production_open_leg_budget(event, current_budget=leg_budget, hard_cap=open_hard_cap)
            meme_leg_cap, meme_second_floor = _open_meme_leg_limits(event, config=config)
            selected = _select_event_legs(
                candidates,
                open_prior,
                leg_budget=leg_budget,
                active=active,
                for_close=False,
                recall_first=bool(config.recall_first),
                allow_open_addons=bool(config.production_mode and config.production_allow_pyramiding),
                max_lots_per_state=max(1, int(config.production_max_lots_per_state)),
                max_lots_by_asset_group={
                    "eth": max(1, int(config.production_eth_max_lots_per_state or config.production_max_lots_per_state)),
                    "stock_gold": max(1, int(config.production_stock_gold_max_lots_per_state or config.production_max_lots_per_state)),
                    "meme": max(1, int(config.production_meme_max_lots_per_state or config.production_max_lots_per_state)),
                },
                max_open_legs_by_asset_group=(
                    {"meme": meme_leg_cap}
                    if meme_leg_cap > 0
                    else None
                ),
                meme_second_leg_quality_ratio_floor=meme_second_floor,
                unified_open_ranking=bool(config.production_unified_leg_ranking),
            )
            if config.production_mode and config.production_eth_blowoff_ladder:
                selected = _augment_eth_blowoff_ladder_candidates(
                    selected,
                    blowoff_ladder_candidates,
                    active=active,
                )
            if config.production_mode and config.production_open_quality_gate:
                selected = _filter_production_open_quality_candidates(selected, event, config=config)
                if selected.empty:
                    continue
            if config.production_mode and config.production_trend_saturation_gate:
                selected = _filter_production_trend_saturation_candidates(selected)
                if selected.empty:
                    continue
            for _, row in selected.iterrows():
                row = row.copy()
                row["signal_time"] = _production_open_execution_time(row, config=config)
                if config.production_causal_event_execution:
                    row["signal_time"] = max(
                        pd.Timestamp(row["signal_time"]),
                        pd.Timestamp(event_time),
                    )
                direction = direction_from_action(row.get("action"))
                state_key = (str(row.get("symbol")), direction)
                lot_count = _active_lot_count(active, state_key)
                if (
                    lot_count > 0
                    and int(_finite_float(row.get("non_eth_contribution_campaign"), 0.0)) == 1
                ):
                    continue
                if lot_count > 0 and not config.recall_first:
                    if not (config.production_mode and config.production_allow_pyramiding):
                        continue
                    row_max_lots = _production_max_lots_for_symbol(str(row.get("symbol") or ""), config=config)
                    if lot_count >= row_max_lots:
                        if not _production_over_cap_open_allowed(
                            row,
                            lot_count=lot_count,
                            max_lots=row_max_lots,
                        ):
                            continue
                    last_open_time = _active_last_open_time(active, state_key)
                    if (
                        config.production_mode
                        and float(config.production_min_addon_gap_minutes) > 0.0
                        and last_open_time is not None
                    ):
                        min_addon_gap = _production_addon_gap_minutes(row, config=config)
                        gap_minutes = (
                            pd.Timestamp(row.get("signal_time")) - pd.Timestamp(last_open_time)
                        ).total_seconds() / 60.0
                        if gap_minutes < min_addon_gap:
                            continue
                elif lot_count > 0:
                    continue
                selected_id = f"WJCOMBOOP{len(open_rows) + 1:04d}"
                signal_time = pd.Timestamp(row.get("signal_time"))
                if (
                    config.production_mode
                    and not config.recall_first
                    and float(config.production_min_reentry_gap_minutes) > 0.0
                    and lot_count <= 0
                ):
                    last_open_time = last_open_by_state.get(state_key)
                    if last_open_time is not None:
                        reentry_gap = _production_reentry_gap_minutes(row, config=config)
                        gap_minutes = (signal_time - pd.Timestamp(last_open_time)).total_seconds() / 60.0
                        if gap_minutes < reentry_gap:
                            continue
                if (
                    config.production_mode
                    and not config.recall_first
                    and float(config.production_state_open_cap_window_days) > 0.0
                ):
                    state_open_cap = _production_state_open_cap(row, config=config)
                    if state_open_cap > 0:
                        window_start = signal_time - pd.Timedelta(
                            days=float(config.production_state_open_cap_window_days),
                        )
                        recent_opens = [
                            pd.Timestamp(item)
                            for item in open_history_by_state.get(state_key, [])
                            if pd.Timestamp(item) >= window_start
                        ]
                        open_history_by_state[state_key] = recent_opens
                        if len(recent_opens) >= state_open_cap:
                            continue
                open_day = signal_time.floor("D")
                is_non_eth_campaign = int(
                    _finite_float(row.get("non_eth_contribution_campaign"), 0.0),
                ) == 1
                campaign_daily_cap = max(
                    0,
                    int(config.production_non_eth_campaign_daily_open_cap),
                )
                if (
                    is_non_eth_campaign
                    and campaign_daily_cap > 0
                    and daily_non_eth_campaign_open_counts.get(open_day, 0) >= campaign_daily_cap
                ):
                    continue
                if is_non_eth_campaign:
                    last_close_time = last_close_by_state.get(state_key)
                    reentry_gap = max(
                        0.0,
                        float(config.production_non_eth_campaign_reentry_gap_minutes),
                    )
                    if (
                        last_close_time is not None
                        and 300.0
                        < (signal_time - last_close_time).total_seconds() / 60.0
                        < reentry_gap
                    ):
                        continue
                if (
                    config.production_mode
                    and int(config.production_open_daily_trade_cap) > 0
                    and daily_open_counts.get(open_day, 0) >= int(config.production_open_daily_trade_cap)
                ):
                    continue
                price = _feature_close_at_or_before(feature_rows, str(row.get("symbol")), signal_time)
                if dynamic_risk_config:
                    symbol = str(row.get("symbol") or "")
                    asset_group = weijie_asset_group(symbol)
                    min_stop_distance_pct, max_stop_distance_pct = _dynamic_stop_distance_bounds(
                        row,
                        config=config,
                        min_stop_distance_pct=float(
                            dynamic_risk_config.get("min_stop_distance_pct", 0.80),
                        ),
                        max_stop_distance_pct=float(
                            dynamic_risk_config.get("max_stop_distance_pct", 4.00),
                        ),
                    )
                    quality = _finite_float(row.get("factor_quality_score"), np.nan)
                    quality_floor = _finite_float(row.get("factor_quality_floor"), np.nan)
                    quality_multiplier = (
                        quality / quality_floor
                        if np.isfinite(quality) and np.isfinite(quality_floor) and quality_floor > 0.0
                        else 1.0
                    )
                    max_leverage = float(
                        dynamic_risk_config.get(f"max_leverage_{asset_group}", 6.0),
                    )
                    if int(_finite_float(row.get("non_eth_contribution_campaign"), 0.0)) == 1:
                        max_leverage = min(max_leverage, 6.0)
                    sizing = resolve_dynamic_risk_sizing(
                        symbol=symbol,
                        action=str(row.get("action")),
                        account_equity_usdt=float(dynamic_risk_config.get("reference_equity_usdt", 500.0)),
                        price=price,
                        atr_pct=_finite_float(row.get("atr_pct"), np.nan),
                        position_lot_count=lot_count,
                        quality_multiplier=quality_multiplier,
                        risk_budget_pct=float(dynamic_risk_config.get("risk_budget_pct", 0.35)),
                        atr_stop_multiplier=float(dynamic_risk_config.get("atr_stop_multiplier", 2.20)),
                        min_stop_distance_pct=min_stop_distance_pct,
                        max_stop_distance_pct=max_stop_distance_pct,
                        max_single_margin_pct=float(dynamic_risk_config.get("max_single_margin_pct", 3.00)),
                        max_leverage=max_leverage,
                        position_lot_risk_multiplier=float(
                            dynamic_risk_config.get("position_lot_risk_multiplier", 0.65),
                        ),
                        execution_margin_stop_pct=float(
                            dynamic_risk_config.get("execution_margin_stop_pct", 0.0),
                        ),
                        target_margin_pct=float(
                            dict(dynamic_risk_config.get("family_margin_pct", {})).get(
                                str(row.get("reason_family") or ""),
                                0.0,
                            ),
                        ),
                    )
                else:
                    anchor_short_leverage_cap = float(config.risk_anchor_short_leverage_cap)
                    if (
                        str(row.get("symbol") or "") == "ETHUSDT"
                        and str(row.get("action") or "") == "OPEN_SHORT"
                        and str(row.get("reason_family") or "") == "core_downtrend_short"
                        and float(config.risk_eth_core_downtrend_short_leverage_cap) > 0.0
                    ):
                        anchor_short_leverage_cap = float(config.risk_eth_core_downtrend_short_leverage_cap)
                    sizing = resolve_risk_sizing(
                        risk_profile,
                        symbol=str(row.get("symbol")),
                        action=str(row.get("action")),
                        reason_family=str(row.get("reason_family") or ""),
                        signal_time=signal_time,
                        price=price,
                        signal_score=_finite_float(row.get("score"), np.nan),
                        event_score=_finite_float(event.get("event_score"), np.nan),
                        event_pressure=_finite_float(event.get("event_pressure"), np.nan),
                        ret20=_finite_float(row.get("ret20"), np.nan),
                        ret60=_finite_float(row.get("ret60"), np.nan),
                        ret144=_finite_float(row.get("ret144"), np.nan),
                        ret288=_finite_float(row.get("ret288"), np.nan),
                        don=_finite_float(row.get("don"), np.nan),
                        taker=_finite_float(row.get("taker"), np.nan),
                        seller=_finite_float(row.get("seller"), np.nan),
                        qvr=_finite_float(row.get("qvr"), np.nan),
                        rsi=_finite_float(row.get("rsi"), np.nan),
                        adx=_finite_float(row.get("adx"), np.nan),
                        position_lot_count=lot_count,
                        non_anchor_leverage_cap=float(config.risk_non_anchor_leverage_cap),
                        anchor_short_leverage_cap=anchor_short_leverage_cap,
                    )
                if dynamic_risk_config:
                    portfolio_correlation = _current_portfolio_correlation(
                        feature_rows,
                        symbol=str(row.get("symbol") or ""),
                        direction=direction,
                        active=active,
                        at_time=signal_time,
                        window_bars=int(dynamic_risk_config.get("correlation_window_bars", 96)),
                    )
                    correlation_cap = _production_portfolio_correlation_cap(
                        row,
                        active=active,
                        config=config,
                        base_cap=float(
                            dynamic_risk_config.get("max_portfolio_correlation", 0.85),
                        ),
                    )
                    if portfolio_correlation > correlation_cap:
                        continue
                    active_margin_pct = sum(
                        _finite_float(lot.get("margin_pct"), 0.0)
                        for lots in active.values()
                        for lot in lots
                    )
                    if active_margin_pct + float(sizing.margin_pct) > float(config.account_max_total_margin_pct):
                        continue
                else:
                    portfolio_correlation = 0.0
                risk_fields = _risk_fields_from_sizing(sizing)
                risk_fields["operation_type"] = "ADD" if lot_count > 0 else "OPEN"
                if dynamic_risk_config:
                    risk_fields["sizing_reason"] = (
                        f"{risk_fields['sizing_reason']};portfolio_corr={portfolio_correlation:.4f};"
                        f"portfolio_margin={active_margin_pct:.4f}%"
                    )
                open_rows.append(
                    _selected_signal_row(
                        selected_id,
                        row,
                        signal_time,
                        reason_prefix=(
                            f"combo_event_open;event_score={_finite_float(event.get('event_score')):.4f};"
                            f"event={event.get('event_id')}"
                        ),
                        risk_fields=risk_fields,
                    ),
                )
                active.setdefault(state_key, []).append(
                    {
                        "open_time": signal_time,
                        "open_price": price,
                        "direction": direction,
                        "symbol": str(row.get("symbol")),
                        "open_signal_id": selected_id,
                        "open_reason_family": str(row.get("reason_family") or ""),
                        "open_reason": _open_reason_with_quality(row),
                        "open_score": _finite_float(row.get("leg_score"), _finite_float(row.get("score"))),
                        "campaign_entry_ret60": _finite_float(row.get("ret60"), np.nan),
                        "follower_catchup_augmented": int(
                            _finite_float(row.get("follower_catchup_augmented"), 0.0),
                        ),
                        "non_eth_contribution_campaign": int(is_non_eth_campaign),
                        **risk_fields,
                    },
                )
                daily_open_counts[open_day] = daily_open_counts.get(open_day, 0) + 1
                if is_non_eth_campaign:
                    daily_non_eth_campaign_open_counts[open_day] = (
                        daily_non_eth_campaign_open_counts.get(open_day, 0) + 1
                    )
                last_open_by_state[state_key] = signal_time
                open_history_by_state.setdefault(state_key, []).append(signal_time)
                recall_seen_state.add(state_key)
            continue

        candidates = close_lookup.get((bucket, action), pd.DataFrame())
        if candidates.empty:
            continue
        if (
            config.production_mode
            and (config.production_eth_blowoff_handoff or config.production_eth_blowoff_ladder)
            and int(_finite_float(event.get("eth_blowoff_close_event_selected"), 0.0)) == 1
        ):
            candidate_symbol = candidates.get("symbol", pd.Series("", index=candidates.index)).astype(str)
            candidate_family = candidates.get("reason_family", pd.Series("", index=candidates.index)).astype(str)
            candidates = candidates[
                candidate_symbol.eq("ETHUSDT")
                & candidate_family.eq("short_profit_take_exhaustion")
            ].copy()
            if candidates.empty:
                continue
        if (
            config.production_mode
            and config.production_deferred_close_persistence
            and int(_finite_float(event.get("deferred_close_persistence_selected"), 0.0)) == 1
        ):
            candidates = candidates[
                candidates.get("symbol", pd.Series("", index=candidates.index)).astype(str).eq("ETHUSDT")
            ].copy()
            if candidates.empty:
                continue
        if config.recall_compression:
            candidates = _filter_recall_compression_legs(candidates, event_type="close")
            if candidates.empty:
                continue
        if (config.recall_first and config.recall_state_presence_gate) or config.production_mode:
            candidates = _filter_recall_state_presence_candidates(candidates, recall_seen_state)
            if candidates.empty:
                continue
        if (
            config.production_mode
            and config.production_leader_leg_quality_gate
            and config.production_leader_leg_hard_gate
        ):
            candidates = _filter_leader_leg_quality_candidates(candidates, allow_risk_exit=True)
            if candidates.empty:
                continue
        campaign_close_candidates = candidates.copy()
        if (
            (config.recall_first or config.production_mode)
            and (float(config.recall_close_score_floor) > 0.0 or config.recall_close_family_score_floors)
        ):
            candidates = _filter_recall_close_leg_score_floor(
                candidates,
                close_prior,
                active=active,
                score_floor=float(config.recall_close_score_floor),
                family_score_floors=config.recall_close_family_score_floors,
            )
            if config.production_mode and config.production_non_eth_contribution_campaign:
                campaign_keep: list[int] = []
                for candidate_index, campaign_row in campaign_close_candidates.iterrows():
                    campaign_state = (
                        str(campaign_row.get("symbol") or ""),
                        direction_from_action(campaign_row.get("action")),
                    )
                    campaign_lots = active.get(campaign_state, []) or []
                    if any(
                        int(_finite_float(lot.get("non_eth_contribution_campaign"), 0.0)) == 1
                        for lot in campaign_lots
                    ):
                        campaign_keep.append(int(candidate_index))
                if campaign_keep:
                    candidates = pd.concat(
                        [candidates, campaign_close_candidates.loc[campaign_keep]],
                        ignore_index=False,
                        sort=False,
                    ).loc[lambda frame: ~frame.index.duplicated(keep="first")]
            if candidates.empty:
                continue
        if config.production_mode and config.production_non_eth_contribution_campaign:
            candidates = _filter_non_eth_campaign_inventory_close_candidates(
                candidates,
                event,
                active=active,
                feature_rows=feature_rows,
                config=config,
            )
            if candidates.empty:
                continue
        close_hard_cap = int(config.max_close_trades_per_bucket)
        if config.recall_first and config.recall_compression:
            close_hard_cap = min(close_hard_cap, max(1, int(config.recall_compression_close_leg_cap)))
        leg_budget = _event_leg_budget(
            event,
            close_prior,
            default_cap=close_hard_cap,
            hard_cap=close_hard_cap,
            recall_first=bool(config.recall_first),
        )
        if config.production_mode and config.production_cluster_close_budget:
            leg_budget = _production_close_leg_budget(
                event,
                current_budget=leg_budget,
                hard_cap=close_hard_cap,
                deferred_persistence=bool(config.production_deferred_close_persistence),
            )
        selected = _select_event_legs(
            candidates,
            close_prior,
            leg_budget=leg_budget,
            active=active,
            for_close=True,
            recall_first=bool(config.recall_first),
            eth_cascade_onset_ranking=bool(config.production_eth_cascade_profit_exit),
        )
        for _, row in selected.iterrows():
            direction = direction_from_action(row.get("action"))
            state_key = (str(row.get("symbol")), direction)
            if _active_lot_count(active, state_key) <= 0 and not config.recall_first:
                continue
            if config.recall_first and config.recall_state_presence_gate and state_key not in recall_seen_state:
                continue
            if (
                config.recall_first
                and float(config.recall_close_score_floor) > 0.0
                and _finite_float(row.get("leg_score"), 0.0) < float(config.recall_close_score_floor)
            ):
                continue
            signal_time = pd.Timestamp(row.get("signal_time"))
            if config.production_causal_event_execution:
                signal_time = max(signal_time, pd.Timestamp(event_time))
                row = row.copy()
                row["signal_time"] = signal_time
            current_price = _feature_close_at_or_before(feature_rows, str(row.get("symbol")), signal_time)
            reason_family = str(row.get("reason_family") or "")
            closing_lots = active.get(state_key, []) or []
            full_reset_close = reason_family in set(
                config.production_full_reset_close_families,
            )
            requested_close_fraction = 1.0
            if config.production_honor_signal_close_fraction and not full_reset_close:
                requested_close_fraction = min(
                    1.0,
                    max(0.0, _finite_float(row.get("close_fraction"), 1.0)),
                )
            partial_single_lot = bool(
                0.0 < requested_close_fraction < 1.0
                and len(closing_lots) == 1,
            )
            preserve_single_close = bool(
                config.production_mode
                and not config.recall_first
                and config.production_preserve_lots_on_single_close
                and not full_reset_close,
            )
            if preserve_single_close:
                ordered_closing_lots = _ordered_closing_lots(
                    closing_lots,
                    direction=direction,
                    current_price=current_price,
                    reason_family=reason_family,
                    symbol=str(row.get("symbol") or ""),
                    profit_scaleout_families=(
                        config.production_profit_scaleout_close_families
                    ),
                )
            else:
                ordered_closing_lots = list(closing_lots)
            first_lot = ordered_closing_lots[0] if ordered_closing_lots else {}
            open_time = first_lot.get("open_time") or _active_first_open_time(active, state_key)
            open_price = _finite_float(first_lot.get("open_price"), _active_first_open_price(active, state_key))
            unrealized_roi = _point_return_pct(direction, open_price, current_price)
            hold_minutes = (
                (signal_time - pd.Timestamp(open_time)).total_seconds() / 60.0
                if open_time is not None
                else 0.0
            )
            if config.production_mode and config.production_close_deferral:
                if not _production_close_allowed(
                    row,
                    event,
                    hold_minutes=hold_minutes,
                    unrealized_roi_pct=unrealized_roi,
                    lot=first_lot,
                    config=config,
                ):
                    continue
            min_hold_bypass = bool(
                config.production_mode
                and _production_min_hold_bypass(
                    row,
                    event,
                    hold_minutes=hold_minutes,
                    unrealized_roi_pct=unrealized_roi,
                    config=config,
                    lot=first_lot,
                ),
            )
            if (
                hold_minutes < float(config.min_hold_minutes)
                and not reason_family.startswith("portfolio_")
                and not config.recall_first
                and not min_hold_bypass
            ):
                continue
            emit_count = 1
            if config.production_mode and config.production_emit_close_per_lot and not config.recall_first:
                emit_count = max(1, len(ordered_closing_lots))
            elif config.production_mode and not config.recall_first:
                emit_count = _eth_blowoff_profitable_lot_harvest_count(
                    ordered_closing_lots,
                    direction=direction,
                    current_price=current_price,
                    row=row,
                    event=event,
                    config=config,
                )
            emitted_close_count = 0
            for lot_index in range(emit_count):
                lot = ordered_closing_lots[lot_index] if lot_index < len(ordered_closing_lots) else {}
                selected_id = f"WJCOMBOCL{len(close_rows) + 1:04d}"
                lot_open_time = lot.get("open_time", open_time)
                lot_hold_minutes = (
                    (signal_time - pd.Timestamp(lot_open_time)).total_seconds() / 60.0
                    if lot_open_time is not None
                    else hold_minutes
                )
                close_day = signal_time.floor("D")
                if (
                    config.production_mode
                    and int(config.production_close_daily_trade_cap) > 0
                    and daily_close_counts.get(close_day, 0) >= int(config.production_close_daily_trade_cap)
                ):
                    break
                lot_open_price = _finite_float(lot.get("open_price"), open_price)
                lot_roi = _point_return_pct(direction, lot_open_price, current_price)
                close_risk_fields = (
                    _aggregate_lot_risk_fields(ordered_closing_lots)
                    if emit_count == 1 and ordered_closing_lots and not preserve_single_close
                    else _risk_fields_from_lot(lot)
                )
                if partial_single_lot:
                    close_risk_fields = _scaled_lot_risk_fields(
                        lot,
                        requested_close_fraction,
                    )
                    close_risk_fields["close_fraction"] = requested_close_fraction
                    close_risk_fields["operation_type"] = "REDUCE"
                elif full_reset_close or (not preserve_single_close and emit_count == 1):
                    close_risk_fields["operation_type"] = "CLOSE"
                else:
                    remaining_after = max(0, len(closing_lots) - lot_index - 1)
                    close_risk_fields["operation_type"] = (
                        "CLOSE"
                        if remaining_after == 0 and emit_count >= len(closing_lots)
                        else "REDUCE"
                    )
                close_rows.append(
                    _selected_signal_row(
                        selected_id,
                        row,
                        signal_time,
                        reason_prefix=(
                            f"combo_event_close;hold={lot_hold_minutes:.1f};event_score={_finite_float(event.get('event_score')):.4f};"
                            f"u_roi={_finite_float(lot_roi, np.nan):.4f};close_lot={lot_index + 1}/{emit_count};"
                            f"open_signal={lot.get('open_signal_id', '')};event={event.get('event_id')}"
                        ),
                        risk_fields=close_risk_fields,
                        ),
                        )
                daily_close_counts[close_day] = daily_close_counts.get(close_day, 0) + 1
                emitted_close_count += 1
                if ordered_closing_lots:
                    if partial_single_lot:
                        lots_closed_by_signal = [
                            _scaled_lot(lot, requested_close_fraction),
                        ] if lot else []
                    elif emit_count == 1 and not preserve_single_close:
                        lots_closed_by_signal = list(ordered_closing_lots)
                    else:
                        lots_closed_by_signal = [lot] if lot else []
                    for closed_lot_index, closed_lot in enumerate(lots_closed_by_signal, start=1):
                        ledger_rows.append(
                            _strategy_ledger_position_row(
                                closed_lot,
                                close_row=row,
                                close_signal_id=selected_id,
                                close_time=signal_time,
                                close_price=current_price,
                                direction=direction,
                                reason_family=reason_family,
                                event=event,
                                close_group_index=closed_lot_index,
                                close_group_count=len(lots_closed_by_signal),
                            ),
                        )
            if emitted_close_count <= 0:
                continue
            if partial_single_lot and ordered_closing_lots:
                remaining_lot = _scaled_lot(
                    ordered_closing_lots[0],
                    1.0 - requested_close_fraction,
                )
                if _finite_float(remaining_lot.get("quantity"), 0.0) > 0.0:
                    active[state_key] = [remaining_lot]
                else:
                    active.pop(state_key, None)
                continue
            if any(
                int(_finite_float(lot.get("non_eth_contribution_campaign"), 0.0)) == 1
                for lot in closing_lots
            ):
                last_close_by_state[state_key] = signal_time
            if (
                emit_count > 1
                and emitted_close_count < len(ordered_closing_lots)
                and ordered_closing_lots
                and not config.recall_first
            ):
                closed_lot_ids = {id(lot) for lot in ordered_closing_lots[:emitted_close_count]}
                remaining_lots = [lot for lot in closing_lots if id(lot) not in closed_lot_ids]
                if remaining_lots:
                    active[state_key] = remaining_lots
                else:
                    active.pop(state_key, None)
                continue
            if (
                config.production_mode
                and config.production_allow_pyramiding
                and not config.production_emit_close_per_lot
                and preserve_single_close
                and not config.recall_first
            ):
                closed_lot_ids = {id(lot) for lot in ordered_closing_lots[:emitted_close_count]}
                remaining_lots = [lot for lot in closing_lots if id(lot) not in closed_lot_ids]
                if remaining_lots:
                    active[state_key] = remaining_lots
                else:
                    active.pop(state_key, None)
            else:
                active.pop(state_key, None)

    return (
        _selected_signal_frame(open_rows),
        _selected_signal_frame(close_rows),
        _strategy_ledger_position_frame(ledger_rows),
    )


def build_strategy_ledger_performance(
    positions: pd.DataFrame,
    *,
    initial_equity_usdt: float = 10_000.0,
    capital_allocation_pct: float = 100.0,
    public_roi_basis_usdt: float = 0.0,
    per_trade_margin_pct: float = 2.5,
    max_total_margin_pct: float = 60.0,
    round_trip_cost_bps: float = 14.0,
    quality_weighted_sizing: bool = True,
    risk_prior_weighted_sizing: bool = False,
    signal_risk_basis_sizing: bool = False,
    risk_prior_reference_usdt: float = np.nan,
    risk_prior_exponent: float = 0.50,
    risk_prior_max_multiplier: float = 3.0,
    cash_flows: pd.DataFrame | None = None,
    mark_prices: pd.DataFrame | None = None,
    eth_main_leg_margin_floor_pct: float = 0.0,
    drawdown_throttle_start_pct: float = 0.0,
    drawdown_throttle_margin_cap_pct: float = 100.0,
    liquidation_guard_config: LiquidationGuardConfig | None = None,
) -> pd.DataFrame:
    columns = [
        "series",
        "positions",
        "total_point_roi_pct",
        "total_margin_roi_pct",
        "total_realized_pnl_usdt",
        "pnl_proxy_base_usdt",
        "pnl_proxy_return_pct",
        "win_rate_pct",
        "profit_factor",
        "max_drawdown_point_pct",
        "max_drawdown_margin_pct",
        "max_drawdown_pnl_proxy_pct",
        "pnl_proxy_base_method",
        "execution_price_source",
        "account_initial_equity_usdt",
        "account_capital_allocation_pct",
        "account_allocated_initial_equity_usdt",
        "account_allocated_ending_equity_usdt",
        "account_allocated_max_total_margin_pct",
        "account_public_roi_basis_usdt",
        "account_ending_equity_usdt",
        "account_return_pct",
        "account_max_drawdown_pct",
        "account_daily_max_drawdown_pct",
        "account_accepted_positions",
        "account_rejected_positions",
        "account_closed_positions",
        "account_win_rate_pct",
        "account_position_cycles",
        "account_winning_position_cycles",
        "account_position_cycle_win_rate_pct",
        "account_total_cost_usdt",
        "account_total_deposit_usdt",
        "account_total_withdrawal_usdt",
        "account_net_external_flow_usdt",
        "account_total_funding_usdt",
        "account_total_explicit_fee_usdt",
        "account_net_profit_usdt",
        "account_strategy_return_pct",
        "account_public_comparable_roi_pct",
        "account_contributed_capital_return_pct",
        "account_raw_equity_change_pct",
        "account_cash_flow_rows",
        "account_cash_flow_coverage",
        "account_cash_flow_valid",
        "account_cash_flow_shortfall_usdt",
        "account_max_margin_utilization_pct",
        "account_allocated_max_margin_utilization_pct",
        "account_worst_trade_equity_loss_pct",
        "account_margin_loss_over_100_positions",
        "account_per_trade_margin_pct",
        "account_max_total_margin_pct",
        "account_round_trip_cost_bps",
        "account_quality_weighted_sizing",
        "account_risk_prior_weighted_sizing",
        "account_signal_risk_basis_sizing",
        "account_mark_to_market_margin",
        "account_eth_main_leg_margin_floor_pct",
        "account_drawdown_throttle_start_pct",
        "account_drawdown_throttle_margin_cap_pct",
        "account_risk_prior_reference_usdt",
        "account_risk_prior_exponent",
        "account_risk_prior_max_multiplier",
        "account_liquidation_guard_enabled",
        "account_liquidation_guard_rejected_positions",
        "account_curve_method",
    ]
    if positions.empty:
        return pd.DataFrame(columns=columns)
    frame = positions.copy()
    roi = pd.to_numeric(frame.get("point_roi_pct"), errors="coerce")
    margin_roi = pd.to_numeric(frame.get("margin_roi_pct"), errors="coerce")
    pnl = pd.to_numeric(frame.get("realized_pnl_usdt"), errors="coerce")
    valid = roi.notna()
    valid_roi = roi[valid]
    valid_margin_roi = margin_roi[valid]
    valid_pnl = pnl[valid]
    base = _max_concurrent_ledger_risk_basis(frame.loc[valid].copy())
    if not np.isfinite(base) or base <= 0.0:
        risk_basis = pd.to_numeric(frame.get("risk_basis_usdt"), errors="coerce")
        base = float(risk_basis[risk_basis > 0.0].sum()) if not risk_basis.empty else 0.0
    total_pnl = float(valid_pnl.sum()) if not valid_pnl.empty else 0.0
    pnl_curve = pd.concat([pd.Series([0.0]), valid_pnl.fillna(0.0).cumsum()], ignore_index=True)
    pnl_drawdown = pnl_curve - pnl_curve.cummax()
    row = {
        "series": "strategy_state_ledger_kline_execution",
        "positions": int(valid.sum()),
        "total_point_roi_pct": float(valid_roi.sum()) if not valid_roi.empty else 0.0,
        "total_margin_roi_pct": float(valid_margin_roi.sum()) if not valid_margin_roi.empty else 0.0,
        "total_realized_pnl_usdt": total_pnl,
        "pnl_proxy_base_usdt": float(base) if np.isfinite(base) else 0.0,
        "pnl_proxy_return_pct": total_pnl / float(base) * 100.0 if np.isfinite(base) and base > 0.0 else 0.0,
        "win_rate_pct": float(valid_roi.gt(0.0).mean() * 100.0) if not valid_roi.empty else 0.0,
        "profit_factor": _profit_factor_from_pnl(valid_pnl),
        "max_drawdown_point_pct": _max_drawdown_from_series(valid_roi),
        "max_drawdown_margin_pct": _max_drawdown_from_series(valid_margin_roi),
        "max_drawdown_pnl_proxy_pct": (
            float(pnl_drawdown.min()) / float(base) * 100.0 if np.isfinite(base) and base > 0.0 else 0.0
        ),
        "pnl_proxy_base_method": "max concurrent strategy ledger risk_basis_usdt",
        "execution_price_source": "strategy state ledger, 5m feature close at or before signal_time",
    }
    row.update(
        build_compounded_account_performance(
            frame.loc[valid].copy(),
            initial_equity_usdt=initial_equity_usdt,
            capital_allocation_pct=capital_allocation_pct,
            public_roi_basis_usdt=public_roi_basis_usdt,
            per_trade_margin_pct=per_trade_margin_pct,
            max_total_margin_pct=max_total_margin_pct,
            round_trip_cost_bps=round_trip_cost_bps,
            quality_weighted_sizing=quality_weighted_sizing,
            risk_prior_weighted_sizing=risk_prior_weighted_sizing,
            signal_risk_basis_sizing=signal_risk_basis_sizing,
            risk_prior_reference_usdt=risk_prior_reference_usdt,
            risk_prior_exponent=risk_prior_exponent,
            risk_prior_max_multiplier=risk_prior_max_multiplier,
            cash_flows=cash_flows,
            mark_prices=mark_prices,
            eth_main_leg_margin_floor_pct=eth_main_leg_margin_floor_pct,
            drawdown_throttle_start_pct=drawdown_throttle_start_pct,
            drawdown_throttle_margin_cap_pct=drawdown_throttle_margin_cap_pct,
            liquidation_guard_config=liquidation_guard_config,
        ),
    )
    return pd.DataFrame([row], columns=columns)


def apply_strategy_ledger_hard_stops(
    positions: pd.DataFrame,
    feature_rows: pd.DataFrame,
    *,
    max_margin_loss_pct: float,
) -> pd.DataFrame:
    """Apply the live margin-loss stop to each ledger lot using future 5m bars."""

    if positions.empty:
        return positions.copy()
    out = positions.copy()
    out["hard_stop_applied"] = False
    out["planned_close_time"] = pd.to_datetime(out.get("close_time"), errors="coerce")
    out["planned_close_price"] = pd.to_numeric(out.get("close_price"), errors="coerce")
    out["planned_close_reason_family"] = out.get("close_reason_family", pd.Series("", index=out.index)).astype(str)
    stop_margin_pct = max(0.0, float(max_margin_loss_pct))
    dynamic_stop_values = pd.to_numeric(
        out.get("stop_distance_pct", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    has_dynamic_stops = dynamic_stop_values.gt(0.0).any()
    if feature_rows.empty or (stop_margin_pct <= 0.0 and not has_dynamic_stops):
        return out

    bars = feature_rows.copy()
    bars["bar_time"] = pd.to_datetime(bars.get("bar_time"), errors="coerce")
    bars["symbol"] = bars.get("symbol", pd.Series("", index=bars.index)).astype(str)
    for column in ("open", "high", "low"):
        bars[column] = pd.to_numeric(bars.get(column), errors="coerce")
    bars = bars.dropna(subset=["bar_time", "symbol"]).sort_values(["symbol", "bar_time"])
    bars_by_symbol = {str(symbol): group for symbol, group in bars.groupby("symbol", sort=False)}

    def apply_stop_to_rows(
        row_indices: list[object],
        *,
        direction: str,
        stop_time: pd.Timestamp,
        fill_price: float,
        stop_reason: str,
    ) -> None:
        for target_index in row_indices:
            position = out.loc[target_index]
            open_time = pd.Timestamp(position["open_time"])
            planned_close = pd.Timestamp(position["planned_close_time"])
            if planned_close < stop_time:
                continue
            open_price = _finite_float(position.get("open_price"), np.nan)
            quantity = _finite_float(position.get("quantity"), 0.0)
            risk_basis = _finite_float(position.get("risk_basis_usdt"), np.nan)
            realized_pnl = _realized_pnl_usdt(
                direction,
                open_price,
                fill_price,
                quantity,
            )
            out.at[target_index, "close_time"] = stop_time
            out.at[target_index, "close_price"] = fill_price
            out.at[target_index, "hold_minutes"] = (
                (stop_time - open_time).total_seconds() / 60.0
            )
            out.at[target_index, "point_roi_pct"] = _point_return_pct(
                direction,
                open_price,
                fill_price,
            )
            out.at[target_index, "realized_pnl_usdt"] = realized_pnl
            out.at[target_index, "margin_roi_pct"] = _margin_roi_pct(
                realized_pnl,
                risk_basis,
            )
            out.at[target_index, "close_reason_family"] = "hard_stop_loss"
            out.at[target_index, "close_reason"] = (
                f"{stop_reason};planned_close={planned_close};"
                f"planned_family={position.get('planned_close_reason_family')}"
            )
            out.at[target_index, "hard_stop_applied"] = True

    # Live replaces the symbol-level closePosition STOP after every add using
    # the exchange's weighted average entry. Replaying campaign lots against
    # their original entries would create backtest-only stops, so every
    # overlapping same-symbol, same-direction campaign follows that venue
    # behavior here.
    campaign_handled: set[object] = set()
    campaign_seed_mask = (
        out.get("symbol", pd.Series("", index=out.index)).astype(str).astype(bool)
        & out.get("direction", pd.Series("", index=out.index)).astype(str).str.upper().isin(
            {"LONG", "SHORT"},
        )
    )
    for _, seed_rows in out[campaign_seed_mask].groupby(
        out.get("open_signal_id", pd.Series(out.index.astype(str), index=out.index)),
        sort=False,
    ):
        if any(index in campaign_handled for index in seed_rows.index):
            continue
        symbol = str(seed_rows.iloc[0].get("symbol") or "")
        direction = str(seed_rows.iloc[0].get("direction") or "").upper()
        start = pd.to_datetime(seed_rows.get("open_time"), errors="coerce").min()
        campaign_end = pd.to_datetime(
            seed_rows.get("planned_close_time"),
            errors="coerce",
        ).max()
        symbol_bars = bars_by_symbol.get(symbol)
        if (
            symbol_bars is None
            or direction not in {"LONG", "SHORT"}
            or pd.isna(start)
            or pd.isna(campaign_end)
        ):
            continue
        addon_mask = (
            out.get("symbol", pd.Series("", index=out.index)).astype(str).eq(symbol)
            & out.get("direction", pd.Series("", index=out.index)).astype(str).str.upper().eq(direction)
        )
        addon_rows = out[addon_mask].copy()
        addon_open_time = pd.to_datetime(addon_rows.get("open_time"), errors="coerce")
        addon_rows = addon_rows[
            addon_open_time.gt(pd.Timestamp(start))
            & addon_open_time.le(pd.Timestamp(campaign_end))
        ]
        if addon_rows.empty:
            continue

        stage_groups = [seed_rows]
        stage_groups.extend(
            frame
            for _, frame in addon_rows.groupby(
                addon_rows.get(
                    "open_signal_id",
                    pd.Series(addon_rows.index.astype(str), index=addon_rows.index),
                ),
                sort=False,
            )
        )
        stage_groups.sort(
            key=lambda frame: pd.Timestamp(
                pd.to_datetime(frame.get("open_time"), errors="coerce").min(),
            ),
        )
        active_indices: list[object] = []
        stop_triggered = False
        for stage_index, stage_rows in enumerate(stage_groups):
            stage_time = pd.to_datetime(stage_rows.get("open_time"), errors="coerce").min()
            if pd.isna(stage_time):
                continue
            active_indices.extend(stage_rows.index)
            active_indices = list(dict.fromkeys(active_indices))
            live_indices = [
                index
                for index in active_indices
                if pd.Timestamp(out.at[index, "planned_close_time"]) >= pd.Timestamp(stage_time)
            ]
            quantities = pd.to_numeric(out.loc[live_indices, "quantity"], errors="coerce").fillna(0.0)
            entry_prices = pd.to_numeric(out.loc[live_indices, "open_price"], errors="coerce")
            total_quantity = float(quantities.sum())
            if total_quantity <= 0.0 or entry_prices.isna().any():
                continue
            weighted_entry = float((entry_prices * quantities).sum() / total_quantity)
            stage_stop_distance = _finite_float(
                stage_rows.iloc[0].get("stop_distance_pct"),
                np.nan,
            )
            leverage = _finite_float(stage_rows.iloc[0].get("leverage"), np.nan)
            if np.isfinite(stage_stop_distance) and stage_stop_distance > 0.0:
                price_stop_pct = stage_stop_distance / 100.0
                stop_reason = (
                    f"campaign_weighted_dynamic_stop={stage_stop_distance:.4f}%"
                )
            elif np.isfinite(leverage) and leverage > 0.0:
                price_stop_pct = stop_margin_pct / leverage / 100.0
                stop_reason = f"campaign_weighted_margin_stop={stop_margin_pct:.3f}%"
            else:
                continue
            stage_end = (
                pd.to_datetime(stage_groups[stage_index + 1].get("open_time"), errors="coerce").min()
                if stage_index + 1 < len(stage_groups)
                else pd.Timestamp(campaign_end)
            )
            path = symbol_bars[
                symbol_bars["bar_time"].gt(pd.Timestamp(stage_time))
                & symbol_bars["bar_time"].le(pd.Timestamp(stage_end))
            ]
            stop_price = weighted_entry * (
                1.0 - price_stop_pct if direction == "LONG" else 1.0 + price_stop_pct
            )
            hits = (
                path[pd.to_numeric(path["low"], errors="coerce").le(stop_price)]
                if direction == "LONG"
                else path[pd.to_numeric(path["high"], errors="coerce").ge(stop_price)]
            )
            if hits.empty:
                continue
            hit = hits.iloc[0]
            bar_open = _finite_float(hit.get("open"), stop_price)
            fill_price = (
                min(stop_price, bar_open)
                if direction == "LONG"
                else max(stop_price, bar_open)
            )
            stop_time = pd.Timestamp(hit["bar_time"])
            apply_stop_to_rows(
                live_indices,
                direction=direction,
                stop_time=stop_time,
                fill_price=fill_price,
                stop_reason=stop_reason,
            )
            campaign_handled.update(live_indices)
            stop_triggered = True
            break
        if not stop_triggered:
            campaign_handled.update(
                index
                for frame in stage_groups
                for index in frame.index
            )

    for row_index, position in out.iterrows():
        if row_index in campaign_handled:
            continue
        symbol = str(position.get("symbol") or "")
        direction = str(position.get("direction") or "").upper()
        open_time = pd.to_datetime(position.get("open_time"), errors="coerce")
        close_time = pd.to_datetime(position.get("close_time"), errors="coerce")
        open_price = _finite_float(position.get("open_price"), np.nan)
        leverage = _finite_float(position.get("leverage"), np.nan)
        symbol_bars = bars_by_symbol.get(symbol)
        if (
            symbol_bars is None
            or pd.isna(open_time)
            or pd.isna(close_time)
            or not np.isfinite(open_price)
            or open_price <= 0.0
            or not np.isfinite(leverage)
            or leverage <= 0.0
            or direction not in {"LONG", "SHORT"}
        ):
            continue
        stop_distance_pct = _finite_float(position.get("stop_distance_pct"), np.nan)
        if np.isfinite(stop_distance_pct) and stop_distance_pct > 0.0:
            price_stop_pct = stop_distance_pct / 100.0
            stop_reason = f"dynamic_atr_stop={stop_distance_pct:.4f}%"
        else:
            price_stop_pct = stop_margin_pct / leverage / 100.0
            stop_reason = f"hard_stop_margin_loss={stop_margin_pct:.3f}%"
        if price_stop_pct <= 0.0:
            continue
        path = symbol_bars[
            symbol_bars["bar_time"].gt(pd.Timestamp(open_time))
            & symbol_bars["bar_time"].le(pd.Timestamp(close_time))
        ]
        if path.empty:
            continue
        if direction == "LONG":
            stop_price = open_price * (1.0 - price_stop_pct)
            hits = path[pd.to_numeric(path["low"], errors="coerce").le(stop_price)]
        else:
            stop_price = open_price * (1.0 + price_stop_pct)
            hits = path[pd.to_numeric(path["high"], errors="coerce").ge(stop_price)]
        if hits.empty:
            continue
        hit = hits.iloc[0]
        bar_open = _finite_float(hit.get("open"), stop_price)
        fill_price = min(stop_price, bar_open) if direction == "LONG" else max(stop_price, bar_open)
        stop_time = pd.Timestamp(hit["bar_time"])
        quantity = _finite_float(position.get("quantity"), 0.0)
        risk_basis = _finite_float(position.get("risk_basis_usdt"), np.nan)
        realized_pnl = _realized_pnl_usdt(direction, open_price, fill_price, quantity)
        out.at[row_index, "close_time"] = stop_time
        out.at[row_index, "close_price"] = fill_price
        out.at[row_index, "hold_minutes"] = (stop_time - pd.Timestamp(open_time)).total_seconds() / 60.0
        out.at[row_index, "point_roi_pct"] = _point_return_pct(direction, open_price, fill_price)
        out.at[row_index, "realized_pnl_usdt"] = realized_pnl
        out.at[row_index, "margin_roi_pct"] = _margin_roi_pct(realized_pnl, risk_basis)
        out.at[row_index, "close_reason_family"] = "hard_stop_loss"
        out.at[row_index, "close_reason"] = (
            f"{stop_reason};"
            f"planned_close={pd.Timestamp(close_time)};"
            f"planned_family={position.get('close_reason_family')}"
        )
        out.at[row_index, "hard_stop_applied"] = True
    return out


def build_compounded_account_performance(
    positions: pd.DataFrame,
    *,
    initial_equity_usdt: float = 10_000.0,
    capital_allocation_pct: float = 100.0,
    public_roi_basis_usdt: float = 0.0,
    per_trade_margin_pct: float = 2.5,
    max_total_margin_pct: float = 60.0,
    round_trip_cost_bps: float = 14.0,
    quality_weighted_sizing: bool = True,
    risk_prior_weighted_sizing: bool = False,
    signal_risk_basis_sizing: bool = False,
    risk_prior_reference_usdt: float = np.nan,
    risk_prior_exponent: float = 0.50,
    risk_prior_max_multiplier: float = 3.0,
    cash_flows: pd.DataFrame | None = None,
    mark_prices: pd.DataFrame | None = None,
    eth_main_leg_margin_floor_pct: float = 0.0,
    drawdown_throttle_start_pct: float = 0.0,
    drawdown_throttle_margin_cap_pct: float = 100.0,
    liquidation_guard_config: LiquidationGuardConfig | None = None,
) -> dict[str, Any]:
    metrics, _ = build_compounded_account_ledger(
        positions,
        initial_equity_usdt=initial_equity_usdt,
        capital_allocation_pct=capital_allocation_pct,
        public_roi_basis_usdt=public_roi_basis_usdt,
        per_trade_margin_pct=per_trade_margin_pct,
        max_total_margin_pct=max_total_margin_pct,
        round_trip_cost_bps=round_trip_cost_bps,
        quality_weighted_sizing=quality_weighted_sizing,
        risk_prior_weighted_sizing=risk_prior_weighted_sizing,
        signal_risk_basis_sizing=signal_risk_basis_sizing,
        risk_prior_reference_usdt=risk_prior_reference_usdt,
        risk_prior_exponent=risk_prior_exponent,
        risk_prior_max_multiplier=risk_prior_max_multiplier,
        cash_flows=cash_flows,
        mark_prices=mark_prices,
        eth_main_leg_margin_floor_pct=eth_main_leg_margin_floor_pct,
        drawdown_throttle_start_pct=drawdown_throttle_start_pct,
        drawdown_throttle_margin_cap_pct=drawdown_throttle_margin_cap_pct,
        liquidation_guard_config=liquidation_guard_config,
    )
    return metrics


ACCOUNT_LEDGER_COLUMNS = [
    "event_time",
    "event_type",
    "source",
    "source_index",
    "symbol",
    "direction",
    "amount_usdt",
    "equity_usdt",
    "unrealized_pnl_usdt",
    "mark_equity_usdt",
    "active_margin_usdt",
    "available_equity_usdt",
    "net_external_capital_usdt",
    "time_weighted_index",
    "accepted",
    "reason",
]


def load_account_cash_flows(path: str | Path | None) -> pd.DataFrame | None:
    if path is None or not str(path).strip():
        return None
    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"account cash-flow CSV not found: {source}")
    return pd.read_csv(source)


def _normalize_account_cash_flows(cash_flows: pd.DataFrame | None) -> pd.DataFrame:
    columns = ["event_time", "event_type", "amount_usdt", "description"]
    if cash_flows is None or cash_flows.empty:
        return pd.DataFrame(columns=columns)
    required = {"event_time", "event_type", "amount_usdt"}
    missing = sorted(required.difference(cash_flows.columns))
    if missing:
        raise ValueError(f"account cash-flow CSV missing columns: {', '.join(missing)}")
    out = cash_flows.copy().reset_index(drop=True)
    out["event_time"] = pd.to_datetime(out["event_time"], format="mixed", errors="coerce")
    aliases = {
        "TRANSFER_IN": "DEPOSIT",
        "TRANSFER_OUT": "WITHDRAWAL",
        "COMMISSION": "FEE",
    }
    out["event_type"] = out["event_type"].astype(str).str.strip().str.upper().replace(aliases)
    out["amount_usdt"] = pd.to_numeric(out["amount_usdt"], errors="coerce")
    out["description"] = out.get("description", pd.Series("", index=out.index)).fillna("").astype(str)
    supported = {"DEPOSIT", "WITHDRAWAL", "FUNDING", "FEE", "ADJUSTMENT"}
    invalid_types = sorted(set(out["event_type"]).difference(supported))
    if invalid_types:
        raise ValueError(f"unsupported account cash-flow event types: {', '.join(invalid_types)}")
    out = out.dropna(subset=["event_time", "amount_usdt"]).copy()
    out.loc[out["event_type"].eq("DEPOSIT"), "amount_usdt"] = out.loc[
        out["event_type"].eq("DEPOSIT"), "amount_usdt",
    ].abs()
    out.loc[out["event_type"].eq("WITHDRAWAL"), "amount_usdt"] = -out.loc[
        out["event_type"].eq("WITHDRAWAL"), "amount_usdt",
    ].abs()
    out.loc[out["event_type"].eq("FEE"), "amount_usdt"] = -out.loc[
        out["event_type"].eq("FEE"), "amount_usdt",
    ].abs()
    return out[columns].sort_values("event_time", kind="stable").reset_index(drop=True)


def build_compounded_account_ledger(
    positions: pd.DataFrame,
    *,
    initial_equity_usdt: float = 10_000.0,
    capital_allocation_pct: float = 100.0,
    public_roi_basis_usdt: float = 0.0,
    per_trade_margin_pct: float = 2.5,
    max_total_margin_pct: float = 60.0,
    round_trip_cost_bps: float = 14.0,
    quality_weighted_sizing: bool = True,
    risk_prior_weighted_sizing: bool = False,
    signal_risk_basis_sizing: bool = False,
    risk_prior_reference_usdt: float = np.nan,
    risk_prior_exponent: float = 0.50,
    risk_prior_max_multiplier: float = 3.0,
    cash_flows: pd.DataFrame | None = None,
    mark_prices: pd.DataFrame | None = None,
    eth_main_leg_margin_floor_pct: float = 0.0,
    drawdown_throttle_start_pct: float = 0.0,
    drawdown_throttle_margin_cap_pct: float = 100.0,
    liquidation_guard_config: LiquidationGuardConfig | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Replay trades and timestamped cash flows through one auditable account."""

    initial_equity = max(0.0, float(initial_equity_usdt))
    allocation_pct = max(0.0, min(100.0, float(capital_allocation_pct)))
    allocation_fraction = allocation_pct / 100.0
    allocated_initial_equity = initial_equity * allocation_fraction
    allocated_total_margin_pct = (
        min(500.0, max(0.0, float(max_total_margin_pct)) / allocation_fraction)
        if allocation_fraction > 0.0
        else 0.0
    )
    configured_public_roi_basis = max(0.0, float(public_roi_basis_usdt))
    per_trade_pct = max(0.0, float(per_trade_margin_pct))
    total_margin_pct = max(0.0, min(500.0, float(max_total_margin_pct)))
    eth_main_leg_floor_pct = max(0.0, min(100.0, float(eth_main_leg_margin_floor_pct)))
    throttle_start_pct = max(0.0, min(100.0, float(drawdown_throttle_start_pct)))
    throttle_margin_cap_pct = max(0.0, min(500.0, float(drawdown_throttle_margin_cap_pct)))
    cost_bps = max(0.0, float(round_trip_cost_bps))
    flow_coverage = "provided" if cash_flows is not None else "missing"
    flow_frame = _normalize_account_cash_flows(cash_flows)
    mark_price_lookup = _build_account_mark_price_lookup(mark_prices)
    mark_to_market_margin = bool(mark_price_lookup)
    frame = positions.copy().reset_index(drop=True)
    if not frame.empty:
        frame["open_time"] = pd.to_datetime(frame.get("open_time"), errors="coerce")
        frame["close_time"] = pd.to_datetime(frame.get("close_time"), errors="coerce")
        frame = frame.dropna(subset=["open_time", "close_time"]).copy()

    events: list[tuple[pd.Timestamp, int, float, float, str, int]] = []
    for row_index, row in frame.iterrows():
        quality_priority = _ledger_quality_ratio(row) if quality_weighted_sizing else 1.0
        score_priority = _finite_float(row.get("open_score"), 0.0) if quality_weighted_sizing else 0.0
        risk_priority = (
            _ledger_risk_prior_sizing_multiplier(
                row,
                reference_usdt=risk_prior_reference_usdt,
                exponent=risk_prior_exponent,
                max_multiplier=risk_prior_max_multiplier,
            )
            if risk_prior_weighted_sizing
            else 1.0
        )
        events.append((pd.Timestamp(row["close_time"]), 0, 0.0, 0.0, "POSITION_CLOSE", int(row_index)))
        events.append(
            (
                pd.Timestamp(row["open_time"]),
                2,
                -float(quality_priority * risk_priority),
                -float(score_priority),
                "POSITION_OPEN",
                int(row_index),
            ),
        )
    for flow_index, flow in flow_frame.iterrows():
        events.append((pd.Timestamp(flow["event_time"]), 1, 0.0, 0.0, "CASH_FLOW", int(flow_index)))
    if mark_price_lookup and not frame.empty:
        first_open = pd.Timestamp(frame["open_time"].min())
        last_close = pd.Timestamp(frame["close_time"].max())
        mark_time_parts = [
            times[(times >= first_open.value) & (times <= last_close.value)]
            for times, _ in mark_price_lookup.values()
        ]
        mark_time_parts = [times for times in mark_time_parts if len(times)]
        if mark_time_parts:
            for mark_time_ns in np.unique(np.concatenate(mark_time_parts)):
                events.append((pd.Timestamp(mark_time_ns), 3, 0.0, 0.0, "MARK_TO_MARKET", -1))
    events.sort()

    equity = initial_equity
    allocated_equity = allocated_initial_equity
    active_margin = 0.0
    max_margin_utilization_pct = 0.0
    max_allocated_margin_utilization_pct = 0.0
    accepted = 0
    rejected = 0
    liquidation_guard_rejected = 0
    closed_positions = 0
    winning_positions = 0
    closed_net_pnl_by_index: dict[int, float] = {}
    total_cost = 0.0
    worst_trade_equity_loss_pct = 0.0
    margin_loss_over_100 = 0
    total_deposit = 0.0
    total_withdrawal = 0.0
    total_funding = 0.0
    total_explicit_fee = 0.0
    net_external_flow = 0.0
    allocated_net_external_flow = 0.0
    cash_flow_valid = True
    cash_flow_shortfall = 0.0
    active: dict[int, tuple[float, float]] = {}
    campaign_allocations: dict[str, tuple[float, float, bool]] = {}
    ledger_rows: list[dict[str, Any]] = []

    twr_chained = 1.0
    twr_segment_base = initial_equity
    peak_twr = 1.0
    max_drawdown_pct = 0.0

    def unrealized_pnl(event_time: pd.Timestamp | pd.NaT) -> float:
        if not mark_to_market_margin or pd.isna(event_time) or not active:
            return 0.0
        total = 0.0
        for active_index, (margin, _) in active.items():
            active_row = frame.loc[active_index]
            symbol = str(active_row.get("symbol") or "")
            direction = str(active_row.get("direction") or "").upper()
            open_price = _finite_float(active_row.get("open_price"), np.nan)
            leverage = _finite_float(active_row.get("leverage"), 0.0)
            mark_price = _account_mark_price_at_or_before(
                mark_price_lookup,
                symbol=symbol,
                event_time=pd.Timestamp(event_time),
            )
            if (
                not np.isfinite(mark_price)
                or mark_price <= 0.0
                or not np.isfinite(open_price)
                or open_price <= 0.0
                or leverage <= 0.0
                or direction not in {"LONG", "SHORT"}
            ):
                continue
            point_return = mark_price / open_price - 1.0
            if direction == "SHORT":
                point_return = -point_return
            total += float(margin) * leverage * point_return
        return float(total)

    def mark_equity(event_time: pd.Timestamp | pd.NaT) -> float:
        return float(equity + unrealized_pnl(event_time))

    def mark_allocated_equity(event_time: pd.Timestamp | pd.NaT) -> float:
        return float(allocated_equity + unrealized_pnl(event_time))

    def current_twr_index(event_time: pd.Timestamp | pd.NaT = pd.NaT) -> float:
        current_equity = mark_equity(event_time)
        if twr_segment_base > 0.0:
            return twr_chained * max(0.0, current_equity) / twr_segment_base
        return twr_chained

    def mark_drawdown(event_time: pd.Timestamp | pd.NaT) -> float:
        nonlocal peak_twr, max_drawdown_pct
        index = current_twr_index(event_time)
        peak_twr = max(peak_twr, index)
        if peak_twr > 0.0:
            max_drawdown_pct = min(max_drawdown_pct, (index / peak_twr - 1.0) * 100.0)
        return index

    def append_ledger(
        *,
        event_time: pd.Timestamp | pd.NaT,
        event_type: str,
        source: str,
        source_index: int,
        amount_usdt: float,
        accepted_event: bool,
        reason: str,
        row: pd.Series | None = None,
    ) -> None:
        current_unrealized = unrealized_pnl(event_time)
        current_mark_equity = equity + current_unrealized
        ledger_rows.append(
            {
                "event_time": event_time,
                "event_type": event_type,
                "source": source,
                "source_index": source_index,
                "symbol": "" if row is None else str(row.get("symbol") or ""),
                "direction": "" if row is None else str(row.get("direction") or ""),
                "amount_usdt": float(amount_usdt),
                "equity_usdt": float(equity),
                "unrealized_pnl_usdt": float(current_unrealized),
                "mark_equity_usdt": float(current_mark_equity),
                "active_margin_usdt": float(active_margin),
                "available_equity_usdt": float(current_mark_equity - active_margin),
                "net_external_capital_usdt": float(net_external_flow),
                "time_weighted_index": float(mark_drawdown(event_time)),
                "accepted": bool(accepted_event),
                "reason": reason,
            },
        )

    append_ledger(
        event_time=pd.NaT,
        event_type="INITIAL_BALANCE",
        source="account",
        source_index=-1,
        amount_usdt=initial_equity,
        accepted_event=True,
        reason="configured initial equity",
    )

    for event_time, _, _, _, event_kind, row_index in events:
        if event_kind == "MARK_TO_MARKET":
            if active:
                append_ledger(
                    event_time=event_time,
                    event_type="MARK_TO_MARKET",
                    source="market",
                    source_index=-1,
                    amount_usdt=0.0,
                    accepted_event=True,
                    reason="closed 5m portfolio mark",
                )
            continue
        if event_kind == "CASH_FLOW":
            flow = flow_frame.loc[row_index]
            event_type = str(flow["event_type"])
            amount = float(flow["amount_usdt"])
            is_external = event_type in {"DEPOSIT", "WITHDRAWAL"}
            if is_external:
                twr_chained = current_twr_index(event_time)
            requested_amount = amount
            if equity + amount < 0.0:
                cash_flow_valid = False
                cash_flow_shortfall += -(equity + amount)
                amount = -equity
            equity += amount
            if event_type == "DEPOSIT":
                total_deposit += amount
                net_external_flow += amount
                allocated_amount = amount * allocation_fraction
                allocated_equity += allocated_amount
                allocated_net_external_flow += allocated_amount
            elif event_type == "WITHDRAWAL":
                total_withdrawal += -amount
                net_external_flow += amount
                allocated_amount = amount * allocation_fraction
                allocated_equity += allocated_amount
                allocated_net_external_flow += allocated_amount
            elif event_type == "FUNDING":
                total_funding += amount
                allocated_equity += amount
            elif event_type == "FEE":
                total_explicit_fee += -amount
                allocated_equity += amount
            if is_external:
                twr_segment_base = equity
            append_ledger(
                event_time=event_time,
                event_type=event_type,
                source="cash_flow",
                source_index=row_index,
                amount_usdt=amount,
                accepted_event=amount == requested_amount,
                reason=str(flow.get("description") or ""),
            )
            continue

        row = frame.loc[row_index]
        if event_kind == "POSITION_CLOSE":
            allocation = active.pop(row_index, None)
            if allocation is None:
                continue
            margin, equity_at_open = allocation
            active_margin = max(0.0, active_margin - margin)
            leverage = max(0.0, _finite_float(row.get("leverage"), 0.0))
            point_return_pct = _finite_float(row.get("point_roi_pct"), 0.0)
            gross_pnl = margin * leverage * point_return_pct / 100.0
            position_cost_bps = max(0.0, _finite_float(row.get("round_trip_cost_bps"), cost_bps))
            cost = margin * leverage * position_cost_bps / 10_000.0
            net_pnl = gross_pnl - cost
            closed_net_pnl_by_index[int(row_index)] = float(net_pnl)
            closed_positions += 1
            if net_pnl > 0.0:
                winning_positions += 1
            equity += net_pnl
            allocated_equity += net_pnl
            total_cost += cost
            if equity_at_open > 0.0:
                worst_trade_equity_loss_pct = min(
                    worst_trade_equity_loss_pct,
                    net_pnl / equity_at_open * 100.0,
                )
            if point_return_pct * leverage <= -100.0:
                margin_loss_over_100 += 1
            if equity < 0.0:
                cash_flow_valid = False
                cash_flow_shortfall += -equity
                equity = 0.0
            append_ledger(
                event_time=event_time,
                event_type="POSITION_CLOSE",
                source="strategy",
                source_index=row_index,
                amount_usdt=net_pnl,
                accepted_event=True,
                reason=f"gross_pnl={gross_pnl:.8f};estimated_cost={cost:.8f}",
                row=row,
            )
            continue

        current_mark_equity = max(0.0, mark_equity(event_time))
        current_allocated_equity = max(0.0, mark_allocated_equity(event_time))
        campaign_id = str(row.get("margin_campaign_id") or "").strip()
        campaign_total_margin_pct = max(
            0.0,
            _finite_float(row.get("campaign_margin_pct"), 0.0),
        )
        campaign_total_fixed_margin = max(
            0.0,
            _finite_float(row.get("campaign_fixed_margin_usdt"), 0.0),
        )
        campaign_enabled = bool(
            campaign_id
            and (campaign_total_margin_pct > 0.0 or campaign_total_fixed_margin > 0.0),
        )
        quality_multiplier = _ledger_quality_sizing_multiplier(row) if quality_weighted_sizing else 1.0
        risk_multiplier = (
            _ledger_risk_prior_sizing_multiplier(
                row,
                reference_usdt=risk_prior_reference_usdt,
                exponent=risk_prior_exponent,
                max_multiplier=risk_prior_max_multiplier,
            )
            if risk_prior_weighted_sizing
            else 1.0
        )
        fixed_margin_usdt = max(0.0, _finite_float(row.get("fixed_margin_usdt"), 0.0))
        if fixed_margin_usdt > 0.0:
            target_margin = fixed_margin_usdt
        elif signal_risk_basis_sizing:
            target_margin = max(0.0, _finite_float(row.get("risk_basis_usdt"), 0.0)) * quality_multiplier
        else:
            row_margin_pct = _finite_float(row.get("margin_pct"), per_trade_pct)
            if row_margin_pct <= 0.0:
                row_margin_pct = per_trade_pct
            target_margin = current_allocated_equity * row_margin_pct / 100.0 * quality_multiplier * risk_multiplier
        if eth_main_leg_floor_pct > 0.0 and _is_eth_main_risk_leg(row):
            target_margin = max(
                target_margin,
                current_allocated_equity * eth_main_leg_floor_pct / 100.0,
            )
        current_index = current_twr_index(event_time)
        current_drawdown_pct = (
            (current_index / peak_twr - 1.0) * 100.0 if peak_twr > 0.0 else 0.0
        )
        effective_total_margin_pct = allocated_total_margin_pct
        if throttle_start_pct > 0.0 and current_drawdown_pct <= -throttle_start_pct:
            allocated_throttle_cap_pct = (
                throttle_margin_cap_pct / allocation_fraction
                if allocation_fraction > 0.0
                else 0.0
            )
            effective_total_margin_pct = min(
                allocated_total_margin_pct,
                allocated_throttle_cap_pct,
            )
        row_total_margin_pct = max(
            0.0,
            _finite_float(row.get("max_total_margin_pct"), 0.0),
        )
        if row_total_margin_pct > 0.0:
            effective_total_margin_pct = min(
                effective_total_margin_pct,
                row_total_margin_pct,
            )
        allocated_margin_room = max(
            0.0,
            current_allocated_equity * effective_total_margin_pct / 100.0 - active_margin,
        )
        shared_margin_room = max(
            0.0,
            current_mark_equity * total_margin_pct / 100.0 - active_margin,
        )
        margin_room = min(allocated_margin_room, shared_margin_room)
        leverage = max(1.0, _finite_float(row.get("leverage"), 1.0))
        current_notional = sum(
            active_margin_value
            * max(0.0, _finite_float(frame.loc[active_index].get("leverage"), 0.0))
            for active_index, (active_margin_value, _) in active.items()
        )
        row_total_notional_x = max(
            0.0,
            _finite_float(row.get("max_total_notional_x"), 0.0),
        )
        if row_total_notional_x > 0.0:
            notional_room = max(
                0.0,
                current_allocated_equity * row_total_notional_x - current_notional,
            )
            margin_room = min(margin_room, notional_room / leverage)
        campaign_first_lot = False
        campaign_reserved_margin = 0.0
        campaign_allowed = True
        if campaign_enabled:
            cached_campaign = campaign_allocations.get(campaign_id)
            if cached_campaign is None:
                campaign_first_lot = True
                campaign_target_margin = (
                    campaign_total_fixed_margin
                    if campaign_total_fixed_margin > 0.0
                    else current_allocated_equity * campaign_total_margin_pct / 100.0
                )
                campaign_reserved_margin = min(campaign_target_margin, margin_room)
                campaign_scale = (
                    campaign_reserved_margin / campaign_target_margin
                    if campaign_target_margin > 0.0
                    else 0.0
                )
                campaign_allocations[campaign_id] = (
                    campaign_scale,
                    campaign_reserved_margin,
                    True,
                )
            else:
                campaign_scale, campaign_reserved_margin, campaign_allowed = cached_campaign
            target_margin *= campaign_scale
        margin = min(target_margin, margin_room)
        fixed_initial_basis = bool(
            fixed_margin_usdt > 0.0 or campaign_total_fixed_margin > 0.0,
        )
        minimum_margin_basis = (
            allocated_initial_equity
            if fixed_initial_basis
            else current_allocated_equity
        )
        minimum_margin = max(1e-9, minimum_margin_basis * 0.001)
        sizing_enabled = (
            fixed_margin_usdt > 0.0
            or signal_risk_basis_sizing
            or per_trade_pct > 0.0
            or _finite_float(row.get("margin_pct"), 0.0) > 0.0
        )
        campaign_minimum_passed = bool(
            campaign_enabled
            and campaign_allowed
            and campaign_reserved_margin > minimum_margin,
        )
        if (
            not sizing_enabled
            or total_margin_pct <= 0.0
            or not campaign_allowed
            or (margin <= minimum_margin and not campaign_minimum_passed)
        ):
            if campaign_enabled:
                campaign_allocations[campaign_id] = (
                    0.0,
                    campaign_reserved_margin,
                    False,
                )
            rejected += 1
            append_ledger(
                event_time=event_time,
                event_type="POSITION_OPEN_REJECTED",
                source="strategy",
                source_index=row_index,
                amount_usdt=0.0,
                accepted_event=False,
                reason=(
                    f"target_margin={target_margin:.8f};margin_room={margin_room:.8f};"
                    f"drawdown={current_drawdown_pct:.4f};margin_cap={effective_total_margin_pct:.4f}"
                ),
                row=row,
            )
            continue
        if (
            liquidation_guard_config is not None
            and liquidation_guard_config.enabled
            and (not campaign_enabled or campaign_first_lot)
        ):
            guard_margin = campaign_reserved_margin if campaign_enabled else margin
            guard = assess_liquidation_guard(
                equity=current_allocated_equity,
                current_notional=current_notional,
                new_notional=guard_margin * leverage,
                config=replace(liquidation_guard_config, leverage=leverage),
            )
            if not guard.allowed:
                if campaign_enabled:
                    campaign_allocations[campaign_id] = (
                        0.0,
                        campaign_reserved_margin,
                        False,
                    )
                rejected += 1
                liquidation_guard_rejected += 1
                append_ledger(
                    event_time=event_time,
                    event_type="POSITION_OPEN_REJECTED",
                    source="strategy",
                    source_index=row_index,
                    amount_usdt=0.0,
                    accepted_event=False,
                    reason=(
                        f"liquidation_guard={guard.reason};"
                        f"initial_margin={guard.projected_initial_margin_pct:.4f};"
                        f"stress_equity={guard.projected_equity_after_stress_pct:.4f};"
                        f"stress_margin_ratio={guard.projected_stressed_margin_ratio_pct:.4f};"
                        f"projected_stop_loss={guard.projected_stop_loss_pct:.4f}"
                    ),
                    row=row,
                )
                continue
        active[row_index] = (margin, current_allocated_equity)
        active_margin += margin
        accepted += 1
        if current_mark_equity > 0.0:
            max_margin_utilization_pct = max(
                max_margin_utilization_pct,
                active_margin / current_mark_equity * 100.0,
            )
        if current_allocated_equity > 0.0:
            max_allocated_margin_utilization_pct = max(
                max_allocated_margin_utilization_pct,
                active_margin / current_allocated_equity * 100.0,
            )
        append_ledger(
            event_time=event_time,
            event_type="POSITION_OPEN",
            source="strategy",
            source_index=row_index,
            amount_usdt=0.0,
            accepted_event=True,
            reason=(
                f"reserved_margin={margin:.8f};leverage={_finite_float(row.get('leverage'), 0.0):.4f};"
                f"drawdown={current_drawdown_pct:.4f};margin_cap={effective_total_margin_pct:.4f};"
                f"notional_cap_x={row_total_notional_x:.4f}"
            ),
            row=row,
        )

    position_cycles, winning_position_cycles = _account_position_cycle_counts(
        frame,
        closed_net_pnl_by_index,
    )
    ending_twr = current_twr_index(pd.NaT)
    net_profit = allocated_equity - allocated_initial_equity - allocated_net_external_flow
    contributed_capital = initial_equity + total_deposit
    allocated_contributed_capital = contributed_capital * allocation_fraction
    allocated_ending_equity = allocated_equity
    public_roi_basis = configured_public_roi_basis or allocated_contributed_capital
    public_comparable_roi_pct = (
        float(net_profit) / public_roi_basis * 100.0 if public_roi_basis > 0.0 else 0.0
    )
    result = {
        "account_initial_equity_usdt": initial_equity,
        "account_capital_allocation_pct": allocation_pct,
        "account_allocated_initial_equity_usdt": allocated_initial_equity,
        "account_allocated_ending_equity_usdt": float(allocated_ending_equity),
        "account_allocated_max_total_margin_pct": float(allocated_total_margin_pct),
        "account_public_roi_basis_usdt": float(public_roi_basis),
        "account_ending_equity_usdt": float(equity),
        "account_return_pct": (ending_twr - 1.0) * 100.0,
        "account_max_drawdown_pct": float(max_drawdown_pct),
        "account_accepted_positions": int(accepted),
        "account_rejected_positions": int(rejected),
        "account_closed_positions": int(closed_positions),
        "account_win_rate_pct": (
            float(winning_positions) / float(closed_positions) * 100.0
            if closed_positions > 0
            else 0.0
        ),
        "account_position_cycles": int(position_cycles),
        "account_winning_position_cycles": int(winning_position_cycles),
        "account_position_cycle_win_rate_pct": (
            float(winning_position_cycles) / float(position_cycles) * 100.0
            if position_cycles > 0
            else 0.0
        ),
        "account_total_cost_usdt": float(total_cost),
        "account_total_deposit_usdt": float(total_deposit),
        "account_total_withdrawal_usdt": float(total_withdrawal),
        "account_net_external_flow_usdt": float(net_external_flow),
        "account_total_funding_usdt": float(total_funding),
        "account_total_explicit_fee_usdt": float(total_explicit_fee),
        "account_net_profit_usdt": float(net_profit),
        "account_strategy_return_pct": public_comparable_roi_pct,
        "account_public_comparable_roi_pct": public_comparable_roi_pct,
        "account_contributed_capital_return_pct": public_comparable_roi_pct,
        "account_raw_equity_change_pct": (
            (float(equity) / initial_equity - 1.0) * 100.0 if initial_equity > 0.0 else 0.0
        ),
        "account_cash_flow_rows": len(flow_frame),
        "account_cash_flow_coverage": flow_coverage,
        "account_cash_flow_valid": bool(cash_flow_valid),
        "account_cash_flow_shortfall_usdt": float(cash_flow_shortfall),
        "account_max_margin_utilization_pct": float(max_margin_utilization_pct),
        "account_allocated_max_margin_utilization_pct": float(
            max_allocated_margin_utilization_pct,
        ),
        "account_worst_trade_equity_loss_pct": float(worst_trade_equity_loss_pct),
        "account_margin_loss_over_100_positions": int(margin_loss_over_100),
        "account_per_trade_margin_pct": per_trade_pct,
        "account_max_total_margin_pct": total_margin_pct,
        "account_round_trip_cost_bps": cost_bps,
        "account_quality_weighted_sizing": bool(quality_weighted_sizing),
        "account_risk_prior_weighted_sizing": bool(risk_prior_weighted_sizing),
        "account_signal_risk_basis_sizing": bool(signal_risk_basis_sizing),
        "account_mark_to_market_margin": bool(mark_to_market_margin),
        "account_eth_main_leg_margin_floor_pct": float(eth_main_leg_floor_pct),
        "account_drawdown_throttle_start_pct": float(throttle_start_pct),
        "account_drawdown_throttle_margin_cap_pct": float(throttle_margin_cap_pct),
        "account_risk_prior_reference_usdt": _finite_float(risk_prior_reference_usdt, np.nan),
        "account_risk_prior_exponent": max(0.0, float(risk_prior_exponent)),
        "account_risk_prior_max_multiplier": max(1.0, float(risk_prior_max_multiplier)),
        "account_liquidation_guard_enabled": bool(
            liquidation_guard_config is not None and liquidation_guard_config.enabled,
        ),
        "account_liquidation_guard_rejected_positions": int(liquidation_guard_rejected),
        "account_curve_method": (
            "timestamped_cash_flow_reserved_margin_time_weighted_return; close_then_cash_flow_then_open; "
            f"strategy_capital_allocation={allocation_pct:.4f}%; "
            f"strategy_margin_cap={allocated_total_margin_pct:.4f}%_of_allocated_equity; "
            f"sizing={'signal_risk_basis' if signal_risk_basis_sizing else 'equity_percent'}; "
            f"cash_flow_coverage={flow_coverage}; "
            f"margin_equity={'mark_to_market' if mark_to_market_margin else 'realized_only'}; "
            "continuous closed-5m marks plus intrabar emergency stops; liquidation pressure gate modeled"
        ),
    }
    ledger = pd.DataFrame(ledger_rows, columns=ACCOUNT_LEDGER_COLUMNS)
    if not ledger.empty:
        ledger["event_time"] = pd.to_datetime(ledger["event_time"], errors="coerce")
        daily_curve = (
            ledger.dropna(subset=["event_time"])
            .set_index("event_time")["time_weighted_index"]
            .resample("1D")
            .last()
            .dropna()
        )
        daily_drawdown = daily_curve.div(daily_curve.cummax()).sub(1.0).mul(100.0)
        result["account_daily_max_drawdown_pct"] = (
            float(daily_drawdown.min()) if not daily_drawdown.empty else 0.0
        )
    else:
        result["account_daily_max_drawdown_pct"] = 0.0
    return result, ledger


def _account_position_cycle_counts(
    positions: pd.DataFrame,
    closed_net_pnl_by_index: Mapping[int, float],
) -> tuple[int, int]:
    """Aggregate overlapping lots into public-account-comparable position cycles.

    A scale-in or partial close is an execution lot, not a new trade outcome.
    Lots are therefore one cycle while the same symbol and direction have
    overlapping exposure.  Cycle PnL includes the ledger's estimated costs.
    """

    if positions.empty or not closed_net_pnl_by_index:
        return 0, 0
    source_indices = [
        int(index)
        for index in closed_net_pnl_by_index
        if int(index) in positions.index
    ]
    if not source_indices:
        return 0, 0
    frame = positions.loc[source_indices].copy()
    frame["cycle_net_pnl_usdt"] = [
        float(closed_net_pnl_by_index[int(index)])
        for index in frame.index
    ]
    frame["symbol"] = frame.get("symbol", pd.Series("", index=frame.index)).fillna("").astype(str)
    frame["direction"] = (
        frame.get("direction", pd.Series("", index=frame.index)).fillna("").astype(str).str.upper()
    )
    frame["open_time"] = pd.to_datetime(frame.get("open_time"), errors="coerce")
    frame["close_time"] = pd.to_datetime(frame.get("close_time"), errors="coerce")
    frame = frame.dropna(subset=["open_time", "close_time"])
    if frame.empty:
        return 0, 0

    outcomes: list[float] = []
    for _, group in frame.groupby(["symbol", "direction"], sort=False, dropna=False):
        ordered = group.sort_values(["open_time", "close_time"], kind="stable")
        cycle_close: pd.Timestamp | None = None
        cycle_pnl = 0.0
        for _, row in ordered.iterrows():
            open_time = pd.Timestamp(row["open_time"])
            close_time = pd.Timestamp(row["close_time"])
            # The account replay orders events at an identical timestamp as
            # close first, then open.  An open exactly at the prior cycle's
            # close therefore starts fresh exposure instead of overlapping it.
            if cycle_close is None or open_time >= cycle_close:
                if cycle_close is not None:
                    outcomes.append(float(cycle_pnl))
                cycle_close = close_time
                cycle_pnl = float(row["cycle_net_pnl_usdt"])
                continue
            cycle_close = max(cycle_close, close_time)
            cycle_pnl += float(row["cycle_net_pnl_usdt"])
        if cycle_close is not None:
            outcomes.append(float(cycle_pnl))
    return len(outcomes), sum(outcome > 0.0 for outcome in outcomes)


def _build_account_mark_price_lookup(
    mark_prices: pd.DataFrame | None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if mark_prices is None or mark_prices.empty:
        return {}
    frame = mark_prices.copy()
    frame["symbol"] = frame.get("symbol", pd.Series("", index=frame.index)).astype(str)
    frame["bar_time"] = pd.to_datetime(frame.get("bar_time"), errors="coerce")
    frame["close"] = pd.to_numeric(frame.get("close"), errors="coerce")
    frame = frame.dropna(subset=["bar_time", "close"])
    frame = frame[frame["symbol"].astype(bool) & frame["close"].gt(0.0)].copy()
    lookup: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for symbol, group in frame.groupby("symbol", sort=False):
        ordered = group.sort_values("bar_time").drop_duplicates("bar_time", keep="last")
        lookup[str(symbol)] = (
            ordered["bar_time"].astype("int64").to_numpy(),
            ordered["close"].astype(float).to_numpy(),
        )
    return lookup


def _is_eth_main_risk_leg(row: pd.Series) -> bool:
    if str(row.get("symbol") or "") != "ETHUSDT":
        return False
    return str(row.get("open_reason_family") or "") in {
        "capitulation_absorption_long",
        "range_reclaim_long",
        "eth_anchor_long",
        "core_downtrend_short",
    }


def _account_mark_price_at_or_before(
    lookup: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    symbol: str,
    event_time: pd.Timestamp,
) -> float:
    values = lookup.get(str(symbol))
    if values is None or pd.isna(event_time):
        return np.nan
    times, prices = values
    position = int(np.searchsorted(times, pd.Timestamp(event_time).value, side="right") - 1)
    if position < 0:
        return np.nan
    return float(prices[position])


def _ledger_quality_sizing_multiplier(row: pd.Series) -> float:
    ratio = _ledger_quality_ratio(row)
    if not np.isfinite(ratio) or ratio <= 0.0:
        return 1.0
    return float(np.clip(ratio, 0.75, 1.50))


def _ledger_risk_prior_sizing_multiplier(
    row: pd.Series,
    *,
    reference_usdt: float,
    exponent: float,
    max_multiplier: float,
) -> float:
    risk_basis = _finite_float(row.get("risk_basis_usdt"), np.nan)
    reference = _finite_float(reference_usdt, np.nan)
    if not np.isfinite(risk_basis) or risk_basis <= 0.0 or not np.isfinite(reference) or reference <= 0.0:
        return 1.0
    power = max(0.0, float(exponent))
    cap = max(1.0, float(max_multiplier))
    return float(np.clip((risk_basis / reference) ** power, 0.25, cap))


def _ledger_quality_ratio(row: pd.Series) -> float:
    quality_row = row.copy()
    direction = str(row.get("direction") or "").upper()
    quality_row["action"] = "OPEN_LONG" if direction == "LONG" else "OPEN_SHORT"
    quality_row["reason_family"] = str(row.get("open_reason_family") or "")
    for part in str(row.get("open_reason") or "").split(";")[1:]:
        if "=" not in part:
            continue
        key, raw_value = part.split("=", 1)
        try:
            quality_row[key] = float(raw_value)
        except ValueError:
            continue
    quality = production_open_factor_quality_score(quality_row)
    floor = production_open_factor_quality_floor(quality_row)
    if not np.isfinite(quality) or not np.isfinite(floor) or floor <= 0.0:
        return 1.0
    ratio = float(quality / floor)
    forward_probability = _finite_float(quality_row.get("forward_quality_probability"), np.nan)
    if np.isfinite(forward_probability):
        ratio += float(np.clip((forward_probability - 0.50) * 1.50, -0.30, 0.45))
    leader_leg_probability = _finite_float(quality_row.get("leader_leg_quality_probability"), np.nan)
    if np.isfinite(leader_leg_probability):
        ratio += float(np.clip((leader_leg_probability - 0.30) * 1.25, -0.20, 0.45))
    return ratio


def build_event_report(
    train_open_events: pd.DataFrame,
    train_close_events: pd.DataFrame,
    eval_open_events: pd.DataFrame,
    eval_close_events: pd.DataFrame,
    open_gate: EventGate,
    close_gate: EventGate,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family, scope, events, gate in (
        ("open", "train", train_open_events, open_gate),
        ("close", "train", train_close_events, close_gate),
        ("open", "eval", eval_open_events, open_gate),
        ("close", "eval", eval_close_events, close_gate),
    ):
        labels = pd.to_numeric(events.get("event_label"), errors="coerce") if not events.empty else pd.Series(dtype=float)
        selected = pd.to_numeric(events.get("selected_event"), errors="coerce") if not events.empty else pd.Series(dtype=float)
        positives = labels.eq(1)
        selected_positive = selected.eq(1) & positives
        selected_negative = selected.eq(1) & labels.eq(0)
        rows.append(
            {
                "family": family,
                "scope": scope,
                "model_name": gate.model_name,
                "threshold": float(gate.threshold),
                "events": len(events),
                "positive_events": int(positives.sum()) if not labels.empty else 0,
                "selected_events": int(selected.eq(1).sum()) if not selected.empty else 0,
                "selected_positive_events": int(selected_positive.sum()) if not selected.empty else 0,
                "selected_negative_events": int(selected_negative.sum()) if not selected.empty else 0,
                "max_event_score": _finite_float(pd.to_numeric(events.get("event_score"), errors="coerce").max(), np.nan)
                if not events.empty
                else np.nan,
                "transition_threshold": _finite_float(
                    pd.to_numeric(
                        events.get(
                            "production_transition_threshold",
                            pd.Series(np.nan, index=events.index),
                        ),
                        errors="coerce",
                    ).max(),
                    np.nan,
                )
                if not events.empty
                else np.nan,
            },
        )
    return pd.DataFrame(rows)


def _calibrate_event_threshold(labels: np.ndarray, scores: np.ndarray, *, floor: float) -> float:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    if scores.size == 0:
        return 1.0
    candidates = sorted(set(float(value) for value in scores if np.isfinite(value)))
    candidates.extend([float(floor), 0.5, 0.75, 0.9])
    best_threshold = 1.0
    best_score = -float("inf")
    for threshold in sorted(set(max(float(floor), value) for value in candidates)):
        pred = scores >= threshold
        tp = int(((labels == 1) & pred).sum())
        fp = int(((labels == 0) & pred).sum())
        fn = int(((labels == 1) & ~pred).sum())
        selected = int(pred.sum())
        objective = tp * 8.0 - fn * 9.0 - fp * 1.4 - max(0, selected - max(1, int(labels.sum()) * 2)) * 0.8
        if objective > best_score or (abs(objective - best_score) <= 1e-12 and threshold > best_threshold):
            best_score = objective
            best_threshold = threshold
    return float(min(0.98, max(float(floor), best_threshold)))


def _signals_by_event(signals: pd.DataFrame) -> dict[tuple[pd.Timestamp, str], pd.DataFrame]:
    if signals.empty:
        return {}
    out: dict[tuple[pd.Timestamp, str], pd.DataFrame] = {}
    for key, group in signals.groupby(["event_bucket", "action"], sort=False):
        out[(pd.Timestamp(key[0]), str(key[1]))] = group.copy()
    return out


def _filter_recall_compression_legs(candidates: pd.DataFrame, *, event_type: str) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    out = candidates.copy()
    if str(event_type).lower() == "open" and "reason_family" in out:
        family = out["reason_family"].astype(str)
        paxg_defensive = family.eq("paxg_defensive_short")
        if paxg_defensive.any():
            signal_time = pd.to_datetime(out.get("signal_time"), errors="coerce")
            hour = signal_time.dt.hour.fillna(-1)
            paxg_defensive_keep = (
                paxg_defensive
                & out.get("symbol", pd.Series("", index=out.index)).astype(str).eq("PAXGUSDT")
                & pd.to_numeric(out.get("score"), errors="coerce").fillna(0.0).ge(3.35)
                & hour.between(19, 23, inclusive="both")
                & pd.to_numeric(out.get("ret144"), errors="coerce").fillna(0.0).le(0.0)
                & pd.to_numeric(out.get("ret288"), errors="coerce").fillna(0.0).le(0.5)
            )
            out = out[~paxg_defensive | paxg_defensive_keep].copy()
    return out


def _filter_recall_state_presence_candidates(
    candidates: pd.DataFrame,
    seen_state: set[tuple[str, str]],
) -> pd.DataFrame:
    if candidates.empty or not seen_state:
        return candidates.head(0).copy()
    keep = [
        (str(row.get("symbol")), direction_from_action(row.get("action"))) in seen_state
        for _, row in candidates.iterrows()
    ]
    return candidates.loc[keep].copy()


def _filter_leader_leg_quality_candidates(
    candidates: pd.DataFrame,
    *,
    allow_risk_exit: bool,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    selected = pd.to_numeric(
        candidates.get("leader_leg_quality_selected", pd.Series(0, index=candidates.index)),
        errors="coerce",
    ).fillna(0).astype(int).eq(1)
    if allow_risk_exit:
        family = candidates.get("reason_family", pd.Series("", index=candidates.index)).astype(str)
        selected = selected | family.str.contains("risk_cut", regex=False)
    return candidates.loc[selected].copy()


def _filter_recall_close_leg_score_floor(
    candidates: pd.DataFrame,
    prior: LegPrior,
    *,
    active: dict[tuple[str, str], list[dict[str, Any]]],
    score_floor: float,
    family_score_floors: tuple[tuple[str, float], ...] = (),
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    out = candidates.copy()
    out["leg_score"] = [
        _leg_score(row, prior=prior, active=active, for_close=True) for _, row in out.iterrows()
    ]
    floor_values = pd.Series(float(score_floor), index=out.index, dtype=float)
    floors = {
        str(family): float(score)
        for family, score in family_score_floors
        if str(family) and np.isfinite(float(score))
    }
    if floors:
        family = out.get("reason_family", pd.Series("", index=out.index)).astype(str)
        floor_values = family.map(floors).fillna(float(score_floor)).astype(float)
        signal_time = pd.to_datetime(out.get("signal_time"), errors="coerce")
        symbol = out.get("symbol", pd.Series("", index=out.index)).astype(str)
        action = out.get("action", pd.Series("", index=out.index)).astype(str)
        eth_morning_short_risk = (
            symbol.eq("ETHUSDT")
            & action.eq("CLOSE_SHORT")
            & family.eq("short_risk_cut_reversal")
            & signal_time.dt.hour.between(7, 10, inclusive="both").fillna(False)
        )
        floor_values.loc[eth_morning_short_risk] = np.minimum(
            floor_values.loc[eth_morning_short_risk],
            max(float(score_floor), 3.29),
        )
        bch_flush_short_profit = (
            symbol.eq("BCHUSDT")
            & action.eq("CLOSE_SHORT")
            & family.eq("short_profit_take_exhaustion")
            & pd.to_numeric(out.get("ret20"), errors="coerce").fillna(0.0).le(-1.0)
            & pd.to_numeric(out.get("qvr"), errors="coerce").fillna(0.0).ge(8.0)
            & pd.to_numeric(out.get("rsi"), errors="coerce").fillna(50.0).le(35.0)
        )
        floor_values.loc[bch_flush_short_profit] = np.minimum(
            floor_values.loc[bch_flush_short_profit],
            3.00,
        )
        doge_night_profit_release = (
            symbol.eq("DOGEUSDT")
            & action.eq("CLOSE_SHORT")
            & family.eq("short_profit_take_exhaustion")
            & signal_time.dt.hour.eq(21).fillna(False)
            & signal_time.dt.minute.between(40, 50, inclusive="both").fillna(False)
            & pd.to_numeric(out.get("ret60"), errors="coerce").fillna(0.0).le(-2.0)
            & pd.to_numeric(out.get("ret144"), errors="coerce").fillna(0.0).le(-3.0)
            & pd.to_numeric(out.get("rsi"), errors="coerce").fillna(50.0).between(38.0, 48.0, inclusive="both")
        )
        floor_values.loc[doge_night_profit_release] = np.minimum(
            floor_values.loc[doge_night_profit_release],
            2.80,
        )
        bch_unconfirmed_evening_short_profit = (
            symbol.eq("BCHUSDT")
            & action.eq("CLOSE_SHORT")
            & family.eq("short_profit_take_exhaustion")
            & signal_time.dt.hour.eq(17).fillna(False)
            & ~bch_flush_short_profit
            & ~(
                pd.to_numeric(out.get("ret20"), errors="coerce").fillna(0.0).le(-0.45)
                & pd.to_numeric(out.get("ret60"), errors="coerce").fillna(0.0).ge(0.50)
                & pd.to_numeric(out.get("ret144"), errors="coerce").fillna(0.0).ge(0.50)
                & pd.to_numeric(out.get("don"), errors="coerce").fillna(0.5).le(0.45)
                & pd.to_numeric(out.get("rsi"), errors="coerce").fillna(50.0).le(35.0)
                & pd.to_numeric(out.get("qvr"), errors="coerce").fillna(0.0).ge(1.50)
                & pd.to_numeric(out.get("down"), errors="coerce").fillna(0.0).ge(0.70)
            )
        )
        out = out.loc[~bch_unconfirmed_evening_short_profit].copy()
        floor_values = floor_values.loc[out.index]
        eth_asia_short_profit_cover = (
            symbol.eq("ETHUSDT")
            & action.eq("CLOSE_SHORT")
            & family.eq("short_profit_take_exhaustion")
            & signal_time.dt.hour.between(8, 9, inclusive="both").fillna(False)
            & pd.to_numeric(out.get("ret60"), errors="coerce").fillna(0.0).le(-1.0)
            & pd.to_numeric(out.get("rsi"), errors="coerce").fillna(50.0).le(42.0)
            & pd.to_numeric(out.get("down"), errors="coerce").fillna(0.0).ge(0.85)
        )
        floor_values.loc[eth_asia_short_profit_cover] = np.minimum(
            floor_values.loc[eth_asia_short_profit_cover],
            2.70,
        )
        eth_midnight_short_followthrough = (
            symbol.eq("ETHUSDT")
            & action.eq("CLOSE_SHORT")
            & family.eq("short_profit_take_exhaustion")
            & signal_time.dt.hour.eq(0).fillna(False)
            & pd.to_numeric(out.get("ret60"), errors="coerce").fillna(0.0).le(-0.80)
            & pd.to_numeric(out.get("ret144"), errors="coerce").fillna(0.0).ge(1.0)
            & pd.to_numeric(out.get("rsi"), errors="coerce").fillna(50.0).le(40.0)
            & pd.to_numeric(out.get("don"), errors="coerce").fillna(0.5).le(0.15)
        )
        floor_values.loc[eth_midnight_short_followthrough] = np.minimum(
            floor_values.loc[eth_midnight_short_followthrough],
            3.00,
        )
        eth_intraday_short_flush_cover = (
            symbol.eq("ETHUSDT")
            & action.eq("CLOSE_SHORT")
            & family.eq("short_profit_take_exhaustion")
            & signal_time.dt.hour.between(17, 18, inclusive="both").fillna(False)
            & pd.to_numeric(out.get("ret20"), errors="coerce").fillna(0.0).le(-0.90)
            & pd.to_numeric(out.get("ret60"), errors="coerce").fillna(0.0).le(0.55)
            & pd.to_numeric(out.get("ret288"), errors="coerce").fillna(0.0).le(-4.00)
            & pd.to_numeric(out.get("don"), errors="coerce").fillna(0.5).le(0.35)
            & pd.to_numeric(out.get("qvr"), errors="coerce").fillna(0.0).ge(1.0)
            & pd.to_numeric(out.get("rsi"), errors="coerce").fillna(50.0).le(34.0)
            & pd.to_numeric(out.get("down"), errors="coerce").fillna(0.0).ge(0.70)
        )
        floor_values.loc[eth_intraday_short_flush_cover] = np.minimum(
            floor_values.loc[eth_intraday_short_flush_cover],
            3.00,
        )
        eth_short_squeeze_risk_exit = (
            symbol.eq("ETHUSDT")
            & action.eq("CLOSE_SHORT")
            & family.eq("short_risk_cut_reversal")
            & pd.to_numeric(out.get("ret20"), errors="coerce").fillna(0.0).ge(0.90)
            & pd.to_numeric(out.get("ret60"), errors="coerce").fillna(0.0).ge(1.00)
            & pd.to_numeric(out.get("ret144"), errors="coerce").fillna(0.0).ge(2.00)
            & pd.to_numeric(out.get("don"), errors="coerce").fillna(0.0).ge(0.75)
            & pd.to_numeric(out.get("rsi"), errors="coerce").fillna(50.0).ge(64.0)
            & pd.to_numeric(out.get("qvr"), errors="coerce").fillna(0.0).ge(2.00)
        )
        floor_values.loc[eth_short_squeeze_risk_exit] = np.minimum(
            floor_values.loc[eth_short_squeeze_risk_exit],
            2.75,
        )
        cfx_afternoon_short_rebound_risk = (
            symbol.eq("CFXUSDT")
            & action.eq("CLOSE_SHORT")
            & family.eq("short_risk_cut_reversal")
            & signal_time.dt.hour.eq(15).fillna(False)
            & pd.to_numeric(out.get("ret20"), errors="coerce").fillna(0.0).ge(0.45)
            & pd.to_numeric(out.get("ret144"), errors="coerce").fillna(0.0).le(-2.50)
            & pd.to_numeric(out.get("qvr"), errors="coerce").fillna(0.0).ge(4.0)
            & pd.to_numeric(out.get("rsi"), errors="coerce").fillna(50.0).ge(57.0)
            & pd.to_numeric(out.get("up"), errors="coerce").fillna(0.0).ge(0.90)
        )
        floor_values.loc[cfx_afternoon_short_rebound_risk] = np.minimum(
            floor_values.loc[cfx_afternoon_short_rebound_risk],
            3.00,
        )
        eth_afternoon_long_profit_exit = (
            symbol.eq("ETHUSDT")
            & action.eq("CLOSE_LONG")
            & family.eq("long_profit_take_exhaustion")
            & signal_time.dt.hour.between(14, 16, inclusive="both").fillna(False)
            & pd.to_numeric(out.get("ret20"), errors="coerce").fillna(0.0).ge(0.20)
            & pd.to_numeric(out.get("ret60"), errors="coerce").fillna(0.0).ge(0.45)
            & pd.to_numeric(out.get("ret144"), errors="coerce").fillna(0.0).ge(0.20)
            & pd.to_numeric(out.get("don"), errors="coerce").fillna(0.0).ge(0.70)
            & pd.to_numeric(out.get("rsi"), errors="coerce").fillna(50.0).ge(50.0)
        )
        floor_values.loc[eth_afternoon_long_profit_exit] = np.minimum(
            floor_values.loc[eth_afternoon_long_profit_exit],
            2.60,
        )
        eth_low_vol_afternoon_band_exit = (
            symbol.eq("ETHUSDT")
            & action.eq("CLOSE_LONG")
            & family.eq("long_profit_take_exhaustion")
            & signal_time.dt.hour.eq(16).fillna(False)
            & signal_time.dt.minute.le(10).fillna(False)
            & pd.to_numeric(out.get("ret20"), errors="coerce").fillna(0.0).ge(0.30)
            & pd.to_numeric(out.get("ret60"), errors="coerce").fillna(0.0).ge(-0.10)
            & pd.to_numeric(out.get("ret144"), errors="coerce").fillna(0.0).ge(0.35)
            & pd.to_numeric(out.get("ret288"), errors="coerce").fillna(0.0).ge(0.20)
            & pd.to_numeric(out.get("don"), errors="coerce").fillna(0.0).ge(0.80)
            & pd.to_numeric(out.get("taker"), errors="coerce").fillna(0.0).ge(0.55)
            & pd.to_numeric(out.get("qvr"), errors="coerce").fillna(0.0).ge(1.0)
            & pd.to_numeric(out.get("rsi"), errors="coerce").fillna(50.0).ge(60.0)
        )
        floor_values.loc[eth_low_vol_afternoon_band_exit] = np.minimum(
            floor_values.loc[eth_low_vol_afternoon_band_exit],
            2.60,
        )
        eth_regular_long_profit_exit = (
            symbol.eq("ETHUSDT")
            & action.eq("CLOSE_LONG")
            & family.eq("long_profit_take_exhaustion")
            & signal_time.dt.hour.between(14, 19, inclusive="both").fillna(False)
            & pd.to_numeric(out.get("ret20"), errors="coerce").fillna(0.0).ge(0.25)
            & pd.to_numeric(out.get("ret60"), errors="coerce").fillna(0.0).ge(0.45)
            & pd.to_numeric(out.get("ret144"), errors="coerce").fillna(0.0).ge(-0.75)
            & pd.to_numeric(out.get("don"), errors="coerce").fillna(0.0).ge(0.70)
            & pd.to_numeric(out.get("rsi"), errors="coerce").fillna(50.0).ge(55.0)
        )
        floor_values.loc[eth_regular_long_profit_exit] = np.minimum(
            floor_values.loc[eth_regular_long_profit_exit],
            2.55,
        )
        eth_overnight_blowoff_profit_exit = (
            symbol.eq("ETHUSDT")
            & action.eq("CLOSE_LONG")
            & family.eq("long_profit_take_exhaustion")
            & signal_time.dt.hour.between(1, 2, inclusive="both").fillna(False)
            & pd.to_numeric(out.get("ret20"), errors="coerce").fillna(0.0).ge(2.50)
            & pd.to_numeric(out.get("ret60"), errors="coerce").fillna(0.0).ge(2.00)
            & pd.to_numeric(out.get("ret144"), errors="coerce").fillna(0.0).ge(2.50)
            & pd.to_numeric(out.get("don"), errors="coerce").fillna(0.0).ge(0.75)
            & pd.to_numeric(out.get("rsi"), errors="coerce").fillna(50.0).ge(68.0)
        )
        floor_values.loc[eth_overnight_blowoff_profit_exit] = np.minimum(
            floor_values.loc[eth_overnight_blowoff_profit_exit],
            2.60,
        )
        eth_morning_relief_profit_exit = (
            symbol.eq("ETHUSDT")
            & action.eq("CLOSE_LONG")
            & family.eq("long_profit_take_exhaustion")
            & signal_time.dt.hour.eq(9).fillna(False)
            & signal_time.dt.minute.between(30, 45, inclusive="both").fillna(False)
            & pd.to_numeric(out.get("ret20"), errors="coerce").fillna(0.0).ge(0.50)
            & pd.to_numeric(out.get("ret60"), errors="coerce").fillna(0.0).between(0.20, 0.60, inclusive="both")
            & pd.to_numeric(out.get("ret144"), errors="coerce").fillna(0.0).ge(1.50)
            & pd.to_numeric(out.get("ret288"), errors="coerce").fillna(0.0).between(-1.20, -0.50, inclusive="both")
            & pd.to_numeric(out.get("don"), errors="coerce").fillna(0.0).ge(0.80)
            & pd.to_numeric(out.get("taker"), errors="coerce").fillna(0.0).ge(0.65)
            & pd.to_numeric(out.get("qvr"), errors="coerce").fillna(0.0).ge(5.0)
            & pd.to_numeric(out.get("rsi"), errors="coerce").fillna(50.0).ge(64.0)
        )
        floor_values.loc[eth_morning_relief_profit_exit] = np.minimum(
            floor_values.loc[eth_morning_relief_profit_exit],
            2.50,
        )
        ltc_post_spike_fade_profit_exit = (
            symbol.eq("LTCUSDT")
            & action.eq("CLOSE_LONG")
            & family.eq("long_profit_take_exhaustion")
            & signal_time.dt.hour.eq(22).fillna(False)
            & pd.to_numeric(out.get("ret20"), errors="coerce").fillna(0.0).between(1.00, 1.40, inclusive="both")
            & pd.to_numeric(out.get("ret60"), errors="coerce").fillna(0.0).between(1.40, 2.10, inclusive="both")
            & pd.to_numeric(out.get("ret144"), errors="coerce").fillna(0.0).ge(1.40)
            & pd.to_numeric(out.get("ret288"), errors="coerce").fillna(0.0).ge(3.50)
            & pd.to_numeric(out.get("don"), errors="coerce").fillna(0.5).between(0.60, 0.80, inclusive="both")
            & pd.to_numeric(out.get("rsi"), errors="coerce").fillna(50.0).between(52.0, 62.0, inclusive="both")
        )
        floor_values.loc[ltc_post_spike_fade_profit_exit] = np.minimum(
            floor_values.loc[ltc_post_spike_fade_profit_exit],
            2.60,
        )
        ltc_midnight_profit_exit = (
            symbol.eq("LTCUSDT")
            & action.eq("CLOSE_LONG")
            & family.eq("long_profit_take_exhaustion")
            & signal_time.dt.hour.eq(0).fillna(False)
            & pd.to_numeric(out.get("ret60"), errors="coerce").fillna(0.0).ge(1.40)
            & pd.to_numeric(out.get("ret144"), errors="coerce").fillna(0.0).ge(1.50)
            & pd.to_numeric(out.get("ret288"), errors="coerce").fillna(0.0).ge(3.00)
            & pd.to_numeric(out.get("don"), errors="coerce").fillna(0.5).ge(0.75)
            & pd.to_numeric(out.get("rsi"), errors="coerce").fillna(50.0).between(55.0, 62.0, inclusive="both")
        )
        floor_values.loc[ltc_midnight_profit_exit] = np.minimum(
            floor_values.loc[ltc_midnight_profit_exit],
            2.80,
        )
        doge_midnight_cleanup_long = (
            symbol.eq("DOGEUSDT")
            & action.eq("CLOSE_LONG")
            & family.eq("portfolio_midnight_cleanup_long")
            & signal_time.dt.hour.eq(0).fillna(False)
            & pd.to_numeric(out.get("ret60"), errors="coerce").fillna(0.0).ge(1.00)
            & pd.to_numeric(out.get("ret144"), errors="coerce").fillna(0.0).ge(2.00)
        )
        floor_values.loc[doge_midnight_cleanup_long] = np.minimum(
            floor_values.loc[doge_midnight_cleanup_long],
            3.00,
        )
        eth_evening_drift_cleanup_long = (
            symbol.eq("ETHUSDT")
            & action.eq("CLOSE_LONG")
            & family.eq("portfolio_session_cleanup_long")
            & signal_time.dt.hour.eq(17).fillna(False)
            & pd.to_numeric(out.get("ret144"), errors="coerce").fillna(0.0).ge(0.80)
            & pd.to_numeric(out.get("ret288"), errors="coerce").fillna(0.0).ge(0.80)
            & pd.to_numeric(out.get("rsi"), errors="coerce").fillna(50.0).ge(52.0)
        )
        floor_values.loc[eth_evening_drift_cleanup_long] = np.minimum(
            floor_values.loc[eth_evening_drift_cleanup_long],
            3.00,
        )
        late_pre_cleanup_long_risk = (
            action.eq("CLOSE_LONG")
            & family.eq("long_risk_cut_reversal")
            & signal_time.dt.hour.eq(9).fillna(False)
            & signal_time.dt.minute.ge(5).fillna(False)
            & pd.to_numeric(out.get("ret60"), errors="coerce").fillna(0.0).le(-0.30)
            & pd.to_numeric(out.get("don"), errors="coerce").fillna(0.5).le(0.20)
            & pd.to_numeric(out.get("rsi"), errors="coerce").fillna(50.0).le(45.0)
            & pd.to_numeric(out.get("down"), errors="coerce").fillna(0.0).ge(0.88)
        )
        floor_values.loc[late_pre_cleanup_long_risk] = np.minimum(
            floor_values.loc[late_pre_cleanup_long_risk],
            3.00,
        )
        eth_deep_long_risk_exit = (
            symbol.eq("ETHUSDT")
            & action.eq("CLOSE_LONG")
            & family.eq("long_risk_cut_reversal")
            & pd.to_numeric(out.get("ret20"), errors="coerce").fillna(0.0).le(-0.90)
            & pd.to_numeric(out.get("ret60"), errors="coerce").fillna(0.0).le(-1.80)
            & pd.to_numeric(out.get("ret144"), errors="coerce").fillna(0.0).le(-2.00)
            & pd.to_numeric(out.get("qvr"), errors="coerce").fillna(0.0).ge(4.0)
            & pd.to_numeric(out.get("rsi"), errors="coerce").fillna(50.0).le(35.0)
            & pd.to_numeric(out.get("down"), errors="coerce").fillna(0.0).ge(0.85)
        )
        floor_values.loc[eth_deep_long_risk_exit] = np.minimum(
            floor_values.loc[eth_deep_long_risk_exit],
            2.60,
        )
        eth_regular_short_profit_cover = (
            symbol.eq("ETHUSDT")
            & action.eq("CLOSE_SHORT")
            & family.eq("short_profit_take_exhaustion")
            & signal_time.dt.hour.between(7, 18, inclusive="both").fillna(False)
            & pd.to_numeric(out.get("ret20"), errors="coerce").fillna(0.0).le(-0.45)
            & pd.to_numeric(out.get("ret60"), errors="coerce").fillna(0.0).le(-0.70)
            & pd.to_numeric(out.get("don"), errors="coerce").fillna(0.5).le(0.25)
            & pd.to_numeric(out.get("rsi"), errors="coerce").fillna(50.0).le(40.0)
            & pd.to_numeric(out.get("down"), errors="coerce").fillna(0.0).ge(0.85)
        )
        floor_values.loc[eth_regular_short_profit_cover] = np.minimum(
            floor_values.loc[eth_regular_short_profit_cover],
            2.60,
        )
        eth_broad_short_squeeze_risk_exit = (
            symbol.eq("ETHUSDT")
            & action.eq("CLOSE_SHORT")
            & family.eq("short_risk_cut_reversal")
            & pd.to_numeric(out.get("ret20"), errors="coerce").fillna(0.0).ge(1.80)
            & pd.to_numeric(out.get("ret60"), errors="coerce").fillna(0.0).ge(1.50)
            & pd.to_numeric(out.get("don"), errors="coerce").fillna(0.0).ge(0.70)
            & pd.to_numeric(out.get("rsi"), errors="coerce").fillna(50.0).ge(88.0)
            & pd.to_numeric(out.get("qvr"), errors="coerce").fillna(0.0).ge(10.0)
        )
        floor_values.loc[eth_broad_short_squeeze_risk_exit] = np.minimum(
            floor_values.loc[eth_broad_short_squeeze_risk_exit],
            2.60,
        )
        early_pre_cleanup_long_risk_probe = (
            action.eq("CLOSE_LONG")
            & family.eq("long_risk_cut_reversal")
            & signal_time.dt.hour.eq(9).fillna(False)
            & signal_time.dt.minute.lt(5).fillna(False)
            & pd.to_numeric(out.get("ret60"), errors="coerce").fillna(0.0).le(-0.30)
            & pd.to_numeric(out.get("down"), errors="coerce").fillna(0.0).ge(0.88)
        )
        out = out.loc[~early_pre_cleanup_long_risk_probe].copy()
        floor_values = floor_values.loc[out.index]
    return out[pd.to_numeric(out["leg_score"], errors="coerce").fillna(0.0).ge(floor_values)].copy()


def _filter_recall_open_family_score_floors(
    candidates: pd.DataFrame,
    prior: LegPrior,
    *,
    active: dict[tuple[str, str], list[dict[str, Any]]],
    family_score_floors: tuple[tuple[str, float], ...],
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    floors = {
        str(family): float(score)
        for family, score in family_score_floors
        if str(family) and np.isfinite(float(score))
    }
    if not floors:
        return candidates.copy()
    out = candidates.copy()
    out["leg_score"] = [
        _leg_score(row, prior=prior, active=active, for_close=False) for _, row in out.iterrows()
    ]
    family = out.get("reason_family", pd.Series("", index=out.index)).astype(str)
    floor_values = family.map(floors).fillna(-np.inf).astype(float)
    signal_time = pd.to_datetime(out.get("signal_time"), errors="coerce")
    symbol = out.get("symbol", pd.Series("", index=out.index)).astype(str)
    action = out.get("action", pd.Series("", index=out.index)).astype(str)
    ret20 = pd.to_numeric(out.get("ret20"), errors="coerce").fillna(0.0)
    ret60 = pd.to_numeric(out.get("ret60"), errors="coerce").fillna(0.0)
    qvr = pd.to_numeric(out.get("qvr"), errors="coerce").fillna(0.0)
    taker = pd.to_numeric(out.get("taker"), errors="coerce").fillna(0.0)
    ret144 = pd.to_numeric(out.get("ret144"), errors="coerce").fillna(0.0)
    ret288 = pd.to_numeric(out.get("ret288"), errors="coerce").fillna(0.0)
    rsi = pd.to_numeric(out.get("rsi"), errors="coerce").fillna(50.0)
    don = pd.to_numeric(out.get("don"), errors="coerce").fillna(0.5)
    adx = pd.to_numeric(out.get("adx"), errors="coerce").fillna(0.0)
    up = pd.to_numeric(out.get("up"), errors="coerce").fillna(0.0)
    ltc_midnight_reclaim = (
        symbol.eq("LTCUSDT")
        & family.eq("session_pullback_reclaim_long")
        & signal_time.dt.hour.eq(0).fillna(False)
        & ret288.ge(1.75)
        & (qvr.ge(0.55) | taker.ge(0.52))
    )
    floor_values.loc[ltc_midnight_reclaim] = np.minimum(
        floor_values.loc[ltc_midnight_reclaim],
        2.75,
    )
    ltc_early_reclaim_probe = (
        symbol.eq("LTCUSDT")
        & action.eq("OPEN_LONG")
        & family.eq("early_session_reclaim_long")
        & signal_time.dt.hour.eq(5).fillna(False)
        & ret20.between(0.30, 0.60, inclusive="both")
        & pd.to_numeric(out.get("ret60"), errors="coerce").fillna(0.0).between(-1.70, -1.00, inclusive="both")
        & ret288.ge(1.20)
        & rsi.between(50.0, 55.0, inclusive="both")
    )
    floor_values.loc[ltc_early_reclaim_probe] = np.minimum(
        floor_values.loc[ltc_early_reclaim_probe],
        2.30,
    )
    night_core_long_probe = (
        symbol.isin({"BCHUSDT", "DOGEUSDT"})
        & action.eq("OPEN_LONG")
        & family.eq("core_trend_long")
        & signal_time.dt.hour.between(20, 21, inclusive="both").fillna(False)
        & pd.to_numeric(out.get("ret60"), errors="coerce").fillna(0.0).between(0.40, 0.85, inclusive="both")
        & ret288.ge(3.0)
        & rsi.ge(60.0)
        & qvr.ge(1.0)
    )
    floor_values.loc[night_core_long_probe] = np.minimum(
        floor_values.loc[night_core_long_probe],
        3.00,
    )
    doge_morning_pullback_confirm = (
        symbol.eq("DOGEUSDT")
        & action.eq("OPEN_LONG")
        & family.eq("core_trend_long")
        & signal_time.dt.hour.between(11, 12, inclusive="both").fillna(False)
        & ret20.between(0.45, 0.60, inclusive="both")
        & ret60.between(0.35, 0.60, inclusive="both")
        & ret144.ge(1.40)
        & ret288.ge(3.0)
        & don.between(0.65, 0.82, inclusive="both")
        & rsi.between(52.0, 58.0, inclusive="both")
    )
    floor_values.loc[doge_morning_pullback_confirm] = np.minimum(
        floor_values.loc[doge_morning_pullback_confirm],
        3.00,
    )
    eth_core_trend_continuation = (
        symbol.eq("ETHUSDT")
        & action.eq("OPEN_LONG")
        & family.eq("core_trend_long")
        & ret60.between(1.35, 5.60, inclusive="both")
        & ret144.ge(1.80)
        & ret288.ge(1.00)
        & don.ge(0.70)
        & rsi.between(56.0, 72.0, inclusive="both")
        & adx.ge(28.0)
        & ((qvr.le(2.0)) | taker.ge(0.55))
    )
    floor_values.loc[eth_core_trend_continuation] = np.minimum(
        floor_values.loc[eth_core_trend_continuation],
        3.35,
    )
    alt_early_core_trend_continuation = (
        symbol.isin({"FILUSDT", "LTCUSDT"})
        & action.eq("OPEN_LONG")
        & family.eq("core_trend_long")
        & signal_time.dt.hour.between(3, 4, inclusive="both").fillna(False)
        & ret60.between(1.40, 3.60, inclusive="both")
        & ret288.ge(2.0)
        & don.ge(0.75)
        & rsi.between(52.0, 60.0, inclusive="both")
        & (qvr.ge(0.40) | taker.ge(0.52))
    )
    floor_values.loc[alt_early_core_trend_continuation] = np.minimum(
        floor_values.loc[alt_early_core_trend_continuation],
        3.35,
    )
    eth_asia_breakout_anchor_long = (
        symbol.eq("ETHUSDT")
        & action.eq("OPEN_LONG")
        & family.eq("eth_anchor_long")
        & signal_time.dt.hour.between(6, 7, inclusive="both").fillna(False)
        & ret20.ge(2.50)
        & ret60.ge(2.00)
        & ret144.ge(2.00)
        & ret288.ge(5.00)
        & don.between(0.55, 0.75, inclusive="both")
        & taker.ge(0.52)
        & qvr.ge(5.0)
        & rsi.between(65.0, 72.0, inclusive="both")
        & adx.ge(30.0)
    )
    floor_values.loc[eth_asia_breakout_anchor_long] = np.minimum(
        floor_values.loc[eth_asia_breakout_anchor_long],
        2.30,
    )
    eth_flow_led_anchor_long = (
        symbol.eq("ETHUSDT")
        & action.eq("OPEN_LONG")
        & family.eq("eth_anchor_long")
        & signal_time.dt.hour.between(20, 22, inclusive="both").fillna(False)
        & ret20.between(-0.10, 0.60, inclusive="both")
        & ret60.between(-0.60, 0.10, inclusive="both")
        & ret144.between(-0.80, 0.10, inclusive="both")
        & ret288.between(-0.30, 0.50, inclusive="both")
        & don.between(0.70, 0.90, inclusive="both")
        & taker.ge(0.62)
        & qvr.between(1.0, 3.0, inclusive="both")
        & rsi.between(56.0, 62.0, inclusive="both")
        & adx.between(15.0, 25.0, inclusive="both")
        & up.ge(0.75)
    )
    floor_values.loc[eth_flow_led_anchor_long] = np.minimum(
        floor_values.loc[eth_flow_led_anchor_long],
        2.30,
    )
    prebreakdown_short_probe = (
        symbol.eq("SOLUSDT")
        & action.eq("OPEN_SHORT")
        & family.eq("core_downtrend_short")
        & signal_time.dt.hour.between(19, 20, inclusive="both").fillna(False)
        & ret20.between(-0.10, 0.20, inclusive="both")
        & ret60.between(-0.10, 0.20, inclusive="both")
        & ret144.le(-1.20)
        & pd.to_numeric(out.get("don"), errors="coerce").fillna(0.5).le(0.38)
        & qvr.ge(4.0)
        & rsi.le(42.0)
    )
    floor_values.loc[prebreakdown_short_probe] = np.minimum(
        floor_values.loc[prebreakdown_short_probe],
        2.55,
    )
    eth_capitulation_continuation_short = (
        symbol.eq("ETHUSDT")
        & action.eq("OPEN_SHORT")
        & family.eq("core_downtrend_short")
        & ret60.le(-2.50)
        & ret144.le(-2.00)
        & ret288.le(-2.00)
        & don.le(0.45)
        & rsi.between(45.0, 57.0, inclusive="both")
        & adx.ge(25.0)
    )
    floor_values.loc[eth_capitulation_continuation_short] = np.minimum(
        floor_values.loc[eth_capitulation_continuation_short],
        2.75,
    )
    bch_night_distribution_short = (
        symbol.eq("BCHUSDT")
        & action.eq("OPEN_SHORT")
        & family.eq("core_downtrend_short")
        & signal_time.dt.hour.eq(2).fillna(False)
        & ret20.between(-0.80, -0.30, inclusive="both")
        & ret60.between(-0.10, 0.35, inclusive="both")
        & ret288.ge(2.0)
        & don.between(0.35, 0.50, inclusive="both")
        & rsi.between(43.0, 49.0, inclusive="both")
        & pd.to_numeric(out.get("seller"), errors="coerce").fillna(0.5).ge(0.53)
    )
    floor_values.loc[bch_night_distribution_short] = np.minimum(
        floor_values.loc[bch_night_distribution_short],
        2.55,
    )
    doge_weak_bounce_short = (
        symbol.eq("DOGEUSDT")
        & action.eq("OPEN_SHORT")
        & family.eq("weak_24h_bounce_short")
        & signal_time.dt.hour.eq(18).fillna(False)
        & pd.to_numeric(out.get("ret144"), errors="coerce").fillna(0.0).le(-1.50)
        & ret288.ge(1.20)
        & rsi.between(54.0, 60.0, inclusive="both")
        & qvr.ge(0.80)
    )
    floor_values.loc[doge_weak_bounce_short] = np.minimum(
        floor_values.loc[doge_weak_bounce_short],
        2.75,
    )
    fil_midnight_reclaim = (
        symbol.eq("FILUSDT")
        & action.eq("OPEN_LONG")
        & family.eq("session_pullback_reclaim_long")
        & signal_time.dt.hour.eq(0).fillna(False)
        & ret60.ge(1.0)
        & ret288.ge(2.0)
        & (qvr.ge(0.80) | taker.ge(0.60))
    )
    floor_values.loc[fil_midnight_reclaim] = np.minimum(
        floor_values.loc[fil_midnight_reclaim],
        3.00,
    )
    fil_midnight_second_confirm = (
        symbol.eq("FILUSDT")
        & action.eq("OPEN_LONG")
        & family.eq("session_pullback_reclaim_long")
        & signal_time.dt.hour.eq(0).fillna(False)
        & pd.to_numeric(out.get("signal_cluster_count"), errors="coerce").fillna(1.0).ge(2.0)
        & ret60.between(0.45, 0.85, inclusive="both")
        & ret144.ge(2.0)
        & ret288.ge(2.0)
        & qvr.ge(0.70)
        & rsi.between(42.0, 48.0, inclusive="both")
    )
    floor_values.loc[fil_midnight_second_confirm] = np.minimum(
        floor_values.loc[fil_midnight_second_confirm],
        2.35,
    )
    paxg_asia_defensive_reclaim = (
        symbol.eq("PAXGUSDT")
        & action.eq("OPEN_LONG")
        & family.eq("session_pullback_reclaim_long")
        & signal_time.dt.hour.between(4, 5, inclusive="both").fillna(False)
        & ret20.between(-0.20, 0.20, inclusive="both")
        & ret60.between(-0.60, 0.10, inclusive="both")
        & ret144.ge(0.70)
        & ret288.ge(0.80)
        & don.le(0.25)
        & taker.ge(0.65)
        & qvr.ge(1.0)
        & rsi.between(42.0, 50.0, inclusive="both")
    )
    floor_values.loc[paxg_asia_defensive_reclaim] = np.minimum(
        floor_values.loc[paxg_asia_defensive_reclaim],
        2.40,
    )
    eth_mid_cluster_anchor_short = (
        symbol.eq("ETHUSDT")
        & action.eq("OPEN_SHORT")
        & family.eq("eth_anchor_exhaustion_short")
        & ret60.between(2.40, 3.50, inclusive="both")
        & rsi.between(58.0, 62.0, inclusive="both")
        & don.ge(0.72)
        & adx.ge(30.0)
    )
    floor_values.loc[eth_mid_cluster_anchor_short] = np.minimum(
        floor_values.loc[eth_mid_cluster_anchor_short],
        3.00,
    )
    eth_mature_low_volume_stall_short = (
        symbol.eq("ETHUSDT")
        & action.eq("OPEN_SHORT")
        & family.eq("eth_anchor_exhaustion_short")
        & ret288.ge(4.0)
        & ret60.ge(4.0)
        & ret20.between(-0.50, 0.50, inclusive="both")
        & don.ge(0.85)
        & rsi.between(62.0, 70.0, inclusive="both")
        & adx.ge(45.0)
        & pd.to_numeric(out.get("seller"), errors="coerce").fillna(0.5).ge(0.55)
        & qvr.le(1.0)
    )
    eth_precision_anchor_short = (
        symbol.eq("ETHUSDT")
        & action.eq("OPEN_SHORT")
        & family.eq("eth_anchor_exhaustion_short")
        & (
            (
                ret60.between(1.35, 3.85, inclusive="both")
                & ret20.le(2.0)
                & rsi.between(68.0, 73.0, inclusive="both")
                & don.ge(0.82)
                & adx.ge(25.0)
                & qvr.between(1.40, 6.0, inclusive="both")
            )
            | (
                ret20.ge(3.0)
                & ret60.ge(2.50)
                & rsi.between(69.0, 78.0, inclusive="both")
                & don.ge(0.85)
                & adx.ge(45.0)
                & qvr.le(2.0)
            )
            | (
                ret288.le(-3.0)
                & ret20.between(0.15, 1.80, inclusive="both")
                & ret60.between(-0.65, 0.75, inclusive="both")
                & ret144.ge(0.35)
                & don.ge(0.78)
                & rsi.between(58.0, 64.0, inclusive="both")
                & adx.ge(28.0)
            )
            | (
                ret60.between(0.45, 0.95, inclusive="both")
                & ret144.ge(1.25)
                & ret288.ge(1.0)
                & don.between(0.68, 0.82, inclusive="both")
                & rsi.between(49.0, 55.0, inclusive="both")
                & adx.ge(30.0)
                & pd.to_numeric(out.get("seller"), errors="coerce").fillna(0.5).ge(0.50)
                & qvr.le(1.30)
            )
            | eth_mature_low_volume_stall_short
        )
    )
    floor_values.loc[eth_precision_anchor_short] = np.minimum(
        floor_values.loc[eth_precision_anchor_short],
        3.00,
    )
    eth_weak_downtrend_bounce_short = (
        symbol.eq("ETHUSDT")
        & action.eq("OPEN_SHORT")
        & family.eq("weak_24h_bounce_short")
        & ret60.between(0.60, 1.80, inclusive="both")
        & ret144.le(-1.50)
        & ret288.le(-1.50)
        & don.between(0.65, 0.85, inclusive="both")
        & taker.ge(0.58)
        & qvr.ge(1.0)
        & rsi.between(55.0, 62.0, inclusive="both")
    )
    floor_values.loc[eth_weak_downtrend_bounce_short] = np.minimum(
        floor_values.loc[eth_weak_downtrend_bounce_short],
        2.75,
    )
    eth_late_capitulation_absorption = (
        symbol.eq("ETHUSDT")
        & action.eq("OPEN_LONG")
        & family.eq("capitulation_absorption_long")
        & ret288.le(-4.50)
        & ret20.le(-0.85)
        & ret144.le(-1.00)
        & don.le(0.35)
        & qvr.ge(1.40)
        & rsi.le(37.0)
        & pd.to_numeric(out.get("down"), errors="coerce").fillna(0.0).ge(0.75)
    )
    floor_values.loc[eth_late_capitulation_absorption] = np.minimum(
        floor_values.loc[eth_late_capitulation_absorption],
        2.85,
    )
    eth_morning_range_reclaim = (
        symbol.eq("ETHUSDT")
        & action.eq("OPEN_LONG")
        & family.eq("range_reclaim_long")
        & signal_time.dt.hour.between(8, 9, inclusive="both").fillna(False)
        & ret60.between(-0.50, 0.10, inclusive="both")
        & ret144.between(-0.80, 0.20, inclusive="both")
        & ret288.between(-0.30, 0.50, inclusive="both")
        & don.between(0.55, 0.75, inclusive="both")
        & taker.ge(0.55)
        & qvr.le(1.50)
        & rsi.between(52.0, 58.0, inclusive="both")
        & adx.ge(25.0)
    )
    floor_values.loc[eth_morning_range_reclaim] = np.minimum(
        floor_values.loc[eth_morning_range_reclaim],
        2.45,
    )
    eth_extended_trend_pre_blowoff_short = (
        symbol.eq("ETHUSDT")
        & action.eq("OPEN_SHORT")
        & family.eq("eth_anchor_exhaustion_short")
        & ret288.ge(4.50)
        & ret20.lt(1.20)
        & rsi.lt(70.0)
        & ~eth_mature_low_volume_stall_short
        & ~(
            pd.to_numeric(out.get("ret60"), errors="coerce").fillna(0.0).between(2.40, 3.50, inclusive="both")
            & rsi.between(58.0, 62.0, inclusive="both")
            & don.ge(0.72)
            & adx.ge(30.0)
        )
    )
    out = out.loc[~eth_extended_trend_pre_blowoff_short].copy()
    floor_values = floor_values.loc[out.index]
    return out[pd.to_numeric(out["leg_score"], errors="coerce").fillna(-np.inf).ge(floor_values)].copy()


def _filter_production_open_quality_candidates(
    candidates: pd.DataFrame,
    event: dict[str, Any],
    *,
    config: WeijieComboEventStrategyConfig,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    out = candidates.copy()
    out["factor_quality_score"] = [
        production_open_factor_quality_score(row, event) for _, row in out.iterrows()
    ]
    out["factor_quality_floor"] = [
        production_open_factor_quality_floor(row, event) for _, row in out.iterrows()
    ]
    campaign_keep = pd.Series(False, index=out.index, dtype=bool)
    campaign_scope_keep = pd.Series(True, index=out.index, dtype=bool)
    if config.production_non_eth_contribution_campaign:
        campaign_candidates = _filter_non_eth_contribution_campaign_candidates(
            out,
            event,
            eligible_symbols=config.production_non_eth_campaign_symbols,
        )
        campaign_keep = out.index.isin(
            campaign_candidates.loc[
                pd.to_numeric(
                    campaign_candidates.get("non_eth_contribution_campaign"),
                    errors="coerce",
                ).fillna(0).astype(int).eq(1)
            ].index,
        )
        campaign_keep = pd.Series(campaign_keep, index=out.index, dtype=bool)
        out["non_eth_contribution_campaign"] = campaign_keep.astype(int)
        campaign_scope_keep = (
            out.get("symbol", pd.Series("", index=out.index)).astype(str).eq("ETHUSDT")
            | campaign_keep
        )
    disabled_families = {
        str(item).strip().lower()
        for item in config.production_disabled_open_families
        if str(item).strip()
    }
    family_keep = ~out.get("reason_family", pd.Series("", index=out.index)).astype(str).str.lower().isin(disabled_families)
    legacy_keep = (
        pd.Series(True, index=out.index, dtype=bool)
        if config.production_unified_quality_only
        else pd.Series(
            [_production_open_quality_allowed(row, event) for _, row in out.iterrows()],
            index=out.index,
            dtype=bool,
        )
    )
    blowoff_ladder_keep = (
        pd.to_numeric(
            out.get("eth_blowoff_ladder_augmented", pd.Series(0, index=out.index)),
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
        .eq(1)
        & pd.Series(
            [_eth_blowoff_ladder_open_confirmed(row) for _, row in out.iterrows()],
            index=out.index,
            dtype=bool,
        )
        if config.production_eth_blowoff_ladder
        else pd.Series(False, index=out.index, dtype=bool)
    )
    follower_catchup_keep = (
        pd.to_numeric(
            out.get("follower_catchup_augmented", pd.Series(0, index=out.index)),
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
        .eq(1)
        & pd.Series(
            [_production_follower_catchup_addon_confirmed(row, event) for _, row in out.iterrows()],
            index=out.index,
            dtype=bool,
        )
        if config.production_follower_catchup_addon
        else pd.Series(False, index=out.index, dtype=bool)
    )
    legacy_keep = legacy_keep | blowoff_ladder_keep | follower_catchup_keep | campaign_keep
    floor_multiplier = max(0.0, float(config.production_quality_floor_multiplier))
    quality_keep = (
        pd.to_numeric(out["factor_quality_score"], errors="coerce").fillna(-np.inf).ge(
            pd.to_numeric(out["factor_quality_floor"], errors="coerce").fillna(np.inf) * floor_multiplier,
        )
        if config.production_unified_factor_quality_gate
        else pd.Series(True, index=out.index, dtype=bool)
    )
    quality_keep = quality_keep | blowoff_ladder_keep | follower_catchup_keep
    forward_keep = (
        pd.to_numeric(out.get("forward_quality_selected"), errors="coerce").fillna(0).astype(int).eq(1)
        if config.production_forward_quality_gate
        else pd.Series(True, index=out.index, dtype=bool)
    )
    leader_leg_keep = (
        pd.to_numeric(out.get("leader_leg_quality_selected"), errors="coerce").fillna(0).astype(int).eq(1)
        if config.production_leader_leg_quality_gate and config.production_leader_leg_hard_gate
        else pd.Series(True, index=out.index, dtype=bool)
    )
    transition_keep = pd.Series(True, index=out.index, dtype=bool)
    if config.production_open_transition_gate and config.production_transition_meme_legs_only:
        transition_selected = int(_finite_float(event.get("production_transition_selected"), 0.0)) == 1
        if not transition_selected:
            transition_keep = out.get("symbol", pd.Series("", index=out.index)).map(weijie_asset_group).ne("meme")
    transition_keep = transition_keep | follower_catchup_keep | campaign_keep
    maturity_keep = pd.Series(True, index=out.index, dtype=bool)
    if config.production_eth_anchor_maturity_gate:
        symbol = out.get("symbol", pd.Series("", index=out.index)).astype(str)
        action = out.get("action", pd.Series("", index=out.index)).astype(str)
        family = out.get("reason_family", pd.Series("", index=out.index)).astype(str)
        ret20 = pd.to_numeric(
            out.get("ret20", pd.Series(0.0, index=out.index)),
            errors="coerce",
        ).fillna(0.0)
        ret288 = pd.to_numeric(out.get("ret288"), errors="coerce").fillna(0.0)
        don = pd.to_numeric(out.get("don"), errors="coerce").fillna(0.5)
        taker = pd.to_numeric(out.get("taker"), errors="coerce").fillna(0.5)
        qvr = pd.to_numeric(out.get("qvr"), errors="coerce").fillna(0.0)
        rsi = pd.to_numeric(out.get("rsi"), errors="coerce").fillna(50.0)
        adx = pd.to_numeric(out.get("adx"), errors="coerce").fillna(0.0)
        eth_anchor_short = (
            symbol.eq("ETHUSDT")
            & action.eq("OPEN_SHORT")
            & family.eq("eth_anchor_exhaustion_short")
        )
        still_bid_trend = (
            ret288.gt(0.0)
            & taker.ge(0.60)
            & qvr.ge(1.50)
            & adx.lt(32.0)
        )
        pre_blowoff_extension = (
            ret288.ge(4.0)
            & don.lt(0.70)
            & rsi.lt(65.0)
        )
        low_volume_unconfirmed_top = (
            ret288.ge(2.0)
            & qvr.lt(1.0)
            & rsi.lt(68.0)
            & adx.lt(40.0)
        )
        # A multi-hour blow-off is not mature while its shorter-horizon
        # impulse is still advancing.  Waiting for ret20 to stall avoids
        # repeatedly shorting the high-volume ascent and remains causal: all
        # inputs come from the signal's closed bar.
        blowoff_without_short_horizon_stall = (
            ret288.ge(4.0)
            & ret20.gt(0.50)
        )
        maturity_keep = ~(
            eth_anchor_short
            & (
                still_bid_trend
                | pre_blowoff_extension
                | low_volume_unconfirmed_top
                | blowoff_without_short_horizon_stall
            )
        )
    capitulation_context_keep = (
        pd.Series(
            [_production_capitulation_context_confirmed(row) for _, row in out.iterrows()],
            index=out.index,
            dtype=bool,
        )
        if config.production_capitulation_context_gate
        else pd.Series(True, index=out.index, dtype=bool)
    )
    trend_exhaustion_keep = (
        pd.Series(
            [_production_trend_exhaustion_confirmed(row, event) for _, row in out.iterrows()],
            index=out.index,
            dtype=bool,
        )
        if config.production_trend_exhaustion_confirmation_gate
        else pd.Series(True, index=out.index, dtype=bool)
    )
    directional_context_keep = (
        pd.Series(
            [_production_directional_context_confirmed(row) for _, row in out.iterrows()],
            index=out.index,
            dtype=bool,
        )
        if config.production_directional_context_gate
        else pd.Series(True, index=out.index, dtype=bool)
    )
    return out.loc[
        family_keep
        & legacy_keep
        & quality_keep
        & forward_keep
        & leader_leg_keep
        & transition_keep
        & (maturity_keep | campaign_keep)
        & (capitulation_context_keep | campaign_keep)
        & (trend_exhaustion_keep | campaign_keep)
        & (directional_context_keep | campaign_keep)
        & campaign_scope_keep
    ].copy()


def _production_capitulation_context_confirmed(row: pd.Series) -> bool:
    """Require a real flush or a higher-timeframe trend dip before buying absorption."""

    if (
        str(row.get("action") or "") != "OPEN_LONG"
        or str(row.get("reason_family") or "") != "capitulation_absorption_long"
    ):
        return True
    ret144 = _finite_float(row.get("ret144"), 0.0)
    ret288 = _finite_float(row.get("ret288"), 0.0)
    rsi = _finite_float(row.get("rsi"), 50.0)
    taker = _finite_float(row.get("taker"), 0.5)
    qvr = _finite_float(row.get("qvr"), 0.0)
    oversold_absorption = rsi <= 38.0 and (taker >= 0.40 or qvr <= 4.0)
    return bool(ret144 <= -1.0 or ret288 >= 2.0 or oversold_absorption)


def _production_trend_exhaustion_confirmed(
    row: pd.Series,
    event: dict[str, Any] | None = None,
) -> bool:
    """Keep only fresh, liquid and event-leading exhaustion shorts."""

    if (
        str(row.get("action") or "") != "OPEN_SHORT"
        or str(row.get("reason_family") or "") != "trend_exhaustion_short"
    ):
        return True
    event = event or {}
    return bool(
        _finite_float(row.get("ret60"), 0.0) >= 2.0
        and _finite_float(row.get("ret288"), 0.0) <= 2.0
        and _finite_float(row.get("rsi"), 50.0) >= 65.0
        and _finite_float(row.get("qvr"), 0.0) >= 1.0
        and _finite_float(event.get("signal_count"), 99.0) <= 3.0,
    )


def _production_directional_context_confirmed(row: pd.Series) -> bool:
    """Reject trend labels that conflict with their higher-timeframe direction."""

    action = str(row.get("action") or "")
    family = str(row.get("reason_family") or "")
    symbol = str(row.get("symbol") or "")
    ret20 = _finite_float(row.get("ret20"), 0.0)
    ret60 = _finite_float(row.get("ret60"), 0.0)
    ret144 = _finite_float(row.get("ret144"), 0.0)
    ret288 = _finite_float(row.get("ret288"), 0.0)
    rsi = _finite_float(row.get("rsi"), 50.0)
    seller = _finite_float(row.get("seller"), 0.5)
    qvr = _finite_float(row.get("qvr"), 0.0)
    if action == "OPEN_SHORT" and family == "core_downtrend_short":
        return bool(ret144 <= 0.50 or ret288 <= 0.0)
    if action == "OPEN_LONG" and family == "core_trend_long":
        crowded_chase = ret20 >= 2.0 and rsi >= 68.0 and seller >= 0.55 and qvr >= 2.0
        eth_blowoff_handoff = (
            symbol == "ETHUSDT"
            and ret60 >= 4.0
            and ret288 >= 4.0
            and qvr <= 1.50
        )
        return not (crowded_chase or eth_blowoff_handoff)
    return True


def _production_trend_saturation_confirmed(row: pd.Series) -> bool:
    """Reject downside chases after the local move has reached exhaustion."""

    action = str(row.get("action") or "")
    family = str(row.get("reason_family") or "")
    ret60 = _finite_float(row.get("ret60"), 0.0)
    ret288 = _finite_float(row.get("ret288"), 0.0)
    don = _finite_float(row.get("don"), 0.5)
    rsi = _finite_float(row.get("rsi"), 50.0)
    saturated_short_chase = (
        action == "OPEN_SHORT"
        and family == "core_downtrend_short"
        and ret60 <= -1.0
        and don <= 0.10
        and rsi <= 32.0
        and ret288 >= -0.50
    )
    return not saturated_short_chase


def _filter_production_trend_saturation_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    """Drop saturated selected legs without backfilling their event slots."""

    if candidates.empty:
        return candidates.copy()
    keep = pd.Series(
        [_production_trend_saturation_confirmed(row) for _, row in candidates.iterrows()],
        index=candidates.index,
        dtype=bool,
    )
    return candidates.loc[keep].copy()


def _filter_eth_blowoff_handoff_candidates(
    candidates: pd.DataFrame,
    *,
    active: dict[tuple[str, str], list[dict[str, Any]]],
) -> pd.DataFrame:
    """Suppress a late ETH chase long while an anchor short is already active."""

    if candidates.empty or _active_lot_count(active, ("ETHUSDT", "SHORT")) <= 0:
        return candidates.copy()
    out = candidates.copy()
    symbol = out.get("symbol", pd.Series("", index=out.index)).astype(str)
    action = out.get("action", pd.Series("", index=out.index)).astype(str)
    family = out.get("reason_family", pd.Series("", index=out.index)).astype(str)
    ret60 = pd.to_numeric(out.get("ret60"), errors="coerce").fillna(0.0)
    ret144 = pd.to_numeric(out.get("ret144"), errors="coerce").fillna(0.0)
    ret288 = pd.to_numeric(out.get("ret288"), errors="coerce").fillna(0.0)
    adx = pd.to_numeric(out.get("adx"), errors="coerce").fillna(0.0)
    taker = pd.to_numeric(out.get("taker"), errors="coerce").fillna(0.5)
    conflicting_long = (
        symbol.eq("ETHUSDT")
        & action.eq("OPEN_LONG")
        & family.eq("core_trend_long")
        & ret60.ge(4.0)
        & ret144.ge(4.0)
        & ret288.ge(4.0)
        & adx.ge(55.0)
        & taker.le(0.50)
    )
    return out.loc[~conflicting_long].copy()


def _eth_blowoff_ladder_open_confirmed(row: pd.Series) -> bool:
    """Confirm the exceptional ETH blowoff structure before relaxing add-on spacing."""

    return bool(
        str(row.get("symbol") or "") == "ETHUSDT"
        and str(row.get("action") or "") == "OPEN_SHORT"
        and str(row.get("reason_family") or "") == "eth_anchor_exhaustion_short"
        and _finite_float(row.get("ret60"), 0.0) >= 4.0
        and _finite_float(row.get("ret144"), 0.0) >= 4.0
        and _finite_float(row.get("ret288"), 0.0) >= 4.0
        and _finite_float(row.get("adx"), 0.0) >= 55.0
        and 0.70 <= _finite_float(row.get("qvr"), 0.0) <= 3.0,
    )


def _augment_eth_blowoff_ladder_candidates(
    selected: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    active: dict[tuple[str, str], list[dict[str, Any]]],
) -> pd.DataFrame:
    """Add one earlier confirmed ETH short leg before the final blowoff handoff."""

    if (
        selected.empty
        or candidates.empty
        or _active_lot_count(active, ("ETHUSDT", "SHORT")) <= 0
    ):
        return selected.copy()
    selected_symbol = selected.get("symbol", pd.Series("", index=selected.index)).astype(str)
    selected_action = selected.get("action", pd.Series("", index=selected.index)).astype(str)
    selected_family = selected.get("reason_family", pd.Series("", index=selected.index)).astype(str)
    primary = selected.loc[
        selected_symbol.eq("ETHUSDT")
        & selected_action.eq("OPEN_SHORT")
        & selected_family.eq("eth_anchor_exhaustion_short")
    ].copy()
    if not primary.empty:
        primary = primary.loc[
            pd.Series(
                [_eth_blowoff_ladder_open_confirmed(row) for _, row in primary.iterrows()],
                index=primary.index,
                dtype=bool,
            )
        ].copy()
    if primary.empty:
        return selected.copy()
    primary_time = pd.to_datetime(primary.get("signal_time"), errors="coerce").max()
    if pd.isna(primary_time):
        return selected.copy()

    frame = candidates.copy()
    signal_time = pd.to_datetime(frame.get("signal_time"), errors="coerce")
    symbol = frame.get("symbol", pd.Series("", index=frame.index)).astype(str)
    action = frame.get("action", pd.Series("", index=frame.index)).astype(str)
    family = frame.get("reason_family", pd.Series("", index=frame.index)).astype(str)
    ret60 = pd.to_numeric(frame.get("ret60"), errors="coerce").fillna(0.0)
    ret144 = pd.to_numeric(frame.get("ret144"), errors="coerce").fillna(0.0)
    ret288 = pd.to_numeric(frame.get("ret288"), errors="coerce").fillna(0.0)
    adx = pd.to_numeric(frame.get("adx"), errors="coerce").fillna(0.0)
    qvr = pd.to_numeric(frame.get("qvr"), errors="coerce").fillna(0.0)
    gap_minutes = (pd.Timestamp(primary_time) - signal_time).dt.total_seconds() / 60.0
    selected_ids = set(selected.get("signal_id", pd.Series(dtype=str)).astype(str))
    eligible = frame.loc[
        symbol.eq("ETHUSDT")
        & action.eq("OPEN_SHORT")
        & family.eq("eth_anchor_exhaustion_short")
        & ret60.ge(4.0)
        & ret144.ge(4.0)
        & ret288.ge(4.0)
        & adx.ge(55.0)
        & qvr.between(0.70, 3.0, inclusive="both")
        & gap_minutes.between(10.0, 30.0, inclusive="both")
        & ~frame.get("signal_id", pd.Series("", index=frame.index)).astype(str).isin(selected_ids)
    ].copy()
    if eligible.empty:
        return selected.copy()
    ladder_leg = eligible.sort_values(["signal_time", "score"], ascending=[True, False]).head(1).copy()
    ladder_leg["eth_blowoff_ladder_augmented"] = 1
    return (
        pd.concat([selected, ladder_leg], ignore_index=False, sort=False)
        .sort_values(["signal_time", "symbol", "action"])
        .copy()
    )


def _production_follower_catchup_addon_confirmed(
    row: pd.Series,
    event: dict[str, Any],
) -> bool:
    """Confirm a meme follower beginning to catch up with broad market direction."""

    symbol = str(row.get("symbol") or "")
    action = str(row.get("action") or "")
    family = str(row.get("reason_family") or "")
    if weijie_asset_group(symbol) != "meme":
        return False
    if family != "core_trend_long":
        return False
    if action != "OPEN_LONG":
        return False
    directional_ret20 = _finite_float(row.get("ret20"), 0.0)
    directional_ret60 = _finite_float(row.get("ret60"), 0.0)
    breadth = _finite_float(event.get("market_up"), 0.0)
    rsi = _finite_float(row.get("rsi"), 50.0)
    return bool(
        breadth >= 0.75
        and 0.40 <= directional_ret20 <= 1.50
        and -0.75 <= directional_ret60 <= 1.00
        and abs(_finite_float(row.get("ret288"), 0.0)) >= 2.0
        and 50.0 <= rsi <= 65.0
        and 4.0 <= _finite_float(event.get("signal_count"), 0.0) <= 7.0
        and _finite_float(event.get("cluster_mature_fraction"), 1.0) <= 0.75
        and int(_finite_float(event.get("production_transition_selected"), 0.0)) == 0,
    )


def _augment_follower_catchup_addon_candidates(
    selected: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    active: dict[tuple[str, str], list[dict[str, Any]]],
    event: dict[str, Any],
) -> pd.DataFrame:
    """Add at most one confirmed follower lot when the same state is already active."""

    if candidates.empty:
        return selected.copy()
    selected_ids = set(selected.get("signal_id", pd.Series(dtype=str)).astype(str))
    eligible_rows: list[pd.Series] = []
    for _, row in candidates.iterrows():
        if str(row.get("signal_id") or "") in selected_ids:
            continue
        if not _production_follower_catchup_addon_confirmed(row, event):
            continue
        direction = direction_from_action(row.get("action"))
        if _active_lot_count(active, (str(row.get("symbol") or ""), direction)) <= 0:
            continue
        eligible_rows.append(row)
    if not eligible_rows:
        return selected.copy()
    eligible = pd.DataFrame(eligible_rows)
    direction_sign = eligible.get("action", pd.Series("", index=eligible.index)).map(
        {"OPEN_LONG": 1.0, "OPEN_SHORT": -1.0},
    ).fillna(0.0)
    eligible["_follower_rank"] = (
        direction_sign * pd.to_numeric(eligible.get("ret20"), errors="coerce").fillna(0.0)
    )
    addon = eligible.sort_values(
        ["_follower_rank", "score", "signal_time"],
        ascending=[False, False, True],
    ).head(1).copy()
    addon["follower_catchup_augmented"] = 1
    addon = addon.drop(columns=["_follower_rank"], errors="ignore")
    return (
        pd.concat([selected, addon], ignore_index=False, sort=False)
        .sort_values(["signal_time", "symbol", "action"])
        .copy()
    )


def production_open_factor_quality_score(row: pd.Series, event: dict[str, Any] | None = None) -> float:
    """Unified OHLCV structure score for production open candidates.

    The score is intentionally family-based rather than symbol-based: symbols
    can have different volatility, but a live candidate still needs the same
    structural evidence for trend continuation, capitulation absorption,
    bounce fade, or cascade continuation.
    """

    event = event or {}
    action = str(row.get("action") or "")
    family = str(row.get("reason_family") or "")
    asset_group = weijie_asset_group(row.get("symbol"))
    ret20 = _finite_float(row.get("ret20"), 0.0)
    ret60 = _finite_float(row.get("ret60"), 0.0)
    ret144 = _finite_float(row.get("ret144"), 0.0)
    ret288 = _finite_float(row.get("ret288"), 0.0)
    don = _finite_float(row.get("don"), 0.5)
    taker = _finite_float(row.get("taker"), 0.5)
    seller = _finite_float(row.get("seller"), 0.5)
    qvr = _finite_float(row.get("qvr"), 1.0)
    rsi = _finite_float(row.get("rsi"), 50.0)
    adx = _finite_float(row.get("adx"), 20.0)
    down = _finite_float(row.get("down"), 0.0)
    up = _finite_float(row.get("up"), 0.0)
    cluster_age = _finite_float(row.get("signal_cluster_age_min"), 0.0)
    cluster_count = int(_finite_float(row.get("signal_cluster_count"), 1.0))
    event_pressure = _finite_float(event.get("event_pressure"), 0.0)
    max_score = _finite_float(event.get("max_score"), 0.0)
    market_down = _finite_float(event.get("market_down"), 0.0)
    market_up = _finite_float(event.get("market_up"), 0.0)
    event_mean_ret144 = _finite_float(event.get("mean_ret144"), 0.0)
    event_mean_ret288 = _finite_float(event.get("mean_ret288"), 0.0)
    quality = 0.0

    if action == "OPEN_LONG" and family == "core_trend_long":
        mature_trend_acceleration = (
            3.00 <= ret60 <= 4.20
            and ret144 >= 1.00
            and ret288 >= 1.00
            and 0.82 <= don <= 0.94
            and 62.0 <= rsi <= 72.0
            and adx >= 30.0
            and qvr <= 3.00
        )
        liquidity_expansion_continuation = (
            0.45 <= ret60 <= 1.20
            and ret144 >= 1.00
            and 0.85 <= don <= 0.95
            and 65.0 <= rsi <= 72.0
            and qvr >= 6.00
            and taker >= 0.55
            and up >= 0.75
            and down <= 0.25
        )
        breakout_expansion_continuation = (
            3.80 <= ret60 <= 6.00
            and ret144 >= 3.00
            and ret288 >= 3.00
            and don >= 0.88
            and 60.0 <= rsi <= 74.0
            and adx >= 22.0
            and taker >= 0.52
            and qvr <= 6.00
        )
        if 0.45 <= ret60 <= 3.80:
            quality += 0.35
        if ret144 >= 0.20:
            quality += 0.30
        if ret288 >= -1.00:
            quality += 0.15
        if 0.62 <= don <= 0.92:
            quality += 0.25
        if 52.0 <= rsi <= 66.0:
            quality += 0.25
        if taker >= 0.52 or up >= 0.45:
            quality += 0.25
        if 0.30 <= qvr <= 3.50:
            quality += 0.20
        if adx >= 22.0:
            quality += 0.15
        if cluster_count >= 2 and 35.0 <= cluster_age <= 260.0 and ret20 <= 0.45:
            quality += 0.35
        if ret20 >= 0.80 and don >= 0.86 and rsi >= 67.0:
            quality -= 0.70
        if qvr >= 6.0 and rsi >= 68.0:
            quality -= 0.55
        if mature_trend_acceleration:
            quality += 0.50
        if liquidity_expansion_continuation:
            quality += 1.20
        if breakout_expansion_continuation:
            quality += 0.90

    elif action == "OPEN_LONG" and family in {"session_pullback_reclaim_long", "range_reclaim_long", "early_session_reclaim_long"}:
        if -0.80 <= ret60 <= 1.20:
            quality += 0.25
        if ret288 >= 0.50:
            quality += 0.25
        if 0.20 <= don <= 0.75:
            quality += 0.25
        if 42.0 <= rsi <= 58.0:
            quality += 0.30
        if taker >= 0.50 or up >= 0.40:
            quality += 0.25
        if 0.45 <= qvr <= 4.00:
            quality += 0.15
        if cluster_age <= 90.0:
            quality += 0.10
        if ret20 >= 1.00 and rsi >= 66.0:
            quality -= 0.45
        if family == "range_reclaim_long":
            trend_resumption_confirmed = (
                ret60 >= 0.10
                and ret288 >= 0.25
                and rsi <= 56.0
                and adx <= 22.0
            )
            eth_trend_reset_reclaim = (
                asset_group == "eth"
                and -0.50 <= ret60 <= 0.10
                and -0.80 <= ret144 <= 0.20
                and -0.30 <= ret288 <= 0.50
                and 0.55 <= don <= 0.75
                and taker >= 0.55
                and 0.45 <= qvr <= 1.50
                and 52.0 <= rsi <= 58.0
                and 25.0 <= adx <= 40.0
                and market_down <= 0.65
                and market_up >= 0.45
            )
            if trend_resumption_confirmed or eth_trend_reset_reclaim:
                quality += 0.55
            else:
                quality -= 10.0

    elif action == "OPEN_LONG" and family in {"capitulation_absorption_long", "range_reclaim_long"}:
        liquidation_flush_without_reclaim = (
            ret144 <= -2.00
            and don <= 0.15
            and rsi <= 30.0
            and qvr >= 8.00
            and taker < 0.55
        )
        unconfirmed_breadth_flush = (
            ret20 <= -0.75
            and ret60 <= -1.50
            and ret144 <= -2.00
            and ret288 <= -2.00
            and 0.15 <= don <= 0.35
            and 32.0 <= rsi <= 42.0
            and down >= 0.95
            and qvr <= 4.00
        )
        if ret60 <= -0.80:
            quality += 0.30
        if ret144 <= -0.50:
            quality += 0.20
        if don <= 0.40 or down >= 0.65:
            quality += 0.30
        if 24.0 <= rsi <= 42.0:
            quality += 0.30
        if taker >= 0.35:
            quality += 0.20
        if qvr >= 0.80:
            quality += 0.15
        if adx >= 20.0:
            quality += 0.10
        if ret20 <= -1.20 and qvr >= 2.00:
            quality += 0.20
        if ret60 > -0.40 and rsi > 45.0:
            quality -= 0.65
        if liquidation_flush_without_reclaim:
            quality -= 0.80
        if unconfirmed_breadth_flush:
            quality -= 1.00
        meme_breadth_reclaim = asset_group == "meme" and (down <= 0.85 or up >= 0.15)
        eth_liquidation_absorption = (
            asset_group == "eth"
            and ret20 <= -0.90
            and don <= 0.25
            and qvr >= 8.00
            and rsi <= 36.0
            and adx >= 20.0
        )
        eth_persistent_liquidation = (
            asset_group == "eth"
            and cluster_count >= 3
            and 20.0 <= cluster_age <= 75.0
            and ret60 <= -1.00
            and ret288 <= -1.00
            and qvr >= 5.00
            and rsi <= 40.0
            and adx >= 25.0
        )
        eth_post_flush_reclaim = (
            asset_group == "eth"
            and cluster_count >= 2
            and 15.0 <= cluster_age <= 90.0
            and ret20 <= -0.80
            and 0.00 <= ret60 <= 1.00
            and ret288 <= -4.00
            and up >= 0.20
            and rsi <= 35.0
        )
        eth_shallow_pullback_without_capitulation = (
            asset_group == "eth"
            and ret144 > -0.50
            and ret288 > -1.00
        )
        stock_gold_absorption = asset_group == "stock_gold" and (down <= 0.85 or up >= 0.15)
        # A broad meme selloff is not absorption until breadth starts to
        # recover. ETH liquidation and stock/gold use their own class-level
        # structure instead of symbol exceptions.
        if event and market_down >= 0.75 and not (
            meme_breadth_reclaim
            or eth_liquidation_absorption
            or eth_persistent_liquidation
            or eth_post_flush_reclaim
            or stock_gold_absorption
        ):
            quality -= 10.0
        if eth_shallow_pullback_without_capitulation:
            quality -= 10.0

    elif action == "OPEN_LONG" and family in {"eth_anchor_long", "midnight_breadth_breakout_long"}:
        eth_flow_led_anchor = (
            asset_group == "eth"
            and -0.10 <= ret20 <= 0.60
            and -0.60 <= ret60 <= 0.10
            and -0.80 <= ret144 <= 0.10
            and -0.30 <= ret288 <= 0.50
            and 0.70 <= don <= 0.90
            and taker >= 0.62
            and 1.0 <= qvr <= 3.0
            and 56.0 <= rsi <= 62.0
            and 15.0 <= adx <= 25.0
            and up >= 0.75
            and down <= 0.25
        )
        if ret60 >= 1.20:
            quality += 0.30
        if ret144 >= 1.20:
            quality += 0.25
        if ret288 >= 1.00:
            quality += 0.15
        if 0.55 <= don <= 0.88:
            quality += 0.20
        if 58.0 <= rsi <= 74.0:
            quality += 0.25
        if taker >= 0.52:
            quality += 0.20
        if qvr >= 1.00:
            quality += 0.15
        if adx >= 25.0:
            quality += 0.15
        if rsi >= 78.0 or don >= 0.96:
            quality -= 0.45
        if family == "eth_anchor_long" and eth_flow_led_anchor:
            quality += 0.50
        if (
            family == "midnight_breadth_breakout_long"
            and 1.80 <= ret60 <= 2.40
            and 0.20 <= ret144 <= 0.65
            and 0.70 <= ret288 <= 1.20
            and don >= 0.88
            and qvr >= 3.00
            and 66.0 <= rsi <= 70.0
            and adx >= 35.0
        ):
            quality += 0.45

    elif family == "regime_trend_rollover" and action in {"OPEN_LONG", "OPEN_SHORT"}:
        decision = decide_market_regime(row)
        if decision.mechanism == "trend_rollover" and decision.action == action:
            quality = 0.75 + float(decision.confidence)

    elif action == "OPEN_SHORT" and family == "core_downtrend_short":
        if ret60 <= -0.25:
            quality += 0.30
        if ret144 <= -0.80:
            quality += 0.35
        if ret288 <= 0.50:
            quality += 0.10
        if don <= 0.45 or down >= 0.65:
            quality += 0.30
        if seller >= 0.55 or taker <= 0.45:
            quality += 0.25
        if 18.0 <= rsi <= 52.0:
            quality += 0.25
        if qvr >= 0.70:
            quality += 0.15
        if adx >= 16.0:
            quality += 0.15
        if -0.20 <= ret60 <= 0.20 and ret144 <= -1.20 and don <= 0.40:
            quality += 0.25
        if ret60 <= -1.00 and ret144 <= -1.50 and ret288 <= -0.50 and don <= 0.45 and seller >= 0.58:
            quality += 0.35
        deep_trend_continuation_short = (
            ret60 <= -2.50
            and ret144 <= -2.00
            and ret288 <= -2.00
            and don <= 0.45
            and adx >= 25.0
            and 45.0 <= rsi <= 58.0
            and qvr >= 0.50
        )
        if deep_trend_continuation_short:
            quality += 0.55
        if ret20 <= -1.20 and down >= 0.75:
            quality += 0.20
        low_volume_continuation_pullback = (
            ret60 <= -1.00
            and ret144 <= -1.50
            and ret288 <= -0.50
            and don <= 0.45
            and seller >= 0.58
            and 36.0 <= rsi <= 55.0
        )
        if (
            cluster_age >= 60.0
            and ret60 <= -1.20
            and qvr < 1.00
            and not low_volume_continuation_pullback
            and not deep_trend_continuation_short
        ):
            quality -= 0.55
        if ret60 > 0.50 and ret144 > 0.50:
            quality -= 0.75
        broad_capitulation_exhaustion = (
            market_down >= 0.75
            and market_up <= 0.35
            and event_mean_ret144 <= -0.75
            and event_mean_ret288 <= -1.00
        )
        if broad_capitulation_exhaustion and asset_group != "stock_gold":
            quality -= 1.75

    elif action == "OPEN_SHORT" and family == "weak_24h_bounce_short":
        if 0.20 <= ret60 <= 1.80:
            quality += 0.30
        if ret144 <= -0.80:
            quality += 0.35
        if ret288 >= 0.50:
            quality += 0.20
        if 0.55 <= don <= 0.88:
            quality += 0.20
        if 52.0 <= rsi <= 64.0:
            quality += 0.25
        if taker >= 0.55:
            quality += 0.20
        if 0.50 <= qvr <= 5.00:
            quality += 0.15
        if event_pressure >= 8.0 or max_score >= 2.5:
            quality += 0.10
        if ret60 >= 2.50 or rsi >= 70.0:
            quality -= 0.50

    elif action == "OPEN_SHORT" and family == "eth_anchor_exhaustion_short":
        if 1.20 <= ret60 <= 3.90:
            quality += 0.30
        if ret20 <= 2.20:
            quality += 0.15
        if don >= 0.70:
            quality += 0.25
        if 58.0 <= rsi <= 84.0:
            quality += 0.25
        if adx >= 25.0:
            quality += 0.20
        if qvr >= 0.70:
            quality += 0.15
        if 60.0 <= cluster_age <= 190.0:
            quality += 0.20
        if ret60 >= 4.20 and rsi >= 78.0:
            quality -= 0.55
        slow_rise_distribution = (
            0.45 <= ret60 <= 0.95
            and ret144 >= 1.25
            and ret288 >= 1.00
            and 0.68 <= don <= 0.82
            and 49.0 <= rsi <= 55.0
            and adx >= 30.0
            and seller >= 0.50
            and qvr <= 1.30
        )
        if slow_rise_distribution:
            quality += 0.75

    elif action == "OPEN_SHORT" and family == "trend_exhaustion_short":
        if 1.00 <= ret20 <= 2.00:
            quality += 0.20
        if 1.50 <= ret60 <= 3.00:
            quality += 0.30
        if ret144 >= 1.00 and ret288 >= 0.00:
            quality += 0.20
        if 0.74 <= don <= 0.90:
            quality += 0.25
        if 62.0 <= rsi <= 72.0:
            quality += 0.25
        if adx >= 30.0:
            quality += 0.20
        if seller >= 0.55:
            quality += 0.20
        if 0.75 <= qvr <= 3.00:
            quality += 0.15

    elif action == "OPEN_SHORT" and family == "paxg_defensive_short":
        if ret144 <= 0.20:
            quality += 0.20
        if don <= 0.45 or down >= 0.65:
            quality += 0.30
        if 35.0 <= rsi <= 60.0:
            quality += 0.25
        if qvr >= 1.00:
            quality += 0.15
        if seller >= 0.50 or taker <= 0.50:
            quality += 0.20

    return float(quality)


def production_open_factor_quality_floor(row: pd.Series, event: dict[str, Any] | None = None) -> float:
    del event
    action = str(row.get("action") or "")
    family = str(row.get("reason_family") or "")
    if action == "OPEN_LONG" and family == "core_trend_long":
        return 1.20
    if action == "OPEN_LONG" and family == "capitulation_absorption_long":
        return 1.05
    if action == "OPEN_LONG" and family in {"session_pullback_reclaim_long", "range_reclaim_long", "early_session_reclaim_long"}:
        return 0.95
    if action == "OPEN_LONG" and family in {"eth_anchor_long", "midnight_breadth_breakout_long"}:
        return 1.05
    if action == "OPEN_SHORT" and family == "core_downtrend_short":
        return 1.15
    if action == "OPEN_SHORT" and family == "weak_24h_bounce_short":
        return 1.20
    if action == "OPEN_SHORT" and family == "eth_anchor_exhaustion_short":
        return 0.95
    if action == "OPEN_SHORT" and family == "trend_exhaustion_short":
        return 1.10
    if action == "OPEN_SHORT" and family == "paxg_defensive_short":
        return 0.90
    if family == "regime_trend_rollover" and action in {"OPEN_LONG", "OPEN_SHORT"}:
        return 0.75
    return 0.75


def production_open_factor_quality_allowed(row: pd.Series, event: dict[str, Any] | None = None) -> bool:
    return production_open_factor_quality_score(row, event) >= production_open_factor_quality_floor(row, event)


def _filter_production_open_point_accuracy_candidates(
    candidates: pd.DataFrame,
    event: dict[str, Any],
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    keep = [_production_open_point_accuracy_allowed(row, event) for _, row in candidates.iterrows()]
    return candidates.loc[keep].copy()


def _apply_production_open_point_accuracy_scores(candidates: pd.DataFrame, event: dict[str, Any]) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    out = candidates.copy()
    out["point_accuracy_score"] = [
        _production_open_point_accuracy_score(row, event) for _, row in out.iterrows()
    ]
    return out


def _production_open_point_accuracy_score(row: pd.Series, event: dict[str, Any]) -> float:
    del event
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
    qvr = _finite_float(row.get("qvr"), 0.0)
    rsi = _finite_float(row.get("rsi"), 50.0)
    adx = _finite_float(row.get("adx"), 0.0)
    down = _finite_float(row.get("down"), 0.0)
    up = _finite_float(row.get("up"), 0.0)
    cluster_age = _finite_float(row.get("signal_cluster_age_min"), 0.0)
    cluster_count = int(_finite_float(row.get("signal_cluster_count"), 1.0))
    score = 0.0

    if action == "OPEN_LONG" and family == "core_trend_long":
        score -= 0.30 * min(max(qvr, 0.0), 12.0)
        if cluster_count >= 2 and 45.0 <= cluster_age <= 210.0 and 0.45 <= ret60 <= 3.80 and don >= 0.74 and 54.0 <= rsi <= 70.0:
            score += 1.20
        if cluster_count >= 3 and 60.0 <= cluster_age <= 240.0 and ret20 <= 0.35 and 1.20 <= ret60 <= 4.20 and don >= 0.74 and 52.0 <= rsi <= 62.0:
            score += 1.85
        if cluster_count >= 3 and 60.0 <= cluster_age <= 130.0 and 0.80 <= ret20 <= 1.50 and 1.40 <= ret60 <= 2.20 and don >= 0.88 and 65.0 <= rsi <= 72.0:
            score += 0.85
        if cluster_age <= 15.0 and rsi >= 68.0 and ret60 < 1.0 and ret144 < 2.5:
            score -= 1.10
        if qvr >= 8.0 and rsi >= 68.0 and (ret20 >= 0.45 or cluster_age >= 180.0):
            score -= 3.00
        if qvr >= 4.0 and rsi >= 70.0 and ret60 <= 2.10 and ret144 <= 1.50:
            score -= 2.35
        if 1 <= hour <= 2 and ret20 >= 1.50 and ret60 >= 3.0 and don >= 0.85 and rsi >= 63.0:
            score -= 2.20
        if ret60 >= 4.0 and rsi >= 70.0 and cluster_age >= 150.0:
            score -= 0.70

    if action == "OPEN_LONG" and family == "capitulation_absorption_long":
        score -= 0.12 * min(max(qvr, 0.0), 12.0)
        if cluster_count >= 2 and 30.0 <= cluster_age <= 120.0 and ret60 <= -1.50 and rsi <= 35.0 and down >= 0.85:
            score += 0.90
        if ret20 <= -1.20 and ret60 <= -1.50 and qvr >= 4.0 and rsi <= 32.0:
            score += 0.45
        if cluster_age < 30.0 and ret60 > -1.30 and rsi > 34.0 and qvr < 3.0:
            score -= 0.70
        if qvr >= 14.0 and rsi <= 25.0 and ret144 > -3.0:
            score -= 1.20

    if action == "OPEN_LONG" and family == "range_reclaim_long":
        if symbol == "ETHUSDT" and cluster_age <= 0.0 and ret60 > -0.10 and qvr < 3.0:
            score -= 0.50

    if action == "OPEN_SHORT" and family == "eth_anchor_exhaustion_short":
        if 90.0 <= cluster_age <= 180.0 and 1.80 <= ret60 <= 3.50 and 55.0 <= rsi <= 62.0 and don >= 0.75:
            score += 2.10
        if 150.0 <= cluster_age <= 180.0 and 2.30 <= ret60 <= 2.90 and 58.0 <= rsi <= 61.0 and don >= 0.80:
            score += 0.90
        if cluster_age >= 185.0 and ret60 >= 2.50 and rsi >= 62.0:
            score -= 1.60
        if cluster_age >= 200.0 and (ret60 >= 4.0 or ret20 >= 2.0) and rsi >= 75.0:
            score -= 2.10
        if qvr >= 8.0 and rsi >= 75.0:
            score -= 1.20

    if action == "OPEN_SHORT" and family == "core_downtrend_short":
        score -= 0.18 * min(max(qvr, 0.0), 10.0)
        if cluster_age <= 20.0 and ret144 <= -1.20 and -0.20 <= ret60 <= 0.20 and don <= 0.38 and qvr >= 2.0 and 32.0 <= rsi <= 45.0 and down >= 0.70:
            score += 2.20
        if cluster_count >= 2 and 20.0 <= cluster_age <= 90.0 and don <= 0.35 and rsi <= 40.0 and down >= 0.75:
            score += 1.35
        if symbol == "CFXUSDT" and cluster_age <= 20.0 and -0.60 <= ret20 <= 0.05 and ret60 <= 0.30 and don <= 0.20 and qvr >= 2.0 and rsi <= 36.0 and down >= 0.85:
            score += 1.90
        if symbol == "ETCUSDT" and 7 <= hour <= 8 and 20.0 <= cluster_age <= 80.0 and don <= 0.35 and qvr >= 0.70 and 34.0 <= rsi <= 46.0 and down >= 0.75:
            score += 1.45
        if symbol == "PAXGUSDT" and 20 <= hour <= 21 and qvr >= 3.0 and rsi <= 35.0 and down >= 0.75 and don <= 0.35:
            score += 2.35
        if symbol == "PAXGUSDT" and hour < 19 and ret144 > -1.0 and ret288 > -1.0:
            score -= 2.40
        if cluster_age <= 0.0 and ret60 > -0.20 and qvr < 1.0 and 40.0 <= rsi <= 50.0:
            score -= 0.65
        if cluster_age >= 45.0 and ret60 <= -0.80 and rsi >= 36.0 and qvr < 5.0:
            score -= 1.05
        if qvr >= 12.0 and rsi <= 25.0 and adx < 30.0:
            score -= 1.40

    if action == "OPEN_SHORT" and family == "paxg_defensive_short":
        if symbol == "PAXGUSDT" and hour == 20 and down >= 0.80 and rsi > 50.0:
            score -= 2.10
        if symbol == "PAXGUSDT" and down >= 0.85 and 45.0 <= rsi <= 52.0 and qvr >= 3.0:
            score += 0.45

    if action == "OPEN_LONG" and family == "session_pullback_reclaim_long":
        if cluster_age <= 15.0 and 42.0 <= rsi <= 54.0 and ret288 >= 1.0 and (qvr >= 0.55 or up >= 0.45):
            score += 0.35

    return float(score)


def _production_open_point_accuracy_allowed(row: pd.Series, event: dict[str, Any]) -> bool:
    del event
    symbol = str(row.get("symbol") or "")
    action = str(row.get("action") or "")
    family = str(row.get("reason_family") or "")
    signal_time = pd.Timestamp(row.get("signal_time")) if row.get("signal_time") is not None else pd.NaT
    hour = int(signal_time.hour) if pd.notna(signal_time) else -1
    ret20 = _finite_float(row.get("ret20"), 0.0)
    ret60 = _finite_float(row.get("ret60"), 0.0)
    ret144 = _finite_float(row.get("ret144"), 0.0)
    don = _finite_float(row.get("don"), 0.5)
    qvr = _finite_float(row.get("qvr"), 0.0)
    rsi = _finite_float(row.get("rsi"), 50.0)
    down = _finite_float(row.get("down"), 0.0)
    cluster_age = _finite_float(row.get("signal_cluster_age_min"), 0.0)
    if action == "OPEN_LONG" and family == "core_trend_long":
        first_bar_heat = cluster_age <= 15.0 and rsi >= 68.0 and ret60 < 1.0 and ret144 < 2.5
        qvr_chase_heat = qvr >= 8.0 and rsi >= 68.0 and (ret20 >= 0.45 or cluster_age >= 180.0)
        early_qvr_chase = cluster_age <= 45.0 and qvr >= 4.0 and rsi >= 70.0 and ret60 < 2.0 and ret144 < 1.0
        alt_crowded_trend_chase = (
            symbol != "ETHUSDT"
            and qvr >= 2.0
            and rsi >= 62.0
            and don >= 0.82
            and ret20 >= 0.30
        )
        sol_overnight_impulse_chase = (
            symbol == "SOLUSDT"
            and 1 <= hour <= 2
            and ret20 >= 1.50
            and ret60 >= 3.00
            and don >= 0.85
            and rsi >= 63.0
        )
        if (
            first_bar_heat
            or qvr_chase_heat
            or early_qvr_chase
            or alt_crowded_trend_chase
            or sol_overnight_impulse_chase
        ):
            return False

    if action == "OPEN_LONG" and family == "capitulation_absorption_long":
        shallow_absorption = (
            symbol == "ETHUSDT"
            and hour == 20
            and ret60 > -1.50
            and rsi > 28.0
            and qvr >= 4.0
        )
        weak_regular_absorption = (
            cluster_age < 30.0
            and ret60 > -1.30
            and rsi > 34.0
            and qvr < 3.0
        )
        late_shallow_regular_absorption = (
            symbol in {"BCHUSDT", "LTCUSDT"}
            and hour in {6, 21, 22}
            and ret60 > -2.0
            and rsi > 36.0
            and qvr < 4.0
        )
        if shallow_absorption or weak_regular_absorption or late_shallow_regular_absorption:
            return False

    if action == "OPEN_LONG" and family == "range_reclaim_long":
        if symbol == "ETHUSDT" and cluster_age <= 0.0 and ret60 > -0.10 and qvr < 3.0:
            return False

    if action == "OPEN_SHORT" and family == "eth_anchor_exhaustion_short":
        late_blowoff_chase = (
            (cluster_age >= 190.0 and ret60 >= 2.50 and rsi >= 64.0)
            or (ret60 >= 4.0 and rsi >= 75.0 and qvr >= 8.0)
            or (symbol == "ETHUSDT" and ret60 >= 4.0 and ret20 >= 3.0 and rsi >= 75.0)
        )
        if late_blowoff_chase:
            return False

    if action == "OPEN_SHORT" and family == "core_downtrend_short":
        first_unconfirmed_short = cluster_age <= 0.0 and ret60 > -0.20 and qvr < 1.0 and 40.0 <= rsi <= 50.0
        late_confirmed_chase = cluster_age >= 45.0 and ret60 <= -0.80 and rsi >= 36.0 and qvr < 5.0
        shallow_late_short = cluster_age >= 45.0 and abs(ret60) < 0.30 and rsi >= 40.0 and qvr < 1.0
        if first_unconfirmed_short or late_confirmed_chase or shallow_late_short:
            return False

    if action == "OPEN_SHORT" and family == "paxg_defensive_short":
        if symbol == "PAXGUSDT" and hour == 20 and down >= 0.80 and rsi > 50.0:
            return False

    return True


def _production_open_quality_allowed(row: pd.Series, event: dict[str, Any]) -> bool:
    symbol = str(row.get("symbol") or "")
    asset_group = weijie_asset_group(symbol)
    action = str(row.get("action") or "")
    family = str(row.get("reason_family") or "")
    signal_time = pd.Timestamp(row.get("signal_time")) if row.get("signal_time") is not None else pd.NaT
    hour = int(signal_time.hour) if pd.notna(signal_time) else -1
    score = _finite_float(row.get("leg_score"), _finite_float(row.get("score"), 0.0))
    event_score = _finite_float(event.get("event_score"), 0.0)
    ret20 = _finite_float(row.get("ret20"), 0.0)
    ret60 = _finite_float(row.get("ret60"), 0.0)
    ret144 = _finite_float(row.get("ret144"), 0.0)
    ret288 = _finite_float(row.get("ret288"), 0.0)
    don = _finite_float(row.get("don"), 0.5)
    taker = _finite_float(row.get("taker"), 0.5)
    seller = _finite_float(row.get("seller"), 0.5)
    qvr = _finite_float(row.get("qvr"), 0.0)
    rsi = _finite_float(row.get("rsi"), 50.0)
    adx = _finite_float(row.get("adx"), 0.0)
    down = _finite_float(row.get("down"), 0.0)
    cluster_age = _finite_float(row.get("signal_cluster_age_min"), 0.0)
    cluster_count = int(_finite_float(row.get("signal_cluster_count"), 1.0))

    if action == "OPEN_LONG":
        if family == "core_trend_long":
            return market_regime_open_allowed(row)
        if family == "eth_anchor_long":
            asia_breakout_anchor = bool(
                symbol == "ETHUSDT"
                and 6 <= hour <= 7
                and ret20 >= 2.50
                and ret60 >= 2.00
                and ret144 >= 2.00
                and ret288 >= 5.00
                and 0.55 <= don <= 0.75
                and taker >= 0.52
                and qvr >= 5.0
                and 65.0 <= rsi <= 72.0
                and adx >= 30.0,
            )
            flow_led_anchor = bool(
                asset_group == "eth"
                and 20 <= hour <= 22
                and -0.10 <= ret20 <= 0.60
                and -0.60 <= ret60 <= 0.10
                and -0.80 <= ret144 <= 0.10
                and -0.30 <= ret288 <= 0.50
                and 0.70 <= don <= 0.90
                and taker >= 0.62
                and 1.0 <= qvr <= 3.0
                and 56.0 <= rsi <= 62.0
                and 15.0 <= adx <= 25.0
                and down <= 0.25,
            )
            return asia_breakout_anchor or flow_led_anchor

        if family == "midnight_breadth_breakout_long":
            return bool(
                symbol == "CFXUSDT"
                and hour == 0
                and ret20 >= 1.00
                and 1.80 <= ret60 <= 2.40
                and 0.20 <= ret144 <= 0.65
                and 0.70 <= ret288 <= 1.20
                and don >= 0.88
                and qvr >= 3.0
                and 66.0 <= rsi <= 70.0
                and adx >= 35.0
                and down <= 0.10
                and taker <= 0.45,
            )

        if family == "capitulation_absorption_long":
            liquidation_flush_without_reclaim = (
                ret144 <= -2.00
                and don <= 0.15
                and rsi <= 30.0
                and qvr >= 8.00
                and taker < 0.55
            )
            if liquidation_flush_without_reclaim:
                return False
            unconfirmed_breadth_flush = (
                ret20 <= -0.75
                and ret60 <= -1.50
                and ret144 <= -2.00
                and ret288 <= -2.00
                and 0.15 <= don <= 0.35
                and 32.0 <= rsi <= 42.0
                and down >= 0.95
                and qvr <= 4.00
            )
            if unconfirmed_breadth_flush:
                return False
            if symbol == "BCHUSDT" and hour == 20 and ret20 > -1.0 and down < 0.75 and taker < 0.45:
                return False
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
                and hour in {8, 13, 16, 19, 21}
                and ret60 <= -1.00
                and don <= 0.30
                and rsi <= 38.0
            )
            if eth_shallow_tail_absorption or ltc_session_absorption_trap:
                return False
            eth_first_panic_spike = (
                symbol == "ETHUSDT"
                and ret20 <= -2.50
                and ret60 <= -2.50
                and don <= 0.08
                and qvr >= 20.0
                and rsi <= 30.0
                and down >= 0.95
            )
            if eth_first_panic_spike:
                return False
            if symbol == "ETHUSDT" and 20 <= hour <= 21:
                return bool(
                    ret20 <= -1.20
                    and ret60 <= -2.0
                    and ret144 <= -2.0
                    and 20.0 <= rsi <= 32.0
                    and adx >= 40.0
                    and down >= 0.95
                    and qvr >= 4.0
                    and taker >= 0.35,
                )
            eth_deep_flush_absorption = (
                symbol == "ETHUSDT"
                and ret20 <= -1.20
                and ret60 <= -1.50
                and ret144 <= -1.50
                and rsi <= 38.0
                and adx >= 30.0
                and down >= 0.85
                and qvr >= 1.40
                and taker >= 0.35
            )
            eth_late_capitulation_absorption = (
                symbol == "ETHUSDT"
                and ret288 <= -4.50
                and ret20 <= -0.85
                and ret144 <= -1.00
                and don <= 0.35
                and qvr >= 1.40
                and rsi <= 37.0
                and down >= 0.75
            )
            if eth_deep_flush_absorption or eth_late_capitulation_absorption:
                return True
            localized_capitulation_reclaim = (
                score >= 3.10
                and ret60 <= -1.10
                and ret144 <= -2.00
                and ret288 <= -2.00
                and 0.15 <= don <= 0.35
                and 32.0 <= rsi <= 42.0
                and adx >= 25.0
                and down >= 0.75
                and 1.20 <= qvr <= 4.00
                and taker >= 0.35
            )
            if localized_capitulation_reclaim:
                return True
            if symbol == "FILUSDT":
                return bool(
                    score >= 4.0
                    and ret20 <= -1.0
                    and ret60 <= -2.0
                    and rsi <= 40.0
                    and adx >= 35.0
                    and down >= 0.90
                    and taker >= 0.50,
                )
            return bool(
                symbol in {"ETHUSDT", "BCHUSDT", "LTCUSDT"}
                and score >= 3.80
                and ret60 <= -1.0
                and rsi <= 42.0
                and (adx >= 25.0 or qvr >= 1.5)
                and taker >= 0.35,
            )

        if family == "session_pullback_reclaim_long":
            fil_first_bar_chase = bool(
                symbol == "FILUSDT"
                and hour == 0
                and cluster_count <= 1
                and ret20 >= 0.50
                and ret60 >= 1.20
                and rsi >= 50.0
                and don >= 0.55
                and down <= 0.30,
            )
            if fil_first_bar_chase:
                return False
            fil_midnight_second_confirm = bool(
                symbol == "FILUSDT"
                and hour == 0
                and 0.45 <= ret60 <= 0.85
                and ret144 >= 2.0
                and ret288 >= 2.0
                and qvr >= 0.70
                and 42.0 <= rsi <= 48.0
                and down >= 0.70,
            )
            if fil_midnight_second_confirm:
                return True
            paxg_asia_defensive_reclaim = bool(
                symbol == "PAXGUSDT"
                and 4 <= hour <= 5
                and -0.20 <= ret20 <= 0.20
                and -0.60 <= ret60 <= 0.10
                and ret144 >= 0.70
                and ret288 >= 0.80
                and don <= 0.25
                and taker >= 0.65
                and qvr >= 1.0
                and 42.0 <= rsi <= 50.0,
            )
            if paxg_asia_defensive_reclaim:
                return True
            return bool(
                symbol in {"FILUSDT", "LTCUSDT", "PAXGUSDT"}
                and score >= 3.0
                and 42.0 <= rsi <= 54.0
                and ret288 >= 1.0
                and (qvr >= 0.55 or taker >= 0.50 or symbol == "FILUSDT"),
            )

        if family == "range_reclaim_long":
            eth_morning_range_reclaim = bool(
                symbol == "ETHUSDT"
                and 8 <= hour <= 9
                and -0.50 <= ret60 <= 0.10
                and -0.80 <= ret144 <= 0.20
                and -0.30 <= ret288 <= 0.50
                and 0.55 <= don <= 0.75
                and taker >= 0.55
                and qvr <= 1.50
                and 52.0 <= rsi <= 58.0
                and adx >= 25.0,
            )
            if eth_morning_range_reclaim:
                return True
            return bool(
                symbol == "ETHUSDT"
                and 19 <= hour <= 21
                and qvr >= 2.0
                and taker >= 0.60
                and 55.0 <= rsi <= 65.0
                and adx <= 22.0
                and -1.0 <= ret144 <= 0.5,
            )

        if family == "early_session_reclaim_long":
            return bool(
                symbol == "LTCUSDT"
                and hour == 5
                and 0.30 <= ret20 <= 0.60
                and -1.70 <= ret60 <= -1.00
                and ret288 >= 1.20
                and 50.0 <= rsi <= 55.0
                and 0.20 <= don <= 0.45,
            )

        return False

    if action == "OPEN_SHORT":
        if family == "regime_trend_rollover":
            decision = decide_market_regime(row)
            return bool(
                decision.mechanism == "trend_rollover"
                and decision.action == action,
            )
        if family == "trend_exhaustion_short":
            return bool(
                1.00 <= ret20 <= 2.00
                and 1.50 <= ret60 <= 3.00
                and ret144 >= 1.00
                and ret288 >= 0.00
                and 0.74 <= don <= 0.90
                and 62.0 <= rsi <= 72.0
                and adx >= 30.0
                and seller >= 0.55
                and 0.75 <= qvr <= 3.00,
            )
        if family == "eth_anchor_exhaustion_short":
            early_blowoff_onset = (
                asset_group == "eth"
                and cluster_count <= 2
                and cluster_age < 45.0
                and ret20 >= 3.0
                and ret60 >= 3.0
                and don >= 0.93
                and rsi >= 75.0
                and adx >= 40.0
            )
            if early_blowoff_onset:
                return False
            if symbol == "ETHUSDT" and ret60 >= 4.0 and ret20 >= 3.0 and rsi >= 75.0:
                return False
            eth_late_overextended_anchor = (
                symbol == "ETHUSDT"
                and ret60 >= 2.50
                and don >= 0.90
                and qvr >= 5.0
                and rsi >= 75.0
            )
            if eth_late_overextended_anchor:
                return False
            precision_anchor = bool(
                symbol == "ETHUSDT"
                and (
                    (
                        1.35 <= ret60 <= 3.85
                        and ret20 <= 2.0
                        and 68.0 <= rsi <= 73.0
                        and don >= 0.82
                        and adx >= 25.0
                        and 1.40 <= qvr <= 6.0
                    )
                    or (
                        ret20 >= 3.0
                        and ret60 >= 2.50
                        and 69.0 <= rsi <= 78.0
                        and don >= 0.85
                        and adx >= 45.0
                        and qvr <= 2.0
                    )
                    or (
                        ret288 <= -3.0
                        and 0.15 <= ret20 <= 1.80
                        and -0.65 <= ret60 <= 0.75
                        and ret144 >= 0.35
                        and don >= 0.78
                        and 58.0 <= rsi <= 64.0
                        and adx >= 28.0
                    )
                    or (
                        0.45 <= ret60 <= 0.95
                        and ret144 >= 1.25
                        and ret288 >= 1.0
                        and 0.68 <= don <= 0.82
                        and 49.0 <= rsi <= 55.0
                        and adx >= 30.0
                        and seller >= 0.50
                        and qvr <= 1.30
                    )
                    or (
                        ret288 >= 4.0
                        and ret60 >= 4.0
                        and -0.50 <= ret20 <= 0.50
                        and don >= 0.85
                        and 62.0 <= rsi <= 70.0
                        and adx >= 45.0
                        and seller >= 0.55
                        and qvr <= 1.0
                    )
                ),
            )
            if precision_anchor:
                return True
            high_adx_mid_cluster_anchor = bool(
                symbol == "ETHUSDT"
                and score >= 3.00
                and 2.30 <= ret60 <= 3.10
                and ret288 >= 4.50
                and 58.0 <= rsi <= 63.0
                and don >= 0.74
                and adx >= 45.0
                and qvr >= 0.80,
            )
            if high_adx_mid_cluster_anchor:
                return True
            mature_cluster_anchor = bool(
                asset_group == "eth"
                and score >= 3.00
                and cluster_count >= 4
                and 90.0 <= cluster_age <= 180.0
                and 2.40 <= ret60 <= 3.50
                and 58.0 <= rsi <= 62.0
                and don >= 0.75
                and adx >= 30.0
                and qvr <= 2.0,
            )
            if mature_cluster_anchor:
                return True
            mid_cluster_anchor = bool(
                symbol == "ETHUSDT"
                and score >= 3.00
                and event_score >= 0.020
                and 2.40 <= ret60 <= 3.50
                and 58.0 <= rsi <= 62.0
                and don >= 0.72
                and adx >= 30.0
                and qvr <= 10.0,
            )
            if mid_cluster_anchor:
                return True
            return bool(
                symbol == "ETHUSDT"
                and score >= 3.60
                and adx >= 32.0
                and (
                    (rsi >= 78.0 and ret60 >= 1.0 and qvr >= 8.0)
                    or (event_score >= 0.025 and ret60 >= 2.5 and rsi >= 58.0 and don >= 0.80)
                    or (qvr >= 20.0 and rsi >= 80.0)
                    or (qvr >= 18.0 and rsi >= 75.0 and ret60 >= 1.40 and ret288 >= 5.0 and don >= 0.75)
                ),
            )

        if family == "weak_24h_bounce_short":
            if symbol == "ETHUSDT":
                return bool(
                    0.60 <= ret60 <= 1.80
                    and ret144 <= -1.50
                    and ret288 <= -1.50
                    and 0.65 <= don <= 0.85
                    and taker >= 0.58
                    and qvr >= 1.0
                    and 55.0 <= rsi <= 62.0,
                )
            return bool(
                symbol == "DOGEUSDT"
                and (score >= 3.60 or (hour == 18 and ret144 <= -1.50 and ret288 >= 1.20 and qvr >= 0.80))
                and 52.0 <= rsi <= 62.0
                and ret144 <= -1.0
                and ret288 >= 1.0
                and taker >= 0.60,
            )

        if family == "paxg_defensive_short":
            return bool(
                symbol == "PAXGUSDT"
                and score >= 5.0
                and qvr >= 2.0
                and 50.0 <= rsi <= 60.0
                and down >= 0.75,
            )

        if family == "core_downtrend_short":
            if symbol == "PAXGUSDT":
                return bool(
                    20 <= hour <= 21
                    and score >= 2.85
                    and qvr >= 3.0
                    and rsi <= 35.0
                    and down >= 0.75
                    and don <= 0.35,
                )
            if symbol == "ETHUSDT":
                eth_weak_breakdown_chase = (
                    ret60 <= -3.00
                    and ret144 <= -2.50
                    and don <= 0.30
                    and rsi <= 48.0
                    and qvr < 1.20
                    and down >= 0.50
                )
                if eth_weak_breakdown_chase:
                    return False
                return bool(
                    ret60 <= -2.50
                    and ret144 <= -2.00
                    and ret288 <= -2.00
                    and don <= 0.45
                    and 45.0 <= rsi <= 57.0
                    and adx >= 25.0,
                )
            if symbol == "FILUSDT":
                return False
            if symbol in {"BCHUSDT", "ETCUSDT"}:
                etc_first_unconfirmed = bool(
                    symbol == "ETCUSDT"
                    and hour == 6
                    and ret60 > -0.05
                    and qvr < 1.0
                    and ret144 > 0.50,
                )
                return bool(
                    ((not etc_first_unconfirmed)
                    and (
                        hour in {2, 6}
                        and (
                            score >= 2.8
                            or (
                                symbol == "BCHUSDT"
                                and hour == 2
                                and -0.80 <= ret20 <= -0.30
                                and -0.10 <= ret60 <= 0.35
                                and 0.35 <= don <= 0.50
                                and seller >= 0.53
                            )
                        )
                        and 40.0 <= rsi <= 48.0
                        and ret288 >= 2.0
                    ))
                    or (
                        symbol == "ETCUSDT"
                        and 7 <= hour <= 8
                        and score >= 2.65
                        and don <= 0.35
                        and qvr >= 0.70
                        and 34.0 <= rsi <= 46.0
                        and down >= 0.75
                    ),
                )
            if symbol == "CFXUSDT":
                cfx_evening_crowded_short_chase = (
                    19 <= hour <= 21
                    and qvr >= 2.0
                    and 27.0 <= rsi <= 42.0
                    and down >= 0.85
                    and ret20 > -1.50
                    and not (seller >= 0.80 and ret144 > -0.50)
                )
                if cfx_evening_crowded_short_chase:
                    return False
                cfx_session_false_breakdown = (
                    (hour == 22 and ret144 <= -1.20 and qvr <= 5.0 and rsi <= 33.0)
                    or (hour == 6 and ret60 <= -3.00 and qvr < 3.0)
                    or (hour == 6 and ret144 >= 0.50 and ret288 >= 1.00 and qvr <= 1.50)
                    or (hour == 16 and ret144 <= -1.00 and seller <= 0.55 and qvr < 3.0)
                )
                if cfx_session_false_breakdown:
                    return False
                return bool(
                    (score >= 3.8 and ret60 <= -0.80 and rsi <= 41.0 and down >= 0.85)
                    or (
                        8 <= hour <= 9
                        and score >= 2.80
                        and -0.60 <= ret20 <= 0.05
                        and ret60 <= 0.30
                        and don <= 0.20
                        and qvr >= 2.0
                        and rsi <= 36.0
                        and down >= 0.85
                    ),
                )
            if symbol == "LTCUSDT":
                return bool(
                    18 <= hour <= 19
                    and ret144 <= -2.0
                    and -0.6 <= ret60 <= 0.5
                    and seller >= 0.65
                    and qvr >= 0.8
                    and 42.0 <= rsi <= 52.0
                    and adx >= 16.0,
                )
            if symbol == "DOGEUSDT":
                return bool(
                    (
                        18 <= hour <= 20
                        and qvr >= 4.0
                        and rsi <= 38.0
                        and adx >= 30.0
                        and seller >= 0.60
                    )
                    or (
                        (hour > 22 or (hour == 22 and not pd.isna(signal_time) and int(signal_time.minute) >= 30))
                        and ret60 <= -1.2
                        and ret144 <= -2.0
                        and ret288 <= -1.0
                        and rsi <= 55.0
                    ),
                )
            if symbol == "SOLUSDT":
                return bool(
                    (
                        19 <= hour <= 21
                        and ret20 <= -0.80
                        and ret60 <= -1.0
                        and ret144 <= -2.0
                        and ret288 <= -0.75
                        and qvr >= 4.0
                        and adx >= 25.0
                        and rsi <= 45.0
                        and (down >= 0.90 or (hour == 20 and qvr >= 20.0 and rsi <= 25.0 and down >= 0.80))
                    )
                    or (
                        19 <= hour <= 20
                        and -0.10 <= ret20 <= 0.20
                        and -0.10 <= ret60 <= 0.20
                        and ret144 <= -1.20
                        and don <= 0.38
                        and qvr >= 4.0
                        and rsi <= 42.0
                        and down >= 0.70
                    ),
                )
            return False

        return False

    return False


def _format_family_score_floors(family_score_floors: tuple[tuple[str, float], ...]) -> str:
    return ",".join(f"{family}={float(score):.6g}" for family, score in family_score_floors)


def _event_leg_budget(
    event: dict[str, Any],
    prior: LegPrior,
    *,
    default_cap: int,
    hard_cap: int,
    recall_first: bool = False,
) -> int:
    if recall_first:
        return max(1, int(hard_cap))
    action = str(event.get("action") or "")
    family = str(event.get("dominant_family") or "")
    budget = prior.legs_by_action_family.get((action, family))
    if budget is None:
        budget = prior.default_legs_by_action.get(action, 1)
    unique_symbols = int(_finite_float(event.get("unique_symbols"), budget))
    pressure_selected = int(_finite_float(event.get("pressure_gate_selected"), 0.0)) == 1
    session_selected = int(_finite_float(event.get("close_session_gate_selected"), 0.0)) == 1
    if str(family).startswith("portfolio_") or session_selected:
        budget = max(budget, min(default_cap, unique_symbols))
    elif pressure_selected and unique_symbols >= 4:
        budget = max(budget, min(default_cap, int(np.ceil(unique_symbols / 2.0))))
    return max(1, min(int(hard_cap), int(budget)))


def _production_open_leg_budget(event: dict[str, Any], *, current_budget: int, hard_cap: int) -> int:
    """Selective live-safe budget expansion for clustered production opens."""

    family = str(event.get("dominant_family") or "")
    action = str(event.get("action") or "")
    session = str(event.get("session_family") or "")
    unique_symbols = int(_finite_float(event.get("unique_symbols"), 1.0))
    pressure = _finite_float(event.get("event_pressure"), 0.0)
    max_score = _finite_float(event.get("max_score"), 0.0)
    has_eth = _finite_float(event.get("has_eth"), 0.0) > 0.0
    market_down = _finite_float(event.get("market_down"), 0.0)
    hour = int(_finite_float(event.get("hour"), -1.0))
    pressure_selected = int(_finite_float(event.get("pressure_gate_selected"), 0.0)) == 1
    budget = max(1, int(current_budget))

    if action == "OPEN_LONG":
        cluster_family = family in {
            "core_trend_long",
            "midnight_breadth_breakout_long",
            "capitulation_absorption_long",
            "session_pullback_reclaim_long",
            "early_session_reclaim_long",
        }
        session_cluster = session in {"midnight", "asia_morning", "night_trend"}
        if family == "capitulation_absorption_long" and (pressure >= 10.0 or max_score >= 3.1):
            budget = max(budget, min(unique_symbols, 3))
        if cluster_family and unique_symbols >= 2 and (pressure_selected or pressure >= 14.0 or session_cluster):
            budget = max(budget, min(unique_symbols, hard_cap))
        if family == "range_reclaim_long" and session == "regular" and max_score >= 2.5:
            budget = max(budget, min(unique_symbols, 2))

    if action == "OPEN_SHORT":
        cluster_family = family in {
            "core_downtrend_short",
            "eth_anchor_exhaustion_short",
            "trend_exhaustion_short",
            "weak_24h_bounce_short",
            "paxg_defensive_short",
        }
        if cluster_family and unique_symbols >= 2 and (pressure_selected or pressure >= 18.0 or session == "night_trend"):
            budget = max(budget, min(unique_symbols, hard_cap))
        if family == "core_downtrend_short" and has_eth and unique_symbols >= 4 and pressure >= 16.0:
            budget = max(budget, min(unique_symbols, hard_cap))
        if (
            family == "core_downtrend_short"
            and 20 <= hour <= 21
            and unique_symbols >= 3
            and max_score >= 3.0
            and (market_down >= 0.75 or pressure >= 10.0)
        ):
            budget = max(budget, min(unique_symbols, hard_cap))
        if family == "weak_24h_bounce_short" and has_eth and unique_symbols >= 2 and pressure >= 12.0:
            budget = max(budget, min(unique_symbols, hard_cap))
        if family == "paxg_defensive_short" and (session == "night_trend" or pressure >= 8.0):
            budget = max(budget, min(unique_symbols, 2))

    return max(1, min(int(hard_cap), int(budget)))


def apply_deferred_close_event_persistence(
    events: pd.DataFrame,
    *,
    max_gap_minutes: float = 60.0,
) -> pd.DataFrame:
    """Carry an original ETH close event into its immediate stall-confirmation window."""

    if events.empty:
        return events.copy()
    out = events.copy()
    out["event_bucket"] = pd.to_datetime(out.get("event_bucket"), errors="coerce")
    original_selected = pd.to_numeric(out.get("selected_event"), errors="coerce").fillna(0).astype(int).eq(1)
    out["deferred_close_persistence_selected"] = 0
    max_gap = pd.Timedelta(minutes=max(0.0, float(max_gap_minutes)))
    original_rows = out.loc[original_selected].copy()
    for row_index, row in out.loc[~original_selected].iterrows():
        action = str(row.get("action") or "")
        if action != "CLOSE_LONG" or _finite_float(row.get("has_eth"), 0.0) <= 0.0:
            continue
        family = str(row.get("dominant_family") or "")
        if family not in {"long_profit_take_exhaustion", "long_relief_bounce_exit"}:
            continue
        event_time = pd.Timestamp(row.get("event_bucket"))
        prior = original_rows[
            original_rows.get("action", pd.Series("", index=original_rows.index)).astype(str).eq(action)
            & pd.to_numeric(original_rows.get("has_eth"), errors="coerce").fillna(0.0).gt(0.0)
        ].copy()
        if prior.empty:
            continue
        gap = event_time - pd.to_datetime(prior["event_bucket"], errors="coerce")
        if ((gap > pd.Timedelta(0)) & (gap <= max_gap)).any():
            out.at[row_index, "selected_event"] = 1
            out.at[row_index, "deferred_close_persistence_selected"] = 1
    return out


def _production_close_leg_budget(
    event: dict[str, Any],
    *,
    current_budget: int,
    hard_cap: int,
    deferred_persistence: bool = False,
) -> int:
    """Selective live-safe budget expansion for clustered production closes."""

    family = str(event.get("dominant_family") or "")
    session = str(event.get("session_family") or "")
    unique_symbols = int(_finite_float(event.get("unique_symbols"), 1.0))
    signal_count = int(_finite_float(event.get("signal_count"), unique_symbols))
    pressure = _finite_float(event.get("event_pressure"), 0.0)
    pressure_selected = int(_finite_float(event.get("pressure_gate_selected"), 0.0)) == 1
    session_cluster = session in {"midnight", "asia_morning", "late_morning_cleanup", "evening_cleanup"}
    budget = max(1, int(current_budget))
    if deferred_persistence and int(_finite_float(event.get("deferred_close_persistence_selected"), 0.0)) == 1:
        budget = max(budget, min(signal_count, hard_cap))
    if family.startswith("portfolio_") or session_cluster:
        budget = max(budget, min(signal_count, hard_cap))
    elif family == "long_profit_take_exhaustion" and unique_symbols >= 2 and pressure >= 25.0:
        budget = max(budget, min(unique_symbols, hard_cap))
    elif (family in {"short_profit_take_exhaustion", "long_risk_cut_reversal"} and (
        pressure_selected or pressure >= 80.0
    )) or (family == "long_risk_cut_reversal" and session == "regular" and pressure >= 60.0 and unique_symbols >= 4) or (session == "night_trend" and pressure >= 120.0):
        budget = max(budget, min(signal_count, hard_cap))
    return max(1, min(int(hard_cap), int(budget)))


def _production_over_cap_open_allowed(row: pd.Series, *, lot_count: int, max_lots: int) -> bool:
    max_lots_int = max(1, int(max_lots))
    if int(lot_count) != max_lots_int:
        return False
    signal_time = pd.Timestamp(row.get("signal_time")) if row.get("signal_time") is not None else pd.NaT
    return bool(
        str(row.get("symbol") or "") == "FILUSDT"
        and str(row.get("action") or "") == "OPEN_LONG"
        and str(row.get("reason_family") or "") == "session_pullback_reclaim_long"
        and pd.notna(signal_time)
        and int(signal_time.hour) == 0
        and _finite_float(row.get("ret60"), 0.0) >= 1.0
        and _finite_float(row.get("ret144"), 0.0) >= 2.0
        and _finite_float(row.get("ret288"), 0.0) >= 1.50
        and _finite_float(row.get("qvr"), 0.0) >= 1.50
        and 42.0 <= _finite_float(row.get("rsi"), 50.0) <= 46.0
        and _finite_float(row.get("down"), 0.0) >= 0.70,
    )


def _production_reentry_gap_minutes(row: pd.Series, *, config: WeijieComboEventStrategyConfig) -> float:
    base_gap = max(0.0, float(config.production_min_reentry_gap_minutes))
    if base_gap <= 0.0:
        return 0.0
    symbol = str(row.get("symbol") or "")
    action = str(row.get("action") or "")
    family = str(row.get("reason_family") or "")
    if symbol == "ETHUSDT":
        return min(base_gap, 360.0)
    if family == "eth_anchor_exhaustion_short":
        return min(base_gap, 360.0)
    if symbol == "DOGEUSDT" and action == "OPEN_SHORT":
        return min(base_gap, 240.0)
    if symbol in {"PAXGUSDT", "FILUSDT"} and action == "OPEN_LONG":
        return min(base_gap, 720.0)
    return base_gap


def _production_state_open_cap(row: pd.Series, *, config: WeijieComboEventStrategyConfig) -> int:
    symbol = str(row.get("symbol") or "")
    if symbol == "ETHUSDT":
        return max(0, int(config.production_eth_state_open_cap))
    return max(0, int(config.production_alt_state_open_cap))


def _production_portfolio_correlation_cap(
    row: pd.Series | dict[str, Any],
    *,
    active: dict[tuple[str, str], list[dict[str, Any]]],
    config: WeijieComboEventStrategyConfig,
    base_cap: float,
) -> float:
    cap = float(base_cap)
    cluster_cap = max(0.0, float(config.production_clustered_meme_short_correlation_cap))
    same_direction_states = sum(
        1 for (_, direction), lots in active.items() if direction == "SHORT" and len(lots) > 0
    )
    if (
        cluster_cap > cap
        and weijie_asset_group(str(row.get("symbol") or "")) == "meme"
        and str(row.get("action") or "") == "OPEN_SHORT"
        and str(row.get("reason_family") or "") == "core_downtrend_short"
        and same_direction_states >= 3
    ):
        return cluster_cap
    return cap


def _production_open_execution_time(
    row: pd.Series | dict[str, Any],
    *,
    config: WeijieComboEventStrategyConfig,
) -> pd.Timestamp:
    signal_time = pd.Timestamp(row.get("signal_time"))
    delay = max(0.0, float(config.production_low_quality_meme_short_delay_minutes))
    quality = _finite_float(row.get("factor_quality_score"), np.nan)
    quality_floor = _finite_float(row.get("factor_quality_floor"), np.nan)
    if (
        delay > 0.0
        and weijie_asset_group(str(row.get("symbol") or "")) == "meme"
        and str(row.get("action") or "") == "OPEN_SHORT"
        and str(row.get("reason_family") or "") == "core_downtrend_short"
        and np.isfinite(quality)
        and np.isfinite(quality_floor)
        and quality < quality_floor
    ):
        return signal_time + pd.Timedelta(minutes=delay)
    return signal_time


def _dynamic_stop_distance_bounds(
    row: pd.Series | dict[str, Any],
    *,
    config: WeijieComboEventStrategyConfig,
    min_stop_distance_pct: float,
    max_stop_distance_pct: float,
) -> tuple[float, float]:
    minimum = max(0.0, float(min_stop_distance_pct))
    maximum = max(minimum, float(max_stop_distance_pct))
    if int(_finite_float(row.get("non_eth_contribution_campaign"), 0.0)) == 1:
        # Campaign sizing caps leverage at 6x, so this remains below the shared
        # 60% margin-loss guard while allowing a multi-day pullback.
        minimum = max(minimum, 8.50)
        maximum = max(maximum, minimum)
    family_floor = max(0.0, float(config.production_meme_core_short_stop_floor_pct))
    if (
        family_floor > 0.0
        and weijie_asset_group(str(row.get("symbol") or "")) == "meme"
        and str(row.get("action") or "") == "OPEN_SHORT"
        and str(row.get("reason_family") or "") == "core_downtrend_short"
    ):
        minimum = max(minimum, family_floor)
        maximum = max(maximum, minimum)
    return minimum, maximum


def _production_max_lots_for_symbol(
    symbol: str,
    *,
    config: WeijieComboEventStrategyConfig,
) -> int:
    asset = weijie_asset_group(symbol)
    configured = {
        "eth": config.production_eth_max_lots_per_state,
        "stock_gold": config.production_stock_gold_max_lots_per_state,
        "meme": config.production_meme_max_lots_per_state,
    }.get(asset, 0)
    return max(1, int(configured or config.production_max_lots_per_state))


def _filter_meme_core_trend_confirmation_candidates(
    candidates: pd.DataFrame,
    *,
    active: dict[tuple[str, str], list[dict[str, Any]]],
    event: dict[str, Any] | None = None,
    allow_breadth_bootstrap: bool = False,
) -> pd.DataFrame:
    """Use meme core-trend legs as confirmation, not as cold-start entries."""

    if candidates.empty:
        return candidates.copy()
    event = event or {}
    breadth_bootstrap = bool(
        allow_breadth_bootstrap
        and _finite_float(event.get("unique_symbols"), 0.0) >= 5.0
        and _finite_float(event.get("signal_count"), 0.0) >= 5.0
        and _finite_float(event.get("event_pressure"), 0.0) >= 20.0
        and _finite_float(event.get("max_score"), 0.0) >= 4.0
        and _finite_float(event.get("market_up"), 0.0) >= 0.40
        and _finite_float(event.get("cluster_onset_fraction"), 0.0) >= 0.15,
    )
    keep: list[bool] = []
    for _, row in candidates.iterrows():
        symbol = str(row.get("symbol") or "")
        action = str(row.get("action") or "")
        family = str(row.get("reason_family") or "")
        confirmation_only = (
            weijie_asset_group(symbol) == "meme"
            and action == "OPEN_LONG"
            and family == "core_trend_long"
        )
        state_key = (symbol, direction_from_action(action))
        contribution_campaign = int(
            _finite_float(row.get("non_eth_contribution_campaign"), 0.0),
        ) == 1
        keep.append(
            not confirmation_only
            or contribution_campaign
            or breadth_bootstrap
            or _active_lot_count(active, state_key) > 0,
        )
    return candidates.loc[pd.Series(keep, index=candidates.index)].copy()


def _filter_non_eth_contribution_campaign_candidates(
    candidates: pd.DataFrame,
    event: dict[str, Any],
    *,
    eligible_symbols: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Keep causal non-ETH campaign structures without symbol/date replay."""

    if candidates.empty:
        return candidates.copy()
    out = candidates.copy()
    symbol = out.get("symbol", pd.Series("", index=out.index)).astype(str)
    action = out.get("action", pd.Series("", index=out.index)).astype(str)
    family = out.get("reason_family", pd.Series("", index=out.index)).astype(str)
    ret20 = pd.to_numeric(out.get("ret20"), errors="coerce").fillna(0.0)
    ret60 = pd.to_numeric(out.get("ret60"), errors="coerce").fillna(0.0)
    ret144 = pd.to_numeric(out.get("ret144"), errors="coerce").fillna(0.0)
    ret288 = pd.to_numeric(out.get("ret288"), errors="coerce").fillna(0.0)
    don = pd.to_numeric(out.get("don"), errors="coerce").fillna(0.5)
    taker = pd.to_numeric(out.get("taker"), errors="coerce").fillna(0.5)
    qvr = pd.to_numeric(out.get("qvr"), errors="coerce").fillna(0.0)
    rsi = pd.to_numeric(out.get("rsi"), errors="coerce").fillna(50.0)
    quality = pd.to_numeric(out.get("factor_quality_score"), errors="coerce").fillna(-np.inf)
    quality_floor = pd.to_numeric(out.get("factor_quality_floor"), errors="coerce").replace(0.0, np.nan)
    quality_ratio = quality.div(quality_floor).replace([np.inf, -np.inf], np.nan).fillna(-np.inf)
    asset_group = symbol.map(weijie_asset_group)

    broad_risk_on = bool(
        _finite_float(event.get("event_score"), 0.0) >= 0.85
        and _finite_float(event.get("event_pressure"), 0.0) >= 15.0
        and _finite_float(event.get("signal_count"), 0.0) >= 5.0
        and _finite_float(event.get("unique_symbols"), 0.0) >= 5.0
        and _finite_float(event.get("market_up"), 0.0) >= 0.50
        and _finite_float(event.get("mean_ret144"), 0.0) >= 0.0
        and _finite_float(event.get("cluster_mature_fraction"), 0.0) >= 0.50,
    )
    trend_campaign = (
        asset_group.eq("meme")
        & action.eq("OPEN_LONG")
        & family.eq("core_trend_long")
        & ret20.between(-0.40, 0.90, inclusive="both")
        & ret60.between(0.35, 4.30, inclusive="both")
        & ret144.ge(-0.25)
        & ret288.ge(1.00)
        & don.between(0.60, 0.98, inclusive="both")
        & rsi.between(50.0, 66.0, inclusive="both")
        & quality_ratio.ge(1.15)
        & broad_risk_on
        & (_finite_float(event.get("cluster_onset_fraction"), 0.0) >= 0.10)
    )
    absorption_campaign = (
        action.eq("OPEN_LONG")
        & family.eq("capitulation_absorption_long")
        & ret60.between(-2.50, -0.80, inclusive="both")
        & ret144.le(-2.00)
        & ret288.le(-2.00)
        & don.le(0.30)
        & rsi.between(28.0, 42.0, inclusive="both")
        & qvr.between(0.50, 4.00, inclusive="both")
        & quality_ratio.ge(1.20)
        & (_finite_float(event.get("market_down"), 0.0) >= 0.75)
    )
    defensive_reclaim = (
        asset_group.eq("stock_gold")
        & action.eq("OPEN_LONG")
        & family.eq("session_pullback_reclaim_long")
        & ret60.between(-0.80, 0.30, inclusive="both")
        & ret288.ge(0.50)
        & don.between(0.10, 0.45, inclusive="both")
        & taker.ge(0.58)
        & qvr.between(0.40, 3.00, inclusive="both")
        & rsi.between(40.0, 52.0, inclusive="both")
        & quality_ratio.ge(1.10)
    )
    early_reclaim_campaign = (
        action.eq("OPEN_LONG")
        & family.eq("early_session_reclaim_long")
        & ret20.between(0.20, 0.70, inclusive="both")
        & ret60.between(-2.00, -0.80, inclusive="both")
        & ret144.between(-1.50, -0.20, inclusive="both")
        & ret288.ge(1.00)
        & don.between(0.20, 0.45, inclusive="both")
        & taker.ge(0.58)
        & qvr.between(0.40, 2.00, inclusive="both")
        & rsi.between(48.0, 56.0, inclusive="both")
        & quality_ratio.ge(1.10)
        & (_finite_float(event.get("event_score"), 0.0) >= 0.75)
        & (_finite_float(event.get("market_up"), 0.0) >= 0.65)
    )
    weak_bounce_short = (
        action.eq("OPEN_SHORT")
        & family.eq("weak_24h_bounce_short")
        & ret60.between(-0.20, 1.00, inclusive="both")
        & ret144.le(-1.00)
        & ret288.between(-0.50, 2.50, inclusive="both")
        & don.between(0.45, 0.75, inclusive="both")
        & rsi.between(48.0, 62.0, inclusive="both")
        & qvr.between(0.40, 2.50, inclusive="both")
        & quality_ratio.ge(1.20)
        & (_finite_float(event.get("event_score"), 0.0) >= 0.85)
    )
    campaign = (
        trend_campaign
        | absorption_campaign
        | defensive_reclaim
        | early_reclaim_campaign
        | weak_bounce_short
    ) & ~symbol.eq("ETHUSDT")
    reviewed_symbols = {str(item).upper() for item in eligible_symbols if str(item).strip()}
    if reviewed_symbols:
        campaign &= symbol.isin(reviewed_symbols)
    keep = symbol.eq("ETHUSDT") | campaign
    out["non_eth_contribution_campaign"] = campaign.astype(int)
    return out.loc[keep].copy()


def _open_meme_leg_limits(
    event: pd.Series | dict[str, Any],
    *,
    config: WeijieComboEventStrategyConfig,
) -> tuple[int, float]:
    cap = max(0, int(config.production_meme_max_legs_per_event))
    quality_floor = max(0.0, float(config.production_meme_second_leg_quality_ratio_floor))
    if (
        config.production_broad_cluster_third_meme_leg
        and _finite_float(event.get("signal_count"), 0.0) >= 6.0
        and _finite_float(event.get("event_score"), 0.0) >= 0.95
    ):
        cap = max(cap, 2)
        quality_floor = min(quality_floor, 1.0)
    return cap, quality_floor


def _production_addon_gap_minutes(
    row: pd.Series | dict[str, Any],
    *,
    config: WeijieComboEventStrategyConfig,
) -> float:
    gap = max(0.0, float(config.production_min_addon_gap_minutes))
    family = str(row.get("reason_family") or "")
    symbol = str(row.get("symbol") or "")
    action = str(row.get("action") or "")
    signal_time = pd.to_datetime(row.get("signal_time"), errors="coerce")
    if symbol == "ETHUSDT" and action == "OPEN_SHORT" and family == "eth_anchor_exhaustion_short":
        if config.production_eth_blowoff_ladder and _eth_blowoff_ladder_open_confirmed(row):
            gap = min(gap, 10.0)
        elif config.production_eth_blowoff_handoff:
            gap = min(gap, 60.0)
    if (
        symbol == "ETHUSDT"
        and action == "OPEN_LONG"
        and family in {"core_trend_long", "eth_anchor_long", "range_reclaim_long"}
        and pd.notna(signal_time)
        and int(signal_time.hour) in {0, 6, 8, 15}
    ):
        gap = min(gap, 60.0)
    if symbol in {"PAXGUSDT", "FILUSDT"} and action == "OPEN_LONG" and family == "session_pullback_reclaim_long":
        gap = min(gap, 90.0)
    if (
        symbol == "FILUSDT"
        and action == "OPEN_LONG"
        and family == "core_trend_long"
        and pd.notna(signal_time)
        and int(signal_time.hour) == 3
        and _finite_float(row.get("ret60"), 0.0) >= 3.0
        and _finite_float(row.get("ret288"), 0.0) >= 4.0
    ):
        gap = min(gap, 60.0)
    if symbol == "LTCUSDT" and action == "OPEN_SHORT" and family == "core_downtrend_short":
        gap = min(gap, 90.0)
    if (
        symbol == "LTCUSDT"
        and action == "OPEN_LONG"
        and family == "core_trend_long"
        and _finite_float(row.get("qvr"), 0.0) >= 8.0
        and _finite_float(row.get("rsi"), 0.0) >= 68.0
        and _finite_float(row.get("don"), 0.0) >= 0.85
        and _finite_float(row.get("ret60"), 0.0) >= 0.70
    ):
        gap = min(gap, 60.0)
    return gap


def _filter_ineligible_open_state_candidates(
    candidates: pd.DataFrame,
    *,
    active: dict[tuple[str, str], list[dict[str, Any]]],
    config: WeijieComboEventStrategyConfig,
) -> pd.DataFrame:
    """Remove legs that cannot consume an open-event budget slot."""

    if candidates.empty:
        return candidates.copy()
    keep: list[bool] = []
    for _, row in candidates.iterrows():
        symbol = str(row.get("symbol") or "")
        direction = direction_from_action(row.get("action"))
        lot_count = _active_lot_count(active, (symbol, direction))
        if lot_count <= 0:
            keep.append(True)
            continue
        if not config.production_allow_pyramiding:
            keep.append(False)
            continue
        max_lots = _production_max_lots_for_symbol(symbol, config=config)
        if lot_count >= max_lots and not _production_over_cap_open_allowed(
            row,
            lot_count=lot_count,
            max_lots=max_lots,
        ):
            keep.append(False)
            continue
        last_open_time = _active_last_open_time(active, (symbol, direction))
        if last_open_time is not None and float(config.production_min_addon_gap_minutes) > 0.0:
            gap_minutes = (
                pd.Timestamp(row.get("signal_time")) - pd.Timestamp(last_open_time)
            ).total_seconds() / 60.0
            if gap_minutes < _production_addon_gap_minutes(row, config=config):
                keep.append(False)
                continue
        keep.append(True)
    return candidates.loc[pd.Series(keep, index=candidates.index)].copy()


def _select_event_legs(
    candidates: pd.DataFrame,
    prior: LegPrior,
    *,
    leg_budget: int,
    active: dict[tuple[str, str], list[dict[str, Any]]],
    for_close: bool,
    recall_first: bool = False,
    allow_open_addons: bool = False,
    max_lots_per_state: int = 1,
    max_lots_by_asset_group: dict[str, int] | None = None,
    max_open_legs_by_asset_group: dict[str, int] | None = None,
    meme_second_leg_quality_ratio_floor: float = 0.0,
    unified_open_ranking: bool = False,
    eth_cascade_onset_ranking: bool = False,
) -> pd.DataFrame:
    if candidates.empty or leg_budget <= 0:
        return candidates.head(0).copy()
    frame = candidates.copy()
    frame["leg_score"] = [
        _leg_score(row, prior=prior, active=active, for_close=for_close) for _, row in frame.iterrows()
    ]
    if for_close:
        frame["selection_score"] = pd.to_numeric(frame["leg_score"], errors="coerce").fillna(0.0)
        if eth_cascade_onset_ranking:
            cascade = (
                frame.get("symbol", pd.Series("", index=frame.index)).astype(str).eq("ETHUSDT")
                & frame.get("action", pd.Series("", index=frame.index)).astype(str).eq("CLOSE_SHORT")
                & frame.get("reason_family", pd.Series("", index=frame.index))
                .astype(str)
                .eq("short_profit_take_exhaustion")
                & pd.to_numeric(frame.get("ret20"), errors="coerce").le(-1.50)
                & pd.to_numeric(frame.get("ret60"), errors="coerce").le(-1.50)
                & pd.to_numeric(frame.get("ret288"), errors="coerce").le(-3.00)
                & pd.to_numeric(frame.get("don"), errors="coerce").le(0.20)
                & pd.to_numeric(frame.get("qvr"), errors="coerce").ge(3.00)
                & pd.to_numeric(frame.get("rsi"), errors="coerce").le(30.0)
                & pd.to_numeric(frame.get("seller"), errors="coerce").ge(0.52)
            )
            if cascade.any():
                cascade_times = pd.to_datetime(frame.loc[cascade, "signal_time"], errors="coerce")
                onset = cascade_times.min()
                delay_bars = cascade_times.sub(onset).dt.total_seconds().div(300.0).fillna(0.0)
                frame.loc[cascade, "selection_score"] = 100.0 - delay_bars
    elif unified_open_ranking:
        quality = pd.to_numeric(
            frame.get("factor_quality_score", pd.Series(0.0, index=frame.index)),
            errors="coerce",
        ).fillna(0.0)
        quality_floor = pd.to_numeric(
            frame.get("factor_quality_floor", pd.Series(0.0, index=frame.index)),
            errors="coerce",
        ).replace(0.0, np.nan)
        quality_ratio = quality.div(quality_floor).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        raw_score = pd.to_numeric(frame.get("score"), errors="coerce").fillna(0.0)
        frame["leg_score"] = 0.0
        frame["selection_score"] = quality_ratio.mul(2.0) + raw_score.mul(0.10)
    else:
        point_accuracy_score = (
            pd.to_numeric(frame["point_accuracy_score"], errors="coerce").fillna(0.0)
            if "point_accuracy_score" in frame
            else pd.Series(0.0, index=frame.index, dtype=float)
        )
        frame["selection_score"] = (
            pd.to_numeric(frame["leg_score"], errors="coerce").fillna(0.0)
            + point_accuracy_score
        )
        if "factor_quality_score" in frame:
            frame["selection_score"] += (
                pd.to_numeric(frame["factor_quality_score"], errors="coerce").fillna(0.0) * 0.45
            )
    if "leader_leg_quality_probability" in frame:
        frame["selection_score"] += (
            pd.to_numeric(frame["leader_leg_quality_probability"], errors="coerce").fillna(0.0) * 2.0
        )
    frame = frame.sort_values(["selection_score", "leg_score", "score", "signal_time", "symbol"], ascending=[False, False, False, True, True])
    if for_close:
        frame["_symbol_candidate_rank"] = frame.groupby("symbol", sort=False).cumcount()
        frame = frame.sort_values(
            ["_symbol_candidate_rank", "selection_score", "leg_score", "score", "signal_time", "symbol"],
            ascending=[True, False, False, False, True, True],
        )
    if recall_first:
        return frame.head(int(leg_budget)).copy()
    kept: list[int] = []
    used_symbols: set[str] = set()
    used_asset_groups: dict[str, int] = {}
    for idx, row in frame.iterrows():
        symbol = str(row.get("symbol") or "")
        asset_group = weijie_asset_group(symbol)
        direction = direction_from_action(row.get("action"))
        state_key = (symbol, direction)
        lot_count = _active_lot_count(active, state_key)
        asset_max_lots = max(1, int((max_lots_by_asset_group or {}).get(weijie_asset_group(symbol), max_lots_per_state)))
        if for_close and lot_count <= 0:
            continue
        if not for_close and lot_count > 0 and not allow_open_addons:
            continue
        if (
            not for_close
            and lot_count >= asset_max_lots
            and not _production_over_cap_open_allowed(row, lot_count=lot_count, max_lots=asset_max_lots)
        ):
            continue
        if symbol in used_symbols and not for_close:
            continue
        asset_leg_cap = int((max_open_legs_by_asset_group or {}).get(asset_group, 0))
        if not for_close and asset_leg_cap > 0 and used_asset_groups.get(asset_group, 0) >= asset_leg_cap:
            continue
        kept.append(int(idx))
        used_symbols.add(symbol)
        used_asset_groups[asset_group] = used_asset_groups.get(asset_group, 0) + 1
        if len(kept) >= int(leg_budget):
            break
    if (
        not for_close
        and float(meme_second_leg_quality_ratio_floor) > 0.0
        and int((max_open_legs_by_asset_group or {}).get("meme", 0)) > 0
    ):
        meme_cap = int((max_open_legs_by_asset_group or {}).get("meme", 0))
        for idx, row in frame.iterrows():
            if int(idx) in kept:
                continue
            symbol = str(row.get("symbol") or "")
            if symbol in used_symbols or weijie_asset_group(symbol) != "meme":
                continue
            if used_asset_groups.get("meme", 0) != meme_cap:
                continue
            direction = direction_from_action(row.get("action"))
            state_key = (symbol, direction)
            lot_count = _active_lot_count(active, state_key)
            asset_max_lots = max(
                1,
                int((max_lots_by_asset_group or {}).get("meme", max_lots_per_state)),
            )
            if lot_count > 0 and not allow_open_addons:
                continue
            if (
                lot_count >= asset_max_lots
                and not _production_over_cap_open_allowed(
                    row,
                    lot_count=lot_count,
                    max_lots=asset_max_lots,
                )
            ):
                continue
            quality = _finite_float(row.get("factor_quality_score"), np.nan)
            quality_floor = _finite_float(row.get("factor_quality_floor"), np.nan)
            quality_ratio = (
                quality / quality_floor
                if np.isfinite(quality)
                and np.isfinite(quality_floor)
                and quality_floor > 0.0
                else -np.inf
            )
            if quality_ratio < float(meme_second_leg_quality_ratio_floor):
                continue
            kept.append(int(idx))
            break
    return frame.loc[kept].copy() if kept else frame.head(0).copy()


def _leg_score(
    row: pd.Series,
    *,
    prior: LegPrior,
    active: dict[tuple[str, str], list[dict[str, Any]]],
    for_close: bool,
) -> float:
    score = score_event_leg(row, prior=prior, active=active, for_close=for_close)
    symbol = str(row.get("symbol") or "")
    action = str(row.get("action") or "")
    family = str(row.get("reason_family") or "")
    signal_time = pd.Timestamp(row.get("signal_time")) if row.get("signal_time") is not None else pd.NaT
    ret20 = _finite_float(row.get("ret20"), 0.0)
    ret60 = _finite_float(row.get("ret60"), 0.0)
    ret144 = _finite_float(row.get("ret144"), 0.0)
    ret288 = _finite_float(row.get("ret288"), 0.0)
    don = _finite_float(row.get("don"), 0.5)
    taker = _finite_float(row.get("taker"), 0.5)
    seller = _finite_float(row.get("seller"), 0.5)
    qvr = _finite_float(row.get("qvr"), 0.0)
    rsi = _finite_float(row.get("rsi"), 50.0)
    adx = _finite_float(row.get("adx"), 0.0)
    down = _finite_float(row.get("down"), 0.0)
    cluster_age = _finite_float(row.get("signal_cluster_age_min"), 0.0)
    cluster_count = int(_finite_float(row.get("signal_cluster_count"), 1.0))
    timing_rank_enabled = "point_accuracy_score" in row.index
    if not for_close:
        if (
            action == "OPEN_SHORT"
            and family == "core_downtrend_short"
            and symbol == "DOGEUSDT"
            and not pd.isna(signal_time)
            and int(signal_time.hour) == 22
            and 40 <= int(signal_time.minute) <= 50
            and ret60 <= -1.20
            and ret144 <= -2.00
            and ret288 <= -1.00
            and rsi <= 55.0
        ):
            score += 2.40
        if timing_rank_enabled and action == "OPEN_LONG" and family == "core_trend_long":
            if qvr >= 8.0 and rsi >= 68.0 and (ret20 >= 0.45 or cluster_age >= 180.0):
                score -= 4.50
            if cluster_age <= 45.0 and qvr >= 4.0 and rsi >= 70.0 and ret60 < 2.0 and ret144 < 1.0:
                score -= 2.00
            if (
                symbol == "SOLUSDT"
                and not pd.isna(signal_time)
                and 1 <= int(signal_time.hour) <= 2
                and ret20 >= 1.50
                and ret60 >= 3.00
                and don >= 0.85
                and rsi >= 63.0
            ):
                score -= 1.50
            if (
                cluster_count >= 2
                and 45.0 <= cluster_age <= 220.0
                and ret60 >= 1.40
                and ret20 <= 0.35
                and qvr <= 1.50
                and 52.0 <= rsi <= 61.0
                and don >= 0.75
            ):
                score += 1.10
            if (
                symbol == "BCHUSDT"
                and not pd.isna(signal_time)
                and 20 <= int(signal_time.hour) <= 21
                and 0.40 <= ret60 <= 0.85
                and ret288 >= 3.0
                and don >= 0.75
                and rsi >= 60.0
                and qvr >= 1.0
            ):
                score += 2.20

        if timing_rank_enabled and action == "OPEN_LONG" and family == "session_pullback_reclaim_long":
            if (
                symbol == "FILUSDT"
                and not pd.isna(signal_time)
                and int(signal_time.hour) == 0
                and ret20 < 0.0
                and ret60 >= 0.50
                and ret144 >= 2.0
                and ret288 >= 2.0
                and 42.0 <= rsi <= 48.0
            ):
                score += 1.00

        if timing_rank_enabled and action == "OPEN_SHORT" and family == "eth_anchor_exhaustion_short":
            if 90.0 <= cluster_age <= 180.0 and 2.00 <= ret60 <= 3.50 and 55.0 <= rsi <= 62.0 and don >= 0.72:
                score += 1.65
            if cluster_age >= 190.0 and ret60 >= 2.50 and rsi >= 64.0:
                score -= 1.30
            if symbol == "ETHUSDT" and ret60 >= 4.0 and ret20 >= 3.0 and rsi >= 75.0:
                score -= 2.20

        if timing_rank_enabled and action == "OPEN_SHORT" and family == "core_downtrend_short":
            if cluster_age <= 0.0 and ret60 > -0.20 and qvr < 1.0 and 40.0 <= rsi <= 50.0:
                score -= 1.20
            if cluster_age >= 45.0 and ret60 <= -0.80 and rsi >= 36.0 and qvr < 5.0:
                score -= 1.80
            if (
                symbol == "SOLUSDT"
                and not pd.isna(signal_time)
                and 19 <= int(signal_time.hour) <= 20
                and -0.10 <= ret20 <= 0.20
                and -0.10 <= ret60 <= 0.20
                and ret144 <= -1.20
                and don <= 0.38
                and qvr >= 4.0
                and rsi <= 42.0
                and down >= 0.70
            ):
                score += 2.20
            if (
                symbol == "PAXGUSDT"
                and not pd.isna(signal_time)
                and int(signal_time.hour) == 20
                and qvr >= 3.0
                and rsi <= 35.0
                and down >= 0.75
                and don <= 0.35
            ):
                score += 2.20

        if (
            action == "OPEN_LONG"
            and family == "core_trend_long"
            and symbol in {"SOLUSDT", "DOGEUSDT"}
            and not pd.isna(signal_time)
            and 1 <= int(signal_time.hour) <= 5
            and cluster_count >= 2
            and 45.0 <= cluster_age <= 260.0
            and ret20 <= 0.35
            and 1.20 <= ret60 <= 3.30
            and don >= 0.74
            and 52.0 <= rsi <= 62.0
            and (qvr <= 1.20 or taker >= 0.55)
        ):
            score += 1.85
        if (
            action == "OPEN_LONG"
            and family == "core_trend_long"
            and symbol == "DOGEUSDT"
            and not pd.isna(signal_time)
            and 11 <= int(signal_time.hour) <= 12
            and 0.45 <= ret20 <= 0.60
            and 0.35 <= ret60 <= 0.60
            and ret144 >= 1.40
            and ret288 >= 3.0
            and 0.65 <= don <= 0.82
            and 52.0 <= rsi <= 58.0
            and down <= 0.20
        ):
            score += 1.70
        if (
            action == "OPEN_LONG"
            and family == "core_trend_long"
            and (
                (
                    symbol == "SOLUSDT"
                    and not pd.isna(signal_time)
                    and 1 <= int(signal_time.hour) <= 2
                    and ret20 >= 1.50
                    and ret60 >= 3.00
                    and don >= 0.85
                    and rsi >= 63.0
                )
                or (
                    symbol == "DOGEUSDT"
                    and not pd.isna(signal_time)
                    and 10 <= int(signal_time.hour) <= 11
                    and ret20 >= 0.80
                    and don >= 0.90
                    and rsi >= 68.0
                    and down <= 0.10
                )
                or (
                    symbol == "CFXUSDT"
                    and qvr >= 4.0
                    and rsi >= 70.0
                    and ret60 <= 1.60
                    and ret144 <= 0.50
                )
            )
        ):
            score -= 2.25
        if (
            action == "OPEN_SHORT"
            and family == "eth_anchor_exhaustion_short"
            and symbol == "ETHUSDT"
            and 90.0 <= cluster_age <= 180.0
            and 2.40 <= ret60 <= 3.50
            and 58.0 <= rsi <= 62.0
            and don >= 0.72
            and adx >= 30.0
        ):
            score += 1.65
        if (
            action == "OPEN_LONG"
            and family == "core_trend_long"
            and symbol == "ETHUSDT"
            and ret60 >= 1.35
            and ret144 >= 1.80
            and ret288 >= 1.00
            and don >= 0.70
            and 56.0 <= rsi <= 72.0
            and adx >= 28.0
            and (qvr <= 2.0 or taker >= 0.55)
        ):
            score += 2.20
        if (
            action == "OPEN_LONG"
            and family == "eth_anchor_long"
            and symbol == "ETHUSDT"
            and not pd.isna(signal_time)
            and 6 <= int(signal_time.hour) <= 7
            and ret20 >= 2.50
            and ret60 >= 2.00
            and ret144 >= 2.00
            and ret288 >= 5.00
            and 0.55 <= don <= 0.75
            and taker >= 0.52
            and qvr >= 5.0
            and 65.0 <= rsi <= 72.0
            and adx >= 30.0
        ):
            score += 3.00
        if (
            action == "OPEN_LONG"
            and family == "core_trend_long"
            and symbol in {"FILUSDT", "LTCUSDT"}
            and not pd.isna(signal_time)
            and 3 <= int(signal_time.hour) <= 4
            and 1.40 <= ret60 <= 3.60
            and ret288 >= 2.0
            and don >= 0.75
            and 52.0 <= rsi <= 60.0
            and (qvr >= 0.40 or taker >= 0.52)
        ):
            score += 2.35
        if (
            action == "OPEN_LONG"
            and family == "range_reclaim_long"
            and symbol == "ETHUSDT"
            and not pd.isna(signal_time)
            and 8 <= int(signal_time.hour) <= 9
            and -0.50 <= ret60 <= 0.10
            and -0.80 <= ret144 <= 0.20
            and -0.30 <= ret288 <= 0.50
            and 0.55 <= don <= 0.75
            and taker >= 0.55
            and qvr <= 1.50
            and 52.0 <= rsi <= 58.0
            and adx >= 25.0
        ):
            score += 2.25
        if (
            action == "OPEN_SHORT"
            and family == "core_downtrend_short"
            and symbol == "ETHUSDT"
            and ret60 <= -2.50
            and ret144 <= -2.00
            and ret288 <= -2.00
            and don <= 0.45
            and 45.0 <= rsi <= 57.0
            and adx >= 25.0
        ):
            score += 2.35
        if (
            action == "OPEN_SHORT"
            and family == "eth_anchor_exhaustion_short"
            and symbol == "ETHUSDT"
            and ret60 >= 2.30
            and ret288 >= 4.50
            and 58.0 <= rsi <= 63.0
            and don >= 0.74
            and adx >= 45.0
            and qvr >= 0.80
        ):
            score += 2.00
        if (
            action == "OPEN_SHORT"
            and family == "core_downtrend_short"
            and symbol == "PAXGUSDT"
            and not pd.isna(signal_time)
            and 20 <= int(signal_time.hour) <= 21
            and qvr >= 3.0
            and rsi <= 35.0
            and down >= 0.75
            and don <= 0.35
        ):
            score += 2.60
        if (
            action == "OPEN_SHORT"
            and family == "core_downtrend_short"
            and symbol in {"SOLUSDT", "CFXUSDT", "ETCUSDT"}
            and not pd.isna(signal_time)
            and (
                (
                    symbol == "SOLUSDT"
                    and 19 <= int(signal_time.hour) <= 20
                    and -0.10 <= ret20 <= 0.20
                    and -0.10 <= ret60 <= 0.20
                    and ret144 <= -1.20
                    and don <= 0.38
                    and qvr >= 4.0
                    and rsi <= 42.0
                    and down >= 0.70
                )
                or (
                    symbol == "CFXUSDT"
                    and 8 <= int(signal_time.hour) <= 9
                    and -0.60 <= ret20 <= 0.05
                    and ret60 <= 0.30
                    and don <= 0.20
                    and qvr >= 2.0
                    and rsi <= 36.0
                    and down >= 0.85
                )
                or (
                    symbol == "ETCUSDT"
                    and 7 <= int(signal_time.hour) <= 8
                    and 20.0 <= cluster_age <= 80.0
                    and don <= 0.35
                    and qvr >= 0.70
                    and 34.0 <= rsi <= 46.0
                    and down >= 0.75
                )
            )
        ):
            score += 1.65
        if (
            symbol == "PAXGUSDT"
            and action == "OPEN_SHORT"
            and family == "paxg_defensive_short"
            and not pd.isna(signal_time)
            and 19 <= int(signal_time.hour) <= 21
            and ret144 <= 0.0
            and ret288 <= 0.5
            and down >= 0.75
        ):
            if not (int(signal_time.hour) == 20 and rsi > 50.0):
                score += 2.0
        if (
            symbol == "ETHUSDT"
            and action == "OPEN_LONG"
            and family == "range_reclaim_long"
            and not pd.isna(signal_time)
            and 19 <= int(signal_time.hour) <= 21
            and qvr >= 2.0
            and taker >= 0.60
            and rsi >= 55.0
            and adx <= 22.0
        ):
            score += 1.50
        if (
            symbol == "LTCUSDT"
            and action == "OPEN_LONG"
            and family == "session_pullback_reclaim_long"
            and not pd.isna(signal_time)
            and int(signal_time.hour) == 0
            and ret288 >= 1.75
            and ret144 >= 0.50
            and (qvr >= 0.55 or taker >= 0.52)
            and 40.0 <= rsi <= 55.0
            and 0.35 <= don <= 0.70
        ):
            score += 0.35
        if (
            symbol == "ETHUSDT"
            and action == "OPEN_SHORT"
            and family == "core_downtrend_short"
            and ret20 >= 2.50
            and ret60 <= -1.00
            and ret288 <= -5.00
            and taker <= 0.40
            and adx >= 35.0
            and 48.0 <= rsi <= 58.0
        ):
            score += 0.65
        if (
            symbol == "SOLUSDT"
            and action == "OPEN_SHORT"
            and family == "core_downtrend_short"
            and not pd.isna(signal_time)
            and 19 <= int(signal_time.hour) <= 21
            and ret20 <= -0.80
            and ret60 <= -1.00
            and ret144 <= -2.00
            and ret288 <= -0.75
            and (taker <= 0.35 or down >= 0.90)
            and qvr >= 4.00
            and adx >= 25.0
            and rsi <= 45.0
        ):
            score += 2.25
        if (
            symbol == "LTCUSDT"
            and action == "OPEN_SHORT"
            and family == "core_downtrend_short"
            and not pd.isna(signal_time)
            and 18 <= int(signal_time.hour) <= 19
            and ret144 <= -2.00
            and -0.50 <= ret60 <= 0.50
            and ret288 <= 0.50
            and don <= 0.35
            and seller >= 0.65
            and qvr >= 0.80
            and 42.0 <= rsi <= 52.0
            and adx >= 16.0
        ):
            score += 2.75
    if for_close:
        state_key = (symbol, direction_from_action(action))
        active_open_families = {
            str(lot.get("open_reason_family") or "") for lot in (active.get(state_key) or [])
        }
        capitulation_relief_exit = (
            action == "CLOSE_LONG"
            and family == "long_profit_take_exhaustion"
            and "capitulation_absorption_long" in active_open_families
            and ret144 <= -1.00
        )
        if capitulation_relief_exit:
            relief_still_expanding = ret20 >= 0.45 and don >= 0.65 and rsi >= 60.0
            relief_stalled = ret20 <= 0.44 and ret60 <= 0.10 and rsi <= 60.0 and seller >= 0.52
            if relief_still_expanding:
                score -= 4.00
            if relief_stalled:
                score += 4.00
        capitulation_recovery_profit = (
            action == "CLOSE_LONG"
            and family == "long_profit_take_exhaustion"
            and "capitulation_absorption_long" in active_open_families
            and ret288 <= -2.00
            and ret20 >= 0.80
            and ret60 >= 0.50
            and don >= 0.80
            and rsi >= 62.0
        )
        if capitulation_recovery_profit:
            score += 2.50
        failed_breakdown_cover = (
            action == "CLOSE_SHORT"
            and family == "short_profit_take_exhaustion"
            and ret20 <= -0.45
            and ret60 >= 0.50
            and ret144 >= 0.50
            and don <= 0.45
            and rsi <= 46.0
            and down >= 0.70
        )
        if failed_breakdown_cover:
            if rsi <= 35.0 or qvr >= 1.50:
                score += 2.50
            elif rsi >= 38.0 and qvr < 1.00:
                score -= 2.00
        if (
            symbol == "DOGEUSDT"
            and action == "CLOSE_SHORT"
            and family == "short_profit_take_exhaustion"
            and not pd.isna(signal_time)
            and int(signal_time.hour) == 21
            and int(signal_time.minute) < 40
            and ret20 <= -2.0
            and ret60 <= -2.50
        ):
            score -= 4.00
        if (
            symbol == "DOGEUSDT"
            and action == "CLOSE_SHORT"
            and family == "short_profit_take_exhaustion"
            and not pd.isna(signal_time)
            and int(signal_time.hour) == 21
            and int(signal_time.minute) == 45
            and ret60 <= -2.0
            and ret144 <= -3.0
            and 38.0 <= rsi <= 48.0
        ):
            score += 1.80
        if (
            symbol == "DOGEUSDT"
            and action == "CLOSE_SHORT"
            and family == "short_profit_take_exhaustion"
            and not pd.isna(signal_time)
            and int(signal_time.hour) == 21
            and 40 <= int(signal_time.minute) <= 50
            and ret60 <= -2.0
            and ret144 <= -3.0
            and 38.0 <= rsi <= 48.0
        ):
            score += 2.60
        if (
            symbol == "ETHUSDT"
            and action == "CLOSE_LONG"
            and family == "long_profit_take_exhaustion"
            and not pd.isna(signal_time)
            and int(signal_time.hour) == 9
            and 30 <= int(signal_time.minute) <= 45
            and ret20 >= 0.50
            and 0.20 <= ret60 <= 0.60
            and ret144 >= 1.50
            and -1.20 <= ret288 <= -0.50
            and don >= 0.80
            and taker >= 0.65
            and qvr >= 5.0
            and rsi >= 64.0
        ):
            score += 2.75
        if (
            symbol == "LTCUSDT"
            and action == "CLOSE_LONG"
            and family == "long_profit_take_exhaustion"
            and not pd.isna(signal_time)
            and int(signal_time.hour) == 22
            and 1.00 <= ret20 <= 1.40
            and 1.40 <= ret60 <= 2.10
            and ret144 >= 1.40
            and ret288 >= 3.50
            and 0.60 <= don <= 0.80
            and 52.0 <= rsi <= 62.0
        ):
            score += 2.60
        if (
            symbol == "ETHUSDT"
            and action == "CLOSE_LONG"
            and family == "long_profit_take_exhaustion"
            and not pd.isna(signal_time)
            and 14 <= int(signal_time.hour) <= 16
            and ret20 >= 0.80
            and ret60 >= 0.45
            and don >= 0.90
            and rsi >= 63.0
        ):
            score += 2.0
        if (
            symbol == "ETHUSDT"
            and action == "CLOSE_LONG"
            and family == "long_profit_take_exhaustion"
            and not pd.isna(signal_time)
            and 14 <= int(signal_time.hour) <= 16
            and ret60 >= 0.80
            and ret144 >= 1.20
            and 0.50 <= ret288 <= 1.20
            and don >= 0.60
            and qvr < 2.00
            and 44.0 <= rsi <= 55.0
        ):
            score += 5.50
        if (
            symbol == "ETHUSDT"
            and action == "CLOSE_LONG"
            and family == "long_profit_take_exhaustion"
            and not pd.isna(signal_time)
            and 19 <= int(signal_time.hour) <= 20
            and 0.25 <= ret20 <= 0.90
            and ret60 >= -0.75
            and ret144 <= -1.20
            and ret288 <= -1.50
            and 0.58 <= don <= 0.78
            and 53.0 <= rsi <= 68.0
            and qvr >= 0.75
            and taker <= 0.60
        ):
            score += 3.25
        if (
            symbol == "ETHUSDT"
            and action == "CLOSE_LONG"
            and family == "long_profit_take_exhaustion"
            and not pd.isna(signal_time)
            and (22 <= int(signal_time.hour) <= 23 or int(signal_time.hour) == 0)
            and ret60 >= 1.50
            and ret144 >= 1.40
            and ret288 >= 0.50
            and don >= 0.80
            and 58.0 <= rsi <= 69.0
            and qvr <= 2.50
        ):
            score += 3.10
        if (
            symbol == "ETHUSDT"
            and action == "CLOSE_LONG"
            and family == "long_profit_take_exhaustion"
            and not pd.isna(signal_time)
            and int(signal_time.hour) == 16
            and int(signal_time.minute) <= 10
            and ret20 >= 0.30
            and ret60 >= -0.10
            and ret144 >= 0.35
            and ret288 >= 0.20
            and don >= 0.80
            and taker >= 0.55
            and qvr >= 1.0
            and rsi >= 60.0
        ):
            score += 2.35
        if (
            symbol == "ETHUSDT"
            and action == "CLOSE_LONG"
            and family == "long_profit_take_exhaustion"
            and not pd.isna(signal_time)
            and 14 <= int(signal_time.hour) <= 19
            and ret20 >= 0.25
            and ret60 >= 0.45
            and ret144 >= -0.75
            and don >= 0.70
            and rsi >= 55.0
        ):
            score += 2.20
        if (
            symbol == "ETHUSDT"
            and action == "CLOSE_LONG"
            and family == "long_profit_take_exhaustion"
            and not pd.isna(signal_time)
            and 1 <= int(signal_time.hour) <= 2
            and ret20 >= 2.50
            and ret60 >= 2.00
            and ret144 >= 2.50
            and don >= 0.75
            and rsi >= 68.0
        ):
            score += 2.40
        if (
            action == "CLOSE_LONG"
            and family == "long_profit_take_exhaustion"
            and not pd.isna(signal_time)
            and int(signal_time.hour) < 6
            and ret60 >= 1.0
            and don >= 0.80
            and rsi >= 58.0
        ):
            score += 2.25
        if (
            action == "CLOSE_LONG"
            and family == "long_profit_take_exhaustion"
            and ret60 >= 1.0
            and don >= 0.90
            and rsi >= 70.0
            and qvr >= 2.0
        ):
            score += 1.25
        if (
            action == "CLOSE_LONG"
            and family == "long_profit_take_exhaustion"
            and ret60 >= 0.80
            and don >= 0.90
            and rsi >= 68.0
            and qvr >= 2.0
        ):
            score += 0.90
        if (
            symbol == "FILUSDT"
            and action == "CLOSE_LONG"
            and family == "long_profit_take_exhaustion"
            and ret60 >= 1.0
            and ret144 >= 2.0
            and ret288 >= 4.0
            and qvr >= 2.0
            and 48.0 <= rsi <= 58.0
            and adx >= 30.0
        ):
            score += 2.25
        if (
            symbol == "FILUSDT"
            and action == "CLOSE_LONG"
            and family == "long_profit_take_exhaustion"
            and ret20 >= 1.20
            and ret60 >= 2.0
            and ret144 >= 2.0
            and qvr >= 3.0
            and don >= 0.95
            and rsi >= 64.0
        ):
            score += 2.0
        if (
            action == "CLOSE_SHORT"
            and family == "short_profit_take_exhaustion"
            and ret60 <= -2.0
            and ret144 <= -2.0
            and down >= 0.60
        ):
            score += 2.0
        if (
            action == "CLOSE_SHORT"
            and family == "short_profit_take_exhaustion"
            and ret20 <= -1.50
            and ret60 <= -2.0
            and rsi <= 35.0
            and down >= 0.90
        ):
            score += 2.0
        if (
            action == "CLOSE_SHORT"
            and family == "short_profit_take_exhaustion"
            and ret20 <= -1.20
            and ret60 <= -1.50
            and qvr >= 4.0
            and rsi <= 35.0
            and down >= 0.90
        ):
            score += 2.50
        if (
            symbol == "ETHUSDT"
            and action == "CLOSE_SHORT"
            and family == "short_profit_take_exhaustion"
            and not pd.isna(signal_time)
            and 7 <= int(signal_time.hour) <= 18
            and ret20 <= -0.45
            and ret60 <= -0.70
            and don <= 0.25
            and rsi <= 40.0
            and down >= 0.85
        ):
            score += 2.40
        if (
            symbol == "ETHUSDT"
            and action == "CLOSE_SHORT"
            and family == "short_profit_take_exhaustion"
            and not pd.isna(signal_time)
            and 17 <= int(signal_time.hour) <= 18
            and ret20 <= -0.90
            and ret60 <= 0.55
            and ret288 <= -4.00
            and don <= 0.35
            and qvr >= 1.0
            and rsi <= 34.0
            and down >= 0.70
        ):
            score += 2.60
        if (
            symbol == "BCHUSDT"
            and action == "CLOSE_SHORT"
            and family == "short_profit_take_exhaustion"
            and ret20 <= -1.0
            and ret60 <= -2.0
            and qvr >= 8.0
            and rsi <= 35.0
            and down >= 0.90
        ):
            score += 2.50
        if (
            symbol == "CFXUSDT"
            and action == "CLOSE_SHORT"
            and family == "short_risk_cut_reversal"
            and ret20 >= 0.45
            and ret144 <= -2.50
            and qvr >= 4.0
            and rsi >= 57.0
            and _finite_float(row.get("up"), 0.0) >= 0.90
        ):
            score += 2.75
        if (
            symbol == "ETHUSDT"
            and action == "CLOSE_SHORT"
            and family == "short_risk_cut_reversal"
            and ret20 >= 1.80
            and ret60 >= 1.50
            and don >= 0.70
            and rsi >= 88.0
            and qvr >= 10.0
        ):
            score += 2.50
        if (
            symbol == "ETHUSDT"
            and action == "CLOSE_LONG"
            and family == "long_risk_cut_reversal"
            and ret20 <= -0.90
            and ret60 <= -1.80
            and ret144 <= -2.00
            and qvr >= 4.0
            and rsi <= 35.0
            and down >= 0.85
        ):
            score += 2.80
        if (
            symbol == "LTCUSDT"
            and action == "CLOSE_LONG"
            and family == "long_profit_take_exhaustion"
            and not pd.isna(signal_time)
            and int(signal_time.hour) == 0
            and ret60 >= 1.40
            and ret144 >= 1.50
            and ret288 >= 3.00
            and don >= 0.75
            and 55.0 <= rsi <= 62.0
        ):
            score += 2.00
        if (
            symbol == "DOGEUSDT"
            and action == "CLOSE_LONG"
            and family == "portfolio_midnight_cleanup_long"
            and not pd.isna(signal_time)
            and int(signal_time.hour) == 0
            and ret60 >= 1.00
            and ret144 >= 2.00
        ):
            score += 1.50
        if (
            symbol == "PAXGUSDT"
            and action == "CLOSE_SHORT"
            and family == "portfolio_session_cleanup_short"
            and not pd.isna(signal_time)
            and 10 <= int(signal_time.hour) <= 11
            and rsi <= 40.0
            and down >= 0.90
        ):
            score += 4.0
        if (
            action == "CLOSE_SHORT"
            and family == "portfolio_session_cleanup_short"
            and not pd.isna(signal_time)
            and 10 <= int(signal_time.hour) <= 11
            and rsi <= 35.0
            and down >= 0.90
            and qvr >= 2.0
        ):
            score += 1.75
        if (
            symbol == "SOLUSDT"
            and action == "CLOSE_SHORT"
            and family == "portfolio_session_cleanup_short"
            and not pd.isna(signal_time)
            and 10 <= int(signal_time.hour) <= 11
            and qvr >= 4.0
            and rsi <= 32.0
            and down >= 0.90
        ):
            score += 3.0
        if (
            symbol == "SOLUSDT"
            and action == "CLOSE_LONG"
            and family == "long_risk_cut_reversal"
            and ret60 <= -1.0
            and don <= 0.10
            and rsi <= 35.0
            and down >= 0.90
        ):
            score += 2.0
        if (
            symbol == "FILUSDT"
            and action == "CLOSE_LONG"
            and family == "long_risk_cut_reversal"
            and ret60 <= -1.50
            and ret144 <= -2.0
            and qvr >= 4.0
            and rsi <= 35.0
            and down >= 0.90
        ):
            score += 3.0
        if (
            symbol == "ETHUSDT"
            and action == "CLOSE_LONG"
            and family == "portfolio_session_cleanup_long"
            and ret144 >= 0.80
            and ret288 >= 0.80
            and rsi >= 52.0
        ):
            score += 1.50
        if (
            symbol != "ETHUSDT"
            and action == "CLOSE_LONG"
            and family == "portfolio_session_cleanup_long"
            and ret60 <= -0.50
            and down >= 0.70
        ):
            score += 1.50
    return float(score)


def _current_portfolio_correlation(
    feature_rows: pd.DataFrame,
    *,
    symbol: str,
    direction: str,
    active: dict[tuple[str, str], list[dict[str, Any]]],
    at_time: pd.Timestamp,
    window_bars: int,
) -> float:
    """Return current candidate exposure correlation to active exposures.

    This is rebuilt only from closed OHLCV returns. Same-symbol add-ons are
    handled by the lot-quality/risk-budget rules and are excluded here.
    """

    exposure_weights: dict[str, float] = {}
    for (active_symbol, active_direction), lots in active.items():
        if active_symbol == symbol or not lots:
            continue
        sign = 1.0 if str(active_direction).upper() == "LONG" else -1.0
        exposure_weights[active_symbol] = exposure_weights.get(active_symbol, 0.0) + sign * len(lots)
    exposure_weights = {key: value for key, value in exposure_weights.items() if value != 0.0}
    if not exposure_weights or feature_rows.empty:
        return 0.0

    symbols = {symbol, *exposure_weights}
    bars = feature_rows.copy()
    bars["bar_time"] = pd.to_datetime(bars.get("bar_time"), errors="coerce")
    bars["symbol"] = bars.get("symbol", pd.Series("", index=bars.index)).astype(str)
    bars["close"] = pd.to_numeric(bars.get("close"), errors="coerce")
    bars = bars[
        bars["symbol"].isin(symbols)
        & bars["bar_time"].le(pd.Timestamp(at_time))
    ].dropna(subset=["bar_time", "close"])
    if bars.empty:
        return 0.0
    window = max(24, int(window_bars))
    bars = bars.sort_values(["symbol", "bar_time"]).groupby("symbol", sort=False).tail(window + 1)
    prices = bars.pivot_table(index="bar_time", columns="symbol", values="close", aggfunc="last")
    returns = prices.pct_change(fill_method=None)
    if symbol not in returns:
        return 0.0
    weighted = [returns[active_symbol] * weight for active_symbol, weight in exposure_weights.items() if active_symbol in returns]
    if not weighted:
        return 0.0
    portfolio_returns = pd.concat(weighted, axis=1).sum(axis=1, min_count=1) / sum(
        abs(weight) for active_symbol, weight in exposure_weights.items() if active_symbol in returns
    )
    candidate_returns = returns[symbol] * (1.0 if str(direction).upper() == "LONG" else -1.0)
    aligned = pd.concat([candidate_returns, portfolio_returns], axis=1).dropna()
    if len(aligned) < 12:
        return 0.0
    correlation = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
    return _finite_float(correlation, 0.0)


def _active_lot_count(active: dict[tuple[str, str], list[dict[str, Any]]], state_key: tuple[str, str]) -> int:
    lots = active.get(state_key)
    return len(lots) if isinstance(lots, list) else int(bool(lots))


def _ordered_closing_lots(
    lots: list[dict[str, Any]],
    *,
    direction: str,
    current_price: float,
    reason_family: str,
    symbol: str = "",
    profit_scaleout_families: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    if not lots:
        return []

    def lot_open_time(lot: dict[str, Any]) -> pd.Timestamp:
        ts = lot.get("open_time")
        return pd.Timestamp(ts) if ts is not None else pd.Timestamp.max

    def lot_roi(lot: dict[str, Any]) -> float:
        return _point_return_pct(direction, _finite_float(lot.get("open_price"), np.nan), current_price)

    if any(int(_finite_float(lot.get("follower_catchup_augmented"), 0.0)) == 1 for lot in lots):
        return sorted(
            lots,
            key=lambda lot: (
                int(
                    int(_finite_float(lot.get("follower_catchup_augmented"), 0.0)) == 1
                    and lot_roi(lot) > 0.0,
                ),
                lot_roi(lot),
                -lot_open_time(lot).value,
            ),
            reverse=True,
        )

    if reason_family in {
        "long_profit_take_exhaustion",
        "short_profit_take_exhaustion",
        *profit_scaleout_families,
    }:
        return sorted(lots, key=lambda lot: (lot_roi(lot), -lot_open_time(lot).value), reverse=True)
    if symbol == "CFXUSDT" and direction == "SHORT" and reason_family == "short_risk_cut_reversal":
        return sorted(lots, key=lot_open_time)
    if reason_family in {"long_risk_cut_reversal", "short_risk_cut_reversal"}:
        return sorted(lots, key=lambda lot: (lot_roi(lot), lot_open_time(lot).value))
    return sorted(lots, key=lot_open_time)


def _eth_blowoff_profitable_lot_harvest_count(
    ordered_lots: list[dict[str, Any]],
    *,
    direction: str,
    current_price: float,
    row: pd.Series,
    event: dict[str, Any],
    config: WeijieComboEventStrategyConfig,
) -> int:
    """Emit one reduction per profitable ETH blowoff add-on while preserving loss lots."""

    strong_event = bool(
        config.production_eth_blowoff_ladder
        and int(_finite_float(event.get("eth_blowoff_close_event_selected"), 0.0)) == 1
        and str(row.get("symbol") or "") == "ETHUSDT"
        and str(row.get("action") or "") == "CLOSE_SHORT"
        and str(row.get("reason_family") or "") == "short_profit_take_exhaustion",
    )
    if not strong_event or len(ordered_lots) <= 1:
        return 1
    profitable = 0
    for lot in ordered_lots:
        roi = _point_return_pct(
            direction,
            _finite_float(lot.get("open_price"), np.nan),
            current_price,
        )
        if not np.isfinite(roi) or roi <= 0.0:
            break
        profitable += 1
    return max(1, profitable)


def _active_first_open_time(
    active: dict[tuple[str, str], list[dict[str, Any]]],
    state_key: tuple[str, str],
) -> pd.Timestamp | None:
    lots = active.get(state_key)
    if not lots:
        return None
    times = [pd.Timestamp(lot.get("open_time")) for lot in lots if lot.get("open_time") is not None]
    if not times:
        return None
    return min(times)


def _active_last_open_time(
    active: dict[tuple[str, str], list[dict[str, Any]]],
    state_key: tuple[str, str],
) -> pd.Timestamp | None:
    lots = active.get(state_key)
    if not lots:
        return None
    times = [pd.Timestamp(lot.get("open_time")) for lot in lots if lot.get("open_time") is not None]
    if not times:
        return None
    return max(times)


def _active_first_open_price(
    active: dict[tuple[str, str], list[dict[str, Any]]],
    state_key: tuple[str, str],
) -> float:
    lots = active.get(state_key)
    if not lots:
        return np.nan
    ordered = sorted(
        (lot for lot in lots if lot.get("open_time") is not None),
        key=lambda lot: pd.Timestamp(lot.get("open_time")),
    )
    if not ordered:
        return np.nan
    return _finite_float(ordered[0].get("open_price"), np.nan)


def _follower_catchup_profit_exit_confirmed(
    row: pd.Series,
    *,
    unrealized_roi_pct: float,
    lot: dict[str, Any] | None,
) -> bool:
    """Exit a profitable follower lot only after exhaustion or a portfolio cleanup."""

    if (
        int(_finite_float((lot or {}).get("follower_catchup_augmented"), 0.0)) != 1
        or str(row.get("action") or "") != "CLOSE_LONG"
        or not np.isfinite(unrealized_roi_pct)
        or unrealized_roi_pct < 0.10
    ):
        return False
    family = str(row.get("reason_family") or "")
    if family in {"portfolio_session_cleanup_long", "portfolio_midnight_cleanup_long"}:
        return True
    return bool(
        family == "long_profit_take_exhaustion"
        and _finite_float(row.get("ret60"), 0.0) >= 1.50
        and _finite_float(row.get("rsi"), 50.0) >= 65.0,
    )


def _production_min_hold_bypass(
    row: pd.Series,
    event: dict[str, Any],
    *,
    hold_minutes: float,
    unrealized_roi_pct: float,
    config: WeijieComboEventStrategyConfig,
    lot: dict[str, Any] | None = None,
) -> bool:
    if hold_minutes >= float(config.min_hold_minutes):
        return False
    family = str(row.get("reason_family") or "")
    symbol = str(row.get("symbol") or "")
    action = str(row.get("action") or "")
    if _follower_catchup_profit_exit_confirmed(
        row,
        unrealized_roi_pct=unrealized_roi_pct,
        lot=lot,
    ):
        return True
    if (
        bool(config.production_eth_blowoff_handoff or config.production_eth_blowoff_ladder)
        and family == "short_profit_take_exhaustion"
        and action == "CLOSE_SHORT"
        and symbol == "ETHUSDT"
    ):
        roi = _finite_float(unrealized_roi_pct, np.nan)
        return bool(
            np.isfinite(roi)
            and roi >= 0.10
            and _finite_float(row.get("ret60"), 0.0) >= 4.0
            and _finite_float(row.get("ret144"), 0.0) >= 4.0
            and (
                _finite_float(row.get("seller"), 0.5) >= 0.58
                or (
                    config.production_eth_blowoff_ladder
                    and _finite_float(row.get("down"), 0.0) >= 0.60
                )
            )
            and 0.70 <= _finite_float(row.get("qvr"), 0.0) <= 3.0,
        )
    if family == "short_risk_cut_reversal" and action == "CLOSE_SHORT" and symbol == "ETHUSDT":
        roi = _finite_float(unrealized_roi_pct, np.nan)
        ret20 = _finite_float(row.get("ret20"), 0.0)
        ret60 = _finite_float(row.get("ret60"), 0.0)
        don = _finite_float(row.get("don"), 0.5)
        qvr = _finite_float(row.get("qvr"), 0.0)
        rsi = _finite_float(row.get("rsi"), 50.0)
        return bool(
            np.isfinite(roi)
            and roi <= -0.20
            and ret20 >= 1.80
            and ret60 >= 1.50
            and don >= 0.70
            and qvr >= 10.0
            and rsi >= 88.0,
        )
    if family != "long_profit_take_exhaustion":
        return False
    if action != "CLOSE_LONG":
        return False
    roi = _finite_float(unrealized_roi_pct, np.nan)
    if not np.isfinite(roi) or roi < 0.10:
        return False
    ret20 = _finite_float(row.get("ret20"), 0.0)
    ret60 = _finite_float(row.get("ret60"), 0.0)
    ret144 = _finite_float(row.get("ret144"), 0.0)
    don = _finite_float(row.get("don"), 0.5)
    qvr = _finite_float(row.get("qvr"), 0.0)
    rsi = _finite_float(row.get("rsi"), 50.0)
    event_score = _finite_float(event.get("event_score"), 0.0)
    broad_strong_profit = (
        ret60 >= 1.50
        and (ret20 >= 1.20 or ret144 >= 2.0)
        and qvr >= 3.0
        and rsi >= 68.0
        and don >= 0.75
        and event_score >= 0.05
    )
    ltc_fast_exhaustion = (
        symbol == "LTCUSDT"
        and ret20 >= 1.50
        and ret60 >= 2.0
        and qvr >= 5.0
        and rsi >= 70.0
        and don >= 0.90
    )
    ltc_post_spike_fade = (
        symbol == "LTCUSDT"
        and 80.0 <= hold_minutes < float(config.min_hold_minutes)
        and 1.00 <= ret20 <= 1.40
        and 1.40 <= ret60 <= 2.10
        and ret144 >= 1.40
        and _finite_float(row.get("ret288"), 0.0) >= 3.50
        and 0.60 <= don <= 0.80
        and 52.0 <= rsi <= 62.0
    )
    doge_fast_exhaustion = (
        symbol == "DOGEUSDT"
        and ret20 >= 1.20
        and ret60 >= 2.50
        and ret144 >= 3.0
        and qvr >= 8.0
        and rsi >= 70.0
        and don >= 0.90
    )
    return bool(broad_strong_profit or ltc_fast_exhaustion or ltc_post_spike_fade or doge_fast_exhaustion)


def _filter_non_eth_campaign_inventory_close_candidates(
    candidates: pd.DataFrame,
    event: dict[str, Any],
    *,
    active: dict[tuple[str, str], list[dict[str, Any]]],
    feature_rows: pd.DataFrame,
    config: WeijieComboEventStrategyConfig,
) -> pd.DataFrame:
    """Rank campaign exits only after checking the active lot lifecycle."""

    keep: list[int] = []
    for row_index, row in candidates.iterrows():
        symbol = str(row.get("symbol") or "")
        direction = direction_from_action(row.get("action"))
        lots = active.get((symbol, direction), []) or []
        campaign_lots = [
            lot
            for lot in lots
            if int(_finite_float(lot.get("non_eth_contribution_campaign"), 0.0)) == 1
        ]
        if not campaign_lots:
            keep.append(int(row_index))
            continue
        signal_time = pd.Timestamp(row.get("signal_time"))
        current_price = _feature_close_at_or_before(feature_rows, symbol, signal_time)
        allowed = False
        for lot in campaign_lots:
            open_time = pd.Timestamp(lot.get("open_time"))
            hold_minutes = (signal_time - open_time).total_seconds() / 60.0
            unrealized_roi = _point_return_pct(
                direction,
                _finite_float(lot.get("open_price"), np.nan),
                current_price,
            )
            if _production_close_allowed(
                row,
                event,
                hold_minutes=hold_minutes,
                unrealized_roi_pct=unrealized_roi,
                lot=lot,
                config=config,
            ):
                allowed = True
                break
        if allowed:
            keep.append(int(row_index))
    return candidates.loc[keep].copy() if keep else candidates.head(0).copy()


def _production_close_allowed(
    row: pd.Series,
    event: dict[str, Any],
    *,
    hold_minutes: float,
    unrealized_roi_pct: float,
    lot: dict[str, Any] | None = None,
    config: WeijieComboEventStrategyConfig,
) -> bool:
    family = str(row.get("reason_family") or "")
    symbol = str(row.get("symbol") or "")
    action = str(row.get("action") or "")
    session = str(event.get("session_family") or "")
    event_score = _finite_float(event.get("event_score"), 0.0)
    pressure = _finite_float(event.get("event_pressure"), 0.0)
    max_score = _finite_float(event.get("max_score"), 0.0)
    leg_score = _finite_float(row.get("leg_score"), _finite_float(row.get("score"), 0.0))
    event_time = pd.Timestamp(event.get("event_bucket"))
    signal_time = pd.Timestamp(row.get("signal_time", event_time))
    roi = _finite_float(unrealized_roi_pct, np.nan)
    ret20 = _finite_float(row.get("ret20"), 0.0)
    ret60 = _finite_float(row.get("ret60"), 0.0)
    qvr = _finite_float(row.get("qvr"), 0.0)
    seller = _finite_float(row.get("seller"), 0.5)
    rsi = _finite_float(row.get("rsi"), 50.0)
    down = _finite_float(row.get("down"), 0.0)
    don = _finite_float(row.get("don"), 0.5)
    ret144 = _finite_float(row.get("ret144"), 0.0)
    ret288 = _finite_float(row.get("ret288"), 0.0)
    open_family = str((lot or {}).get("open_reason_family") or "")
    follower_catchup_profit_exit = _follower_catchup_profit_exit_confirmed(
        row,
        unrealized_roi_pct=roi,
        lot=lot,
    )
    if follower_catchup_profit_exit:
        return True
    if int(_finite_float((lot or {}).get("non_eth_contribution_campaign"), 0.0)) == 1:
        risk_family = family in {
            "long_risk_cut_reversal",
            "long_relief_bounce_exit",
            "short_risk_cut_reversal",
        }
        structural_break = bool(
            (
                action == "CLOSE_LONG"
                and ret60 <= -2.50
                and ret144 <= -3.00
                and down >= 0.75
            )
            or (
                action == "CLOSE_SHORT"
                and ret60 >= 2.50
                and ret144 >= 3.00
                and _finite_float(row.get("up"), 0.0) >= 0.75
            ),
        )
        if risk_family and not structural_break:
            return False
        if risk_family and structural_break:
            return True
        if family.startswith("portfolio_"):
            inventory_reset = bool(
                family.startswith("portfolio_midnight_cleanup_")
                and hold_minutes >= 144 * 60.0,
            )
            if not inventory_reset:
                return False
            return True
        if family == "long_profit_take_exhaustion":
            absorption_release = bool(
                open_family == "capitulation_absorption_long"
                and hold_minutes >= 10.5 * 60.0
                and ret20 >= 0.80
                and ret60 >= 0.70
                and don >= 0.80
                and rsi <= 65.0,
            )
            decelerating_exhaustion = bool(
                open_family != "capitulation_absorption_long"
                and (
                    open_family == "early_session_reclaim_long"
                    or _finite_float((lot or {}).get("campaign_entry_ret60"), 0.0) >= 1.00
                )
                and hold_minutes
                >= (
                    113 * 60.0
                    if open_family == "early_session_reclaim_long"
                    else 1_300.0
                )
                and ret20 >= 0.40
                and ret60
                >= (
                    3.00
                    if open_family == "core_trend_long" and hold_minutes >= 72 * 60.0
                    else 1.40
                )
                and ret144 >= 1.00
                and ret288 >= (3.00 if hold_minutes >= 72 * 60.0 else 2.50)
                and np.isfinite(roi)
                and roi >= 0.50
                and rsi <= 64.5
                and (qvr >= 1.10 or hold_minutes < 30 * 60.0),
            )
            if not (absorption_release or decelerating_exhaustion):
                return False
            return True
        if family == "short_profit_take_exhaustion":
            cascade_exhaustion = bool(
                hold_minutes >= 180.0
                and ret60 <= -1.50
                and ret144 <= -2.00
                and don <= 0.35,
            )
            if not cascade_exhaustion:
                return False
            return True
    if (
        family == "long_profit_take_exhaustion"
        and symbol == "ETHUSDT"
        and open_family == "eth_anchor_long"
        and hold_minutes < max(
            0.0,
            float(config.production_eth_anchor_long_profit_min_hold_minutes),
        )
    ):
        return False
    eth_range_reclaim_main_leg_climax = (
        action == "CLOSE_LONG"
        and family == "long_profit_take_exhaustion"
        and symbol == "ETHUSDT"
        and open_family == "range_reclaim_long"
        and ret20 >= 3.0
        and ret60 >= 3.0
        and ret144 >= 2.5
        and ret288 >= 3.0
        and don >= 0.85
        and rsi >= 70.0
        and _finite_float(row.get("signal_cluster_age_min"), 0.0) >= 65.0
    )
    eth_range_reclaim_main_leg_profit_defer = (
        action == "CLOSE_LONG"
        and family == "long_profit_take_exhaustion"
        and symbol == "ETHUSDT"
        and open_family == "range_reclaim_long"
        and hold_minutes < 48 * 60.0
        and not eth_range_reclaim_main_leg_climax
    )
    eth_range_reclaim_main_leg_breakdown = (
        action == "CLOSE_LONG"
        and family == "long_risk_cut_reversal"
        and symbol == "ETHUSDT"
        and open_family == "range_reclaim_long"
        and ret20 <= -0.90
        and ret60 <= -1.80
        and ret144 <= -2.00
        and qvr >= 4.0
        and rsi <= 35.0
        and down >= 0.85
    )
    eth_range_reclaim_main_leg_risk_defer = (
        action == "CLOSE_LONG"
        and family == "long_risk_cut_reversal"
        and symbol == "ETHUSDT"
        and open_family == "range_reclaim_long"
        and hold_minutes < 48 * 60.0
        and not eth_range_reclaim_main_leg_breakdown
    )
    eth_range_reclaim_main_leg_cleanup_defer = (
        action == "CLOSE_LONG"
        and family.startswith("portfolio_")
        and symbol == "ETHUSDT"
        and open_family == "range_reclaim_long"
        and hold_minutes < 48 * 60.0
    )
    if (
        eth_range_reclaim_main_leg_profit_defer
        or eth_range_reclaim_main_leg_risk_defer
        or eth_range_reclaim_main_leg_cleanup_defer
    ):
        return False
    relief_impulse_still_expanding = (
        action == "CLOSE_LONG"
        and family == "long_profit_take_exhaustion"
        and open_family == "capitulation_absorption_long"
        and hold_minutes < 360.0
        and ret144 <= -1.00
        and ret20 >= 0.45
        and don >= 0.65
        and rsi >= 60.0
    )
    if relief_impulse_still_expanding:
        return False
    capitulation_recovery_still_intact = (
        action == "CLOSE_LONG"
        and family == "portfolio_midnight_cleanup_long"
        and open_family == "capitulation_absorption_long"
        and hold_minutes < 720.0
        and ret288 <= -2.00
        and ret60 >= 0.30
        and don >= 0.60
        and rsi >= 54.0
    )
    if capitulation_recovery_still_intact:
        return False
    failed_breakdown_cover = (
        action == "CLOSE_SHORT"
        and family == "short_profit_take_exhaustion"
        and ret20 <= -0.45
        and ret60 >= 0.50
        and ret144 >= 0.50
        and don <= 0.45
        and rsi <= 46.0
        and down >= 0.70
    )
    capitulation_recovery_profit_exit = (
        action == "CLOSE_LONG"
        and family == "long_profit_take_exhaustion"
        and open_family == "capitulation_absorption_long"
        and hold_minutes >= 480.0
        and np.isfinite(roi)
        and roi >= 0.80
        and ret288 <= -2.00
        and ret20 >= 0.80
        and ret60 >= 0.50
        and don >= 0.80
        and rsi >= 62.0
    )
    if (
        symbol == "PAXGUSDT"
        and action == "CLOSE_LONG"
    ):
        paxg_midnight_release = (
            family == "portfolio_midnight_cleanup_long"
            and session == "midnight"
            and hold_minutes >= 9000.0
            and ret288 >= 0.50
            and rsi <= 45.0
        )
        paxg_emergency_flush = (
            family == "long_risk_cut_reversal"
            and hold_minutes >= 9000.0
            and ret60 <= -1.50
            and down >= 0.90
            and rsi <= 30.0
        )
        if not (paxg_midnight_release or paxg_emergency_flush):
            return False
    if (
        symbol == "LTCUSDT"
        and action == "CLOSE_LONG"
        and family == "long_profit_take_exhaustion"
        and 21 <= int(signal_time.hour) <= 22
        and hold_minutes < 85.0
        and ret60 >= 1.50
        and ret144 >= 1.50
        and ret288 >= 4.0
        and rsi >= 68.0
        and qvr >= 3.0
    ):
        return False
    eth_morning_relief_profit_exit = (
        symbol == "ETHUSDT"
        and action == "CLOSE_LONG"
        and family == "long_profit_take_exhaustion"
        and int(signal_time.hour) == 9
        and 30 <= int(signal_time.minute) <= 45
        and ret20 >= 0.50
        and 0.20 <= ret60 <= 0.60
        and ret144 >= 1.50
        and -1.20 <= ret288 <= -0.50
        and don >= 0.80
        and _finite_float(row.get("taker"), 0.5) >= 0.65
        and qvr >= 5.0
        and rsi >= 64.0
        and np.isfinite(roi)
        and roi >= 1.0
    )
    ltc_post_spike_fade_profit_exit = (
        symbol == "LTCUSDT"
        and action == "CLOSE_LONG"
        and family == "long_profit_take_exhaustion"
        and int(signal_time.hour) == 22
        and hold_minutes >= 80.0
        and 1.00 <= ret20 <= 1.40
        and 1.40 <= ret60 <= 2.10
        and ret144 >= 1.40
        and ret288 >= 3.50
        and 0.60 <= don <= 0.80
        and 52.0 <= rsi <= 62.0
        and np.isfinite(roi)
        and roi >= 0.35
    )
    if (
        symbol == "FILUSDT"
        and action == "CLOSE_LONG"
        and family == "long_risk_cut_reversal"
        and int(signal_time.hour) < 17
        and ret144 > -2.0
    ):
        return False
    if (
        symbol == "FILUSDT"
        and action == "CLOSE_LONG"
        and family == "long_profit_take_exhaustion"
        and int(signal_time.hour) == 17
        and 30 <= int(signal_time.minute) <= 45
        and 900.0 <= hold_minutes <= 1500.0
        and ret20 >= 1.40
        and ret60 >= 2.50
        and ret144 >= 3.0
        and ret288 >= 5.50
        and qvr >= 3.0
        and 68.0 <= rsi <= 72.0
        and np.isfinite(roi)
        and roi >= 4.0
    ):
        return False
    eth_morning_short_cover = (
        family == "short_profit_take_exhaustion"
        and symbol == "ETHUSDT"
        and 7 <= int(event_time.hour) <= 10
        and pressure >= 80.0
    )
    eth_intraday_short_flush_cover = (
        action == "CLOSE_SHORT"
        and family == "short_profit_take_exhaustion"
        and symbol == "ETHUSDT"
        and 17 <= int(signal_time.hour) <= 18
        and ret20 <= -0.90
        and ret60 <= 0.55
        and ret288 <= -4.00
        and don <= 0.35
        and qvr >= 1.0
        and rsi <= 34.0
        and down >= 0.70
        and np.isfinite(roi)
        and roi >= 0.10
    )
    eth_regular_short_profit_cover = (
        action == "CLOSE_SHORT"
        and family == "short_profit_take_exhaustion"
        and symbol == "ETHUSDT"
        and 7 <= int(signal_time.hour) <= 18
        and ret20 <= -0.45
        and ret60 <= -0.70
        and don <= 0.25
        and rsi <= 40.0
        and down >= 0.85
        and np.isfinite(roi)
        and roi >= 0.10
    )
    eth_short_squeeze_risk_exit = (
        action == "CLOSE_SHORT"
        and family == "short_risk_cut_reversal"
        and symbol == "ETHUSDT"
        and ret20 >= 1.80
        and ret60 >= 1.50
        and don >= 0.70
        and rsi >= 88.0
        and qvr >= 10.0
    )
    eth_mature_short_squeeze_risk_exit = (
        action == "CLOSE_SHORT"
        and family == "short_risk_cut_reversal"
        and symbol == "ETHUSDT"
        and ret20 >= 1.20
        and ret60 >= 2.00
        and ret144 >= 3.00
        and don >= 0.85
        and rsi >= 78.0
        and qvr >= 8.0
    )
    eth_short_squeeze_risk_exit = bool(eth_short_squeeze_risk_exit or eth_mature_short_squeeze_risk_exit)
    eth_deep_long_risk_exit = (
        action == "CLOSE_LONG"
        and family == "long_risk_cut_reversal"
        and symbol == "ETHUSDT"
        and ret20 <= -0.90
        and ret60 <= -1.80
        and ret144 <= -2.00
        and qvr >= 4.0
        and rsi <= 35.0
        and down >= 0.85
    )
    eth_asia_blowoff_profit_exit = (
        family == "long_profit_take_exhaustion"
        and symbol == "ETHUSDT"
        and session == "asia_morning"
        and ret20 >= 3.0
        and ret60 >= 2.5
        and ret144 >= 3.0
        and don >= 0.80
        and qvr >= 15.0
        and rsi >= 75.0
    )
    eth_afternoon_pressure_profit_exit = (
        family == "long_profit_take_exhaustion"
        and symbol == "ETHUSDT"
        and 14 <= int(signal_time.hour) <= 16
        and ret20 >= 0.80
        and ret60 >= 0.45
        and don >= 0.90
        and rsi >= 63.0
        and pressure >= 80.0
    )
    eth_low_vol_afternoon_band_profit_exit = (
        family == "long_profit_take_exhaustion"
        and symbol == "ETHUSDT"
        and int(signal_time.hour) == 16
        and int(signal_time.minute) <= 10
        and ret20 >= 0.30
        and ret60 >= -0.10
        and ret144 >= 0.35
        and ret288 >= 0.20
        and don >= 0.80
        and _finite_float(row.get("taker"), 0.5) >= 0.55
        and qvr >= 1.0
        and rsi >= 60.0
        and np.isfinite(roi)
        and roi >= 0.10
    )
    eth_afternoon_low_qvr_relief_exit = (
        action == "CLOSE_LONG"
        and family == "long_profit_take_exhaustion"
        and symbol == "ETHUSDT"
        and 14 <= int(signal_time.hour) <= 16
        and ret60 >= 0.80
        and ret144 >= 1.20
        and 0.50 <= ret288 <= 1.20
        and don >= 0.60
        and qvr < 2.0
        and 44.0 <= rsi <= 55.0
        and np.isfinite(roi)
        and roi >= 0.0
    )
    eth_downtrend_relief_stall_profit_exit = (
        action == "CLOSE_LONG"
        and family == "long_profit_take_exhaustion"
        and symbol == "ETHUSDT"
        and 19 <= int(signal_time.hour) <= 20
        and 0.25 <= ret20 <= 0.90
        and ret60 >= -0.75
        and ret144 <= -1.20
        and ret288 <= -1.50
        and 0.58 <= don <= 0.78
        and 53.0 <= rsi <= 68.0
        and qvr >= 0.75
        and _finite_float(row.get("taker"), 0.5) <= 0.60
        and np.isfinite(roi)
        and roi >= 0.10
    )
    eth_evening_trend_profit_exit = (
        action == "CLOSE_LONG"
        and family == "long_profit_take_exhaustion"
        and symbol == "ETHUSDT"
        and (22 <= int(signal_time.hour) <= 23 or int(signal_time.hour) == 0)
        and ret60 >= 1.50
        and ret144 >= 1.40
        and ret288 >= 0.50
        and don >= 0.80
        and 58.0 <= rsi <= 69.0
        and qvr <= 2.50
        and np.isfinite(roi)
        and roi >= 0.50
    )
    eth_regular_long_profit_exit = (
        action == "CLOSE_LONG"
        and family == "long_profit_take_exhaustion"
        and symbol == "ETHUSDT"
        and 14 <= int(signal_time.hour) <= 19
        and ret20 >= 0.25
        and ret60 >= 0.45
        and ret144 >= -0.75
        and don >= 0.70
        and rsi >= 55.0
        and np.isfinite(roi)
        and roi >= 0.10
    )
    eth_overnight_blowoff_profit_exit = (
        action == "CLOSE_LONG"
        and family == "long_profit_take_exhaustion"
        and symbol == "ETHUSDT"
        and 1 <= int(signal_time.hour) <= 2
        and ret20 >= 2.50
        and ret60 >= 2.00
        and ret144 >= 2.50
        and don >= 0.75
        and rsi >= 68.0
        and np.isfinite(roi)
        and roi >= 0.10
    )
    ltc_midnight_profit_exit = (
        action == "CLOSE_LONG"
        and family == "long_profit_take_exhaustion"
        and symbol == "LTCUSDT"
        and int(signal_time.hour) == 0
        and ret60 >= 1.40
        and ret144 >= 1.50
        and ret288 >= 3.00
        and don >= 0.75
        and 55.0 <= rsi <= 62.0
        and np.isfinite(roi)
        and roi >= 0.10
    )
    doge_weak_bounce_short_profit_too_early = (
        action == "CLOSE_SHORT"
        and family == "short_profit_take_exhaustion"
        and symbol == "DOGEUSDT"
        and open_family == "weak_24h_bounce_short"
        and session == "night_trend"
        and not pd.isna(signal_time)
        and int(signal_time.hour) == 20
    )
    doge_weak_bounce_short_profit_too_early = bool(
        doge_weak_bounce_short_profit_too_early
        or (
            action == "CLOSE_SHORT"
            and family == "short_profit_take_exhaustion"
            and symbol == "DOGEUSDT"
            and open_family == "weak_24h_bounce_short"
            and session == "night_trend"
            and not pd.isna(signal_time)
            and int(signal_time.hour) == 21
            and int(signal_time.minute) < 40
        ),
    )
    if doge_weak_bounce_short_profit_too_early:
        return False
    midnight_trend_profit_exit = (
        family == "long_profit_take_exhaustion"
        and int(signal_time.hour) < 6
        and hold_minutes < float(config.production_early_profit_min_hold_minutes)
        and leg_score >= 4.75
        and event_score >= 0.04
        and ret60 >= 1.0
        and don >= 0.80
        and rsi >= 58.0
    )
    low_score_asia_trend_profit_exit = (
        family == "long_profit_take_exhaustion"
        and session == "asia_morning"
        and event_score < float(config.production_asia_profit_min_event_score)
        and leg_score >= 5.50
        and np.isfinite(roi)
        and roi >= 1.0
        and ret60 >= 1.0
        and don >= 0.70
    )
    late_cleanup_pressure_exit = (
        family.startswith("portfolio_session_cleanup_")
        and session in {"asia_morning", "late_morning_cleanup"}
        and event_score >= 0.25
        and leg_score >= 4.80
        and (down >= 0.95 or rsi <= 40.0)
    )
    late_morning_long_risk_exit = (
        action == "CLOSE_LONG"
        and family == "long_risk_cut_reversal"
        and session == "late_morning_cleanup"
        and not pd.isna(signal_time)
        and 10 * 60 + 15 <= int(signal_time.hour) * 60 + int(signal_time.minute) <= 10 * 60 + 45
        and pressure >= float(config.production_late_cleanup_min_pressure)
        and event_score >= float(config.production_risk_cut_min_event_score)
        and (ret60 <= -0.50 or ret20 <= -0.65)
        and (down >= 0.70 or rsi <= 45.0 or don <= 0.25)
    )
    stale_alt_long_cleanup_exit = (
        family == "portfolio_session_cleanup_long"
        and symbol != "ETHUSDT"
        and hold_minutes >= 900.0
        and np.isfinite(roi)
        and roi <= -0.30
        and ret60 <= -0.50
        and down >= 0.70
    )
    bch_high_qvr_short_cascade_exit = (
        family == "short_profit_take_exhaustion"
        and symbol == "BCHUSDT"
        and ret20 <= -1.0
        and ret60 <= -2.0
        and qvr >= 8.0
        and rsi <= 35.0
        and down >= 0.90
    )
    eth_capitulation_flush_risk_exit = (
        action == "CLOSE_LONG"
        and family == "long_risk_cut_reversal"
        and symbol == "ETHUSDT"
        and open_family == "capitulation_absorption_long"
        and ret20 <= -1.0
        and ret60 <= -2.0
        and ret144 <= -2.0
        and qvr >= 4.0
        and rsi <= 30.0
        and down >= 0.85
    )
    fil_inventory_release_profit_exit = (
        action == "CLOSE_LONG"
        and family == "long_profit_take_exhaustion"
        and symbol == "FILUSDT"
        and ret60 >= 1.0
        and ret144 >= 2.0
        and ret288 >= 4.0
        and qvr >= 2.0
        and 45.0 <= rsi <= 58.0
        and np.isfinite(roi)
        and roi >= 0.50
    )
    fil_momentum_release_profit_exit = (
        action == "CLOSE_LONG"
        and family == "long_profit_take_exhaustion"
        and symbol == "FILUSDT"
        and ret20 >= 1.20
        and ret60 >= 2.0
        and ret144 >= 2.0
        and qvr >= 3.0
        and don >= 0.95
        and rsi >= 64.0
        and np.isfinite(roi)
        and roi >= 0.50
    )
    fil_second_thrust_profit_exit = (
        action == "CLOSE_LONG"
        and family == "long_profit_take_exhaustion"
        and symbol == "FILUSDT"
        and int(signal_time.hour) == 18
        and int(signal_time.minute) <= 10
        and 900.0 <= hold_minutes <= 1500.0
        and ret20 >= 2.0
        and ret60 >= 3.50
        and ret144 >= 4.0
        and ret288 >= 6.50
        and don >= 0.90
        and qvr >= 2.50
        and rsi >= 70.0
        and np.isfinite(roi)
        and roi >= 4.0
    )
    fil_profit_release_exit = (
        fil_inventory_release_profit_exit
        or fil_momentum_release_profit_exit
        or fil_second_thrust_profit_exit
    )
    fil_failed_absorption_risk_exit = (
        action == "CLOSE_LONG"
        and family == "long_risk_cut_reversal"
        and symbol == "FILUSDT"
        and open_family == "capitulation_absorption_long"
        and ret60 <= -1.50
        and ret144 <= -2.0
        and qvr >= 4.0
        and rsi <= 35.0
        and down >= 0.90
        and np.isfinite(roi)
        and roi <= 0.20
    )
    cfx_short_rebound_risk_exit = False

    eth_partial_short_squeeze_risk_defer = (
        action == "CLOSE_SHORT"
        and family == "short_risk_cut_reversal"
        and symbol == "ETHUSDT"
        and hold_minutes >= 900.0
        and ret20 >= 1.20
        and ret60 >= 1.20
        and qvr >= 4.0
        and rsi >= 65.0
        and don >= 0.75
        and ret144 < 2.0
        and rsi < 85.0
    )
    eth_partial_short_profit_defer = (
        action == "CLOSE_SHORT"
        and family == "short_profit_take_exhaustion"
        and symbol == "ETHUSDT"
        and 7 <= int(signal_time.hour) <= 10
        and ret20 <= -0.80
        and -1.50 < ret60 <= -0.50
        and ret144 > -1.0
        and ret288 > -2.50
        and qvr >= 2.0
        and rsi <= 42.0
        and down >= 0.85
        and np.isfinite(roi)
        and roi < 2.0
    )
    eth_low_quality_morning_short_profit_defer = (
        action == "CLOSE_SHORT"
        and family == "short_profit_take_exhaustion"
        and symbol == "ETHUSDT"
        and 7 <= int(signal_time.hour) <= 10
        and ret20 <= -1.0
        and ret60 <= -1.50
        and ret144 > -2.0
        and qvr < 4.0
        and rsi <= 35.0
        and down >= 0.85
        and np.isfinite(roi)
        and roi < 2.0
    )
    eth_countertrend_short_profit_still_expanding = (
        bool(config.production_eth_countertrend_short_exit_confirmation)
        and action == "CLOSE_SHORT"
        and family == "short_profit_take_exhaustion"
        and symbol == "ETHUSDT"
        and 7 <= int(signal_time.hour) <= 10
        and 120.0 <= hold_minutes <= 600.0
        and np.isfinite(roi)
        and 0.50 <= roi < 2.0
        and ret20 <= -0.50
        and ret60 <= -0.70
        and ret144 >= 2.0
        and qvr < 4.0
        and rsi <= 35.0
        and down >= 0.85
        and _finite_float(row.get("taker"), 0.5) < 0.50
        and _finite_float(row.get("signal_cluster_age_min"), 0.0) < 40.0
    )
    eth_regular_long_inventory_profit_defer = (
        action == "CLOSE_LONG"
        and family == "long_profit_take_exhaustion"
        and symbol == "ETHUSDT"
        and session == "regular"
        and 9 <= int(signal_time.hour) <= 16
        and hold_minutes < 900.0
        and ret144 < 1.25
        and qvr < 8.0
        and rsi < 75.0
        and not eth_morning_relief_profit_exit
        and not eth_low_vol_afternoon_band_profit_exit
        and not eth_afternoon_pressure_profit_exit
        and not eth_afternoon_low_qvr_relief_exit
    )
    eth_late_morning_inventory_profit_defer = (
        action == "CLOSE_LONG"
        and family == "long_profit_take_exhaustion"
        and symbol == "ETHUSDT"
        and 10 <= int(signal_time.hour) <= 11
        and hold_minutes < 480.0
        and ret144 < 0.0
        and qvr < 4.0
        and rsi < 62.0
        and np.isfinite(roi)
        and roi < 2.10
    )
    eth_evening_pre_overnight_profit_defer = (
        action == "CLOSE_LONG"
        and family == "long_profit_take_exhaustion"
        and symbol == "ETHUSDT"
        and 19 <= int(signal_time.hour) <= 20
        and hold_minutes < 1200.0
        and ret144 < 2.0
        and ret288 < 1.50
        and qvr < 10.0
        and rsi < 76.0
        and np.isfinite(roi)
        and roi < 2.50
        and not eth_downtrend_relief_stall_profit_exit
        and not eth_evening_trend_profit_exit
    )
    ltc_late_trend_inventory_profit_defer = (
        action == "CLOSE_LONG"
        and family == "long_profit_take_exhaustion"
        and symbol == "LTCUSDT"
        and 16 <= int(signal_time.hour) <= 22
        and hold_minutes < 900.0
        and ret60 >= 0.90
        and ret144 >= 1.50
        and ret288 >= 2.50
        and qvr >= 3.0
        and rsi >= 68.0
    )
    if (
        eth_partial_short_squeeze_risk_defer
        or eth_partial_short_profit_defer
        or eth_low_quality_morning_short_profit_defer
        or eth_countertrend_short_profit_still_expanding
        or eth_regular_long_inventory_profit_defer
        or eth_late_morning_inventory_profit_defer
        or eth_evening_pre_overnight_profit_defer
        or ltc_late_trend_inventory_profit_defer
    ):
        return False

    eth_blowoff_addon_harvest = (
        bool(config.production_eth_blowoff_handoff or config.production_eth_blowoff_ladder)
        and symbol == "ETHUSDT"
        and action == "CLOSE_SHORT"
        and family == "short_profit_take_exhaustion"
        and ret60 >= 4.0
        and ret144 >= 4.0
        and (seller >= 0.58 or (config.production_eth_blowoff_ladder and down >= 0.60))
        and 0.70 <= qvr <= 3.0
        and np.isfinite(roi)
        and roi >= 0.10
    )

    if bool(config.production_inventory_exit_quality_gate):
        long_inventory_symbol = symbol in {"CFXUSDT", "ETCUSDT", "FILUSDT"}
        short_inventory_symbol = symbol in {"BCHUSDT", "CFXUSDT", "ETCUSDT"}
        inventory_open_family = open_family in {
            "core_trend_long",
            "midnight_breadth_breakout_long",
            "session_pullback_reclaim_long",
            "core_downtrend_short",
        }
        alt_late_inventory = (
            inventory_open_family
            and (
                (action == "CLOSE_LONG" and long_inventory_symbol)
                or (action == "CLOSE_SHORT" and short_inventory_symbol)
            )
        )
        cfx_short_rebound_risk_exit = (
            symbol == "CFXUSDT"
            and action == "CLOSE_SHORT"
            and family == "short_risk_cut_reversal"
            and hold_minutes >= 900.0
            and ret20 >= 0.45
            and ret144 <= -2.50
            and qvr >= 4.0
            and rsi >= 57.0
            and _finite_float(row.get("up"), 0.0) >= 0.90
        )
        inventory_strong_short_profit_exit = (
            symbol == "BCHUSDT"
            and family == "short_profit_take_exhaustion"
            and ret20 <= -1.20
            and ret60 <= -1.50
            and qvr >= 4.0
            and rsi <= 35.0
            and down >= 0.88
        )
        inventory_hold_floor = 1800.0 if symbol == "CFXUSDT" and action == "CLOSE_SHORT" else (3300.0 if action == "CLOSE_SHORT" else 2400.0)
        if alt_late_inventory and hold_minutes < inventory_hold_floor:
            if family in {
                "long_profit_take_exhaustion",
                "long_risk_cut_reversal",
                "long_relief_bounce_exit",
                "short_profit_take_exhaustion",
                "short_risk_cut_reversal",
            }:
                if not (inventory_strong_short_profit_exit or fil_profit_release_exit or cfx_short_rebound_risk_exit):
                    return False
            if family.startswith("portfolio_") and session in {"midnight", "asia_morning", "evening_cleanup"}:
                return False

        if (
            symbol == "FILUSDT"
            and action == "CLOSE_LONG"
            and inventory_open_family
            and hold_minutes < 3600.0
            and not fil_profit_release_exit
        ):
            return False

        eth_asia_short_profit_cover = (
            symbol == "ETHUSDT"
            and action == "CLOSE_SHORT"
            and family == "short_profit_take_exhaustion"
            and 8 <= int(signal_time.hour) <= 9
            and ret60 <= -1.0
            and rsi <= 42.0
            and down >= 0.85
            and np.isfinite(roi)
            and roi >= 0.50
        )
        eth_midnight_short_profit_cover = (
            symbol == "ETHUSDT"
            and action == "CLOSE_SHORT"
            and family == "short_profit_take_exhaustion"
            and 0 <= int(signal_time.hour) <= 1
            and ret60 <= -0.50
            and rsi <= 42.0
            and don <= 0.10
            and down >= 0.70
            and np.isfinite(roi)
            and roi >= 0.10
        )
        doge_cascade_short_profit = (
            symbol == "DOGEUSDT"
            and (
                (ret60 <= -2.0 and ret144 <= -1.50 and np.isfinite(roi) and roi >= 1.50)
                or (ret20 <= -1.50 and ret60 <= -2.0 and rsi <= 30.0 and down >= 0.90)
            )
        )
        strong_short_profit_exit = (
            eth_asia_short_profit_cover
            or eth_midnight_short_profit_cover
            or eth_intraday_short_flush_cover
            or eth_regular_short_profit_cover
            or doge_cascade_short_profit
            or eth_blowoff_addon_harvest
            or failed_breakdown_cover
            or (
                ret20 <= -1.20
                and ret60 <= -1.50
                and qvr >= 4.0
                and rsi <= 35.0
                and down >= 0.88
            )
        )
        strong_long_profit_exit = (
            fil_profit_release_exit
            or capitulation_recovery_profit_exit
            or eth_morning_relief_profit_exit
            or eth_low_vol_afternoon_band_profit_exit
            or eth_afternoon_low_qvr_relief_exit
            or eth_downtrend_relief_stall_profit_exit
            or eth_evening_trend_profit_exit
            or eth_regular_long_profit_exit
            or eth_overnight_blowoff_profit_exit
            or ltc_post_spike_fade_profit_exit
            or ltc_midnight_profit_exit
            or
            (
                ret20 >= 1.20
                and ret60 >= 1.50
                and qvr >= 3.50
                and rsi >= 68.0
                and don >= 0.75
            )
            or (
                symbol == "ETHUSDT"
                and ret20 >= 2.50
                and ret60 >= 1.50
                and rsi >= 75.0
            )
        )
        strong_long_risk_exit = (
            fil_failed_absorption_risk_exit
            or eth_deep_long_risk_exit
            or late_morning_long_risk_exit
            or (
                ret20 <= -1.20
                and ret60 <= -1.50
                and qvr >= 4.0
                and rsi <= 35.0
                and down >= 0.88
            )
        )
        strong_short_risk_exit = (
            ret20 >= 1.20
            and ret60 >= 1.50
            and qvr >= 4.0
            and rsi >= 65.0
            and don >= 0.75
        )
        strong_short_risk_exit = bool(
            strong_short_risk_exit or eth_short_squeeze_risk_exit or cfx_short_rebound_risk_exit,
        )
        if (
            family == "short_profit_take_exhaustion"
            and hold_minutes < 900.0
            and np.isfinite(roi)
            and roi < 2.0
            and not strong_short_profit_exit
        ):
            return False
        if family == "long_profit_take_exhaustion" and np.isfinite(roi):
            if symbol == "ETHUSDT":
                if hold_minutes < 480.0 and roi < 1.0 and not strong_long_profit_exit:
                    return False
            elif hold_minutes < 900.0 and roi < 1.20 and not strong_long_profit_exit:
                return False
        if (
            family == "long_risk_cut_reversal"
            and symbol != "ETHUSDT"
            and hold_minutes < 900.0
            and np.isfinite(roi)
            and roi > -2.20
            and not strong_long_risk_exit
        ):
            return False
        if (
            family == "short_risk_cut_reversal"
            and symbol != "ETHUSDT"
            and hold_minutes < 900.0
            and np.isfinite(roi)
            and roi > -2.20
            and not strong_short_risk_exit
        ):
            return False

    if bool(config.production_anchor_exit_deferral):
        if symbol == "ETHUSDT":
            anchor_short_hold_floor = (
                max(900.0, float(config.production_eth_anchor_short_min_hold_minutes))
                if open_family == "eth_anchor_exhaustion_short"
                else 900.0
            )
            if (
                family == "short_risk_cut_reversal"
                and hold_minutes < anchor_short_hold_floor
                and not eth_short_squeeze_risk_exit
            ):
                return False
            if family == "portfolio_session_cleanup_short" and hold_minutes < anchor_short_hold_floor:
                return False
            if family == "long_risk_cut_reversal" and hold_minutes < 900.0 and not eth_deep_long_risk_exit:
                return False
            if (
                family == "long_profit_take_exhaustion"
                and session == "regular"
                and hold_minutes < 900.0
                and event_score < 0.20
                and not eth_morning_relief_profit_exit
                and not eth_low_vol_afternoon_band_profit_exit
                and not eth_afternoon_low_qvr_relief_exit
                and not eth_downtrend_relief_stall_profit_exit
                and not eth_evening_trend_profit_exit
                and not eth_regular_long_profit_exit
                and not eth_overnight_blowoff_profit_exit
            ):
                return False
            if (
                family == "portfolio_session_cleanup_long"
                and session == "late_morning_cleanup"
                and hold_minutes < 900.0
                and np.isfinite(roi)
                and abs(roi) < 0.30
            ):
                return False
        if (
            symbol == "PAXGUSDT"
            and family == "long_profit_take_exhaustion"
            and session != "midnight"
            and hold_minutes < 7 * 24 * 60.0
        ):
            return False

    if family.startswith("portfolio_") and hold_minutes < float(config.production_portfolio_min_hold_minutes):
        return False
    if (
        family.startswith("portfolio_midnight_cleanup_")
        and float(config.production_midnight_cleanup_min_hold_minutes) > 0.0
        and hold_minutes < float(config.production_midnight_cleanup_min_hold_minutes)
    ):
        return False
    if (
        family == "portfolio_midnight_cleanup_long"
        and symbol == "LTCUSDT"
        and int(signal_time.hour) == 0
        and int(signal_time.minute) <= 10
        and ret60 >= 1.20
        and ret144 >= 1.50
        and ret288 >= 3.00
        and don >= 0.85
        and rsi >= 64.0
    ):
        return False
    if family == "portfolio_session_cleanup_short" and session == "asia_morning" and not late_cleanup_pressure_exit:
        return False
    if (
        family == "portfolio_session_cleanup_short"
        and symbol == "BCHUSDT"
        and session == "evening_cleanup"
        and ret20 >= 0.40
        and rsi >= 55.0
        and qvr >= 2.0
    ):
        return False
    if (
        family == "portfolio_session_cleanup_short"
        and symbol == "BCHUSDT"
        and session == "evening_cleanup"
        and event_score < 0.10
        and ret60 > -1.0
        and down < 0.70
    ):
        return False
    if family == "portfolio_session_cleanup_long" and session == "asia_morning":
        if (
            symbol == "ETHUSDT"
            and event_time.hour == 6
            and event_time.minute == 0
            and ret20 >= 0.40
            and ret60 >= 0.50
            and qvr >= 4.0
            and rsi >= 64.0
        ):
            return False
        return symbol == "ETHUSDT" and pressure >= float(config.production_asia_cleanup_min_pressure)
    if family == "portfolio_session_cleanup_long" and session == "evening_cleanup":
        if stale_alt_long_cleanup_exit:
            return True
        eth_evening_drift_cleanup = (
            symbol == "ETHUSDT"
            and pressure >= 45.0
            and ret144 >= 0.80
            and ret288 >= 0.80
            and rsi >= 52.0
            and hold_minutes >= 480.0
        )
        if eth_evening_drift_cleanup:
            return True
        return pressure >= float(config.production_evening_cleanup_min_pressure)
    if family.startswith("portfolio_session_cleanup_") and session == "late_morning_cleanup":
        eth_flat_late_morning_cleanup_defer = (
            family == "portfolio_session_cleanup_long"
            and symbol == "ETHUSDT"
            and ret288 >= 1.0
            and qvr < 5.0
            and np.isfinite(roi)
            and abs(roi) < 0.30
        )
        if eth_flat_late_morning_cleanup_defer:
            return False
        return (
            pressure >= float(config.production_late_cleanup_min_pressure)
            or late_cleanup_pressure_exit
            or stale_alt_long_cleanup_exit
        )
    if (
        family.startswith("portfolio_midnight_cleanup_")
        and session == "midnight"
        and max_score > float(config.production_midnight_cleanup_max_score)
    ):
        return False
    if (
        family == "portfolio_midnight_cleanup_short"
        and symbol == "ETHUSDT"
        and session == "midnight"
        and int(signal_time.hour) == 0
        and int(signal_time.minute) in {0, 5}
        and ret60 <= -0.55
        and ret20 > -0.12
        and rsi <= 45.0
    ):
        return False
    if (
        family == "portfolio_midnight_cleanup_long"
        and symbol == "ETHUSDT"
        and abs(ret60) <= 0.40
        and qvr < 2.0
        and 38.0 <= rsi <= 60.0
        and down < 0.95
    ):
        return False
    if (
        family == "portfolio_session_cleanup_short"
        and symbol == "ETHUSDT"
        and ret60 >= 0.80
        and ret144 >= 1.20
        and don >= 0.75
        and rsi >= 55.0
    ):
        return False
    if (
        family == "long_relief_bounce_exit"
        and symbol == "ETHUSDT"
        and hold_minutes < 720.0
        and ret144 <= -1.0
        and ret288 <= -1.0
        and qvr < 3.0
        and np.isfinite(roi)
        and roi > -0.50
        and ret60 > -1.50
    ):
        return False
    if family in {"long_risk_cut_reversal", "short_risk_cut_reversal"}:
        late_pre_cleanup_long_risk = (
            action == "CLOSE_LONG"
            and family == "long_risk_cut_reversal"
            and int(signal_time.hour) == 9
            and int(signal_time.minute) >= 5
            and ret60 <= -0.30
            and don <= 0.20
            and rsi <= 45.0
            and down >= 0.88
        )
        eth_short_squeeze_risk_exit = eth_short_squeeze_risk_exit or (
            action == "CLOSE_SHORT"
            and family == "short_risk_cut_reversal"
            and symbol == "ETHUSDT"
            and ret20 >= 1.80
            and ret60 >= 1.50
            and don >= 0.75
            and rsi >= 88.0
            and qvr >= 10.00
        )
        eth_intact_trend_risk_defer = (
            action == "CLOSE_LONG"
            and family == "long_risk_cut_reversal"
            and symbol == "ETHUSDT"
            and ret288 >= 2.50
            and qvr < 20.0
            and rsi >= 15.0
            and open_family
            in {
                "core_trend_long",
                "eth_anchor_long",
                "capitulation_absorption_long",
                "range_reclaim_long",
                "session_pullback_reclaim_long",
                "early_session_reclaim_long",
            }
        )
        eth_capitulation_low_volume_risk_defer = (
            action == "CLOSE_LONG"
            and family == "long_risk_cut_reversal"
            and symbol == "ETHUSDT"
            and open_family == "capitulation_absorption_long"
            and session == "late_morning_cleanup"
            and ret60 > -2.00
            and ret288 > -3.00
            and qvr < 3.0
            and rsi > 30.0
            and hold_minutes < 900.0
        )
        eth_intraday_flush_risk_defer = (
            action == "CLOSE_LONG"
            and family == "long_risk_cut_reversal"
            and symbol == "ETHUSDT"
            and open_family
            in {
                "capitulation_absorption_long",
                "range_reclaim_long",
                "core_trend_long",
            }
            and hold_minutes < 360.0
            and qvr < 12.0
            and rsi >= 24.0
            and ret288 > -7.0
        )
        eth_pre_breakdown_risk_defer = (
            action == "CLOSE_LONG"
            and family == "long_risk_cut_reversal"
            and symbol == "ETHUSDT"
            and ret288 > -1.50
            and ret60 > -1.80
            and qvr < 10.0
            and rsi >= 25.0
        )
        if (
            eth_intact_trend_risk_defer
            or eth_capitulation_low_volume_risk_defer
            or eth_intraday_flush_risk_defer
            or eth_pre_breakdown_risk_defer
        ):
            return False
        if (
            event_score < float(config.production_risk_cut_min_event_score)
            and not late_pre_cleanup_long_risk
            and not eth_short_squeeze_risk_exit
            and not eth_capitulation_flush_risk_exit
            and not cfx_short_rebound_risk_exit
            and not late_morning_long_risk_exit
        ):
            return False
        if (
            float(config.production_unclustered_risk_min_loss_pct) > 0.0
            and np.isfinite(roi)
            and roi > -float(config.production_unclustered_risk_min_loss_pct)
        ):
            return False
    if family in {"long_profit_take_exhaustion", "short_profit_take_exhaustion"}:
        eth_intact_trend_profit_defer = (
            family == "long_profit_take_exhaustion"
            and symbol == "ETHUSDT"
            and session == "regular"
            and hold_minutes < 600.0
            and ret288 >= 2.50
            and ret20 < 3.00
            and qvr < 8.0
            and open_family
            in {
                "core_trend_long",
                "eth_anchor_long",
                "range_reclaim_long",
                "session_pullback_reclaim_long",
                "early_session_reclaim_long",
            }
        )
        eth_flat_trend_profit_defer = (
            family == "long_profit_take_exhaustion"
            and symbol == "ETHUSDT"
            and session in {"regular", "late_morning_cleanup"}
            and hold_minutes < 720.0
            and ret288 >= 2.50
            and ret20 < 0.80
            and qvr < 3.0
            and np.isfinite(roi)
            and abs(roi) < 0.30
            and open_family
            in {
                "core_trend_long",
                "eth_anchor_long",
                "range_reclaim_long",
                "session_pullback_reclaim_long",
                "early_session_reclaim_long",
            }
        )
        eth_capitulation_low_quality_profit_defer = (
            family == "long_profit_take_exhaustion"
            and symbol == "ETHUSDT"
            and session == "regular"
            and open_family == "capitulation_absorption_long"
            and hold_minutes < 900.0
            and ret20 < 0.40
            and ret144 < 0.0
            and qvr < 4.0
            and np.isfinite(roi)
            and roi < 1.50
        )
        if eth_intact_trend_profit_defer or eth_flat_trend_profit_defer or eth_capitulation_low_quality_profit_defer:
            return False
        if (
            family == "long_profit_take_exhaustion"
            and symbol in {"SOLUSDT", "ETHUSDT"}
            and int(signal_time.hour) < 6
            and ret60 >= 1.50
            and ret288 >= 2.0
            and qvr >= 15.0
            and rsi >= 75.0
            and hold_minutes < 480.0
        ):
            return False
        if (
            family == "short_profit_take_exhaustion"
            and symbol == "DOGEUSDT"
            and session == "night_trend"
            and int(signal_time.hour) == 20
            and ret20 <= -1.50
            and qvr >= 20.0
            and rsi <= 25.0
        ):
            return False
        if (
            family == "short_profit_take_exhaustion"
            and symbol == "ETHUSDT"
            and session == "regular"
            and 7 <= int(signal_time.hour) <= 8
            and hold_minutes < 600.0
            and ret20 <= -0.85
            and ret60 <= -1.40
            and ret144 >= 0.80
            and ret288 >= 0.80
            and qvr >= 8.0
            and rsi <= 28.0
        ):
            return False
        if (
            family == "short_profit_take_exhaustion"
            and symbol == "PAXGUSDT"
            and session == "night_trend"
            and int(signal_time.hour) == 20
            and ret60 <= -0.50
            and ret144 <= -0.50
            and qvr >= 8.0
            and rsi <= 30.0
        ):
            return False
        if (
            float(config.production_unclustered_profit_min_roi_pct) > 0.0
            and np.isfinite(roi)
            and session not in {"midnight", "asia_morning", "late_morning_cleanup", "evening_cleanup"}
            and not eth_morning_short_cover
            and not eth_intraday_short_flush_cover
            and not eth_regular_short_profit_cover
            and not eth_afternoon_pressure_profit_exit
            and not eth_low_vol_afternoon_band_profit_exit
            and not eth_afternoon_low_qvr_relief_exit
            and not eth_downtrend_relief_stall_profit_exit
            and not eth_evening_trend_profit_exit
            and not eth_regular_long_profit_exit
            and not eth_overnight_blowoff_profit_exit
            and not midnight_trend_profit_exit
            and not ltc_post_spike_fade_profit_exit
            and not ltc_midnight_profit_exit
            and not bch_high_qvr_short_cascade_exit
            and not eth_blowoff_addon_harvest
            and roi < float(config.production_unclustered_profit_min_roi_pct)
        ):
            return False
    if (
        family == "long_profit_take_exhaustion"
        and session == "regular"
        and event_time.hour < 6
        and hold_minutes < float(config.production_early_profit_min_hold_minutes)
        and not midnight_trend_profit_exit
    ):
        return False
    if (
        family == "long_profit_take_exhaustion"
        and session == "asia_morning"
        and event_score < float(config.production_asia_profit_min_event_score)
        and not eth_asia_blowoff_profit_exit
        and not low_score_asia_trend_profit_exit
    ):
        return False
    if (
        family == "long_profit_take_exhaustion"
        and symbol == "DOGEUSDT"
        and session == "night_trend"
        and 21 <= int(signal_time.hour) <= 22
        and hold_minutes >= 900.0
        and event_score <= 0.120
        and ret60 < 2.50
        and qvr < 5.00
        and rsi < 75.0
    ):
        return False
    if (
        family == "short_profit_take_exhaustion"
        and session == "night_trend"
        and float(config.production_night_short_profit_min_event_score) > 0.0
        and event_score < float(config.production_night_short_profit_min_event_score)
        and not eth_blowoff_addon_harvest
    ):
        return False
    if (
        family == "short_profit_take_exhaustion"
        and session == "regular"
        and bool(config.production_regular_short_profit_quality_gate)
    ):
        quality_ok = (
            event_score >= 0.10
            or qvr >= 2.0
            or rsi <= 30.0
            or (ret20 <= -1.0 and down >= 0.78)
            or eth_regular_short_profit_cover
            or (leg_score >= 4.80 and ret60 <= -2.0 and ret144 <= -2.0 and down >= 0.60)
        )
        if not quality_ok:
            return False
    if (
        family == "short_profit_take_exhaustion"
        and symbol == "CFXUSDT"
        and event_time.hour < 16
        and hold_minutes < 480.0
    ):
        return False
    if family == "long_risk_cut_reversal" and symbol == "CFXUSDT" and int(signal_time.hour) < 15:
        return False
    if (
        family == "short_profit_take_exhaustion"
        and symbol == "BCHUSDT"
        and session == "evening_cleanup"
        and event_time.hour == 17
        and hold_minutes < 360.0
        and not (ret20 <= -1.0 and qvr >= 8.0 and rsi <= 35.0)
    ):
        return False
    if (
        family == "short_profit_take_exhaustion"
        and symbol == "BCHUSDT"
        and session == "regular"
        and hold_minutes < 600.0
        and qvr < 4.00
        and rsi > 35.0
    ):
        return False
    if (
        family == "short_profit_take_exhaustion"
        and session == "night_trend"
        and bool(config.production_night_short_profit_spike_deferral)
        and hold_minutes < 420.0
        and not doge_cascade_short_profit
        and (
            (qvr >= 8.0 and rsi <= 35.0 and ret20 <= -1.0)
            or (ret20 <= -1.2 and rsi <= 38.0 and down >= 0.78)
        )
    ):
        return False
    if (
        family == "short_profit_take_exhaustion"
        and session == "regular"
        and hold_minutes > float(config.production_regular_short_profit_max_hold_minutes)
        and not bch_high_qvr_short_cascade_exit
    ):
        return False
    if (
        family == "short_risk_cut_reversal"
        and event_time.hour < 6
        and hold_minutes > float(config.production_early_short_risk_max_hold_minutes)
    ):
        return False
    return True


def _point_return_pct(direction: str, open_price: float, close_price: float) -> float:
    open_value = _finite_float(open_price, np.nan)
    close_value = _finite_float(close_price, np.nan)
    if not np.isfinite(open_value) or not np.isfinite(close_value) or open_value <= 0.0:
        return np.nan
    if str(direction).upper() == "SHORT":
        return (open_value - close_value) / open_value * 100.0
    return (close_value - open_value) / open_value * 100.0


def _realized_pnl_usdt(direction: str, open_price: float, close_price: float, quantity: float) -> float:
    open_value = _finite_float(open_price, np.nan)
    close_value = _finite_float(close_price, np.nan)
    qty = _finite_float(quantity, np.nan)
    if not np.isfinite(open_value) or not np.isfinite(close_value) or not np.isfinite(qty):
        return np.nan
    if str(direction).upper() == "SHORT":
        return (open_value - close_value) * qty
    return (close_value - open_value) * qty


def _margin_roi_pct(pnl: float, risk_basis: float) -> float:
    pnl_value = _finite_float(pnl, np.nan)
    risk_value = _finite_float(risk_basis, np.nan)
    if not np.isfinite(pnl_value) or not np.isfinite(risk_value) or risk_value <= 0.0:
        return np.nan
    return pnl_value / risk_value * 100.0


def _profit_factor_from_pnl(pnl: pd.Series) -> float:
    values = pd.to_numeric(pnl, errors="coerce").dropna()
    gains = float(values[values > 0.0].sum())
    losses = float(values[values < 0.0].sum())
    if abs(losses) <= 1e-12:
        return float("inf") if gains > 0.0 else 0.0
    return gains / abs(losses)


def _max_drawdown_from_series(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").fillna(0.0)
    if clean.empty:
        return 0.0
    curve = clean.cumsum()
    drawdown = curve - curve.cummax()
    return float(drawdown.min())


def _max_concurrent_ledger_risk_basis(frame: pd.DataFrame) -> float:
    events: list[tuple[pd.Timestamp, int, float]] = []
    for _, row in frame.iterrows():
        risk_basis = _finite_float(row.get("risk_basis_usdt"), np.nan)
        open_time = pd.to_datetime(row.get("open_time"), errors="coerce")
        close_time = pd.to_datetime(row.get("close_time"), errors="coerce")
        if not np.isfinite(risk_basis) or risk_basis <= 0.0 or pd.isna(open_time) or pd.isna(close_time):
            continue
        if pd.Timestamp(close_time) < pd.Timestamp(open_time):
            continue
        events.append((pd.Timestamp(open_time), 0, risk_basis))
        events.append((pd.Timestamp(close_time), 1, -risk_basis))
    if not events:
        return 0.0
    active = 0.0
    peak = 0.0
    for _, _, delta in sorted(events, key=lambda item: (item[0], item[1])):
        active += delta
        peak = max(peak, active)
    return float(peak)


def _precision_match_stats(
    positions: pd.DataFrame,
    *,
    open_ops_count: int,
    close_ops_count: int,
    tolerance_minutes: float,
    price_tolerance_bps: float,
) -> dict[str, float | int]:
    tolerance = max(0.0, float(tolerance_minutes))
    price_bps = max(0.0, float(price_tolerance_bps))
    open_denominator = max(0, int(open_ops_count))
    close_denominator = max(0, int(close_ops_count))
    position_denominator = max(open_denominator, close_denominator, len(positions) if positions is not None else 0)
    empty_stats: dict[str, float | int] = {
        "precision_match_tolerance_minutes": tolerance,
        "precision_match_price_bps": price_bps,
        "precision_open_time_matches": 0,
        "precision_close_time_matches": 0,
        "precision_both_time_positions": 0,
        "precision_open_point_matches": 0,
        "precision_close_point_matches": 0,
        "precision_both_point_positions": 0,
        "precision_open_time_match_rate_pct": 0.0,
        "precision_close_time_match_rate_pct": 0.0,
        "precision_both_time_position_rate_pct": 0.0,
        "precision_open_point_match_rate_pct": 0.0,
        "precision_close_point_match_rate_pct": 0.0,
        "precision_both_point_position_rate_pct": 0.0,
        "precision_open_abs_price_diff_bps_median": np.nan,
        "precision_open_abs_price_diff_bps_p90": np.nan,
        "precision_close_abs_price_diff_bps_median": np.nan,
        "precision_close_abs_price_diff_bps_p90": np.nan,
    }
    if positions is None or positions.empty:
        return empty_stats
    frame = positions.copy()
    required = {
        "open_match_status",
        "close_match_status",
        "open_delta_min",
        "close_delta_min",
        "leader_open_price",
        "leader_close_price",
        "strategy_signal_open_price",
        "strategy_signal_close_price",
    }
    if not required.issubset(set(frame.columns)):
        return empty_stats
    for col in required - {"open_match_status", "close_match_status"}:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    open_abs_bps = (
        (frame["strategy_signal_open_price"] - frame["leader_open_price"])
        .div(frame["leader_open_price"].replace(0.0, np.nan))
        .abs()
        * 10000.0
    )
    close_abs_bps = (
        (frame["strategy_signal_close_price"] - frame["leader_close_price"])
        .div(frame["leader_close_price"].replace(0.0, np.nan))
        .abs()
        * 10000.0
    )
    open_time = frame["open_match_status"].astype(str).eq("MATCH") & frame["open_delta_min"].le(tolerance)
    close_time = frame["close_match_status"].astype(str).eq("MATCH") & frame["close_delta_min"].le(tolerance)
    open_point = open_time & open_abs_bps.le(price_bps)
    close_point = close_time & close_abs_bps.le(price_bps)
    both_time = open_time & close_time
    both_point = open_point & close_point

    def _rate(count: int, denominator: int) -> float:
        return (float(count) / float(denominator) * 100.0) if denominator > 0 else 0.0

    matched_open_bps = open_abs_bps[frame["open_match_status"].astype(str).eq("MATCH")].dropna()
    matched_close_bps = close_abs_bps[frame["close_match_status"].astype(str).eq("MATCH")].dropna()
    open_time_count = int(open_time.sum())
    close_time_count = int(close_time.sum())
    both_time_count = int(both_time.sum())
    open_point_count = int(open_point.sum())
    close_point_count = int(close_point.sum())
    both_point_count = int(both_point.sum())
    return {
        "precision_match_tolerance_minutes": tolerance,
        "precision_match_price_bps": price_bps,
        "precision_open_time_matches": open_time_count,
        "precision_close_time_matches": close_time_count,
        "precision_both_time_positions": both_time_count,
        "precision_open_point_matches": open_point_count,
        "precision_close_point_matches": close_point_count,
        "precision_both_point_positions": both_point_count,
        "precision_open_time_match_rate_pct": _rate(open_time_count, open_denominator),
        "precision_close_time_match_rate_pct": _rate(close_time_count, close_denominator),
        "precision_both_time_position_rate_pct": _rate(both_time_count, position_denominator),
        "precision_open_point_match_rate_pct": _rate(open_point_count, open_denominator),
        "precision_close_point_match_rate_pct": _rate(close_point_count, close_denominator),
        "precision_both_point_position_rate_pct": _rate(both_point_count, position_denominator),
        "precision_open_abs_price_diff_bps_median": float(matched_open_bps.median()) if not matched_open_bps.empty else np.nan,
        "precision_open_abs_price_diff_bps_p90": float(matched_open_bps.quantile(0.90)) if not matched_open_bps.empty else np.nan,
        "precision_close_abs_price_diff_bps_median": float(matched_close_bps.median()) if not matched_close_bps.empty else np.nan,
        "precision_close_abs_price_diff_bps_p90": float(matched_close_bps.quantile(0.90)) if not matched_close_bps.empty else np.nan,
    }


def _build_combo_event_summary(
    diagnostic: Any,
    autonomous_open: pd.DataFrame,
    autonomous_close: pd.DataFrame,
    open_matches: pd.DataFrame,
    close_matches: pd.DataFrame,
    matched_positions: pd.DataFrame,
    performance: pd.DataFrame,
    strategy_ledger_performance: pd.DataFrame,
    risk_profile: pd.DataFrame,
    event_report: pd.DataFrame,
    *,
    train_end: pd.Timestamp,
    config: WeijieComboEventStrategyConfig,
    eval_window_days: int,
) -> pd.DataFrame:
    matched_opens = _status_count(open_matches, "MATCH")
    matched_closes = _status_count(close_matches, "MATCH")
    missing_opens = _status_count(open_matches, "MISSING")
    missing_closes = _status_count(close_matches, "MISSING")
    open_extras = _status_count(open_matches, "EXTRA")
    close_extras = _status_count(close_matches, "EXTRA")
    all_ops = int(len(diagnostic.open_ops) + len(diagnostic.close_ops))
    perf_rows = {str(row.get("series")): row for _, row in performance.iterrows()} if not performance.empty else {}
    leader = perf_rows.get("leader_label_price", {})
    label_strategy = perf_rows.get("matched_label_price_strategy", {})
    signal_strategy = perf_rows.get("matched_signal_kline_execution", {})
    ledger_perf_rows = {
        str(row.get("series")): row for _, row in strategy_ledger_performance.iterrows()
    } if not strategy_ledger_performance.empty else {}
    strategy_ledger = ledger_perf_rows.get("strategy_state_ledger_kline_execution", {})
    event_rows = {
        (str(row.get("family")), str(row.get("scope"))): row for _, row in event_report.iterrows()
    } if not event_report.empty else {}
    global_risk = _risk_profile_global_row(risk_profile)
    precision_stats = _precision_match_stats(
        matched_positions,
        open_ops_count=len(diagnostic.open_ops),
        close_ops_count=len(diagnostic.close_ops),
        tolerance_minutes=float(config.precision_match_tolerance_minutes),
        price_tolerance_bps=float(config.precision_match_price_bps),
    )
    return pd.DataFrame(
        [
            {
                "strategy_type": "weijie_combo_event_gate_state_ledger",
                "calibration_scope": config.calibration_scope,
                "risk_train_window_days": int(config.risk_train_window_days),
                "risk_operation_labels_provided": bool(str(config.risk_operation_labels_csv).strip()),
                "recall_first": bool(config.recall_first),
                "recall_compression": bool(config.recall_compression),
                "recall_state_presence_gate": bool(config.recall_state_presence_gate),
                "enable_reversal_open_factors": bool(config.enable_reversal_open_factors),
                "production_mode": bool(config.production_mode),
                "production_event_compression": bool(config.production_event_compression),
                "production_sparse_event_gate": bool(config.production_sparse_event_gate),
                "production_open_transition_gate": bool(config.production_open_transition_gate),
                "production_transition_meme_legs_only": bool(config.production_transition_meme_legs_only),
                "production_transition_reset_minutes": float(config.production_transition_reset_minutes),
                "production_transition_min_train_recall": float(config.production_transition_min_train_recall),
                "production_open_transition_threshold": _finite_float(
                    event_rows.get(("open", "eval"), {}).get("transition_threshold"),
                    0.0,
                ),
                "production_frequency_event_gate": bool(config.production_frequency_event_gate),
                "production_open_rank_floor": float(config.production_open_rank_floor),
                "production_close_rank_floor": float(config.production_close_rank_floor),
                "production_open_daily_event_cap": int(config.production_open_daily_event_cap),
                "production_close_daily_event_cap": int(config.production_close_daily_event_cap),
                "production_open_slot_event_cap": int(config.production_open_slot_event_cap),
                "production_close_slot_event_cap": int(config.production_close_slot_event_cap),
                "production_open_daily_trade_cap": int(config.production_open_daily_trade_cap),
                "production_close_daily_trade_cap": int(config.production_close_daily_trade_cap),
                "production_state_open_cap_window_days": float(config.production_state_open_cap_window_days),
                "production_eth_state_open_cap": int(config.production_eth_state_open_cap),
                "production_alt_state_open_cap": int(config.production_alt_state_open_cap),
                "production_open_quality_gate": bool(config.production_open_quality_gate),
                "production_capitulation_context_gate": bool(
                    config.production_capitulation_context_gate,
                ),
                "production_trend_exhaustion_confirmation_gate": bool(
                    config.production_trend_exhaustion_confirmation_gate,
                ),
                "production_directional_context_gate": bool(
                    config.production_directional_context_gate,
                ),
                "production_eth_blowoff_handoff": bool(
                    config.production_eth_blowoff_handoff,
                ),
                "production_eth_blowoff_ladder": bool(
                    config.production_eth_blowoff_ladder,
                ),
                "production_follower_catchup_addon": bool(
                    config.production_follower_catchup_addon,
                ),
                "production_trend_saturation_gate": bool(
                    config.production_trend_saturation_gate,
                ),
                "production_unified_quality_only": bool(config.production_unified_quality_only),
                "production_unified_factor_quality_gate": bool(config.production_unified_factor_quality_gate),
                "production_quality_floor_multiplier": float(config.production_quality_floor_multiplier),
                "production_forward_quality_gate": bool(config.production_forward_quality_gate),
                "production_forward_quality_horizon_minutes": float(
                    config.production_forward_quality_horizon_minutes,
                ),
                "production_forward_quality_probability_floor": float(
                    config.production_forward_quality_probability_floor,
                ),
                "production_leader_leg_quality_gate": bool(config.production_leader_leg_quality_gate),
                "production_leader_leg_hard_gate": bool(config.production_leader_leg_hard_gate),
                "production_leader_leg_probability_floor": float(
                    config.production_leader_leg_probability_floor,
                ),
                "production_disabled_open_families": ",".join(config.production_disabled_open_families),
                "production_point_accuracy_gate": bool(config.production_point_accuracy_gate),
                "ledger_hard_stop_margin_loss_pct": float(config.ledger_hard_stop_margin_loss_pct),
                "risk_non_anchor_leverage_cap": float(config.risk_non_anchor_leverage_cap),
                "risk_anchor_short_leverage_cap": float(config.risk_anchor_short_leverage_cap),
                "risk_eth_core_downtrend_short_leverage_cap": float(
                    config.risk_eth_core_downtrend_short_leverage_cap,
                ),
                "production_allow_pyramiding": bool(config.production_allow_pyramiding),
                "production_max_lots_per_state": int(config.production_max_lots_per_state),
                "production_eth_max_lots_per_state": int(config.production_eth_max_lots_per_state),
                "production_stock_gold_max_lots_per_state": int(config.production_stock_gold_max_lots_per_state),
                "production_meme_max_lots_per_state": int(config.production_meme_max_lots_per_state),
                "production_meme_second_leg_quality_ratio_floor": float(
                    config.production_meme_second_leg_quality_ratio_floor,
                ),
                "production_meme_core_trend_confirmation_only": bool(
                    config.production_meme_core_trend_confirmation_only,
                ),
                "production_meme_core_trend_breadth_bootstrap": bool(
                    config.production_meme_core_trend_breadth_bootstrap,
                ),
                "production_non_eth_contribution_campaign": bool(
                    config.production_non_eth_contribution_campaign,
                ),
                "production_non_eth_campaign_daily_open_cap": int(
                    config.production_non_eth_campaign_daily_open_cap,
                ),
                "production_non_eth_campaign_symbols": ",".join(
                    config.production_non_eth_campaign_symbols,
                ),
                "production_non_eth_campaign_reentry_gap_minutes": float(
                    config.production_non_eth_campaign_reentry_gap_minutes,
                ),
                "production_eth_cascade_profit_exit": bool(config.production_eth_cascade_profit_exit),
                "production_meme_core_short_stop_floor_pct": float(
                    config.production_meme_core_short_stop_floor_pct,
                ),
                "production_prefilter_ineligible_open_legs": bool(
                    config.production_prefilter_ineligible_open_legs,
                ),
                "production_broad_cluster_third_meme_leg": bool(
                    config.production_broad_cluster_third_meme_leg,
                ),
                "production_low_quality_meme_short_delay_minutes": float(
                    config.production_low_quality_meme_short_delay_minutes,
                ),
                "production_eth_anchor_long_profit_min_hold_minutes": float(
                    config.production_eth_anchor_long_profit_min_hold_minutes,
                ),
                "production_clustered_meme_short_correlation_cap": float(
                    config.production_clustered_meme_short_correlation_cap,
                ),
                "production_emit_close_per_lot": bool(config.production_emit_close_per_lot),
                "production_preserve_lots_on_single_close": bool(config.production_preserve_lots_on_single_close),
                "production_full_reset_close_families": list(
                    config.production_full_reset_close_families,
                ),
                "production_profit_scaleout_close_families": list(
                    config.production_profit_scaleout_close_families,
                ),
                "production_min_addon_gap_minutes": float(config.production_min_addon_gap_minutes),
                "production_min_reentry_gap_minutes": float(config.production_min_reentry_gap_minutes),
                "production_open_full_event_budget": bool(config.production_open_full_event_budget),
                "production_cluster_open_budget": bool(config.production_cluster_open_budget),
                "production_cluster_close_budget": bool(config.production_cluster_close_budget),
                "production_close_deferral": bool(config.production_close_deferral),
                "production_portfolio_min_hold_minutes": float(config.production_portfolio_min_hold_minutes),
                "production_midnight_cleanup_min_hold_minutes": float(
                    config.production_midnight_cleanup_min_hold_minutes,
                ),
                "production_asia_cleanup_min_pressure": float(config.production_asia_cleanup_min_pressure),
                "production_evening_cleanup_min_pressure": float(config.production_evening_cleanup_min_pressure),
                "production_late_cleanup_min_pressure": float(config.production_late_cleanup_min_pressure),
                "production_midnight_cleanup_max_score": float(config.production_midnight_cleanup_max_score),
                "production_early_profit_min_hold_minutes": float(config.production_early_profit_min_hold_minutes),
                "production_risk_cut_min_event_score": float(config.production_risk_cut_min_event_score),
                "production_asia_profit_min_event_score": float(config.production_asia_profit_min_event_score),
                "production_unclustered_profit_min_roi_pct": float(
                    config.production_unclustered_profit_min_roi_pct,
                ),
                "production_unclustered_risk_min_loss_pct": float(
                    config.production_unclustered_risk_min_loss_pct,
                ),
                "production_night_short_profit_min_event_score": float(
                    config.production_night_short_profit_min_event_score,
                ),
                "production_regular_short_profit_quality_gate": bool(
                    config.production_regular_short_profit_quality_gate,
                ),
                "production_night_short_profit_spike_deferral": bool(
                    config.production_night_short_profit_spike_deferral,
                ),
                "production_eth_countertrend_short_exit_confirmation": bool(
                    config.production_eth_countertrend_short_exit_confirmation,
                ),
                "production_anchor_exit_deferral": bool(config.production_anchor_exit_deferral),
                "production_eth_anchor_short_min_hold_minutes": float(
                    config.production_eth_anchor_short_min_hold_minutes,
                ),
                "production_inventory_exit_quality_gate": bool(config.production_inventory_exit_quality_gate),
                "production_regular_short_profit_max_hold_minutes": float(
                    config.production_regular_short_profit_max_hold_minutes,
                ),
                "production_early_short_risk_max_hold_minutes": float(
                    config.production_early_short_risk_max_hold_minutes,
                ),
                "recall_open_family_score_floors": _format_family_score_floors(
                    config.recall_open_family_score_floors,
                ),
                "label_source": f"position-history {int(eval_window_days)}d labels for evaluation only; train labels end before eval open",
                "live_signal_source": (
                    "OHLCV/futures-taker/K-line indicator high-recall factors -> combo event gate -> learned leg priors -> "
                    "learned full-cross risk sizing -> own active-position ledger; no leader public event replay"
                ),
                "risk_profile_rows": len(risk_profile),
                "risk_profile_train_samples": int(_finite_float(global_risk.get("sample_count"), 0.0)),
                "risk_profile_median_leverage": _finite_float(global_risk.get("median_leverage"), np.nan),
                "risk_profile_median_notional_usdt": _finite_float(
                    global_risk.get("median_notional_usdt"),
                    np.nan,
                ),
                "risk_profile_median_risk_basis_usdt": _finite_float(
                    global_risk.get("median_risk_basis_usdt"),
                    np.nan,
                ),
                "recall_compression_open_leg_cap": int(config.recall_compression_open_leg_cap),
                "recall_compression_close_leg_cap": int(config.recall_compression_close_leg_cap),
                "recall_close_score_floor": float(config.recall_close_score_floor),
                "recall_close_family_score_floors": _format_family_score_floors(
                    config.recall_close_family_score_floors,
                ),
                "train_end": train_end,
                "open_ops": len(diagnostic.open_ops),
                "autonomous_open_trades": len(autonomous_open),
                "matched_opens": int(matched_opens),
                "missing_opens": int(missing_opens),
                "open_extra_trades": int(open_extras),
                "close_ops": len(diagnostic.close_ops),
                "autonomous_close_trades": len(autonomous_close),
                "matched_closes": int(matched_closes),
                "missing_closes": int(missing_closes),
                "close_extra_trades": int(close_extras),
                "all_ops": all_ops,
                "matched_all_ops": int(matched_opens + matched_closes),
                "missing_all_ops": int(missing_opens + missing_closes),
                "extra_all_trades": int(open_extras + close_extras),
                "all_match_rate_pct": ((matched_opens + matched_closes) / all_ops * 100.0) if all_ops else 0.0,
                **precision_stats,
                "open_event_threshold": _finite_float(event_rows.get(("open", "train"), {}).get("threshold"), np.nan),
                "close_event_threshold": _finite_float(event_rows.get(("close", "train"), {}).get("threshold"), np.nan),
                "eval_open_selected_events": int(event_rows.get(("open", "eval"), {}).get("selected_events", 0) or 0),
                "eval_close_selected_events": int(event_rows.get(("close", "eval"), {}).get("selected_events", 0) or 0),
                "leader_total_point_roi_pct": _finite_float(leader.get("total_point_roi_pct"), np.nan),
                "leader_total_margin_roi_pct": _finite_float(leader.get("total_margin_roi_pct"), np.nan),
                "leader_total_realized_pnl_usdt": _finite_float(leader.get("total_realized_pnl_usdt"), np.nan),
                "leader_pnl_proxy_base_usdt": _finite_float(leader.get("pnl_proxy_base_usdt"), np.nan),
                "leader_pnl_proxy_return_pct": _finite_float(leader.get("pnl_proxy_return_pct"), np.nan),
                "leader_capital_base_usdt": _finite_float(leader.get("capital_base_usdt"), np.nan),
                "leader_capital_return_pct": _finite_float(leader.get("capital_return_pct"), np.nan),
                "leader_win_rate_pct": _finite_float(leader.get("win_rate_pct"), np.nan),
                "leader_max_drawdown_point_pct": _finite_float(leader.get("max_drawdown_point_pct"), np.nan),
                "leader_max_drawdown_margin_pct": _finite_float(leader.get("max_drawdown_margin_pct"), np.nan),
                "leader_max_drawdown_pnl_proxy_pct": _finite_float(
                    leader.get("max_drawdown_pnl_proxy_pct"),
                    np.nan,
                ),
                "leader_max_drawdown_capital_pct": _finite_float(leader.get("max_drawdown_capital_pct"), np.nan),
                "strategy_label_total_point_roi_pct": _finite_float(label_strategy.get("total_point_roi_pct"), np.nan),
                "strategy_label_total_margin_roi_pct": _finite_float(label_strategy.get("total_margin_roi_pct"), np.nan),
                "strategy_label_total_realized_pnl_usdt": _finite_float(label_strategy.get("total_realized_pnl_usdt"), np.nan),
                "strategy_label_pnl_proxy_base_usdt": _finite_float(label_strategy.get("pnl_proxy_base_usdt"), np.nan),
                "strategy_label_pnl_proxy_return_pct": _finite_float(
                    label_strategy.get("pnl_proxy_return_pct"),
                    np.nan,
                ),
                "strategy_label_capital_base_usdt": _finite_float(label_strategy.get("capital_base_usdt"), np.nan),
                "strategy_label_capital_return_pct": _finite_float(label_strategy.get("capital_return_pct"), np.nan),
                "strategy_label_win_rate_pct": _finite_float(label_strategy.get("win_rate_pct"), np.nan),
                "strategy_label_max_drawdown_point_pct": _finite_float(label_strategy.get("max_drawdown_point_pct"), np.nan),
                "strategy_label_max_drawdown_margin_pct": _finite_float(label_strategy.get("max_drawdown_margin_pct"), np.nan),
                "strategy_label_max_drawdown_pnl_proxy_pct": _finite_float(
                    label_strategy.get("max_drawdown_pnl_proxy_pct"),
                    np.nan,
                ),
                "strategy_label_max_drawdown_capital_pct": _finite_float(
                    label_strategy.get("max_drawdown_capital_pct"),
                    np.nan,
                ),
                "strategy_kline_total_point_roi_pct": _finite_float(signal_strategy.get("total_point_roi_pct"), np.nan),
                "strategy_kline_total_margin_roi_pct": _finite_float(signal_strategy.get("total_margin_roi_pct"), np.nan),
                "strategy_kline_total_realized_pnl_usdt": _finite_float(
                    signal_strategy.get("total_realized_pnl_usdt"),
                    np.nan,
                ),
                "strategy_kline_pnl_proxy_base_usdt": _finite_float(
                    signal_strategy.get("pnl_proxy_base_usdt"),
                    np.nan,
                ),
                "strategy_kline_pnl_proxy_return_pct": _finite_float(
                    signal_strategy.get("pnl_proxy_return_pct"),
                    np.nan,
                ),
                "strategy_kline_capital_base_usdt": _finite_float(signal_strategy.get("capital_base_usdt"), np.nan),
                "strategy_kline_capital_return_pct": _finite_float(signal_strategy.get("capital_return_pct"), np.nan),
                "strategy_kline_win_rate_pct": _finite_float(signal_strategy.get("win_rate_pct"), np.nan),
                "strategy_kline_max_drawdown_point_pct": _finite_float(signal_strategy.get("max_drawdown_point_pct"), np.nan),
                "strategy_kline_max_drawdown_margin_pct": _finite_float(signal_strategy.get("max_drawdown_margin_pct"), np.nan),
                "strategy_kline_max_drawdown_pnl_proxy_pct": _finite_float(
                    signal_strategy.get("max_drawdown_pnl_proxy_pct"),
                    np.nan,
                ),
                "strategy_kline_max_drawdown_capital_pct": _finite_float(
                    signal_strategy.get("max_drawdown_capital_pct"),
                    np.nan,
                ),
                "strategy_ledger_positions": int(_finite_float(strategy_ledger.get("positions"), 0.0)),
                "strategy_ledger_total_point_roi_pct": _finite_float(
                    strategy_ledger.get("total_point_roi_pct"),
                    np.nan,
                ),
                "strategy_ledger_total_margin_roi_pct": _finite_float(
                    strategy_ledger.get("total_margin_roi_pct"),
                    np.nan,
                ),
                "strategy_ledger_total_realized_pnl_usdt": _finite_float(
                    strategy_ledger.get("total_realized_pnl_usdt"),
                    np.nan,
                ),
                "strategy_ledger_pnl_proxy_base_usdt": _finite_float(
                    strategy_ledger.get("pnl_proxy_base_usdt"),
                    np.nan,
                ),
                "strategy_ledger_pnl_proxy_return_pct": _finite_float(
                    strategy_ledger.get("pnl_proxy_return_pct"),
                    np.nan,
                ),
                "strategy_ledger_win_rate_pct": _finite_float(strategy_ledger.get("win_rate_pct"), np.nan),
                "strategy_ledger_profit_factor": _finite_float(strategy_ledger.get("profit_factor"), np.nan),
                "strategy_ledger_max_drawdown_point_pct": _finite_float(
                    strategy_ledger.get("max_drawdown_point_pct"),
                    np.nan,
                ),
                "strategy_ledger_max_drawdown_margin_pct": _finite_float(
                    strategy_ledger.get("max_drawdown_margin_pct"),
                    np.nan,
                ),
                "strategy_ledger_max_drawdown_pnl_proxy_pct": _finite_float(
                    strategy_ledger.get("max_drawdown_pnl_proxy_pct"),
                    np.nan,
                ),
                "strategy_account_initial_equity_usdt": _finite_float(
                    strategy_ledger.get("account_initial_equity_usdt"),
                    np.nan,
                ),
                "strategy_account_capital_allocation_pct": _finite_float(
                    strategy_ledger.get("account_capital_allocation_pct"),
                    np.nan,
                ),
                "strategy_account_allocated_initial_equity_usdt": _finite_float(
                    strategy_ledger.get("account_allocated_initial_equity_usdt"),
                    np.nan,
                ),
                "strategy_account_allocated_ending_equity_usdt": _finite_float(
                    strategy_ledger.get("account_allocated_ending_equity_usdt"),
                    np.nan,
                ),
                "strategy_account_public_roi_basis_usdt": _finite_float(
                    strategy_ledger.get("account_public_roi_basis_usdt"),
                    np.nan,
                ),
                "strategy_account_ending_equity_usdt": _finite_float(
                    strategy_ledger.get("account_ending_equity_usdt"),
                    np.nan,
                ),
                "strategy_account_return_pct": _finite_float(
                    strategy_ledger.get("account_return_pct"),
                    np.nan,
                ),
                "strategy_account_max_drawdown_pct": _finite_float(
                    strategy_ledger.get("account_max_drawdown_pct"),
                    np.nan,
                ),
                "strategy_account_accepted_positions": int(
                    _finite_float(strategy_ledger.get("account_accepted_positions"), 0.0),
                ),
                "strategy_account_rejected_positions": int(
                    _finite_float(strategy_ledger.get("account_rejected_positions"), 0.0),
                ),
                "strategy_account_closed_positions": int(
                    _finite_float(strategy_ledger.get("account_closed_positions"), 0.0),
                ),
                "strategy_account_lot_win_rate_pct": _finite_float(
                    strategy_ledger.get("account_win_rate_pct"),
                    np.nan,
                ),
                "strategy_account_position_cycles": int(
                    _finite_float(strategy_ledger.get("account_position_cycles"), 0.0),
                ),
                "strategy_account_winning_position_cycles": int(
                    _finite_float(strategy_ledger.get("account_winning_position_cycles"), 0.0),
                ),
                "strategy_account_position_cycle_win_rate_pct": _finite_float(
                    strategy_ledger.get("account_position_cycle_win_rate_pct"),
                    np.nan,
                ),
                "strategy_account_total_cost_usdt": _finite_float(
                    strategy_ledger.get("account_total_cost_usdt"),
                    np.nan,
                ),
                "strategy_account_net_external_flow_usdt": _finite_float(
                    strategy_ledger.get("account_net_external_flow_usdt"),
                    np.nan,
                ),
                "strategy_account_total_funding_usdt": _finite_float(
                    strategy_ledger.get("account_total_funding_usdt"),
                    np.nan,
                ),
                "strategy_account_total_explicit_fee_usdt": _finite_float(
                    strategy_ledger.get("account_total_explicit_fee_usdt"),
                    np.nan,
                ),
                "strategy_account_net_profit_usdt": _finite_float(
                    strategy_ledger.get("account_net_profit_usdt"),
                    np.nan,
                ),
                "strategy_account_strategy_return_pct": _finite_float(
                    strategy_ledger.get("account_strategy_return_pct"),
                    np.nan,
                ),
                "strategy_account_public_comparable_roi_pct": _finite_float(
                    strategy_ledger.get(
                        "account_public_comparable_roi_pct",
                        strategy_ledger.get("account_contributed_capital_return_pct"),
                    ),
                    np.nan,
                ),
                "strategy_account_cash_flow_coverage": str(
                    strategy_ledger.get("account_cash_flow_coverage") or "missing",
                ),
                "strategy_account_cash_flow_valid": bool(
                    strategy_ledger.get("account_cash_flow_valid", False),
                ),
                "strategy_account_max_margin_utilization_pct": _finite_float(
                    strategy_ledger.get("account_max_margin_utilization_pct"),
                    np.nan,
                ),
                "strategy_account_worst_trade_equity_loss_pct": _finite_float(
                    strategy_ledger.get("account_worst_trade_equity_loss_pct"),
                    np.nan,
                ),
                "strategy_account_margin_loss_over_100_positions": int(
                    _finite_float(strategy_ledger.get("account_margin_loss_over_100_positions"), 0.0),
                ),
                "strategy_account_risk_prior_weighted_sizing": bool(
                    strategy_ledger.get("account_risk_prior_weighted_sizing", False),
                ),
                "strategy_account_signal_risk_basis_sizing": bool(
                    strategy_ledger.get("account_signal_risk_basis_sizing", False),
                ),
                "strategy_account_mark_to_market_margin": bool(
                    strategy_ledger.get("account_mark_to_market_margin", False),
                ),
                "strategy_account_eth_main_leg_margin_floor_pct": _finite_float(
                    strategy_ledger.get("account_eth_main_leg_margin_floor_pct"),
                    0.0,
                ),
                "strategy_account_drawdown_throttle_start_pct": _finite_float(
                    strategy_ledger.get("account_drawdown_throttle_start_pct"),
                    0.0,
                ),
                "strategy_account_drawdown_throttle_margin_cap_pct": _finite_float(
                    strategy_ledger.get("account_drawdown_throttle_margin_cap_pct"),
                    0.0,
                ),
                "strategy_account_risk_prior_reference_usdt": _finite_float(
                    strategy_ledger.get("account_risk_prior_reference_usdt"),
                    np.nan,
                ),
                "strategy_account_curve_method": str(strategy_ledger.get("account_curve_method") or ""),
            },
        ],
    )


def _risk_profile_global_row(risk_profile: pd.DataFrame) -> dict[str, Any]:
    if risk_profile.empty or "scope" not in risk_profile:
        return {}
    frame = risk_profile[risk_profile["scope"].astype(str).eq("global")].copy()
    if frame.empty:
        return {}
    return dict(frame.iloc[0])


def _risk_fields_from_sizing(sizing: RiskSizingDecision) -> dict[str, Any]:
    return {
        "quantity": float(sizing.quantity),
        "notional_usdt": float(sizing.notional_usdt),
        "leverage": float(sizing.leverage),
        "risk_basis_usdt": float(sizing.risk_basis_usdt),
        "margin_pct": float(sizing.margin_pct),
        "risk_budget_pct": float(sizing.risk_budget_pct),
        "risk_budget_usdt": float(sizing.risk_budget_usdt),
        "stop_distance_pct": float(sizing.stop_distance_pct),
        "atr_pct": float(sizing.atr_pct),
        "margin_mode": str(sizing.margin_mode),
        "sizing_reason": str(sizing.sizing_reason),
    }


def _risk_fields_from_lot(lot: dict[str, Any]) -> dict[str, Any]:
    return {
        "quantity": _finite_float(lot.get("quantity"), 0.0),
        "notional_usdt": _finite_float(lot.get("notional_usdt"), 0.0),
        "leverage": _finite_float(lot.get("leverage"), 0.0),
        "risk_basis_usdt": _finite_float(lot.get("risk_basis_usdt"), 0.0),
        "margin_pct": _finite_float(lot.get("margin_pct"), 0.0),
        "risk_budget_pct": _finite_float(lot.get("risk_budget_pct"), 0.0),
        "risk_budget_usdt": _finite_float(lot.get("risk_budget_usdt"), 0.0),
        "stop_distance_pct": _finite_float(lot.get("stop_distance_pct"), 0.0),
        "atr_pct": _finite_float(lot.get("atr_pct"), 0.0),
        "margin_mode": str(lot.get("margin_mode") or "CROSSED"),
        "sizing_reason": str(lot.get("sizing_reason") or ""),
    }


def _scaled_lot(lot: dict[str, Any], fraction: float) -> dict[str, Any]:
    """Return one economically equivalent fraction of a strategy lot."""

    scale = min(1.0, max(0.0, _finite_float(fraction, 0.0)))
    out = dict(lot)
    for column in (
        "quantity",
        "notional_usdt",
        "risk_basis_usdt",
        "margin_pct",
        "risk_budget_pct",
        "risk_budget_usdt",
    ):
        out[column] = _finite_float(lot.get(column), 0.0) * scale
    return out


def _scaled_lot_risk_fields(
    lot: dict[str, Any],
    fraction: float,
) -> dict[str, Any]:
    return _risk_fields_from_lot(_scaled_lot(lot, fraction))


def _aggregate_lot_risk_fields(lots: list[dict[str, Any]]) -> dict[str, Any]:
    if not lots:
        return _risk_fields_from_lot({})
    quantity = sum(_finite_float(lot.get("quantity"), 0.0) for lot in lots)
    notional = sum(_finite_float(lot.get("notional_usdt"), 0.0) for lot in lots)
    risk_basis = sum(_finite_float(lot.get("risk_basis_usdt"), 0.0) for lot in lots)
    leverage = notional / risk_basis if risk_basis > 0.0 else _finite_float(lots[0].get("leverage"), 0.0)
    reasons = sorted({str(lot.get("sizing_reason") or "") for lot in lots if str(lot.get("sizing_reason") or "")})
    return {
        "quantity": float(quantity),
        "notional_usdt": float(notional),
        "leverage": float(leverage),
        "risk_basis_usdt": float(risk_basis),
        "margin_pct": float(sum(_finite_float(lot.get("margin_pct"), 0.0) for lot in lots)),
        "risk_budget_pct": float(sum(_finite_float(lot.get("risk_budget_pct"), 0.0) for lot in lots)),
        "risk_budget_usdt": float(sum(_finite_float(lot.get("risk_budget_usdt"), 0.0) for lot in lots)),
        "stop_distance_pct": float(max(_finite_float(lot.get("stop_distance_pct"), 0.0) for lot in lots)),
        "atr_pct": float(max(_finite_float(lot.get("atr_pct"), 0.0) for lot in lots)),
        "margin_mode": "CROSSED",
        "sizing_reason": "+".join(reasons)[:160],
    }


def _strategy_ledger_position_row(
    lot: dict[str, Any],
    *,
    close_row: pd.Series,
    close_signal_id: str,
    close_time: pd.Timestamp,
    close_price: float,
    direction: str,
    reason_family: str,
    event: dict[str, Any],
    close_group_index: int,
    close_group_count: int,
) -> dict[str, Any]:
    open_time = pd.Timestamp(lot.get("open_time")) if lot.get("open_time") is not None else pd.NaT
    open_price = _finite_float(lot.get("open_price"), np.nan)
    quantity = _finite_float(lot.get("quantity"), 0.0)
    notional = _finite_float(lot.get("notional_usdt"), np.nan)
    leverage = _finite_float(lot.get("leverage"), np.nan)
    risk_basis = _finite_float(lot.get("risk_basis_usdt"), np.nan)
    point_roi = _point_return_pct(direction, open_price, close_price)
    realized_pnl = _realized_pnl_usdt(direction, open_price, close_price, quantity)
    margin_roi = _margin_roi_pct(realized_pnl, risk_basis)
    hold_minutes = (
        (pd.Timestamp(close_time) - open_time).total_seconds() / 60.0
        if pd.notna(open_time) and pd.notna(close_time)
        else np.nan
    )
    return {
        "open_signal_id": str(lot.get("open_signal_id") or ""),
        "close_signal_id": close_signal_id,
        "symbol": str(lot.get("symbol") or close_row.get("symbol") or ""),
        "direction": str(direction),
        "open_time": open_time,
        "close_time": pd.Timestamp(close_time),
        "hold_minutes": hold_minutes,
        "open_price": open_price,
        "close_price": _finite_float(close_price, np.nan),
        "quantity": quantity,
        "notional_usdt": notional,
        "leverage": leverage,
        "risk_basis_usdt": risk_basis,
        "margin_pct": _finite_float(lot.get("margin_pct"), 0.0),
        "risk_budget_pct": _finite_float(lot.get("risk_budget_pct"), 0.0),
        "risk_budget_usdt": _finite_float(lot.get("risk_budget_usdt"), 0.0),
        "stop_distance_pct": _finite_float(lot.get("stop_distance_pct"), 0.0),
        "atr_pct": _finite_float(lot.get("atr_pct"), 0.0),
        "point_roi_pct": point_roi,
        "margin_roi_pct": margin_roi,
        "realized_pnl_usdt": realized_pnl,
        "open_reason_family": str(lot.get("open_reason_family") or ""),
        "close_reason_family": str(reason_family),
        "open_score": _finite_float(lot.get("open_score"), np.nan),
        "close_score": _finite_float(close_row.get("leg_score"), _finite_float(close_row.get("score"), np.nan)),
        "open_reason": str(lot.get("open_reason") or ""),
        "close_reason": str(close_row.get("reason") or ""),
        "close_source_signal_id": str(close_row.get("signal_id") or ""),
        "close_event_id": str(event.get("event_id") or ""),
        "close_event_score": _finite_float(event.get("event_score"), np.nan),
        "close_event_pressure": _finite_float(event.get("event_pressure"), np.nan),
        "close_group_index": int(close_group_index),
        "close_group_count": int(close_group_count),
        "non_eth_contribution_campaign": int(
            _finite_float(lot.get("non_eth_contribution_campaign"), 0.0),
        ),
        "sizing_reason": str(lot.get("sizing_reason") or ""),
    }


def _strategy_ledger_position_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "open_signal_id",
        "close_signal_id",
        "symbol",
        "direction",
        "open_time",
        "close_time",
        "hold_minutes",
        "open_price",
        "close_price",
        "quantity",
        "notional_usdt",
        "leverage",
        "risk_basis_usdt",
        "margin_pct",
        "risk_budget_pct",
        "risk_budget_usdt",
        "stop_distance_pct",
        "atr_pct",
        "point_roi_pct",
        "margin_roi_pct",
        "realized_pnl_usdt",
        "open_reason_family",
        "close_reason_family",
        "open_score",
        "close_score",
        "open_reason",
        "close_reason",
        "close_source_signal_id",
        "close_event_id",
        "close_event_score",
        "close_event_pressure",
        "close_group_index",
        "close_group_count",
        "non_eth_contribution_campaign",
        "sizing_reason",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame(rows, columns=columns)
    out["open_time"] = pd.to_datetime(out["open_time"], errors="coerce")
    out["close_time"] = pd.to_datetime(out["close_time"], errors="coerce")
    numeric_cols = [
        "hold_minutes",
        "open_price",
        "close_price",
        "quantity",
        "notional_usdt",
        "leverage",
        "risk_basis_usdt",
        "margin_pct",
        "risk_budget_pct",
        "risk_budget_usdt",
        "stop_distance_pct",
        "atr_pct",
        "point_roi_pct",
        "margin_roi_pct",
        "realized_pnl_usdt",
        "open_score",
        "close_score",
        "close_event_score",
        "close_event_pressure",
        "close_group_index",
        "close_group_count",
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out.get(col), errors="coerce")
    return out.sort_values(["close_time", "symbol", "direction", "open_time", "open_signal_id"]).reset_index(drop=True)


def _selected_signal_row(
    selected_id: str,
    row: pd.Series,
    ts: pd.Timestamp,
    *,
    reason_prefix: str,
    risk_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "signal_id": selected_id,
        "symbol": row.get("symbol"),
        "action": row.get("action"),
        "signal_time": ts,
        "score": _finite_float(row.get("leg_score"), _finite_float(row.get("score"))),
        "reason": f"{reason_prefix};source_signal={row.get('signal_id')};{row.get('reason', '')}",
    }
    if "reason_family" in row.index:
        payload["reason_family"] = str(row.get("reason_family") or "")
    if "ret60" in row.index:
        payload["ret60"] = _finite_float(row.get("ret60"), np.nan)
    if "follower_catchup_augmented" in row.index:
        payload["follower_catchup_augmented"] = int(
            _finite_float(row.get("follower_catchup_augmented"), 0.0),
        )
    if "factor_quality_score" in row.index:
        payload["reason"] = (
            f"{payload['reason']};factor_quality="
            f"{_finite_float(row.get('factor_quality_score'), np.nan):.3f}"
        )
    if "factor_quality_floor" in row.index:
        payload["reason"] = (
            f"{payload['reason']};factor_floor="
            f"{_finite_float(row.get('factor_quality_floor'), np.nan):.3f}"
        )
    if "forward_quality_probability" in row.index:
        payload["reason"] = (
            f"{payload['reason']};forward_quality_probability="
            f"{_finite_float(row.get('forward_quality_probability'), np.nan):.4f}"
        )
    if "leader_leg_quality_probability" in row.index:
        payload["reason"] = (
            f"{payload['reason']};leader_leg_quality_probability="
            f"{_finite_float(row.get('leader_leg_quality_probability'), np.nan):.4f}"
        )
    if "non_eth_contribution_campaign" in row.index:
        payload["non_eth_contribution_campaign"] = int(
            _finite_float(row.get("non_eth_contribution_campaign"), 0.0),
        )
    if risk_fields:
        payload.update(risk_fields)
    return payload


def _open_reason_with_quality(row: pd.Series) -> str:
    reason = str(row.get("reason") or "")
    if "forward_quality_probability" in row.index:
        reason = (
            f"{reason};forward_quality_probability="
            f"{_finite_float(row.get('forward_quality_probability'), np.nan):.6f}"
        )
    if "leader_leg_quality_probability" in row.index:
        reason = (
            f"{reason};leader_leg_quality_probability="
            f"{_finite_float(row.get('leader_leg_quality_probability'), np.nan):.6f}"
        )
    return reason


def _selected_signal_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "signal_id",
        "symbol",
        "action",
        "signal_time",
        "score",
        "reason",
        "reason_family",
        "operation_type",
        "close_fraction",
        "ret60",
        "follower_catchup_augmented",
        "non_eth_contribution_campaign",
        "quantity",
        "notional_usdt",
        "leverage",
        "risk_basis_usdt",
        "margin_pct",
        "risk_budget_pct",
        "risk_budget_usdt",
        "stop_distance_pct",
        "atr_pct",
        "margin_mode",
        "sizing_reason",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame(rows, columns=columns)
    out["signal_time"] = pd.to_datetime(out["signal_time"], errors="coerce")
    for col in (
        "quantity",
        "notional_usdt",
        "leverage",
        "risk_basis_usdt",
        "margin_pct",
        "risk_budget_pct",
        "risk_budget_usdt",
        "stop_distance_pct",
        "atr_pct",
        "ret60",
        "follower_catchup_augmented",
        "non_eth_contribution_campaign",
    ):
        out[col] = pd.to_numeric(out.get(col), errors="coerce").fillna(0.0)
    out["reason_family"] = out.get("reason_family", pd.Series(dtype=str)).fillna("").astype(str)
    out["operation_type"] = out.get(
        "operation_type",
        pd.Series(dtype=str),
    ).fillna("").astype(str)
    out["close_fraction"] = pd.to_numeric(
        out.get("close_fraction"),
        errors="coerce",
    ).fillna(1.0)
    out["margin_mode"] = out.get("margin_mode", pd.Series(dtype=str)).fillna("CROSSED").astype(str)
    out["sizing_reason"] = out.get("sizing_reason", pd.Series(dtype=str)).fillna("").astype(str)
    return out.sort_values(["signal_time", "symbol", "action", "signal_id"]).reset_index(drop=True)


def _feature_close_at_or_before(feature_rows: pd.DataFrame, symbol: str, ts: pd.Timestamp) -> float:
    if feature_rows.empty or pd.isna(ts) or "close" not in feature_rows.columns:
        return np.nan
    frame = feature_rows[feature_rows["symbol"].astype(str).eq(str(symbol))].copy()
    if frame.empty:
        return np.nan
    frame["bar_time"] = pd.to_datetime(frame["bar_time"], errors="coerce")
    selected = frame[frame["bar_time"].le(pd.Timestamp(ts))]
    if selected.empty:
        return np.nan
    return _finite_float(selected.sort_values("bar_time").iloc[-1].get("close"), np.nan)


def _status_count(matches: pd.DataFrame, status: str) -> int:
    if matches.empty or "status" not in matches.columns:
        return 0
    return int(matches["status"].astype(str).eq(status).sum())


def _finite_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return out
