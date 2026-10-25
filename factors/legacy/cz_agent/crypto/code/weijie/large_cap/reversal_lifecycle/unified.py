# ruff: noqa: C901, PLR0912, PLR0915, RUF002, RUF003
"""统一生成、筛选和执行大盘反转生命周期候选活动。"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from factors.crypto.code.weijie.large_cap.countertrend import (
    CountertrendCampaignPolicy,
    CountertrendFactorPolicy,
    build_countertrend_scores,
    evaluate_scored_countertrend_campaigns,
)

from .components import directional_market_components
from .inventory import classify_lot_inventory_lifecycle
from .point_recovery import PointRecoveryPolicy, build_point_recovery_actions
from .post_exit import (
    NORMALIZED_RUNNER_EXIT_STATE,
    POST_EXIT_BREADTH_RECOVERY,
    PostExitRecoveryPolicy,
    build_post_exit_recovery_actions,
    build_post_exit_rotation_actions,
)
from .relative_lag_reclaim import (
    RELATIVE_LAG_RECLAIM,
    RelativeLagReclaimPolicy,
    build_relative_lag_reclaim_actions,
)
from .rotation import (
    ROTATION_MECHANISMS,
    RegimeRotationPolicy,
    build_regime_rotation_actions,
)

LIQUIDITY_SHOCK = "liquidity_shock_exhaustion"
FAILED_RECLAIM = "failed_reclaim_recovery"
LOW_LIQUIDITY_RETEST_PROBE = "low_liquidity_retest_probe"
SUPPORTED_MECHANISMS = (
    LIQUIDITY_SHOCK,
    FAILED_RECLAIM,
    RELATIVE_LAG_RECLAIM,
    *ROTATION_MECHANISMS,
    POST_EXIT_BREADTH_RECOVERY,
)
MINIMUM_THRESHOLD_MULTIPLIER = 0.8
MAXIMUM_THRESHOLD_MULTIPLIER = 1.2
_INVENTORY_EPSILON = 1e-9
_DIRECTIONAL_RANGE_MIDPOINT = 0.50
_MAXIMUM_RSI = 100.0


@dataclass(frozen=True, slots=True)
class UnifiedRegimeLifecyclePolicy:
    """统一反转机制、候选选择、库存接管和风险边界配置。"""

    enabled_mechanisms: tuple[str, ...] = SUPPORTED_MECHANISMS
    rotation_policy: RegimeRotationPolicy = field(
        default_factory=RegimeRotationPolicy,
    )
    point_recovery_policy: PointRecoveryPolicy = field(
        default_factory=PointRecoveryPolicy,
    )
    relative_lag_reclaim_policy: RelativeLagReclaimPolicy = field(
        default_factory=RelativeLagReclaimPolicy,
    )
    post_exit_policy: PostExitRecoveryPolicy = field(
        default_factory=PostExitRecoveryPolicy,
    )
    shock_score_floor: float = 0.60
    shock_threshold_multiplier: float = 1.0
    shock_recovery_target_pct: float = 1.50
    shock_maximum_adds: int = 3
    shock_add_distance_atr_multiple: float = 1.0
    shock_minimum_holding_bars: int = 6
    shock_maximum_holding_bars: int = 576
    shock_hard_stop_pct: float = 6.0
    shock_confirmation_timeout_bars: int = 72
    shock_trend_following_liquid_rejection_enabled: bool = True
    shock_immediate_quote_volume_ratio: float = 8.0
    shock_immediate_relative_edge_pct: float = 0.25
    shock_closed_bar_liquid_capitulation_enabled: bool = False
    shock_closed_bar_liquid_capitulation_short_enabled: bool = False
    shock_closed_bar_liquid_capitulation_quote_volume_ratio: float = 5.5
    shock_closed_bar_liquid_capitulation_adverse_mid_return_pct: float = 1.4
    shock_closed_bar_liquid_capitulation_adverse_long_return_pct: float = 1.2
    shock_closed_bar_liquid_capitulation_maximum_location: float = 0.10
    shock_closed_bar_liquid_capitulation_minimum_rsi: float = 25.0
    shock_closed_bar_liquid_capitulation_maximum_rsi: float = 30.0
    shock_absorption_quote_volume_ratio: float = 5.0
    shock_absorption_adverse_mid_return_pct: float = 1.20
    shock_absorption_location: float = 0.30
    shock_absorption_rsi: float = 40.0
    shock_absorption_adverse_breadth: float = 0.85
    shock_absorption_candle_close_location: float = 0.60
    shock_absorption_first_scaleout_pct: float = 0.30
    shock_absorption_final_target_pct: float = 0.80
    shock_absorption_entry_units: float = 2.0
    shock_absorption_scaleout_units: float = 1.0
    shock_absorption_deep_scaleout_rsi: float = 20.0
    shock_absorption_deep_scaleout_minimum_adx: float = 35.0
    shock_absorption_deep_scaleout_fraction: float = 0.80
    shock_absorption_deep_scaleout_lookback_bars: int = 1
    shock_absorption_deep_scaleout_first_target_pct: float = 0.05
    shock_absorption_maximum_holding_bars: int = 576
    shock_absorption_recovery_return_pct: float = 0.50
    shock_absorption_recovery_location: float = 0.60
    shock_absorption_recovery_rsi: float = 55.0
    shock_absorption_recovery_breadth: float = 0.75
    shock_absorption_counter_breadth: float = 0.66
    shock_absorption_resilient_volatility_ratio: float = 1.25
    shock_absorption_maturity_location: float = 0.75
    shock_absorption_maturity_rsi: float = 70.0
    shock_absorption_maturity_minimum_adx: float = 25.0
    shock_absorption_maturity_quote_volume_ratio: float = 2.0
    shock_trend_preserving_long_band_pct: float = 0.50
    shock_trend_preserving_exit_location: float = 0.85
    shock_trend_preserving_exit_rsi: float = 60.0
    shock_trend_stall_minimum_adx: float = 35.0
    shock_crash_recovery_adverse_long_return_pct: float = 2.0
    shock_crash_recovery_location: float = 0.30
    shock_crash_recovery_rsi: float = 45.0
    shock_crash_recovery_tiny_return_pct: float = 0.50
    shock_severe_capitulation_quote_volume_ratio: float = 10.0
    shock_severe_capitulation_adverse_mid_return_pct: float = 3.0
    shock_severe_capitulation_location: float = 0.10
    shock_severe_capitulation_rsi: float = 15.0
    shock_severe_capitulation_adverse_breadth: float = 0.85
    shock_orderly_washout_adverse_mid_minimum_pct: float = 1.5
    shock_orderly_washout_adverse_mid_maximum_pct: float = 2.5
    shock_orderly_washout_adverse_long_minimum_pct: float = 2.0
    shock_orderly_washout_minimum_adx: float = 50.0
    shock_orderly_washout_capitulation_rsi: float = 20.0
    shock_orderly_washout_requires_cooling_confirmation: bool = False
    shock_generic_countertrend_requires_cooling_confirmation: bool = False
    shock_generic_momentum_extension_enabled: bool = False
    shock_orderly_washout_confirmation_minimum_bars: int = 48
    shock_orderly_washout_confirmation_maximum_abs_tiny_return_pct: float = 0.02
    shock_orderly_washout_confirmation_maximum_abs_recovery_return_pct: float = 0.20
    shock_orderly_washout_confirmation_location_minimum: float = 0.55
    shock_orderly_washout_confirmation_location_maximum: float = 0.80
    shock_orderly_washout_confirmation_rsi_minimum: float = 40.0
    shock_orderly_washout_confirmation_rsi_maximum: float = 60.0
    shock_orderly_washout_confirmation_maximum_adx: float = 15.0
    shock_orderly_washout_confirmation_maximum_quote_volume_ratio: float = 1.50
    shock_orderly_washout_scaleout_locations: tuple[float, ...] = (
        0.25,
        0.45,
        0.60,
    )
    shock_orderly_washout_immediate_scaleout_fraction: float = 0.25
    shock_orderly_washout_scaleout_fraction: float = 0.50
    shock_orderly_washout_scaleout_minimum_return_pct: float = 0.10
    shock_orderly_washout_reentry_units: float = 0.15
    shock_orderly_washout_reentry_minimum_giveback_pct: float = 0.40
    shock_orderly_washout_reentry_minimum_recovery_return_pct: float = 1.0
    shock_orderly_washout_reentry_minimum_adverse_candle_return_pct: float = 0.10
    shock_orderly_washout_exit_location: float = 0.80
    shock_orderly_washout_exit_rsi: float = 55.0
    shock_orderly_washout_exit_maximum_adx: float = 25.0
    shock_orderly_washout_exit_reversal_tiny_return_pct: float = 0.10
    shock_orderly_washout_exit_reversal_adverse_breadth3: float = 0.66
    shock_orderly_washout_terminal_mid_return_pct: float = 2.0
    shock_orderly_washout_terminal_location: float = 0.95
    shock_orderly_washout_terminal_rsi: float = 85.0
    shock_orderly_washout_terminal_minimum_adx: float = 40.0
    shock_orderly_washout_maximum_holding_bars: int = 288
    shock_blowoff_entry_adverse_mid_minimum_pct: float = 2.5
    shock_blowoff_entry_adverse_mid_maximum_pct: float = 3.5
    shock_blowoff_entry_adverse_long_minimum_pct: float = 2.0
    shock_blowoff_entry_minimum_adx: float = 50.0
    shock_blowoff_entry_minimum_atr_ratio: float = 0.0
    shock_blowoff_entry_minimum_quote_volume_ratio: float = 0.0
    shock_blowoff_add_adverse_mid_return_pct: float = 4.5
    shock_blowoff_add_maximum_location: float = 0.20
    shock_blowoff_add_maximum_rsi: float = 25.0
    shock_blowoff_add_minimum_adx: float = 50.0
    shock_blowoff_second_add_gap_bars: int = 8
    shock_blowoff_probe_units: float = 0.50
    shock_blowoff_probe_take_profit_pct: float = 0.0
    shock_blowoff_primary_add_units: float = 2.0
    shock_blowoff_secondary_add_units: float = 0.50
    shock_blowoff_reduction_fraction: float = 0.50
    shock_blowoff_followthrough_return_pct: float = 0.50
    shock_blowoff_followthrough_breadth: float = 0.75
    shock_blowoff_followthrough_reduction_fraction: float = 2.0
    shock_blowoff_residual_minimum_holding_bars: int = 288
    shock_blowoff_residual_location: float = 0.85
    shock_blowoff_residual_rsi: float = 55.0
    shock_blowoff_residual_maximum_adx: float = 20.0
    shock_blowoff_maximum_holding_bars: int = 576
    shock_countertrend_strong_long_return_pct: float = 2.5
    shock_strong_trend_defer_pct: float = 3.0
    shock_deep_exhaustion_minimum_adx: float = 25.0
    shock_confirmation_minimum_adverse_mid_return_pct: float = 1.20
    shock_confirmation_minimum_adverse_long_return_pct: float = 1.20
    shock_confirmation_minimum_quote_volume_ratio: float = 2.0
    shock_minimum_directional_dislocation_pct: float = 0.75
    shock_structural_override_score_ratio: float = 0.80
    shock_rejection_tiny_return_pct: float = 0.10
    shock_rejection_breadth: float = 0.66
    shock_relief_location: float = 0.20
    shock_retest_location: float = 0.10
    shock_relief_retest_confirmation_enabled: bool = True
    post_exit_cooldown_bars: int = 0
    opposite_direction_post_exit_cooldown_bars: int = 0
    post_exit_liquidity_shock_cooldown_bars: int = 0
    controller_takeover_edge: float = 0.10
    round_trip_cost_bps: float = 14.0
    threshold_consensus_enabled: bool = False
    threshold_consensus_multipliers: tuple[float, ...] = (0.85, 1.0, 1.15)
    threshold_consensus_required_votes: int = 2
    threshold_consensus_time_tolerance_bars: int = 18
    threshold_consensus_dispersion_multiplier: float = 1.0

    def with_threshold_multiplier(
        self,
        multiplier: float,
    ) -> UnifiedRegimeLifecyclePolicy:
        """扰动信号灵敏度，同时保留流动性冲击风险安全下限。

        ``shock_threshold_multiplier`` 控制方向错配、成交量确认和拒绝强度，
        放宽它会把未成熟冲击提前变成持仓，并可能在真正信号到来前触发账户
        止损。它属于灾难风险下限，不是可优化的信号灵敏度。其余自主机制的
        连续阈值仍完整接受 0.8/1.0/1.2 扰动。
        """

        if not (MINIMUM_THRESHOLD_MULTIPLIER <= multiplier <= MAXIMUM_THRESHOLD_MULTIPLIER):
            message = "unified threshold multiplier must be in [0.8, 1.2]"
            raise ValueError(message)
        if self.threshold_consensus_enabled:
            return replace(
                self,
                threshold_consensus_dispersion_multiplier=float(multiplier),
            )
        return replace(
            self,
            rotation_policy=self.rotation_policy.with_threshold_multiplier(
                multiplier,
            ),
            point_recovery_policy=(self.point_recovery_policy.with_threshold_multiplier(multiplier)),
            relative_lag_reclaim_policy=(
                self.relative_lag_reclaim_policy.with_threshold_multiplier(
                    multiplier,
                )
            ),
            post_exit_policy=self.post_exit_policy.with_threshold_multiplier(
                multiplier,
            ),
        )


def evaluate_unified_regime_lifecycle(
    market_panel: pd.DataFrame,
    *,
    policy: UnifiedRegimeLifecyclePolicy | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """evaluate_unified_regime_lifecycle：计算可解释的市场结构因子。"""

    cfg = policy or UnifiedRegimeLifecyclePolicy()
    _validate_policy(cfg)
    market = _prepare_market(market_panel)
    if market.empty:
        return _empty_cycles(), _empty_actions()
    if cfg.threshold_consensus_enabled:
        return _evaluate_threshold_consensus(market, policy=cfg)

    proposals: list[dict[str, Any]] = []
    if LIQUIDITY_SHOCK in cfg.enabled_mechanisms:
        proposals.extend(_liquidity_shock_proposals(market, policy=cfg))
    if FAILED_RECLAIM in cfg.enabled_mechanisms:
        proposals.extend(
            _failed_reclaim_proposals(
                market,
                policy=cfg.point_recovery_policy,
            ),
        )
    if RELATIVE_LAG_RECLAIM in cfg.enabled_mechanisms:
        proposals.extend(
            _relative_lag_reclaim_proposals(
                market,
                policy=cfg.relative_lag_reclaim_policy,
            ),
        )
    if set(cfg.enabled_mechanisms).intersection(ROTATION_MECHANISMS):
        proposals.extend(
            _rotation_proposals(
                market,
                enabled_mechanisms=cfg.enabled_mechanisms,
                policy=cfg.rotation_policy,
            ),
        )
    base_proposals = proposals
    selected = _select_non_overlapping_proposals(
        proposals,
        market=market,
        cooldown_bars=cfg.post_exit_cooldown_bars,
        opposite_direction_cooldown_bars=(cfg.opposite_direction_post_exit_cooldown_bars),
        post_exit_liquidity_shock_cooldown_bars=(cfg.post_exit_liquidity_shock_cooldown_bars),
        takeover_edge=cfg.controller_takeover_edge,
    )
    if POST_EXIT_BREADTH_RECOVERY in cfg.enabled_mechanisms:
        for _ in range(4):
            controller_actions = _concat_action_frames(
                [proposal["actions"] for proposal in selected],
            )
            post_exit_proposals = _post_exit_proposals(
                market,
                controller_actions=controller_actions,
                policy=cfg.post_exit_policy,
            )
            next_selected = _select_non_overlapping_proposals(
                [*base_proposals, *post_exit_proposals],
                market=market,
                cooldown_bars=cfg.post_exit_cooldown_bars,
                opposite_direction_cooldown_bars=(cfg.opposite_direction_post_exit_cooldown_bars),
                post_exit_liquidity_shock_cooldown_bars=(cfg.post_exit_liquidity_shock_cooldown_bars),
                takeover_edge=cfg.controller_takeover_edge,
            )
            if _selection_signature(next_selected) == _selection_signature(
                selected,
            ):
                break
            selected = next_selected
    if not selected:
        return _empty_cycles(), _empty_actions()

    actions = _concat_action_frames(
        [proposal["actions"] for proposal in selected],
    ).sort_values(["signal_time", "signal_id"], kind="stable")
    actions = _close_actual_controller_inventory(actions.reset_index(drop=True))
    actions = classify_lot_inventory_lifecycle(actions.reset_index(drop=True))
    cycles = pd.DataFrame.from_records(
        [
            _summarize_selected_campaign(
                group,
                round_trip_cost_bps=cfg.round_trip_cost_bps,
            )
            for _, group in actions.groupby("chain_id", sort=False)
        ],
    )
    return cycles, actions


def _evaluate_threshold_consensus(
    market: pd.DataFrame,
    *,
    policy: UnifiedRegimeLifecyclePolicy,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """只保留在多套连续阈值下已因果确认的完整基线生命周期。"""

    variants: list[tuple[float, pd.DataFrame, pd.DataFrame]] = []
    dispersion = float(policy.threshold_consensus_dispersion_multiplier)
    single = replace(policy, threshold_consensus_enabled=False)
    for declared in policy.threshold_consensus_multipliers:
        effective = 1.0 + (float(declared) - 1.0) * dispersion
        variant_policy = single.with_threshold_multiplier(effective)
        cycles, actions = evaluate_unified_regime_lifecycle(
            market,
            policy=variant_policy,
        )
        variants.append((float(declared), cycles, actions))

    baseline_index = min(
        range(len(variants)),
        key=lambda index: abs(variants[index][0] - 1.0),
    )
    _, baseline_cycles, baseline_actions = variants[baseline_index]
    if baseline_actions.empty:
        return baseline_cycles, baseline_actions

    tolerance = pd.Timedelta(
        minutes=5 * int(policy.threshold_consensus_time_tolerance_bars),
    )
    prepared: list[pd.DataFrame] = []
    for _, _, actions in variants:
        frame = actions.copy()
        if not frame.empty:
            frame["signal_time"] = pd.to_datetime(frame["signal_time"], utc=True)
        prepared.append(frame)

    accepted_chains: list[str] = []
    baseline = prepared[baseline_index]
    for chain_id, chain in baseline.groupby("chain_id", sort=False):
        complete_consensus = True
        for _, action in chain.iterrows():
            action_time = pd.Timestamp(action["signal_time"])
            votes = 1
            for index, candidate_frame in enumerate(prepared):
                if index == baseline_index or candidate_frame.empty:
                    continue
                candidates = candidate_frame.loc[
                    candidate_frame["operation_type"]
                    .astype(str)
                    .eq(
                        str(action["operation_type"]),
                    )
                    & candidate_frame["action"]
                    .astype(str)
                    .eq(
                        str(action["action"]),
                    )
                    & candidate_frame["reason_family"]
                    .astype(str)
                    .eq(
                        str(action["reason_family"]),
                    )
                ]
                if candidates.empty:
                    continue
                candidate_times = pd.to_datetime(
                    candidates["signal_time"],
                    utc=True,
                )
                causal_match = candidate_times.le(action_time) & candidate_times.ge(
                    action_time - tolerance,
                )
                if bool(causal_match.any()):
                    votes += 1
            if votes < int(policy.threshold_consensus_required_votes):
                complete_consensus = False
                break
        if complete_consensus:
            accepted_chains.append(str(chain_id))

    if not accepted_chains:
        return _empty_cycles(), _empty_actions()
    accepted = set(accepted_chains)
    actions = baseline.loc[baseline["chain_id"].astype(str).isin(accepted)].reset_index(drop=True)
    cycles = baseline_cycles.loc[baseline_cycles["chain_id"].astype(str).isin(accepted)].reset_index(drop=True)
    return cycles, actions


def _liquidity_shock_proposals(
    market: pd.DataFrame,
    *,
    policy: UnifiedRegimeLifecyclePolicy,
) -> list[dict[str, Any]]:
    """生成流动性冲击、吸收和恢复类候选活动。"""

    mechanism = LIQUIDITY_SHOCK
    # 计算 factor，仅使用当前及历史数据。
    # 计算 factor，仅使用当前及历史数据。
    factor = CountertrendFactorPolicy(
        enabled_mechanisms=(mechanism,),
        liquidity_shock_exhaustion_floor=policy.shock_score_floor,
    )
    campaign = CountertrendCampaignPolicy(
        maximum_adds=policy.shock_maximum_adds,
        add_distance_atr_multiple=policy.shock_add_distance_atr_multiple,
        recovery_target_pct=policy.shock_recovery_target_pct,
        hard_stop_pct=policy.shock_hard_stop_pct,
        minimum_holding_bars=policy.shock_minimum_holding_bars,
        maximum_holding_bars=policy.shock_maximum_holding_bars,
        post_exit_cooldown_bars=0,
        round_trip_cost_bps=policy.round_trip_cost_bps,
    )
    raw_scores = build_countertrend_scores(market, policy=factor)
    confirmed_scores = _build_shock_confirmation_scores(
        market,
        raw_scores=raw_scores,
        factor_policy=factor,
        policy=policy,
    )
    cycles = evaluate_scored_countertrend_campaigns(
        market,
        scores=confirmed_scores,
        mechanism=mechanism,
        score_threshold=factor.threshold(mechanism),
        release_hysteresis=factor.release_hysteresis,
        minimum_directional_edge=factor.minimum_directional_edge,
        campaign_policy=campaign,
    )
    close_by_time = market.set_index("signal_time")["close"]
    symbol = str(market.iloc[0]["symbol"])
    proposals: list[dict[str, Any]] = []
    for cycle_index, cycle in cycles.reset_index(drop=True).iterrows():
        chain_id = f"unified:{mechanism}:{cycle_index + 1:06d}"
        direction = "LONG" if str(cycle["action"]) == "OPEN_LONG" else "SHORT"
        fill_times = [pd.Timestamp(value) for value in str(cycle["fill_times"]).split("|") if value]
        entry_time = pd.Timestamp(fill_times[0])
        entry_bar = market.loc[market["signal_time"].eq(entry_time)].iloc[0]
        direction_value = 1.0 if direction == "LONG" else -1.0
        if _is_orderly_washout_entry(
            entry_bar,
            direction=direction_value,
            policy=policy,
        ):
            requires_cooling = _requires_orderly_washout_cooling(
                entry_bar,
                direction=direction_value,
                policy=policy,
            )
            if requires_cooling:
                confirmed_entry = _confirmed_orderly_washout_entry(
                    market,
                    setup_time=entry_time,
                    direction=direction_value,
                    policy=policy,
                )
                if confirmed_entry is None:
                    continue
                entry_time = pd.Timestamp(confirmed_entry["signal_time"])
                entry_price = float(confirmed_entry["close"])
            else:
                entry_price = float(close_by_time.loc[entry_time])
            records, _ = _orderly_washout_actions(
                market,
                chain_id=chain_id,
                symbol=symbol,
                direction=direction,
                entry_time=entry_time,
                entry_price=entry_price,
                score=float(cycle["entry_score"]),
                policy=policy,
                confirmed_cooling=requires_cooling,
            )
        elif _is_blowoff_ladder_entry(
            entry_bar,
            direction=direction_value,
            policy=policy,
        ):
            records, _ = _blowoff_ladder_actions(
                market,
                chain_id=chain_id,
                symbol=symbol,
                direction=direction,
                entry_time=entry_time,
                entry_price=float(close_by_time.loc[entry_time]),
                score=float(cycle["entry_score"]),
                policy=policy,
            )
        elif _is_blowoff_ladder_shape(
            entry_bar,
            direction=direction_value,
            policy=policy,
        ):
            # 末端冲高形态缺少波动或成交扩张时，不能降级为普通冲击。
            continue
        elif _is_severe_capitulation(
            entry_bar,
            direction=direction_value,
            policy=policy,
        ):
            records, _ = _crash_recovery_actions(
                market,
                chain_id=chain_id,
                symbol=symbol,
                direction=direction,
                entry_time=entry_time,
                entry_price=float(close_by_time.loc[entry_time]),
                score=float(cycle["entry_score"]),
                policy=policy,
            )
        elif _is_trend_stall_entry(
            entry_bar,
            direction=direction_value,
            policy=policy,
        ):
            records, _ = _trend_stall_actions(
                market,
                chain_id=chain_id,
                symbol=symbol,
                direction=direction,
                entry_time=entry_time,
                entry_price=float(close_by_time.loc[entry_time]),
                score=float(cycle["entry_score"]),
                policy=policy,
            )
        elif _is_trend_preserving_absorption(
            entry_bar,
            direction=direction_value,
            policy=policy,
        ):
            records, _ = _trend_preserving_absorption_actions(
                market,
                chain_id=chain_id,
                symbol=symbol,
                direction=direction,
                entry_time=entry_time,
                entry_price=float(close_by_time.loc[entry_time]),
                score=float(cycle["entry_score"]),
                policy=policy,
            )
        elif _is_high_quality_absorption(
            entry_bar,
            direction=direction_value,
            policy=policy,
        ) or _is_closed_bar_liquid_capitulation(
            entry_bar,
            direction=direction_value,
            policy=policy,
        ):
            records, _ = _absorption_recovery_actions(
                market,
                chain_id=chain_id,
                symbol=symbol,
                direction=direction,
                entry_time=entry_time,
                entry_price=float(close_by_time.loc[entry_time]),
                score=float(cycle["entry_score"]),
                policy=policy,
            )
        else:
            generic_fill_times = list(fill_times)
            if len(generic_fill_times) == 1:
                confirmed_entry = _confirmed_generic_shock_entry(
                    market,
                    entry_bar=entry_bar,
                    setup_time=entry_time,
                    direction=direction_value,
                    policy=policy,
                )
                if confirmed_entry is None:
                    continue
                entry_time = pd.Timestamp(confirmed_entry["signal_time"])
                generic_fill_times = [entry_time]
            campaign_risk_profile = (
                LOW_LIQUIDITY_RETEST_PROBE
                if _is_low_liquidity_retest_entry(
                    entry_bar,
                    direction=direction_value,
                    policy=policy,
                )
                else ""
            )
            records = []
            for fill_index, fill_time in enumerate(generic_fill_times):
                operation_type = "OPEN" if fill_index == 0 else "ADD"
                records.append(
                    _action_record(
                        chain_id=chain_id,
                        symbol=symbol,
                        action=f"OPEN_{direction}",
                        operation_type=operation_type,
                        signal_time=fill_time,
                        signal_price=float(close_by_time.loc[fill_time]),
                        position_fraction=1.0,
                        mechanism=mechanism,
                        score=float(cycle["entry_score"]),
                        campaign_risk_profile=campaign_risk_profile,
                    ),
                )
            if str(cycle["exit_reason"]) != "development_window_end":
                exit_bar = _normalized_shock_runner_exit(
                    market,
                    proposed_exit_time=pd.Timestamp(cycle["exit_time"]),
                    direction=direction_value,
                    policy=policy,
                )
                if exit_bar is not None:
                    normalized_runner = pd.Timestamp(exit_bar["signal_time"]) != pd.Timestamp(cycle["exit_time"])
                    records.append(
                        _action_record(
                            chain_id=chain_id,
                            symbol=symbol,
                            action=f"CLOSE_{direction}",
                            operation_type="CLOSE",
                            signal_time=pd.Timestamp(exit_bar["signal_time"]),
                            signal_price=float(exit_bar["close"]),
                            position_fraction=float(len(generic_fill_times)),
                            mechanism=mechanism,
                            score=float(cycle["entry_score"]),
                            reason_detail=(NORMALIZED_RUNNER_EXIT_STATE if normalized_runner else ""),
                        ),
                    )
        actions = pd.DataFrame.from_records(records)
        closes = actions.loc[
            actions["operation_type"].astype(str).eq("CLOSE"),
            "signal_time",
        ]
        proposals.append(
            {
                "chain_id": chain_id,
                "mechanism": mechanism,
                "score": float(cycle["entry_score"]),
                "start": entry_time,
                # 说明：以下逻辑仅使用决策时点可获得的数据。
                # 说明：以下逻辑仅使用决策时点可获得的数据。
                "end": (
                    pd.Timestamp(closes.iloc[-1]) if not closes.empty else pd.Timestamp(market.iloc[-1]["signal_time"])
                ),
                "actions": actions,
            },
        )
    return proposals


def _normalized_shock_runner_exit(
    market: pd.DataFrame,
    *,
    proposed_exit_time: pd.Timestamp,
    direction: float,
    policy: UnifiedRegimeLifecyclePolicy,
) -> pd.Series | None:
    """为标准化冲击活动计算 runner 的退出动作。"""

    normalized = _base_normalized_shock_runner_exit(
        market,
        proposed_exit_time=proposed_exit_time,
        direction=direction,
        policy=policy,
    )
    if normalized is None:
        return None
    matches = market.index[market["signal_time"].eq(pd.Timestamp(normalized["signal_time"]))]
    if matches.empty:
        return normalized
    extension_active, extension_exit = _generic_shock_momentum_extension_exit(
        market,
        proposed_exit_index=int(matches[0]),
        direction=direction,
        policy=policy,
    )
    return extension_exit if extension_active else normalized


def _base_normalized_shock_runner_exit(
    market: pd.DataFrame,
    *,
    proposed_exit_time: pd.Timestamp,
    direction: float,
    policy: UnifiedRegimeLifecyclePolicy,
) -> pd.Series | None:
    """先运行既有 runner 标准化，供可选趋势延续层继续判断。"""

    matches = market.index[market["signal_time"].eq(proposed_exit_time)]
    if matches.empty:
        return None
    exit_index = int(matches[0])
    proposed = market.iloc[exit_index]
    directional = _shock_directional(proposed, direction)
    hot_trend_exit = bool(
        float(proposed["atr14_ratio120"]) >= policy.shock_absorption_resilient_volatility_ratio
        and directional["recovery_return"] >= policy.shock_recovery_target_pct
        and directional["location"] >= policy.shock_absorption_maturity_location
        and directional["rsi"] >= policy.shock_absorption_maturity_rsi
        and directional["favorable_breadth20"] >= policy.shock_absorption_recovery_breadth
        and float(proposed["quote_volume_ratio_med30"]) >= policy.shock_absorption_maturity_quote_volume_ratio,
    )
    if not hot_trend_exit:
        return proposed

    future = market.iloc[exit_index + 1 : exit_index + 1 + policy.shock_orderly_washout_maximum_holding_bars]
    for _, bar in future.iterrows():
        directional = _shock_directional(bar, direction)
        if (
            float(bar["atr14_ratio120"]) <= policy.shock_absorption_resilient_volatility_ratio
            and directional["location"] >= policy.shock_absorption_maturity_location
            and directional["rsi"] >= policy.shock_absorption_maturity_rsi
            and directional["favorable_breadth20"] >= policy.shock_absorption_recovery_breadth
            and float(bar["quote_volume_ratio_med30"]) >= policy.shock_absorption_maturity_quote_volume_ratio
        ):
            return bar
    if len(future) >= policy.shock_orderly_washout_maximum_holding_bars:
        return future.iloc[-1]
    # 返回基于当前及历史数据得到的结果。
    return None


def _confirmed_generic_shock_entry(
    market: pd.DataFrame,
    *,
    entry_bar: pd.Series,
    setup_time: pd.Timestamp,
    direction: float,
    policy: UnifiedRegimeLifecyclePolicy,
) -> pd.Series | None:
    """让强趋势中的普通冲击等待同一套公开市场冷却确认。"""

    if not policy.shock_generic_countertrend_requires_cooling_confirmation:
        return entry_bar
    directional = _shock_directional(entry_bar, direction)
    strong_countertrend = bool(
        directional["adverse_mid_return"] >= policy.shock_orderly_washout_adverse_mid_minimum_pct
        and float(entry_bar["adx14"]) >= policy.shock_orderly_washout_minimum_adx
        and float(entry_bar["atr14_ratio120"]) >= policy.shock_absorption_resilient_volatility_ratio,
    )
    if not strong_countertrend:
        return entry_bar
    return _confirmed_orderly_washout_entry(
        market,
        setup_time=setup_time,
        direction=direction,
        policy=policy,
    )


def _generic_shock_momentum_extension_exit(
    market: pd.DataFrame,
    *,
    proposed_exit_index: int,
    direction: float,
    policy: UnifiedRegimeLifecyclePolicy,
) -> tuple[bool, pd.Series | None]:
    """在广度和量价仍单边时延续 runner，等待高潮或反转。"""

    if not policy.shock_generic_momentum_extension_enabled:
        return False, None
    proposed = market.iloc[proposed_exit_index]
    directional = _shock_directional(proposed, direction)
    extension_active = bool(
        directional["mid_return"] >= policy.shock_absorption_recovery_return_pct
        and directional["adverse_long_return"] <= -policy.shock_trend_preserving_long_band_pct
        and directional["location"] >= policy.shock_orderly_washout_exit_location
        and directional["rsi"] >= policy.shock_absorption_maturity_rsi
        and directional["favorable_breadth20"] >= policy.shock_absorption_recovery_breadth
        and float(proposed["quote_volume_ratio_med30"]) >= policy.shock_absorption_maturity_quote_volume_ratio,
    )
    if not extension_active:
        return False, None
    future = market.iloc[proposed_exit_index + 1 : proposed_exit_index + 1 + policy.shock_confirmation_timeout_bars]
    for _, bar in future.iterrows():
        directional = _shock_directional(bar, direction)
        terminal_extension = bool(
            directional["mid_return"] >= policy.shock_orderly_washout_terminal_mid_return_pct
            and directional["rsi"] >= policy.shock_orderly_washout_terminal_rsi
            and float(bar["adx14"]) >= policy.shock_orderly_washout_terminal_minimum_adx
            and float(bar["quote_volume_ratio_med30"]) >= policy.shock_severe_capitulation_quote_volume_ratio,
        )
        reversal_exit = bool(
            directional["location"] >= policy.shock_orderly_washout_exit_location
            and directional["rsi"] >= policy.shock_orderly_washout_exit_rsi
            and directional["adverse_long_return"] <= 0.0
            and float(bar["adx14"]) <= policy.shock_orderly_washout_exit_maximum_adx
            and directional["tiny_return"] <= -policy.shock_orderly_washout_exit_reversal_tiny_return_pct
            and directional["adverse_breadth3"] >= policy.shock_orderly_washout_exit_reversal_adverse_breadth3,
        )
        if terminal_extension or reversal_exit:
            return True, bar
    if len(future) >= policy.shock_confirmation_timeout_bars:
        return True, future.iloc[-1]
    return True, None


def _is_high_quality_absorption(
    bar: pd.Series,
    *,
    direction: float,
    policy: UnifiedRegimeLifecyclePolicy,
) -> bool:
    """判断冲击后的成交吸收是否具备足够恢复质量。"""

    directional = _shock_directional(bar, direction)
    long_horizon_dislocated = bool(
        abs(directional["adverse_long_return"]) > policy.shock_trend_preserving_long_band_pct,
    )
    broad_short_reversal = bool(
        directional["favorable_breadth3"] >= policy.shock_rejection_breadth,
    )
    seller_rejection = bool(
        directional["tiny_return"] >= policy.shock_rejection_tiny_return_pct
        or (
            directional["candle_return"] >= policy.shock_rejection_tiny_return_pct
            and directional["candle_close_location"] >= policy.shock_absorption_candle_close_location
        ),
    )
    return bool(
        directional["mid_return"] <= -policy.shock_absorption_adverse_mid_return_pct
        # 吸收信号必须伴随长周期错位或宽基空头反转，
        # 避免仅凭局部波动触发生命周期状态切换。
        and (long_horizon_dislocated or broad_short_reversal)
        and directional["location"] <= policy.shock_absorption_location
        and directional["rsi"] <= policy.shock_absorption_rsi
        and directional["adverse_breadth20"] >= policy.shock_absorption_adverse_breadth
        and seller_rejection
        and float(bar["quote_volume_ratio_med30"]) >= policy.shock_absorption_quote_volume_ratio,
    )


def _is_severe_capitulation(
    bar: pd.Series,
    *,
    direction: float,
    policy: UnifiedRegimeLifecyclePolicy,
) -> bool:
    """判断价格、广度和流动性是否构成严重投降状态。"""

    directional = _shock_directional(bar, direction)
    return bool(
        directional["adverse_mid_return"] >= policy.shock_severe_capitulation_adverse_mid_return_pct
        and directional["adverse_long_return"] <= 0.0
        and directional["location"] <= policy.shock_severe_capitulation_location
        and directional["rsi"] <= policy.shock_severe_capitulation_rsi
        and directional["adverse_breadth20"] >= policy.shock_severe_capitulation_adverse_breadth
        and float(bar["quote_volume_ratio_med30"]) >= policy.shock_severe_capitulation_quote_volume_ratio,
    )


def _is_trend_stall_entry(
    bar: pd.Series,
    *,
    direction: float,
    policy: UnifiedRegimeLifecyclePolicy,
) -> bool:
    """识别强趋势成熟后的停滞反转入口。"""

    directional = _shock_directional(bar, direction)
    return bool(
        policy.shock_confirmation_minimum_adverse_mid_return_pct
        <= directional["adverse_mid_return"]
        < policy.shock_orderly_washout_adverse_mid_maximum_pct
        and policy.shock_orderly_washout_adverse_long_minimum_pct
        <= directional["adverse_long_return"]
        < policy.shock_strong_trend_defer_pct
        and directional["location"] <= policy.shock_orderly_washout_scaleout_locations[0]
        and directional["tiny_return"] >= policy.shock_rejection_tiny_return_pct
        and directional["favorable_breadth3"] >= policy.shock_rejection_breadth
        and float(bar["adx14"]) >= policy.shock_trend_stall_minimum_adx
        and float(bar["quote_volume_ratio_med30"]) >= policy.shock_confirmation_minimum_quote_volume_ratio,
    )


def _is_trend_preserving_absorption(
    bar: pd.Series,
    *,
    direction: float,
    policy: UnifiedRegimeLifecyclePolicy,
) -> bool:
    """识别不破坏原趋势结构的回撤吸收。"""

    directional = _shock_directional(bar, direction)
    return bool(
        abs(directional["adverse_long_return"]) <= policy.shock_trend_preserving_long_band_pct
        and _is_high_quality_absorption(
            bar,
            direction=direction,
            policy=policy,
        ),
    )


def _trend_stall_actions(
    market: pd.DataFrame,
    *,
    chain_id: str,
    symbol: str,
    direction: str,
    entry_time: pd.Timestamp,
    entry_price: float,
    score: float,
    policy: UnifiedRegimeLifecyclePolicy,
) -> tuple[list[dict[str, object]], pd.Timestamp | None]:
    """构建趋势停滞反转的开仓和退出动作。"""

    direction_value = 1.0 if direction == "LONG" else -1.0
    entry_index = int(market.index[market["signal_time"].eq(entry_time)][0])
    future = market.iloc[entry_index + 1 : entry_index + 1 + policy.shock_maximum_holding_bars]
    exit_bar: pd.Series | None = None
    for _, bar in future.iterrows():
        directional = _shock_directional(bar, direction_value)
        if (
            directional["mid_return"] >= policy.shock_strong_trend_defer_pct
            and directional["location"] >= 1.0 - policy.shock_retest_location
            and directional["rsi"] >= 100.0 - policy.shock_orderly_washout_capitulation_rsi
            and directional["favorable_breadth20"] >= policy.shock_absorption_adverse_breadth
            and float(bar["quote_volume_ratio_med30"]) >= policy.shock_absorption_quote_volume_ratio
        ):
            exit_bar = bar
            break
    if exit_bar is None:
        exit_bar = future.iloc[-1] if len(future) >= policy.shock_maximum_holding_bars else None
    return _single_open_close_actions(
        chain_id=chain_id,
        symbol=symbol,
        direction=direction,
        entry_time=entry_time,
        entry_price=entry_price,
        score=score,
        exit_bar=exit_bar,
    )


def _crash_recovery_actions(
    market: pd.DataFrame,
    *,
    chain_id: str,
    symbol: str,
    direction: str,
    entry_time: pd.Timestamp,
    entry_price: float,
    score: float,
    policy: UnifiedRegimeLifecyclePolicy,
) -> tuple[list[dict[str, object]], pd.Timestamp | None]:
    """构建极端下跌后获得确认的恢复动作。"""

    direction_value = 1.0 if direction == "LONG" else -1.0
    entry_index = int(market.index[market["signal_time"].eq(entry_time)][0])
    future = market.iloc[entry_index + 1 : entry_index + 1 + policy.shock_absorption_maximum_holding_bars]
    exit_bar: pd.Series | None = None
    for _, bar in future.iterrows():
        directional = _shock_directional(bar, direction_value)
        if (
            directional["adverse_long_return"] >= policy.shock_crash_recovery_adverse_long_return_pct
            and directional["location"] >= policy.shock_crash_recovery_location
            and directional["rsi"] >= policy.shock_crash_recovery_rsi
            and directional["tiny_return"] >= policy.shock_crash_recovery_tiny_return_pct
            and directional["favorable_breadth20"] >= policy.shock_absorption_recovery_breadth
        ):
            exit_bar = bar
            break
    if exit_bar is None:
        exit_bar = future.iloc[-1] if len(future) >= policy.shock_absorption_maximum_holding_bars else None
    return _single_open_close_actions(
        chain_id=chain_id,
        symbol=symbol,
        direction=direction,
        entry_time=entry_time,
        entry_price=entry_price,
        score=score,
        exit_bar=exit_bar,
    )


def _trend_preserving_absorption_actions(
    market: pd.DataFrame,
    *,
    chain_id: str,
    symbol: str,
    direction: str,
    entry_time: pd.Timestamp,
    entry_price: float,
    score: float,
    policy: UnifiedRegimeLifecyclePolicy,
) -> tuple[list[dict[str, object]], pd.Timestamp | None]:
    """构建顺趋势吸收后的恢复持仓动作。"""

    direction_value = 1.0 if direction == "LONG" else -1.0
    entry_index = int(market.index[market["signal_time"].eq(entry_time)][0])
    future = market.iloc[entry_index + 1 : entry_index + 1 + policy.shock_absorption_maximum_holding_bars]
    exit_bar: pd.Series | None = None
    for _, bar in future.iterrows():
        directional = _shock_directional(bar, direction_value)
        if (
            directional["recovery_return"] >= policy.shock_absorption_recovery_return_pct
            and directional["location"] >= policy.shock_trend_preserving_exit_location
            and directional["rsi"] >= policy.shock_trend_preserving_exit_rsi
            and directional["favorable_breadth20"] >= policy.shock_absorption_recovery_breadth
        ):
            exit_bar = bar
            break
    if exit_bar is None:
        exit_bar = future.iloc[-1] if len(future) >= policy.shock_absorption_maximum_holding_bars else None
    return _single_open_close_actions(
        chain_id=chain_id,
        symbol=symbol,
        direction=direction,
        entry_time=entry_time,
        entry_price=entry_price,
        score=score,
        exit_bar=exit_bar,
    )


def _single_open_close_actions(
    *,
    chain_id: str,
    symbol: str,
    direction: str,
    entry_time: pd.Timestamp,
    entry_price: float,
    score: float,
    exit_bar: pd.Series | None,
) -> tuple[list[dict[str, object]], pd.Timestamp | None]:
    """为单次机制生成配对的开仓和平仓动作。"""

    records = [
        _action_record(
            chain_id=chain_id,
            symbol=symbol,
            action=f"OPEN_{direction}",
            operation_type="OPEN",
            signal_time=entry_time,
            signal_price=entry_price,
            position_fraction=1.0,
            mechanism=LIQUIDITY_SHOCK,
            score=score,
        ),
    ]
    if exit_bar is None:
        return records, None
    exit_time = pd.Timestamp(exit_bar["signal_time"])
    records.append(
        _action_record(
            chain_id=chain_id,
            symbol=symbol,
            action=f"CLOSE_{direction}",
            operation_type="CLOSE",
            signal_time=exit_time,
            signal_price=float(exit_bar["close"]),
            position_fraction=1.0,
            mechanism=LIQUIDITY_SHOCK,
            score=score,
        ),
    )
    return records, exit_time


def _is_orderly_washout_entry(
    bar: pd.Series,
    *,
    direction: float,
    policy: UnifiedRegimeLifecyclePolicy,
) -> bool:
    """判断有序清洗是否同时满足冲击和结构保留条件。"""

    directional = _shock_directional(bar, direction)
    rejected_washout = bool(
        policy.shock_orderly_washout_adverse_mid_minimum_pct
        <= directional["adverse_mid_return"]
        < policy.shock_orderly_washout_adverse_mid_maximum_pct
        and directional["tiny_return"] >= policy.shock_rejection_tiny_return_pct
        and directional["favorable_breadth3"] >= policy.shock_rejection_breadth,
    )
    capitulation_probe = _is_orderly_washout_capitulation_probe(
        bar,
        direction=direction,
        policy=policy,
    )
    return bool(
        (rejected_washout or capitulation_probe)
        and directional["adverse_long_return"] >= policy.shock_orderly_washout_adverse_long_minimum_pct
        and float(bar["adx14"]) >= policy.shock_orderly_washout_minimum_adx,
    )


def _is_orderly_washout_capitulation_probe(
    bar: pd.Series,
    *,
    direction: float,
    policy: UnifiedRegimeLifecyclePolicy,
) -> bool:
    """识别需等待冷却确认的高波动清洗探针。"""

    directional = _shock_directional(bar, direction)
    return bool(
        directional["adverse_mid_return"] >= policy.shock_orderly_washout_adverse_long_minimum_pct
        and directional["adverse_mid_return"] < policy.shock_blowoff_entry_adverse_mid_maximum_pct
        and directional["location"] <= policy.shock_severe_capitulation_location
        and directional["rsi"] <= policy.shock_orderly_washout_capitulation_rsi
        and float(bar["quote_volume_ratio_med30"]) >= policy.shock_absorption_quote_volume_ratio,
    )


def _requires_orderly_washout_cooling(
    bar: pd.Series,
    *,
    direction: float,
    policy: UnifiedRegimeLifecyclePolicy,
) -> bool:
    """仅让未达到即时高潮强度的清洗探针等待冷却确认。"""

    return bool(
        policy.shock_orderly_washout_requires_cooling_confirmation
        and _is_orderly_washout_capitulation_probe(
            bar,
            direction=direction,
            policy=policy,
        )
        and float(bar["quote_volume_ratio_med30"]) < policy.shock_immediate_quote_volume_ratio,
    )


def _confirmed_orderly_washout_entry(
    market: pd.DataFrame,
    *,
    setup_time: pd.Timestamp,
    direction: float,
    policy: UnifiedRegimeLifecyclePolicy,
) -> pd.Series | None:
    """等待冲击后的波动与趋势冷却，再确认有序清洗入口。"""

    matches = market.index[market["signal_time"].eq(setup_time)]
    if matches.empty:
        return None
    setup_index = int(matches[0])
    start_index = setup_index + policy.shock_orderly_washout_confirmation_minimum_bars
    end_index = min(
        len(market) - 1,
        setup_index + policy.shock_confirmation_timeout_bars,
    )
    for bar_index in range(start_index, end_index + 1):
        bar = market.iloc[bar_index]
        directional = _shock_directional(bar, direction)
        if (
            abs(directional["tiny_return"]) <= policy.shock_orderly_washout_confirmation_maximum_abs_tiny_return_pct
            and abs(directional["recovery_return"])
            <= policy.shock_orderly_washout_confirmation_maximum_abs_recovery_return_pct
            and policy.shock_orderly_washout_confirmation_location_minimum
            <= directional["location"]
            <= policy.shock_orderly_washout_confirmation_location_maximum
            and policy.shock_orderly_washout_confirmation_rsi_minimum
            <= directional["rsi"]
            <= policy.shock_orderly_washout_confirmation_rsi_maximum
            and float(bar["adx14"]) <= policy.shock_orderly_washout_confirmation_maximum_adx
            and float(bar["quote_volume_ratio_med30"])
            <= policy.shock_orderly_washout_confirmation_maximum_quote_volume_ratio
        ):
            return bar
    return None


def _orderly_washout_actions(
    market: pd.DataFrame,
    *,
    chain_id: str,
    symbol: str,
    direction: str,
    entry_time: pd.Timestamp,
    entry_price: float,
    score: float,
    policy: UnifiedRegimeLifecyclePolicy,
    confirmed_cooling: bool = False,
) -> tuple[list[dict[str, object]], pd.Timestamp | None]:
    """构建确认后的有序清洗、单次止盈、回抽加回和末端退出。"""

    if not confirmed_cooling:
        return _immediate_orderly_washout_actions(
            market,
            chain_id=chain_id,
            symbol=symbol,
            direction=direction,
            entry_time=entry_time,
            entry_price=entry_price,
            score=score,
            policy=policy,
        )
    direction_value = 1.0 if direction == "LONG" else -1.0
    entry_index = int(market.index[market["signal_time"].eq(entry_time)][0])
    future = market.iloc[entry_index + 1 : entry_index + 1 + policy.shock_orderly_washout_maximum_holding_bars]
    records = [
        _action_record(
            chain_id=chain_id,
            symbol=symbol,
            action=f"OPEN_{direction}",
            operation_type="OPEN",
            signal_time=entry_time,
            signal_price=entry_price,
            position_fraction=1.0,
            mechanism=LIQUIDITY_SHOCK,
            score=score,
        ),
    ]
    reduced = False
    reentered = False
    best_directional_return_pct = 0.0
    exit_bar: pd.Series | None = None
    terminal_extension_exit = False
    for _, bar in future.iterrows():
        directional = _shock_directional(bar, direction_value)
        directional_return_pct = direction_value * (float(bar["close"]) / entry_price - 1.0) * 100.0
        best_directional_return_pct = max(
            best_directional_return_pct,
            directional_return_pct,
        )
        if (
            not reduced
            and directional["location"] >= policy.shock_orderly_washout_exit_location
            and directional_return_pct >= policy.shock_orderly_washout_scaleout_minimum_return_pct
        ):
            reduced = True
            records.append(
                _action_record(
                    chain_id=chain_id,
                    symbol=symbol,
                    action=f"CLOSE_{direction}",
                    operation_type="REDUCE",
                    signal_time=pd.Timestamp(bar["signal_time"]),
                    signal_price=float(bar["close"]),
                    position_fraction=policy.shock_orderly_washout_scaleout_fraction,
                    mechanism=LIQUIDITY_SHOCK,
                    score=score,
                ),
            )
            continue
        giveback_pct = best_directional_return_pct - directional_return_pct
        if (
            reduced
            and not reentered
            and giveback_pct >= policy.shock_orderly_washout_reentry_minimum_giveback_pct
            and directional["recovery_return"] >= policy.shock_orderly_washout_reentry_minimum_recovery_return_pct
            and directional["location"] >= policy.shock_absorption_maturity_location
            and directional["candle_return"]
            <= (-policy.shock_orderly_washout_reentry_minimum_adverse_candle_return_pct)
        ):
            reentered = True
            records.append(
                _action_record(
                    chain_id=chain_id,
                    symbol=symbol,
                    action=f"OPEN_{direction}",
                    operation_type="ADD",
                    signal_time=pd.Timestamp(bar["signal_time"]),
                    signal_price=float(bar["close"]),
                    position_fraction=policy.shock_orderly_washout_reentry_units,
                    mechanism=LIQUIDITY_SHOCK,
                    score=score,
                ),
            )
            continue
        terminal_extension = bool(
            directional["mid_return"] >= policy.shock_orderly_washout_terminal_mid_return_pct
            and directional["location"] >= policy.shock_orderly_washout_terminal_location
            and directional["rsi"] >= policy.shock_orderly_washout_terminal_rsi
            and float(bar["adx14"]) >= policy.shock_orderly_washout_terminal_minimum_adx
            and float(bar["quote_volume_ratio_med30"]) >= policy.shock_severe_capitulation_quote_volume_ratio,
        )
        reversal_exit = bool(
            directional["location"] >= policy.shock_orderly_washout_exit_location
            and directional["rsi"] >= policy.shock_orderly_washout_exit_rsi
            and directional["adverse_long_return"] <= 0.0
            and float(bar["adx14"]) <= policy.shock_orderly_washout_exit_maximum_adx
            and directional["tiny_return"] <= -policy.shock_orderly_washout_exit_reversal_tiny_return_pct
            and directional["adverse_breadth3"] >= policy.shock_orderly_washout_exit_reversal_adverse_breadth3,
        )
        if terminal_extension or reversal_exit:
            exit_bar = bar
            terminal_extension_exit = terminal_extension
            break
    if exit_bar is None:
        exit_bar = future.iloc[-1] if len(future) >= policy.shock_orderly_washout_maximum_holding_bars else None
    remaining_units = max(
        0.0,
        1.0
        - (policy.shock_orderly_washout_scaleout_fraction if reduced else 0.0)
        + (policy.shock_orderly_washout_reentry_units if reentered else 0.0),
    )
    exit_time = pd.Timestamp(exit_bar["signal_time"]) if exit_bar is not None else None
    if remaining_units > 0.0 and exit_bar is not None:
        records.append(
            _action_record(
                chain_id=chain_id,
                symbol=symbol,
                action=f"CLOSE_{direction}",
                operation_type="CLOSE",
                signal_time=exit_time,
                signal_price=float(exit_bar["close"]),
                position_fraction=remaining_units,
                mechanism=LIQUIDITY_SHOCK,
                score=score,
                reason_detail=("exit=terminal_extension" if terminal_extension_exit else "exit=reversal_or_timeout"),
            ),
        )
    return records, exit_time


def _immediate_orderly_washout_actions(
    market: pd.DataFrame,
    *,
    chain_id: str,
    symbol: str,
    direction: str,
    entry_time: pd.Timestamp,
    entry_price: float,
    score: float,
    policy: UnifiedRegimeLifecyclePolicy,
) -> tuple[list[dict[str, object]], pd.Timestamp | None]:
    """保留真正即时高潮样本的原有分级退出。"""

    direction_value = 1.0 if direction == "LONG" else -1.0
    entry_index = int(market.index[market["signal_time"].eq(entry_time)][0])
    future = market.iloc[entry_index + 1 : entry_index + 1 + policy.shock_orderly_washout_maximum_holding_bars]
    records = [
        _action_record(
            chain_id=chain_id,
            symbol=symbol,
            action=f"OPEN_{direction}",
            operation_type="OPEN",
            signal_time=entry_time,
            signal_price=entry_price,
            position_fraction=1.0,
            mechanism=LIQUIDITY_SHOCK,
            score=score,
        ),
    ]
    next_scaleout = 0
    exit_bar: pd.Series | None = None
    for _, bar in future.iterrows():
        directional = _shock_directional(bar, direction_value)
        while (
            next_scaleout < len(policy.shock_orderly_washout_scaleout_locations)
            and directional["location"] >= policy.shock_orderly_washout_scaleout_locations[next_scaleout]
            and direction_value * (float(bar["close"]) / entry_price - 1.0) * 100.0
            >= policy.shock_orderly_washout_scaleout_minimum_return_pct
        ):
            records.append(
                _action_record(
                    chain_id=chain_id,
                    symbol=symbol,
                    action=f"CLOSE_{direction}",
                    operation_type="REDUCE",
                    signal_time=pd.Timestamp(bar["signal_time"]),
                    signal_price=float(bar["close"]),
                    position_fraction=(policy.shock_orderly_washout_immediate_scaleout_fraction),
                    mechanism=LIQUIDITY_SHOCK,
                    score=score,
                ),
            )
            next_scaleout += 1
        if (
            directional["location"] >= policy.shock_orderly_washout_exit_location
            and directional["rsi"] >= policy.shock_orderly_washout_exit_rsi
            and directional["adverse_long_return"] <= 0.0
            and float(bar["adx14"]) <= policy.shock_orderly_washout_exit_maximum_adx
        ):
            exit_bar = bar
            break
    if exit_bar is None:
        exit_bar = future.iloc[-1] if len(future) >= policy.shock_orderly_washout_maximum_holding_bars else None
    remaining_units = max(
        0.0,
        1.0 - next_scaleout * policy.shock_orderly_washout_immediate_scaleout_fraction,
    )
    exit_time = pd.Timestamp(exit_bar["signal_time"]) if exit_bar is not None else None
    if remaining_units > 0.0 and exit_bar is not None:
        records.append(
            _action_record(
                chain_id=chain_id,
                symbol=symbol,
                action=f"CLOSE_{direction}",
                operation_type="CLOSE",
                signal_time=exit_time,
                signal_price=float(exit_bar["close"]),
                position_fraction=remaining_units,
                mechanism=LIQUIDITY_SHOCK,
                score=score,
            ),
        )
    return records, exit_time


def _is_blowoff_ladder_entry(
    bar: pd.Series,
    *,
    direction: float,
    policy: UnifiedRegimeLifecyclePolicy,
) -> bool:
    """识别末端加速衰竭并允许分级反向建仓。"""

    expansion_confirmed = bool(
        float(bar["atr14_ratio120"]) >= policy.shock_blowoff_entry_minimum_atr_ratio
        or float(bar["quote_volume_ratio_med30"])
        >= policy.shock_blowoff_entry_minimum_quote_volume_ratio,
    )
    return _is_blowoff_ladder_shape(
        bar,
        direction=direction,
        policy=policy,
    ) and bool(expansion_confirmed)


def _is_blowoff_ladder_shape(
    bar: pd.Series,
    *,
    direction: float,
    policy: UnifiedRegimeLifecyclePolicy,
) -> bool:
    """识别末端加速形态，不把扩张不足的候选降级到普通冲击。"""

    directional = _shock_directional(bar, direction)
    return bool(
        policy.shock_blowoff_entry_adverse_mid_minimum_pct
        <= directional["adverse_mid_return"]
        <= policy.shock_blowoff_entry_adverse_mid_maximum_pct
        and directional["adverse_long_return"] >= policy.shock_blowoff_entry_adverse_long_minimum_pct
        and float(bar["adx14"]) >= policy.shock_blowoff_entry_minimum_adx,
    )


def _blowoff_ladder_actions(
    market: pd.DataFrame,
    *,
    chain_id: str,
    symbol: str,
    direction: str,
    entry_time: pd.Timestamp,
    entry_price: float,
    score: float,
    policy: UnifiedRegimeLifecyclePolicy,
) -> tuple[list[dict[str, object]], pd.Timestamp | None]:
    """构建末端衰竭的阶梯开仓、加仓和止盈动作。"""

    direction_value = 1.0 if direction == "LONG" else -1.0
    entry_index = int(market.index[market["signal_time"].eq(entry_time)][0])
    future = market.iloc[entry_index + 1 : entry_index + 1 + policy.shock_blowoff_maximum_holding_bars]
    records = [
        _action_record(
            chain_id=chain_id,
            symbol=symbol,
            action=f"OPEN_{direction}",
            operation_type="OPEN",
            signal_time=entry_time,
            signal_price=entry_price,
            position_fraction=policy.shock_blowoff_probe_units,
            mechanism=LIQUIDITY_SHOCK,
            score=score,
        ),
    ]
    first_add_index: int | None = None
    second_add_index: int | None = None
    reduced = False
    first_reduction_index: int | None = None
    followthrough_reduced = False
    exit_bar: pd.Series | None = None
    for bar_index, bar in future.iterrows():
        directional = _shock_directional(bar, direction_value)
        add_state = bool(
            directional["adverse_mid_return"] >= policy.shock_blowoff_add_adverse_mid_return_pct
            and directional["location"] <= policy.shock_blowoff_add_maximum_location
            and directional["rsi"] <= policy.shock_blowoff_add_maximum_rsi
            and float(bar["adx14"]) >= policy.shock_blowoff_add_minimum_adx,
        )
        if first_add_index is None and add_state:
            first_add_index = bar_index
            records.append(
                _action_record(
                    chain_id=chain_id,
                    symbol=symbol,
                    action=f"OPEN_{direction}",
                    operation_type="ADD",
                    signal_time=pd.Timestamp(bar["signal_time"]),
                    signal_price=float(bar["close"]),
                    position_fraction=policy.shock_blowoff_primary_add_units,
                    mechanism=LIQUIDITY_SHOCK,
                    score=score,
                ),
            )
            continue
        if (
            first_add_index is not None
            and second_add_index is None
            and bar_index - first_add_index >= policy.shock_blowoff_second_add_gap_bars
            and add_state
        ):
            second_add_index = bar_index
            records.append(
                _action_record(
                    chain_id=chain_id,
                    symbol=symbol,
                    action=f"OPEN_{direction}",
                    operation_type="ADD",
                    signal_time=pd.Timestamp(bar["signal_time"]),
                    signal_price=float(bar["close"]),
                    position_fraction=policy.shock_blowoff_secondary_add_units,
                    mechanism=LIQUIDITY_SHOCK,
                    score=score,
                ),
            )
            continue
        if (
            second_add_index is not None
            and not reduced
            and directional["tiny_return"] >= policy.shock_rejection_tiny_return_pct
            and directional["favorable_breadth3"] >= policy.shock_rejection_breadth
        ):
            reduced = True
            first_reduction_index = bar_index
            records.append(
                _action_record(
                    chain_id=chain_id,
                    symbol=symbol,
                    action=f"CLOSE_{direction}",
                    operation_type="REDUCE",
                    signal_time=pd.Timestamp(bar["signal_time"]),
                    signal_price=float(bar["close"]),
                    position_fraction=policy.shock_blowoff_reduction_fraction,
                    mechanism=LIQUIDITY_SHOCK,
                    score=score,
                ),
            )
        elif (
            reduced
            and not followthrough_reduced
            and first_reduction_index is not None
            and bar_index > first_reduction_index
            and directional["tiny_return"] >= policy.shock_blowoff_followthrough_return_pct
            and directional["favorable_breadth3"] >= policy.shock_blowoff_followthrough_breadth
        ):
            followthrough_reduced = True
            records.append(
                _action_record(
                    chain_id=chain_id,
                    symbol=symbol,
                    action=f"CLOSE_{direction}",
                    operation_type="REDUCE",
                    signal_time=pd.Timestamp(bar["signal_time"]),
                    signal_price=float(bar["close"]),
                    position_fraction=(policy.shock_blowoff_followthrough_reduction_fraction),
                    mechanism=LIQUIDITY_SHOCK,
                    score=score,
                ),
            )
        probe_return_pct = direction_value * (float(bar["close"]) / entry_price - 1.0) * 100.0
        if (
            first_add_index is None
            and policy.shock_blowoff_probe_take_profit_pct > 0.0
            and probe_return_pct >= policy.shock_blowoff_probe_take_profit_pct
        ):
            exit_bar = bar
            break
        held_bars = bar_index - entry_index
        if (
            held_bars >= policy.shock_blowoff_residual_minimum_holding_bars
            and directional["location"] >= policy.shock_blowoff_residual_location
            and directional["rsi"] >= policy.shock_blowoff_residual_rsi
            and float(bar["adx14"]) <= policy.shock_blowoff_residual_maximum_adx
        ):
            exit_bar = bar
            break
    if exit_bar is None:
        exit_bar = future.iloc[-1] if len(future) >= policy.shock_blowoff_maximum_holding_bars else None
    inventory_units = (
        policy.shock_blowoff_probe_units
        + (policy.shock_blowoff_primary_add_units if first_add_index is not None else 0.0)
        + (policy.shock_blowoff_secondary_add_units if second_add_index is not None else 0.0)
    )
    remaining_units = max(
        0.0,
        inventory_units
        - (policy.shock_blowoff_reduction_fraction if reduced else 0.0)
        - (policy.shock_blowoff_followthrough_reduction_fraction if followthrough_reduced else 0.0),
    )
    exit_time = pd.Timestamp(exit_bar["signal_time"]) if exit_bar is not None else None
    if exit_bar is not None:
        records.append(
            _action_record(
                chain_id=chain_id,
                symbol=symbol,
                action=f"CLOSE_{direction}",
                operation_type="CLOSE",
                signal_time=exit_time,
                signal_price=float(exit_bar["close"]),
                position_fraction=remaining_units,
                mechanism=LIQUIDITY_SHOCK,
                score=score,
            ),
        )
    return records, exit_time


def _absorption_recovery_actions(
    market: pd.DataFrame,
    *,
    chain_id: str,
    symbol: str,
    direction: str,
    entry_time: pd.Timestamp,
    entry_price: float,
    score: float,
    policy: UnifiedRegimeLifecyclePolicy,
) -> tuple[list[dict[str, object]], pd.Timestamp | None]:
    """构建吸收确认后的恢复、成熟和退出动作。"""

    direction_value = 1.0 if direction == "LONG" else -1.0
    entry_index = int(market.index[market["signal_time"].eq(entry_time)][0])
    future = market.iloc[entry_index + 1 : entry_index + 1 + policy.shock_absorption_maximum_holding_bars]
    recent_start = max(
        0,
        entry_index - policy.shock_absorption_deep_scaleout_lookback_bars + 1,
    )
    recent_setup = market.iloc[recent_start : entry_index + 1]
    deep_setup = recent_setup.loc[
        pd.to_numeric(recent_setup["rsi14"], errors="coerce").le(
            policy.shock_absorption_deep_scaleout_rsi,
        )
        & pd.to_numeric(recent_setup["adx14"], errors="coerce").ge(
            policy.shock_absorption_deep_scaleout_minimum_adx,
        )
    ]
    deep_scaleout = not deep_setup.empty
    first_scaleout_pct = (
        policy.shock_absorption_deep_scaleout_first_target_pct
        if deep_scaleout
        else policy.shock_absorption_first_scaleout_pct
    )
    first_target = entry_price * (1.0 + direction_value * first_scaleout_pct / 100.0)
    final_target = entry_price * (1.0 + direction_value * policy.shock_absorption_final_target_pct / 100.0)
    first_fill = _first_target_fill(
        future,
        direction=direction_value,
        target=first_target,
    )
    final_fill = _first_target_fill(
        future,
        direction=direction_value,
        target=final_target,
    )
    structured_fill = _first_absorption_recovery_fill(
        future,
        direction=direction_value,
        not_before=(pd.Timestamp(first_fill["signal_time"]) if first_fill is not None else entry_time),
        policy=policy,
    )
    if structured_fill is not None:
        if _is_resilient_absorption_recovery(
            structured_fill,
            policy=policy,
        ):
            # 计算 final_fill，仅使用当前及历史数据。
            # 计算 final_fill，仅使用当前及历史数据。
            final_fill = _first_absorption_maturity_fill(
                future,
                direction=direction_value,
                not_before=pd.Timestamp(structured_fill["signal_time"]),
                policy=policy,
            )
        else:
            final_fill = structured_fill
        if final_fill is not None:
            final_target = float(final_fill["close"])
    if final_fill is None and len(future) >= policy.shock_absorption_maximum_holding_bars:
        final_fill = future.iloc[-1]
        final_target = float(final_fill["close"])
    scaleout_units = (
        policy.shock_absorption_entry_units * policy.shock_absorption_deep_scaleout_fraction
        if deep_scaleout
        else policy.shock_absorption_scaleout_units
    )
    records = [
        _action_record(
            chain_id=chain_id,
            symbol=symbol,
            action=f"OPEN_{direction}",
            operation_type="OPEN",
            signal_time=entry_time,
            signal_price=entry_price,
            position_fraction=1.0,
            mechanism=LIQUIDITY_SHOCK,
            score=score,
        ),
        _action_record(
            chain_id=chain_id,
            symbol=symbol,
            action=f"OPEN_{direction}",
            operation_type="ADD",
            signal_time=entry_time,
            signal_price=entry_price,
            position_fraction=policy.shock_absorption_entry_units - 1.0,
            mechanism=LIQUIDITY_SHOCK,
            score=score,
        ),
    ]
    if first_fill is not None and (
        final_fill is None or pd.Timestamp(first_fill["signal_time"]) <= pd.Timestamp(final_fill["signal_time"])
    ):
        records.append(
            _action_record(
                chain_id=chain_id,
                symbol=symbol,
                action=f"CLOSE_{direction}",
                operation_type="REDUCE",
                signal_time=pd.Timestamp(first_fill["signal_time"]),
                signal_price=first_target,
                position_fraction=scaleout_units,
                mechanism=LIQUIDITY_SHOCK,
                score=score,
                decision_time=entry_time,
            ),
        )
    reduced_units = scaleout_units if any(record["operation_type"] == "REDUCE" for record in records) else 0.0
    remaining_units = max(
        0.0,
        policy.shock_absorption_entry_units - reduced_units,
    )
    exit_time = pd.Timestamp(final_fill["signal_time"]) if final_fill is not None else None
    if final_fill is not None:
        records.append(
            _action_record(
                chain_id=chain_id,
                symbol=symbol,
                action=f"CLOSE_{direction}",
                operation_type="CLOSE",
                signal_time=exit_time,
                signal_price=final_target,
                position_fraction=remaining_units,
                mechanism=LIQUIDITY_SHOCK,
                score=score,
                decision_time=entry_time,
            ),
        )
    return records, exit_time


def _first_target_fill(
    future: pd.DataFrame,
    *,
    direction: float,
    target: float,
) -> pd.Series | None:
    """返回目标价在后续已完成柱上首次可成交的索引和价格。"""

    eligible = future.loc[future["high"].ge(target)] if direction > 0.0 else future.loc[future["low"].le(target)]
    return None if eligible.empty else eligible.iloc[0]


def _first_absorption_recovery_fill(
    future: pd.DataFrame,
    *,
    direction: float,
    not_before: pd.Timestamp,
    policy: UnifiedRegimeLifecyclePolicy,
) -> pd.Series | None:
    """寻找吸收活动首次满足恢复确认的后续柱。"""

    candidates = future.loc[future["signal_time"].ge(not_before)]
    for _, bar in candidates.iterrows():
        directional = _shock_directional(bar, direction)
        if (
            directional["recovery_return"] >= policy.shock_absorption_recovery_return_pct
            and directional["location"] >= policy.shock_absorption_recovery_location
            and directional["rsi"] >= policy.shock_absorption_recovery_rsi
            and directional["favorable_breadth20"] >= policy.shock_absorption_recovery_breadth
            and directional["adverse_breadth3"] >= policy.shock_absorption_counter_breadth
        ):
            return bar
    return None


def _is_resilient_absorption_recovery(
    bar: pd.Series,
    *,
    policy: UnifiedRegimeLifecyclePolicy,
) -> bool:
    """判断恢复过程中是否保持吸收结构和广度韧性。"""

    return bool(
        float(bar["atr14_ratio120"]) >= policy.shock_absorption_resilient_volatility_ratio,
    )


def _first_absorption_maturity_fill(
    future: pd.DataFrame,
    *,
    direction: float,
    not_before: pd.Timestamp,
    policy: UnifiedRegimeLifecyclePolicy,
) -> pd.Series | None:
    """寻找吸收恢复达到成熟退出条件的首个后续柱。"""

    candidates = future.loc[future["signal_time"].ge(not_before)]
    for _, bar in candidates.iterrows():
        directional = _shock_directional(bar, direction)
        if (
            directional["recovery_return"] >= policy.shock_absorption_recovery_return_pct
            and directional["location"] >= policy.shock_absorption_maturity_location
            and directional["rsi"] >= policy.shock_absorption_maturity_rsi
            # 说明：以下逻辑仅使用决策时点可获得的数据。
            # 说明：以下逻辑仅使用决策时点可获得的数据。
            and float(bar["adx14"]) >= policy.shock_absorption_maturity_minimum_adx
            and directional["favorable_breadth20"] >= policy.shock_absorption_recovery_breadth
            # 说明：以下逻辑仅使用决策时点可获得的数据。
            # 说明：以下逻辑仅使用决策时点可获得的数据。
            and directional["relative_return"] >= 0.0
            and float(bar["quote_volume_ratio_med30"]) >= policy.shock_absorption_maturity_quote_volume_ratio
        ):
            return bar
    return None


def _build_shock_confirmation_scores(
    market: pd.DataFrame,
    *,
    raw_scores: pd.DataFrame,
    factor_policy: CountertrendFactorPolicy,
    policy: UnifiedRegimeLifecyclePolicy,
) -> pd.DataFrame:
    """计算冲击严重度、吸收质量和恢复确认评分。"""

    mechanism = LIQUIDITY_SHOCK
    score_column = f"{mechanism}_score"
    threshold = factor_policy.threshold(mechanism)
    structure_multiplier = policy.shock_threshold_multiplier
    raw = raw_scores.set_index(["signal_time", "action"])[score_column]
    confirmed = raw_scores.loc[:, ["signal_time", "action"]].copy()
    confirmed[score_column] = 0.0
    output_index = {
        (pd.Timestamp(row["signal_time"]), str(row["action"])): index for index, row in confirmed.iterrows()
    }
    states = {
        action: {
            "phase": "idle",
            "start_index": -10_000,
            "countertrend": False,
        }
        for action in ("OPEN_LONG", "OPEN_SHORT")
    }
    active = dict.fromkeys(states, False)
    for bar_index, bar in market.iterrows():
        timestamp = pd.Timestamp(bar["signal_time"])
        for action, direction in (("OPEN_LONG", 1.0), ("OPEN_SHORT", -1.0)):
            score = float(raw.get((timestamp, action), 0.0))
            rising = False
            if score < threshold - factor_policy.release_hysteresis:
                active[action] = False
            elif score >= threshold:
                rising = not active[action]
                active[action] = True
            state = states[action]
            if rising and str(state["phase"]) == "idle":
                state["phase"] = "newly_armed"
                state["start_index"] = bar_index
            if (
                str(state["phase"]) != "idle"
                and bar_index - int(state["start_index"]) > policy.shock_confirmation_timeout_bars
            ):
                state["phase"] = "idle"
                continue
            directional = _shock_directional(bar, direction)
            newly_armed = str(state["phase"]) == "newly_armed"
            if newly_armed:
                countertrend = directional["adverse_long_return"] > 0.0
                state["countertrend"] = countertrend
                state["phase"] = "countertrend_armed" if countertrend else "armed"
            rejection = bool(
                directional["tiny_return"] >= (policy.shock_rejection_tiny_return_pct * structure_multiplier)
                and directional["favorable_breadth3"] >= policy.shock_rejection_breadth,
            )
            phase = str(state["phase"])
            immediate_climax = bool(
                float(bar["quote_volume_ratio_med30"])
                >= (policy.shock_immediate_quote_volume_ratio * structure_multiplier)
                and directional["relative_return"] >= (policy.shock_immediate_relative_edge_pct * structure_multiplier),
            )
            closed_bar_liquid_capitulation = _is_closed_bar_liquid_capitulation(
                bar,
                direction=direction,
                policy=policy,
            )
            high_quality_absorption = _is_high_quality_absorption(
                bar,
                direction=direction,
                policy=policy,
            )
            severe_capitulation = _is_severe_capitulation(
                bar,
                direction=direction,
                policy=policy,
            )
            orderly_washout = _is_orderly_washout_entry(
                bar,
                direction=direction,
                policy=policy,
            )
            blowoff_ladder = _is_blowoff_ladder_entry(
                bar,
                direction=direction,
                policy=policy,
            )
            blowoff_shape = _is_blowoff_ladder_shape(
                bar,
                direction=direction,
                policy=policy,
            )
            if phase in {"armed", "countertrend_armed"} and blowoff_shape and not blowoff_ladder:
                state["phase"] = "unexpanded_blowoff_blocked"
                continue
            if phase == "unexpanded_blowoff_blocked":
                if blowoff_ladder:
                    confirmed.loc[
                        output_index[(timestamp, action)],
                        score_column,
                    ] = max(score, threshold)
                    state["phase"] = "idle"
                continue
            deep_countertrend_exhaustion = bool(
                directional["adverse_mid_return"] >= policy.shock_strong_trend_defer_pct
                and directional["adverse_long_return"] >= policy.shock_blowoff_entry_adverse_long_minimum_pct
                and directional["location"] <= policy.shock_relief_location
                and directional["favorable_breadth20"] >= policy.shock_rejection_breadth
                and float(bar["adx14"]) >= policy.shock_deep_exhaustion_minimum_adx,
            )
            liquid_rejection = bool(
                rejection
                and directional["adverse_mid_return"]
                >= (policy.shock_confirmation_minimum_adverse_mid_return_pct * structure_multiplier)
                and float(bar["quote_volume_ratio_med30"])
                >= (policy.shock_confirmation_minimum_quote_volume_ratio * structure_multiplier),
            )
            moderate_countertrend_rejection = bool(
                liquid_rejection
                and directional["adverse_long_return"]
                >= (policy.shock_confirmation_minimum_adverse_long_return_pct * structure_multiplier)
                and directional["adverse_long_return"] < policy.shock_strong_trend_defer_pct,
            )
            structural_absorption_override = bool(
                score >= threshold * policy.shock_structural_override_score_ratio and high_quality_absorption,
            )
            # 计算 directionally_dislocated，仅使用当前及历史数据。
            # 计算 directionally_dislocated，仅使用当前及历史数据。
            directionally_dislocated = bool(
                directional["adverse_mid_return"]
                >= (policy.shock_minimum_directional_dislocation_pct * structure_multiplier),
            )
            if directionally_dislocated and (
                structural_absorption_override
                or (phase == "armed" and severe_capitulation)
                or (
                    phase == "countertrend_armed"
                    and (
                        immediate_climax
                        or closed_bar_liquid_capitulation
                        or high_quality_absorption
                        or orderly_washout
                        or blowoff_ladder
                        or deep_countertrend_exhaustion
                        or moderate_countertrend_rejection
                    )
                )
                or (phase == "armed" and high_quality_absorption)
            ):
                confirmed.loc[
                    output_index[(timestamp, action)],
                    score_column,
                ] = max(score, threshold)
                state["phase"] = "idle"
            elif phase in {"armed", "countertrend_armed"} and liquid_rejection:
                if phase == "countertrend_armed":
                    if directional["adverse_long_return"] >= policy.shock_countertrend_strong_long_return_pct:
                        state["phase"] = "rejected"
                elif (
                    policy.shock_trend_following_liquid_rejection_enabled
                    and directional["adverse_long_return"] <= 0.0
                    and directional["location"] <= _DIRECTIONAL_RANGE_MIDPOINT
                ):
                    confirmed.loc[
                        output_index[(timestamp, action)],
                        score_column,
                    ] = max(score, threshold)
                    state["phase"] = "idle"
                else:
                    state["phase"] = "rejected"
            elif phase == "rejected" and directional["location"] >= policy.shock_relief_location:
                state["phase"] = "relief"
            elif (
                policy.shock_relief_retest_confirmation_enabled
                and phase == "relief"
                and directional["location"] <= policy.shock_retest_location
                and rejection
                and directionally_dislocated
                and (
                    not bool(state["countertrend"])
                    or directional["adverse_long_return"]
                    >= (policy.shock_confirmation_minimum_adverse_long_return_pct * structure_multiplier)
                )
            ):
                confirmed.loc[
                    output_index[(timestamp, action)],
                    score_column,
                ] = max(score, threshold)
                state["phase"] = "idle"
    return confirmed


def _is_closed_bar_liquid_capitulation(
    bar: pd.Series,
    *,
    direction: float,
    policy: UnifiedRegimeLifecyclePolicy,
) -> bool:
    """识别无需未来反弹确认即可成交的闭合柱流动性投降。"""

    if not policy.shock_closed_bar_liquid_capitulation_enabled:
        return False
    if direction < 0.0 and not policy.shock_closed_bar_liquid_capitulation_short_enabled:
        return False
    directional = _shock_directional(bar, direction)
    multiplier = policy.shock_threshold_multiplier
    return bool(
        float(bar["quote_volume_ratio_med30"])
        >= (policy.shock_closed_bar_liquid_capitulation_quote_volume_ratio * multiplier)
        and directional["adverse_mid_return"]
        >= (policy.shock_closed_bar_liquid_capitulation_adverse_mid_return_pct * multiplier)
        and directional["adverse_long_return"]
        >= (policy.shock_closed_bar_liquid_capitulation_adverse_long_return_pct * multiplier)
        and directional["location"] <= policy.shock_closed_bar_liquid_capitulation_maximum_location
        and directional["rsi"] >= policy.shock_closed_bar_liquid_capitulation_minimum_rsi
        and directional["rsi"] <= policy.shock_closed_bar_liquid_capitulation_maximum_rsi,
    )


def _is_low_liquidity_retest_entry(
    bar: pd.Series,
    *,
    direction: float,
    policy: UnifiedRegimeLifecyclePolicy,
) -> bool:
    """识别强趋势末端的低流动性回测，仅允许探针风险预算。"""

    directional = _shock_directional(bar, direction)
    return bool(
        float(bar["quote_volume_ratio_med30"]) < policy.shock_confirmation_minimum_quote_volume_ratio
        and directional["adverse_mid_return"] >= policy.shock_strong_trend_defer_pct
        and directional["adverse_long_return"] >= policy.shock_strong_trend_defer_pct
        and directional["location"] <= policy.shock_retest_location
        and directional["favorable_breadth3"] >= policy.shock_rejection_breadth,
    )


def _shock_directional(bar: pd.Series, direction: float) -> dict[str, float]:
    """提取相对于冲击反转方向一致的市场分量。"""

    values = directional_market_components(bar, direction)
    return {
        "tiny_return": values["tiny_return"],
        "candle_return": values["candle_return"],
        "candle_close_location": values["candle_close_location"],
        "recovery_return": values["recovery_return"],
        "relative_return": values["relative_momentum"],
        "mid_return": values["mid_momentum"],
        "adverse_mid_return": values["adverse_mid_momentum"],
        "adverse_long_return": values["adverse_long_momentum"],
        "location": values["location"],
        "favorable_breadth3": values["favorable_breadth3"],
        "adverse_breadth3": values["adverse_breadth3"],
        "favorable_breadth20": values["favorable_breadth20"],
        "adverse_breadth20": values["adverse_breadth20"],
        "rsi": values["rsi"],
    }


def _failed_reclaim_proposals(
    market: pd.DataFrame,
    *,
    policy: PointRecoveryPolicy,
) -> list[dict[str, Any]]:
    """生成回收失败后再次确认的点位恢复候选。"""

    empty_inventory = pd.DataFrame(columns=["signal_time", "action"])
    actions = build_point_recovery_actions(
        market,
        empty_inventory,
        policy=policy,
    )
    actions = actions.loc[actions["chain_id"].astype(str).str.startswith("point_recovery:")].copy()
    if actions.empty:
        return []
    market_end = pd.Timestamp(market.iloc[-1]["signal_time"])
    proposals: list[dict[str, Any]] = []
    for source_chain_id, group in actions.groupby("chain_id", sort=False):
        ordered = group.sort_values(["signal_time", "signal_id"], kind="stable").copy()
        chain_id = "unified:failed_reclaim:" + str(source_chain_id).removeprefix(
            "point_recovery:",
        )
        ordered["source_signal_id"] = ordered["signal_id"].astype(str)
        ordered["chain_id"] = chain_id
        ordered["signal_id"] = [
            f"{chain_id}:{operation_type}:{pd.Timestamp(signal_time).isoformat()}"
            for operation_type, signal_time in zip(
                ordered["operation_type"],
                ordered["signal_time"],
                strict=True,
            )
        ]
        ordered["reason_family"] = FAILED_RECLAIM
        ordered["reason"] = "unified_market_regime;mechanism=failed_reclaim_recovery"
        closes = ordered.loc[ordered["operation_type"].eq("CLOSE"), "signal_time"]
        proposals.append(
            {
                "chain_id": chain_id,
                "mechanism": FAILED_RECLAIM,
                "score": float(
                    pd.to_numeric(ordered["score"], errors="coerce").fillna(0.0).iloc[0],
                ),
                "start": pd.Timestamp(ordered.iloc[0]["signal_time"]),
                "end": pd.Timestamp(closes.iloc[-1]) if not closes.empty else market_end,
                "actions": ordered,
            },
        )
    return proposals


def _relative_lag_reclaim_proposals(
    market: pd.DataFrame,
    *,
    policy: RelativeLagReclaimPolicy,
) -> list[dict[str, Any]]:
    """生成 ETH 相对落后后价格与市场广度同步回收的候选。"""

    actions = build_relative_lag_reclaim_actions(market, policy=policy)
    if actions.empty:
        return []
    market_end = pd.Timestamp(market.iloc[-1]["signal_time"])
    proposals: list[dict[str, Any]] = []
    for chain_id, group in actions.groupby("chain_id", sort=False):
        ordered = group.sort_values(
            ["signal_time", "signal_id"],
            kind="stable",
        ).copy()
        closes = ordered.loc[
            ordered["operation_type"].eq("CLOSE"),
            "signal_time",
        ]
        proposals.append(
            {
                "chain_id": str(chain_id),
                "mechanism": RELATIVE_LAG_RECLAIM,
                "score": float(ordered.iloc[0]["score"]),
                "start": pd.Timestamp(ordered.iloc[0]["signal_time"]),
                "end": (pd.Timestamp(closes.iloc[-1]) if not closes.empty else market_end),
                "actions": ordered,
            },
        )
    return proposals


def _rotation_proposals(
    market: pd.DataFrame,
    *,
    enabled_mechanisms: tuple[str, ...],
    policy: RegimeRotationPolicy,
) -> list[dict[str, Any]]:
    """生成趋势重入和状态反转轮动候选。"""

    enabled = set(enabled_mechanisms).intersection(ROTATION_MECHANISMS)
    actions = build_regime_rotation_actions(market, policy=policy)
    actions = actions.loc[actions["reason_family"].isin(enabled)].copy()
    if actions.empty:
        return []
    market_end = pd.Timestamp(market.iloc[-1]["signal_time"])
    proposals: list[dict[str, Any]] = []
    for source_chain_id, group in actions.groupby("chain_id", sort=False):
        ordered = group.sort_values(["signal_time", "signal_id"], kind="stable").copy()
        mechanism = str(ordered.iloc[0]["reason_family"])
        chain_suffix = str(source_chain_id).rsplit(":", maxsplit=1)[-1]
        chain_id = f"unified:{mechanism}:{chain_suffix}"
        ordered["source_signal_id"] = ordered["signal_id"].astype(str)
        ordered["chain_id"] = chain_id
        ordered["signal_id"] = [
            f"{chain_id}:{operation_type}:{pd.Timestamp(signal_time).isoformat()}"
            for operation_type, signal_time in zip(
                ordered["operation_type"],
                ordered["signal_time"],
                strict=True,
            )
        ]
        ordered["reason"] = f"unified_market_regime;mechanism={mechanism}"
        closes = ordered.loc[ordered["operation_type"].eq("CLOSE"), "signal_time"]
        proposals.append(
            {
                "chain_id": chain_id,
                "mechanism": mechanism,
                "score": float(ordered.iloc[0]["score"]),
                "start": pd.Timestamp(ordered.iloc[0]["signal_time"]),
                "end": pd.Timestamp(closes.iloc[-1]) if not closes.empty else market_end,
                "actions": ordered,
            },
        )
    return proposals


def _post_exit_proposals(
    market: pd.DataFrame,
    *,
    controller_actions: pd.DataFrame,
    policy: PostExitRecoveryPolicy,
) -> list[dict[str, Any]]:
    """生成控制器退出后的广度恢复和反向衰竭候选。"""

    actions = _concat_action_frames(
        [
            build_post_exit_recovery_actions(
                market,
                controller_actions,
                policy=policy,
            ),
            build_post_exit_rotation_actions(
                market,
                controller_actions,
                policy=policy,
            ),
        ],
    )
    if actions.empty:
        return []
    market_end = pd.Timestamp(market.iloc[-1]["signal_time"])
    proposals: list[dict[str, Any]] = []
    episodes: list[tuple[str, pd.DataFrame]] = []
    for source_chain_id, group in actions.groupby("chain_id", sort=False):
        episodes.extend((str(source_chain_id), episode) for episode in _split_lifecycle_episodes(group))
    episodes.sort(
        key=lambda item: (
            pd.Timestamp(item[1].iloc[0]["signal_time"]),
            item[0],
        ),
    )
    for episode_index, (_, ordered) in enumerate(episodes, start=1):
        chain_id = f"unified:{POST_EXIT_BREADTH_RECOVERY}:{episode_index:06d}"
        ordered["source_signal_id"] = ordered["signal_id"].astype(str)
        ordered["chain_id"] = chain_id
        ordered["signal_id"] = [
            (f"{chain_id}:{operation_type}:{pd.Timestamp(signal_time).isoformat()}")
            for operation_type, signal_time in zip(
                ordered["operation_type"],
                ordered["signal_time"],
                strict=True,
            )
        ]
        closes = ordered.loc[
            ordered["operation_type"].eq("CLOSE"),
            "signal_time",
        ]
        proposals.append(
            {
                "chain_id": chain_id,
                "mechanism": POST_EXIT_BREADTH_RECOVERY,
                "score": float(ordered.iloc[0]["score"]),
                "start": pd.Timestamp(ordered.iloc[0]["signal_time"]),
                "end": (pd.Timestamp(closes.iloc[-1]) if not closes.empty else market_end),
                "actions": ordered,
            },
        )
    return proposals


def _split_lifecycle_episodes(actions: pd.DataFrame) -> list[pd.DataFrame]:
    """按 chain_id 和完整库存闭合边界拆分候选活动。"""

    priority = {"REDUCE": 0, "CLOSE": 1, "OPEN": 2, "ADD": 3}
    ordered = actions.copy()
    ordered["_priority"] = ordered["operation_type"].astype(str).map(priority).fillna(4)
    ordered = ordered.sort_values(
        ["signal_time", "_priority", "signal_id"],
        kind="stable",
    ).drop(columns="_priority")
    episodes: list[pd.DataFrame] = []
    current: list[pd.Series] = []
    inventory = 0.0
    for _, action in ordered.iterrows():
        operation_type = str(action["operation_type"])
        units = float(action["position_fraction"])
        if inventory <= _INVENTORY_EPSILON:
            if operation_type != "OPEN":
                message = "post-exit lifecycle starts without OPEN"
                raise ValueError(message)
            current = []
        current.append(action)
        if operation_type in {"OPEN", "ADD"}:
            inventory += units
        else:
            inventory = max(0.0, inventory - units)
        if inventory <= _INVENTORY_EPSILON:
            episodes.append(pd.DataFrame(current).reset_index(drop=True))
            current = []
    if current:
        episodes.append(pd.DataFrame(current).reset_index(drop=True))
    return episodes


def _selection_signature(
    proposals: list[dict[str, Any]],
) -> tuple[tuple[str, str, str], ...]:
    """构造只依赖候选机制、方向和因果市场状态的排序签名。"""

    return tuple(
        (
            str(proposal["chain_id"]),
            pd.Timestamp(proposal["start"]).isoformat(),
            pd.Timestamp(proposal["end"]).isoformat(),
        )
        for proposal in proposals
    )


def _select_non_overlapping_proposals(
    proposals: list[dict[str, Any]],
    *,
    market: pd.DataFrame,
    cooldown_bars: int,
    opposite_direction_cooldown_bars: int = 0,
    post_exit_liquidity_shock_cooldown_bars: int = 0,
    takeover_edge: float,
) -> list[dict[str, Any]]:
    """按评分与库存约束选择时间上不重叠的候选活动。"""

    if not proposals:
        return []
    bar_times = pd.Index(market["signal_time"])
    bar_number = {pd.Timestamp(signal_time): index for index, signal_time in enumerate(bar_times)}
    ordered = sorted(
        proposals,
        key=lambda proposal: (
            pd.Timestamp(proposal["start"]),
            -float(proposal["score"]),
            str(proposal["mechanism"]),
        ),
    )
    selected: list[dict[str, Any]] = []
    active_until: pd.Timestamp | None = None
    last_exit_bar = -max(0, int(cooldown_bars)) - 1
    index = 0
    while index < len(ordered):
        start = pd.Timestamp(ordered[index]["start"])
        same_start: list[dict[str, Any]] = []
        while index < len(ordered) and pd.Timestamp(ordered[index]["start"]) == start:
            same_start.append(ordered[index])
            index += 1
        if active_until is not None and start < active_until:
            active = selected[-1]
            strongest = max(
                same_start,
                key=lambda proposal: (
                    float(proposal["score"]),
                    str(proposal["mechanism"]),
                ),
            )
            same_direction = _proposal_direction(active) == _proposal_direction(
                strongest,
            )
            stronger = float(strongest["score"]) >= (float(active["score"]) + float(takeover_edge))
            if same_direction and stronger:
                selected[-1] = _take_over_controller(
                    active,
                    strongest,
                    takeover_time=start,
                )
                active_until = pd.Timestamp(selected[-1]["end"])
                last_exit_bar = int(
                    bar_number.get(active_until, bar_number.get(start, -1)),
                )
            continue
        chosen = max(
            same_start,
            key=lambda proposal: (
                float(proposal["score"]),
                str(proposal["mechanism"]),
            ),
        )
        start_bar = int(bar_number.get(start, -1))
        required_cooldown = max(0, int(cooldown_bars))
        if (
            selected
            and str(selected[-1].get("mechanism") or "") == POST_EXIT_BREADTH_RECOVERY
            and str(chosen.get("mechanism") or "") == LIQUIDITY_SHOCK
        ):
            required_cooldown = max(
                required_cooldown,
                int(post_exit_liquidity_shock_cooldown_bars),
            )
        if (
            selected
            and _proposal_direction(selected[-1]) != _proposal_direction(chosen)
            and _proposal_requires_opposite_direction_cooldown(selected[-1])
        ):
            required_cooldown = max(
                required_cooldown,
                0,
                int(opposite_direction_cooldown_bars),
            )
        if start_bar < 0 or start_bar - last_exit_bar < required_cooldown:
            continue
        selected.append(chosen)
        active_until = pd.Timestamp(chosen["end"])
        last_exit_bar = int(bar_number.get(active_until, start_bar))
    return selected


def _proposal_requires_opposite_direction_cooldown(
    proposal: dict[str, Any],
) -> bool:
    """末端延伸退出后拒绝任意即时反向噪声。"""

    actions = proposal.get("actions")
    if not isinstance(actions, pd.DataFrame) or actions.empty:
        return False
    if "operation_type" not in actions:
        return False
    closes = actions.loc[actions["operation_type"].astype(str).eq("CLOSE")]
    if closes.empty or "reason" not in closes:
        return False
    return bool(
        closes["reason"].astype(str).str.contains("exit=terminal_extension", regex=False).any(),
    )


def _summarize_selected_campaign(
    actions: pd.DataFrame,
    *,
    round_trip_cost_bps: float,
) -> dict[str, object]:
    """汇总已选活动的时间、方向、评分和库存变化。"""

    actions = actions.sort_values(
        ["signal_time", "signal_id"],
        kind="stable",
    )
    opening = actions.loc[actions["action"].astype(str).str.startswith("OPEN_")]
    closing = actions.loc[actions["action"].astype(str).str.startswith("CLOSE_")]
    direction = 1.0 if str(opening.iloc[0]["action"]).endswith("_LONG") else -1.0
    entry_units = pd.to_numeric(
        opening["position_fraction"],
        errors="coerce",
    ).fillna(1.0)
    average_entry = float(
        np.average(
            pd.to_numeric(opening["signal_price"], errors="coerce"),
            weights=entry_units,
        ),
    )
    realized = 0.0
    close_units = 0.0
    for _, action in closing.iterrows():
        units = float(action["position_fraction"])
        realized += units * direction * (float(action["signal_price"]) / average_entry - 1.0) * 100.0
        close_units += units
    closed = bool(close_units + 1e-9 >= float(entry_units.sum()))
    cost_pct = float(round_trip_cost_bps) / 100.0 * float(entry_units.sum())
    return {
        "chain_id": str(actions.iloc[0]["chain_id"]),
        "mechanism": "->".join(
            dict.fromkeys(actions["reason_family"].astype(str)),
        ),
        "entry_time": pd.Timestamp(opening.iloc[0]["signal_time"]),
        "exit_time": (pd.Timestamp(closing.iloc[-1]["signal_time"]) if closed and not closing.empty else pd.NaT),
        "action": str(opening.iloc[0]["action"]),
        "fill_count": len(opening),
        "reduction_count": int(
            actions["operation_type"].astype(str).eq("REDUCE").sum(),
        ),
        "average_entry_price": average_entry,
        "gross_return_pct": realized if closed else np.nan,
        "net_return_pct": realized - cost_pct if closed else np.nan,
        "status": "CLOSED" if closed else "OPEN",
    }


def _proposal_direction(proposal: dict[str, Any]) -> str:
    """从候选动作链解析其主要交易方向。"""

    opening = proposal["actions"].loc[proposal["actions"]["action"].astype(str).str.startswith("OPEN_")]
    if opening.empty:
        message = "unified lifecycle proposal has no opening action"
        raise ValueError(message)
    return "LONG" if str(opening.iloc[0]["action"]).endswith("_LONG") else "SHORT"


def _take_over_controller(
    active: dict[str, Any],
    incoming: dict[str, Any],
    *,
    takeover_time: pd.Timestamp,
) -> dict[str, Any]:
    """在新活动接管前关闭控制器实际持有的冲突库存。"""

    chain_id = str(active["chain_id"])
    before = active["actions"].loc[active["actions"]["signal_time"].lt(takeover_time)].copy()
    inherited_units = _net_inventory_units(before)
    after = incoming["actions"].copy()
    incoming_end = pd.Timestamp(incoming["end"])
    residual_tail = (
        active["actions"]
        .loc[
            active["actions"]["signal_time"].gt(incoming_end)
            & active["actions"]["action"].astype(str).str.startswith("CLOSE_")
        ]
        .copy()
    )
    if not residual_tail.empty:
        residual_tail = residual_tail.iloc[[0]].copy()
        residual_tail["operation_type"] = "CLOSE"
    if inherited_units > 0.0 and not residual_tail.empty:
        # 只有旧控制器在新活动结束后仍有独立退出点时，才把新活动的最终
        # CLOSE 降级为 REDUCE。若旧控制器更早结束，新活动的 CLOSE 必须
        # 同时关闭继承库存，否则会残留孤儿仓位并与下一周期重叠。
        incoming_closes = after["operation_type"].astype(str).eq("CLOSE")
        after.loc[incoming_closes, "operation_type"] = "REDUCE"
    after["source_chain_id"] = after["chain_id"].astype(str)
    after["chain_id"] = chain_id
    after["signal_id"] = [
        f"{chain_id}:TAKEOVER:{index + 1:03d}:{pd.Timestamp(signal_time).isoformat()}"
        for index, signal_time in enumerate(after["signal_time"])
    ]
    actions = _concat_action_frames([before, after, residual_tail]).sort_values(
        ["signal_time", "signal_id"],
        kind="stable",
    )
    return {
        "chain_id": chain_id,
        "mechanism": (f"{active['mechanism']}->{incoming['mechanism']}"),
        "score": float(incoming["score"]),
        "start": pd.Timestamp(active["start"]),
        "end": max(
            pd.Timestamp(active["end"]),
            pd.Timestamp(incoming["end"]),
        ),
        "actions": actions.reset_index(drop=True),
    }


def _net_inventory_units(actions: pd.DataFrame) -> float:
    """按动作语义计算一组记录的净库存单位。"""

    units = pd.to_numeric(
        actions.get("position_fraction"),
        errors="coerce",
    ).fillna(1.0)
    opening = actions["action"].astype(str).str.startswith("OPEN_")
    return max(0.0, float(units.where(opening, -units).sum()))


def _close_actual_controller_inventory(actions: pd.DataFrame) -> pd.DataFrame:
    """为控制器尚未闭合的真实库存生成接管平仓动作。"""

    frame = actions.copy()
    priority = {"REDUCE": 0, "CLOSE": 1, "OPEN": 2, "ADD": 3}
    frame["_operation_priority"] = frame["operation_type"].astype(str).map(priority).fillna(4)
    frame = frame.sort_values(
        ["signal_time", "_operation_priority", "signal_id"],
        kind="stable",
    ).reset_index(drop=True)
    inventory: dict[tuple[str, str], float] = {}
    for index, action in frame.iterrows():
        name = str(action["action"])
        direction = "LONG" if name.endswith("_LONG") else "SHORT"
        key = (str(action["symbol"]), direction)
        before = float(inventory.get(key, 0.0))
        units = float(action["position_fraction"])
        if name.startswith("OPEN_"):
            inventory[key] = before + units
            continue
        if str(action["operation_type"]) == "CLOSE":
            units = before
            frame.loc[index, "position_fraction"] = units
        inventory[key] = max(0.0, before - units)
    return frame.drop(columns="_operation_priority")


def _concat_action_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """合并动作表并保持统一列、时间顺序和稳定优先级。"""

    available = [frame.copy() for frame in frames if not frame.empty]
    if not available:
        return _empty_actions()
    columns = list(
        dict.fromkeys(column for frame in available for column in frame.columns),
    )
    cleaned = [frame.dropna(axis="columns", how="all") for frame in available]
    combined = pd.concat(cleaned, ignore_index=True, sort=False)
    for column in columns:
        if column not in combined:
            combined[column] = pd.NA
    return combined.reindex(columns=columns)


def _action_record(
    *,
    chain_id: str,
    symbol: str,
    action: str,
    operation_type: str,
    signal_time: pd.Timestamp,
    signal_price: float,
    position_fraction: float,
    mechanism: str,
    score: float,
    decision_time: pd.Timestamp | None = None,
    reason_detail: str = "",
    campaign_risk_profile: str = "",
) -> dict[str, object]:
    """生成统一生命周期模块的标准动作记录。"""

    detail = f";{reason_detail}" if reason_detail else ""
    return {
        "signal_id": f"{chain_id}:{operation_type}:{signal_time.isoformat()}",
        "chain_id": chain_id,
        "symbol": symbol,
        "action": action,
        "operation_type": operation_type,
        "signal_time": signal_time,
        "signal_price": float(signal_price),
        "position_fraction": float(position_fraction),
        "decision_time": decision_time or signal_time,
        "reason_family": mechanism,
        "campaign_risk_profile": campaign_risk_profile,
        "reason": (f"unified_market_regime;mechanism={mechanism}{detail}"),
        "score": float(score),
    }


def _prepare_market(market_panel: pd.DataFrame) -> pd.DataFrame:
    """校验统一生命周期输入并按 UTC 信号时间排序。"""

    required = {"signal_time", "symbol", "close"}
    missing = sorted(required.difference(market_panel.columns))
    if missing:
        raise ValueError("unified lifecycle panel missing columns: " + ", ".join(missing))
    market = market_panel.copy().sort_values("signal_time", kind="stable").reset_index(drop=True)
    market["signal_time"] = pd.to_datetime(market["signal_time"], utc=True)
    symbols = market["symbol"].dropna().astype(str).unique()
    if len(symbols) != 1:
        message = "unified lifecycle requires exactly one symbol"
        raise ValueError(message)
    if market["signal_time"].duplicated().any():
        message = "unified lifecycle panel contains duplicate signal_time"
        raise ValueError(message)
    return market


def _validate_policy(policy: UnifiedRegimeLifecyclePolicy) -> None:
    """_validate_policy：校验输入结构、时间顺序与数值边界。"""

    unknown = sorted(set(policy.enabled_mechanisms).difference(SUPPORTED_MECHANISMS))
    if unknown:
        raise ValueError("unknown unified lifecycle mechanisms: " + ", ".join(unknown))
    if policy.post_exit_cooldown_bars < 0:
        message = "unified lifecycle cooldown cannot be negative"
        raise ValueError(message)
    if policy.opposite_direction_post_exit_cooldown_bars < 0:
        message = "unified lifecycle opposite-direction cooldown cannot be negative"
        raise ValueError(message)
    if policy.post_exit_liquidity_shock_cooldown_bars < 0:
        message = "unified lifecycle post-exit shock cooldown cannot be negative"
        raise ValueError(message)
    if policy.controller_takeover_edge < 0.0:
        message = "unified lifecycle takeover edge cannot be negative"
        raise ValueError(message)
    consensus = tuple(float(multiplier) for multiplier in policy.threshold_consensus_multipliers)
    if (
        not consensus
        or tuple(sorted(consensus)) != consensus
        or any(
            multiplier < MINIMUM_THRESHOLD_MULTIPLIER or multiplier > MAXIMUM_THRESHOLD_MULTIPLIER
            for multiplier in consensus
        )
        or not any(abs(multiplier - 1.0) <= _INVENTORY_EPSILON for multiplier in consensus)
    ):
        message = "unified lifecycle consensus multipliers must be sorted in [0.8, 1.2] and include 1.0"
        raise ValueError(message)
    if not 1 <= policy.threshold_consensus_required_votes <= len(consensus):
        message = "unified lifecycle consensus votes must fit the multiplier grid"
        raise ValueError(message)
    if policy.threshold_consensus_time_tolerance_bars < 0:
        message = "unified lifecycle consensus tolerance cannot be negative"
        raise ValueError(message)
    if not (
        MINIMUM_THRESHOLD_MULTIPLIER <= policy.threshold_consensus_dispersion_multiplier <= MAXIMUM_THRESHOLD_MULTIPLIER
    ):
        message = "unified lifecycle consensus dispersion must be in [0.8, 1.2]"
        raise ValueError(message)
    if policy.shock_confirmation_timeout_bars <= 0:
        message = "unified lifecycle confirmation timeout must be positive"
        raise ValueError(message)
    if not (0 < policy.shock_orderly_washout_confirmation_minimum_bars <= policy.shock_confirmation_timeout_bars):
        message = "unified lifecycle orderly confirmation window is invalid"
        raise ValueError(message)
    if not (
        0.0
        <= policy.shock_orderly_washout_confirmation_location_minimum
        < policy.shock_orderly_washout_confirmation_location_maximum
        <= 1.0
    ):
        message = "unified lifecycle orderly confirmation location is invalid"
        raise ValueError(message)
    if not (
        0.0
        <= policy.shock_orderly_washout_confirmation_rsi_minimum
        < policy.shock_orderly_washout_confirmation_rsi_maximum
        <= _MAXIMUM_RSI
    ):
        message = "unified lifecycle orderly confirmation RSI is invalid"
        raise ValueError(message)
    if not (
        0.0
        < policy.shock_orderly_washout_immediate_scaleout_fraction
        * len(policy.shock_orderly_washout_scaleout_locations)
        < 1.0
        and 0.0 < policy.shock_orderly_washout_scaleout_fraction < 1.0
        and 0.0 < policy.shock_orderly_washout_reentry_units < policy.shock_orderly_washout_scaleout_fraction
    ):
        message = "unified lifecycle orderly washout inventory is invalid"
        raise ValueError(message)
    if not (0.0 <= policy.shock_orderly_washout_exit_reversal_adverse_breadth3 <= 1.0):
        message = "unified lifecycle orderly exit breadth is invalid"
        raise ValueError(message)
    if not 0.0 < policy.shock_score_floor <= 1.0:
        message = "unified lifecycle shock score floor must be in (0, 1]"
        raise ValueError(message)
    if policy.shock_minimum_directional_dislocation_pct <= 0.0:
        message = "unified lifecycle directional dislocation must be positive"
        raise ValueError(message)
    if policy.shock_absorption_maximum_holding_bars <= 0:
        message = "unified lifecycle absorption holding time must be positive"
        raise ValueError(message)
    if policy.shock_absorption_resilient_volatility_ratio <= 0.0:
        message = "unified lifecycle absorption volatility ratio must be positive"
        raise ValueError(message)
    if not 0.0 <= policy.shock_absorption_maturity_location <= 1.0:
        message = "unified lifecycle absorption maturity location is invalid"
        raise ValueError(message)
    if not 0.0 <= policy.shock_absorption_maturity_rsi <= _MAXIMUM_RSI:
        message = "unified lifecycle absorption maturity RSI is invalid"
        raise ValueError(message)
    if policy.shock_absorption_maturity_minimum_adx <= 0.0:
        message = "unified lifecycle absorption maturity ADX must be positive"
        raise ValueError(message)
    if policy.shock_absorption_maturity_quote_volume_ratio <= 0.0:
        message = "unified lifecycle absorption maturity volume must be positive"
        raise ValueError(message)
    if not (0.0 < policy.shock_absorption_first_scaleout_pct < policy.shock_absorption_final_target_pct):
        message = "unified lifecycle absorption targets are inconsistent"
        raise ValueError(message)
    if not (
        policy.shock_absorption_entry_units > 1.0
        and 0.0 < policy.shock_absorption_scaleout_units < policy.shock_absorption_entry_units
    ):
        message = "unified lifecycle absorption inventory is inconsistent"
        raise ValueError(message)
    if not (
        0.0 <= policy.shock_absorption_deep_scaleout_rsi <= _MAXIMUM_RSI
        and policy.shock_absorption_deep_scaleout_minimum_adx > 0.0
        and 0.0 < policy.shock_absorption_deep_scaleout_fraction < 1.0
        and policy.shock_absorption_deep_scaleout_lookback_bars > 0
        and 0.0 < policy.shock_absorption_deep_scaleout_first_target_pct <= policy.shock_absorption_first_scaleout_pct
    ):
        message = "unified lifecycle deep absorption scaleout is invalid"
        raise ValueError(message)
    if not (
        policy.shock_closed_bar_liquid_capitulation_quote_volume_ratio > 0.0
        and policy.shock_closed_bar_liquid_capitulation_adverse_mid_return_pct > 0.0
        and policy.shock_closed_bar_liquid_capitulation_adverse_long_return_pct > 0.0
        and 0.0 <= policy.shock_closed_bar_liquid_capitulation_maximum_location <= 1.0
        and 0.0
        <= policy.shock_closed_bar_liquid_capitulation_minimum_rsi
        <= policy.shock_closed_bar_liquid_capitulation_maximum_rsi
        <= _MAXIMUM_RSI
    ):
        message = "unified lifecycle closed-bar capitulation is invalid"
        raise ValueError(message)
    blowoff_entries = (
        policy.shock_blowoff_probe_units
        + policy.shock_blowoff_primary_add_units
        + policy.shock_blowoff_secondary_add_units
    )
    blowoff_reductions = policy.shock_blowoff_reduction_fraction + policy.shock_blowoff_followthrough_reduction_fraction
    if not (
        policy.shock_blowoff_probe_units > 0.0
        and policy.shock_blowoff_primary_add_units > 0.0
        and policy.shock_blowoff_secondary_add_units > 0.0
        and 0.0 < blowoff_reductions < blowoff_entries
    ):
        message = "unified lifecycle blowoff inventory is inconsistent"
        raise ValueError(message)
    if policy.shock_blowoff_probe_take_profit_pct < 0.0:
        message = "unified lifecycle blowoff probe target cannot be negative"
        raise ValueError(message)
    if (
        policy.shock_blowoff_entry_minimum_atr_ratio < 0.0
        or policy.shock_blowoff_entry_minimum_quote_volume_ratio < 0.0
    ):
        message = "unified lifecycle blowoff expansion floors cannot be negative"
        raise ValueError(message)
    if not 0.0 <= policy.shock_rejection_breadth <= 1.0:
        message = "unified lifecycle rejection breadth must be in [0, 1]"
        raise ValueError(message)
    if not (0.0 <= policy.shock_retest_location < policy.shock_relief_location <= 1.0):
        message = "unified lifecycle shock locations are inconsistent"
        raise ValueError(message)


def _empty_actions() -> pd.DataFrame:
    """返回具有标准动作列的空表。"""

    return pd.DataFrame(
        columns=[
            "signal_id",
            "chain_id",
            "symbol",
            "action",
            "operation_type",
            "signal_time",
            "signal_price",
            "position_fraction",
            "decision_time",
            "reason_family",
            "reason",
            "score",
        ],
    )


def _empty_cycles() -> pd.DataFrame:
    """返回具有标准活动汇总列的空表。"""

    return pd.DataFrame(
        columns=[
            "chain_id",
            "mechanism",
            "entry_time",
            "exit_time",
            "action",
            "fill_count",
            "reduction_count",
            "average_entry_price",
            "gross_return_pct",
            "net_return_pct",
            "status",
        ],
    )


__all__ = [
    "FAILED_RECLAIM",
    "LIQUIDITY_SHOCK",
    "RELATIVE_LAG_RECLAIM",
    "SUPPORTED_MECHANISMS",
    "UnifiedRegimeLifecyclePolicy",
    "evaluate_unified_regime_lifecycle",
]
