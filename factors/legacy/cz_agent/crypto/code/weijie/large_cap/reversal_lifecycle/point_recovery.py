# ruff: noqa: C901, PLR0911, PLR0912, PLR2004
"""从价格冲击、回收确认和库存状态构建点位恢复活动。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import pandas as pd

from .components import (
    directional_market_components,
)
from .components import (
    linear_ramp as _ramp,
)
from .components import (
    logarithmic_ramp as _log_ramp,
)

REQUIRED_POINT_COLUMNS = frozenset(
    {
        "signal_time",
        "symbol",
        "high",
        "low",
        "close",
        "ret3",
        "ret12",
        "ret20",
        "ret60",
        "tf1h_ret12",
        "donchian60_pos",
        "rsi14",
        "adx14",
        "atr14_pct",
        "atr14_ratio120",
        "quote_volume_ratio_med30",
        "market_down_breadth20",
        "market_up_breadth20",
        "market_down_breadth3",
        "market_up_breadth3",
        "countertrend_data_valid",
    },
)
_DEFERRED_REVERSAL_MINIMUM_COMPONENTS = 2
_DEFERRED_REVERSAL_GUARD_RATIO = 0.80
_SETUP_ADX_STATE_RANGE = (18.0, 35.0)
_SETUP_ATR_EXPANSION_RANGE = (0.75, 1.50)
_SETUP_MID_DISPLACEMENT_RANGE = (0.60, 1.80)
_SETUP_LONG_DISPLACEMENT_RANGE = (0.50, 4.00)
_SECOND_RECLAIM_CURRENT_TREND_WEIGHT = 0.65
_DEFERRED_BLOWOFF_QUOTE_VOLUME_RANGE = (2.0, 20.0)
_DEFERRED_BLOWOFF_ATR_RANGE = (1.0, 2.0)
_BLOWOFF_SHORT_MOMENTUM_RANGE = (2.0, 4.0)
_BLOWOFF_MID_MOMENTUM_RANGE = (2.0, 4.0)
_BLOWOFF_LOCATION_RANGE = (0.85, 0.98)
_BLOWOFF_RSI_RANGE = (65.0, 80.0)
_BLOWOFF_ADX_RANGE = (25.0, 50.0)
_CONSOLIDATION_ATR_ACTIVITY_RANGE = (1.0, 1.8)
_CONSOLIDATION_SHORT_MOMENTUM_RANGE = (0.0, 0.40)
_CONSOLIDATION_COUNTER_MOVE_RANGE = (0.0, 0.20)


@dataclass(frozen=True, slots=True)
class PointRecoveryPolicy:
    """点位恢复、延迟确认、加仓和分级止盈的策略参数。"""

    setup_adverse_mid_return_pct: float = -0.60
    setup_maximum_directional_location: float = 0.40
    setup_minimum_trend_strength: float = 0.48
    reclaim_location: float = 0.50
    failed_reclaim_location: float = 0.35
    minimum_breadth: float = 0.66
    reclaim_rsi_minimum: float = 45.0
    reclaim_rsi_maximum: float = 60.0
    probe_timeout_bars: int = 18
    second_reclaim_timeout_bars: int = 24
    strong_trend_defer_pct: float = 1.0
    deferred_entry_timeout_bars: int = 288
    deferred_exhaustion_mid_return_pct: float = 3.0
    deferred_extreme_location: float = 0.20
    deferred_reversal_breadth: float = 0.66
    deferred_blowoff_mid_return_pct: float = 1.50
    deferred_blowoff_location: float = 0.10
    deferred_blowoff_rsi: float = 20.0
    deferred_blowoff_minimum_liquidity_score: float = 0.75
    deferred_add_adverse_return_pct: float = 1.0
    deferred_add_quote_volume_ratio: float = 1.5
    deferred_recovery_target_pct: float = 1.50
    deferred_minimum_holding_bars: int = 6
    deferred_maximum_holding_bars: int = 288

    consolidation_minimum_bars: int = 72
    consolidation_mid_return_minimum_pct: float = -0.20
    consolidation_mid_return_maximum_pct: float = 0.30
    consolidation_location_minimum: float = 0.55
    consolidation_location_maximum: float = 0.72
    consolidation_rsi_minimum: float = 48.0
    consolidation_rsi_maximum: float = 55.0
    consolidation_maximum_adx: float = 20.0
    consolidation_minimum_quality: float = 0.60

    extension_mid_return_pct: float = 0.80
    extension_location: float = 0.85
    extension_rsi: float = 62.0
    extension_minimum_adx: float = 18.0
    extension_retrace_location: float = 0.55
    extension_retrace_rsi: float = 45.0

    deep_pullback_location: float = 0.20
    deep_pullback_short_return_pct: float = -0.15
    deep_pullback_mid_return_pct: float = -0.50
    deep_pullback_rsi: float = 42.0
    minimum_action_gap_bars: int = 3

    recovery_stage_one_short_return_pct: float = 0.25
    recovery_stage_one_rsi: float = 54.0
    recovery_stage_two_return_pct: float = 0.85
    recovery_stage_two_location: float = 0.80
    recovery_stage_two_rsi: float = 68.0
    recovery_stage_two_short_return_pct: float = 1.25

    blowoff_minimum_strength: float = 0.65
    maximum_campaign_bars: int = 576

    profit_rebound_mid_return_pct: float = 0.70
    profit_rebound_location: float = 0.75
    profit_rebound_rsi: float = 58.0
    profit_rebound_counter_move_pct: float = -0.10
    profit_exhaustion_short_return_pct: float = 0.70
    profit_exhaustion_mid_return_pct: float = 0.80
    profit_exhaustion_location: float = 0.88
    profit_exhaustion_rsi: float = 65.0
    profit_exhaustion_quote_volume_ratio: float = 2.0
    profit_limit_atr_multiple: float = 1.0
    profit_limit_expiry_bars: int = 6
    profit_exit_cooldown_bars: int = 24

    def with_threshold_multiplier(
        self,
        multiplier: float,
    ) -> PointRecoveryPolicy:
        """统一扰动连续阈值，用于检验点位恢复机制的稳定性。"""

        if not 0.8 <= multiplier <= 1.2:
            message = "point recovery threshold multiplier must be in [0.8, 1.2]"
            raise ValueError(message)
        fields = (
            "setup_minimum_trend_strength",
            "strong_trend_defer_pct",
            "deferred_exhaustion_mid_return_pct",
            "deferred_blowoff_mid_return_pct",
            "deferred_blowoff_minimum_liquidity_score",
            "deferred_add_adverse_return_pct",
            "deferred_add_quote_volume_ratio",
            "deferred_recovery_target_pct",
            "consolidation_minimum_quality",
            "extension_mid_return_pct",
            "extension_minimum_adx",
            "deep_pullback_short_return_pct",
            "deep_pullback_mid_return_pct",
            "recovery_stage_one_short_return_pct",
            "recovery_stage_two_return_pct",
            "recovery_stage_two_short_return_pct",
            "blowoff_minimum_strength",
            "profit_rebound_mid_return_pct",
            "profit_exhaustion_short_return_pct",
            "profit_exhaustion_mid_return_pct",
            "profit_exhaustion_quote_volume_ratio",
        )
        changes = {
            name: float(getattr(self, name)) * multiplier
            for name in fields
        }
        return replace(self, **changes)


def build_point_recovery_actions(
    market_panel: pd.DataFrame,
    base_actions: pd.DataFrame,
    *,
    policy: PointRecoveryPolicy | None = None,
) -> pd.DataFrame:
    """结合市场面板和基础动作，构建因果的点位恢复动作链。"""

    cfg = policy or PointRecoveryPolicy()
    market = _prepare_market(market_panel)
    core = _prepare_base_actions(base_actions)
    inventory_events = _inventory_events(core)
    records = [
        *_build_symmetric_campaign_actions(
            market,
            inventory_events=inventory_events,
            policy=cfg,
        ),
        *_build_symmetric_profit_exits(
            market,
            inventory_events=inventory_events,
            policy=cfg,
        ),
    ]
    columns = (
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
    )
    return (
        pd.DataFrame.from_records(records, columns=columns)
        .sort_values(["signal_time", "signal_id"], kind="stable")
        .reset_index(drop=True)
    )


def _build_symmetric_campaign_actions(
    market: pd.DataFrame,
    *,
    inventory_events: dict[pd.Timestamp, list[tuple[float, float]]],
    policy: PointRecoveryPolicy,
) -> list[dict[str, object]]:
    """对多空方向使用同一套方向化规则构建恢复活动。"""

    setup = {
        direction: {
            "phase": "idle",
            "probe_index": -10_000,
            "failure_index": -10_000,
            "deferred_index": -10_000,
            "setup_trend_strength": 0.0,
        }
        for direction in (1.0, -1.0)
    }
    core_inventory = {1.0: 0.0, -1.0: 0.0}
    position: dict[str, Any] | None = None
    records: list[dict[str, object]] = []
    for bar_index, bar in market.iterrows():
        timestamp = pd.Timestamp(bar["signal_time"])
        _apply_inventory_events(
            core_inventory,
            inventory_events.get(timestamp, []),
        )
        if position is not None:
            emitted, closed = _advance_campaign(
                market,
                bar_index=bar_index,
                position=position,
                policy=policy,
            )
            records.extend(emitted)
            if closed:
                position = None
            continue
        if sum(core_inventory.values()) > 0.0:
            continue
        candidates: list[tuple[float, dict[str, Any]]] = []
        for direction in (1.0, -1.0):
            candidate = _advance_reclaim_setup(
                market,
                bar_index=bar_index,
                direction=direction,
                state=setup[direction],
                policy=policy,
            )
            if candidate is not None:
                candidates.append(candidate)
        if not candidates:
            continue
        candidates.sort(key=lambda item: (-item[0], -item[1]["direction"]))
        _, position = candidates[0]
        records.append(
            _campaign_action_row(
                position,
                timestamp=timestamp,
                operation_type="OPEN",
                price=float(bar["close"]),
                reason_family="failed_reclaim_second_reclaim",
            ),
        )
    return records


def _advance_reclaim_setup(
    market: pd.DataFrame,
    *,
    bar_index: int,
    direction: float,
    state: dict[str, Any],
    policy: PointRecoveryPolicy,
) -> tuple[float, dict[str, Any]] | None:
    """推进冲击后的回收设置，等待价格和趋势确认。"""

    bar = market.iloc[bar_index]
    directional = _directional(bar, direction)
    previous_location = (
        _directional(market.iloc[bar_index - 1], direction)["location"]
        if bar_index > 0
        else directional["location"]
    )
    reclaimed = bool(
        directional["location"] >= policy.reclaim_location
        and previous_location < policy.reclaim_location,
    )
    phase = str(state["phase"])
    if phase == "trend_exhaustion":
        if (
            bar_index - int(state["deferred_index"])
            > policy.deferred_entry_timeout_bars
        ):
            state["phase"] = "idle"
            return None
        reversal_confirmation = _deferred_reversal_confirmation(
            directional,
            policy=policy,
        )
        blowoff_confirmation = bool(
            directional["mid_return"]
            <= -policy.deferred_blowoff_mid_return_pct
            and directional["location"] <= policy.deferred_blowoff_location
            and directional["rsi"] <= policy.deferred_blowoff_rsi
            and _deferred_blowoff_liquidity_strength(bar)
            >= policy.deferred_blowoff_minimum_liquidity_score,
        )
        if not (reversal_confirmation or blowoff_confirmation):
            return None
        state["phase"] = "idle"
        severity = (
            abs(directional["mid_return"])
            + abs(directional["long_return"])
            + float(bar["adx14"]) / 100.0
        )
        return severity, _new_campaign_position(
            bar,
            bar_index=bar_index,
            direction=direction,
            entry_mode="deferred_trend_exhaustion",
        )
    if phase == "idle":
        trend_strength = _setup_trend_strength(bar, directional)
        if (
            directional["mid_return"] <= policy.setup_adverse_mid_return_pct
            and directional["location"] <= policy.setup_maximum_directional_location
            and trend_strength >= policy.setup_minimum_trend_strength
        ):
            state["phase"] = "drawdown"
            state["setup_trend_strength"] = trend_strength
        return None
    if phase == "drawdown":
        if (
            reclaimed
            and directional["favorable_breadth3"] >= policy.minimum_breadth
            and policy.reclaim_rsi_minimum
            <= directional["rsi"]
            <= policy.reclaim_rsi_maximum
        ):
            state["phase"] = "probe"
            state["probe_index"] = bar_index
        return None
    if phase == "probe":
        if bar_index - int(state["probe_index"]) > policy.probe_timeout_bars:
            state["phase"] = "idle"
        elif (
            directional["location"] <= policy.failed_reclaim_location
            and directional["adverse_breadth3"] >= policy.minimum_breadth
        ):
            state["phase"] = "failed"
            state["failure_index"] = bar_index
        return None
    if bar_index - int(state["failure_index"]) > policy.second_reclaim_timeout_bars:
        state["phase"] = "idle"
        return None
    if not (
        reclaimed
        and directional["favorable_breadth3"] >= policy.minimum_breadth
        and policy.reclaim_rsi_minimum
        <= directional["rsi"]
        <= policy.reclaim_rsi_maximum
        and _second_reclaim_trend_confirmation(
            bar,
            state=state,
            policy=policy,
        )
        and policy.setup_adverse_mid_return_pct * 2.5
        <= directional["mid_return"]
        <= 0.30
    ):
        return None
    severity = abs(directional["mid_return"]) + float(bar["adx14"]) / 100.0
    if directional["long_return"] <= -policy.strong_trend_defer_pct:
        state["phase"] = "trend_exhaustion"
        state["deferred_index"] = bar_index
        return None
    state["phase"] = "idle"
    return severity, _new_campaign_position(
        bar,
        bar_index=bar_index,
        direction=direction,
        entry_mode="immediate_failed_reclaim",
    )


def _deferred_reversal_confirmation(
    directional: dict[str, float],
    *,
    policy: PointRecoveryPolicy,
) -> bool:
    """判断延迟一段时间后的反转是否获得足够确认。"""

    adverse_mid = -directional["mid_return"]
    location = directional["location"]
    breadth = directional["favorable_breadth20"]
    broad_guard = bool(
        adverse_mid
        >= (
            policy.deferred_exhaustion_mid_return_pct
            * _DEFERRED_REVERSAL_GUARD_RATIO
        )
        and location <= policy.deferred_extreme_location
        and breadth
        >= (
            policy.deferred_reversal_breadth
            * _DEFERRED_REVERSAL_GUARD_RATIO
        ),
    )
    strong_components = sum(
        (
            adverse_mid >= policy.deferred_exhaustion_mid_return_pct,
            location <= policy.deferred_extreme_location,
            breadth >= policy.deferred_reversal_breadth,
        ),
    )
    return bool(
        broad_guard
        and strong_components >= _DEFERRED_REVERSAL_MINIMUM_COMPONENTS,
    )


def _setup_trend_strength(
    bar: pd.Series,
    directional: dict[str, float],
) -> float:
    """计算设置形成时的多周期方向趋势强度。"""

    adverse_mid = -directional["mid_return"]
    adverse_long = -directional["long_return"]
    trend_maturity = _ramp(
        float(bar["adx14"]),
        *_SETUP_ADX_STATE_RANGE,
    )
    market_activity = max(
        _ramp(
            float(bar["atr14_ratio120"]),
            *_SETUP_ATR_EXPANSION_RANGE,
        ),
        _ramp(
            adverse_long,
            *_SETUP_LONG_DISPLACEMENT_RANGE,
        ),
    )
    directional_displacement = max(
        _ramp(
            adverse_mid,
            *_SETUP_MID_DISPLACEMENT_RANGE,
        ),
        _ramp(
            adverse_long,
            *_SETUP_LONG_DISPLACEMENT_RANGE,
        ),
    )
    joint_regime = trend_maturity * (
        0.50 * market_activity
        + 0.50 * directional_displacement
    )
    return float(joint_regime**0.5)


def _second_reclaim_trend_confirmation(
    bar: pd.Series,
    *,
    state: dict[str, Any],
    policy: PointRecoveryPolicy,
) -> bool:
    """验证第二次回收是否延续而非重复短暂反弹。"""

    current_trend_maturity = _ramp(
        float(bar["adx14"]),
        *_SETUP_ADX_STATE_RANGE,
    )
    setup_trend_strength = float(state["setup_trend_strength"])
    confirmation_strength = (
        _SECOND_RECLAIM_CURRENT_TREND_WEIGHT * current_trend_maturity
        + (1.0 - _SECOND_RECLAIM_CURRENT_TREND_WEIGHT)
        * setup_trend_strength
    )
    return bool(
        confirmation_strength >= policy.setup_minimum_trend_strength,
    )


def _deferred_blowoff_liquidity_strength(bar: pd.Series) -> float:
    """衡量延迟衰竭阶段的波动与流动性冲击强度。"""

    quote_volume_expansion = _log_ramp(
        float(bar["quote_volume_ratio_med30"]),
        *_DEFERRED_BLOWOFF_QUOTE_VOLUME_RANGE,
    )
    volatility_expansion = _ramp(
        float(bar["atr14_ratio120"]),
        *_DEFERRED_BLOWOFF_ATR_RANGE,
    )
    return float(
        0.70 * quote_volume_expansion
        + 0.30 * volatility_expansion,
    )


def _campaign_blowoff_strength(
    bar: pd.Series,
    directional: dict[str, float],
) -> float:
    """计算持仓活动内末端衰竭的综合强度。"""

    short_momentum = _ramp(
        directional["short_return"],
        *_BLOWOFF_SHORT_MOMENTUM_RANGE,
    )
    mid_momentum = _ramp(
        directional["mid_return"],
        *_BLOWOFF_MID_MOMENTUM_RANGE,
    )
    joint_momentum = (short_momentum * mid_momentum) ** 0.5
    extension_state = (
        _ramp(
            directional["location"],
            *_BLOWOFF_LOCATION_RANGE,
        )
        + _ramp(
            directional["rsi"],
            *_BLOWOFF_RSI_RANGE,
        )
        + _ramp(
            float(bar["adx14"]),
            *_BLOWOFF_ADX_RANGE,
        )
    ) / 3.0
    return float(
        0.60 * joint_momentum
        + 0.40 * extension_state,
    )


def _consolidation_quality(
    bar: pd.Series,
    directional: dict[str, float],
) -> float:
    """评估冲击后整理阶段的收敛、位置和广度质量。"""

    volatility_activity = _ramp(
        float(bar["atr14_ratio120"]),
        *_CONSOLIDATION_ATR_ACTIVITY_RANGE,
    )
    constructive_momentum = _ramp(
        directional["short_return"],
        *_CONSOLIDATION_SHORT_MOMENTUM_RANGE,
    )
    counter_move_stability = _ramp(
        -directional["tiny_return"],
        *_CONSOLIDATION_COUNTER_MOVE_RANGE,
    )
    return float(
        0.50 * volatility_activity
        + 0.30 * constructive_momentum
        + 0.20 * counter_move_stability,
    )


def _new_campaign_position(
    bar: pd.Series,
    *,
    bar_index: int,
    direction: float,
    entry_mode: str,
) -> dict[str, Any]:
    """创建点位恢复活动的初始库存和状态记录。"""

    price = float(bar["close"])
    return {
        "chain_id": f"point_recovery:{pd.Timestamp(bar['signal_time']).isoformat()}",
        "symbol": str(bar["symbol"]),
        "direction": direction,
        "entry_mode": entry_mode,
        "entry_index": bar_index,
        "last_action_index": bar_index,
        "inventory_units": 1.0,
        "fill_prices": [price],
        "reload_done": False,
        "first_reduce": False,
        "extension_probe": False,
        "extension_failed": False,
        "recovery_stage": 0,
        "deep_locked": False,
    }


def _advance_campaign(
    market: pd.DataFrame,
    *,
    bar_index: int,
    position: dict[str, Any],
    policy: PointRecoveryPolicy,
) -> tuple[list[dict[str, object]], bool]:
    """按当前柱推进恢复活动的确认、加仓、减仓和退出。"""

    if str(position["entry_mode"]) == "deferred_trend_exhaustion":
        return _advance_deferred_exhaustion_campaign(
            market,
            bar_index=bar_index,
            position=position,
            policy=policy,
        )
    bar = market.iloc[bar_index]
    directional = _directional(bar, float(position["direction"]))
    timestamp = pd.Timestamp(bar["signal_time"])
    gap = bar_index - int(position["last_action_index"])
    records: list[dict[str, object]] = []
    if bar_index - int(position["entry_index"]) >= policy.maximum_campaign_bars:
        return [
            _campaign_action_row(
                position,
                timestamp=timestamp,
                operation_type="CLOSE",
                price=float(bar["close"]),
                reason_family="maximum_campaign_holding",
            ),
        ], True
    consolidation = bool(
        directional["tiny_return"] <= 0.0
        and 0.0 <= directional["short_return"] <= 0.50
        and policy.consolidation_mid_return_minimum_pct
        <= directional["mid_return"]
        <= policy.consolidation_mid_return_maximum_pct
        and policy.consolidation_location_minimum
        <= directional["location"]
        <= policy.consolidation_location_maximum
        and policy.consolidation_rsi_minimum
        <= directional["rsi"]
        <= policy.consolidation_rsi_maximum
        and float(bar["adx14"]) <= policy.consolidation_maximum_adx
        and directional["favorable_breadth20"] >= 0.75
        and _consolidation_quality(bar, directional)
        >= policy.consolidation_minimum_quality,
    )
    if (
        not bool(position["reload_done"])
        and consolidation
        and gap >= policy.consolidation_minimum_bars
    ):
        records.append(
            _campaign_action_row(
                position,
                timestamp=timestamp,
                operation_type="ADD",
                price=float(bar["close"]),
                reason_family="post_drawdown_consolidation_reload",
            ),
        )
        position["reload_done"] = True
        position["last_action_index"] = bar_index
        return records, False
    first_extension = bool(
        directional["mid_return"] >= policy.extension_mid_return_pct
        and directional["location"] >= policy.extension_location
        and directional["rsi"] >= policy.extension_rsi
        and float(bar["adx14"]) >= policy.extension_minimum_adx
        and directional["favorable_breadth20"] >= 0.75,
    )
    if bool(position["reload_done"]) and not bool(position["first_reduce"]):
        if first_extension and not bool(position["extension_probe"]):
            position["extension_probe"] = True
        elif (
            bool(position["extension_probe"])
            and (
                directional["location"] <= policy.extension_retrace_location
                or directional["rsi"] <= policy.extension_retrace_rsi
                or directional["adverse_breadth20"] >= 0.75
            )
        ):
            position["extension_failed"] = True
        elif first_extension and bool(position["extension_failed"]) and gap >= 6:
            records.append(
                _campaign_action_row(
                    position,
                    timestamp=timestamp,
                    operation_type="REDUCE",
                    price=float(bar["close"]),
                    reason_family="second_extension_first_scaleout",
                ),
            )
            position["first_reduce"] = True
            position["last_action_index"] = bar_index
            position["recovery_stage"] = 1
            position["deep_locked"] = False
            return records, False
    deep_pullback = bool(
        directional["location"] <= policy.deep_pullback_location
        and directional["short_return"] <= policy.deep_pullback_short_return_pct
        and directional["mid_return"] <= policy.deep_pullback_mid_return_pct
        and directional["rsi"] <= policy.deep_pullback_rsi
        and directional["adverse_breadth20"] >= 0.75,
    )
    if (
        bool(position["first_reduce"])
        and deep_pullback
        and not bool(position["deep_locked"])
        and gap >= 6
    ):
        records.append(
            _campaign_action_row(
                position,
                timestamp=timestamp,
                operation_type="ADD",
                price=float(bar["close"]),
                reason_family="deep_pullback_inventory_add",
            ),
        )
        position["last_action_index"] = bar_index
        position["recovery_stage"] = 0
        position["deep_locked"] = True
        return records, False
    blowoff = bool(
        _campaign_blowoff_strength(bar, directional)
        >= policy.blowoff_minimum_strength,
    )
    if blowoff and gap >= policy.minimum_action_gap_bars:
        return [
            _campaign_action_row(
                position,
                timestamp=timestamp,
                operation_type="CLOSE",
                price=float(bar["close"]),
                reason_family="trend_blowoff_full_exit",
            ),
        ], True
    recovery_one = bool(
        (
            directional["tiny_return"]
            >= policy.recovery_stage_one_short_return_pct
            and directional["favorable_breadth3"] >= policy.minimum_breadth
            and directional["rsi"] >= policy.recovery_stage_one_rsi
        )
        or (
            directional["short_return"] >= 0.15
            and directional["location"] >= 0.75
            and directional["rsi"] >= 60.0
        ),
    )
    recovery_two = bool(
        (
            directional["recovery_return"]
            >= policy.recovery_stage_two_return_pct
            and directional["location"] >= policy.recovery_stage_two_location
            and directional["rsi"] >= policy.recovery_stage_two_rsi
        )
        or (
            directional["short_return"]
            >= policy.recovery_stage_two_short_return_pct
            and directional["rsi"] >= policy.extension_rsi
        ),
    )
    stage = int(position["recovery_stage"])
    if bool(position["first_reduce"]) and stage < 1 and recovery_one and gap >= 3:
        records.append(
            _campaign_action_row(
                position,
                timestamp=timestamp,
                operation_type="REDUCE",
                price=float(bar["close"]),
                reason_family="recovery_stage_1",
            ),
        )
        position["last_action_index"] = bar_index
        position["recovery_stage"] = 1
        return records, False
    if bool(position["first_reduce"]) and stage < 2 and recovery_two and gap >= 3:
        records.append(
            _campaign_action_row(
                position,
                timestamp=timestamp,
                operation_type="REDUCE",
                price=float(bar["close"]),
                reason_family="recovery_stage_2",
            ),
        )
        position["last_action_index"] = bar_index
        position["recovery_stage"] = 2
        position["deep_locked"] = False
        return records, False
    return records, False


def _advance_deferred_exhaustion_campaign(
    market: pd.DataFrame,
    *,
    bar_index: int,
    position: dict[str, Any],
    policy: PointRecoveryPolicy,
) -> tuple[list[dict[str, object]], bool]:
    """推进延迟衰竭活动，并在确认失败或成熟时退出。"""

    bar = market.iloc[bar_index]
    direction = float(position["direction"])
    directional = _directional(bar, direction)
    timestamp = pd.Timestamp(bar["signal_time"])
    bars_held = bar_index - int(position["entry_index"])
    gap = bar_index - int(position["last_action_index"])
    average_entry = sum(position["fill_prices"]) / len(position["fill_prices"])
    favorable_move_pct = direction * (
        float(bar["close"]) / average_entry - 1.0
    ) * 100.0
    adverse_move_pct = -direction * (
        float(bar["close"]) / float(position["fill_prices"][-1]) - 1.0
    ) * 100.0
    if (
        len(position["fill_prices"]) == 1
        and gap >= policy.minimum_action_gap_bars
        and adverse_move_pct >= policy.deferred_add_adverse_return_pct
        and directional["recovery_return"]
        <= -policy.deferred_add_adverse_return_pct
        and float(bar["quote_volume_ratio_med30"])
        >= policy.deferred_add_quote_volume_ratio
    ):
        position["fill_prices"].append(float(bar["close"]))
        position["last_action_index"] = bar_index
        return [
            _campaign_action_row(
                position,
                timestamp=timestamp,
                operation_type="ADD",
                price=float(bar["close"]),
                reason_family="trend_exhaustion_adverse_extension_add",
            ),
        ], False
    if (
        bars_held >= policy.deferred_minimum_holding_bars
        and favorable_move_pct >= policy.deferred_recovery_target_pct
    ):
        target = average_entry * (
            1.0 + direction * policy.deferred_recovery_target_pct / 100.0
        )
        return [
            _campaign_action_row(
                position,
                timestamp=timestamp,
                operation_type="CLOSE",
                price=target,
                reason_family="trend_exhaustion_recovery_exit",
            ),
        ], True
    if bars_held >= policy.deferred_maximum_holding_bars:
        return [
            _campaign_action_row(
                position,
                timestamp=timestamp,
                operation_type="CLOSE",
                price=float(bar["close"]),
                reason_family="trend_exhaustion_maximum_holding",
            ),
        ], True
    return [], False


def _build_symmetric_profit_exits(
    market: pd.DataFrame,
    *,
    inventory_events: dict[pd.Timestamp, list[tuple[float, float]]],
    policy: PointRecoveryPolicy,
) -> list[dict[str, object]]:
    """按方向对称地构建分级止盈和剩余库存退出。"""

    inventory = {1.0: 0.0, -1.0: 0.0}
    active = {1.0: False, -1.0: False}
    last_exit = {1.0: -10_000, -1.0: -10_000}
    records: list[dict[str, object]] = []
    for bar_index, bar in market.iterrows():
        timestamp = pd.Timestamp(bar["signal_time"])
        _apply_inventory_events(inventory, inventory_events.get(timestamp, []))
        for direction in (1.0, -1.0):
            directional = _directional(bar, direction)
            rebound = bool(
                directional["mid_return"] >= policy.profit_rebound_mid_return_pct
                and directional["location"] >= policy.profit_rebound_location
                and directional["rsi"] >= policy.profit_rebound_rsi
                and directional["favorable_breadth20"] >= 0.75
                and directional["adverse_breadth3"] >= policy.minimum_breadth
                and directional["tiny_return"]
                <= policy.profit_rebound_counter_move_pct,
            )
            exhaustion = bool(
                directional["short_return"]
                >= policy.profit_exhaustion_short_return_pct
                and directional["mid_return"]
                >= policy.profit_exhaustion_mid_return_pct
                and directional["location"] >= policy.profit_exhaustion_location
                and directional["rsi"] >= policy.profit_exhaustion_rsi
                and directional["favorable_breadth20"] >= 0.90
                and float(bar["quote_volume_ratio_med30"])
                >= policy.profit_exhaustion_quote_volume_ratio,
            )
            condition = rebound or exhaustion
            if not condition:
                active[direction] = False
                continue
            if (
                active[direction]
                or inventory[direction] <= 0.0
                or bar_index - last_exit[direction]
                < policy.profit_exit_cooldown_bars
            ):
                continue
            active[direction] = True
            fill_time = timestamp
            fill_price = float(bar["close"])
            reason = "counter_move_profit_rebound"
            if exhaustion:
                fill_time, fill_price = _causal_profit_limit_fill(
                    market,
                    bar_index=bar_index,
                    direction=direction,
                    policy=policy,
                )
                reason = "directional_profit_exhaustion_limit"
            position = {
                "chain_id": f"point_profit:{timestamp.isoformat()}:{direction:+.0f}",
                "symbol": str(bar["symbol"]),
                "direction": direction,
            }
            units = min(0.25, inventory[direction])
            records.append(
                _action_row(
                    position,
                    timestamp=fill_time,
                    decision_time=timestamp,
                    operation_type="REDUCE",
                    price=fill_price,
                    reason_family=reason,
                    position_fraction=units,
                ),
            )
            # 每次减仓消耗一个库存单位；实际下单比例由
            # position_fraction 单独记录，避免把库存计数与仓位比例混用。
            inventory[direction] = max(0.0, inventory[direction] - 1.0)
            last_exit[direction] = bar_index
    return records


def _causal_profit_limit_fill(
    market: pd.DataFrame,
    *,
    bar_index: int,
    direction: float,
    policy: PointRecoveryPolicy,
) -> tuple[pd.Timestamp, float]:
    """只用后续柱的开高低价判断止盈限价首次可成交位置。"""

    bar = market.iloc[bar_index]
    target = float(bar["close"]) * (
        1.0
        + direction
        * policy.profit_limit_atr_multiple
        * float(bar["atr14_pct"])
        / 100.0
    )
    future = market.iloc[
        bar_index + 1 : bar_index + 1 + policy.profit_limit_expiry_bars
    ]
    fills = (
        future.loc[future["high"].ge(target)]
        if direction > 0.0
        else future.loc[future["low"].le(target)]
    )
    if fills.empty:
        return pd.Timestamp(bar["signal_time"]), float(bar["close"])
    fill = fills.iloc[0]
    return pd.Timestamp(fill["signal_time"]), target


def _directional(bar: pd.Series, direction: float) -> dict[str, float]:
    """提取相对于目标方向一致的收益、位置、RSI 和广度分量。"""

    values = directional_market_components(bar, direction)
    return {
        "tiny_return": values["tiny_return"],
        "recovery_return": values["recovery_return"],
        "short_return": values["short_momentum"],
        "mid_return": values["mid_momentum"],
        "long_return": values["long_momentum"],
        "location": values["location"],
        "rsi": values["rsi"],
        "favorable_breadth20": values["favorable_breadth20"],
        "adverse_breadth20": values["adverse_breadth20"],
        "favorable_breadth3": values["favorable_breadth3"],
        "adverse_breadth3": values["adverse_breadth3"],
    }


def _inventory_events(
    actions: pd.DataFrame,
) -> dict[pd.Timestamp, list[tuple[float, float]]]:
    """把同一时点的控制器动作整理为库存变更事件。"""

    events: dict[pd.Timestamp, list[tuple[float, float]]] = {}
    for _, action in actions.iterrows():
        name = str(action["action"])
        direction = 1.0 if name.endswith("_LONG") else -1.0
        raw_units = pd.to_numeric(action.get("position_fraction"), errors="coerce")
        units = (
            float(raw_units)
            if pd.notna(raw_units) and float(raw_units) > 0.0
            else 1.0
        )
        delta = units if name.startswith("OPEN_") else -units
        event = (direction, delta)
        events.setdefault(pd.Timestamp(action["signal_time"]), []).append(event)
    return events


def _apply_inventory_events(
    inventory: dict[float, float],
    events: list[tuple[float, float]],
) -> None:
    """按 OPEN、ADD、REDUCE、CLOSE 语义更新方向库存。"""

    for direction, delta in events:
        inventory[direction] = max(0.0, inventory[direction] + delta)


def _prepare_market(panel: pd.DataFrame) -> pd.DataFrame:
    """校验点位恢复所需字段并按 UTC 信号时间稳定排序。"""

    missing = sorted(REQUIRED_POINT_COLUMNS.difference(panel.columns))
    if missing:
        message = "point recovery panel missing columns: " + ", ".join(missing)
        raise ValueError(message)
    market = panel.copy().sort_values("signal_time", kind="stable").reset_index(drop=True)
    market["signal_time"] = pd.to_datetime(market["signal_time"], utc=True)
    market = market.loc[market["countertrend_data_valid"].astype(bool)].reset_index(drop=True)
    symbols = market["symbol"].dropna().astype(str).unique()
    if len(symbols) > 1:
        message = "point recovery requires one symbol per lifecycle stream"
        raise ValueError(message)
    return market


def _prepare_base_actions(actions: pd.DataFrame) -> pd.DataFrame:
    """规范基础动作的时间、数量和同柱执行优先级。"""

    required = {"signal_time", "action"}
    missing = sorted(required.difference(actions.columns))
    if missing:
        message = "point recovery base actions missing columns: " + ", ".join(missing)
        raise ValueError(message)
    frame = actions.copy()
    times = pd.to_datetime(frame["signal_time"], errors="coerce")
    if times.dt.tz is None:
        times = times.dt.tz_localize("Asia/Shanghai")
    frame["signal_time"] = times.dt.tz_convert("UTC")
    return frame.sort_values(["signal_time", "action"], kind="stable").reset_index(drop=True)


def _action_row(
    position: dict[str, Any],
    *,
    timestamp: pd.Timestamp,
    operation_type: str,
    price: float,
    reason_family: str,
    decision_time: pd.Timestamp | None = None,
    position_fraction: float = 1.0,
) -> dict[str, object]:
    """生成普通点位恢复动作的标准记录。"""

    direction = "LONG" if float(position["direction"]) > 0.0 else "SHORT"
    prefix = "OPEN" if operation_type in {"OPEN", "ADD"} else "CLOSE"
    return {
        "signal_id": f"{position['chain_id']}:{operation_type}:{timestamp.isoformat()}",
        "chain_id": str(position["chain_id"]),
        "symbol": str(position["symbol"]),
        "action": f"{prefix}_{direction}",
        "operation_type": operation_type,
        "signal_time": timestamp,
        "signal_price": float(price),
        "position_fraction": float(position_fraction),
        "decision_time": decision_time,
        "reason_family": reason_family,
        "reason": f"market_point_recovery;family={reason_family}",
        "score": 1.0,
    }


def _campaign_action_row(
    position: dict[str, Any],
    *,
    timestamp: pd.Timestamp,
    operation_type: str,
    price: float,
    reason_family: str,
) -> dict[str, object]:
    """生成带活动阶段、机制和库存腿信息的标准动作记录。"""

    inventory = float(position["inventory_units"])
    if operation_type == "OPEN":
        units = inventory
    elif operation_type == "ADD":
        units = 1.0
        position["inventory_units"] = inventory + units
    elif operation_type == "REDUCE":
        units = min(0.25, inventory)
        position["inventory_units"] = max(0.0, inventory - units)
    elif operation_type == "CLOSE":
        units = inventory
        position["inventory_units"] = 0.0
    else:
        message = f"unsupported point recovery operation type: {operation_type}"
        raise ValueError(message)
    if units <= 0.0:
        message = (
            "point recovery attempted a non-positive inventory action: "
            f"{position['chain_id']}:{operation_type}"
        )
        raise ValueError(message)
    return _action_row(
        position,
        timestamp=timestamp,
        operation_type=operation_type,
        price=price,
        reason_family=reason_family,
        position_fraction=units,
    )


__all__ = [
    "REQUIRED_POINT_COLUMNS",
    "PointRecoveryPolicy",
    "build_point_recovery_actions",
]
