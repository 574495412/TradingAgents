# ruff: noqa: C901, PLR0911, PLR0912, PLR2004
"""在控制器退出后，依据市场广度恢复或方向轮动构建自主交易动作。"""

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

POST_EXIT_BREADTH_RECOVERY = "post_exit_breadth_recovery"
NORMALIZED_RUNNER_EXIT_STATE = "exit_state=normalized_runner"
MINIMUM_THRESHOLD_MULTIPLIER = 0.8
MAXIMUM_THRESHOLD_MULTIPLIER = 1.2
_EXHAUSTION_RECOVERY_RANGE = (0.40, 1.50)
_EXHAUSTION_ATR_ACTIVITY_RANGE = (0.50, 1.20)
_EXHAUSTION_QUOTE_VOLUME_RANGE = (0.75, 3.0)
_BREADTH_RECOVERY_ADVERSE_LONG_CENTER = 1.125
_BREADTH_RECOVERY_ADVERSE_LONG_RADIUS = 1.0


@dataclass(frozen=True, slots=True)
class PostExitRecoveryPolicy:
    """退出后恢复、反向衰竭和趋势回踩机制的阈值与持仓边界。"""

    maximum_exit_age_bars: int = 144
    maximum_holding_bars: int = 288
    exhaustion_rotation_exit_age_bars: int = 3
    exhaustion_rotation_requires_confirmation: bool = False
    exhaustion_rotation_minimum_strength: float = 0.45
    exhaustion_rotation_location: float = 0.30
    exhaustion_rotation_rsi: float = 35.0
    exhaustion_rotation_adverse_breadth3: float = 0.75
    exhaustion_rotation_quote_volume_ratio: float = 1.0
    exhaustion_rotation_maximum_quote_volume_ratio: float = 3.0
    exhaustion_rotation_maximum_atr_ratio: float = 1.20
    exhaustion_rotation_maximum_adx: float = 30.0
    exhaustion_rotation_confirmation_bars: int = 12
    exhaustion_rotation_confirmation_maximum_adverse_tiny_return_pct: float = (
        0.05
    )
    exhaustion_rotation_confirmation_minimum_favorable_breadth3: float = 0.66
    exhaustion_confirmation_return_pct: float = 1.50
    exhaustion_confirmation_location: float = 0.75
    exhaustion_confirmation_location_maximum: float = 0.90
    exhaustion_confirmation_rsi: float = 60.0
    exhaustion_confirmation_quote_volume_ratio: float = 2.0
    exhaustion_exit_mid_momentum_pct: float = 1.50
    exhaustion_exit_long_momentum_pct: float = 1.50
    exhaustion_exit_location: float = 0.85
    exhaustion_exit_rsi: float = 60.0
    continuation_exit_age_bars: int = 24
    continuation_long_momentum_pct: float = 1.50
    continuation_mid_momentum_pct: float = 1.50
    continuation_pullback_return_pct: float = -0.20
    continuation_location_minimum: float = 0.55
    continuation_location_maximum: float = 0.80
    continuation_rsi: float = 55.0
    continuation_adverse_breadth3: float = 0.66
    continuation_atr_ratio: float = 1.20
    continuation_add_recovery_return_pct: float = -0.50
    continuation_add_pullback_return_pct: float = -0.30
    continuation_add_rsi: float = 45.0
    continuation_add_adverse_breadth3: float = 0.75
    continuation_stall_long_momentum_pct: float = 1.0
    continuation_stall_mid_momentum_pct: float = 0.50
    continuation_stall_location: float = 0.80
    continuation_stall_rsi: float = 60.0
    continuation_stall_breadth_minimum: float = 0.75
    continuation_stall_breadth_maximum: float = 0.90
    continuation_stall_atr_ratio_minimum: float = 0.60
    continuation_stall_atr_ratio_maximum: float = 0.90
    continuation_stall_quote_volume_ratio: float = 1.50
    adverse_long_momentum_minimum_pct: float = 0.75
    adverse_long_momentum_maximum_pct: float = 1.50
    maximum_absolute_mid_momentum_pct: float = 0.30
    location_minimum: float = 0.55
    location_maximum: float = 0.80
    rsi_minimum: float = 50.0
    rsi_maximum: float = 68.0
    minimum_breadth: float = 0.90
    minimum_quote_volume_ratio: float = 0.75
    maximum_quote_volume_ratio: float = 2.0
    minimum_breadth_recovery_strength: float = 0.68
    exit_recovery_return_pct: float = 1.0
    exit_mid_momentum_pct: float = 1.0
    exit_location: float = 0.90
    exit_rsi: float = 75.0
    exit_quote_volume_ratio: float = 2.0

    def with_threshold_multiplier(
        self,
        multiplier: float,
    ) -> PostExitRecoveryPolicy:
        """按统一比例扰动连续阈值，用于检验策略对参数变化的稳定性。"""

        if not (
            MINIMUM_THRESHOLD_MULTIPLIER
            <= multiplier
            <= MAXIMUM_THRESHOLD_MULTIPLIER
        ):
            message = "post-exit threshold multiplier must be in [0.8, 1.2]"
            raise ValueError(message)
        fields = (
            "exhaustion_rotation_minimum_strength",
            "exhaustion_rotation_quote_volume_ratio",
            "exhaustion_rotation_maximum_quote_volume_ratio",
            "exhaustion_rotation_maximum_atr_ratio",
            "exhaustion_rotation_maximum_adx",
            "exhaustion_rotation_confirmation_maximum_adverse_tiny_return_pct",
            "exhaustion_rotation_confirmation_minimum_favorable_breadth3",
            "exhaustion_confirmation_return_pct",
            "exhaustion_confirmation_quote_volume_ratio",
            "exhaustion_exit_mid_momentum_pct",
            "exhaustion_exit_long_momentum_pct",
            "continuation_long_momentum_pct",
            "continuation_mid_momentum_pct",
            "continuation_pullback_return_pct",
            "continuation_atr_ratio",
            "continuation_add_recovery_return_pct",
            "continuation_add_pullback_return_pct",
            "continuation_stall_long_momentum_pct",
            "continuation_stall_mid_momentum_pct",
            "continuation_stall_atr_ratio_minimum",
            "continuation_stall_atr_ratio_maximum",
            "continuation_stall_quote_volume_ratio",
            "adverse_long_momentum_minimum_pct",
            "adverse_long_momentum_maximum_pct",
            "maximum_absolute_mid_momentum_pct",
            "minimum_quote_volume_ratio",
            "maximum_quote_volume_ratio",
            "minimum_breadth_recovery_strength",
            "exit_recovery_return_pct",
            "exit_mid_momentum_pct",
            "exit_quote_volume_ratio",
        )
        return replace(
            self,
            **{
                name: float(getattr(self, name)) * multiplier
                for name in fields
            },
        )


def build_post_exit_recovery_actions(
    market_panel: pd.DataFrame,
    controller_actions: pd.DataFrame,
    *,
    policy: PostExitRecoveryPolicy | None = None,
) -> pd.DataFrame:
    """在控制器平掉反方向仓位后，扫描广度恢复并生成新方向动作。"""

    cfg = policy or PostExitRecoveryPolicy()
    market = _prepare_market(market_panel)
    controller = _prepare_controller_events(controller_actions)
    controller_events = {  # noqa: C416
        timestamp: group
        for timestamp, group in controller.groupby("signal_time", sort=False)
    }

    last_exit_index: dict[float, int | None] = {1.0: None, -1.0: None}
    controller_inventory = {1.0: 0.0, -1.0: 0.0}
    position: dict[str, Any] | None = None
    records: list[dict[str, object]] = []
    sequence = 0
    for bar_index, bar in market.iterrows():
        timestamp = pd.Timestamp(bar["signal_time"])
        for _, event in controller_events.get(
            timestamp,
            pd.DataFrame(),
        ).iterrows():
            direction = (
                1.0 if str(event["action"]).endswith("LONG") else -1.0
            )
            units = float(event["position_fraction"])
            operation_type = str(event["operation_type"])
            if operation_type == "CLOSE":
                controller_inventory[direction] = 0.0
                if NORMALIZED_RUNNER_EXIT_STATE not in str(
                    event.get("reason", ""),
                ):
                    last_exit_index[direction] = bar_index
            elif operation_type == "REDUCE":
                controller_inventory[direction] = max(
                    0.0,
                    controller_inventory[direction] - units,
                )
            else:
                controller_inventory[direction] += units
        if position is not None:
            direction = float(position["direction"])
            held_bars = bar_index - int(position["entry_index"])
            if _is_exit(bar, direction=direction, policy=cfg) or (
                held_bars >= cfg.maximum_holding_bars
            ):
                records.append(
                    _action_row(
                        bar,
                        chain_id=str(position["chain_id"]),
                        direction=direction,
                        operation_type="CLOSE",
                        score=float(position["score"]),
                        mode="breadth_recovery",
                    ),
                )
                position = None
            continue

        if sum(controller_inventory.values()) > 0.0:
            continue
        candidates: list[tuple[float, float]] = []
        for direction in (1.0, -1.0):
            opposite_exit = last_exit_index[-direction]
            if opposite_exit is None:
                continue
            age = bar_index - opposite_exit
            if not 0 <= age <= cfg.maximum_exit_age_bars:
                continue
            active, score = _entry_state(bar, direction=direction, policy=cfg)
            if active:
                candidates.append((score, direction))
        if not candidates:
            continue
        score, direction = max(candidates, key=lambda item: (item[0], item[1]))
        sequence += 1
        chain_id = f"post_exit:{sequence:06d}"
        records.append(
            _action_row(
                bar,
                chain_id=chain_id,
                direction=direction,
                operation_type="OPEN",
                score=score,
                mode="breadth_recovery",
            ),
        )
        last_exit_index[-direction] = None
        position = {
            "chain_id": chain_id,
            "direction": direction,
            "entry_index": bar_index,
            "score": score,
        }
    return pd.DataFrame.from_records(
        records,
        columns=(
            "signal_id",
            "chain_id",
            "symbol",
            "action",
            "operation_type",
            "signal_time",
            "signal_price",
            "position_fraction",
            "reason_family",
            "reason",
            "score",
        ),
    )


def build_post_exit_rotation_actions(
    market_panel: pd.DataFrame,
    controller_actions: pd.DataFrame,
    *,
    policy: PostExitRecoveryPolicy | None = None,
) -> pd.DataFrame:
    """从控制器退出点开始，构建反向衰竭轮动及后续趋势回踩动作。"""

    cfg = policy or PostExitRecoveryPolicy()
    market = _prepare_market(market_panel)
    controller = _prepare_controller_events(controller_actions)
    normalized_runner = controller.get(
        "reason",
        pd.Series("", index=controller.index, dtype=str),
    ).astype(str).str.contains(
        NORMALIZED_RUNNER_EXIT_STATE,
        regex=False,
    )
    closes = controller.loc[
        controller["operation_type"].astype(str).eq("CLOSE")
        & ~normalized_runner
    ].copy()
    records: list[dict[str, object]] = []
    sequence = 0
    for _, event in closes.iterrows():
        matches = market.index[
            market["signal_time"].eq(pd.Timestamp(event["signal_time"]))
        ]
        if matches.empty:
            continue
        exit_index = int(matches[0])
        exited_direction = (
            1.0 if str(event["action"]).endswith("LONG") else -1.0
        )
        direction = -exited_direction
        entry_index = _first_entry_index(
            market,
            start_index=exit_index + 1,
            end_index=exit_index + cfg.exhaustion_rotation_exit_age_bars,
            direction=direction,
            policy=cfg,
            mode="exhaustion_rotation",
        )
        if entry_index is None:
            continue
        sequence += 1
        chain_id = f"post_exit_rotation:{sequence:06d}"
        emitted, close_index = _build_rotation_campaign(
            market,
            entry_index=entry_index,
            chain_id=chain_id,
            direction=direction,
            score=0.49,
            mode="exhaustion_rotation",
            policy=cfg,
        )
        records.extend(emitted)
        if close_index is None:
            continue
        continuation_index = _first_entry_index(
            market,
            start_index=close_index + 1,
            end_index=close_index + cfg.continuation_exit_age_bars,
            direction=direction,
            policy=cfg,
            mode="continuation_pullback",
        )
        if continuation_index is None:
            continue
        sequence += 1
        continuation, _ = _build_rotation_campaign(
            market,
            entry_index=continuation_index,
            chain_id=f"post_exit_rotation:{sequence:06d}",
            direction=direction,
            score=0.48,
            mode="continuation_pullback",
            policy=cfg,
        )
        records.extend(continuation)
    return pd.DataFrame.from_records(
        records,
        columns=(
            "signal_id",
            "chain_id",
            "symbol",
            "action",
            "operation_type",
            "signal_time",
            "signal_price",
            "position_fraction",
            "reason_family",
            "reason",
            "score",
        ),
    )


def _first_entry_index(
    market: pd.DataFrame,
    *,
    start_index: int,
    end_index: int,
    direction: float,
    policy: PostExitRecoveryPolicy,
    mode: str,
) -> int | None:
    """返回限定窗口内首个满足指定轮动机制的已收盘柱索引。"""

    if (
        mode == "exhaustion_rotation"
        and policy.exhaustion_rotation_requires_confirmation
    ):
        return _first_confirmed_exhaustion_rotation_index(
            market,
            start_index=start_index,
            end_index=end_index,
            direction=direction,
            policy=policy,
        )
    for bar_index in range(
        max(0, start_index),
        min(len(market), end_index + 1),
    ):
        bar = market.iloc[bar_index]
        if mode == "exhaustion_rotation":
            active = _is_exhaustion_rotation_entry(
                bar,
                direction=direction,
                policy=policy,
            )
        else:
            active = _is_continuation_pullback_entry(
                bar,
                direction=direction,
                policy=policy,
            )
        if active:
            return bar_index
    return None


def _first_confirmed_exhaustion_rotation_index(
    market: pd.DataFrame,
    *,
    start_index: int,
    end_index: int,
    direction: float,
    policy: PostExitRecoveryPolicy,
) -> int | None:
    """先记录衰竭候选, 再等待价格、广度和波动共同确认企稳。"""

    setup_index: int | None = None
    setup_end = min(len(market) - 1, end_index)
    confirmation_end = min(
        len(market) - 1,
        setup_end + policy.exhaustion_rotation_confirmation_bars,
    )
    for bar_index in range(max(0, start_index), confirmation_end + 1):
        bar = market.iloc[bar_index]
        if bar_index <= setup_end and _is_exhaustion_rotation_entry(
            bar,
            direction=direction,
            policy=policy,
        ):
            setup_index = bar_index
            continue
        if setup_index is None:
            continue
        if (
            bar_index - setup_index
            > policy.exhaustion_rotation_confirmation_bars
        ):
            continue
        if _is_exhaustion_rotation_confirmation(
            bar,
            direction=direction,
            policy=policy,
        ):
            return bar_index
    return None


def _build_rotation_campaign(
    market: pd.DataFrame,
    *,
    entry_index: int,
    chain_id: str,
    direction: float,
    score: float,
    mode: str,
    policy: PostExitRecoveryPolicy,
) -> tuple[list[dict[str, object]], int | None]:
    """从入口柱推进一条轮动持仓链，并返回动作及实际退出索引。"""

    entry = market.iloc[entry_index]
    records = [
        _action_row(
            entry,
            chain_id=chain_id,
            direction=direction,
            operation_type="OPEN",
            score=score,
            mode=mode,
        ),
    ]
    position: dict[str, Any] = {
        "chain_id": chain_id,
        "direction": direction,
        "entry_index": entry_index,
        "last_action_index": entry_index,
        "inventory_units": 1.0,
        "added": False,
        "mode": mode,
        "score": score,
    }
    for bar_index in range(entry_index + 1, len(market)):
        emitted, closed = _advance_position(
            market.iloc[bar_index],
            position=position,
            held_bars=bar_index - entry_index,
            policy=policy,
        )
        records.extend(emitted)
        if closed:
            return records, bar_index
    return records, None


def _advance_position(
    bar: pd.Series,
    *,
    position: dict[str, Any],
    held_bars: int,
    policy: PostExitRecoveryPolicy,
) -> tuple[list[dict[str, object]], bool]:
    """依据当前柱推进加仓、减仓或退出，并保持库存动作成对。"""

    direction = float(position["direction"])
    mode = str(position["mode"])
    values = _directional(bar, direction)
    gap = int(bar.name) - int(position["last_action_index"])
    common = {
        "chain_id": str(position["chain_id"]),
        "direction": direction,
        "score": float(position["score"]),
        "mode": mode,
    }
    if mode == "exhaustion_rotation":
        if (
            not bool(position["added"])
            and gap >= 3
            and values["recovery_return"]
            >= policy.exhaustion_confirmation_return_pct
            and values["location"] >= policy.exhaustion_confirmation_location
            and values["location"]
            <= policy.exhaustion_confirmation_location_maximum
            and values["rsi"] >= policy.exhaustion_confirmation_rsi
            and values["favorable_breadth20"] >= policy.minimum_breadth
            and float(bar["quote_volume_ratio_med30"])
            >= policy.exhaustion_confirmation_quote_volume_ratio
        ):
            position["added"] = True
            position["inventory_units"] = 2.0
            position["last_action_index"] = int(bar.name)
            return [
                _action_row(
                    bar,
                    operation_type="ADD",
                    **common,
                ),
            ], False
        if (
            bool(position["added"])
            and gap >= 3
            and values["mid_momentum"]
            >= policy.exhaustion_exit_mid_momentum_pct
            and values["long_momentum"]
            >= policy.exhaustion_exit_long_momentum_pct
            and values["location"] >= policy.exhaustion_exit_location
            and values["rsi"] >= policy.exhaustion_exit_rsi
            and values["favorable_breadth20"] >= policy.minimum_breadth
        ):
            return [
                _action_row(
                    bar,
                    operation_type="CLOSE",
                    position_fraction=float(position["inventory_units"]),
                    **common,
                ),
            ], True
    elif mode == "continuation_pullback":
        if (
            not bool(position["added"])
            and gap >= 3
            and values["long_momentum"]
            >= policy.continuation_long_momentum_pct
            and values["mid_momentum"]
            >= policy.continuation_mid_momentum_pct
            and values["tiny_return"]
            <= policy.continuation_add_pullback_return_pct
            and values["recovery_return"]
            <= policy.continuation_add_recovery_return_pct
            and values["location"] >= policy.continuation_location_minimum
            and values["rsi"] <= policy.continuation_add_rsi
            and values["adverse_breadth3"]
            >= policy.continuation_add_adverse_breadth3
        ):
            position["added"] = True
            position["inventory_units"] = 2.0
            position["last_action_index"] = int(bar.name)
            return [
                _action_row(
                    bar,
                    operation_type="ADD",
                    **common,
                ),
            ], False
        if (
            bool(position["added"])
            and gap >= 3
            and 0.0
            <= values["long_momentum"]
            <= policy.continuation_stall_long_momentum_pct
            and 0.0
            <= values["mid_momentum"]
            <= policy.continuation_stall_mid_momentum_pct
            and values["location"] >= policy.continuation_stall_location
            and values["rsi"] >= policy.continuation_stall_rsi
            and policy.continuation_stall_breadth_minimum
            <= values["favorable_breadth20"]
            <= policy.continuation_stall_breadth_maximum
            and policy.continuation_stall_atr_ratio_minimum
            <= float(bar["atr14_ratio120"])
            <= policy.continuation_stall_atr_ratio_maximum
            and float(bar["quote_volume_ratio_med30"])
            <= policy.continuation_stall_quote_volume_ratio
        ):
            return [
                _action_row(
                    bar,
                    operation_type="REDUCE",
                    leg="scaleout",
                    **common,
                ),
                _action_row(
                    bar,
                    operation_type="CLOSE",
                    leg="residual",
                    **common,
                ),
            ], True
    elif _is_exit(bar, direction=direction, policy=policy):
        return [
            _action_row(
                bar,
                operation_type="CLOSE",
                **common,
            ),
        ], True

    if held_bars < policy.maximum_holding_bars:
        return [], False
    return [
        _action_row(
            bar,
            operation_type="CLOSE",
            position_fraction=float(position["inventory_units"]),
            leg="maximum_holding",
            **common,
        ),
    ], True


def _is_exhaustion_rotation_entry(
    bar: pd.Series,
    *,
    direction: float,
    policy: PostExitRecoveryPolicy,
) -> bool:
    """判断反方向是否形成衰竭候选; 候选本身不会触发开仓。"""

    values = _directional(bar, direction)
    return bool(
        values["recovery_return"] < 0.0
        and values["location"] <= policy.exhaustion_rotation_location
        and values["rsi"] <= policy.exhaustion_rotation_rsi
        and values["adverse_breadth3"]
        >= policy.exhaustion_rotation_adverse_breadth3
        and float(bar["quote_volume_ratio_med30"])
        >= policy.exhaustion_rotation_quote_volume_ratio
        and float(bar["quote_volume_ratio_med30"])
        <= policy.exhaustion_rotation_maximum_quote_volume_ratio
        and float(bar["atr14_ratio120"])
        <= policy.exhaustion_rotation_maximum_atr_ratio
        and float(bar["adx14"]) <= policy.exhaustion_rotation_maximum_adx
        and _exhaustion_rotation_strength(bar, values)
        >= policy.exhaustion_rotation_minimum_strength,
    )


def _is_exhaustion_rotation_confirmation(
    bar: pd.Series,
    *,
    direction: float,
    policy: PostExitRecoveryPolicy,
) -> bool:
    """确认短周期跌势已停止, 且广度、流动性和趋势强度允许反转。"""

    values = _directional(bar, direction)
    maximum_adverse_return = (
        policy.exhaustion_rotation_confirmation_maximum_adverse_tiny_return_pct
    )
    return bool(
        values["tiny_return"] >= -maximum_adverse_return
        and values["favorable_breadth3"]
        >= policy.exhaustion_rotation_confirmation_minimum_favorable_breadth3
        and float(bar["quote_volume_ratio_med30"])
        <= policy.exhaustion_rotation_maximum_quote_volume_ratio
        and float(bar["atr14_ratio120"])
        <= policy.exhaustion_rotation_maximum_atr_ratio
        and float(bar["adx14"]) <= policy.exhaustion_rotation_maximum_adx,
    )


def _exhaustion_rotation_strength(
    bar: pd.Series,
    values: dict[str, float],
) -> float:
    """综合逆向冲击、市场活跃度、极端位置和趋势降温计算强度。"""

    adverse_recovery = _ramp(
        -values["recovery_return"],
        *_EXHAUSTION_RECOVERY_RANGE,
    )
    market_activity = (
        0.55
        * _ramp(
            float(bar["atr14_ratio120"]),
            *_EXHAUSTION_ATR_ACTIVITY_RANGE,
        )
        + 0.45
        * _log_ramp(
            float(bar["quote_volume_ratio_med30"]),
            *_EXHAUSTION_QUOTE_VOLUME_RANGE,
        )
    )
    impulse = (adverse_recovery * market_activity) ** 0.5
    structure = (
        _ramp(
            0.30 - values["location"],
            0.0,
            0.30,
        )
        + _ramp(
            35.0 - values["rsi"],
            0.0,
            20.0,
        )
        + _ramp(
            values["adverse_breadth3"],
            0.75,
            1.0,
        )
    ) / 3.0
    transition = 1.0 - _ramp(
        float(bar["adx14"]),
        20.0,
        40.0,
    )
    return float(
        0.65 * impulse
        + 0.25 * structure
        + 0.10 * transition,
    )


def _is_continuation_pullback_entry(
    bar: pd.Series,
    *,
    direction: float,
    policy: PostExitRecoveryPolicy,
) -> bool:
    """判断强趋势中的短周期回踩是否适合延续方向重新入场。"""

    values = _directional(bar, direction)
    return bool(
        values["long_momentum"] >= policy.continuation_long_momentum_pct
        and values["mid_momentum"] >= policy.continuation_mid_momentum_pct
        and values["tiny_return"] <= policy.continuation_pullback_return_pct
        and policy.continuation_location_minimum
        <= values["location"]
        <= policy.continuation_location_maximum
        and values["rsi"] <= policy.continuation_rsi
        and values["adverse_breadth3"]
        >= policy.continuation_adverse_breadth3
        and values["favorable_breadth20"] >= policy.minimum_breadth
        and float(bar["atr14_ratio120"])
        >= policy.continuation_atr_ratio,
    )


def _breadth_recovery_strength(
    bar: pd.Series,
    values: dict[str, float],
) -> float:
    """计算价格企稳、广度修复、流动性和趋势降温的综合恢复分。"""

    adverse_long = -values["long_momentum"]
    dislocation = max(
        0.0,
        1.0
        - abs(
            adverse_long
            - _BREADTH_RECOVERY_ADVERSE_LONG_CENTER,
        )
        / _BREADTH_RECOVERY_ADVERSE_LONG_RADIUS,
    )
    stabilization = 1.0 - _ramp(
        abs(values["mid_momentum"]),
        0.10,
        0.80,
    )
    recovery_structure = (
        _ramp(values["location"], 0.50, 0.85)
        + _ramp(values["rsi"], 45.0, 70.0)
        + _ramp(values["favorable_breadth20"], 0.75, 1.0)
        + _ramp(values["tiny_return"], 0.0, 0.40)
    ) / 4.0
    liquidity = _ramp(
        float(bar["quote_volume_ratio_med30"]),
        0.50,
        2.0,
    )
    trend_calmness = 1.0 - _ramp(
        float(bar["adx14"]),
        20.0,
        45.0,
    )
    return float(
        0.25 * dislocation
        + 0.20 * stabilization
        + 0.30 * recovery_structure
        + 0.10 * liquidity
        + 0.15 * trend_calmness,
    )


def _entry_state(
    bar: pd.Series,
    *,
    direction: float,
    policy: PostExitRecoveryPolicy,
) -> tuple[bool, float]:
    """评估退出后的广度恢复入口，并同时返回是否激活和强度分。"""

    values = _directional(bar, direction)
    score = _breadth_recovery_strength(bar, values)
    active = bool(
        policy.adverse_long_momentum_minimum_pct
        <= -values["long_momentum"]
        <= policy.adverse_long_momentum_maximum_pct
        and abs(values["mid_momentum"])
        <= policy.maximum_absolute_mid_momentum_pct
        and policy.location_minimum
        <= values["location"]
        <= policy.location_maximum
        and policy.rsi_minimum <= values["rsi"] <= policy.rsi_maximum
        and values["favorable_breadth20"] >= policy.minimum_breadth
        and values["tiny_return"] >= 0.0
        and float(bar["quote_volume_ratio_med30"])
        >= policy.minimum_quote_volume_ratio
        # 排除极端放量，避免把尚未完成的冲击误判为稳定恢复。
        and float(bar["quote_volume_ratio_med30"])
        <= policy.maximum_quote_volume_ratio
        and score >= policy.minimum_breadth_recovery_strength,
    )
    return active, score


def _is_exit(
    bar: pd.Series,
    *,
    direction: float,
    policy: PostExitRecoveryPolicy,
) -> bool:
    """在收益、动量、位置、RSI、广度和成交量共同成熟时退出。"""

    values = _directional(bar, direction)
    return bool(
        values["recovery_return"] >= policy.exit_recovery_return_pct
        and values["mid_momentum"] >= policy.exit_mid_momentum_pct
        and values["location"] >= policy.exit_location
        and values["rsi"] >= policy.exit_rsi
        and values["favorable_breadth20"] >= policy.minimum_breadth
        and float(bar["quote_volume_ratio_med30"])
        >= policy.exit_quote_volume_ratio,
    )


def _directional(bar: pd.Series, direction: float) -> dict[str, float]:
    """把市场特征转换为相对于目标方向一致的数值分量。"""

    values = directional_market_components(bar, direction)
    return {
        "tiny_return": values["tiny_return"],
        "recovery_return": values["recovery_return"],
        "mid_momentum": values["mid_momentum"],
        "long_momentum": values["long_momentum"],
        "location": values["location"],
        "rsi": values["rsi"],
        "favorable_breadth3": values["favorable_breadth3"],
        "favorable_breadth20": values["favorable_breadth20"],
        "adverse_breadth3": values["adverse_breadth3"],
    }


def _action_row(
    bar: pd.Series,
    *,
    chain_id: str,
    direction: float,
    operation_type: str,
    score: float,
    mode: str,
    position_fraction: float = 1.0,
    leg: str = "",
) -> dict[str, object]:
    """生成带稳定信号 ID、库存动作和机制说明的标准动作记录。"""

    direction_name = "LONG" if direction > 0.0 else "SHORT"
    action_type = (
        "OPEN" if operation_type in {"OPEN", "ADD"} else "CLOSE"
    )
    timestamp = pd.Timestamp(bar["signal_time"])
    suffix = f":{leg}" if leg else ""
    return {
        "signal_id": (
            f"{chain_id}:{operation_type}:{timestamp.isoformat()}{suffix}"
        ),
        "chain_id": chain_id,
        "symbol": str(bar["symbol"]),
        "action": f"{action_type}_{direction_name}",
        "operation_type": operation_type,
        "signal_time": timestamp,
        "signal_price": float(bar["close"]),
        "position_fraction": position_fraction,
        "reason_family": POST_EXIT_BREADTH_RECOVERY,
        "reason": (
            "autonomous_post_exit_regime;"
            f"mechanism={POST_EXIT_BREADTH_RECOVERY};mode={mode}"
        ),
        "score": float(score),
    }


def _prepare_market(market_panel: pd.DataFrame) -> pd.DataFrame:
    """校验单标的市场面板、统一 UTC 时间并保持稳定时间顺序。"""

    required = {
        "signal_time",
        "symbol",
        "close",
        "ret3",
        "ret12",
        "ret60",
        "tf1h_ret12",
        "donchian60_pos",
        "rsi14",
        "adx14",
        "atr14_ratio120",
        "quote_volume_ratio_med30",
        "market_down_breadth3",
        "market_up_breadth3",
        "market_down_breadth20",
        "market_up_breadth20",
    }
    missing = sorted(required.difference(market_panel.columns))
    if missing:
        raise ValueError("post-exit panel missing columns: " + ", ".join(missing))
    market = (
        market_panel.copy()
        .sort_values("signal_time", kind="stable")
        .reset_index(drop=True)
    )
    market["signal_time"] = pd.to_datetime(market["signal_time"], utc=True)
    symbols = market["symbol"].dropna().astype(str).unique()
    if len(symbols) != 1:
        message = "post-exit lifecycle requires exactly one symbol"
        raise ValueError(message)
    return market


def _prepare_controller_events(actions: pd.DataFrame) -> pd.DataFrame:
    """过滤本模块自产动作并按同柱库存语义整理控制器事件。"""

    required = {"signal_time", "action", "operation_type"}
    missing = sorted(required.difference(actions.columns))
    if missing:
        raise ValueError(
            "post-exit controller actions missing columns: " + ", ".join(missing),
        )
    events = actions.copy()
    if "reason_family" in events:
        events = events.loc[
            ~events["reason_family"].eq(POST_EXIT_BREADTH_RECOVERY)
        ].copy()
    events["signal_time"] = pd.to_datetime(events["signal_time"], utc=True)
    if "position_fraction" not in events:
        events["position_fraction"] = 1.0
    events["position_fraction"] = pd.to_numeric(
        events["position_fraction"],
        errors="coerce",
    ).fillna(1.0)
    priority = {"CLOSE": 0, "REDUCE": 1, "OPEN": 2, "ADD": 3}
    events["_priority"] = (
        events["operation_type"].astype(str).map(priority).fillna(4)
    )
    return (
        events.sort_values(
            ["signal_time", "_priority", "action"],
            kind="stable",
        )
        .drop(columns="_priority")
        .reset_index(drop=True)
    )


__all__ = [
    "NORMALIZED_RUNNER_EXIT_STATE",
    "POST_EXIT_BREADTH_RECOVERY",
    "PostExitRecoveryPolicy",
    "build_post_exit_recovery_actions",
    "build_post_exit_rotation_actions",
]
