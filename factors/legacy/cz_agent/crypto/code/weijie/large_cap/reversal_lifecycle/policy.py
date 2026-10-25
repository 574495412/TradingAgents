"""基础反转生命周期的入场、加仓、退出和冷却参数。"""

from __future__ import annotations

from dataclasses import dataclass, replace

MINIMUM_THRESHOLD_MULTIPLIER = 0.8
MAXIMUM_THRESHOLD_MULTIPLIER = 1.2


@dataclass(frozen=True, slots=True)
class ReversalLifecyclePolicy:
    """基础反转活动的评分阈值、持仓期限和风险边界。"""

    entry_threshold: float = 0.62
    entry_release_hysteresis: float = 0.10
    minimum_directional_edge: float = 0.06
    minimum_setup_bars: int = 2
    maximum_setup_bars: int = 24
    post_exit_cooldown_bars: int = 6

    maximum_adds: int = 3
    minimum_bars_between_adds: int = 6
    add_score_floor: float = 0.56
    add_distance_atr_multiple: float = 0.80

    maximum_reductions: int = 3
    minimum_bars_between_reductions: int = 3
    reduce_score_floor: float = 0.56
    close_score_floor: float = 0.68
    recovery_release_hysteresis: float = 0.10
    minimum_holding_bars: int = 3
    recovery_profit_floor_pct: float = 0.20
    final_profit_target_pct: float = 0.45

    hard_stop_pct: float = 6.0
    maximum_holding_bars: int = 576
    round_trip_cost_bps: float = 14.0

    def with_entry_threshold_multiplier(
        self,
        multiplier: float,
    ) -> ReversalLifecyclePolicy:
        if not MINIMUM_THRESHOLD_MULTIPLIER <= multiplier <= MAXIMUM_THRESHOLD_MULTIPLIER:
            message = "entry threshold multiplier must be in [0.8, 1.2]"
            raise ValueError(message)
        return replace(
            self,
            entry_threshold=min(1.0, self.entry_threshold * multiplier),
            add_score_floor=min(1.0, self.add_score_floor * multiplier),
        )

    def with_recovery_threshold_multiplier(
        self,
        multiplier: float,
    ) -> ReversalLifecyclePolicy:
        if not MINIMUM_THRESHOLD_MULTIPLIER <= multiplier <= MAXIMUM_THRESHOLD_MULTIPLIER:
            message = "recovery threshold multiplier must be in [0.8, 1.2]"
            raise ValueError(message)
        return replace(
            self,
            reduce_score_floor=min(1.0, self.reduce_score_floor * multiplier),
            close_score_floor=min(1.0, self.close_score_floor * multiplier),
        )


__all__ = ["ReversalLifecyclePolicy"]
