"""大盘标的市场结构模块：提供 policy 因子逻辑。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LargeCapRegimePolicy:
    """大盘标的状态信号的阈值、方向和风险边界。"""

    enabled_mechanisms: tuple[str, ...] = (
        "trend_rollover",
        "liquidation_absorption",
        "pullback_recovery",
    )
    enabled_trend_rollover_components: tuple[str, ...] = (
        "trend_divergence",
        "low_liquidity_blowoff",
        "volume_climax",
        "cross_horizon_stall",
    )
    trend_rollover_floor: float = 0.60
    liquidation_absorption_floor: float = 0.60
    pullback_recovery_floor: float = 0.62
    deleveraging_reversal_floor: float = 0.40
    crowded_extension_floor: float = 0.40
    minimum_confirmation_bars: int = 3
    release_hysteresis: float = 0.15
    recent_pullback_bars: int = 12
    open_cooldown_minutes: float = 120.0
    minimum_holding_bars: int = 12
    maximum_holding_bars: int = 72
    # 说明：以下逻辑仅使用决策时点可获得的数据。
    # 说明：以下逻辑仅使用决策时点可获得的数据。
    risk_budget_pct: float = 0.35
    max_single_margin_pct: float = 3.0
    minimum_stop_distance_pct: float = 0.80
    maximum_stop_distance_pct: float = 4.00
    stop_atr_multiple: float = 2.20
    minimum_profit_decay_pct: float = 0.40
    profit_atr_multiple: float = 1.5

    def threshold(self, mechanism: str) -> float:
        """按信号方向返回对应的开仓阈值。"""

        values = {
            "trend_rollover": self.trend_rollover_floor,
            "liquidation_absorption": self.liquidation_absorption_floor,
            "pullback_recovery": self.pullback_recovery_floor,
            "deleveraging_reversal": self.deleveraging_reversal_floor,
            "crowded_extension": self.crowded_extension_floor,
        }
        if mechanism not in values:
            raise ValueError(f"unknown large-cap mechanism: {mechanism}")
        return float(values[mechanism])
