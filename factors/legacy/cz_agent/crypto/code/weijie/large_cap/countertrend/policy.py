"""大盘标的反趋势活动的入口、退出和风险参数。"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class CountertrendFactorPolicy:
    enabled_mechanisms: tuple[str, ...] = (
        "liquidity_shock_exhaustion",
        "relative_capitulation",
        "multi_timeframe_overshoot",
    )
    disabled_components: tuple[str, ...] = ()
    short_impulse_start: float = 0.75
    short_impulse_full: float = 2.50
    mid_impulse_start: float = 1.00
    mid_impulse_full: float = 3.00
    relative_impulse_start: float = 0.50
    relative_impulse_full: float = 2.00
    extreme_position_start: float = 0.65
    extreme_position_full: float = 0.95
    rsi_extreme_start: float = 10.0
    rsi_extreme_full: float = 30.0
    adx_start: float = 18.0
    adx_full: float = 38.0
    atr_expansion_start: float = 1.10
    atr_expansion_full: float = 2.00
    range_expansion_start: float = 1.10
    range_expansion_full: float = 2.50
    quote_expansion_start: float = 1.50
    quote_expansion_full: float = 5.00
    breadth_start: float = 0.55
    breadth_full: float = 0.85
    liquidity_shock_exhaustion_floor: float = 0.62
    relative_capitulation_floor: float = 0.62
    multi_timeframe_overshoot_floor: float = 0.62
    release_hysteresis: float = 0.12
    minimum_directional_edge: float = 0.05
    open_cooldown_bars: int = 12

    def threshold(self, mechanism: str) -> float:
        values = {
            "liquidity_shock_exhaustion": self.liquidity_shock_exhaustion_floor,
            "relative_capitulation": self.relative_capitulation_floor,
            "multi_timeframe_overshoot": self.multi_timeframe_overshoot_floor,
        }
        if mechanism not in values:
            raise ValueError(f"unknown countertrend mechanism: {mechanism}")
        return float(values[mechanism])

    def with_threshold_multiplier(self, multiplier: float) -> CountertrendFactorPolicy:
        if not 0.5 <= multiplier <= 1.5:
            raise ValueError("countertrend threshold multiplier must be in [0.5, 1.5]")
        return replace(
            self,
            liquidity_shock_exhaustion_floor=min(
                1.0,
                self.liquidity_shock_exhaustion_floor * multiplier,
            ),
            relative_capitulation_floor=min(
                1.0,
                self.relative_capitulation_floor * multiplier,
            ),
            multi_timeframe_overshoot_floor=min(
                1.0,
                self.multi_timeframe_overshoot_floor * multiplier,
            ),
        )


@dataclass(frozen=True, slots=True)
class CountertrendLifecyclePolicy:
    allocated_account_fraction: float = 0.50
    maximum_total_margin_fraction: float = 0.50
    maximum_single_margin_pct: float = 3.00
    stop_atr_multiple: float = 3.00
    minimum_stop_distance_pct: float = 1.20
    maximum_stop_distance_pct: float = 6.00
    minimum_holding_bars: int = 6
    maximum_holding_bars: int = 144
    round_trip_cost_bps: float = 14.0
