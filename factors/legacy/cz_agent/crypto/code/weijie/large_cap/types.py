"""大盘标的市场结构模块：提供 types 因子逻辑。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DirectionalMomentum:
    """DirectionalMomentum：封装相关市场结构数据与行为。"""

    short: float
    mid: float
    long: float
    impulse: float


@dataclass(frozen=True)
class LiquidityQuality:
    """LiquidityQuality：封装相关市场结构数据与行为。"""

    volume_support: float
    flow_support: float
    dislocation_risk: float
    quality: float


@dataclass(frozen=True)
class RejectionQuality:
    """RejectionQuality：封装相关市场结构数据与行为。"""

    wick_rejection: float
    close_rejection: float
    flow_rejection: float
    quality: float


@dataclass(frozen=True)
class MarketStructure:
    """MarketStructure：封装相关市场结构数据与行为。"""

    trend_alignment: float
    adverse_crowding: float
    breadth_confirmation: float
    multi_timeframe_alignment: float
    atr_pct: float
    volatility_state: float


@dataclass(frozen=True)
class ExhaustionComponents:
    """ExhaustionComponents：封装相关市场结构数据与行为。"""

    trend_divergence: float
    low_liquidity_blowoff: float
    volume_climax: float
    cross_horizon_stall: float


@dataclass(frozen=True)
class AbsorptionComponents:
    """AbsorptionComponents：封装相关市场结构数据与行为。"""

    liquidation_flush: float
    deep_value: float
    flow_reversal: float


@dataclass(frozen=True)
class RecoveryComponents:
    """RecoveryComponents：封装相关市场结构数据与行为。"""

    flow_reclaim: float
    trend_pullback: float
    continuation: float
    pullback_memory: float


@dataclass(frozen=True)
class PositioningQuality:
    """PositioningQuality：封装相关市场结构数据与行为。"""

    available: bool
    oi_deleveraging: float
    oi_value_deleveraging: float
    oi_build: float
    oi_value_build: float
    value_ratio_low: float
    value_ratio_high: float
    opposing_funding_crowding: float


@dataclass(frozen=True)
class PositioningComponents:
    """PositioningComponents：封装相关市场结构数据与行为。"""

    deleveraging_reversal: float
    crowded_extension: float


@dataclass(frozen=True)
class MechanismScores:
    """MechanismScores：封装相关市场结构数据与行为。"""

    trend_rollover: float
    liquidation_absorption: float
    pullback_recovery: float
    deleveraging_reversal: float
    crowded_extension: float
    shock_pressure: float


@dataclass(frozen=True)
class LargeCapFactorSnapshot:
    """LargeCapFactorSnapshot：保存因果运行状态，不包含未来信息。"""

    action: str
    momentum: DirectionalMomentum
    liquidity: LiquidityQuality
    rejection: RejectionQuality
    structure: MarketStructure
    exhaustion: ExhaustionComponents
    absorption: AbsorptionComponents
    recovery: RecoveryComponents
    positioning: PositioningQuality
    positioning_components: PositioningComponents
    mechanisms: MechanismScores


@dataclass(frozen=True)
class LargeCapRegimeDecision:
    """LargeCapRegimeDecision：封装当前时点的计算结果。"""

    action: str
    mechanism: str
    confidence: float
    snapshot: LargeCapFactorSnapshot
