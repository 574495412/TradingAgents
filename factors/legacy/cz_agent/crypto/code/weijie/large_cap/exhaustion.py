"""大盘标的市场结构模块：提供 exhaustion 因子逻辑。"""

from __future__ import annotations

from factors.crypto.code.weijie.large_cap.normalization import clip01
from factors.crypto.code.weijie.large_cap.types import (
    DirectionalMomentum,
    ExhaustionComponents,
    LiquidityQuality,
    MarketStructure,
    RejectionQuality,
)


def score_exhaustion_components(
    *,
    momentum: DirectionalMomentum,
    liquidity: LiquidityQuality,
    rejection: RejectionQuality,
    structure: MarketStructure,
) -> ExhaustionComponents:
    """score_exhaustion_components：计算可解释的市场结构因子。"""

    opposing_short = max(0.0, -momentum.short)
    opposing_mid = max(0.0, -momentum.mid)
    opposing_long = max(0.0, -momentum.long)
    directional_turn = max(0.0, momentum.short)
    short_stall = clip01(1.0 - abs(momentum.short) / 0.20)
    low_liquidity = 1.0 - liquidity.volume_support
    return ExhaustionComponents(
        trend_divergence=clip01(
            0.30 * opposing_mid
            + 0.24 * opposing_long
            + 0.16 * directional_turn
            + 0.16 * structure.breadth_confirmation
            + 0.14 * rejection.quality,
        ),
        low_liquidity_blowoff=clip01(
            0.28 * opposing_short
            + 0.24 * opposing_mid
            + 0.18 * structure.volatility_state
            + 0.18 * low_liquidity
            + 0.12 * rejection.quality,
        ),
        volume_climax=clip01(
            0.24 * opposing_mid
            + 0.24 * liquidity.dislocation_risk
            + 0.22 * structure.adverse_crowding
            + 0.16 * rejection.quality
            + 0.14 * structure.volatility_state,
        ),
        cross_horizon_stall=clip01(
            0.30 * opposing_mid
            + 0.22 * opposing_long
            + 0.20 * short_stall
            + 0.16 * low_liquidity
            + 0.12 * structure.breadth_confirmation,
        ),
    )
