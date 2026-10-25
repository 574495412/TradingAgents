"""大盘标的市场结构模块：提供 absorption 因子逻辑。"""

from __future__ import annotations

from factors.crypto.code.weijie.large_cap.normalization import clip01
from factors.crypto.code.weijie.large_cap.types import (
    AbsorptionComponents,
    DirectionalMomentum,
    LiquidityQuality,
    MarketStructure,
    RejectionQuality,
)


def score_absorption_components(
    *,
    momentum: DirectionalMomentum,
    liquidity: LiquidityQuality,
    rejection: RejectionQuality,
    structure: MarketStructure,
) -> AbsorptionComponents:
    """score_absorption_components：计算可解释的市场结构因子。"""

    opposing_short = max(0.0, -momentum.short)
    opposing_mid = max(0.0, -momentum.mid)
    opposing_long = max(0.0, -momentum.long)
    return AbsorptionComponents(
        liquidation_flush=clip01(
            0.26 * opposing_short
            + 0.20 * opposing_mid
            + 0.16 * max(liquidity.volume_support, liquidity.dislocation_risk)
            + 0.14 * structure.breadth_confirmation
            + 0.12 * structure.volatility_state
            + 0.12 * rejection.quality,
        ),
        deep_value=clip01(
            0.24 * opposing_long
            + 0.22 * opposing_mid
            + 0.20 * structure.adverse_crowding
            + 0.16 * liquidity.volume_support
            + 0.18 * rejection.quality,
        ),
        flow_reversal=clip01(
            0.30 * opposing_short
            + 0.28 * rejection.quality
            + 0.22 * liquidity.flow_support
            + 0.12 * structure.breadth_confirmation
            + 0.08 * structure.volatility_state,
        ),
    )
