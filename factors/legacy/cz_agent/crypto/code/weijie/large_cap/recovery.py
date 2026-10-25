"""大盘标的市场结构模块：提供 recovery 因子逻辑。"""

from __future__ import annotations

from factors.crypto.code.weijie.large_cap.normalization import clip01
from factors.crypto.code.weijie.large_cap.types import (
    DirectionalMomentum,
    LiquidityQuality,
    MarketStructure,
    RecoveryComponents,
    RejectionQuality,
)


def score_recovery_components(
    *,
    momentum: DirectionalMomentum,
    liquidity: LiquidityQuality,
    rejection: RejectionQuality,
    structure: MarketStructure,
    recent_pullback: float,
) -> RecoveryComponents:
    """score_recovery_components：计算可解释的市场结构因子。"""

    pullback_memory = clip01(recent_pullback)
    directional_short = max(0.0, momentum.short)
    directional_mid = max(0.0, momentum.mid)
    directional_long = max(0.0, momentum.long)
    neutral_long = clip01(1.0 - abs(momentum.long) / 0.35)
    return RecoveryComponents(
        flow_reclaim=clip01(
            0.30 * liquidity.flow_support
            + 0.20 * liquidity.volume_support
            + 0.20 * rejection.quality
            + 0.15 * neutral_long
            + 0.15 * structure.multi_timeframe_alignment,
        ),
        trend_pullback=clip01(
            0.26 * directional_mid
            + 0.20 * directional_long
            + 0.22 * pullback_memory
            + 0.17 * liquidity.flow_support
            + 0.15 * rejection.quality,
        ),
        continuation=clip01(
            0.28 * structure.trend_alignment
            + 0.22 * structure.multi_timeframe_alignment
            + 0.20 * liquidity.quality
            + 0.16 * directional_short
            + 0.14 * pullback_memory,
        ),
        pullback_memory=pullback_memory,
    )
