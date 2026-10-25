"""大盘标的市场结构模块：提供 mechanisms 因子逻辑。"""

from __future__ import annotations

import pandas as pd

from factors.crypto.code.weijie.large_cap.absorption import (
    score_absorption_components,
)
from factors.crypto.code.weijie.large_cap.exhaustion import (
    score_exhaustion_components,
)
from factors.crypto.code.weijie.large_cap.liquidity import (
    score_liquidity_quality,
    score_rejection_quality,
)
from factors.crypto.code.weijie.large_cap.momentum import score_directional_momentum
from factors.crypto.code.weijie.large_cap.normalization import clip01
from factors.crypto.code.weijie.large_cap.positioning import (
    score_positioning_components,
    score_positioning_quality,
)
from factors.crypto.code.weijie.large_cap.recovery import (
    score_recovery_components,
)
from factors.crypto.code.weijie.large_cap.structure import score_market_structure
from factors.crypto.code.weijie.large_cap.types import (
    LargeCapFactorSnapshot,
    MechanismScores,
)


def score_large_cap_mechanisms(
    row: pd.Series,
    *,
    action: str,
    recent_pullback: float = 0.0,
    enabled_trend_rollover_components: tuple[str, ...] | None = None,
) -> LargeCapFactorSnapshot:
    """score_large_cap_mechanisms：计算可解释的市场结构因子。"""

    momentum = score_directional_momentum(row, action=action)
    liquidity = score_liquidity_quality(row, action=action)
    rejection = score_rejection_quality(row, action=action)
    structure = score_market_structure(row, action=action, momentum=momentum)
    exhaustion = score_exhaustion_components(
        momentum=momentum,
        liquidity=liquidity,
        rejection=rejection,
        structure=structure,
    )
    absorption = score_absorption_components(
        momentum=momentum,
        liquidity=liquidity,
        rejection=rejection,
        structure=structure,
    )
    recovery = score_recovery_components(
        momentum=momentum,
        liquidity=liquidity,
        rejection=rejection,
        structure=structure,
        recent_pullback=recent_pullback,
    )
    positioning = score_positioning_quality(row, action=action)
    positioning_components = score_positioning_components(
        positioning=positioning,
        momentum=momentum,
        liquidity=liquidity,
        rejection=rejection,
        structure=structure,
    )
    shock_pressure = clip01(
        0.24 * max(0.0, -momentum.short)
        + 0.20 * max(0.0, -momentum.mid)
        + 0.18 * liquidity.dislocation_risk
        + 0.18 * structure.volatility_state
        + 0.20 * structure.breadth_confirmation,
    )
    rollover_scores = {
        "trend_divergence": exhaustion.trend_divergence,
        "low_liquidity_blowoff": exhaustion.low_liquidity_blowoff,
        "volume_climax": exhaustion.volume_climax,
        "cross_horizon_stall": exhaustion.cross_horizon_stall,
    }
    enabled_rollover = (
        tuple(rollover_scores)
        if enabled_trend_rollover_components is None
        else tuple(enabled_trend_rollover_components)
    )
    unknown_rollover = sorted(set(enabled_rollover).difference(rollover_scores))
    if unknown_rollover:
        raise ValueError(
            "unknown trend-rollover component: " + ",".join(unknown_rollover),
        )
    trend_rollover = max(
        (rollover_scores[name] for name in enabled_rollover),
        default=0.0,
    )
    liquidation_absorption = max(
        absorption.liquidation_flush,
        absorption.deep_value,
        absorption.flow_reversal,
    )
    pullback_recovery = max(
        recovery.flow_reclaim,
        recovery.trend_pullback,
        recovery.continuation,
    )
    return LargeCapFactorSnapshot(
        action=str(action).upper(),
        momentum=momentum,
        liquidity=liquidity,
        rejection=rejection,
        structure=structure,
        exhaustion=exhaustion,
        absorption=absorption,
        recovery=recovery,
        positioning=positioning,
        positioning_components=positioning_components,
        mechanisms=MechanismScores(
            trend_rollover=trend_rollover,
            liquidation_absorption=liquidation_absorption,
            pullback_recovery=pullback_recovery,
            deleveraging_reversal=positioning_components.deleveraging_reversal,
            crowded_extension=positioning_components.crowded_extension,
            shock_pressure=shock_pressure,
        ),
    )
