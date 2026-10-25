"""大盘标的市场结构模块：提供 positioning 因子逻辑。"""

from __future__ import annotations

import pandas as pd

from factors.crypto.code.weijie.large_cap.normalization import (
    action_direction,
    clip01,
    finite,
    unit_scale,
)
from factors.crypto.code.weijie.large_cap.types import (
    DirectionalMomentum,
    LiquidityQuality,
    MarketStructure,
    PositioningComponents,
    PositioningQuality,
    RejectionQuality,
)


def score_positioning_quality(
    row: pd.Series,
    *,
    action: str,
) -> PositioningQuality:
    """score_positioning_quality：计算可解释的市场结构因子。"""

    available = finite(row.get("positioning_available")) >= 0.5
    if not available:
        return PositioningQuality(
            available=False,
            oi_deleveraging=0.0,
            oi_value_deleveraging=0.0,
            oi_build=0.0,
            oi_value_build=0.0,
            value_ratio_low=0.0,
            value_ratio_high=0.0,
            opposing_funding_crowding=0.0,
        )
    oi_change = finite(row.get("oi_change12"))
    oi_value_change = finite(row.get("oi_value_change12"))
    value_ratio = finite(row.get("oi_value_ratio96"), default=1.0)
    direction = action_direction(action)
    opposing_funding_bps = -direction * finite(row.get("funding_rate")) * 10_000.0
    return PositioningQuality(
        available=True,
        oi_deleveraging=unit_scale(-oi_change, 0.10, 1.00),
        oi_value_deleveraging=unit_scale(-oi_value_change, 0.20, 3.00),
        oi_build=unit_scale(oi_change, 0.15, 1.50),
        oi_value_build=unit_scale(oi_value_change, 0.30, 3.00),
        value_ratio_low=unit_scale(1.0 - value_ratio, 0.005, 0.050),
        value_ratio_high=unit_scale(value_ratio - 1.0, 0.005, 0.050),
        opposing_funding_crowding=unit_scale(opposing_funding_bps, 0.10, 1.00),
    )


def score_positioning_components(
    *,
    positioning: PositioningQuality,
    momentum: DirectionalMomentum,
    liquidity: LiquidityQuality,
    rejection: RejectionQuality,
    structure: MarketStructure,
) -> PositioningComponents:
    """score_positioning_components：计算可解释的市场结构因子。"""

    if not positioning.available:
        return PositioningComponents(
            deleveraging_reversal=0.0,
            crowded_extension=0.0,
        )
    adverse_impulse = max(0.0, -momentum.short, -momentum.mid)
    deleveraging_reversal = clip01(
        0.30 * adverse_impulse
        + 0.25 * positioning.oi_deleveraging
        + 0.20 * positioning.oi_value_deleveraging
        + 0.10 * positioning.value_ratio_low
        + 0.10 * structure.breadth_confirmation
        + 0.05 * positioning.opposing_funding_crowding,
    )
    crowded_extension = clip01(
        0.28 * adverse_impulse
        + 0.22 * positioning.oi_build
        + 0.18 * positioning.oi_value_build
        + 0.14 * positioning.value_ratio_high
        + 0.10 * positioning.opposing_funding_crowding
        + 0.08 * rejection.quality,
    )
    # 计算 _，仅使用当前及历史数据。
    # 计算 _，仅使用当前及历史数据。
    _ = liquidity
    return PositioningComponents(
        deleveraging_reversal=deleveraging_reversal,
        crowded_extension=crowded_extension,
    )


__all__ = ["score_positioning_components", "score_positioning_quality"]
