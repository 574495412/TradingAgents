"""大盘标的市场结构模块：提供 eligibility 因子逻辑。"""

from __future__ import annotations

from factors.crypto.code.weijie.large_cap.types import LargeCapFactorSnapshot


def eligible_market_support(snapshot: LargeCapFactorSnapshot) -> dict[str, float]:
    """判断市场面板是否具备大盘标的机制所需的数据支持。"""

    momentum = snapshot.momentum
    opposing_short = max(0.0, -momentum.short)
    opposing_mid = max(0.0, -momentum.mid)
    opposing_long = max(0.0, -momentum.long)
    directional_short = max(0.0, momentum.short)
    directional_mid = max(0.0, momentum.mid)
    directional_long = max(0.0, momentum.long)
    positioning = snapshot.positioning
    has_positioning_liquidity = snapshot.liquidity.quality >= 0.10

    has_opposing_trend = opposing_mid >= 0.08 and (
        opposing_long >= 0.06
        or opposing_short >= 0.12
        or snapshot.rejection.quality >= 0.20
    )
    has_liquidation_shock = max(opposing_short, opposing_mid) >= 0.12 and (
        snapshot.liquidity.volume_support >= 0.12
        or snapshot.liquidity.dislocation_risk >= 0.10
        or snapshot.rejection.quality >= 0.20
    )
    has_directional_recovery = (
        directional_short >= 0.08
        and directional_mid >= 0.06
        and directional_long >= 0.06
        and snapshot.recovery.pullback_memory >= 0.20
    )
    has_deleveraging_reversal = (
        positioning.available
        and max(opposing_short, opposing_mid) >= 0.12
        and snapshot.structure.adverse_crowding >= 0.50
        and snapshot.rejection.quality >= 0.20
        and max(
            positioning.oi_deleveraging,
            positioning.oi_value_deleveraging,
        ) >= 0.12
        and has_positioning_liquidity
    )
    has_crowded_extension = (
        positioning.available
        and max(opposing_short, opposing_mid) >= 0.12
        and snapshot.mechanisms.shock_pressure >= 0.50
        and positioning.oi_build >= 0.10
        and max(positioning.oi_value_build, positioning.value_ratio_high) >= 0.10
        and has_positioning_liquidity
    )
    return {
        "trend_rollover": (
            snapshot.mechanisms.trend_rollover if has_opposing_trend else 0.0
        ),
        "liquidation_absorption": (
            snapshot.mechanisms.liquidation_absorption
            if has_liquidation_shock
            else 0.0
        ),
        "pullback_recovery": (
            snapshot.mechanisms.pullback_recovery
            if has_directional_recovery
            else 0.0
        ),
        "deleveraging_reversal": (
            snapshot.mechanisms.deleveraging_reversal
            if has_deleveraging_reversal
            else 0.0
        ),
        "crowded_extension": (
            snapshot.mechanisms.crowded_extension
            if has_crowded_extension
            else 0.0
        ),
    }


__all__ = ["eligible_market_support"]
