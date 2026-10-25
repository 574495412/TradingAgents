"""大盘标的市场结构模块：提供 liquidity 因子逻辑。"""

from __future__ import annotations

import pandas as pd

from factors.crypto.code.weijie.large_cap.normalization import (
    action_direction,
    clip01,
    finite,
    unit_scale,
)
from factors.crypto.code.weijie.large_cap.types import (
    LiquidityQuality,
    RejectionQuality,
)


def score_liquidity_quality(row: pd.Series, *, action: str) -> LiquidityQuality:
    """score_liquidity_quality：计算可解释的市场结构因子。"""

    direction = action_direction(action)
    qvr = max(0.0, finite(row.get("quote_volume_ratio_med30"), 1.0))
    taker_buy = finite(row.get("taker_buy_ratio"), 0.50)
    directional_flow = taker_buy if direction > 0.0 else 1.0 - taker_buy
    volume_support = unit_scale(qvr, 0.50, 4.00)
    flow_support = unit_scale(directional_flow, 0.42, 0.66)
    dislocation_risk = unit_scale(qvr, 8.00, 20.00)
    quality = clip01(
        0.50 * volume_support
        + 0.35 * flow_support
        + 0.15 * (1.0 - dislocation_risk),
    )
    return LiquidityQuality(
        volume_support=volume_support,
        flow_support=flow_support,
        dislocation_risk=dislocation_risk,
        quality=quality,
    )


def score_rejection_quality(row: pd.Series, *, action: str) -> RejectionQuality:
    """score_rejection_quality：计算可解释的市场结构因子。"""

    direction = action_direction(action)
    upper_wick = max(0.0, finite(row.get("upper_wick_pct")))
    lower_wick = max(0.0, finite(row.get("lower_wick_pct")))
    wick_total = upper_wick + lower_wick
    directional_wick_share = (
        (lower_wick if direction > 0.0 else upper_wick) / wick_total
        if wick_total > 1e-12
        else 0.50
    )
    close_position = finite(row.get("close_pos"), 0.50)
    directional_close = close_position if direction > 0.0 else 1.0 - close_position
    taker_buy = finite(row.get("taker_buy_ratio"), 0.50)
    directional_flow = taker_buy if direction > 0.0 else 1.0 - taker_buy
    wick_rejection = unit_scale(directional_wick_share, 0.45, 0.85)
    close_rejection = unit_scale(directional_close, 0.40, 0.85)
    flow_rejection = unit_scale(directional_flow, 0.45, 0.65)
    return RejectionQuality(
        wick_rejection=wick_rejection,
        close_rejection=close_rejection,
        flow_rejection=flow_rejection,
        quality=clip01(
            0.40 * wick_rejection
            + 0.35 * close_rejection
            + 0.25 * flow_rejection,
        ),
    )
