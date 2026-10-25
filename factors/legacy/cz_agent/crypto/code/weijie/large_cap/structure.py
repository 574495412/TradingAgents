"""大盘标的市场结构模块：提供 structure 因子逻辑。"""

from __future__ import annotations

import pandas as pd

from factors.crypto.code.weijie.large_cap.normalization import (
    action_direction,
    clip01,
    finite,
    signed_scale,
    unit_scale,
)
from factors.crypto.code.weijie.large_cap.types import (
    DirectionalMomentum,
    MarketStructure,
)


def score_market_structure(
    row: pd.Series,
    *,
    action: str,
    momentum: DirectionalMomentum,
) -> MarketStructure:
    """score_market_structure：计算可解释的市场结构因子。"""

    direction = action_direction(action)
    donchian = finite(row.get("donchian60_pos"), 0.50)
    directional_donchian = donchian if direction > 0.0 else 1.0 - donchian
    rsi = finite(row.get("rsi14"), 50.0)
    opposing_rsi_extreme = (
        unit_scale(50.0 - rsi, 0.0, 30.0)
        if direction > 0.0
        else unit_scale(rsi - 50.0, 0.0, 30.0)
    )
    opposing_donchian_extreme = unit_scale(0.50 - directional_donchian, 0.0, 0.45)
    down_breadth = finite(row.get("market_down_breadth20"), 0.50)
    breadth_confirmation = down_breadth if direction > 0.0 else 1.0 - down_breadth
    adx = finite(row.get("adx14"), 20.0)
    tf15 = signed_scale(finite(row.get("tf15_ret4")) * direction, 1.50)
    tf1h = signed_scale(finite(row.get("tf1h_ret4")) * direction, 3.00)
    multi_timeframe = clip01(0.50 + 0.30 * tf15 + 0.20 * tf1h)
    trend_alignment = clip01(
        0.28 * max(0.0, momentum.mid)
        + 0.24 * max(0.0, momentum.long)
        + 0.18 * unit_scale(adx, 15.0, 45.0)
        + 0.16 * unit_scale(directional_donchian, 0.50, 0.90)
        + 0.14 * multi_timeframe,
    )
    adverse_crowding = clip01(
        0.24 * max(0.0, -momentum.mid)
        + 0.24 * max(0.0, -momentum.long)
        + 0.20 * opposing_rsi_extreme
        + 0.18 * opposing_donchian_extreme
        + 0.14 * clip01(breadth_confirmation),
    )
    atr_pct = max(0.0, finite(row.get("atr14_pct")))
    return MarketStructure(
        trend_alignment=trend_alignment,
        adverse_crowding=adverse_crowding,
        breadth_confirmation=clip01(breadth_confirmation),
        multi_timeframe_alignment=multi_timeframe,
        atr_pct=atr_pct,
        volatility_state=unit_scale(atr_pct, 0.20, 2.00),
    )
