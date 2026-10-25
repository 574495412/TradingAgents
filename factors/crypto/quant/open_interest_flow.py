"""H03持仓扩张与滞后成交方向交互，保留有效零和未知的区别。"""

import numpy as np
import pandas as pd

from .position_tilt import HOUR, MINUTE, _window, numeric

OI = "sum_open_interest"
GROWTH = "positive_log_oi_change_60m"
FLOW = "flow_60m_ending_raw_oi_clock"
NAME = "oi_expansion_flow_interaction"


def components(hourly, minutes):
    """完整小时OI和连续分钟表相接，F精确结束于t−10分钟。"""
    if (
        not pd.api.types.is_integer_dtype(hourly.clock_ms.dtype)
        or hourly.clock_ms.tolist() != sorted(set(hourly.clock_ms))
        or (len(hourly) > 1 and not np.all(np.diff(hourly.clock_ms) == HOUR))
        or not hourly.requested_metrics_raw_T_ms.eq(hourly.clock_ms - 10 * MINUTE).all()
        or not hourly.loc[hourly.raw_T_ms.notna(), "raw_T_ms"]
        .eq(hourly.loc[hourly.raw_T_ms.notna(), "requested_metrics_raw_T_ms"])
        .all()
        or not pd.api.types.is_integer_dtype(minutes.index.dtype)
        or minutes.index.has_duplicates
        or (len(minutes) > 1 and not np.all(np.diff(minutes.index) == MINUTE))
    ):
        raise ValueError("H03必须使用连续且精确对齐的完整小时／分钟日历")
    oi = numeric(hourly, OI).where(lambda v: v > 0)
    if (oi.notna() & hourly.raw_T_ms.isna()).any():
        raise ValueError("已知OI缺少对应原始时钟")
    growth = (np.log(oi) - np.log(oi.shift(1))).clip(lower=0)
    volume = numeric(minutes, "quote_volume").where(lambda v: v >= 0)
    buy = numeric(minutes, "taker_buy_quote_volume").where(lambda v: v >= 0)
    buy = buy.where(buy <= volume)
    denominator, numerator = _window(volume, 60), _window(buy, 60)
    flow = (2 * numerator / denominator - 1).where(denominator > 0)
    # 从精确u钟检索，不把小时聚合移十分钟，也不压缩未知分钟。
    aligned = flow.reindex(hourly.clock_ms.to_numpy() - 10 * MINUTE).to_numpy()
    result = pd.DataFrame(index=hourly.index)
    result[GROWTH], result[FLOW] = growth, aligned
    result[NAME] = result[GROWTH] * result[FLOW]
    return result.replace([np.inf, -np.inf], np.nan)
