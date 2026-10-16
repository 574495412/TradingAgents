"""按已声明覆盖区间和五分钟可见延迟构造固定资金费因子。"""

from collections.abc import Mapping
from decimal import Decimal, localcontext
from itertools import pairwise
from numbers import Integral, Real

import numpy as np
import pandas as pd

NAMES = ("funding_carry_24h", "funding_carry_change_24h")
DAY = 86_400_000
LAG = 300_000


def _integer_ms(value, field):
    """拒绝布尔值、浮点数及其他非整数毫秒值。"""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{field} 必须是整数毫秒")
    return int(value)


def _decimal_factors(rates, right, middle, left):
    """精确累积十进制费率，避免窗口前的历史余额改变相等因子的名次。"""
    decimals = [Decimal(str(rate)) for rate in rates]
    minimum_exponent = min((value.as_tuple().exponent for value in decimals), default=0)
    maximum_adjusted = max((value.adjusted() for value in decimals), default=0)
    count_digits = len(str(max(1, len(decimals))))
    # 最小十进制单位到最大总和之间的全部数位都须保留，不能按结果事后舍入。
    precision = max(1, maximum_adjusted - minimum_exponent + 1 + count_digits)
    with localcontext() as context:
        context.prec = precision
        context.Emax = max(0, maximum_adjusted + count_digits)
        context.Emin = min(0, minimum_exponent)
        prefix = [Decimal(0)]
        for rate in decimals:
            prefix.append(prefix[-1] + rate)
        carry, change = [], []
        for end, split, start in zip(right, middle, left, strict=True):
            current_sum = prefix[end] - prefix[split]
            previous_sum = prefix[split] - prefix[start]
            carry.append(float(-current_sum))
            change.append(float(-(current_sum - previous_sum)))
    return np.asarray(carry), np.asarray(change)


def calculate(clocks, events, coverage_start_ms, coverage_end_ms):
    """用截止于时钟前五分钟的事件计算两个固定窗口因子。

    覆盖区间两端均包含；调用者负责将来源的右开端点转换为最后已覆盖毫秒。
    每个二十四小时窗口左开右闭，并且至少有一个实际事件才算已知。
    两个因子分别是最近窗口费率和的相反数，以及最近与此前窗口和之差的相反数。
    此函数只接收已规范化的单一合约事件，不推断结算频率或筛选事件类型。
    """
    coverage_start = _integer_ms(coverage_start_ms, "覆盖起点")
    coverage_end = _integer_ms(coverage_end_ms, "覆盖终点")
    if coverage_start > coverage_end:
        raise ValueError("覆盖起点不能晚于覆盖终点")

    index = pd.Index(clocks)
    clock_values = [_integer_ms(value, "研究时钟") for value in index]
    if any(current <= previous for previous, current in pairwise(clock_values)):
        raise ValueError("研究时钟必须唯一且严格递增")

    normalized = []
    symbols = set()
    for event in events:
        if not isinstance(event, Mapping):
            raise ValueError("资金费事件必须是规范化字典")
        symbol = event.get("symbol")
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("资金费事件必须声明合约 symbol")
        symbols.add(symbol)
        timestamp = _integer_ms(event.get("time_ms"), "事件时间")
        if not coverage_start <= timestamp <= coverage_end:
            raise ValueError("资金费事件越出声明覆盖区间")
        rate = event.get("rate")
        if isinstance(rate, (bool, np.bool_)) or not isinstance(rate, Real):
            raise ValueError("资金费率必须是有限数值")
        rate = float(rate)
        if not np.isfinite(rate):
            raise ValueError("资金费率必须是有限数值")
        normalized.append((timestamp, rate))
    if len(symbols) > 1:
        raise ValueError("资金费事件只能属于单一合约")
    normalized.sort(key=lambda event: event[0])
    if any(previous[0] == current[0] for previous, current in pairwise(normalized)):
        raise ValueError("资金费事件时间戳不能重复")

    # 对象整数避免极端毫秒值在减去窗口长度时发生整数溢出。
    times = np.asarray([event[0] for event in normalized], dtype=object)
    cutoffs = np.asarray([clock - LAG for clock in clock_values], dtype=object)
    right = np.searchsorted(times, cutoffs, side="right")
    middle = np.searchsorted(times, cutoffs - DAY, side="right")
    left = np.searchsorted(times, cutoffs - 2 * DAY, side="right")
    carry, change = _decimal_factors([event[1] for event in normalized], right, middle, left)

    current_count = right - middle
    previous_count = middle - left
    current_ready = (
        (cutoffs - DAY >= coverage_start) & (cutoffs <= coverage_end) & (current_count > 0)
    )
    change_ready = current_ready & (cutoffs - 2 * DAY >= coverage_start) & (previous_count > 0)
    result = pd.DataFrame(index=index)
    result[NAMES[0]] = np.where(current_ready, carry, np.nan)
    result[NAMES[1]] = np.where(change_ready, change, np.nan)
    result["funding_event_count_24h"] = current_count
    result["funding_previous_event_count_24h"] = previous_count
    result["funding_latest_event_ms"] = [
        times[position - 1] if position else np.nan for position in right
    ]
    return result.replace([np.inf, -np.inf], np.nan)
