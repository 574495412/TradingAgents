"""原生研究假说 H01：普通资金费结算后的固定短时退出压力。"""

from collections.abc import Mapping
from decimal import Decimal, localcontext
from itertools import pairwise
from numbers import Real

import numpy as np
import pandas as pd

from .funding import DAY, LAG, _integer_ms

H01_NAME = "settlement_unwind_h01"
MINUTE = 60_000
DECAY = 60 * MINUTE


def calculate(clocks, events, coverage_start_ms, coverage_end_ms):
    """只使用五分钟前已结算的最近普通事件，已知零与未知严格区分。

    输入事件须由调用者筛选为普通结算。覆盖两端均包含，完整窗口为
    [t-五分钟-四十八小时,t-五分钟]；窗口无事件时最近事件必已过衰减期。
    """
    first = _integer_ms(coverage_start_ms, "覆盖起点")
    last = _integer_ms(coverage_end_ms, "覆盖终点")
    if first > last:
        raise ValueError("覆盖起点不能晚于终点")
    index = pd.Index(clocks)
    stamps = [_integer_ms(value, "研究时钟") for value in index]
    if any(b <= a for a, b in pairwise(stamps)):
        raise ValueError("研究时钟必须唯一且递增")
    normalized, symbols = [], set()
    for event in events:
        if not isinstance(event, Mapping):
            raise ValueError("结算事件必须为规范化字典")
        symbol = event.get("symbol")
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("结算事件缺少合约")
        symbols.add(symbol)
        clock = _integer_ms(event.get("time_ms"), "事件时间")
        rate = event.get("rate")
        if (
            not first <= clock <= last
            or isinstance(rate, (bool, np.bool_))
            or not isinstance(rate, Real)
            or not np.isfinite(float(rate))
        ):
            raise ValueError("结算事件越界或费率不是有限数字")
        normalized.append((clock, Decimal(str(rate))))
    if len(symbols) > 1:
        raise ValueError("一次只能计算单一合约")
    normalized.sort(key=lambda value: value[0])
    if any(a[0] == b[0] for a, b in pairwise(normalized)):
        raise ValueError("普通结算事件时钟重复或冲突")
    times = np.asarray([v[0] for v in normalized], dtype=object)
    result = []
    for clock in stamps:
        cut = clock - LAG
        end = int(np.searchsorted(times, cut, side="right"))
        start = int(np.searchsorted(times, cut - 2 * DAY, side="left"))
        ready = cut - 2 * DAY >= first and cut <= last
        stamp = normalized[end - 1][0] if end else None
        age = (clock - stamp) / MINUTE if stamp is not None else np.nan
        value = np.nan
        if ready:
            remaining = DECAY - (clock - stamp - LAG) if stamp is not None else 0
            if remaining <= 0:
                value = 0.0
            else:
                # 不在窗口前累计费率，极端未来事件不能改变当前十进制精度。
                rate = normalized[end - 1][1]
                with localcontext() as context:
                    context.prec = max(40, len(rate.as_tuple().digits) + 24)
                    context.Emax = max(999999, rate.adjusted() + 24)
                    context.Emin = min(-999999, rate.as_tuple().exponent - 24)
                    value = float(rate * Decimal(remaining) / Decimal(DECAY))
                if not np.isfinite(value):
                    value = np.nan
        result.append(
            (
                value,
                stamp if ready and stamp is not None else np.nan,
                age if ready else np.nan,
                end - start,
            )
        )
    return pd.DataFrame(
        result,
        index=index,
        columns=[
            H01_NAME,
            "settlement_latest_event_ms",
            "settlement_age_minutes",
            "settlement_event_count_48h",
        ],
    )
