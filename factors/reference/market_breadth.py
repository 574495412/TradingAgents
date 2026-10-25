"""市场广度：同一截面收益为正的标的占比；change 为相隔 window 个收盘截面的差。

沿用当前策略口径：截面中的 NaN 收益计入分母但不计入上涨数。时间窗口按观测柱计数。
"""

import pandas as pd


def calculate(values: pd.Series) -> pd.Series:
    return values.gt(0.0).groupby(level="bar_time", sort=False).transform("mean")


def change(breadth: pd.Series, window: int = 5) -> pd.Series:
    by_time = breadth.groupby(level="bar_time", sort=False).first().diff(window)
    return pd.Series(breadth.index.get_level_values("bar_time").map(by_time), index=breadth.index)
