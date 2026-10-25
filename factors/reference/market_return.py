"""市场中心收益和收益离散度：同一已收盘截面的中位数或样本标准差，单位百分数。"""

import pandas as pd


def median(values: pd.Series) -> pd.Series:
    return values.groupby(level="bar_time", sort=False).transform("median")


def dispersion(values: pd.Series) -> pd.Series:
    return values.groupby(level="bar_time", sort=False).transform("std")
