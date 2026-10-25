"""相对市场动量：自身百分数收益减去同时点市场中位数收益；非回归残差。"""

import pandas as pd

from .market_return import median


def calculate(values: pd.Series) -> pd.Series:
    return values - median(values)
