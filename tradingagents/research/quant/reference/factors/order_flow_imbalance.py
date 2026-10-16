"""主动成交不平衡：最近 window 根 (2×主动买入比例-1) 的等权均值，范围 [-1,1]。"""

import pandas as pd


def calculate(buy_share: pd.Series, window: int) -> pd.Series:
    return (buy_share * 2.0 - 1.0).rolling(window, min_periods=window).mean()
