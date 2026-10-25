"""价格距区间极值：100×(C/滚动最高价或最低价-1)，单位百分数。"""

import pandas as pd


def calculate(frame: pd.DataFrame, window: int, side: str) -> pd.Series:
    if side == "high":
        extreme = frame["high"].rolling(window, min_periods=window).max()
    elif side == "low":
        extreme = frame["low"].rolling(window, min_periods=window).min()
    else:
        raise ValueError("range side must be high or low")
    return (frame["close"] / extreme - 1.0) * 100.0
