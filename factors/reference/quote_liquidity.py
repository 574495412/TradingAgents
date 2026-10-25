"""成交额流动性：最近 window 根收盘柱的计价币成交额之和。"""

import pandas as pd


def calculate(frame: pd.DataFrame, window: int = 12) -> pd.Series:
    return frame["quote_volume"].rolling(window, min_periods=window).sum()
