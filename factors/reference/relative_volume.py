"""放量倍数：当前成交额 / 前 window 根成交额中位数；基线不含当前柱。"""

import numpy as np
import pandas as pd


def calculate(frame: pd.DataFrame, window: int = 30, min_periods: int = 12) -> pd.Series:
    baseline = frame["quote_volume"].shift(1).rolling(window, min_periods=min_periods).median()
    return frame["quote_volume"] / baseline.replace(0.0, np.nan)
