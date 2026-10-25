"""收盘动量，单位百分数：100 × (C[t] / C[t-window] - 1)。"""

import pandas as pd


def calculate(frame: pd.DataFrame, window: int) -> pd.Series:
    return frame["close"].pct_change(window, fill_method=None) * 100.0
