"""已实现波动：单柱百分数收益的滚动样本标准差 (ddof=1)，不做年化。"""

import pandas as pd


def calculate(frame: pd.DataFrame, window: int) -> pd.Series:
    return (
        (frame["close"].pct_change(fill_method=None) * 100.0)
        .rolling(window, min_periods=window)
        .std()
    )
