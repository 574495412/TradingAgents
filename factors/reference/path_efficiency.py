"""路径效率：窗口总收益绝对值 / 单柱收益绝对值之和，截断到 [0,1]。"""

import numpy as np
import pandas as pd


def calculate(frame: pd.DataFrame, window: int = 20) -> pd.Series:
    returns = frame["close"].pct_change(fill_method=None) * 100.0
    path = returns.abs().rolling(window, min_periods=window).sum()
    displacement = frame["close"].pct_change(window, fill_method=None) * 100.0
    return (displacement.abs() / path.replace(0.0, np.nan)).clip(0.0, 1.0)
