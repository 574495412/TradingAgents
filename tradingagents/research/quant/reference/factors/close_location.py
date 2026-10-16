"""收盘位置：(C-L)/(H-L)，范围 [0,1]；零振幅柱定义为中性 0.5。"""

import numpy as np
import pandas as pd


def calculate(frame: pd.DataFrame) -> pd.Series:
    return (
        ((frame["close"] - frame["low"]) / (frame["high"] - frame["low"]).replace(0.0, np.nan))
        .clip(0.0, 1.0)
        .fillna(0.5)
    )
