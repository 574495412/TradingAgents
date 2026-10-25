"""同时点横截面百分位排名：并列取平均秩，缺失值不参加排名，范围 (0,1]。"""

import pandas as pd


def calculate(values: pd.Series) -> pd.Series:
    return values.groupby(level="bar_time", sort=False).rank(method="average", pct=True)
