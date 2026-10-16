"""真实波幅：TR 的 window 期简单均值 / 当前收盘价 × 100；非 Wilder 平滑。"""

import pandas as pd


def calculate(frame: pd.DataFrame, window: int = 14) -> pd.Series:
    previous = frame["close"].shift(1)
    tr = pd.concat(
        (
            frame["high"] - frame["low"],
            (frame["high"] - previous).abs(),
            (frame["low"] - previous).abs(),
        ),
        axis=1,
    ).max(axis=1)
    return tr.rolling(window, min_periods=window).mean() / frame["close"] * 100.0
