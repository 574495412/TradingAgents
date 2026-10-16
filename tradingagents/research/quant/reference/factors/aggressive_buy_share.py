"""主动买入比例：主动买入量 / 总成交量，范围 [0,1]；无流量证据保留 NaN。

优先使用计价币成交额，其次使用同单位基础币成交量，最后使用明确提供的比例。
这是公共数据单位适配，不把缺失的交易流猜成买入信号。
"""

import numpy as np
import pandas as pd


def calculate(frame: pd.DataFrame) -> pd.Series:
    if "taker_buy_quote" in frame and frame["taker_buy_quote"].notna().any():
        ratio = frame["taker_buy_quote"] / frame["quote_volume"].replace(0.0, np.nan)
    elif "taker_buy_base" in frame:
        ratio = frame["taker_buy_base"] / frame["volume"].replace(0.0, np.nan)
    elif "taker_buy_ratio" in frame:
        ratio = frame["taker_buy_ratio"]
    else:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    if not ratio.dropna().between(0.0, 1.0).all():
        raise ValueError("taker buy ratio must be in [0, 1]")
    return ratio.astype(float)
