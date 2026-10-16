"""用历史价格与主动成交构造固定的价格响应因子，不读取未来标签。"""

import numpy as np
import pandas as pd

NAMES = ("impact_residual_fade_20m", "flow_path_concordance_20m", "skip_momentum_4h")


def calculate(minutes):
    """先用当前窗口之前的120分钟拟合冲击，再度量当前20分钟的偏离。"""
    if minutes.index.has_duplicates or (
        len(minutes) > 1 and not (np.diff(minutes.index) == 60_000).all()
    ):
        raise ValueError("响应因子必须保留唯一、连续分钟日历")
    log_price = np.log(minutes["close"])
    returns = log_price.diff()
    quote = minutes["quote_volume"]
    flow = 2 * minutes["taker_buy_quote"] / quote.where(quote > 0) - 1
    # 拟合区间止于 t-20，当前冲击不能进入其自身的正常响应估计。
    past_return, past_flow = returns.shift(20), flow.shift(20)
    pair_ready = past_return.notna() & past_flow.notna()
    x, y = past_flow.where(pair_ready), past_return.where(pair_ready)
    mean_x, mean_y = x.rolling(120, min_periods=120).mean(), y.rolling(120, min_periods=120).mean()
    variance = x.rolling(120, min_periods=120).var(ddof=1)
    beta = x.rolling(120, min_periods=120).cov(y) / variance.where(variance > 0)
    intercept = mean_y - beta * mean_x
    past_volatility = y.rolling(120, min_periods=120).std(ddof=1)
    current_flow = flow.rolling(20, min_periods=20).sum()
    current_return = returns.rolling(20, min_periods=20).sum()
    predicted_return = 20 * intercept + beta * current_flow
    result = pd.DataFrame(index=minutes.index)
    result[NAMES[0]] = -(current_return - predicted_return) / (
        past_volatility.where(past_volatility > 0) * np.sqrt(20)
    )
    path = returns.abs().rolling(20, min_periods=20).sum()
    flow_20 = (
        2
        * minutes["taker_buy_quote"].rolling(20, min_periods=20).sum()
        / quote.rolling(20, min_periods=20).sum().where(lambda v: v > 0)
        - 1
    )
    result[NAMES[1]] = flow_20 * current_return.abs() / path.where(path > 0)
    lagged_volatility = returns.shift(60).rolling(240, min_periods=240).std(ddof=1)
    result[NAMES[2]] = (log_price.shift(60) - log_price.shift(240)) / (
        lagged_volatility.where(lagged_volatility > 0) * np.sqrt(180)
    )
    return result.replace([np.inf, -np.inf], np.nan)
