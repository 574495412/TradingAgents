"""
VWAP偏离动量因子 - 港美股

使用成交量加权均价(VWAP)作为锚点:
- 价格持续在VWAP上方 → 买盘主导 → 做多
- 价格持续在VWAP下方 → 卖盘主导 → 做空
- 偏离程度+持续时间共同决定信号强度
- 适合捕捉日内/短期资金流向
"""
import pandas as pd
import numpy as np


def calculate_factor(df: pd.DataFrame) -> pd.Series:
    close = df['close'].astype('float64')
    high = df['high'].astype('float64')
    low = df['low'].astype('float64')
    volume = df['volume'].astype('float64')
    n = len(close)

    if n < 30:
        return pd.Series(0.0, index=df.index, dtype='float64')

    # VWAP近似: 典型价格 × 成交量的累积均值 (rolling 20)
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).rolling(20, min_periods=5).sum() / (volume.rolling(20, min_periods=5).sum() + 1e-10)

    # VWAP偏离度
    vwap_dev = (close - vwap) / (vwap + 1e-10)
    vwap_signal = np.clip(vwap_dev / 0.03, -1, 1)

    # 偏离持续性: 连续几根K线在VWAP同侧
    above_vwap = (close > vwap).astype(float)
    below_vwap = (close < vwap).astype(float)
    streak_above = above_vwap.rolling(5, min_periods=1).sum()
    streak_below = below_vwap.rolling(5, min_periods=1).sum()
    streak_signal = (streak_above - streak_below) / 5

    # OBV (On-Balance Volume) 趋势
    obv = (volume * np.sign(close.diff().fillna(0))).cumsum()
    obv_ma5 = obv.rolling(5, min_periods=2).mean()
    obv_ma20 = obv.rolling(20, min_periods=5).mean()
    obv_trend = (obv_ma5 - obv_ma20) / (obv_ma20.abs() + 1e-10)
    obv_signal = np.clip(obv_trend / 0.1, -1, 1)

    # 价格动量
    mom_3 = close.pct_change(3, fill_method=None)
    mom_10 = close.pct_change(10, fill_method=None)
    mom_signal = np.clip((mom_3 * 0.6 + mom_10 * 0.4) / 0.04, -1, 1)

    # 综合
    raw = vwap_signal * 0.30 + streak_signal * 0.20 + obv_signal * 0.25 + mom_signal * 0.25

    raw_mean = raw.rolling(40, min_periods=15).mean()
    raw_std = raw.rolling(40, min_periods=15).std()
    zscore = (raw - raw_mean) / (raw_std + 1e-8)
    result = np.clip(zscore / 2.5, -1, 1)

    return pd.Series(result, index=df.index).fillna(0).astype('float64')
