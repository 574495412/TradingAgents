"""
RSI动态反转因子 - 港美股

改进: 从脉冲型(几乎不触发)改为连续型。
- RSI偏离中轴(50)越远，反转期望越强
- 价格动量衰减时信号增强
- 成交量异常(放量/缩量)提供额外信息
"""
import pandas as pd
import numpy as np


def calculate_factor(df: pd.DataFrame) -> pd.Series:
    close = df['close'].astype('float64')
    high = df['high'].astype('float64')
    low = df['low'].astype('float64')
    volume = df['volume'].astype('float64')
    n = len(close)

    if n < 60:
        return pd.Series(0.0, index=df.index, dtype='float64')

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # RSI信号: 超卖做多, 超买做空 (连续型)
    # RSI < 30 → 强做多, RSI > 70 → 强做空
    rsi_signal = -(rsi - 50) / 30  # RSI=30 → +0.67, RSI=70 → -0.67
    rsi_signal = np.clip(rsi_signal, -1, 1)

    # RSI变化率 (动量衰减检测)
    rsi_change = rsi - rsi.shift(3)
    # 超买区RSI开始下降 → 做空信号增强; 超卖区RSI开始上升 → 做多信号增强
    rsi_momentum = np.where(
        rsi > 60, np.clip(-rsi_change / 10, 0, 0.5),
        np.where(rsi < 40, np.clip(rsi_change / 10, 0, 0.5), 0)
    )
    rsi_mom_directed = np.where(rsi > 50, -rsi_momentum, rsi_momentum)

    # 价格动量
    mom_5 = close.pct_change(5, fill_method=None)
    mom_10 = close.pct_change(10, fill_method=None)
    # 动量衰减: 短期动量 vs 长期动量方向不一致
    mom_decel = np.where(
        (mom_10 > 0.01) & (mom_5 < mom_10 * 0.5), -0.3,
        np.where((mom_10 < -0.01) & (mom_5 > mom_10 * 0.5), 0.3, 0)
    )

    # 成交量异动
    vol_ma20 = volume.rolling(20, min_periods=5).mean()
    vol_ratio = volume / (vol_ma20 + 1e-10)
    vol_extreme = np.clip((vol_ratio - 1) / 2, -0.3, 0.3)

    raw = (rsi_signal * 0.45 +
           pd.Series(rsi_mom_directed, index=df.index) * 0.25 +
           pd.Series(mom_decel, index=df.index) * 0.15 +
           pd.Series(vol_extreme, index=df.index) * 0.15)

    raw_mean = raw.rolling(50, min_periods=20).mean()
    raw_std = raw.rolling(50, min_periods=20).std()
    zscore = (raw - raw_mean) / (raw_std + 1e-8)
    result = np.clip(zscore / 2.5, -1, 1)

    return pd.Series(result, index=df.index).fillna(0).astype('float64')
