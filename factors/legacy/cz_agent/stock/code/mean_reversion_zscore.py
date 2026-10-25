"""
均值回归Z-Score因子 - 港美股

价格偏离短期均值过远时做反向交易。
- 使用布林带Z-Score衡量偏离
- ATR标准化避免高波动股误判
- 成交量萎缩时反转概率更高
- 连续型信号，交易频率高
"""
import pandas as pd
import numpy as np


def calculate_factor(df: pd.DataFrame) -> pd.Series:
    close = df['close'].astype('float64')
    high = df['high'].astype('float64')
    low = df['low'].astype('float64')
    volume = df['volume'].astype('float64')
    n = len(close)

    if n < 40:
        return pd.Series(0.0, index=df.index, dtype='float64')

    # 价格Z-Score (偏离20日均线)
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    zscore = (close - ma20) / (std20 + 1e-10)

    # 反转信号: 偏高做空, 偏低做多
    revert_signal = -np.clip(zscore / 2, -1, 1)

    # ATR标准化的偏离度
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14, min_periods=5).mean()
    price_dev = (close - ma20) / (atr14 + 1e-10)
    atr_signal = -np.clip(price_dev / 3, -1, 1)

    # 速度: 3日涨跌幅过快则反转概率高
    ret_3 = close.pct_change(3, fill_method=None)
    speed_signal = -np.clip(ret_3 / 0.05, -1, 1)

    # 成交量: 缩量反转更可靠
    vol_ma20 = volume.rolling(20, min_periods=5).mean()
    vol_ratio = volume / (vol_ma20 + 1e-10)
    vol_factor = np.clip(1.5 - vol_ratio, 0.5, 1.3)

    # RSI辅助
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(7).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(7).mean()
    rs = gain / (loss + 1e-10)
    rsi7 = 100 - (100 / (1 + rs))
    rsi_signal = -(rsi7 - 50) / 35
    rsi_signal = np.clip(rsi_signal, -1, 1)

    raw = (revert_signal * 0.30 + atr_signal * 0.25 + speed_signal * 0.20 + rsi_signal * 0.25) * vol_factor

    raw_mean = raw.rolling(30, min_periods=10).mean()
    raw_std = raw.rolling(30, min_periods=10).std()
    final_z = (raw - raw_mean) / (raw_std + 1e-8)
    result = np.clip(final_z / 2, -1, 1)

    return pd.Series(result, index=df.index).fillna(0).astype('float64')
