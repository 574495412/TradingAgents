"""
布林带动态因子 - 港美股

改进: 输出连续信号而非脉冲。
- 价格在布林带上轨附近 → 多头信号
- 价格在布林带下轨附近 → 空头信号
- 布林带宽度越窄(压缩)，突破后信号加权越大
- 成交量放大时信号增强
"""
import pandas as pd
import numpy as np


def calculate_factor(df: pd.DataFrame) -> pd.Series:
    close = df['close'].astype('float64')
    volume = df['volume'].astype('float64')
    n = len(close)

    if n < 60:
        return pd.Series(0.0, index=df.index, dtype='float64')

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    bb_width = (bb_upper - bb_lower) / (bb_mid + 1e-10)

    # %B 指标: 0=下轨, 0.5=中轨, 1=上轨
    pct_b = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)

    # 位置信号: %B > 0.8 做多, %B < 0.2 做空, 线性过渡
    position_signal = np.clip((pct_b - 0.5) / 0.3, -1, 1)

    # 压缩系数: 宽度越小，突破后信号越强
    bb_width_ma = bb_width.rolling(60, min_periods=20).mean()
    squeeze_ratio = bb_width_ma / (bb_width + 1e-10)
    squeeze_boost = np.clip(squeeze_ratio / 2, 0.5, 1.5)

    # 动量确认
    mom_5 = close.pct_change(5, fill_method=None)
    mom_signal = np.clip(mom_5 / 0.03, -1, 1)

    # 成交量加权
    vol_ma20 = volume.rolling(20, min_periods=5).mean()
    vol_ratio = volume / (vol_ma20 + 1e-10)
    vol_weight = np.clip(vol_ratio / 1.5, 0.5, 1.5)

    # 趋势过滤: 20日均线斜率
    ma20_slope = (bb_mid - bb_mid.shift(5)) / (bb_mid.shift(5) + 1e-10)
    trend_signal = np.clip(ma20_slope / 0.02, -1, 1)

    raw = (position_signal * 0.35 + mom_signal * 0.30 + trend_signal * 0.35) * squeeze_boost * vol_weight

    # 标准化
    raw_mean = pd.Series(raw).rolling(50, min_periods=20).mean()
    raw_std = pd.Series(raw).rolling(50, min_periods=20).std()
    zscore = (raw - raw_mean) / (raw_std + 1e-8)
    result = np.clip(zscore / 2.5, -1, 1)

    return pd.Series(result, index=df.index).fillna(0).astype('float64')
