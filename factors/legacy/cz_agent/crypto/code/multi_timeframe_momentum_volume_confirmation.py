"""
动量成交量均线因子 - 多时间框架动量共振与成交量确认复合因子

结合快(20h)、中(50h)、慢(100h)三个周期的价格动量，
要求快慢周期动量同向且快周期更强，同时成交量Z-score > 1进行确认，
最后用波动率标准化信号强度。旨在捕捉高置信度的趋势启动点，过滤假突破。
"""
import pandas as pd
import numpy as np


def calculate_factor(df: pd.DataFrame) -> pd.Series:
    """
    多时间框架动量共振与成交量确认复合因子。
    结合快(20h)、中(50h)、慢(100h)三个周期的价格动量，
    要求快慢周期动量同向且快周期更强，同时成交量Z-score > 1进行确认，
    最后用波动率标准化信号强度。
    """
    # 超参数
    fast_period = 20
    mid_period = 50
    slow_period = 100
    volume_window = 20
    volatility_window = 20
    zscore_window = 100
    clip_bound = 3
    
    # 1. 计算快、中、慢周期动量 (使用 fill_method=None 避免警告)
    mom_fast = df['close'].pct_change(fast_period, fill_method=None)
    mom_mid = df['close'].pct_change(mid_period, fill_method=None)
    mom_slow = df['close'].pct_change(slow_period, fill_method=None)
    
    # 2. 计算成交量Z-score
    volume_mean = df['volume'].rolling(volume_window).mean()
    volume_std = df['volume'].rolling(volume_window).std()
    volume_z = (df['volume'] - volume_mean) / (volume_std + 1e-8)
    
    # 3. 动量共振条件
    cond_resonance = (mom_fast * mom_slow > 0) & (np.abs(mom_fast) > np.abs(mom_slow) * 0.8)
    
    # 4. 成交量确认条件
    cond_volume = volume_z > 1
    
    # 5. 原始信号
    raw_signal = np.where(cond_resonance & cond_volume, mom_fast, 0.0)
    
    # 6. 波动率标准化
    returns = df['close'].pct_change(fill_method=None)
    vol = returns.rolling(volatility_window).std()
    adj_signal = raw_signal / (vol + 1e-8)
    
    # 7. Z-score标准化
    signal_mean = adj_signal.rolling(zscore_window).mean()
    signal_std = adj_signal.rolling(zscore_window).std()
    signal_z = (adj_signal - signal_mean) / (signal_std + 1e-8)
    
    # 8. 裁剪并缩放
    clipped = np.clip(signal_z, -clip_bound, clip_bound)
    final_signal = clipped / clip_bound
    
    # 转换为 Series
    result = pd.Series(final_signal, index=df.index, dtype='float64')
    
    # 处理 NaN
    result = result.ffill().fillna(0)
    
    # 确保返回 Series
    if isinstance(result, pd.DataFrame):
        result = result.iloc[:, 0] if result.shape[1] > 0 else pd.Series(dtype='float64', index=df.index)
    
    return result.astype('float64')
