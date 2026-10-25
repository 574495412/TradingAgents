"""
SAR趋势信号 - v2脉冲式SAR趋势反转因子

只在SAR发生翻转(从做多变做空或反之)时输出脉冲信号，
配合成交量放大和价格动量确认。
修复v1连续smoothed tanh输出导致信号被稀释为0交易的问题。
"""
import pandas as pd
import numpy as np


def calculate_factor(df: pd.DataFrame) -> pd.Series:
    """
    v2脉冲式SAR趋势反转因子。只在SAR发生翻转(趋势反转点)时
    输出脉冲信号，而非连续偏离值。翻转+成交量确认+价格动量确认。
    """
    high = df['high'].values.astype('float64')
    low = df['low'].values.astype('float64')
    close = df['close'].values.astype('float64')
    volume = df['volume'].astype('float64')
    n = len(close)
    
    # SAR计算
    initial_af = 0.02
    af_increment = 0.02
    max_af = 0.2
    sar = np.full(n, np.nan)
    trend = np.full(n, 0)
    ep = np.full(n, np.nan)
    af_arr = np.full(n, np.nan)
    
    sar[0] = low[0]
    trend[0] = 1
    ep[0] = high[0]
    af_arr[0] = initial_af
    
    if n < 3:
        return pd.Series(0.0, index=df.index, dtype='float64')
    
    if high[1] > ep[0]:
        ep[1] = high[1]
        af_arr[1] = min(af_arr[0] + af_increment, max_af)
    else:
        ep[1] = ep[0]
        af_arr[1] = af_arr[0]
    trend[1] = trend[0]
    sar[1] = sar[0] + af_arr[0] * (ep[0] - sar[0])
    if trend[0] == 1:
        sar[1] = min(sar[1], low[0], low[1])
    else:
        sar[1] = max(sar[1], high[0], high[1])
    
    for i in range(2, n):
        if trend[i-1] == 1:
            if low[i] < sar[i-1]:
                trend[i] = -1
                sar[i] = ep[i-1]
                ep[i] = low[i]
                af_arr[i] = initial_af
            else:
                trend[i] = 1
                ep[i] = max(ep[i-1], high[i])
                af_arr[i] = min(af_arr[i-1] + af_increment, max_af) if ep[i] > ep[i-1] else af_arr[i-1]
                sar[i] = sar[i-1] + af_arr[i] * (ep[i] - sar[i-1])
                sar[i] = min(sar[i], low[i-1], low[i-2])
        else:
            if high[i] > sar[i-1]:
                trend[i] = 1
                sar[i] = ep[i-1]
                ep[i] = high[i]
                af_arr[i] = initial_af
            else:
                trend[i] = -1
                ep[i] = min(ep[i-1], low[i])
                af_arr[i] = min(af_arr[i-1] + af_increment, max_af) if ep[i] < ep[i-1] else af_arr[i-1]
                sar[i] = sar[i-1] + af_arr[i] * (ep[i] - sar[i-1])
                sar[i] = max(sar[i], high[i-1], high[i-2])
    
    # 检测SAR翻转(脉冲事件)
    trend_s = pd.Series(trend, index=df.index)
    sar_flip_bull = (trend_s == 1) & (trend_s.shift(1) == -1)  # 空转多
    sar_flip_bear = (trend_s == -1) & (trend_s.shift(1) == 1)  # 多转空
    
    # 成交量确认
    vol_mean = volume.rolling(20, min_periods=5).mean()
    vol_ratio = volume / (vol_mean + 1e-10)
    vol_confirm = vol_ratio > 1.2
    
    # 价格动量确认(3周期方向一致)
    close_s = pd.Series(close, index=df.index)
    ret3 = close_s.pct_change(3).fillna(0)
    mom_up = ret3 > 0.002
    mom_down = ret3 < -0.002
    
    # SAR偏离度(翻转后价格离SAR越远越好)
    sar_s = pd.Series(sar, index=df.index)
    sar_dist = (close_s - sar_s).abs() / (sar_s + 1e-10)
    dist_ok = sar_dist > 0.001  # 至少0.1%偏离
    
    result = pd.Series(0.0, index=df.index, dtype='float64')
    
    # 多头脉冲：SAR翻转做多 + (放量或动量确认) + 偏离度
    bull = sar_flip_bull & (vol_confirm | mom_up) & dist_ok
    bear = sar_flip_bear & (vol_confirm | mom_down) & dist_ok
    
    result[bull] = 0.7
    result[bear] = -0.7
    
    result = result.fillna(0).astype('float64')
    return result
