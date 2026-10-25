"""
OBV量价确认 - v3纯顺势量价确认因子

完全去掉背离逆势信号(v2背离信号胜率仅6.8%反向了)。
只做量价齐升确认多头、量价齐跌确认空头。
要求价格创20周期新高/新低+OBV同步创新高/新低+成交量异常放大。
纯顺势，不做反转。
"""
import pandas as pd
import numpy as np


def calculate_factor(df: pd.DataFrame) -> pd.Series:
    close = df['close'].astype('float64')
    volume = df['volume'].astype('float64')
    
    # OBV计算
    price_change = close.diff()
    obv_dir = np.sign(price_change).fillna(0)
    obv = (volume * obv_dir).cumsum()
    
    # 价格和OBV是否创20周期新高/新低(真正的突破)
    lookback = 20
    price_new_high = close >= close.rolling(lookback, min_periods=lookback).max()
    price_new_low = close <= close.rolling(lookback, min_periods=lookback).min()
    obv_new_high = obv >= obv.rolling(lookback, min_periods=lookback).max()
    obv_new_low = obv <= obv.rolling(lookback, min_periods=lookback).min()
    
    # 成交量异常放大(必须显著放量)
    vol_mean = volume.rolling(20, min_periods=5).mean()
    vol_std = volume.rolling(20, min_periods=5).std()
    vol_z = (volume - vol_mean) / (vol_std + 1e-10)
    vol_spike = vol_z > 1.0  # 成交量>1个标准差(较严格)
    
    # 价格动量确认(方向一致)
    ret5 = close.pct_change(5, fill_method=None).fillna(0)
    ret10 = close.pct_change(10, fill_method=None).fillna(0)
    strong_up = (ret5 > 0.005) & (ret10 > 0.005)  # 短期和中期都在涨
    strong_down = (ret5 < -0.005) & (ret10 < -0.005)
    
    result = pd.Series(0.0, index=df.index, dtype='float64')
    
    # 量价齐升确认(纯顺势多头)：价格新高 + OBV新高 + 放量 + 动量确认
    bull_confirm = price_new_high & obv_new_high & vol_spike & strong_up
    result[bull_confirm] = 0.7
    
    # 量价齐跌确认(纯顺势空头)：价格新低 + OBV新低 + 放量 + 动量确认
    bear_confirm = price_new_low & obv_new_low & vol_spike & strong_down
    result[bear_confirm] = -0.7
    
    result = result.fillna(0).astype('float64')
    return result
