"""
EMA趋势强度 - v3脉冲式趋势因子

仅在EMA8/21/55完美排列+ADX>30+价格刚穿越EMA8(不用above_3)+成交量放大确认时才发信号。
修复v2的above_3导致频繁触发问题。
"""
import pandas as pd
import numpy as np


def calculate_factor(df: pd.DataFrame) -> pd.Series:
    close = df['close'].astype('float64')
    high = df['high'].astype('float64')
    low = df['low'].astype('float64')
    volume = df['volume'].astype('float64')
    
    # EMA三线
    ema8 = close.ewm(span=8, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema55 = close.ewm(span=55, adjust=False).mean()
    
    # ADX计算
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14, min_periods=1).mean()
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    plus_di = 100 * plus_dm.rolling(14, min_periods=1).mean() / (atr14 + 1e-10)
    minus_di = 100 * minus_dm.rolling(14, min_periods=1).mean() / (atr14 + 1e-10)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(14, min_periods=1).mean()
    
    # 完美排列(必须三线顺序排列)
    bull_perfect = (ema8 > ema21) & (ema21 > ema55)
    bear_perfect = (ema8 < ema21) & (ema21 < ema55)
    
    # 只用价格穿越EMA8(真正的脉冲事件，不用above_3)
    price_cross_above = (close > ema8) & (close.shift(1) <= ema8.shift(1))
    price_cross_below = (close < ema8) & (close.shift(1) >= ema8.shift(1))
    
    # ADX强趋势(提高到30，更严格)
    adx_strong = adx > 30
    
    # DI方向确认：多头时+DI>-DI，空头时-DI>+DI
    di_bull = plus_di > minus_di
    di_bear = minus_di > plus_di
    
    # 成交量放大确认
    vol_mean = volume.rolling(20, min_periods=5).mean()
    vol_ratio = volume / (vol_mean + 1e-10)
    vol_elevated = vol_ratio > 1.2
    
    # 价格动量确认(5周期涨跌方向一致)
    ret5 = close.pct_change(5, fill_method=None).fillna(0)
    mom_up = ret5 > 0.003
    mom_down = ret5 < -0.003
    
    result = pd.Series(0.0, index=df.index, dtype='float64')
    
    # 多头脉冲：完美排列 + ADX>30 + 穿越EMA8 + DI确认 + (放量 或 强动量)
    bull_trigger = bull_perfect & adx_strong & price_cross_above & di_bull & (vol_elevated | mom_up)
    bear_trigger = bear_perfect & adx_strong & price_cross_below & di_bear & (vol_elevated | mom_down)
    
    # 信号强度固定0.7(避免复杂计算引入噪音)
    result[bull_trigger] = 0.7
    result[bear_trigger] = -0.7
    
    result = result.fillna(0).astype('float64')
    return result
