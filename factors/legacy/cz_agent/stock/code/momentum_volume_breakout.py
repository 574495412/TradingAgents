"""
动量成交量因子 v3 (增强版) - 港美股

v2→v3 改进 (参考加密系统盈利经验):
  1. 新增ADX趋势过滤: ADX<15时自动降权 (crypto核心: 无趋势不做)
  2. 新增DI方向一致性: 动量方向需与DI方向一致 (方向确认)
  3. 缩量→放量模式检测: 前3根缩量+当根放量时额外加分
  4. 动量衰减检测: 快动量弱于慢动量时减弱信号 (crypto经验)
  5. 波动率自适应: 高波动降权 (crypto核心)
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

    # 多周期动量
    mom_5 = close.pct_change(5, fill_method=None)
    mom_10 = close.pct_change(10, fill_method=None)
    mom_20 = close.pct_change(20, fill_method=None)

    # 动量综合 (加权)
    mom_score = mom_5 * 0.5 + mom_10 * 0.3 + mom_20 * 0.2

    # ATR标准化动量
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr20 = tr.rolling(20, min_periods=5).mean()
    atr_pct = atr20 / (close + 1e-10)
    mom_norm = mom_score / (atr_pct * 5 + 1e-8)
    mom_signal = np.clip(mom_norm, -2, 2) / 2

    # 【v3新增】动量共振检测: 快慢动量同向且快>慢 (crypto核心逻辑)
    mom_aligned = (np.sign(mom_5) == np.sign(mom_20)) & (mom_5.abs() > mom_20.abs() * 0.8)
    mom_boost = np.where(mom_aligned, 1.3, 0.8)  # 共振放大, 不共振缩小

    # 成交量趋势
    vol_ma5 = volume.rolling(5, min_periods=2).mean()
    vol_ma20 = volume.rolling(20, min_periods=5).mean()
    vol_trend = (vol_ma5 - vol_ma20) / (vol_ma20 + 1e-10)
    vol_signal = np.clip(vol_trend / 0.5, -1, 1)

    # 量价方向一致性
    price_dir = np.sign(mom_5)
    vol_dir_signal = vol_signal * price_dir * 0.3

    # 【v3新增】缩量→放量模式 (crypto核心)
    vol_ratio = volume / (vol_ma20 + 1e-10)
    vol_prev3 = volume.shift(1).rolling(3, min_periods=1).mean()
    vol_contraction = vol_prev3 < vol_ma20 * 0.90
    vol_pattern_boost = np.where(vol_contraction & (vol_ratio > 1.0), 0.15, 0)

    # 价格位置
    high_20 = high.rolling(20).max()
    low_20 = low.rolling(20).min()
    price_range = high_20 - low_20 + 1e-10
    price_pos = (close - low_20) / price_range
    pos_signal = np.clip((price_pos - 0.5) / 0.3, -1, 1)

    # 【v3新增】ADX/DI趋势过滤 (crypto核心)
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    atr14 = tr.rolling(14, min_periods=1).mean()
    plus_di = 100 * plus_dm.rolling(14, min_periods=1).mean() / (atr14 + 1e-10)
    minus_di = 100 * minus_dm.rolling(14, min_periods=1).mean() / (atr14 + 1e-10)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(14, min_periods=1).mean()

    # ADX权重: <15惩罚, >25加强
    adx_weight = np.clip((adx - 10) / 25, 0.3, 1.2)

    # DI方向一致性: 动量方向需与DI方向一致
    di_dir = np.sign(plus_di - minus_di)
    mom_dir = np.sign(mom_score)
    di_confirm = np.where(di_dir == mom_dir, 1.0, 0.6)

    # 【v3新增】波动率自适应 (crypto核心)
    returns = close.pct_change(fill_method=None)
    vol_20d = returns.rolling(20, min_periods=5).std()
    vol_60d = returns.rolling(60, min_periods=20).std()
    vol_regime = vol_20d / (vol_60d + 1e-10)
    vol_penalty = np.clip(1.5 - vol_regime, 0.5, 1.2)

    raw = (mom_signal * mom_boost * 0.40 + pos_signal * 0.20 +
           pd.Series(vol_dir_signal, index=df.index) * 0.25 +
           pd.Series(vol_pattern_boost, index=df.index) * price_dir)
    raw = raw * adx_weight * di_confirm * vol_penalty

    raw_mean = raw.rolling(50, min_periods=20).mean()
    raw_std = raw.rolling(50, min_periods=20).std()
    zscore = (raw - raw_mean) / (raw_std + 1e-8)
    result = np.clip(zscore / 2.5, -1, 1)

    return pd.Series(result, index=df.index).fillna(0).astype('float64')
