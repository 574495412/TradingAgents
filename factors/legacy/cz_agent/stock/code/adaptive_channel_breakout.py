"""
自适应通道突破因子 v2 (增强版) - 港美股

v1→v2 改进 (参考加密系统盈利经验):
  1. 新增ADX/DI趋势过滤 (crypto核心: 无趋势不做)
  2. DI方向一致性确认 (突破方向需与DI一致)
  3. 通道突破+缩量→放量确认 (crypto核心模式)
  4. 波动率极端过滤: ATR>中位数2.5倍不做 (黑天鹅保护)
  5. EMA趋势间距门槛 (crypto核心)
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

    # Keltner Channel
    ema20 = close.ewm(span=20, adjust=False).mean()
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr20 = tr.rolling(20, min_periods=5).mean()
    kc_upper = ema20 + 2.0 * atr20
    kc_lower = ema20 - 2.0 * atr20
    kc_width = kc_upper - kc_lower + 1e-10
    kc_position = (close - kc_lower) / kc_width
    pos_signal = np.clip((kc_position - 0.5) / 0.35, -1, 1)

    # Donchian Channel (10日)
    dc_high = high.rolling(10).max()
    dc_low = low.rolling(10).min()
    dc_width = dc_high - dc_low + 1e-10
    dc_position = (close - dc_low) / dc_width
    dc_signal = np.clip((dc_position - 0.5) / 0.3, -1, 1)

    # 趋势强度: EMA斜率
    ema_slope = (ema20 - ema20.shift(5)) / (ema20.shift(5) + 1e-10)
    slope_signal = np.clip(ema_slope / 0.02, -1, 1)

    # 【v2新增】EMA趋势间距门槛 (crypto核心)
    ema60 = close.ewm(span=60, adjust=False).mean()
    ema_spread = (ema20 - ema60).abs() / (close + 1e-10)
    trend_established = np.where(ema_spread > 0.005, 1.0, 0.6)

    # 动量一致性
    mom_5 = close.pct_change(5, fill_method=None)
    mom_15 = close.pct_change(15, fill_method=None)
    mom_agree = np.sign(mom_5) * np.sign(mom_15)
    mom_strength = np.clip((mom_5.abs() + mom_15.abs()) / 0.06, 0, 1)
    consistency = mom_agree * mom_strength * 0.3

    # 【v2新增】ADX/DI (crypto核心)
    atr14 = tr.rolling(14, min_periods=1).mean()
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    plus_di = 100 * plus_dm.rolling(14, min_periods=1).mean() / (atr14 + 1e-10)
    minus_di = 100 * minus_dm.rolling(14, min_periods=1).mean() / (atr14 + 1e-10)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(14, min_periods=1).mean()

    adx_weight = np.clip((adx - 10) / 25, 0.3, 1.2)

    # DI方向一致性: 突破方向需与DI一致
    di_dir = np.sign(plus_di - minus_di)
    signal_dir = np.sign(pos_signal)
    di_confirm = np.where(di_dir == signal_dir, 1.0, 0.6)

    # 【v2新增】缩量→放量模式 (crypto核心)
    vol_ma20 = volume.rolling(20, min_periods=5).mean()
    vol_ratio = volume / (vol_ma20 + 1e-10)
    vol_prev3 = volume.shift(1).rolling(3, min_periods=1).mean()
    vol_contraction = vol_prev3 < vol_ma20 * 0.90
    vol_breakout = vol_contraction & (vol_ratio > 1.0)
    vol_boost = np.where(vol_breakout, 1.2, np.clip(vol_ratio / 1.5, 0.5, 1.2))

    vol_ma5 = volume.rolling(5, min_periods=2).mean()
    vol_trend = np.clip((vol_ma5 / (vol_ma20 + 1e-10) - 1) / 0.5, -0.5, 0.5)
    vol_directed = vol_trend * np.sign(mom_5)

    # 【v2新增】波动率极端过滤 (crypto核心)
    atr_pct = atr14 / (close + 1e-10)
    atr_pct_median = atr_pct.rolling(60, min_periods=20).median()
    volatility_ok = np.where(atr_pct < atr_pct_median * 2.5, 1.0, 0.3)

    raw = (pos_signal * 0.25 + dc_signal * 0.20 + slope_signal * 0.25 +
           pd.Series(consistency, index=df.index) * 0.15 +
           pd.Series(vol_directed, index=df.index) * 0.15)
    raw = raw * trend_established * adx_weight * di_confirm * vol_boost * volatility_ok

    raw_mean = raw.rolling(40, min_periods=15).mean()
    raw_std = raw.rolling(40, min_periods=15).std()
    zscore = (raw - raw_mean) / (raw_std + 1e-8)
    result = np.clip(zscore / 2.5, -1, 1)

    return pd.Series(result, index=df.index).fillna(0).astype('float64')
