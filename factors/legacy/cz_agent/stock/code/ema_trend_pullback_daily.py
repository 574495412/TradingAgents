"""
EMA趋势回踩因子 (日线渐进版v3) - 港美股

v2→v3 改进 (参考加密系统盈利经验):
  1. 新增趋势间距门槛: EMA间距>0.5%才给趋势分 (crypto核心经验)
  2. 新增DI差值过滤: |+DI - -DI| > 3才确认方向
  3. ADX改为门槛+加分: ADX<15直接降权
  4. 回踩距离分级: 0-1%最佳, 1-3%中等, >3%减分
  5. 缩量→放量确认: 前3根缩量+当根放量加分 (crypto核心逻辑)
  6. K线反转形态加分: 锤子/吞没比单纯阳/阴线权重更高
"""
import pandas as pd
import numpy as np


def calculate_factor(df: pd.DataFrame) -> pd.Series:
    close = df['close'].astype('float64')
    high = df['high'].astype('float64')
    low = df['low'].astype('float64')
    open_ = df['open'].astype('float64')
    volume = df['volume'].astype('float64')
    n = len(close)

    if n < 80:
        return pd.Series(0.0, index=df.index, dtype='float64')

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema60 = close.ewm(span=60, adjust=False).mean()

    ema_spread = (ema20 - ema60) / (close + 1e-10)
    ema_spread_abs = ema_spread.abs()
    ema20_slope = (ema20 - ema20.shift(5)) / (ema20.shift(5) + 1e-10)

    pullback_dist = (close - ema20).abs() / (close + 1e-10)

    vol_ma20 = volume.rolling(20, min_periods=5).mean()
    vol_ratio = volume / (vol_ma20 + 1e-10)
    vol_prev3 = volume.shift(1).rolling(3, min_periods=1).mean()
    vol_contraction = vol_prev3 < vol_ma20 * 0.90

    body = close - open_
    body_abs = body.abs()
    candle_range = high - low + 1e-10
    body_ratio = body_abs / candle_range
    upper_shadow = high - pd.concat([close, open_], axis=1).max(axis=1)
    lower_shadow = pd.concat([close, open_], axis=1).min(axis=1) - low

    # ADX/DI (加密系统核心指标)
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
    di_spread = (plus_di - minus_di).abs()

    # ===== 渐进评分 =====
    bull_score = pd.Series(0.0, index=df.index)
    bear_score = pd.Series(0.0, index=df.index)

    # 1. 趋势方向 (0.20) - 需要趋势间距>0.5%
    trend_valid = ema_spread_abs > 0.005
    bull_score += np.where((ema20 > ema60) & trend_valid, 0.20, 0)
    bear_score += np.where((ema20 < ema60) & trend_valid, 0.20, 0)

    # 2. 趋势强度 (0.10)
    bull_score += np.clip(ema_spread / 0.02, 0, 0.10)
    bear_score += np.clip(-ema_spread / 0.02, 0, 0.10)

    # 3. EMA斜率 (0.10)
    bull_score += np.where(ema20_slope > 0.002, 0.10, 0)
    bear_score += np.where(ema20_slope < -0.002, 0.10, 0)

    # 4. 回踩距离分级 (0.15)
    in_ema_zone = pullback_dist < 0.03
    optimal_pullback = pullback_dist < 0.01
    bull_score += np.where(in_ema_zone & (close > ema20 * 0.99), np.where(optimal_pullback, 0.15, 0.10), 0)
    bear_score += np.where(in_ema_zone & (close < ema20 * 1.01), np.where(optimal_pullback, 0.15, 0.10), 0)

    # 5. 缩量→放量确认 (0.15) - crypto核心逻辑
    vol_pattern = vol_contraction & (vol_ratio > 0.85)
    bull_score += np.where(vol_pattern, 0.15, np.where(vol_ratio > 1.0, 0.05, 0))
    bear_score += np.where(vol_pattern, 0.15, np.where(vol_ratio > 1.0, 0.05, 0))

    # 6. ADX趋势确认 (0.10)
    adx_boost = np.where(adx > 25, 0.10, np.where(adx > 15, 0.05, -0.05))
    bull_score += adx_boost
    bear_score += adx_boost

    # 7. DI方向+强度 (0.10) - crypto核心
    bull_score += np.where((plus_di > minus_di) & (di_spread > 3), 0.10,
                           np.where(plus_di > minus_di, 0.05, 0))
    bear_score += np.where((minus_di > plus_di) & (di_spread > 3), 0.10,
                           np.where(minus_di > plus_di, 0.05, 0))

    # 8. K线形态 (0.10)
    hammer = (lower_shadow >= body_abs * 2) & (body > 0) & (body_ratio > 0.15)
    bullish_engulf = (body > 0) & (body.shift(1) < 0) & (close > open_.shift(1))
    bull_reversal = hammer | bullish_engulf
    bull_score += np.where(bull_reversal, 0.10, np.where(body > 0, 0.03, 0))

    shooting_star = (upper_shadow >= body_abs * 2) & (body < 0) & (body_ratio > 0.15)
    bearish_engulf = (body < 0) & (body.shift(1) > 0) & (close < open_.shift(1))
    bear_reversal = shooting_star | bearish_engulf
    bear_score += np.where(bear_reversal, 0.10, np.where(body < 0, 0.03, 0))

    result = bull_score - bear_score
    result = np.clip(result, -1, 1)

    return pd.Series(result, index=df.index).fillna(0).astype('float64')
