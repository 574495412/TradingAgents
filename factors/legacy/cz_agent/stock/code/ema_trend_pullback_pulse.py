"""
EMA趋势回踩反转因子 (1小时脉冲版) - 港美股

直接复刻加密系统核心因子 ema_trend_pullback_adx (15m K线)
从 15m 适配为 Min_60（与加密同量级 bar 密度，避免日线仅 ~21 根交易 K 无法触发）

核心逻辑 (7重AND链脉冲信号):
  1. EMA20/60趋势方向 + 趋势间距过滤 (趋势已确立)
  2. EMA20斜率确认 (趋势有动量)
  3. 价格回踩到EMA20附近 + 收盘确认 (支撑/阻力有效)
  4. 前3根缩量 + 当前放量 (缩量回踩→放量反转)
  5. K线形态反转确认 (锤子/吞没/阳线收盘上半区)
  6. ADX > 动态阈值 (趋势强度足够)
  7. DI方向 + DI差值 > 5 (趋势方向动量足够)

与加密 15m 的关键差异 (适配 1h，约 4× 时间尺度):
  - 趋势间距: 0.35% → 0.5%
  - EMA斜率(5根): 0.12% → 0.15%
  - 回踩容差带: 0.8% → 1.2%
  - ADX基础: 25 → 22; 高波动: 30 → 27
  - 缩量/放量与加密一致: 85% / 85%
  - 收盘确认: 0.2% → 0.3% 误差
"""
import pandas as pd
import numpy as np


def calculate_factor(df: pd.DataFrame) -> pd.Series:
    """
    EMA趋势回踩反转因子 (1小时脉冲版)

    复刻加密系统核心盈利因子, 适配港美股小时 K 线
    输出: +0.7(做多脉冲) / -0.7(做空脉冲) / 0(无信号)
    """
    close = df['close'].astype('float64')
    high = df['high'].astype('float64')
    low = df['low'].astype('float64')
    open_ = df['open'].astype('float64')
    volume = df['volume'].astype('float64')
    n = len(close)

    if n < 80:
        return pd.Series(0.0, index=df.index, dtype='float64')

    # ===== 条件1: EMA趋势方向 + 趋势间距过滤 =====
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema60 = close.ewm(span=60, adjust=False).mean()
    uptrend = ema20 > ema60
    downtrend = ema20 < ema60

    ema_spread = (ema20 - ema60).abs() / (close + 1e-10)
    trend_established = ema_spread > 0.005  # 0.5%（15m 加密 0.35% → 1h 略放宽）

    ema20_slope = (ema20 - ema20.shift(5)) / (ema20.shift(5) + 1e-10)
    ema_rising = ema20_slope > 0.0015   # 0.15% / 5 根 1h
    ema_falling = ema20_slope < -0.0015

    # ===== 条件2: 价格回踩/反弹到EMA20附近 + 收盘确认 =====
    ema20_band = ema20 * 0.012  # 1.2% 容差带（15m 0.8% → 1h）

    pullback_to_ema_bull = (low <= ema20 + ema20_band) & (low >= ema20 - ema20_band * 0.4)
    pullback_to_ema_bear = (high >= ema20 - ema20_band) & (high <= ema20 + ema20_band * 0.4)

    close_above_ema = close > ema20 * 0.997
    close_below_ema = close < ema20 * 1.003

    # ===== 条件3: 回踩前3根缩量 + 当前放量确认 =====
    vol_ma20 = volume.rolling(20, min_periods=5).mean()
    vol_prev3 = volume.shift(1).rolling(3, min_periods=1).mean()
    vol_contraction = vol_prev3 < vol_ma20 * 0.85

    vol_ratio = volume / (vol_ma20 + 1e-10)
    vol_current_surge = vol_ratio > 0.85

    # ===== 条件4: 反转确认 (K线形态) =====
    body = close - open_
    body_abs = body.abs()
    candle_range = high - low + 1e-10
    body_ratio = body_abs / candle_range
    upper_shadow = high - pd.concat([close, open_], axis=1).max(axis=1)
    lower_shadow = pd.concat([close, open_], axis=1).min(axis=1) - low

    hammer = ((lower_shadow >= body_abs * 2) & (upper_shadow < body_abs) &
              (body > 0) & (body_ratio > 0.15))
    bullish_engulf = ((body > 0) & (body.shift(1) < 0) &
                      (close > open_.shift(1)) & (open_ < close.shift(1)))
    close_in_upper = (body > 0) & (close > (high + low) / 2) & (body_ratio > 0.3)
    bullish_reversal = hammer | bullish_engulf | close_in_upper

    shooting_star = ((upper_shadow >= body_abs * 2) & (lower_shadow < body_abs) &
                     (body < 0) & (body_ratio > 0.15))
    bearish_engulf = ((body < 0) & (body.shift(1) > 0) &
                      (open_ > close.shift(1)) & (close < open_.shift(1)))
    close_in_lower = (body < 0) & (close < (high + low) / 2) & (body_ratio > 0.3)
    bearish_reversal = shooting_star | bearish_engulf | close_in_lower

    # ===== 条件5: ADX + DI方向 + DI差值强度 =====
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14, min_periods=1).mean()

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    plus_di = 100 * plus_dm.rolling(14, min_periods=1).mean() / (atr14 + 1e-10)
    minus_di = 100 * minus_dm.rolling(14, min_periods=1).mean() / (atr14 + 1e-10)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(14, min_periods=1).mean()

    atr_pct = atr14 / (close + 1e-10)
    atr_pct_median = atr_pct.rolling(60, min_periods=20).median()
    high_volatility = atr_pct > atr_pct_median * 1.5
    adx_threshold = pd.Series(22.0, index=df.index)
    adx_threshold[high_volatility] = 27.0
    adx_ok = adx > adx_threshold

    di_bull = plus_di > minus_di
    di_bear = minus_di > plus_di

    di_spread = (plus_di - minus_di).abs()
    di_strong = di_spread > 5.0

    # ===== 综合信号 (7重AND链) =====
    result = pd.Series(0.0, index=df.index, dtype='float64')

    bull = (uptrend & trend_established & ema_rising &
            pullback_to_ema_bull & close_above_ema &
            vol_contraction & vol_current_surge &
            bullish_reversal & adx_ok & di_bull & di_strong)
    bear = (downtrend & trend_established & ema_falling &
            pullback_to_ema_bear & close_below_ema &
            vol_contraction & vol_current_surge &
            bearish_reversal & adx_ok & di_bear & di_strong)

    result[bull] = 0.7
    result[bear] = -0.7

    return result.fillna(0).astype('float64')
