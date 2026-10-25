"""
EMA趋势缩量回踩反转因子 (小时线版) - 港美股

直接复刻加密系统 ema_shrink_pullback_4h (4h K线, 信号分级0.9/0.6)
从4h K线适配为1h K线 (港美股数据一般用小时线)

核心逻辑 (8重AND链脉冲信号):
  1. EMA20/60趋势方向 (多头/空头)
  2. 趋势间距 > 1.5% (趋势已充分展开, 4h用1.0%, 1h需更大)
  3. 价格回踩EMA20附近 (±3%) + 收盘确认 (0.3%误差)
  4. 前一根K线方向确认 (多头前阴=回踩中, 空头前阳=反弹中)
  5. 缩量回踩 (当前K线成交量 < 20均量)
  6. 经典反转K线形态 (锤子/吞没/十字星后突破) → 信号分级
  7. ADX > 20 (趋势强度足够)
  8. ATR极端过滤 (ATR% > 中位数×2.5 不入场)

与加密4h版本的关键差异 (适配1h):
  - 趋势间距: 1h 的 EMA20 覆盖更短时间，spread 通常小于 4h，阈值应低于 4h 的 1.0% → 用 0.8%
  - 最小数据: 与 crypto 4h 一致 60 根即可
  - 无交易时段过滤 (港美股有固定交易时间, 不需要)

信号输出:
  - 强形态(锤子/吞没): ±0.9
  - 弱形态(十字星后突破): ±0.6
  - 无信号: 0
"""
import pandas as pd
import numpy as np


def calculate_factor(df: pd.DataFrame) -> pd.Series:
    """
    EMA趋势缩量回踩反转因子 (小时线版)

    复刻加密系统4h核心因子, 适配港美股小时线
    输出: ±0.9(强反转) / ±0.6(弱反转) / 0(无信号)
    """
    close = df['close'].astype('float64')
    high = df['high'].astype('float64')
    low = df['low'].astype('float64')
    open_ = df['open'].astype('float64')
    volume = df['volume'].astype('float64')
    n = len(close)

    if n < 60:
        return pd.Series(0.0, index=df.index, dtype='float64')

    # ===== 条件1: EMA趋势方向 =====
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema60 = close.ewm(span=60, adjust=False).mean()
    uptrend = ema20 > ema60
    downtrend = ema20 < ema60

    # ===== 条件2: 趋势间距过滤 =====
    # 1h 比 4h 频率高，EMA spread 更窄；阈值应低于 crypto 4h 的 1.0%
    ema_spread = (ema20 - ema60).abs() / (close + 1e-10)
    trend_established = ema_spread > 0.008  # 0.8%

    # ===== 条件3: 价格回踩到EMA20附近 + 收盘确认 =====
    pullback_pct = 0.030  # 3.0% (与crypto相同)
    ema20_band = ema20 * pullback_pct

    # 多头回踩: 低点接近EMA20(上方1.5%内) + 不跌破太远(下方3%)
    pullback_to_ema_bull = (
        (low <= ema20 + ema20_band * 0.5) &
        (low >= ema20 - ema20_band) &
        (close > ema20 * 0.997)  # 收盘确认: >EMA20, 0.3%误差
    )
    # 空头反弹
    pullback_to_ema_bear = (
        (high >= ema20 - ema20_band * 0.5) &
        (high <= ema20 + ema20_band) &
        (close < ema20 * 1.003)  # 收盘确认: <EMA20, 0.3%误差
    )

    # ===== 条件4: 前K方向确认 =====
    # 多头回踩: 前根阴线(确认在回踩中), 当前阳线(反转)
    # 空头反弹: 前根阳线(确认在反弹中), 当前阴线(反转)
    prev_body = close.shift(1) - open_.shift(1)
    prev_bearish = prev_body < 0  # 前根阴线
    prev_bullish = prev_body > 0  # 前根阳线

    # ===== 条件5: 经典反转K线形态 (信号分级) =====
    body = close - open_
    body_abs = body.abs()
    candle_range = high - low + 1e-10
    body_ratio = body_abs / candle_range
    upper_shadow = high - pd.concat([close, open_], axis=1).max(axis=1)
    lower_shadow = pd.concat([close, open_], axis=1).min(axis=1) - low

    # 看涨反转 - 强形态
    hammer = (
        (lower_shadow >= body_abs * 2) &
        (upper_shadow < body_abs * 1.0) &
        (body > 0) & (body_ratio > 0.10)
    )
    bullish_engulf = (
        (body > 0) & (body.shift(1) < 0) &
        (close > open_.shift(1)) & (open_ < close.shift(1))
    )
    strong_bull = hammer | bullish_engulf

    # 看涨反转 - 弱形态 (十字星后阳线)
    prev_doji = body_abs.shift(1) < candle_range.shift(1) * 0.2
    doji_then_bull = prev_doji & (body > 0) & (body_ratio > 0.4)
    weak_bull = doji_then_bull & ~strong_bull
    bullish_reversal = strong_bull | weak_bull

    # 看跌反转 - 强形态
    shooting_star = (
        (upper_shadow >= body_abs * 2) &
        (lower_shadow < body_abs * 1.0) &
        (body < 0) & (body_ratio > 0.10)
    )
    bearish_engulf = (
        (body < 0) & (body.shift(1) > 0) &
        (open_ > close.shift(1)) & (close < open_.shift(1))
    )
    strong_bear = shooting_star | bearish_engulf

    # 看跌反转 - 弱形态
    doji_then_bear = prev_doji & (body < 0) & (body_ratio > 0.4)
    weak_bear = doji_then_bear & ~strong_bear
    bearish_reversal = strong_bear | weak_bear

    # ===== 条件6: ADX + ATR波动率过滤 =====
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
    adx_ok = adx > 20  # 与crypto 4h版本相同

    # ATR%极端过滤: > 中位数×2.5 时不入场 (黑天鹅保护)
    atr_pct = atr14 / (close + 1e-10)
    atr_pct_median = atr_pct.rolling(60, min_periods=20).median()
    volatility_ok = atr_pct < atr_pct_median * 2.5

    # ===== 条件7: 缩量回踩确认 =====
    vol_ma20 = volume.rolling(20, min_periods=5).mean()
    vol_ratio = volume / (vol_ma20 + 1e-10)
    vol_shrink = vol_ratio < 1.0  # 成交量 < 20均量

    # ===== 综合信号 (8重AND链 + 信号分级) =====
    result = pd.Series(0.0, index=df.index, dtype='float64')

    bull_base = (
        uptrend & trend_established &
        pullback_to_ema_bull & vol_shrink &
        prev_bearish &
        bullish_reversal & adx_ok & volatility_ok
    )
    bear_base = (
        downtrend & trend_established &
        pullback_to_ema_bear & vol_shrink &
        prev_bullish &
        bearish_reversal & adx_ok & volatility_ok
    )

    # 信号分级: 强形态=0.9, 弱形态=0.6
    result[bull_base & strong_bull] = 0.9
    result[bull_base & weak_bull] = 0.6
    result[bear_base & strong_bear] = -0.9
    result[bear_base & weak_bear] = -0.6

    return result.fillna(0).astype('float64')
