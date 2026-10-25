"""
布林带收缩突破因子 v1.0 — 波动率收缩后的趋势启动信号

设计理念:
  与 ema_trend_pullback_adx (趋势回踩) 形成互补:
  - 回踩因子: 捕捉已确立趋势中的回调入场机会
  - 本因子: 捕捉盘整收缩后的新趋势启动点

核心逻辑:
  1. 布林带宽度(BB width)收缩到近期极低水平 → 市场蓄力
  2. 价格突破布林带上/下轨 → 方向性突破
  3. 成交量配合放大 → 真突破而非假突破
  4. ATR扩张确认 → 波动率真正打开
  5. K线实体确认 → 收盘站稳突破方向

信号特征:
  - 低频: 盘整→突破是稀缺事件, 预计30天每币5-12次信号
  - 与回踩因子低相关: 回踩因子在趋势中间段发信号, 本因子在趋势起点
  - 适配15m K线

参数设计 (15m级别):
  BB周期=20, 标准差倍数=2.0 (经典参数)
  收缩检测: BB width低于近60根K线的25分位 → 属于"窄带"状态
  突破确认: 收盘价突破BB上/下轨 + 实体>50%在轨外
  量能确认: 当前成交量>MA20×1.3 (较宽松, 突破初期量不一定爆发)
  ATR扩张: 当前ATR > 近10根ATR均值×1.1 (波动率开始扩大)
  方向过滤: EMA20方向一致(不做逆势突破)
"""
import pandas as pd
import numpy as np


def calculate_factor(df: pd.DataFrame) -> pd.Series:
    """布林带收缩突破因子 v1.0 (适配15m K线)

    盘整收缩后突破上/下轨 + 量能确认 + ATR扩张 + EMA方向一致
    """
    close = df['close'].astype('float64')
    high = df['high'].astype('float64')
    low = df['low'].astype('float64')
    open_ = df['open'].astype('float64')
    volume = df['volume'].astype('float64')
    n = len(close)

    if n < 80:
        return pd.Series(0.0, index=df.index, dtype='float64')

    # ===== 布林带计算 =====
    bb_period = 20
    bb_std_mult = 2.0
    bb_ma = close.rolling(bb_period, min_periods=bb_period).mean()
    bb_std = close.rolling(bb_period, min_periods=bb_period).std()
    bb_upper = bb_ma + bb_std_mult * bb_std
    bb_lower = bb_ma - bb_std_mult * bb_std

    # 布林带宽度 (标准化)
    bb_width = (bb_upper - bb_lower) / (bb_ma + 1e-10)

    # ===== 条件1: 布林带收缩 — 宽度处于近期低位 =====
    # 当前BB宽度低于过去60根K线的25分位 → 盘整蓄力状态
    squeeze_lookback = 60
    bb_width_q25 = bb_width.rolling(squeeze_lookback, min_periods=30).quantile(0.25)
    is_squeezed = bb_width < bb_width_q25

    # 要求前一根K线也在收缩区(至少连续2根收缩, 排除偶然缩窄)
    was_squeezed = is_squeezed.shift(1).fillna(False)
    squeeze_confirmed = is_squeezed | was_squeezed  # 当前或前一根在收缩

    # ===== 条件2: 价格突破布林带 =====
    # 多头突破: 收盘价 > BB上轨
    break_upper = close > bb_upper
    # 空头突破: 收盘价 < BB下轨
    break_lower = close < bb_lower

    # 实体方向确认: 突破方向的K线实体要有力度
    body = close - open_
    body_abs = body.abs()
    candle_range = high - low + 1e-10
    body_ratio = body_abs / candle_range

    # 多头突破: 阳线 + 实体占比>35%
    bullish_body = (body > 0) & (body_ratio > 0.35)
    # 空头突破: 阴线 + 实体占比>35%
    bearish_body = (body < 0) & (body_ratio > 0.35)

    # ===== 条件3: 成交量放大确认 =====
    vol_ma20 = volume.rolling(20, min_periods=5).mean()
    vol_ratio = volume / (vol_ma20 + 1e-10)
    vol_surge = vol_ratio > 1.3  # 成交量>均量的130%

    # ===== 条件4: ATR扩张 — 波动率正在打开 =====
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14, min_periods=1).mean()

    # 当前ATR > 近10根ATR均值的1.1倍 → 波动率在扩张
    atr_recent_avg = atr14.rolling(10, min_periods=5).mean()
    atr_expanding = atr14 > atr_recent_avg * 1.1

    # ===== 条件5: EMA方向一致(不做逆势突破) =====
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema60 = close.ewm(span=60, adjust=False).mean()

    # 宽松的方向过滤: 只要EMA20在EMA60同侧, 或至少EMA20斜率方向一致
    ema20_slope = (ema20 - ema20.shift(3)) / (ema20.shift(3) + 1e-10)

    # 多头: EMA20>EMA60 或 EMA20上倾
    ema_bull = (ema20 > ema60) | (ema20_slope > 0.0005)
    # 空头: EMA20<EMA60 或 EMA20下倾
    ema_bear = (ema20 < ema60) | (ema20_slope < -0.0005)

    # ===== 条件6: 不在刚刚突破后的连续信号中(防止重复) =====
    # 要求前一根K线收盘在BB轨内(本根才是突破K线)
    prev_inside_upper = close.shift(1) <= bb_upper.shift(1)
    prev_inside_lower = close.shift(1) >= bb_lower.shift(1)

    # ===== 综合信号 =====
    result = pd.Series(0.0, index=df.index, dtype='float64')

    # 多头突破信号
    bull = (squeeze_confirmed &
            break_upper & prev_inside_upper &
            bullish_body &
            vol_surge &
            atr_expanding &
            ema_bull)

    # 空头突破信号
    bear = (squeeze_confirmed &
            break_lower & prev_inside_lower &
            bearish_body &
            vol_surge &
            atr_expanding &
            ema_bear)

    result[bull] = 0.7
    result[bear] = -0.7

    return result.fillna(0).astype('float64')
