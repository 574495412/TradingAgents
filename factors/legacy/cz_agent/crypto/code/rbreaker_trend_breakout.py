"""
R-Breaker 趋势突破因子 v1.5 (适配15m K线, 加密货币市场)

经典R-Breaker策略源自Richard Saidenberg，曾在Futures Truth杂志
连续多年位列"十大最赚钱策略"。原始策略包含趋势突破和反转两种模式，
本因子**仅使用趋势突破模式**。

版本历史:
  v1.0: 滚动窗口96根 → Bbreak随价格移动 → 0信号
  v1.1: 固定24h锚点(seg=96根) → 24h振幅4-5% → Bbreak距价5%+ → 0信号
  v1.2: seg=24根(6h) → 有信号, 但30天回测3币合计亏损
  v1.3: 五项修复(close确认/EMA门控/ADX25/vol1.3x/body40%)
  v1.4 (2026-04-03): 30天回测诊断6项修复 (见下方v1.5注释)
  v1.5 (2026-04-03): 去掉v1.4的「入场距离≤0.5%」上限
    v1.4问题: 30天回测3个币种零信号!
    根因: 加密15m K线突破Bbreak/Sbreak时几乎都是猛烈冲过去(>0.5%),
      0.3%~0.5%窗口太窄,实测BTC仅10/2960根K线在窗口内,ETH/SOL更少.
      穿越+阳/阴→强度≥0.3%后每币种约3~6个候选,完全够用.
    修复: 去掉max_entry_distance上限,只保留min_break_strength≥0.3%下限.
    保留v1.4其他修复: ATR floor 2.2%, segment 8h, 段尾过滤, TP3 ATR×3.0

核心原理:
  基于前一个8h周期(32根15m K线)的最高价H、最低价L、收盘价C，
  计算关键价位:
    Pivot   = (H + L + C) / 3
    Bbreak  = H + 2 × (Pivot - L)      — 突破买入价
    Sbreak  = L - 2 × (H - Pivot)      — 突破卖出价

  过滤条件(v1.5):
    1. 收盘价突破 (非high/low影线突破)
    2. 突破强度 ≥ 0.3% (close距bbreak/sbreak, 无上限)
    3. EMA20 vs EMA60 方向一致
    4. ADX > 25 (趋势存在且足够强)
    5. DI方向一致 + 差值 > 5
    6. 成交量 > MA20 × 1.30
    7. K线实体 > 40% (实体饱满)
    8. 冷却: 同方向间隔≥8根K线(2h)
    9. 段内位置: 不在segment最后6根K线开仓

与ema_trend_pullback_adx互补:
  - EMA因子: 趋势中段回踩反弹 (均值回归入场)
  - R-Breaker: 关键价位突破 (趋势加速型入场)
"""
import pandas as pd
import numpy as np


def calculate_rbreaker_trend_breakout(df: pd.DataFrame) -> pd.Series:
    """R-Breaker趋势突破因子 v1.5

    8h固定锚点 + 收盘确认突破 + 突破强度≥0.3% (无上限)
    + EMA方向 + ADX≥25/DI/量能强过滤 + 段内时间过滤 + 冷却。
    """
    close = df['close'].astype('float64')
    high = df['high'].astype('float64')
    low = df['low'].astype('float64')
    open_ = df['open'].astype('float64')
    volume = df['volume'].astype('float64')
    n = len(close)

    if n < 65:  # v1.5: 80→65, 只需要2个完整segment(32×2=64)即可开始计算
        return pd.Series(0.0, index=df.index, dtype='float64')

    # ================================================================
    # 第一部分: 固定8h锚点计算R-Breaker价位 (v1.4: 6h→8h)
    # ================================================================
    segment_size = 32  # v1.4: 8h = 32 × 15m (从24根升级)

    segment_id = np.arange(n) // segment_size

    high_arr = high.values
    low_arr = low.values
    close_arr = close.values

    max_seg = int(segment_id[-1]) + 1
    seg_high = np.full(max_seg, np.nan)
    seg_low = np.full(max_seg, np.nan)
    seg_close = np.full(max_seg, np.nan)

    for seg in range(max_seg):
        indices = np.where(segment_id == seg)[0]
        if len(indices) > 0:
            seg_high[seg] = high_arr[indices].max()
            seg_low[seg] = low_arr[indices].min()
            seg_close[seg] = close_arr[indices[-1]]

    ref_h = np.full(n, np.nan)
    ref_l = np.full(n, np.nan)
    ref_c = np.full(n, np.nan)

    for i in range(n):
        seg = int(segment_id[i])
        if seg >= 1:
            ref_h[i] = seg_high[seg - 1]
            ref_l[i] = seg_low[seg - 1]
            ref_c[i] = seg_close[seg - 1]

    ref_high = pd.Series(ref_h, index=df.index)
    ref_low = pd.Series(ref_l, index=df.index)
    ref_close = pd.Series(ref_c, index=df.index)

    # R-Breaker 关键价位 (8h段内固定不变)
    pivot = (ref_high + ref_low + ref_close) / 3.0
    bbreak = ref_high + 2.0 * (pivot - ref_low)
    sbreak = ref_low - 2.0 * (ref_high - pivot)

    # ================================================================
    # 第二部分: 突破信号检测 (v1.4: 收盘确认 + 突破强度 + 入场距离)
    # ================================================================
    # v1.3保留: close > bbreak (非high), 杜绝影线假突破
    close_break_up = close > bbreak
    prev_close_below = close.shift(1) <= bbreak
    close_above_open = close > open_
    break_buy_raw = close_break_up & prev_close_below & close_above_open

    # v1.5: 突破强度下限 (去掉v1.4的0.5%上限——加密15m突破几乎都>0.5%)
    break_strength_up = (close - bbreak) / (bbreak + 1e-10) * 100  # 突破幅度 %
    min_break_strength = 0.3   # 至少突破0.3%才有效 (过滤微弱假突破)
    strength_ok_up = break_strength_up >= min_break_strength
    break_buy_raw = break_buy_raw & strength_ok_up

    # v1.3保留: close < sbreak (非low)
    close_break_down = close < sbreak
    prev_close_above = close.shift(1) >= sbreak
    close_below_open = close < open_
    break_sell_raw = close_break_down & prev_close_above & close_below_open

    # v1.5: 突破强度下限 (同上, 去掉0.5%上限)
    break_strength_down = (sbreak - close) / (sbreak + 1e-10) * 100  # 突破幅度 %
    strength_ok_down = break_strength_down >= min_break_strength
    break_sell_raw = break_sell_raw & strength_ok_down

    # ================================================================
    # 第三部分: 过滤条件 (v1.3收紧 + v1.4段内时间)
    # ================================================================

    # --- v1.4新增④: 段内时间过滤 ---
    # segment最后6根K线 (1.5h) 不开仓, 避免段尾剩余时间不足
    bar_in_segment = np.arange(n) % segment_size  # 0 ~ segment_size-1
    segment_tail_cutoff = segment_size - 6  # 32-6=26, 即bar 26~31不开仓
    not_in_tail = pd.Series(bar_in_segment < segment_tail_cutoff, index=df.index)

    # --- EMA方向过滤 (v1.3) ---
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema60 = close.ewm(span=60, adjust=False).mean()
    ema_bullish = ema20 > ema60  # 仅做多
    ema_bearish = ema20 < ema60  # 仅做空

    # --- ADX + DI (v1.3: ADX>25) ---
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14, min_periods=1).mean()

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index
    )
    plus_di = 100.0 * plus_dm.rolling(14, min_periods=1).mean() / (atr14 + 1e-10)
    minus_di = 100.0 * minus_dm.rolling(14, min_periods=1).mean() / (atr14 + 1e-10)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(14, min_periods=1).mean()

    adx_ok = adx > 25.0        # v1.3: 20→25
    di_bull = plus_di > minus_di
    di_bear = minus_di > plus_di
    di_spread = (plus_di - minus_di).abs()
    di_strong = di_spread > 5.0  # v1.3: 3→5

    # --- 成交量 ---
    vol_ma20 = volume.rolling(20, min_periods=5).mean()
    vol_ratio = volume / (vol_ma20 + 1e-10)
    vol_surge = vol_ratio > 1.30

    # --- K线实体 ---
    body_abs = (close - open_).abs()
    candle_range = high - low + 1e-10
    body_ratio = body_abs / candle_range
    solid_candle = body_ratio > 0.40

    # ================================================================
    # 第四部分: 综合信号 + 冷却
    # ================================================================
    # v1.4: 加入 not_in_tail 段内时间过滤
    bull_signal = (break_buy_raw & not_in_tail & ema_bullish & adx_ok &
                   di_bull & di_strong & vol_surge & solid_candle)
    bear_signal = (break_sell_raw & not_in_tail & ema_bearish & adx_ok &
                   di_bear & di_strong & vol_surge & solid_candle)

    cooldown = 8  # 2小时
    result_arr = np.zeros(n, dtype='float64')
    last_bull_idx = -cooldown - 1
    last_bear_idx = -cooldown - 1

    bull_indices = np.where(bull_signal.values)[0]
    bear_indices = np.where(bear_signal.values)[0]

    all_signals = []
    for idx in bull_indices:
        all_signals.append((idx, 'bull'))
    for idx in bear_indices:
        all_signals.append((idx, 'bear'))
    all_signals.sort(key=lambda x: x[0])

    for idx, direction in all_signals:
        if direction == 'bull':
            if idx - last_bull_idx >= cooldown:
                result_arr[idx] = 0.7
                last_bull_idx = idx
        else:
            if idx - last_bear_idx >= cooldown:
                result_arr[idx] = -0.7
                last_bear_idx = idx

    return pd.Series(result_arr, index=df.index, dtype='float64')


# 兼容实盘加载器 (strict lookup 'calculate_factor')
calculate_factor = calculate_rbreaker_trend_breakout
