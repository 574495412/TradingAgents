"""
EMA趋势回踩反转 4h 大波段因子 v15.0

v15.0 核心改进 (入场时机优化 — 真实K线数据验证):
  数据发现 (2026-04-13~14 BTC+ETH 15m子步骤信号追踪):
    - DI条件(14bar滚动) 在趋势反转时延迟28-32小时 → 最大延迟源
    - BTC: DI在00:00时 +DI=21 vs -DI=27, 03:30才翻转, 整整比实际上涨晚8h
    - ETH: DI延迟5.5h, 仅略好于BTC
    - slope>0.30% 在反转初期也不满足, 额外延迟1-2h
    - spread_shrink_bars=3 要求3根bar不缩, 在反转初期也不满足
  v15优化:
    1. di_spread_min=0 → 禁用DI差值检查 (ADX+EMA趋势+斜率已足够)
    2. spread_shrink_bars=3→2 → 减少spread扩张确认等待
    3. ema_slope_threshold=0.30→0.15 → 降低斜率门槛, 允许更早入场
  v15新增:
    4. di_mode参数 (0=禁用, 1=仅方向, 2=方向+差值) → 可配置DI检查模式
  验证效果 (2026-04-13~14行情):
    BTC: 00:00入场@71989 → 比v14(06:00@73721)早6h, +2.41%
    ETH: 00:30入场@2227 → 比v14(06:00@2295)早5.5h, +3.04%
  v13/v14保留:
    - spread[0.8%,2.0%] + ADX[22,50] + 阳阴线 + RSI75/22
    - signal_cooldown=2, 止盈止损参数不变
"""
import pandas as pd
import numpy as np


def _rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI，与常见行情软件一致。"""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_g = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_l = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_g / (avg_l + 1e-10)
    return 100.0 - (100.0 / (1.0 + rs))


def calculate_factor(df: pd.DataFrame, parameters: dict = None) -> pd.Series:
    """EMA趋势回踩反转因子 v15.0 (4h K线大波段)

    核心: ADX[22,50] + spread[0.8%,2.0%] + 不收缩 + 阳阴线 + DI可配置
    v15: 禁用DI差值(延迟太大) + slope=0.15% + shrink_bars=2
    """
    params = parameters or {}

    # ===== 从 parameters 读取可调参数 =====
    ema_fast_period = int(params.get('ema_fast', 20))
    ema_slow_period = int(params.get('ema_slow', 60))
    adx_base_threshold = float(params.get('adx_base', 22))
    adx_max_threshold = float(params.get('adx_max', 50))
    signal_cooldown_bars = int(params.get('signal_cooldown_bars', 3))  # v11: 4→3
    pullback_pct = float(params.get('pullback_tolerance_pct', 2.5))
    ema_spread_min = float(params.get('ema_spread_min_pct', 0.8)) / 100
    ema_slope_min = float(params.get('ema_slope_threshold', 0.30)) / 100
    di_spread_threshold = float(params.get('di_spread_min', 5.0))
    # v15新增: DI检查模式 (0=禁用DI, 1=仅方向, 2=方向+差值, 默认2=向后兼容)
    di_mode = int(params.get('di_mode', 2))
    signal_val = float(params.get('signal_value', 0.7))
    rsi_period = int(params.get('rsi_period', 14))
    rsi_max_long = float(params.get('rsi_max_for_long', 70))
    rsi_min_short = float(params.get('rsi_min_for_short', 22))
    # v11 新增: spread 收缩检测回看根数 (0=禁用)
    spread_shrink_bars = int(params.get('spread_shrink_bars', 3))
    # v13 新增: spread 上限 (趋势过熟时回踩=反转, 0=禁用)
    ema_spread_max = float(params.get('ema_spread_max_pct', 2.0)) / 100

    close = df['close'].astype('float64')
    high = df['high'].astype('float64')
    low = df['low'].astype('float64')
    open_ = df['open'].astype('float64')
    n = len(close)

    if n < 65:
        return pd.Series(0.0, index=df.index, dtype='float64')

    # ===== 条件1: EMA趋势方向 + 斜率过滤 =====
    ema_fast = close.ewm(span=ema_fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=ema_slow_period, adjust=False).mean()
    uptrend = ema_fast > ema_slow
    downtrend = ema_fast < ema_slow

    ema_spread = (ema_fast - ema_slow).abs() / (close + 1e-10)
    trend_established = ema_spread > ema_spread_min

    ema_fast_slope = (ema_fast - ema_fast.shift(5)) / (ema_fast.shift(5) + 1e-10)
    slope_ok = ema_fast_slope.abs() > ema_slope_min
    ema_rising = (ema_fast_slope > 0) & slope_ok
    ema_falling = (ema_fast_slope < 0) & slope_ok

    # ===== 条件2: 价格回踩到EMA快线附近 + 收盘确认 =====
    ema_band = ema_fast * (pullback_pct / 100)
    pullback_to_ema_bull = (low <= ema_fast + ema_band) & (low >= ema_fast - ema_band)
    pullback_to_ema_bear = (high >= ema_fast - ema_band) & (high <= ema_fast + ema_band)

    close_above_ema = close > ema_fast * 0.998
    close_below_ema = close < ema_fast * 1.002
    # 阳/阴线方向确认 (v10数据验证有效, v11保留)
    bullish_close = close > open_
    bearish_close = close < open_

    # ===== 条件3: ADX + DI方向 + DI差值强度 =====
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

    # ADX下限: 有趋势
    adx_above_min = adx > adx_base_threshold
    # ADX上限: 趋势未过热
    adx_below_max = adx < adx_max_threshold
    # v11: 去掉 adx_rising — 数据验证杀了30%好信号且不提高胜率

    di_bull = plus_di > minus_di
    di_bear = minus_di > plus_di
    di_spread = (plus_di - minus_di).abs()
    di_strong = di_spread > di_spread_threshold

    # v15: DI检查模式
    if di_mode == 0:
        # 完全禁用DI — 不检查方向也不检查差值
        di_ok_bull = pd.Series(True, index=df.index)
        di_ok_bear = pd.Series(True, index=df.index)
    elif di_mode == 1:
        # 仅检查方向 — plus_di > minus_di 即可，不要求差值
        di_ok_bull = di_bull
        di_ok_bear = di_bear
    else:
        # 方向 + 差值 (v14及之前的默认行为)
        di_ok_bull = di_bull & di_strong
        di_ok_bear = di_bear & di_strong

    # ===== 条件4: RSI 极端区过滤 =====
    rsi = _rsi_wilder(close, rsi_period)
    rsi_ok_long = rsi <= rsi_max_long
    rsi_ok_short = rsi >= rsi_min_short

    # ===== 条件5 (v11新增): EMA spread 不在收缩中 =====
    # 趋势衰弱时 spread 下降, 此时不应追趋势
    # 允许 5% 噪声容差
    if spread_shrink_bars > 0:
        spread_not_shrinking = ema_spread >= ema_spread.shift(spread_shrink_bars) * 0.95
    else:
        spread_not_shrinking = pd.Series(True, index=df.index)

    # ===== 条件6 (v13新增): EMA spread 不能太大 =====
    # 数据验证: spread>2% 的信号 0好/16差 = 0%胜率
    # spread大表示趋势过熟, 此时"回踩"实为趋势反转
    if ema_spread_max > 0:
        spread_not_too_wide = ema_spread <= ema_spread_max
    else:
        spread_not_too_wide = pd.Series(True, index=df.index)

    # ===== 综合信号 =====
    result = pd.Series(0.0, index=df.index, dtype='float64')

    bull = (uptrend & trend_established & ema_rising &
            pullback_to_ema_bull & close_above_ema &
            bullish_close &
            adx_above_min & adx_below_max &
            di_ok_bull &
            rsi_ok_long &
            spread_not_shrinking &
            spread_not_too_wide)

    bear = (downtrend & trend_established & ema_falling &
            pullback_to_ema_bear & close_below_ema &
            bearish_close &
            adx_above_min & adx_below_max &
            di_ok_bear &
            rsi_ok_short &
            spread_not_shrinking &
            spread_not_too_wide)

    result[bull] = signal_val
    result[bear] = -signal_val

    # ===== 信号去重: 同方向N根内只保留第一个 =====
    if signal_cooldown_bars > 0:
        cleaned = result.copy()
        last_long_idx = -999
        last_short_idx = -999
        for i in range(len(cleaned)):
            if cleaned.iloc[i] > 0:
                if i - last_long_idx < signal_cooldown_bars:
                    cleaned.iloc[i] = 0.0
                else:
                    last_long_idx = i
            elif cleaned.iloc[i] < 0:
                if i - last_short_idx < signal_cooldown_bars:
                    cleaned.iloc[i] = 0.0
                else:
                    last_short_idx = i
        result = cleaned

    return result.fillna(0).astype('float64')
