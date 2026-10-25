"""
EMA趋势回踩反转 - 15m趋势跟随策略因子 v7.0

v7.0 (基于4h v13成功改进 + 15m波动特征数据分析):
  数据发现 (BTC+ETH 15m 60天信号分析):
    - 15m K线振幅中位0.3-0.4%, 远小于4h的1.6-2.3%
    - TP=1.2% 在8h内到达率 46%(BTC)/47%(ETH)
    - SL=2.5% 在8h内触发率 仅8%(BTC)/21%(ETH) → 宽止损+小TP
    - v6失败原因: TP太高(1.5%→33%到达)、止损太紧(频繁扫损)
  
  v7优化 (从4h v13同步):
    1. 去掉 adx_rising → 4h数据验证杀30%好信号(v11经验)
    2. 新增 spread上限 → 趋势过熟的回踩=反转(v13核心发现)
    3. 新增 spread不缩检测 → 趋势衰弱不入场(v11经验)
    4. RSI做多上限 78→70 → 过热区胜率低(v13经验)
    5. cooldown 3→8根(2小时) → 15m信号太频繁需更强去重
    6. TP1=1.2%(快速锁利, 46%到达率) + TP2=2.0% + 宽止损2.5%
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
    """EMA趋势回踩反转因子 v7.0 (15m K线)

    v7.0: 融合4h v13改进(去adx_rising+spread上限+spread不缩+RSI70)
    """
    params = parameters or {}

    # ===== 可调参数（从 parameters 读取，有默认值）=====
    ema_fast_period = int(params.get('ema_fast', 20))
    ema_slow_period = int(params.get('ema_slow', 60))
    ema_spread_min = float(params.get('ema_spread_min_pct', 0.25)) / 100
    ema_slope_min = float(params.get('ema_slope_threshold', 0.15)) / 100
    pullback_band_pct = float(params.get('pullback_tolerance_pct', 0.8)) / 100
    adx_base = float(params.get('adx_base', 22))
    adx_max_threshold = float(params.get('adx_max', 50))
    adx_rising_bars = int(params.get('adx_rising_bars', 0))  # v7: 默认0(关闭)
    di_spread_threshold = float(params.get('di_spread_min', 5.0))
    signal_val = float(params.get('signal_value', 0.7))
    rsi_period = int(params.get('rsi_period', 14))
    rsi_max_long = float(params.get('rsi_max_for_long', 70))  # v7: 78→70
    rsi_min_short = float(params.get('rsi_min_for_short', 30))  # v7: 22→30
    signal_cooldown_bars = int(params.get('signal_cooldown_bars', 8))  # v7: 3→8
    # v7 新增: spread 收缩检测
    spread_shrink_bars = int(params.get('spread_shrink_bars', 4))
    # v7 新增: spread 上限 (趋势过熟过滤)
    ema_spread_max = float(params.get('ema_spread_max_pct', 1.5)) / 100

    close = df['close'].astype('float64')
    high = df['high'].astype('float64')
    low = df['low'].astype('float64')
    open_ = df['open'].astype('float64')
    n = len(close)

    if n < 80:
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

    # ===== 条件2: 价格回踩/反弹到EMA快线附近 + 收盘确认 =====
    ema_band = ema_fast * pullback_band_pct
    pullback_to_ema_bull = (low <= ema_fast + ema_band) & (low >= ema_fast - ema_band)
    pullback_to_ema_bear = (high >= ema_fast - ema_band) & (high <= ema_fast + ema_band)

    close_above_ema = close > ema_fast * 0.998
    close_below_ema = close < ema_fast * 1.002

    # ===== 条件3: 简化K线确认 (v6.0: 只看阳阴线方向) =====
    bullish_close = close > open_
    bearish_close = close < open_

    # ===== 条件4: ADX + DI方向 + DI差值强度 =====
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
    adx_above_min = adx > adx_base
    # ADX上限: 趋势未过热 (v6.0核心改进)
    adx_below_max = adx < adx_max_threshold
    # ADX方向: 趋势正在加速 (v6.0核心改进)
    if adx_rising_bars > 0:
        adx_rising = adx > adx.shift(adx_rising_bars)
    else:
        adx_rising = pd.Series(True, index=df.index)

    di_bull = plus_di > minus_di
    di_bear = minus_di > plus_di
    di_spread = (plus_di - minus_di).abs()
    di_strong = di_spread > di_spread_threshold

    # ===== 条件5: RSI 极端区过滤 =====
    rsi = _rsi_wilder(close, rsi_period)
    rsi_ok_long = rsi <= rsi_max_long
    rsi_ok_short = rsi >= rsi_min_short

    # ===== 条件6 (v7新增): EMA spread 上限 — 趋势过熟过滤 =====
    if ema_spread_max > 0:
        spread_not_overextended = ema_spread <= ema_spread_max
    else:
        spread_not_overextended = pd.Series(True, index=df.index)

    # ===== 条件7 (v7新增): EMA spread 不在缩小 — 趋势未衰减 =====
    if spread_shrink_bars > 0:
        spread_expanding = ema_spread >= ema_spread.shift(spread_shrink_bars)
    else:
        spread_expanding = pd.Series(True, index=df.index)

    # ===== 综合信号 =====
    result = pd.Series(0.0, index=df.index, dtype='float64')

    bull = (uptrend & trend_established & ema_rising &
            pullback_to_ema_bull & close_above_ema &
            bullish_close &
            adx_above_min & adx_below_max & adx_rising &
            di_bull & di_strong &
            rsi_ok_long &
            spread_not_overextended & spread_expanding)
    bear = (downtrend & trend_established & ema_falling &
            pullback_to_ema_bear & close_below_ema &
            bearish_close &
            adx_above_min & adx_below_max & adx_rising &
            di_bear & di_strong &
            rsi_ok_short &
            spread_not_overextended & spread_expanding)

    result[bull] = signal_val
    result[bear] = -signal_val

    # ===== 信号去重: 同方向N根内只保留第一个 (v6.0核心改进) =====
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
