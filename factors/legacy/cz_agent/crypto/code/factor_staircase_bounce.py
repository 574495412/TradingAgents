"""
阶梯反弹因子 (P3)

基于 Hawkes冷却期 + 空头平仓脉冲 + 整数关口支撑理论：
- 崩盘不是直线下跌，而是"阶梯式"：每跌一段就横盘几分钟
- 横盘 = 空头获利了结(买入平仓) + 整数关口买单墙 + 做市商重新定价
- 每段横盘后的第一次反弹是可以交易的超短线机会

信号：
  +0.5~+1.0 → 短线反弹做多（配合极短止损）
  -0.5~-1.0 → 继续下跌做空
  0 → 无信号

注意：此因子产生的信号持续时间很短（几分钟），需要1m K线配合。
"""

import numpy as np
import pandas as pd


def calculate_factor(df: pd.DataFrame) -> pd.Series:
    """
    阶梯反弹因子

    核心逻辑：
    1. 检测连续大幅下跌后的横盘区间（Hawkes冷却期）
    2. 在横盘区间末端，当成交量脉冲出现时产生反弹信号
    3. 同时检测横盘被击穿的继续下跌信号
    """
    n = len(df)
    signal = pd.Series(0.0, index=df.index)

    if n < 30:
        return signal

    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values

    # ── 1. 下跌幅度检测 ──
    # 最近20根K线内的最大跌幅
    max_drawdown = np.zeros(n)
    for i in range(20, n):
        window_high = max(high[i - 20:i])
        current_low = low[i]
        if window_high > 0:
            max_drawdown[i] = (window_high - current_low) / window_high * 100

    # ── 2. 波动率变化（冷却期检测）──
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - np.roll(close, 1)),
                               np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]

    # 短期 vs 中期波动率比
    atr_3 = pd.Series(tr).rolling(3, min_periods=2).mean().values
    atr_10 = pd.Series(tr).rolling(10, min_periods=5).mean().values
    vol_cooling = np.ones(n)
    for i in range(n):
        if atr_10[i] > 0:
            vol_cooling[i] = atr_3[i] / atr_10[i]

    # ── 3. 价格密度检测（横盘区间识别）──
    # 最近5根K线的价格范围 / ATR → 越小越像横盘
    consolidation = np.ones(n)
    for i in range(5, n):
        window_range = max(high[i - 5:i + 1]) - min(low[i - 5:i + 1])
        if atr_10[i] > 0:
            consolidation[i] = window_range / (atr_10[i] * 5)

    # ── 4. 成交量脉冲检测 ──
    vol_ma_10 = pd.Series(volume).rolling(10, min_periods=5).mean().values
    vol_ratio = np.ones(n)
    for i in range(n):
        if vol_ma_10[i] > 0:
            vol_ratio[i] = volume[i] / vol_ma_10[i]

    # ── 5. 整数关口距离 ──
    # 判断当前价格是否接近整数关口（心理价位支撐）
    round_number_proximity = np.zeros(n)
    for i in range(n):
        price = close[i]
        if price <= 0:
            continue
        # 根据价格量级选择整数关口间距
        if price > 1000:
            step = 100
        elif price > 100:
            step = 10
        elif price > 10:
            step = 1
        elif price > 1:
            step = 0.1
        elif price > 0.1:
            step = 0.01
        else:
            step = 0.001

        nearest_round = round(price / step) * step
        distance_pct = abs(price - nearest_round) / price * 100
        # 越接近整数关口 → 越强的支撐/阻力
        if distance_pct < 1:
            round_number_proximity[i] = 1.0 - distance_pct
        elif distance_pct < 2:
            round_number_proximity[i] = 0.5

    # ── 6. K线形态：锤子线/十字星（反转信号）──
    hammer = np.zeros(n)
    for i in range(n):
        body = abs(close[i] - df['open'].values[i]) if 'open' in df.columns else abs(close[i] - close[max(0, i - 1)])
        total_range = high[i] - low[i]
        if total_range > 0:
            body_ratio = body / total_range
            lower_shadow = (min(close[i], df['open'].values[i] if 'open' in df.columns else close[max(0, i - 1)]) - low[i]) / total_range
            # 锤子线（看涨反转）: 小body + 长下影线
            if body_ratio < 0.3 and lower_shadow > 0.5:
                hammer[i] = 0.8
            # 十字星: 极小body
            elif body_ratio < 0.1:
                hammer[i] = 0.4

    # ── 7. 综合评分 ──
    price_ret_20 = np.zeros(n)
    for i in range(20, n):
        if close[i - 20] > 0:
            price_ret_20[i] = (close[i] - close[i - 20]) / close[i - 20] * 100

    raw_signal = np.zeros(n)
    for i in range(30, n):
        bounce_score = 0.0   # 反弹做多
        continue_score = 0.0  # 继续下跌做空

        # 前提条件：最近20根K线内有显著下跌（1m K线=20分钟，5%已很剧烈）
        if max_drawdown[i] < 5:
            continue  # 没有显著下跌，跳过

        # === 反弹信号 ===

        # a. 波动率冷却（Hawkes衰减，恐慌暂歇）
        if vol_cooling[i] < 0.6:
            bounce_score += min(0.25, (1 - vol_cooling[i]) * 0.5)

        # b. 横盘整理（价格收窄）
        if consolidation[i] < 0.4:
            bounce_score += min(0.2, (1 - consolidation[i]) * 0.3)

        # c. 成交量脉冲（空头获利平仓的买盘）
        if vol_ratio[i] > 2.0 and close[i] > close[i - 1]:
            bounce_score += min(0.2, (vol_ratio[i] - 1) / 5)

        # d. 整数关口支撐
        if round_number_proximity[i] > 0.5:
            bounce_score += 0.1

        # e. 锤子线/十字星形态
        if hammer[i] > 0.3:
            bounce_score += hammer[i] * 0.15

        # f. 短期价格企稳（不再创新低，需要至少3根确认）
        if i >= 5 and low[i] > min(low[i - 5:i]) and low[i - 1] > min(low[i - 5:i - 1]):
            bounce_score += 0.1

        # === 继续下跌信号（横盘被击穿）===

        # a. 波动率重新飙升（新一波恐慌开始）
        if vol_cooling[i] > 1.8:
            continue_score += min(0.3, (vol_cooling[i] - 1) / 3)

        # b. 成交量放大 + 价格破位
        if vol_ratio[i] > 2.5 and close[i] < close[i - 1]:
            continue_score += min(0.25, (vol_ratio[i] - 1) / 5)

        # c. 创新低
        if i >= 5 and close[i] < min(low[i - 5:i]):
            continue_score += 0.2

        # d. 下跌幅度加大
        if max_drawdown[i] > 10:
            continue_score += min(0.15, (max_drawdown[i] - 10) / 20)

        # e. 妖币主跌段：暴跌不是“超跌不追”，而是流动性真空会继续踩踏。
        # 只有在放量、破位、波动扩张同时出现时放开追空，避免普通超跌震荡被误做空。
        panic_breakdown = (
            price_ret_20[i] < -12
            and max_drawdown[i] > 18
            and vol_ratio[i] > 2.5
            and vol_cooling[i] > 1.4
            and close[i] < min(low[i - 5:i])
        )
        if panic_breakdown:
            continue_score += min(0.35, abs(price_ret_20[i]) / 60)

        # 写入原始信号
        if bounce_score > continue_score and bounce_score >= 0.4:
            # 反弹已经拉出一大段时再追，多数是反抽末端。
            if price_ret_20[i] > 8:
                raw_signal[i] = min(0.55, bounce_score)
            else:
                raw_signal[i] = min(1.0, bounce_score * 1.1)
        elif continue_score > bounce_score and continue_score >= 0.45:
            if price_ret_20[i] < -8 and not panic_breakdown:
                raw_signal[i] = -min(0.55, continue_score)
            else:
                mult = 1.35 if panic_breakdown else 1.1
                raw_signal[i] = -min(1.0, continue_score * mult)

    # ── 8. 信号平滑（避免逐bar震荡）──
    smooth = pd.Series(raw_signal).ewm(span=5, min_periods=3).mean().values
    for i in range(30, n):
        if abs(smooth[i]) >= 0.15:
            val = np.clip(smooth[i] * 1.5, -1.0, 1.0)
            if val > 0 and price_ret_20[i] > 8:
                val = min(val, 0.55)
            elif val < 0 and price_ret_20[i] < -8:
                recent_panic = False
                for j in range(max(30, i - 2), i + 1):
                    recent_panic = recent_panic or (
                        price_ret_20[j] < -12
                        and max_drawdown[j] > 18
                        and vol_ratio[j] > 2.5
                        and vol_cooling[j] > 1.4
                        and close[j] < min(low[j - 5:j])
                    )
                if recent_panic:
                    signal.iloc[i] = val
                    continue
                val = max(val, -0.55)
            signal.iloc[i] = val

    return signal
