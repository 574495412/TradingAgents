"""
崩盘背离预警因子 (P3)

基于 LPPLS模型 + 囚徒困境 + 流动性真空理论：
- 检测价格已大幅暴涨后的崩盘前兆
- 资金费率从极端值回归（逼空/逼多阶段结束）
- 成交量放大但价格停滞（庄家出货）
- 波动率收窄后突然放大（临界相变前兆）

信号：
  -0.6~-1.0 → 做空（崩盘预警，或平掉多仓）
  +0.6~+1.0 → 做多（暴跌后的反弹信号）
  0 → 无信号
"""

import numpy as np
import pandas as pd


def calculate_factor(df: pd.DataFrame) -> pd.Series:
    """
    崩盘背离预警因子

    核心逻辑：
    1. 检测"泡沫顶部"特征：价格超涨 + 动量衰竭 + 量价背离
    2. 资金费率回归信号（逼空阶段结束）
    3. LPPLS简化版：超指数增长 + 振荡频率加快
    """
    n = len(df)
    signal = pd.Series(0.0, index=df.index)

    if n < 30:
        return signal

    close = df['close'].values
    open_ = df['open'].values if 'open' in df.columns else np.roll(close, 1)
    open_[0] = close[0]
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values

    # ── 1. 多周期价格变化率 ──
    roc_periods = [5, 10, 20]
    rocs = {}
    for p in roc_periods:
        roc = np.zeros(n)
        for i in range(p, n):
            if close[i - p] > 0:
                roc[i] = (close[i] - close[i - p]) / close[i - p] * 100
        rocs[p] = roc

    # ── 2. RSI ──
    rsi = np.full(n, 50.0)
    period = 14
    for i in range(period + 1, n):
        gains = []
        losses = []
        for j in range(i - period, i):
            change = close[j + 1] - close[j] if j + 1 < n else 0
            if change > 0:
                gains.append(change)
            else:
                losses.append(abs(change))
        avg_gain = np.mean(gains) if gains else 0
        avg_loss = np.mean(losses) if losses else 0.0001
        if avg_loss < 1e-12:
            avg_loss = 1e-12
        rs = avg_gain / avg_loss
        rsi[i] = 100 - (100 / (1 + rs))

    # ── 3. 量价背离检测 ──
    # 价格创新高但成交量递减 → 背离
    vol_ma_10 = pd.Series(volume).rolling(10, min_periods=5).mean().values
    vol_ratio = np.ones(n)
    for i in range(n):
        if vol_ma_10[i] > 0:
            vol_ratio[i] = volume[i] / vol_ma_10[i]

    divergence = np.zeros(n)
    for i in range(20, n):
        # 价格在上涨
        if rocs[10][i] > 5:
            # 成交量在下降
            vol_trend = vol_ma_10[i] / vol_ma_10[i - 10] if vol_ma_10[i - 10] > 0 else 1
            if vol_trend < 0.8:  # 量缩>20%
                divergence[i] = min(1.0, (1 - vol_trend) * 3)  # 量缩越多，背离越强
        # 价格在下跌
        elif rocs[10][i] < -5:
            vol_trend = vol_ma_10[i] / vol_ma_10[i - 10] if vol_ma_10[i - 10] > 0 else 1
            if vol_trend < 0.8:
                divergence[i] = -min(1.0, (1 - vol_trend) * 3)

    # ── 4. 波动率变化率（LPPLS简化：临界相变前波动加速）──
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - np.roll(close, 1)),
                               np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_5 = pd.Series(tr).rolling(5, min_periods=3).mean().values
    atr_20 = pd.Series(tr).rolling(20, min_periods=10).mean().values
    vol_acceleration = np.ones(n)
    for i in range(n):
        if atr_20[i] > 0:
            vol_acceleration[i] = atr_5[i] / atr_20[i]

    # ── 5. 资金费率信号 ──
    has_funding = 'funding_rate' in df.columns
    fr = df['funding_rate'].fillna(0).values if has_funding else np.zeros(n)
    has_oi = 'open_interest' in df.columns
    oi_ret_5 = np.zeros(n)
    if has_oi:
        oi = df['open_interest'].ffill().values
        for i in range(5, n):
            if oi[i - 5] > 0:
                oi_ret_5[i] = (oi[i] - oi[i - 5]) / oi[i - 5] * 100

    # ── 6. 综合崩盘预警评分 ──
    raw_signal = np.zeros(n)
    for i in range(30, n):
        crash_score = 0.0   # 崩盘分数（做空）
        bounce_score = 0.0  # 反弹分数（做多）
        bar_range = max(1e-12, high[i] - low[i])
        upper_wick_ratio = (high[i] - max(open_[i], close[i])) / bar_range
        lower_wick_ratio = (min(open_[i], close[i]) - low[i]) / bar_range

        # === 崩盘信号（做空） ===

        # a. 价格已大幅上涨（泡沫阶段）— 1m K线20根=20分钟
        if rocs[20][i] > 8:
            crash_score += min(0.25, (rocs[20][i] - 8) / 30)

        # b. 短期动量衰竭（涨不动了）
        if rocs[20][i] > 5 and rocs[5][i] < rocs[10][i] * 0.5:
            crash_score += 0.15

        # c. 量价背离（价格涨但量缩）
        if divergence[i] > 0.3:
            crash_score += min(0.2, divergence[i] * 0.25)

        # d. RSI极端超买后回落
        if rsi[i] < 60 and any(rsi[j] > 75 for j in range(max(0, i - 10), i)):
            crash_score += 0.15

        # e. 波动率突然飙升（临界相变）
        if vol_acceleration[i] > 2.0 and rocs[5][i] < 0:
            crash_score += min(0.15, (vol_acceleration[i] - 2) / 5)

        # g. 冲高衰竭：短线急涨但K线给出明显上影/吞吐，且缺少OI燃料
        # 目标：减少RAVE/ORDI这类“顶部追涨后立刻硬止损”。
        if rocs[5][i] > 4 and rocs[20][i] > 10:
            if upper_wick_ratio > 0.45:
                crash_score += min(0.2, (upper_wick_ratio - 0.45) * 0.8)
            if vol_ratio[i] < 1.0 and rocs[5][i] < rocs[10][i] * 0.8:
                crash_score += 0.12
            if has_oi and oi_ret_5[i] <= 0:
                crash_score += min(0.2, abs(oi_ret_5[i]) / 8)
            if rocs[5][i] < 0 and rocs[10][i] > 5:
                crash_score += 0.12

        # f. 资金费率从极端值回归
        if has_funding:
            # 费率从极端负值回归到正常（逼空结束）
            if fr[i] > -0.1 and i >= 5:
                recent_min_fr = min(fr[max(0, i - 10):i])
                if recent_min_fr < -0.3:
                    recovery = fr[i] - recent_min_fr
                    crash_score += min(0.2, recovery / 0.5)

        # === 反弹信号（做多）===

        # a. 价格已大幅下跌 — 1m K线20根=20分钟
        if rocs[20][i] < -8:
            bounce_score += min(0.2, abs(rocs[20][i] + 8) / 20)

        # b. RSI极端超卖
        if rsi[i] < 30:
            bounce_score += min(0.2, (30 - rsi[i]) / 20)

        # c. 波动率开始收敛（恐慌衰退）
        if vol_acceleration[i] < 0.8 and rocs[5][i] > -2:
            bounce_score += 0.15

        # d. 量价背离（价格跌但量缩 → 抛压枯竭）
        if divergence[i] < -0.3:
            bounce_score += min(0.15, abs(divergence[i]) * 0.2)

        # e. 急跌吸筹：长下影 + OI不再扩张，优先看反弹而非继续踩踏
        if rocs[5][i] < -4 and rocs[20][i] < -10:
            if lower_wick_ratio > 0.45:
                bounce_score += min(0.2, (lower_wick_ratio - 0.45) * 0.8)
            if has_oi and oi_ret_5[i] <= 0:
                bounce_score += min(0.15, abs(oi_ret_5[i]) / 10)

        # 输出原始信号（提高激活门槛）
        if crash_score > bounce_score and crash_score >= 0.35:
            raw_signal[i] = -min(1.0, crash_score * 1.3)
        elif bounce_score > crash_score and bounce_score >= 0.35:
            raw_signal[i] = min(1.0, bounce_score * 1.3)

    # 信号平滑
    smooth = pd.Series(raw_signal).ewm(span=5, min_periods=3).mean().values
    for i in range(30, n):
        if abs(smooth[i]) >= 0.15:
            signal.iloc[i] = np.clip(smooth[i] * 1.5, -1.0, 1.0)

    return signal
