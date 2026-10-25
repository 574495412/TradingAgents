"""
OI飙升动量因子 (P1)

基于连环爆仓 Hawkes Process 模型：
- OI急剧增加 → 大量新仓位进场
- 价格向一个方向移动 → 另一侧的仓位开始被强制平仓
- 强平触发更多强平 → 正反饋死亡螺旋

信号逻辑：
  OI飙升 + 价格上涨 → 多头方向的连环爆仓将推高价格 → 做多
  OI飙升 + 价格下跌 → 空头方向的连环爆仓将打压价格 → 做空
  OI急降 + 价格剧烈波动 → 大规模清算正在发生 → 跟随趋势方向
"""

import numpy as np
import pandas as pd


def calculate_factor(df: pd.DataFrame) -> pd.Series:
    """
    OI飙升动量因子

    Args:
        df: OHLCV + open_interest 数据

    Returns:
        信号序列 [-1, 1]
    """
    n = len(df)
    signal = pd.Series(0.0, index=df.index)

    if n < 30:
        return signal

    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values

    # ── 1. 价格动量指标 ──
    # 短周期EMA
    ema_fast = pd.Series(close).ewm(span=8, min_periods=5).mean().values
    ema_slow = pd.Series(close).ewm(span=21, min_periods=10).mean().values

    # 价格变化率
    roc_5 = np.zeros(n)
    for i in range(5, n):
        if close[i - 5] > 0:
            roc_5[i] = (close[i] - close[i - 5]) / close[i - 5] * 100
    roc_20 = np.zeros(n)
    for i in range(20, n):
        if close[i - 20] > 0:
            roc_20[i] = (close[i] - close[i - 20]) / close[i - 20] * 100
    roc_60 = np.zeros(n)
    for i in range(60, n):
        if close[i - 60] > 0:
            roc_60[i] = (close[i] - close[i - 60]) / close[i - 60] * 100
    roc_180 = np.zeros(n)
    for i in range(180, n):
        if close[i - 180] > 0:
            roc_180[i] = (close[i] - close[i - 180]) / close[i - 180] * 100

    # ── 2. 波动率飙升检测（ATR突增）──
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - np.roll(close, 1)),
                               np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_short = pd.Series(tr).rolling(5, min_periods=3).mean().values
    atr_long = pd.Series(tr).rolling(20, min_periods=10).mean().values
    atr_spike = np.ones(n)
    for i in range(n):
        if atr_long[i] > 0:
            atr_spike[i] = atr_short[i] / atr_long[i]

    # ── 3. 成交量脉冲检测 ──
    vol_ma = pd.Series(volume).rolling(20, min_periods=10).mean().values
    vol_ratio = np.ones(n)
    for i in range(n):
        if vol_ma[i] > 0:
            vol_ratio[i] = volume[i] / vol_ma[i]

    # ── 4. OI变化率 ──
    has_oi = 'open_interest' in df.columns
    oi_roc = np.zeros(n)
    oi_accel = np.zeros(n)  # OI加速度
    if has_oi:
        oi = df['open_interest'].ffill().values
        for i in range(3, n):
            if oi[i - 3] > 0:
                oi_roc[i] = (oi[i] - oi[i - 3]) / oi[i - 3] * 100
        for i in range(6, n):
            oi_accel[i] = oi_roc[i] - oi_roc[i - 3]

    # ── 5. 连环爆仓特征检测（Hawkes自激信号）──
    # 特征：连续多根K线同方向大幅波动 + 成交量持续放大
    consecutive_move = np.zeros(n)  # 连续同方向移动计数
    for i in range(1, n):
        if close[i] > close[i - 1]:
            consecutive_move[i] = max(0, consecutive_move[i - 1]) + 1
        elif close[i] < close[i - 1]:
            consecutive_move[i] = min(0, consecutive_move[i - 1]) - 1
        else:
            consecutive_move[i] = 0

    # ── 6. 综合评分 ──
    raw_signal = np.zeros(n)
    for i in range(30, n):
        score = 0.0
        direction = 0  # +1=多, -1=空
        evidence_count = 0
        has_oi_evidence = False

        # a. EMA趋势方向（要求明确分离）
        ema_diff_pct = (ema_fast[i] - ema_slow[i]) / ema_slow[i] * 100 if ema_slow[i] > 0 else 0
        if ema_diff_pct > 0.05:  # 要求明确偏离，而不是微小交叉
            direction = 1
        elif ema_diff_pct < -0.05:
            direction = -1
        else:
            continue

        # b. OI飙升 → 加分
        if has_oi:
            if oi_roc[i] > 5:  # OI 5%以上增长
                score += min(0.3, oi_roc[i] / 20)
                evidence_count += 1
                has_oi_evidence = True
            if oi_accel[i] > 2:  # OI加速增长
                score += 0.1
                evidence_count += 1
                has_oi_evidence = True

        # c. 波动率飙升（爆仓正在发生）
        if atr_spike[i] > 1.8:
            score += min(0.25, (atr_spike[i] - 1.5) / 2)
            evidence_count += 1

        # d. 成交量脉冲
        if vol_ratio[i] > 2.0:
            score += min(0.2, (vol_ratio[i] - 1.5) / 3)
            evidence_count += 1

        # e. 连续同方向移动（Hawkes自激特征）— 要求更长连续
        if abs(consecutive_move[i]) >= 5:
            score += min(0.2, abs(consecutive_move[i]) / 10)
            direction = 1 if consecutive_move[i] > 0 else -1
            evidence_count += 1

        # f. 价格动量确认
        if direction > 0 and roc_5[i] > 2.0:
            score += min(0.15, roc_5[i] / 10)
            evidence_count += 1
        elif direction < 0 and roc_5[i] < -2.0:
            score += min(0.15, abs(roc_5[i]) / 10)
            evidence_count += 1

        # 单纯EMA+量/波动容易追到假突破。没有OI确认时，需要更多独立证据；
        # 已经单边大幅延伸但缺少OI燃料时，只给观察级信号。
        if evidence_count < 2:
            continue
        if not has_oi_evidence and evidence_count < 3:
            continue
        if direction > 0 and roc_60[i] > 12 and not has_oi_evidence:
            raw_signal[i] = min(0.55, score)
            continue
        if direction < 0 and roc_60[i] < -12 and not has_oi_evidence:
            raw_signal[i] = -min(0.55, score)
            continue
        # 大幅单边后再追同向，盈亏比会快速恶化；除非有强 OI 燃料，否则压成观察信号。
        # 但优质meme常见“首分笔急拉/急跌 + 短暂回撤 + 二段延续”，完全压死会错过主盈利段。
        # 因此当出现"回撤后再加速"结构时，允许给到可交易级信号。
        if direction > 0 and roc_20[i] > 8 and oi_roc[i] <= 5:
            continuation_reset_long = (
                roc_5[i] > 1.2
                and roc_5[i] < 7.5
                and consecutive_move[i] >= 3
                and vol_ratio[i] > 1.4
                and atr_spike[i] < 2.8
            )
            if continuation_reset_long:
                raw_signal[i] = min(0.75, score * 1.15)
            else:
                raw_signal[i] = min(0.55, score)
            continue
        if direction < 0 and roc_20[i] < -8 and oi_roc[i] <= 5:
            continuation_reset_short = (
                roc_5[i] < -1.2
                and roc_5[i] > -7.5
                and consecutive_move[i] <= -3
                and vol_ratio[i] > 1.4
                and atr_spike[i] < 2.8
            )
            if continuation_reset_short:
                raw_signal[i] = -min(0.75, score * 1.15)
            else:
                raw_signal[i] = -min(0.55, score)
            continue
        if direction > 0 and roc_60[i] > 6 and roc_180[i] > 12 and oi_roc[i] <= 5:
            raw_signal[i] = min(0.55, score)
            continue
        if direction < 0 and roc_60[i] < -6 and roc_180[i] < -12 and oi_roc[i] <= 5:
            raw_signal[i] = -min(0.55, score)
            continue

        # 输出原始信号（提高激活门槛）
        if score >= 0.4:
            raw_signal[i] = direction * min(1.0, score * 1.35)

    # ── 7. 信号平滑（避免逐bar震荡） ──
    # 用EMA平滑原始信号，减少噪音
    smooth = pd.Series(raw_signal).ewm(span=5, min_periods=3).mean().values
    for i in range(30, n):
        if abs(smooth[i]) >= 0.15:
            signal.iloc[i] = np.clip(smooth[i] * 1.5, -1.0, 1.0)

    return signal
