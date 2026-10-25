"""
资金费率逼空检测因子 (P1)

基于 Kyle's Market Impact Model + Funding Rate Bleed Model：
- 当资金费率极端负值时，空头在持续付费
- 若价格同时上涨+OI增加，说明空头还在加仓（燃料充足）
- 连环爆仓正反馈即将启动

信号：
  +0.8~1.0 → 强烈逼空信号（做多）
  +0.4~0.8 → 逼空前兆
  -0.4~-0.8 → 逼多前兆
  -0.8~-1.0 → 强烈逼多信号（做空）
  0 → 无信号

需要的额外数据列（通过 meme_engine 注入）：
  funding_rate: 当前资金费率（%）
  open_interest: 未平仓合约量
"""

import numpy as np
import pandas as pd


def calculate_factor(df: pd.DataFrame) -> pd.Series:
    """
    资金费率逼空/逼多检测

    Args:
        df: OHLCV + funding_rate + open_interest 数据
            必须列: open, high, low, close, volume
            可选列: funding_rate, open_interest

    Returns:
        信号序列 [-1, 1]
    """
    n = len(df)
    signal = pd.Series(0.0, index=df.index)

    if n < 20:
        return signal

    close = df['close'].values
    volume = df['volume'].values

    # ── 1. 价格动量 ──
    # 短期价格变化率（5根K线）
    price_ret_5 = np.zeros(n)
    for i in range(5, n):
        if close[i - 5] > 0:
            price_ret_5[i] = (close[i] - close[i - 5]) / close[i - 5] * 100

    # 中期价格变化率（20根K线）
    price_ret_20 = np.zeros(n)
    for i in range(20, n):
        if close[i - 20] > 0:
            price_ret_20[i] = (close[i] - close[i - 20]) / close[i - 20] * 100

    price_ret_60 = np.zeros(n)
    for i in range(60, n):
        if close[i - 60] > 0:
            price_ret_60[i] = (close[i] - close[i - 60]) / close[i - 60] * 100

    # ── 2. 成交量异动 ──
    vol_ma_20 = pd.Series(volume).rolling(20, min_periods=10).mean().values
    vol_ratio = np.ones(n)
    for i in range(n):
        if vol_ma_20[i] > 0:
            vol_ratio[i] = volume[i] / vol_ma_20[i]

    # ── 3. 资金费率信号 ──
    has_funding = 'funding_rate' in df.columns
    if has_funding:
        fr = df['funding_rate'].fillna(0).values
    else:
        # 没有资金费率数据时，用价格偏离度近似
        ema_20 = pd.Series(close).ewm(span=20, min_periods=10).mean().values
        fr = np.zeros(n)
        for i in range(20, n):
            if ema_20[i] > 0:
                # 价格大幅高于均线 → 隐含正费率（多头付钱）
                # 价格大幅低于均线 → 隐含负费率（空头付钱）
                fr[i] = (close[i] - ema_20[i]) / ema_20[i] * -10  # 反向：涨多→空头挤压

    # ── 4. OI变化 ──
    has_oi = 'open_interest' in df.columns
    oi_change_rate = np.zeros(n)
    if has_oi:
        oi = df['open_interest'].ffill().values
        for i in range(5, n):
            if oi[i - 5] > 0:
                oi_change_rate[i] = (oi[i] - oi[i - 5]) / oi[i - 5] * 100

    # ── 5. 波动率（ATR Ratio）──
    high = df['high'].values
    low = df['low'].values
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - np.roll(close, 1)),
                               np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).rolling(14, min_periods=7).mean().values
    atr_ratio = np.zeros(n)
    for i in range(n):
        if close[i] > 0:
            atr_ratio[i] = atr_14[i] / close[i] * 100  # ATR%

    # ── 6. 综合评分 ──
    raw_signal = np.zeros(n)
    for i in range(20, n):
        squeeze_long = 0.0   # 逼空分数（做多）
        squeeze_short = 0.0  # 逼多分数（做空）

        long_evidence = 0
        short_evidence = 0

        # --- 逼空信号 ---
        # a. 资金费率极端负值（空头付钱）
        if fr[i] < -0.1:
            squeeze_long += min(0.35, abs(fr[i]) / 1.5)
            long_evidence += 1

        # b. 价格正在上涨
        if price_ret_5[i] > 1:
            squeeze_long += min(0.25, price_ret_5[i] / 15)
            long_evidence += 1

        # c. OI增加（空头还在加仓 = 燃料充足）
        if oi_change_rate[i] > 2:
            squeeze_long += min(0.2, oi_change_rate[i] / 30)
            long_evidence += 1

        # d. 成交量萎缩（流通量被控制 = Kyle λ 增大）
        if vol_ratio[i] < 0.7:
            squeeze_long += 0.1
            long_evidence += 1
        # 或成交量突然放大（爆仓开始）
        elif vol_ratio[i] > 2.5:
            squeeze_long += 0.1
            long_evidence += 1

        # --- 逼多信号（对称）---
        if fr[i] > 0.1:
            squeeze_short += min(0.35, fr[i] / 1.5)
            short_evidence += 1

        if price_ret_5[i] < -1:
            squeeze_short += min(0.25, abs(price_ret_5[i]) / 15)
            short_evidence += 1

        if oi_change_rate[i] > 2:
            squeeze_short += min(0.2, oi_change_rate[i] / 30)
            short_evidence += 1

        if vol_ratio[i] < 0.7:
            squeeze_short += 0.1
            short_evidence += 1
        elif vol_ratio[i] > 2.5:
            squeeze_short += 0.1
            short_evidence += 1

        # 单靠资金费率极端很容易变成“已挤完再追”。没有 OI/量/价格二次确认时，
        # 只输出观察级信号，不能单因子触发开仓。
        if squeeze_long > squeeze_short and squeeze_long > 0.35:
            has_fuel = oi_change_rate[i] > 5
            if price_ret_5[i] <= 1 or price_ret_20[i] <= 5:
                raw_signal[i] = min(0.55, squeeze_long)
            elif not has_fuel:
                raw_signal[i] = min(0.58, squeeze_long)
            elif long_evidence < 3:
                raw_signal[i] = min(0.55, squeeze_long)
            elif price_ret_60[i] > 10:
                raw_signal[i] = min(0.55, squeeze_long)
            else:
                raw_signal[i] = min(1.0, squeeze_long * 1.2)
        elif squeeze_short > squeeze_long and squeeze_short > 0.35:
            has_fuel = oi_change_rate[i] > 5
            if price_ret_5[i] >= -1 or price_ret_20[i] >= -5:
                raw_signal[i] = -min(0.55, squeeze_short)
            elif not has_fuel:
                raw_signal[i] = -min(0.58, squeeze_short)
            elif short_evidence < 3:
                raw_signal[i] = -min(0.55, squeeze_short)
            elif price_ret_60[i] < -10:
                raw_signal[i] = -min(0.55, squeeze_short)
            else:
                raw_signal[i] = -min(1.0, squeeze_short * 1.2)

    # 信号平滑（避免逐bar震荡）
    smooth = pd.Series(raw_signal).ewm(span=5, min_periods=3).mean().values
    for i in range(20, n):
        if abs(smooth[i]) >= 0.15:
            val = np.clip(smooth[i] * 1.5, -1.0, 1.0)
            # 平滑会把前几根强信号拖尾到过热区，末端再做一次强约束。
            if val > 0 and (price_ret_5[i] <= 1 or price_ret_20[i] <= 5):
                val = min(val, 0.55)
            elif val > 0 and price_ret_60[i] > 10:
                val = min(val, 0.58)
            elif val < 0 and (price_ret_5[i] >= -1 or price_ret_20[i] >= -5):
                val = max(val, -0.55)
            elif val < 0 and price_ret_60[i] < -10:
                val = max(val, -0.58)
            signal.iloc[i] = val

    return signal
