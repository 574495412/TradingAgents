"""
多均线趋势强度因子 v3 (增强版) - 港美股

v2→v3 改进 (参考加密系统盈利经验):
  1. 均线排列需持续3日 + 趋势间距>0.5% (crypto核心: 趋势确立)
  2. ADX门槛从soft改为hard: ADX<15惩罚, >25加强
  3. DI差值加权: |+DI - -DI|越大趋势越强 (crypto核心)
  4. MACD需与趋势方向一致才加分 (方向确认)
  5. 波动率自适应: 高波动降权
  6. 新增趋势持续性: EMA斜率确认
"""
import pandas as pd
import numpy as np


def calculate_factor(df: pd.DataFrame) -> pd.Series:
    close = df['close'].astype('float64')
    high = df['high'].astype('float64')
    low = df['low'].astype('float64')
    volume = df['volume'].astype('float64')
    n = len(close)

    if n < 60:
        return pd.Series(0.0, index=df.index, dtype='float64')

    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()

    # 均线排列分数
    score = pd.Series(0.0, index=df.index)
    score += (ma5 > ma10).astype(float) - (ma5 < ma10).astype(float)
    score += (ma10 > ma20).astype(float) - (ma10 < ma20).astype(float)
    score += (ma20 > ma50).astype(float) - (ma20 < ma50).astype(float)
    alignment_score = score / 3.0

    # 趋势持续性: 3日排列均值
    alignment_smooth = alignment_score.rolling(3, min_periods=1).mean()

    # 【v3新增】EMA趋势间距门槛 (crypto核心)
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema60 = close.ewm(span=60, adjust=False).mean()
    ema_spread = (ema20 - ema60).abs() / (close + 1e-10)
    trend_established = np.where(ema_spread > 0.005, 1.0, 0.5)  # 间距不够减半

    # 【v3新增】EMA斜率确认
    ema20_slope = (ema20 - ema20.shift(5)) / (ema20.shift(5) + 1e-10)
    slope_confirm = np.where(
        (alignment_smooth > 0) & (ema20_slope > 0.001), 1.1,
        np.where((alignment_smooth < 0) & (ema20_slope < -0.001), 1.1, 0.8)
    )

    price_vs_ma20 = (close - ma20) / (ma20 + 1e-10)
    deviation_norm = np.clip(price_vs_ma20 / 0.05, -1, 1)

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal
    macd_norm = macd_hist / (close * 0.01 + 1e-10)
    macd_score = np.clip(macd_norm / 2, -1, 1)

    # 【v3改进】MACD方向一致性: MACD需与趋势同向
    macd_dir = np.sign(macd_score)
    trend_dir = np.sign(alignment_smooth)
    macd_confirm = np.where(macd_dir == trend_dir, 1.0, 0.5)

    # ADX/DI (加密系统核心)
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14, min_periods=1).mean()
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    plus_di = 100 * plus_dm.rolling(14, min_periods=1).mean() / (atr14 + 1e-10)
    minus_di = 100 * minus_dm.rolling(14, min_periods=1).mean() / (atr14 + 1e-10)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(14, min_periods=1).mean()

    # 【v3改进】ADX门槛化
    adx_weight = np.clip((adx - 10) / 25, 0.3, 1.2)

    # 【v3新增】DI差值加权 (crypto核心)
    di_spread = (plus_di - minus_di).abs()
    di_weight = np.clip(di_spread / 15, 0.5, 1.3)

    # 波动率自适应
    returns = close.pct_change(fill_method=None)
    vol_20 = returns.rolling(20, min_periods=5).std()
    vol_60 = returns.rolling(60, min_periods=20).std()
    vol_ratio = vol_20 / (vol_60 + 1e-10)
    vol_penalty = np.clip(1.5 - vol_ratio, 0.5, 1.2)

    raw_signal = (alignment_smooth * 0.30 + deviation_norm * 0.25 +
                  pd.Series(macd_score, index=df.index) * macd_confirm * 0.25 +
                  pd.Series(0, index=df.index) * 0.20)
    raw_signal = raw_signal * trend_established * slope_confirm * adx_weight * di_weight * vol_penalty

    signal_mean = raw_signal.rolling(50, min_periods=20).mean()
    signal_std = raw_signal.rolling(50, min_periods=20).std()
    zscore = (raw_signal - signal_mean) / (signal_std + 1e-8)

    zscore_smooth = pd.Series(zscore).ewm(span=3, adjust=False).mean()
    result = np.clip(zscore_smooth / 2.5, -1, 1)

    return pd.Series(result, index=df.index).ffill().fillna(0).astype('float64')
