"""
Multi-alpha ensemble factor.

This factor blends 10 strategy archetypes into one directional score:
1) trend following
2) breakout continuation
3) multi-horizon momentum
4) mean reversion
5) volatility expansion
6) funding carry
7) OI pressure
8) crash continuation
9) squeeze exhaustion reversal
10) microstructure wick imbalance
"""

import os

import numpy as np
import pandas as pd


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _pct(cur: float, prev: float) -> float:
    if prev <= 0:
        return 0.0
    return (cur - prev) / prev * 100.0


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros_like(a, dtype=float)
    mask = b > 1e-12
    out[mask] = a[mask] / b[mask]
    return out


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def calculate_factor(df: pd.DataFrame) -> pd.Series:
    n = len(df)
    out = pd.Series(0.0, index=df.index)
    if n < 80:
        return out

    close = df["close"].astype(float).values
    open_ = df["open"].astype(float).values if "open" in df else np.roll(close, 1)
    open_[0] = close[0]
    high = df["high"].astype(float).values if "high" in df else close
    low = df["low"].astype(float).values if "low" in df else close
    volume = df["volume"].astype(float).values if "volume" in df else np.ones(n)
    oi = df["open_interest"].ffill().astype(float).values if "open_interest" in df else np.zeros(n)
    funding = df["funding_rate"].fillna(0).astype(float).values if "funding_rate" in df else np.zeros(n)

    close_s = pd.Series(close)
    ema8 = close_s.ewm(span=8, min_periods=5).mean().values
    ema21 = close_s.ewm(span=21, min_periods=10).mean().values
    ema55 = close_s.ewm(span=55, min_periods=25).mean().values
    std21 = close_s.rolling(21, min_periods=10).std().fillna(0.0).values

    vol_ma20 = pd.Series(volume).rolling(20, min_periods=10).mean().fillna(0.0).values
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr14 = pd.Series(tr).rolling(14, min_periods=7).mean().fillna(0.0).values
    atr40 = pd.Series(tr).rolling(40, min_periods=20).mean().fillna(0.0).values
    atr_ratio = _safe_div(atr14, np.maximum(atr40, 1e-12))

    hh20 = pd.Series(high).rolling(20, min_periods=10).max().fillna(0.0).values
    ll20 = pd.Series(low).rolling(20, min_periods=10).min().fillna(0.0).values
    range20 = np.maximum(1e-12, hh20 - ll20)
    close_pos = _safe_div(close - ll20, range20)  # [0, 1]

    raw = np.zeros(n, dtype=float)
    min_output = _env_float("MEME_ALPHA_ENSEMBLE_MIN_OUTPUT", 0.22)

    for i in range(60, n):
        ret3 = _pct(close[i], close[i - 3])
        ret5 = _pct(close[i], close[i - 5])
        ret10 = _pct(close[i], close[i - 10])
        ret20 = _pct(close[i], close[i - 20])
        ret60 = _pct(close[i], close[i - 60])
        oi_roc5 = _pct(oi[i], oi[i - 5]) if oi[i - 5] > 0 else 0.0
        oi_roc20 = _pct(oi[i], oi[i - 20]) if oi[i - 20] > 0 else 0.0

        ema_fast_diff = _pct(ema8[i], ema21[i]) if ema21[i] > 0 else 0.0
        ema_slow_diff = _pct(ema21[i], ema55[i]) if ema55[i] > 0 else 0.0
        vol_ratio = volume[i] / vol_ma20[i] if vol_ma20[i] > 0 else 1.0

        bar_range = max(1e-12, high[i] - low[i])
        upper_wick = (high[i] - max(open_[i], close[i])) / bar_range
        lower_wick = (min(open_[i], close[i]) - low[i]) / bar_range

        trend_regime = _clip01(abs(ret60) / 30.0 + abs(ema_fast_diff) / 0.55)
        range_regime = _clip01(1.0 - trend_regime)
        vol_regime = _clip01((atr_ratio[i] - 0.9) / 0.7)

        c1_trend = np.tanh((ema_fast_diff + 0.7 * ema_slow_diff) / 0.45) * (0.16 + 0.12 * trend_regime)
        breakout_bias = (close_pos[i] - 0.5) * 2.0
        c2_breakout = np.tanh(breakout_bias * max(0.6, min(1.6, atr_ratio[i]))) * (0.10 + 0.06 * trend_regime)
        c3_momentum = np.tanh((ret20 / 8.0 + ret60 / 22.0 + ret5 / 5.0) / 2.0) * (0.14 + 0.06 * trend_regime)

        z = (close[i] - ema21[i]) / max(std21[i], close[i] * 0.004, 1e-9)
        c4_mean_revert = -np.tanh(z / 2.0) * (0.12 + 0.08 * range_regime)

        vol_direction = np.tanh(ret3 / 2.5)
        c5_vol_exp = vol_direction * max(0.0, min(1.0, atr_ratio[i] - 1.0)) * (0.10 + 0.06 * vol_regime)

        funding_bias = np.clip(-funding[i] / 0.25, -1.0, 1.0)
        funding_confirm = np.tanh(ret10 / 4.0)
        c6_funding = np.tanh((funding_bias + 0.6 * funding_confirm) / 1.6) * 0.10

        oi_press = np.tanh((oi_roc5 / 8.0 + oi_roc20 / 20.0) / 1.6)
        liquidation_bias = 0.0
        if ret10 < -2.5:
            liquidation_bias = min(1.0, max(0.0, -oi_roc5 / 12.0))
        elif ret10 > 2.5:
            liquidation_bias = -min(1.0, max(0.0, -oi_roc5 / 12.0))
        c7_oi = np.tanh(oi_press + liquidation_bias * np.sign(ret10)) * 0.10

        crash = 0.0
        if ret5 < -2.0 and ret20 < -6.0:
            close_near_low = (close[i] - low[i]) / bar_range
            oi_fuel = max(0.0, oi_roc5 / 10.0, -oi_roc5 / 15.0)
            crash = -min(1.0, (abs(ret5) / 8.0 + abs(ret20) / 16.0) * 0.5 + oi_fuel * 0.5)
            if close_near_low > 0.55:
                crash *= 0.75
        c8_crash_cont = crash * 0.14

        squeeze_rev = 0.0
        if abs(ret20) > 9.0 and atr_ratio[i] > 1.2:
            if ret5 > 0 and upper_wick > 0.45:
                squeeze_rev = -min(1.0, upper_wick * 1.2)
            elif ret5 < 0 and lower_wick > 0.45:
                squeeze_rev = min(1.0, lower_wick * 1.2)
        c9_squeeze_reversal = squeeze_rev * 0.08

        wick_imbalance = np.clip((lower_wick - upper_wick) * 1.4, -1.0, 1.0)
        c10_micro = wick_imbalance * (0.04 + 0.04 * range_regime)

        score = (
            c1_trend
            + c2_breakout
            + c3_momentum
            + c4_mean_revert
            + c5_vol_exp
            + c6_funding
            + c7_oi
            + c8_crash_cont
            + c9_squeeze_reversal
            + c10_micro
        )

        if vol_ratio < 0.6 and abs(score) > 0.4:
            score *= 0.8
        if vol_ratio > 2.8 and abs(ret5) > 8:
            score *= 0.85

        raw[i] = float(np.clip(score, -1.0, 1.0))

    smooth = pd.Series(raw).ewm(span=4, min_periods=3).mean().values
    for i in range(60, n):
        val = float(np.clip(smooth[i], -1.0, 1.0))
        if abs(val) >= min_output:
            out.iloc[i] = val

    return out


def _ewm_adjusted_last(values: np.ndarray, span: int, min_periods: int) -> float:
    alpha = 2.0 / (span + 1.0)
    decay = 1.0 - alpha
    numerator = 0.0
    denominator = 0.0
    count = 0
    result = float("nan")
    for raw in values:
        value = 0.0 if not np.isfinite(raw) else float(raw)
        numerator = value + decay * numerator
        denominator = 1.0 + decay * denominator
        count += 1
        if count >= min_periods:
            result = numerator / denominator
    return result


def calculate_last_factor(df: pd.DataFrame) -> float:
    n = len(df)
    if n < 80:
        return 0.0

    close = df["close"].astype(float).values
    open_ = df["open"].astype(float).values if "open" in df else np.roll(close, 1)
    open_[0] = close[0]
    high = df["high"].astype(float).values if "high" in df else close
    low = df["low"].astype(float).values if "low" in df else close
    volume = df["volume"].astype(float).values if "volume" in df else np.ones(n)
    oi = df["open_interest"].ffill().astype(float).values if "open_interest" in df else np.zeros(n)
    funding = df["funding_rate"].fillna(0).astype(float).values if "funding_rate" in df else np.zeros(n)

    close_s = pd.Series(close)
    ema8 = close_s.ewm(span=8, min_periods=5).mean().values
    ema21 = close_s.ewm(span=21, min_periods=10).mean().values
    ema55 = close_s.ewm(span=55, min_periods=25).mean().values
    std21 = close_s.rolling(21, min_periods=10).std().fillna(0.0).values

    vol_ma20 = pd.Series(volume).rolling(20, min_periods=10).mean().fillna(0.0).values
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr14 = pd.Series(tr).rolling(14, min_periods=7).mean().fillna(0.0).values
    atr40 = pd.Series(tr).rolling(40, min_periods=20).mean().fillna(0.0).values
    atr_ratio = _safe_div(atr14, np.maximum(atr40, 1e-12))

    hh20 = pd.Series(high).rolling(20, min_periods=10).max().fillna(0.0).values
    ll20 = pd.Series(low).rolling(20, min_periods=10).min().fillna(0.0).values
    range20 = np.maximum(1e-12, hh20 - ll20)
    close_pos = _safe_div(close - ll20, range20)

    raw = np.zeros(n, dtype=float)
    min_output = _env_float("MEME_ALPHA_ENSEMBLE_MIN_OUTPUT", 0.22)

    for i in range(60, n):
        ret3 = _pct(close[i], close[i - 3])
        ret5 = _pct(close[i], close[i - 5])
        ret10 = _pct(close[i], close[i - 10])
        ret20 = _pct(close[i], close[i - 20])
        ret60 = _pct(close[i], close[i - 60])
        oi_roc5 = _pct(oi[i], oi[i - 5]) if oi[i - 5] > 0 else 0.0
        oi_roc20 = _pct(oi[i], oi[i - 20]) if oi[i - 20] > 0 else 0.0

        ema_fast_diff = _pct(ema8[i], ema21[i]) if ema21[i] > 0 else 0.0
        ema_slow_diff = _pct(ema21[i], ema55[i]) if ema55[i] > 0 else 0.0
        vol_ratio = volume[i] / vol_ma20[i] if vol_ma20[i] > 0 else 1.0

        bar_range = max(1e-12, high[i] - low[i])
        upper_wick = (high[i] - max(open_[i], close[i])) / bar_range
        lower_wick = (min(open_[i], close[i]) - low[i]) / bar_range

        trend_regime = _clip01(abs(ret60) / 30.0 + abs(ema_fast_diff) / 0.55)
        range_regime = _clip01(1.0 - trend_regime)
        vol_regime = _clip01((atr_ratio[i] - 0.9) / 0.7)

        c1_trend = np.tanh((ema_fast_diff + 0.7 * ema_slow_diff) / 0.45) * (0.16 + 0.12 * trend_regime)
        breakout_bias = (close_pos[i] - 0.5) * 2.0
        c2_breakout = np.tanh(breakout_bias * max(0.6, min(1.6, atr_ratio[i]))) * (0.10 + 0.06 * trend_regime)
        c3_momentum = np.tanh((ret20 / 8.0 + ret60 / 22.0 + ret5 / 5.0) / 2.0) * (0.14 + 0.06 * trend_regime)

        z = (close[i] - ema21[i]) / max(std21[i], close[i] * 0.004, 1e-9)
        c4_mean_revert = -np.tanh(z / 2.0) * (0.12 + 0.08 * range_regime)

        vol_direction = np.tanh(ret3 / 2.5)
        c5_vol_exp = vol_direction * max(0.0, min(1.0, atr_ratio[i] - 1.0)) * (0.10 + 0.06 * vol_regime)

        funding_bias = np.clip(-funding[i] / 0.25, -1.0, 1.0)
        funding_confirm = np.tanh(ret10 / 4.0)
        c6_funding = np.tanh((funding_bias + 0.6 * funding_confirm) / 1.6) * 0.10

        oi_press = np.tanh((oi_roc5 / 8.0 + oi_roc20 / 20.0) / 1.6)
        liquidation_bias = 0.0
        if ret10 < -2.5:
            liquidation_bias = min(1.0, max(0.0, -oi_roc5 / 12.0))
        elif ret10 > 2.5:
            liquidation_bias = -min(1.0, max(0.0, -oi_roc5 / 12.0))
        c7_oi = np.tanh(oi_press + liquidation_bias * np.sign(ret10)) * 0.10

        crash = 0.0
        if ret5 < -2.0 and ret20 < -6.0:
            close_near_low = (close[i] - low[i]) / bar_range
            oi_fuel = max(0.0, oi_roc5 / 10.0, -oi_roc5 / 15.0)
            crash = -min(1.0, (abs(ret5) / 8.0 + abs(ret20) / 16.0) * 0.5 + oi_fuel * 0.5)
            if close_near_low > 0.55:
                crash *= 0.75
        c8_crash_cont = crash * 0.14

        squeeze_rev = 0.0
        if abs(ret20) > 9.0 and atr_ratio[i] > 1.2:
            if ret5 > 0 and upper_wick > 0.45:
                squeeze_rev = -min(1.0, upper_wick * 1.2)
            elif ret5 < 0 and lower_wick > 0.45:
                squeeze_rev = min(1.0, lower_wick * 1.2)
        c9_squeeze_reversal = squeeze_rev * 0.08

        wick_imbalance = np.clip((lower_wick - upper_wick) * 1.4, -1.0, 1.0)
        c10_micro = wick_imbalance * (0.04 + 0.04 * range_regime)

        score = (
            c1_trend
            + c2_breakout
            + c3_momentum
            + c4_mean_revert
            + c5_vol_exp
            + c6_funding
            + c7_oi
            + c8_crash_cont
            + c9_squeeze_reversal
            + c10_micro
        )

        if vol_ratio < 0.6 and abs(score) > 0.4:
            score *= 0.8
        if vol_ratio > 2.8 and abs(ret5) > 8:
            score *= 0.85

        raw[i] = float(np.clip(score, -1.0, 1.0))

    value = float(np.clip(_ewm_adjusted_last(raw, span=4, min_periods=3), -1.0, 1.0))
    return value if abs(value) >= min_output else 0.0
