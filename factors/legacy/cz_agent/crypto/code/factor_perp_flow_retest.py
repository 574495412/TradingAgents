"""Perp flow retest factor for meme research.

This candidate targets a common perp microstructure pattern: a fast range
break or liquidity sweep, followed by immediate acceptance back through a
VWAP/EMA area with aligned taker flow.  It is intentionally lag-safe and uses
only fields already available to live and backtest: OHLCV, taker-buy quote
volume, open interest, and funding.
"""

from __future__ import annotations

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


def _safe_series(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in df:
        return df[name].astype(float).replace([np.inf, -np.inf], np.nan).fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def calculate_factor(df: pd.DataFrame) -> pd.Series:
    n = len(df)
    out = pd.Series(0.0, index=df.index)
    if n < 80:
        return out

    close = _safe_series(df, "close").values
    open_ = _safe_series(df, "open").values if "open" in df else np.roll(close, 1)
    open_[0] = close[0]
    high = _safe_series(df, "high").values if "high" in df else close
    low = _safe_series(df, "low").values if "low" in df else close
    volume = _safe_series(df, "volume", 1.0).values
    quote_volume = _safe_series(df, "quote_volume").values
    taker_buy_quote = _safe_series(df, "taker_buy_quote").values
    oi = _safe_series(df, "open_interest").ffill().values
    funding = _safe_series(df, "funding_rate").values

    close_s = pd.Series(close)
    ema8 = close_s.ewm(span=8, min_periods=4).mean().values
    ema21 = close_s.ewm(span=21, min_periods=10).mean().values
    vol_base = pd.Series(volume).rolling(30, min_periods=12).median().replace(0.0, np.nan).values
    qv_base = pd.Series(quote_volume).rolling(30, min_periods=12).median().replace(0.0, np.nan).values
    hh20 = pd.Series(high).rolling(20, min_periods=10).max().values
    ll20 = pd.Series(low).rolling(20, min_periods=10).min().values

    # Approximate anchored intraday VWAP with rolling quote/base volume.  This
    # avoids session assumptions in 24/7 crypto and stays deterministic.
    base_notional = quote_volume.copy()
    missing_qv = base_notional <= 0
    base_notional[missing_qv] = close[missing_qv] * volume[missing_qv]
    rolling_notional = pd.Series(base_notional).rolling(20, min_periods=8).sum().values
    rolling_volume = pd.Series(volume).rolling(20, min_periods=8).sum().replace(0.0, np.nan).values
    vwap20 = rolling_notional / rolling_volume

    min_output = _env_float("MEME_PERP_FLOW_RETEST_MIN_OUTPUT", 0.26)
    min_vol_ratio = _env_float("MEME_PERP_FLOW_RETEST_MIN_VOL_RATIO", 1.45)
    min_taker_skew = _env_float("MEME_PERP_FLOW_RETEST_MIN_TAKER_SKEW", 0.08)
    max_ret5 = _env_float("MEME_PERP_FLOW_RETEST_MAX_RET5_PCT", 11.0)
    max_sweep_depth = _env_float("MEME_PERP_FLOW_RETEST_MAX_SWEEP_DEPTH_PCT", 7.0)

    raw = np.zeros(n)
    for i in range(35, n):
        ret1 = _pct(close[i], close[i - 1])
        ret3 = _pct(close[i], close[i - 3])
        ret5 = _pct(close[i], close[i - 5])
        ret15 = _pct(close[i], close[i - 15])
        vol_ratio = volume[i] / vol_base[i] if vol_base[i] and vol_base[i] > 0 else 1.0
        qv_ratio = base_notional[i] / qv_base[i] if qv_base[i] and qv_base[i] > 0 else vol_ratio
        oi_roc3 = _pct(oi[i], oi[i - 3]) if oi[i - 3] > 0 else 0.0
        ema_bias = _pct(ema8[i], ema21[i]) if ema21[i] > 0 else 0.0
        vwap_bias = _pct(close[i], vwap20[i]) if vwap20[i] and vwap20[i] > 0 else 0.0

        notional = max(base_notional[i], 1e-12)
        taker_ratio = taker_buy_quote[i] / notional if taker_buy_quote[i] > 0 else 0.5
        taker_skew = taker_ratio - 0.5

        prev_hh = max(float(hh20[i - 1]), 1e-12)
        prev_ll = max(float(ll20[i - 1]), 1e-12)
        swept_low = low[i] < prev_ll * 0.998
        swept_high = high[i] > prev_hh * 1.002
        broke_high = close[i] > prev_hh * 1.001
        broke_low = close[i] < prev_ll * 0.999
        sweep_low_depth = (prev_ll / max(low[i], 1e-12) - 1.0) * 100.0
        sweep_high_depth = (high[i] / prev_hh - 1.0) * 100.0

        bar_range = max(1e-12, high[i] - low[i])
        close_pos = (close[i] - low[i]) / bar_range
        upper_wick = (high[i] - max(open_[i], close[i])) / bar_range
        lower_wick = (min(open_[i], close[i]) - low[i]) / bar_range

        long_score = 0.0
        long_evidence = 0
        if (swept_low and close_pos > 0.62 and ret1 > 0.10) or (broke_high and ret3 > 0.45):
            long_score += 0.24
            long_evidence += 1
        if ret3 > 0.35 and ret15 > -1.5 and ema_bias > -0.10 and vwap_bias > -0.35:
            long_score += min(0.20, max(ret3, 0.0) / 5.0 * 0.20)
            long_evidence += 1
        if vol_ratio >= min_vol_ratio or qv_ratio >= min_vol_ratio:
            long_score += min(0.18, (max(vol_ratio, qv_ratio) - 1.0) / 5.0 * 0.18)
            long_evidence += 1
        if taker_skew >= min_taker_skew:
            long_score += min(0.18, taker_skew / 0.32 * 0.18)
            long_evidence += 1
        if oi_roc3 > 0.35:
            long_score += min(0.12, oi_roc3 / 7.0 * 0.12)
            long_evidence += 1
        if funding[i] < -0.04 and ret3 > 0:
            long_score += 0.06
            long_evidence += 1
        if upper_wick > 0.52 or ret5 > max_ret5 or sweep_low_depth > max_sweep_depth:
            long_score *= 0.45

        short_score = 0.0
        short_evidence = 0
        if (swept_high and close_pos < 0.38 and ret1 < -0.10) or (broke_low and ret3 < -0.45):
            short_score += 0.24
            short_evidence += 1
        if ret3 < -0.35 and ret15 < 1.5 and ema_bias < 0.10 and vwap_bias < 0.35:
            short_score += min(0.20, abs(ret3) / 5.0 * 0.20)
            short_evidence += 1
        if vol_ratio >= min_vol_ratio or qv_ratio >= min_vol_ratio:
            short_score += min(0.18, (max(vol_ratio, qv_ratio) - 1.0) / 5.0 * 0.18)
            short_evidence += 1
        if taker_skew <= -min_taker_skew:
            short_score += min(0.18, abs(taker_skew) / 0.32 * 0.18)
            short_evidence += 1
        if abs(oi_roc3) > 0.45:
            short_score += min(0.12, abs(oi_roc3) / 7.0 * 0.12)
            short_evidence += 1
        if funding[i] > 0.04 and ret3 < 0:
            short_score += 0.06
            short_evidence += 1
        if lower_wick > 0.52 or abs(ret5) > max_ret5 or sweep_high_depth > max_sweep_depth:
            short_score *= 0.45

        if long_evidence >= 3 and long_score >= min_output and long_score >= short_score:
            raw[i] = min(1.0, long_score * 1.55)
        elif short_evidence >= 3 and short_score >= min_output:
            raw[i] = -min(1.0, short_score * 1.55)

    smooth = pd.Series(raw).ewm(span=2, min_periods=1).mean().values
    for i in range(35, n):
        value = float(np.clip(smooth[i], -1.0, 1.0))
        if abs(value) >= min_output:
            out.iloc[i] = value
    return out
