"""Leader-style orderflow rejection short factor.

External meme lead-trader samples are dominated by short, low-leverage shorts
opened after a fast pump starts to fail.  OHLCV-only rejection rules were not
enough in execution tests, so this research factor requires Binance kline
orderflow proxy fields: quote volume, trade count, and taker-buy quote volume.

It emits only short signals and returns zero when those orderflow fields are
missing, keeping old OHLCV caches compatible and avoiding fake precision.
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


def _safe_series(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").ffill().fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _clip01(value: pd.Series) -> pd.Series:
    return value.clip(lower=0.0, upper=1.0)


def calculate_factor(df: pd.DataFrame) -> pd.Series:
    if df is None or len(df) < 90:
        return pd.Series(0.0, index=getattr(df, "index", None))
    required = {"quote_volume", "taker_buy_quote", "trades"}
    if not required.issubset(df.columns):
        return pd.Series(0.0, index=df.index)

    open_ = _safe_series(df, "open")
    high = _safe_series(df, "high")
    low = _safe_series(df, "low")
    close = _safe_series(df, "close")
    volume = _safe_series(df, "volume", 1.0)
    quote_volume = _safe_series(df, "quote_volume", 0.0)
    taker_buy_quote = _safe_series(df, "taker_buy_quote", 0.0)
    trades = _safe_series(df, "trades", 0.0)
    oi = _safe_series(df, "open_interest")
    funding = _safe_series(df, "funding_rate")

    min_output = _env_float("MEME_LEADER_ORDERFLOW_REJECTION_MIN_OUTPUT", 0.24)
    max_output = _env_float("MEME_LEADER_ORDERFLOW_REJECTION_MAX_OUTPUT", 0.86)
    min_vol_ratio = _env_float("MEME_LEADER_ORDERFLOW_REJECTION_MIN_VOL_RATIO", 1.2)
    min_ret5 = _env_float("MEME_LEADER_ORDERFLOW_REJECTION_MIN_RET5_PCT", 1.6)
    min_ret15 = _env_float("MEME_LEADER_ORDERFLOW_REJECTION_MIN_RET15_PCT", 2.4)
    max_ret5 = _env_float("MEME_LEADER_ORDERFLOW_REJECTION_MAX_RET5_PCT", 18.0)
    max_pullback_high20 = _env_float("MEME_LEADER_ORDERFLOW_REJECTION_MAX_PULLBACK_HIGH20_PCT", 3.5)
    max_taker_buy_ratio = _env_float("MEME_LEADER_ORDERFLOW_REJECTION_MAX_TAKER_BUY_RATIO", 0.56)
    min_taker_buy_drop = _env_float("MEME_LEADER_ORDERFLOW_REJECTION_MIN_TAKER_BUY_DROP", 0.04)
    max_close_pos = _env_float("MEME_LEADER_ORDERFLOW_REJECTION_MAX_CLOSE_POS", 0.72)
    min_upper_wick = _env_float("MEME_LEADER_ORDERFLOW_REJECTION_MIN_UPPER_WICK", 0.10)

    ret1 = close.pct_change(1) * 100.0
    ret3 = close.pct_change(3) * 100.0
    ret5 = close.pct_change(5) * 100.0
    ret15 = close.pct_change(15) * 100.0
    ret60 = close.pct_change(60) * 100.0

    high20 = high.rolling(20, min_periods=8).max()
    low20 = low.rolling(20, min_periods=8).min()
    from_high20 = (close / high20.replace(0.0, np.nan) - 1.0) * 100.0
    from_low20 = (close / low20.replace(0.0, np.nan) - 1.0) * 100.0

    vol_base = volume.rolling(30, min_periods=12).median().shift(1).replace(0.0, np.nan)
    vol_ratio = (volume / vol_base).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    quote_base = quote_volume.rolling(30, min_periods=12).median().shift(1).replace(0.0, np.nan)
    quote_ratio = (quote_volume / quote_base).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    taker_buy_ratio = (taker_buy_quote / quote_volume.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    taker_buy_ratio = taker_buy_ratio.fillna(0.5).clip(lower=0.0, upper=1.0)
    taker_buy_ma5 = taker_buy_ratio.rolling(5, min_periods=3).mean()
    taker_buy_ma15 = taker_buy_ratio.rolling(15, min_periods=8).mean()
    taker_buy_peak8 = taker_buy_ratio.rolling(8, min_periods=4).max()
    taker_drop = (taker_buy_peak8 - taker_buy_ratio).fillna(0.0)
    seller_pressure = (1.0 - taker_buy_ratio).fillna(0.5)
    buy_fade = (taker_buy_ma5 - taker_buy_ma15).fillna(0.0)

    bar_range = (high - low).replace(0.0, np.nan)
    upper_wick = (high - pd.concat([open_, close], axis=1).max(axis=1)) / bar_range
    close_pos = (close - low) / bar_range

    ema5 = close.ewm(span=5, min_periods=3, adjust=False).mean()
    ema13 = close.ewm(span=13, min_periods=6, adjust=False).mean()
    oi_roc5 = oi.pct_change(5).replace([np.inf, -np.inf], np.nan).fillna(0.0) * 100.0

    extension_score = pd.concat(
        [
            (ret3.fillna(0.0) - 0.6) / 4.0,
            (ret5.fillna(0.0) - min_ret5) / 8.0,
            (ret15.fillna(0.0) - min_ret15) / 14.0,
            (ret60.fillna(0.0) - 1.0) / 25.0,
            (from_low20.fillna(0.0) - 1.5) / 12.0,
        ],
        axis=1,
    ).max(axis=1)
    near_high_score = 1.0 - _clip01((-from_high20.fillna(-999.0)) / max(max_pullback_high20, 0.5))
    rejection_score = (
        0.42 * _clip01((upper_wick.fillna(0.0) - min_upper_wick) / 0.45)
        + 0.36 * _clip01((max_close_pos - close_pos.fillna(0.5)) / max(max_close_pos, 0.1))
        + 0.22 * (ret1.fillna(0.0) <= 0.25).astype(float)
    )
    orderflow_score = (
        0.45 * _clip01((seller_pressure - 0.46) / 0.22)
        + 0.35 * _clip01((taker_drop - min_taker_buy_drop) / 0.22)
        + 0.20 * _clip01((-buy_fade) / 0.12)
    )
    activity_score = (
        0.50 * _clip01((vol_ratio - min_vol_ratio) / 6.0)
        + 0.30 * _clip01((quote_ratio - min_vol_ratio) / 6.0)
        + 0.20 * _clip01(trades.pct_change(5).fillna(0.0) / 2.0)
    )
    crowding_score = 0.65 * _clip01(oi_roc5 / 12.0) + 0.35 * _clip01(funding.fillna(0.0) / 0.20)
    ema_failure = ((close <= ema5 * 1.003) | (ema5 <= ema13 * 1.010)).astype(float)

    score = (
        0.24 * _clip01(extension_score)
        + 0.18 * near_high_score
        + 0.20 * rejection_score
        + 0.22 * orderflow_score
        + 0.10 * activity_score
        + 0.04 * crowding_score
        + 0.02 * ema_failure
    )

    valid = (
        (vol_ratio >= min_vol_ratio)
        & (quote_ratio >= min_vol_ratio)
        & ((ret5 >= min_ret5) | (ret15 >= min_ret15))
        & (ret5 <= max_ret5)
        & (from_high20 >= -abs(max_pullback_high20))
        & (from_low20 >= 1.0)
        & ((upper_wick >= min_upper_wick) | (close_pos <= max_close_pos) | (ret1 <= 0.25))
        & ((taker_buy_ratio <= max_taker_buy_ratio) | (taker_drop >= min_taker_buy_drop))
        & (close >= ema13 * 0.985)
    )

    out = pd.Series(0.0, index=df.index)
    out = out.mask(valid, -score.clip(lower=0.0, upper=max_output))
    return out.where(out.abs() >= min_output, 0.0).fillna(0.0)
