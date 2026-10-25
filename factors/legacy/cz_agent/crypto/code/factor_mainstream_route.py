"""Mainstream multi-symbol route factor for the Liulian architecture.

This module ports the strongest old main-strategy OHLCV routes into the
independent Liulian strategy surface.  It reads only bar data available at the
decision time; leader history remains labels and benchmarks only.
"""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

from cz.components.factor.manager import get_factors_base_dir


MAINSTREAM_ROUTE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "squeeze_momentum_breakout_1h",
        "code_file": "squeeze_momentum_breakout_1h.py",
        "timeframe": "1h",
        "symbols": ("BTCUSDT", "ETHUSDT", "HYPEUSDT", "LTCUSDT"),
        "hold_minutes": 360.0,
        "weight": 1.0,
    },
    {
        "name": "ema_shrink_pullback_4h",
        "code_file": "ema_shrink_pullback_4h.py",
        "timeframe": "4h",
        "symbols": ("ETHUSDT", "HYPEUSDT", "LTCUSDT", "BNBUSDT"),
        "hold_minutes": 1440.0,
        "weight": 1.0,
    },
    {
        "name": "donchian_volatility_breakout_4h",
        "code_file": "donchian_volatility_breakout_4h.py",
        "timeframe": "4h",
        "symbols": ("BTCUSDT", "SOLUSDT"),
        "hold_minutes": 1440.0,
        "weight": 1.0,
    },
    {
        "name": "squeeze_momentum_breakout_1h_short_guard",
        "code_file": "squeeze_momentum_breakout_1h_short_guard.py",
        "timeframe": "1h",
        "symbols": ("BNBUSDT",),
        "directions": ("SHORT",),
        "hold_minutes": 360.0,
        "weight": 1.0,
    },
)


def calculate_factor(df: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.Series:
    """Return the weighted latest-route signal on the caller's bar index."""

    frame = calculate_mainstream_route_frame(df, symbol=str((params or {}).get("symbol", "")))
    if frame.empty or "signal" not in frame:
        return pd.Series(0.0, index=getattr(df, "index", None), dtype=float)
    return frame["signal"].reindex(getattr(df, "index", frame.index)).fillna(0.0).astype(float)


def calculate_mainstream_route_frame(
    df: pd.DataFrame,
    *,
    symbol: str = "",
    params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Compute old-main-route pulse signals and expose interpretable columns."""

    if df is None or df.empty:
        return pd.DataFrame(index=getattr(df, "index", None))
    bars = _clean_ohlcv(df)
    if bars.empty:
        return pd.DataFrame(index=bars.index)
    symbol_key = _normalize_symbol(symbol)
    route_frames: list[pd.DataFrame] = []
    for spec in MAINSTREAM_ROUTE_SPECS:
        if symbol_key and symbol_key not in set(spec.get("symbols", ())):
            continue
        factor_frame = _route_signal_frame(bars, spec, params=params)
        if not factor_frame.empty:
            route_frames.append(factor_frame)
    if not route_frames:
        return pd.DataFrame({"signal": pd.Series(0.0, index=bars.index)}, index=bars.index)
    pulses = pd.concat(route_frames, ignore_index=False, sort=False)
    pulses = pulses[pulses["signal"].abs().gt(0.0)].copy()
    if pulses.empty:
        return pd.DataFrame({"signal": pd.Series(0.0, index=bars.index)}, index=bars.index)
    pulses["_abs_signal"] = pd.to_numeric(pulses["signal"], errors="coerce").abs()
    pulses = pulses.sort_values(["_abs_signal", "factor"], ascending=[False, True])
    pulses = pulses[~pulses.index.duplicated(keep="first")].sort_index()
    return pulses.drop(columns=["_abs_signal"], errors="ignore")


def _route_signal_frame(
    bars: pd.DataFrame,
    spec: dict[str, Any],
    *,
    params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    timeframe = str(spec.get("timeframe") or "5m")
    factor_bars = _resample_ohlcv(bars, timeframe)
    if factor_bars.empty:
        return pd.DataFrame()
    func = _load_factor_func(str(spec["code_file"]), str(spec["name"]))
    values = func(factor_bars.copy(), dict(params or {}))
    series = values if isinstance(values, pd.Series) else pd.Series(values, index=factor_bars.index)
    series = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    series.index = pd.DatetimeIndex(series.index) + pd.Timedelta(minutes=_timeframe_minutes(timeframe))
    series = series[series.abs().gt(0.0)]
    if series.empty:
        return pd.DataFrame()
    allowed_directions = tuple(str(item).upper() for item in spec.get("directions", ()))
    if allowed_directions:
        keep = []
        for value in series:
            direction = "LONG" if value > 0 else "SHORT"
            keep.append(direction in allowed_directions)
        series = series.loc[keep]
    if series.empty:
        return pd.DataFrame()
    aligned_price = bars["close"].reindex(series.index, method="ffill")
    out = pd.DataFrame(
        {
            "signal": series * float(spec.get("weight", 1.0)),
            "factor": str(spec["name"]),
            "timeframe": timeframe,
            "route_hold_minutes": float(spec.get("hold_minutes", 360.0)),
            "close": aligned_price,
        },
        index=series.index,
    )
    return out[out["close"].gt(0.0).fillna(False)]


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy().sort_index()
    frame.index = pd.to_datetime(frame.index, errors="coerce")
    frame = frame[frame.index.notna()].copy()
    required = ("open", "high", "low", "close", "volume")
    for col in required:
        if col not in frame:
            frame[col] = frame["close"] if "close" in frame else np.nan
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.dropna(subset=["open", "high", "low", "close"])[list(required)]


def _resample_ohlcv(data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    rule = {
        "5m": "5min",
        "15m": "15min",
        "1h": "1h",
        "4h": "4h",
    }.get(str(timeframe), str(timeframe))
    return (
        data.resample(rule, label="left", closed="left")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )


def _timeframe_minutes(timeframe: str) -> int:
    text = str(timeframe or "5m").strip().lower()
    if text.endswith("m"):
        return max(1, int(float(text[:-1] or 1)))
    if text.endswith("h"):
        return max(1, int(float(text[:-1] or 1) * 60))
    return 5


def _normalize_symbol(symbol: str) -> str:
    return str(symbol or "").strip().upper().replace("/", "").replace("-", "")


@lru_cache(maxsize=8)
def _load_factor_func(code_file: str, factor_name: str):
    path = get_factors_base_dir("crypto") / "code" / code_file
    module = _load_module(path, f"cz_liulian_mainstream_{factor_name}")
    func = getattr(module, "calculate_factor", None)
    if not callable(func):
        raise ImportError(f"{code_file} missing calculate_factor")
    return func


def _load_module(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load factor module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
