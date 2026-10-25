"""Reusable causal market-feature engine for research and factor evaluation."""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from factors.crypto.code.weijie.common import (
    weijie_asset_group,
    weijie_market_prior_group,
)

_KLINE_REQUEST_LOCK = threading.Lock()
_LAST_KLINE_REQUEST_MONOTONIC = 0.0

SYMBOL_CONTEXT_SYMBOLS = (
    "ETHUSDT",
    "BTCUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "BCHUSDT",
    "LTCUSDT",
    "ETCUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "CFXUSDT",
    "FILUSDT",
    "PAXGUSDT",
    "DYDXUSDT",
    "LINKUSDT",
    "RAVEUSDT",
)


SYMBOL_CONTEXT_FEATURES = (
    *(
        f"symbol_is_{symbol.lower().replace('usdt', '')}"
        for symbol in SYMBOL_CONTEXT_SYMBOLS
    ),
    "family_eth_main",
    "family_major_basket",
    "family_alt_basket",
    "family_gold_defensive",
    "session_asia_00_08",
    "session_europe_08_16",
    "session_us_16_24",
)


MULTI_TIMEFRAME_FEATURES = (
    "ret12",
    "ret36",
    "ret72",
    "ret144",
    "ret288",
    "from_high12",
    "from_low12",
    "from_high144",
    "from_low144",
    "from_high288",
    "from_low288",
    "ema34_deviation",
    "ema89_deviation",
    "ema144_deviation",
    "ema34_slope12_pct",
    "ema89_slope36_pct",
    "ema144_slope72_pct",
    "vol_ratio_med96",
    "range_ratio_med96",
    "atr_ratio_med96",
    "market_ret12_mean",
    "market_ret36_mean",
    "market_ret72_mean",
    "market_ret144_mean",
    "market_ret288_mean",
    "rel_ret12",
    "rel_ret36",
    "rel_ret72",
    "rel_ret144",
    "rel_ret288",
    "ret12_rank_pct",
    "ret36_rank_pct",
    "ret72_rank_pct",
    "ret144_rank_pct",
    "ret288_rank_pct",
    "from_high144_rank_pct",
    "from_low144_rank_pct",
    "from_high288_rank_pct",
    "from_low288_rank_pct",
    "tf15_ret4",
    "tf15_ret16",
    "tf15_rsi14",
    "tf15_ema21_slope4_pct",
    "tf15_adx14",
    "tf1h_ret4",
    "tf1h_ret12",
    "tf1h_rsi14",
    "tf1h_ema21_slope3_pct",
    "tf1h_adx14",
)


KLINE_INDICATOR_FEATURES = (
    "rsi7",
    "rsi14",
    "rsi28",
    "rsi14_delta5",
    "macd_pct",
    "macd_signal_pct",
    "macd_hist_pct",
    "ema8_dev_pct",
    "ema21_dev_pct",
    "ema55_dev_pct",
    "ema8_21_spread_pct",
    "ema21_55_spread_pct",
    "ema21_slope5_pct",
    "ema55_slope20_pct",
    "bb20_zscore",
    "bb20_width_pct",
    "bb20_width_ratio120",
    "bb20_pct_b",
    "bb20_upper_break_pct",
    "bb20_lower_break_pct",
    "donchian20_pos",
    "donchian60_pos",
    "stoch14_k",
    "stoch14_d",
    "cci20",
    "mfi14",
    "adx14",
    "plus_di14",
    "minus_di14",
    "di_diff14",
    "atr14_pct",
    "atr14_ratio120",
    "vwap20_dev_pct",
    "vwap60_dev_pct",
    "obv_z60",
    "obv_slope20",
    "trend_efficiency20",
    "trend_efficiency60",
    "candle_body_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "wick_imbalance",
    "range_expansion20",
    "volume_price_trend20",
    "squeeze20",
    "squeeze120_rank",
)


FEATURES = (
    "ret1",
    "ret3",
    "ret5",
    "ret20",
    "ret60",
    "ret240",
    "from_high20",
    "from_low20",
    "from_high60",
    "from_low60",
    "from_high240",
    "from_low240",
    "vol_ratio_med30",
    "vol_ratio_mean30",
    "quote_volume_log",
    "quote_volume_ratio_med30",
    "trades_log",
    "trades_ratio_med30",
    "taker_buy_ratio",
    "taker_buy_ratio_ma5",
    "taker_buy_ratio_ma30",
    "taker_buy_ratio_delta5",
    "seller_pressure",
    "bar_range_pct",
    "close_pos",
    "atr_pct",
    "ema20_deviation",
    "market_ret1_mean",
    "market_ret3_mean",
    "market_ret20_mean",
    "market_ret60_mean",
    "market_ret240_mean",
    "rel_ret1",
    "rel_ret3",
    "rel_ret20",
    "rel_ret60",
    "rel_ret240",
    "ret20_rank_pct",
    "ret60_rank_pct",
    "ret240_rank_pct",
    "from_high60_rank_pct",
    "from_low60_rank_pct",
    "quote_volume_rank_pct",
    "market_down_breadth20",
    "market_down_breadth60",
    "market_up_breadth20",
    "market_up_breadth60",
    *SYMBOL_CONTEXT_FEATURES,
    *MULTI_TIMEFRAME_FEATURES,
    "minute_sin",
    "minute_cos",
    "dow_sin",
    "dow_cos",
    *KLINE_INDICATOR_FEATURES,
)


def build_replica_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """从已收盘行情构建自主策略共享的价格、流动性和广度特征。"""

    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy().sort_index()
    for column in ("open", "high", "low", "close", "volume"):
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    close = out["close"].astype(float)
    high = out["high"].astype(float) if "high" in out else close
    low = out["low"].astype(float) if "low" in out else close
    volume = (
        out["volume"].astype(float)
        if "volume" in out
        else pd.Series(np.nan, index=out.index)
    )

    for bars in (1, 3, 5, 15, 20, 60, 240):
        out[f"ret{bars}"] = (close / close.shift(bars) - 1.0) * 100.0
    for bars in (20, 60, 240):
        minimum_periods = max(2, min(10, bars))
        rolling_high = high.rolling(
            bars,
            min_periods=minimum_periods,
        ).max()
        rolling_low = low.rolling(
            bars,
            min_periods=minimum_periods,
        ).min()
        out[f"from_high{bars}"] = (close / rolling_high - 1.0) * 100.0
        out[f"from_low{bars}"] = (close / rolling_low - 1.0) * 100.0

    prior_volume = volume.shift(1)
    out["vol_ratio_med30"] = (
        volume
        / prior_volume.rolling(
            30,
            min_periods=5,
        ).median()
    )
    out["vol_ratio_mean30"] = (
        volume
        / prior_volume.rolling(
            30,
            min_periods=5,
        ).mean()
    )
    out["quote_volume"] = volume * close
    out["bar_range_pct"] = ((high / low - 1.0) * 100.0).replace(
        [np.inf, -np.inf], np.nan,
    )
    out["close_pos"] = (close - low) / (high - low)
    ema20 = close.ewm(span=20, min_periods=10).mean()
    out["ema20_deviation"] = (close / ema20 - 1.0) * 100.0
    out["price_level"] = np.log(close.where(close > 0))

    minute_of_day = pd.Series(
        out.index.hour * 60 + out.index.minute,
        index=out.index,
    )
    out["minute_sin"] = np.sin(
        2.0 * np.pi * minute_of_day / 1440.0,
    )
    out["minute_cos"] = np.cos(
        2.0 * np.pi * minute_of_day / 1440.0,
    )
    day_of_week = pd.Series(out.index.dayofweek, index=out.index)
    out["dow_sin"] = np.sin(2.0 * np.pi * day_of_week / 7.0)
    out["dow_cos"] = np.cos(2.0 * np.pi * day_of_week / 7.0)

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr_pct"] = true_range.rolling(14, min_periods=5).mean() / close * 100.0
    return out


def load_feature_frames(
    *,
    symbols: list[str],
    interval: str,
    start_at: pd.Timestamp,
    end_at: pd.Timestamp,
    enrich_futures_klines: bool = False,
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        frame = pd.DataFrame()
        cols = ("open", "high", "low", "close", "volume")
        microstructure_cols = ("quote_volume", "trades", "taker_buy_quote")
        clean_symbol = _clean_symbol(symbol)
        cache_symbols = [clean_symbol]
        if clean_symbol.endswith("USDT"):
            cache_symbols.append(f"{clean_symbol[:-4]}-USDT")
        raw_cache_symbol = str(symbol).upper().replace("/", "-")
        if raw_cache_symbol not in cache_symbols:
            cache_symbols.append(raw_cache_symbol)
        cache_paths = [
            Path("pickle_cache") / f"ohlcv_{cache_symbol}_{interval}.pkl"
            for cache_symbol in cache_symbols
        ]
        for cache_path in cache_paths:
            if not cache_path.exists():
                continue
            try:
                raw = pd.read_pickle(cache_path)
            except Exception:
                raw = None
            if raw is not None and not raw.empty:
                cached = raw.copy()
                cached.index = pd.to_datetime(cached.index, errors="coerce")
                cached = cached[cached.index.notna()].sort_index()
                requested_start = pd.Timestamp(start_at) - pd.Timedelta(days=3)
                requested_end = pd.Timestamp(end_at)
                if all(col in cached.columns for col in cols) and _frame_covers_window(
                    cached,
                    start_at=requested_start,
                    end_at=requested_end,
                    interval_ms=_interval_to_milliseconds(interval),
                ):
                    available_cols = [
                        col
                        for col in (*cols, *microstructure_cols)
                        if col in cached.columns
                    ]
                    frame = cached.loc[
                        (cached.index >= requested_start)
                        & (cached.index <= requested_end),
                        available_cols,
                    ].copy()
                    break
        if frame.empty and enrich_futures_klines:
            frame = fetch_futures_full_klines(
                symbol=symbol,
                interval=interval,
                start_at=pd.Timestamp(start_at) - pd.Timedelta(days=3),
                end_at=pd.Timestamp(end_at),
            )
            if not frame.empty:
                frame = frame.loc[
                    :,
                    [
                        col
                        for col in frame.columns
                        if col
                        in set(cols) | {"quote_volume", "trades", "taker_buy_quote"}
                    ],
                ].copy()
        if frame.empty or not all(col in frame.columns for col in cols):
            continue
        needs_microstructure = any(
            col not in frame.columns for col in microstructure_cols
        )
        if enrich_futures_klines and needs_microstructure:
            enriched = fetch_futures_full_klines(
                symbol=symbol,
                interval=interval,
                start_at=pd.Timestamp(start_at) - pd.Timedelta(days=3),
                end_at=pd.Timestamp(end_at),
            )
            if not enriched.empty:
                for col in enriched.columns:
                    aligned = enriched[col].reindex(frame.index)
                    if col in frame.columns:
                        frame[col] = aligned.combine_first(frame[col])
                    else:
                        frame[col] = aligned
        if frame.empty:
            continue
        features = build_replica_feature_frame(frame)
        features["symbol"] = symbol
        features["bar_time"] = features.index
        _add_time_features(features)
        frames[symbol] = features.reset_index(drop=True)
    return frames


def fetch_futures_full_klines(
    *,
    symbol: str,
    interval: str,
    start_at: pd.Timestamp,
    end_at: pd.Timestamp,
    allow_spot_fallback: bool = False,
    allow_network: bool = True,
) -> pd.DataFrame:
    """读取合约 K 线；研究回放可显式允许现货备用来源。

    现货和 USDM 合约的微观结构、基差与可成交价格并不相同。因此备用
    来源只能用于研究性信号回放，调用方必须明确开启；它永远不应被实盘
    行情路径或正式的合约收益核验隐式使用。``allow_network=False`` 时只
    读取本地完整缓存，供限流期间的可重复研究使用。
    """

    clean_symbol = _clean_symbol(symbol)
    # ``_normalize_futures_klines`` intentionally stores market bars as
    # timezone-naive Shanghai wall-clock times.  Normalize query boundaries to
    # that same representation before probing the cache or slicing a freshly
    # downloaded frame.  Callers such as the forward-shadow CLI hold their
    # policy boundary in UTC, so leaving those timestamps aware here would
    # compare an aware boundary with a naive bar index after a successful
    # request.
    start_ts = _shanghai_naive_timestamp(start_at).floor("min")
    end_ts = _shanghai_naive_timestamp(end_at).ceil("min")
    cache_dir = Path("pickle_cache") / "weijie_full_klines"
    cache_dir.mkdir(parents=True, exist_ok=True)
    interval_ms = _interval_to_milliseconds(interval)
    if interval_ms <= 0:
        return pd.DataFrame()
    cache_stem = f"{clean_symbol}_{interval}_{start_ts:%Y%m%d%H%M}_{end_ts:%Y%m%d%H%M}"
    cached_futures = _read_complete_kline_cache(
        cache_dir / f"{cache_stem}_binance_usdm_futures.pkl",
        start_at=start_ts,
        end_at=end_ts,
        interval_ms=interval_ms,
        source="binance_usdm_futures",
    )
    if not cached_futures.empty:
        return cached_futures
    if allow_spot_fallback:
        cached_spot = _read_complete_kline_cache(
            cache_dir / f"{cache_stem}_binance_spot_research_fallback.pkl",
            start_at=start_ts,
            end_at=end_ts,
            interval_ms=interval_ms,
            source="binance_spot_research_fallback",
        )
        if not cached_spot.empty:
            return cached_spot

    futures = _fetch_binance_klines_window(
        symbol=clean_symbol,
        interval=interval,
        start_at=start_ts,
        end_at=end_ts,
        interval_ms=interval_ms,
        cache_dir=cache_dir,
        source="binance_usdm_futures",
        base_url="https://fapi.binance.com/fapi/v1/klines",
        allow_legacy_cache=True,
        allow_network=allow_network,
    )
    if not futures.empty:
        return futures
    if not allow_spot_fallback:
        return pd.DataFrame()
    return _fetch_binance_klines_window(
        symbol=clean_symbol,
        interval=interval,
        start_at=start_ts,
        end_at=end_ts,
        interval_ms=interval_ms,
        cache_dir=cache_dir,
        source="binance_spot_research_fallback",
        base_url="https://data-api.binance.vision/api/v3/klines",
        allow_legacy_cache=False,
        allow_network=allow_network,
    )


def _fetch_binance_klines_window(
    *,
    symbol: str,
    interval: str,
    start_at: pd.Timestamp,
    end_at: pd.Timestamp,
    interval_ms: int,
    cache_dir: Path,
    source: str,
    base_url: str,
    allow_legacy_cache: bool,
    allow_network: bool,
) -> pd.DataFrame:
    """读取完整来源窗口；命中部分缓存时只补请求缺失 K 线。"""

    cache_stem = f"{symbol}_{interval}_{start_at:%Y%m%d%H%M}_{end_at:%Y%m%d%H%M}"
    cache_path = cache_dir / f"{cache_stem}_{source}.pkl"
    rolling_path = cache_dir / f"{symbol}_{interval}_{source}_rolling.pkl"
    legacy_path = cache_dir / f"{cache_stem}.pkl"
    candidate_paths = {
        cache_path,
        rolling_path,
        *cache_dir.glob(f"{symbol}_{interval}_*.pkl"),
    }
    if allow_legacy_cache:
        candidate_paths.add(legacy_path)
    cached_frames = [
        frame
        for path in sorted(candidate_paths)
        if not (
            frame := _read_kline_cache_frame(
                path,
                source=source,
                allow_unattributed=allow_legacy_cache,
            )
        ).empty
    ]
    combined = _merge_kline_frames(cached_frames, source=source)
    if _frame_covers_window(
        combined,
        start_at=start_at,
        end_at=end_at,
        interval_ms=interval_ms,
    ):
        return _slice_kline_window(
            combined,
            start_at=start_at,
            end_at=end_at,
            source=source,
        )
    if not allow_network:
        return pd.DataFrame()

    fetched_frames: list[pd.DataFrame] = []
    for missing_start, missing_end in _missing_kline_ranges(
        combined,
        start_at=start_at,
        end_at=end_at,
        interval_ms=interval_ms,
    ):
        fetched = _request_binance_klines(
            symbol=symbol,
            interval=interval,
            start_at=missing_start,
            end_at=missing_end,
            interval_ms=interval_ms,
            base_url=base_url,
        )
        if fetched.empty:
            return pd.DataFrame()
        fetched_frames.append(fetched)
    frame = _merge_kline_frames(
        [combined, *fetched_frames],
        source=source,
    )
    if not _frame_covers_window(
        frame,
        start_at=start_at,
        end_at=end_at,
        interval_ms=interval_ms,
    ):
        return pd.DataFrame()
    output = _slice_kline_window(
        frame,
        start_at=start_at,
        end_at=end_at,
        source=source,
    )
    _write_kline_cache(cache_path, output, source=source)
    _write_kline_cache(rolling_path, frame, source=source)
    return output


def _request_binance_klines(
    *,
    symbol: str,
    interval: str,
    start_at: pd.Timestamp,
    end_at: pd.Timestamp,
    interval_ms: int,
    base_url: str,
) -> pd.DataFrame:
    """Request only one or more contiguous missing ranges from Binance."""

    start_ms = _shanghai_timestamp_ms(start_at)
    end_ms = _shanghai_timestamp_ms(end_at)
    rows: list[list[Any]] = []
    cursor = start_ms
    while cursor <= end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": int(cursor),
            "endTime": int(end_ms),
            "limit": 1500,
        }
        url = base_url + "?" + urllib.parse.urlencode(params)
        payload = None
        for attempt in range(3):
            try:
                _throttle_kline_request()
                with urllib.request.urlopen(url, timeout=15) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                break
            except Exception:
                if attempt < 2:
                    time.sleep(0.50 * (2**attempt))
        if payload is None:
            return pd.DataFrame()
        if not isinstance(payload, list) or not payload:
            break
        rows.extend(payload)
        last_open = int(payload[-1][0])
        next_cursor = last_open + interval_ms
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if cursor <= end_ms:
            time.sleep(0.03)
    frame = _normalize_futures_klines(rows)
    if frame.empty:
        return frame
    return frame.loc[
        (frame.index >= start_at) & (frame.index <= end_at)
    ].copy()


def _read_kline_cache_frame(
    path: Path,
    *,
    source: str,
    allow_unattributed: bool,
) -> pd.DataFrame:
    """Read a partial source-compatible cache without claiming full coverage."""

    if not path.exists():
        return pd.DataFrame()
    try:
        cached = pd.read_pickle(path)
    except Exception:
        return pd.DataFrame()
    if not isinstance(cached, pd.DataFrame) or cached.empty:
        return pd.DataFrame()
    cached_source = str(cached.attrs.get("market_data_source") or "")
    name = path.name
    if cached_source and cached_source != source:
        return pd.DataFrame()
    if not cached_source:
        if not allow_unattributed:
            return pd.DataFrame()
        if "binance_spot" in name and source != "binance_spot_research_fallback":
            return pd.DataFrame()
        if "binance_usdm" in name and source != "binance_usdm_futures":
            return pd.DataFrame()
    output = cached.copy()
    output.index = pd.to_datetime(output.index, errors="coerce")
    output = output.loc[output.index.notna()].sort_index()
    output.attrs["market_data_source"] = source
    return output


def _merge_kline_frames(
    frames: list[pd.DataFrame],
    *,
    source: str,
) -> pd.DataFrame:
    usable = [frame for frame in frames if frame is not None and not frame.empty]
    if not usable:
        return pd.DataFrame()
    output = pd.concat(usable, axis=0, sort=False)
    output = output.loc[~output.index.duplicated(keep="last")].sort_index()
    output.attrs["market_data_source"] = source
    return output


def _slice_kline_window(
    frame: pd.DataFrame,
    *,
    start_at: pd.Timestamp,
    end_at: pd.Timestamp,
    source: str,
) -> pd.DataFrame:
    output = frame.loc[
        (frame.index >= start_at) & (frame.index <= end_at)
    ].copy()
    output.attrs["market_data_source"] = source
    return output


def _missing_kline_ranges(
    frame: pd.DataFrame,
    *,
    start_at: pd.Timestamp,
    end_at: pd.Timestamp,
    interval_ms: int,
) -> tuple[tuple[pd.Timestamp, pd.Timestamp], ...]:
    interval = pd.Timedelta(milliseconds=int(interval_ms))
    expected = pd.date_range(
        start=pd.Timestamp(start_at).ceil(interval),
        end=pd.Timestamp(end_at).floor(interval),
        freq=interval,
    )
    if expected.empty:
        return ()
    available = (
        pd.DatetimeIndex([])
        if frame is None or frame.empty
        else pd.DatetimeIndex(
            pd.to_datetime(frame.index, errors="coerce"),
        ).dropna()
    )
    missing = expected[~expected.isin(available)]
    if missing.empty:
        return ()
    ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    range_start = pd.Timestamp(missing[0])
    previous = range_start
    for value in missing[1:]:
        current = pd.Timestamp(value)
        if current - previous > interval:
            ranges.append((range_start, previous))
            range_start = current
        previous = current
    ranges.append((range_start, previous))
    return tuple(ranges)


def _write_kline_cache(path: Path, frame: pd.DataFrame, *, source: str) -> None:
    output = frame.copy()
    output.attrs["market_data_source"] = source
    temp_path = path.with_suffix(f".{os.getpid()}.tmp")
    try:
        output.to_pickle(temp_path)
        temp_path.replace(path)
    except Exception:
        pass
    finally:
        temp_path.unlink(missing_ok=True)


def _throttle_kline_request() -> None:
    """Apply a process-wide, configurable gap between cache-miss requests."""

    global _LAST_KLINE_REQUEST_MONOTONIC
    try:
        minimum_gap = max(
            0.0,
            float(os.getenv("FACTOR_KLINE_REQUEST_MIN_INTERVAL_SECONDS", "0.10")),
        )
    except (TypeError, ValueError):
        minimum_gap = 0.10
    if minimum_gap <= 0.0:
        return
    with _KLINE_REQUEST_LOCK:
        now = time.monotonic()
        remaining = minimum_gap - (now - _LAST_KLINE_REQUEST_MONOTONIC)
        if remaining > 0.0:
            time.sleep(remaining)
        _LAST_KLINE_REQUEST_MONOTONIC = time.monotonic()


def _read_complete_kline_cache(
    path: Path,
    *,
    start_at: pd.Timestamp,
    end_at: pd.Timestamp,
    interval_ms: int,
    source: str,
) -> pd.DataFrame:
    """仅返回来源匹配、且覆盖完整请求窗口的缓存。"""

    if not path.exists():
        return pd.DataFrame()
    try:
        cached = pd.read_pickle(path)
    except Exception:
        return pd.DataFrame()
    if not isinstance(cached, pd.DataFrame):
        return pd.DataFrame()
    cached_source = cached.attrs.get("market_data_source")
    if cached_source not in {None, source}:
        return pd.DataFrame()
    if not _frame_covers_window(
        cached,
        start_at=start_at,
        end_at=end_at,
        interval_ms=interval_ms,
    ):
        return pd.DataFrame()
    output = cached.loc[(cached.index >= start_at) & (cached.index <= end_at),].copy()
    output.attrs["market_data_source"] = source
    return output


def _frame_covers_window(
    frame: pd.DataFrame,
    *,
    start_at: pd.Timestamp,
    end_at: pd.Timestamp,
    interval_ms: int,
) -> bool:
    if frame is None or frame.empty or int(interval_ms) <= 0:
        return False
    index = pd.to_datetime(frame.index, errors="coerce")
    index = pd.DatetimeIndex(index[index.notna()]).sort_values().unique()
    if index.empty:
        return False
    start_ts = pd.Timestamp(start_at)
    end_ts = pd.Timestamp(end_at)
    interval = pd.Timedelta(milliseconds=int(interval_ms))
    expected = pd.date_range(
        start=start_ts.ceil(interval),
        end=end_ts.floor(interval),
        freq=interval,
    )
    if expected.empty:
        return False
    return bool(expected.isin(index).all())


def build_symbol_feature_rows(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=["symbol", "bar_time", *FEATURES])
    rows = pd.concat(frames.values(), ignore_index=True, sort=False)
    rows["bar_time"] = pd.to_datetime(rows["bar_time"], errors="coerce")
    rows = (
        rows.dropna(subset=["bar_time", "symbol"])
        .sort_values(["symbol", "bar_time"])
        .reset_index(drop=True)
    )
    rows = _add_multi_timeframe_features(rows)
    rows = _add_resampled_timeframe_features(rows)
    rows = _add_single_symbol_liquidity_features(rows)
    rows = _add_kline_indicator_features(rows)
    rows = _add_symbol_context_features(rows)
    rows = _add_cross_sectional_features(rows)
    for feature in FEATURES:
        if feature not in rows:
            rows[feature] = 0.0
        rows[feature] = (
            pd.to_numeric(rows[feature], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
    return (
        rows.dropna(subset=["bar_time", "symbol"])
        .sort_values(["bar_time", "symbol"])
        .reset_index(drop=True)
    )


def build_compact_runtime_feature_rows(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a compact deterministic feature subset for resource-bound callers.

    ``build_symbol_feature_rows`` intentionally materializes the complete
    research feature surface.  Resource-bound consumers need a much smaller
    OHLCV/flow/indicator subset, so retaining every research column creates a
    large transient RSS spike with no effect on factor actions.  Keep the
    formulas identical to the full builder and omit unused columns.
    """

    if not frames:
        return pd.DataFrame()
    required_base = (
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trades",
        "taker_buy_quote",
        "ret1",
        "ret3",
        "ret20",
        "ret60",
        "atr_pct",
        "bar_range_pct",
        "close_pos",
        "symbol",
        "bar_time",
    )
    parts: list[pd.DataFrame] = []
    for frame in frames.values():
        if frame is None or frame.empty:
            continue
        available = [column for column in required_base if column in frame.columns]
        group = frame.loc[:, available].copy()
        group["bar_time"] = pd.to_datetime(group.get("bar_time"), errors="coerce")
        group = group.dropna(subset=["bar_time", "symbol"]).sort_values("bar_time")
        if group.empty:
            continue

        close = pd.to_numeric(group.get("close"), errors="coerce")
        high = pd.to_numeric(group.get("high"), errors="coerce")
        low = pd.to_numeric(group.get("low"), errors="coerce")
        open_ = pd.to_numeric(group.get("open"), errors="coerce")
        for window in (12, 72, 144, 288):
            group[f"ret{window}"] = _pct_change(close, close.shift(window))
        rolling_high12 = high.rolling(12, min_periods=4).max()
        group["from_high12"] = _pct_change(close, rolling_high12)

        quote_volume = pd.to_numeric(
            group.get("quote_volume"), errors="coerce",
        ).replace(
            [np.inf, -np.inf],
            np.nan,
        )
        group["quote_volume_ratio_med30"] = (
            quote_volume
            / quote_volume.shift(1)
            .rolling(30, min_periods=5)
            .median()
            .replace(0.0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)
        taker_buy_quote = pd.to_numeric(
            group.get("taker_buy_quote"),
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)
        group["taker_buy_ratio"] = (
            (taker_buy_quote / quote_volume.replace(0.0, np.nan))
            .clip(lower=0.0, upper=1.0)
            .fillna(0.5)
        )
        group["seller_pressure"] = 1.0 - group["taker_buy_ratio"]

        group["rsi7"] = _rsi(close, 7)
        group["rsi14"] = _rsi(close, 14)
        ema55 = _ema(close, 55)
        group["ema55_slope20_pct"] = _pct_change(ema55, ema55.shift(20))
        mid20 = close.rolling(20, min_periods=10).mean()
        std20 = close.rolling(20, min_periods=10).std()
        upper20 = mid20 + 2.0 * std20
        lower20 = mid20 - 2.0 * std20
        bb_width = _pct_of_price(upper20 - lower20, mid20)
        group["bb20_zscore"] = (close - mid20) / std20.replace(0.0, np.nan)
        group["bb20_width_ratio120"] = (
            bb_width / bb_width.shift(1).rolling(120, min_periods=30).median()
        )
        low20 = low.rolling(20, min_periods=10).min()
        high20 = high.rolling(20, min_periods=10).max()
        group["donchian20_pos"] = (close - low20) / (high20 - low20).replace(
            0.0, np.nan,
        )
        low60 = low.rolling(60, min_periods=20).min()
        high60 = high.rolling(60, min_periods=20).max()
        group["donchian60_pos"] = (close - low60) / (high60 - low60).replace(
            0.0, np.nan,
        )
        _plus_di, _minus_di, adx = _adx(high, low, close, 14)
        group["adx14"] = adx
        previous_close = close.shift(1)
        true_range = pd.concat(
            [
                (high - low).abs(),
                (high - previous_close).abs(),
                (low - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr14 = true_range.ewm(alpha=1.0 / 14.0, adjust=False, min_periods=7).mean()
        group["atr14_pct"] = _pct_of_price(atr14, close)
        group["atr14_ratio120"] = (
            atr14 / atr14.shift(1).rolling(120, min_periods=30).median()
        )
        candle_range = (high - low).replace(0.0, np.nan)
        group["upper_wick_pct"] = (
            high - pd.concat([open_, close], axis=1).max(axis=1)
        ) / candle_range
        group["lower_wick_pct"] = (
            pd.concat([open_, close], axis=1).min(axis=1) - low
        ) / candle_range
        group["wick_imbalance"] = group["upper_wick_pct"] - group["lower_wick_pct"]
        group["squeeze20"] = bb_width / group["atr14_pct"].replace(0.0, np.nan)
        group["squeeze120_rank"] = (
            group["squeeze20"].rolling(120, min_periods=30).rank(pct=True)
        )
        group = _add_resampled_timeframe_group(group)
        parts.append(group)

    if not parts:
        return pd.DataFrame()
    rows = pd.concat(parts, ignore_index=True, sort=False)
    rows = rows.sort_values(["symbol", "bar_time"]).reset_index(drop=True)
    rows = _add_symbol_context_features(rows)
    cross_section = rows["symbol"].map(_market_cross_section_context)
    cross_section_groupers = [rows["bar_time"], cross_section]
    for column in ("ret20", "ret60", "quote_volume"):
        values = pd.to_numeric(rows.get(column), errors="coerce")
        rows[f"{column}_rank_pct"] = values.groupby(cross_section_groupers).rank(
            pct=True,
            method="average",
        )
    for column in ("ret20", "ret60"):
        values = pd.to_numeric(rows.get(column), errors="coerce")
        market_column = f"market_{column}_mean"
        rows[market_column] = values.groupby(cross_section_groupers).transform("mean")
        rows[f"rel_{column}"] = values - rows[market_column]
    ret20 = pd.to_numeric(rows.get("ret20"), errors="coerce")
    ret60 = pd.to_numeric(rows.get("ret60"), errors="coerce")
    rows["market_down_breadth20"] = (
        ret20.lt(0).groupby(cross_section_groupers).transform("mean")
    )
    rows["market_down_breadth60"] = (
        ret60.lt(0).groupby(cross_section_groupers).transform("mean")
    )
    rows["market_up_breadth20"] = (
        ret20.gt(0).groupby(cross_section_groupers).transform("mean")
    )
    rows["market_up_breadth60"] = (
        ret60.gt(0).groupby(cross_section_groupers).transform("mean")
    )
    runtime_numeric = (
        "ret1",
        "ret3",
        "ret12",
        "ret20",
        "ret60",
        "ret72",
        "ret144",
        "ret288",
        "from_high12",
        "atr_pct",
        "bar_range_pct",
        "close_pos",
        "quote_volume_ratio_med30",
        "quote_volume_rank_pct",
        "taker_buy_ratio",
        "seller_pressure",
        "rsi7",
        "rsi14",
        "ema55_slope20_pct",
        "bb20_zscore",
        "bb20_width_ratio120",
        "donchian20_pos",
        "donchian60_pos",
        "adx14",
        "atr14_pct",
        "atr14_ratio120",
        "upper_wick_pct",
        "lower_wick_pct",
        "wick_imbalance",
        "squeeze120_rank",
        "ret20_rank_pct",
        "ret60_rank_pct",
        "market_ret20_mean",
        "market_ret60_mean",
        "rel_ret20",
        "rel_ret60",
        "market_down_breadth20",
        "market_down_breadth60",
        "market_up_breadth20",
        "market_up_breadth60",
        "family_eth_main",
        "family_major_basket",
        "family_alt_basket",
        "family_gold_defensive",
        "session_asia_00_08",
        "session_europe_08_16",
        "session_us_16_24",
        "tf15_ret4",
        "tf15_ret16",
        "tf15_rsi14",
        "tf15_ema21_slope4_pct",
        "tf15_adx14",
        "tf1h_ret4",
        "tf1h_ret12",
        "tf1h_rsi14",
        "tf1h_ema21_slope3_pct",
        "tf1h_adx14",
    )
    for column in runtime_numeric:
        if column not in rows:
            rows[column] = 0.0
        rows[column] = (
            pd.to_numeric(rows[column], errors="coerce")
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0.0)
        )
    return rows.sort_values(["bar_time", "symbol"]).reset_index(drop=True)


def _add_single_symbol_liquidity_features(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    if "quote_volume" not in out:
        out["quote_volume"] = pd.to_numeric(
            out.get("volume"), errors="coerce",
        ) * pd.to_numeric(out.get("close"), errors="coerce")
    out["quote_volume"] = pd.to_numeric(out["quote_volume"], errors="coerce").replace(
        [np.inf, -np.inf], np.nan,
    )
    out["quote_volume_log"] = np.log1p(out["quote_volume"].clip(lower=0.0))
    out["quote_volume_ratio_med30"] = (
        out.groupby("symbol", sort=False)["quote_volume"]
        .transform(lambda s: s / s.shift(1).rolling(30, min_periods=5).median())
        .replace([np.inf, -np.inf], np.nan)
    )
    if "trades" not in out:
        out["trades"] = np.nan
    out["trades"] = pd.to_numeric(out["trades"], errors="coerce").replace(
        [np.inf, -np.inf], np.nan,
    )
    out["trades_log"] = np.log1p(out["trades"].clip(lower=0.0))
    out["trades_ratio_med30"] = (
        out.groupby("symbol", sort=False)["trades"]
        .transform(lambda s: s / s.shift(1).rolling(30, min_periods=5).median())
        .replace([np.inf, -np.inf], np.nan)
    )
    if "taker_buy_quote" not in out:
        out["taker_buy_quote"] = np.nan
    taker = pd.to_numeric(out["taker_buy_quote"], errors="coerce").replace(
        [np.inf, -np.inf], np.nan,
    )
    out["taker_buy_ratio"] = (taker / out["quote_volume"].replace(0.0, np.nan)).clip(
        lower=0.0, upper=1.0,
    )
    out["taker_buy_ratio"] = out["taker_buy_ratio"].fillna(0.5)
    out["taker_buy_ratio_ma5"] = out.groupby("symbol", sort=False)[
        "taker_buy_ratio"
    ].transform(lambda s: s.rolling(5, min_periods=2).mean())
    out["taker_buy_ratio_ma30"] = out.groupby("symbol", sort=False)[
        "taker_buy_ratio"
    ].transform(lambda s: s.rolling(30, min_periods=5).mean())
    out["taker_buy_ratio_delta5"] = out["taker_buy_ratio"] - out["taker_buy_ratio_ma5"]
    out["seller_pressure"] = 1.0 - out["taker_buy_ratio"]
    return out


def _add_multi_timeframe_features(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    parts: list[pd.DataFrame] = []
    for _, group in rows.groupby("symbol", sort=False):
        parts.append(_add_multi_timeframe_group(group.copy()))
    return pd.concat(parts, axis=0).sort_index()


def _add_multi_timeframe_group(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("bar_time").copy()
    close = pd.to_numeric(group.get("close"), errors="coerce")
    high = pd.to_numeric(group.get("high"), errors="coerce")
    low = pd.to_numeric(group.get("low"), errors="coerce")
    volume = pd.to_numeric(group.get("volume"), errors="coerce")
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    for window in (12, 36, 72, 144, 288):
        group[f"ret{window}"] = _pct_change(close, close.shift(window))
    for window in (12, 144, 288):
        rolling_high = high.rolling(window, min_periods=max(4, window // 4)).max()
        rolling_low = low.rolling(window, min_periods=max(4, window // 4)).min()
        group[f"from_high{window}"] = _pct_change(close, rolling_high)
        group[f"from_low{window}"] = _pct_change(close, rolling_low)
    ema34 = _ema(close, 34)
    ema89 = _ema(close, 89)
    ema144 = _ema(close, 144)
    group["ema34_deviation"] = _pct_change(close, ema34)
    group["ema89_deviation"] = _pct_change(close, ema89)
    group["ema144_deviation"] = _pct_change(close, ema144)
    group["ema34_slope12_pct"] = _pct_change(ema34, ema34.shift(12))
    group["ema89_slope36_pct"] = _pct_change(ema89, ema89.shift(36))
    group["ema144_slope72_pct"] = _pct_change(ema144, ema144.shift(72))
    median_volume96 = volume.shift(1).rolling(96, min_periods=24).median()
    median_range96 = tr.shift(1).rolling(96, min_periods=24).median()
    atr14 = tr.ewm(alpha=1.0 / 14.0, adjust=False, min_periods=7).mean()
    median_atr96 = atr14.shift(1).rolling(96, min_periods=24).median()
    group["vol_ratio_med96"] = volume / median_volume96.replace(0.0, np.nan)
    group["range_ratio_med96"] = tr / median_range96.replace(0.0, np.nan)
    group["atr_ratio_med96"] = atr14 / median_atr96.replace(0.0, np.nan)
    return group


def _add_resampled_timeframe_features(rows: pd.DataFrame) -> pd.DataFrame:
    """Add causal indicators from fully closed 15-minute and hourly candles."""

    if rows.empty:
        return rows.copy()
    parts: list[pd.DataFrame] = []
    for _, group in rows.groupby("symbol", sort=False):
        parts.append(_add_resampled_timeframe_group(group.copy()))
    return pd.concat(parts, axis=0).sort_index()


def _add_resampled_timeframe_group(group: pd.DataFrame) -> pd.DataFrame:
    out = group.sort_values("bar_time").copy()
    timestamps = pd.to_datetime(out.get("bar_time"), errors="coerce")
    valid = timestamps.notna()
    base = out.loc[valid].copy()
    if base.empty:
        return out
    row_index = base.index
    base.index = pd.DatetimeIndex(timestamps.loc[valid])
    market = base[["high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    specifications = (
        ("tf15", "15min", 4, 16, 4),
        ("tf1h", "1h", 4, 12, 3),
    )
    for prefix, rule, short_window, long_window, slope_window in specifications:
        candles = market.resample(
            rule,
            closed="right",
            label="right",
            origin="start_day",
        ).agg({"high": "max", "low": "min", "close": "last"})
        close = pd.to_numeric(candles["close"], errors="coerce")
        ema21 = _ema(close, 21)
        _plus_di, _minus_di, adx = _adx(
            pd.to_numeric(candles["high"], errors="coerce"),
            pd.to_numeric(candles["low"], errors="coerce"),
            close,
            14,
        )
        indicators = pd.DataFrame(
            {
                f"{prefix}_ret{short_window}": _pct_change(
                    close,
                    close.shift(short_window),
                ),
                f"{prefix}_ret{long_window}": _pct_change(
                    close,
                    close.shift(long_window),
                ),
                f"{prefix}_rsi14": _rsi(close, 14),
                f"{prefix}_ema21_slope{slope_window}_pct": _pct_change(
                    ema21,
                    ema21.shift(slope_window),
                ),
                f"{prefix}_adx14": adx,
            },
            index=candles.index,
        )
        aligned = indicators.reindex(base.index, method="ffill")
        for column in indicators:
            out.loc[row_index, column] = aligned[column].to_numpy()
    return out


def _add_kline_indicator_features(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    parts: list[pd.DataFrame] = []
    for _, group in rows.groupby("symbol", sort=False):
        parts.append(_add_kline_indicator_group(group.copy()))
    return pd.concat(parts, axis=0).sort_index()


def _add_kline_indicator_group(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("bar_time").copy()
    open_ = pd.to_numeric(group.get("open"), errors="coerce")
    high = pd.to_numeric(group.get("high"), errors="coerce")
    low = pd.to_numeric(group.get("low"), errors="coerce")
    close = pd.to_numeric(group.get("close"), errors="coerce")
    volume = pd.to_numeric(group.get("volume"), errors="coerce").fillna(0.0)
    typical = (high + low + close) / 3.0
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    for window in (7, 14, 28):
        group[f"rsi{window}"] = _rsi(close, window)
    group["rsi14_delta5"] = group["rsi14"] - group["rsi14"].shift(5)

    ema8 = _ema(close, 8)
    ema12 = _ema(close, 12)
    ema21 = _ema(close, 21)
    ema26 = _ema(close, 26)
    ema55 = _ema(close, 55)
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False, min_periods=4).mean()
    group["macd_pct"] = _pct_of_price(macd, close)
    group["macd_signal_pct"] = _pct_of_price(macd_signal, close)
    group["macd_hist_pct"] = _pct_of_price(macd - macd_signal, close)
    group["ema8_dev_pct"] = _pct_change(close, ema8)
    group["ema21_dev_pct"] = _pct_change(close, ema21)
    group["ema55_dev_pct"] = _pct_change(close, ema55)
    group["ema8_21_spread_pct"] = _pct_of_price(ema8 - ema21, close)
    group["ema21_55_spread_pct"] = _pct_of_price(ema21 - ema55, close)
    group["ema21_slope5_pct"] = _pct_change(ema21, ema21.shift(5))
    group["ema55_slope20_pct"] = _pct_change(ema55, ema55.shift(20))

    mid20 = close.rolling(20, min_periods=10).mean()
    std20 = close.rolling(20, min_periods=10).std()
    upper20 = mid20 + 2.0 * std20
    lower20 = mid20 - 2.0 * std20
    bb_width = _pct_of_price(upper20 - lower20, mid20)
    group["bb20_zscore"] = (close - mid20) / std20.replace(0.0, np.nan)
    group["bb20_width_pct"] = bb_width
    group["bb20_width_ratio120"] = (
        bb_width / bb_width.shift(1).rolling(120, min_periods=30).median()
    )
    group["bb20_pct_b"] = (close - lower20) / (upper20 - lower20).replace(0.0, np.nan)
    group["bb20_upper_break_pct"] = _pct_change(close, upper20).clip(lower=0.0)
    group["bb20_lower_break_pct"] = _pct_change(lower20, close).clip(lower=0.0)

    low20 = low.rolling(20, min_periods=10).min()
    high20 = high.rolling(20, min_periods=10).max()
    low60 = low.rolling(60, min_periods=20).min()
    high60 = high.rolling(60, min_periods=20).max()
    group["donchian20_pos"] = (close - low20) / (high20 - low20).replace(0.0, np.nan)
    group["donchian60_pos"] = (close - low60) / (high60 - low60).replace(0.0, np.nan)

    low14 = low.rolling(14, min_periods=7).min()
    high14 = high.rolling(14, min_periods=7).max()
    stoch_k = (close - low14) / (high14 - low14).replace(0.0, np.nan) * 100.0
    group["stoch14_k"] = stoch_k
    group["stoch14_d"] = stoch_k.rolling(3, min_periods=2).mean()

    tp_sma20 = typical.rolling(20, min_periods=10).mean()
    tp_mad20 = typical.rolling(20, min_periods=10).apply(
        lambda values: float(np.mean(np.abs(values - np.mean(values)))), raw=True,
    )
    group["cci20"] = (typical - tp_sma20) / (0.015 * tp_mad20.replace(0.0, np.nan))
    group["mfi14"] = _mfi(typical, volume, 14)

    plus_di, minus_di, adx = _adx(high, low, close, 14)
    group["adx14"] = adx
    group["plus_di14"] = plus_di
    group["minus_di14"] = minus_di
    group["di_diff14"] = plus_di - minus_di

    atr14 = tr.ewm(alpha=1.0 / 14.0, adjust=False, min_periods=7).mean()
    group["atr14_pct"] = _pct_of_price(atr14, close)
    group["atr14_ratio120"] = (
        atr14 / atr14.shift(1).rolling(120, min_periods=30).median()
    )

    vwap20 = _rolling_vwap(typical, volume, 20)
    vwap60 = _rolling_vwap(typical, volume, 60)
    group["vwap20_dev_pct"] = _pct_change(close, vwap20)
    group["vwap60_dev_pct"] = _pct_change(close, vwap60)

    obv = (np.sign(close.diff()).fillna(0.0) * volume).cumsum()
    obv_mean60 = obv.rolling(60, min_periods=20).mean()
    obv_std60 = obv.rolling(60, min_periods=20).std()
    group["obv_z60"] = (obv - obv_mean60) / obv_std60.replace(0.0, np.nan)
    group["obv_slope20"] = (obv - obv.shift(20)) / volume.rolling(
        20, min_periods=10,
    ).sum().replace(0.0, np.nan)

    abs_ret1 = close.pct_change().abs()
    group["trend_efficiency20"] = (
        close / close.shift(20) - 1.0
    ).abs() / abs_ret1.rolling(20, min_periods=10).sum().replace(0.0, np.nan)
    group["trend_efficiency60"] = (
        close / close.shift(60) - 1.0
    ).abs() / abs_ret1.rolling(60, min_periods=20).sum().replace(0.0, np.nan)

    candle_range = (high - low).replace(0.0, np.nan)
    body = (close - open_).abs()
    group["candle_body_pct"] = body / candle_range
    group["upper_wick_pct"] = (
        high - pd.concat([open_, close], axis=1).max(axis=1)
    ) / candle_range
    group["lower_wick_pct"] = (
        pd.concat([open_, close], axis=1).min(axis=1) - low
    ) / candle_range
    group["wick_imbalance"] = group["upper_wick_pct"] - group["lower_wick_pct"]
    group["range_expansion20"] = tr / tr.shift(1).rolling(20, min_periods=10).median()
    group["volume_price_trend20"] = pd.to_numeric(
        group.get("ret20"), errors="coerce",
    ) * pd.to_numeric(group.get("vol_ratio_med30"), errors="coerce")
    group["squeeze20"] = bb_width / group["atr14_pct"].replace(0.0, np.nan)
    group["squeeze120_rank"] = (
        group["squeeze20"].rolling(120, min_periods=30).rank(pct=True)
    )
    return group


def _add_symbol_context_features(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    out = rows.copy()
    clean_symbol = out["symbol"].astype(str).map(_clean_symbol)
    for symbol in SYMBOL_CONTEXT_SYMBOLS:
        feature = f"symbol_is_{symbol.lower().replace('usdt', '')}"
        out[feature] = clean_symbol.eq(symbol).astype(float)
    family = clean_symbol.map(_symbol_strategy_family)
    for name in ("eth_main", "major_basket", "alt_basket", "gold_defensive"):
        out[f"family_{name}"] = family.eq(name).astype(float)
    ts = pd.to_datetime(out["bar_time"], errors="coerce")
    hour = ts.dt.hour.fillna(-1)
    out["session_asia_00_08"] = hour.between(0, 7, inclusive="both").astype(float)
    out["session_europe_08_16"] = hour.between(8, 15, inclusive="both").astype(float)
    out["session_us_16_24"] = hour.between(16, 23, inclusive="both").astype(float)
    return out


def _add_cross_sectional_features(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    cross_section = out["symbol"].map(_market_cross_section_context)
    groupers = [out["bar_time"], cross_section]
    for col in (
        "ret1",
        "ret3",
        "ret12",
        "ret20",
        "ret36",
        "ret60",
        "ret72",
        "ret144",
        "ret240",
        "ret288",
    ):
        values = pd.to_numeric(out.get(col), errors="coerce")
        mean_col = f"market_{col}_mean"
        out[mean_col] = values.groupby(groupers).transform("mean")
        out[f"rel_{col}"] = values - out[mean_col]
    for col in (
        "ret12",
        "ret20",
        "ret36",
        "ret60",
        "ret72",
        "ret144",
        "ret240",
        "ret288",
        "from_high60",
        "from_low60",
        "from_high144",
        "from_low144",
        "from_high288",
        "from_low288",
        "quote_volume",
    ):
        values = pd.to_numeric(out.get(col), errors="coerce")
        out[f"{col}_rank_pct"] = values.groupby(groupers).rank(
            pct=True, method="average",
        )
    ret20 = pd.to_numeric(out.get("ret20"), errors="coerce")
    ret60 = pd.to_numeric(out.get("ret60"), errors="coerce")
    out["market_down_breadth20"] = ret20.lt(0).groupby(groupers).transform("mean")
    out["market_down_breadth60"] = ret60.lt(0).groupby(groupers).transform("mean")
    out["market_up_breadth20"] = ret20.gt(0).groupby(groupers).transform("mean")
    out["market_up_breadth60"] = ret60.gt(0).groupby(groupers).transform("mean")
    return out


def _market_cross_section_context(symbol: object) -> str:
    """Keep tokenized-equity breadth from changing the established core book.

    The boundary is an asset-group prior, never an individual symbol or a
    clock feature.  Signals still merge into one execution and margin ledger.
    """

    return (
        "tradfi_equity"
        if weijie_market_prior_group(symbol) == "tradfi_equity"
        else "core_portfolio"
    )


def _normalize_futures_klines(rows: list[list[Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]
    frame = pd.DataFrame(rows, columns=columns[: len(rows[0])])
    if "open_time" not in frame:
        return pd.DataFrame()
    frame["bar_time"] = (
        pd.to_datetime(
            pd.to_numeric(frame["open_time"], errors="coerce"),
            unit="ms",
            errors="coerce",
            utc=True,
        )
        .dt.tz_convert("Asia/Shanghai")
        .dt.tz_localize(None)
    )
    frame = frame.dropna(subset=["bar_time"]).set_index("bar_time").sort_index()
    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
    ]
    for col in numeric_cols:
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame[[col for col in numeric_cols if col in frame]].copy()


def _ema(series: pd.Series, span: int) -> pd.Series:
    return (
        pd.to_numeric(series, errors="coerce")
        .ewm(span=int(span), adjust=False, min_periods=max(2, int(span) // 2))
        .mean()
    )


def _rsi(close: pd.Series, window: int) -> pd.Series:
    values = pd.to_numeric(close, errors="coerce")
    delta = values.diff()
    gain = (
        delta.clip(lower=0.0)
        .ewm(
            alpha=1.0 / float(window),
            adjust=False,
            min_periods=max(2, int(window) // 2),
        )
        .mean()
    )
    loss = (
        (-delta.clip(upper=0.0))
        .ewm(
            alpha=1.0 / float(window),
            adjust=False,
            min_periods=max(2, int(window) // 2),
        )
        .mean()
    )
    rs = gain / loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _pct_change(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return (
        pd.to_numeric(numerator, errors="coerce")
        / pd.to_numeric(denominator, errors="coerce").replace(0.0, np.nan)
        - 1.0
    ) * 100.0


def _pct_of_price(value: pd.Series, price: pd.Series) -> pd.Series:
    return (
        pd.to_numeric(value, errors="coerce")
        / pd.to_numeric(price, errors="coerce").replace(0.0, np.nan)
        * 100.0
    )


def _rolling_vwap(price: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    money = pd.to_numeric(price, errors="coerce") * pd.to_numeric(
        volume, errors="coerce",
    )
    vol = pd.to_numeric(volume, errors="coerce")
    return money.rolling(
        int(window), min_periods=max(2, int(window) // 2),
    ).sum() / vol.rolling(
        int(window), min_periods=max(2, int(window) // 2),
    ).sum().replace(
        0.0, np.nan,
    )


def _mfi(typical: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    tp = pd.to_numeric(typical, errors="coerce")
    money = tp * pd.to_numeric(volume, errors="coerce")
    delta = tp.diff()
    positive = (
        money.where(delta > 0.0, 0.0)
        .rolling(int(window), min_periods=max(2, int(window) // 2))
        .sum()
    )
    negative = (
        money.where(delta < 0.0, 0.0)
        .rolling(int(window), min_periods=max(2, int(window) // 2))
        .sum()
    )
    ratio = positive / negative.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + ratio))


def _adx(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    high = pd.to_numeric(high, errors="coerce")
    low = pd.to_numeric(low, errors="coerce")
    close = pd.to_numeric(close, errors="coerce")
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0),
        index=high.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0),
        index=high.index,
    )
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    alpha = 1.0 / float(window)
    atr = tr.ewm(alpha=alpha, adjust=False, min_periods=max(2, int(window) // 2)).mean()
    plus_di = (
        100.0
        * plus_dm.ewm(
            alpha=alpha, adjust=False, min_periods=max(2, int(window) // 2),
        ).mean()
        / atr.replace(0.0, np.nan)
    )
    minus_di = (
        100.0
        * minus_dm.ewm(
            alpha=alpha, adjust=False, min_periods=max(2, int(window) // 2),
        ).mean()
        / atr.replace(0.0, np.nan)
    )
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan) * 100.0
    adx = dx.ewm(alpha=alpha, adjust=False, min_periods=max(2, int(window) // 2)).mean()
    return plus_di, minus_di, adx


def _interval_to_milliseconds(interval: str) -> int:
    raw = str(interval or "").strip().lower()
    try:
        amount = int(raw[:-1])
    except Exception:
        return 0
    unit = raw[-1:]
    if unit == "m":
        return amount * 60_000
    if unit == "h":
        return amount * 3_600_000
    if unit == "d":
        return amount * 86_400_000
    return 0


def _shanghai_timestamp_ms(ts: pd.Timestamp) -> int:
    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("Asia/Shanghai")
    else:
        stamp = stamp.tz_convert("Asia/Shanghai")
    return int(stamp.tz_convert("UTC").timestamp() * 1000)


def _shanghai_naive_timestamp(value: object) -> pd.Timestamp:
    """Return the wall-clock representation used by local K-line frames."""

    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("Asia/Shanghai").tz_localize(None)
    return stamp


def _add_time_features(frame: pd.DataFrame) -> None:
    ts = pd.to_datetime(frame["bar_time"], errors="coerce")
    minute = ts.dt.hour * 60 + ts.dt.minute
    frame["minute_sin"] = np.sin(2.0 * np.pi * minute / 1440.0)
    frame["minute_cos"] = np.cos(2.0 * np.pi * minute / 1440.0)
    dow = ts.dt.dayofweek
    frame["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
    frame["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)


def _clean_symbol(value: object) -> str:
    return str(value or "").upper().replace("/", "").replace("-", "").strip()


def _symbol_strategy_family(symbol: str) -> str:
    clean = _clean_symbol(symbol)
    if clean == "ETHUSDT":
        return "eth_main"
    if weijie_asset_group(clean) == "stock_gold":
        return "gold_defensive"
    if clean in {"BTCUSDT", "BNBUSDT", "SOLUSDT", "BCHUSDT", "LTCUSDT"}:
        return "major_basket"
    return "alt_basket"
