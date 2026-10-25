"""Open-interest and funding factors for Weijie event gating."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from factors.crypto.code.weijie.common import clean_symbol, finite_float, pct_change, shanghai_timestamp_ms

POSITIONING_COLUMNS = (
    "oi_change12",
    "oi_value_change12",
    "oi_value_ratio96",
    "funding_rate",
)


def enrich_market_positioning(
    feature_rows: pd.DataFrame,
    *,
    symbols: set[str] | frozenset[str] | tuple[str, ...] | None = None,
    tolerance: pd.Timedelta = pd.Timedelta(minutes=10),
) -> pd.DataFrame:
    """把已公布的 5m 持仓数据向后对齐到闭合 K 线。

    ``merge_asof(direction="backward")`` 保证每根决策 K 线只能看到当时已经
    公布的 OI/资金费率。OI 不可用或仍处于滚动预热期时，数值列置零且
    ``positioning_available=0``；仓位结构因子据此失败关闭，不能把缺失值
    当作中性市场证据。
    """

    if feature_rows is None or feature_rows.empty:
        return feature_rows.copy()
    out = feature_rows.copy()
    out["bar_time"] = pd.to_datetime(out.get("bar_time"), errors="coerce")
    out["symbol"] = out.get("symbol", pd.Series("", index=out.index)).astype(str)
    selected = {
        clean_symbol(symbol)
        for symbol in symbols
        if clean_symbol(symbol)
    } if symbols is not None else None
    parts: list[pd.DataFrame] = []
    for raw_symbol, group in out.groupby("symbol", sort=False, dropna=False):
        symbol = clean_symbol(raw_symbol)
        part = group.sort_values("bar_time", kind="stable").copy()
        for column in (*POSITIONING_COLUMNS, "positioning_available"):
            if column in part:
                part = part.drop(columns=column)
        valid_times = part["bar_time"].dropna()
        if (
            (selected is not None and symbol not in selected)
            or valid_times.empty
        ):
            for column in POSITIONING_COLUMNS:
                part[column] = 0.0
            part["positioning_available"] = 0
            parts.append(part)
            continue
        # 96 根 OI 中位数需要 8 小时预热；额外留出 4 小时覆盖接口边界。
        positioning = fetch_positioning_features(
            symbol,
            start_at=valid_times.min() - pd.Timedelta(hours=12),
            end_at=valid_times.max() + pd.Timedelta(minutes=5),
        )
        if positioning.empty:
            for column in POSITIONING_COLUMNS:
                part[column] = 0.0
            part["positioning_available"] = 0
            parts.append(part)
            continue
        source = positioning[["bar_time", *POSITIONING_COLUMNS]].copy()
        source["positioning_source_time"] = source["bar_time"]
        source = source.drop(columns="bar_time").sort_values(
            "positioning_source_time",
            kind="stable",
        )
        aligned = pd.merge_asof(
            part,
            source,
            left_on="bar_time",
            right_on="positioning_source_time",
            direction="backward",
            tolerance=tolerance,
        )
        required_oi = ("oi_change12", "oi_value_change12", "oi_value_ratio96")
        available = aligned["positioning_source_time"].notna()
        for column in required_oi:
            available &= pd.to_numeric(aligned[column], errors="coerce").replace(
                [np.inf, -np.inf],
                np.nan,
            ).notna()
        aligned["positioning_available"] = available.astype(int)
        for column in POSITIONING_COLUMNS:
            aligned[column] = pd.to_numeric(
                aligned[column],
                errors="coerce",
            ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        parts.append(aligned.drop(columns="positioning_source_time"))
    if not parts:
        return out
    return pd.concat(parts, ignore_index=True, sort=False).sort_values(
        ["bar_time", "symbol"],
        kind="stable",
    ).reset_index(drop=True)


def enrich_signal_positioning(signals: pd.DataFrame) -> pd.DataFrame:
    if signals.empty:
        return signals.copy()
    parts: list[pd.DataFrame] = []
    for symbol, group in signals.groupby("symbol", sort=False):
        symbol_text = str(symbol)
        part = group.sort_values("signal_time").copy()
        start_at = pd.to_datetime(part["signal_time"], errors="coerce").min() - pd.Timedelta(days=3)
        end_at = pd.to_datetime(part["signal_time"], errors="coerce").max() + pd.Timedelta(minutes=15)
        positioning = fetch_positioning_features(symbol_text, start_at=start_at, end_at=end_at)
        if positioning.empty:
            for col in POSITIONING_COLUMNS:
                part[col] = 0.0
            parts.append(part)
            continue
        aligned = pd.merge_asof(
            part.sort_values("signal_time"),
            positioning.sort_values("bar_time"),
            left_on="signal_time",
            right_on="bar_time",
            direction="backward",
            tolerance=pd.Timedelta(minutes=30),
        )
        if "symbol_x" in aligned:
            aligned["symbol"] = aligned["symbol_x"].combine_first(aligned.get("symbol_y"))
            aligned = aligned.drop(columns=[col for col in ("symbol_x", "symbol_y") if col in aligned])
        for col in POSITIONING_COLUMNS:
            if col not in aligned:
                aligned[col] = 0.0
            aligned[col] = pd.to_numeric(aligned[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        parts.append(aligned.drop(columns=[col for col in ("bar_time",) if col in aligned]))
    return pd.concat(parts, ignore_index=True, sort=False) if parts else signals.copy()


def fetch_positioning_features(symbol: str, *, start_at: pd.Timestamp, end_at: pd.Timestamp) -> pd.DataFrame:
    oi = fetch_open_interest_hist(symbol, start_at=start_at, end_at=end_at)
    if oi.empty:
        return pd.DataFrame()
    oi = oi.sort_values("bar_time").copy()
    oi["sum_open_interest"] = pd.to_numeric(oi.get("sum_open_interest"), errors="coerce")
    oi["sum_open_interest_value"] = pd.to_numeric(oi.get("sum_open_interest_value"), errors="coerce")
    oi["oi_change12"] = pct_change(oi["sum_open_interest"], oi["sum_open_interest"].shift(12))
    oi["oi_value_change12"] = pct_change(oi["sum_open_interest_value"], oi["sum_open_interest_value"].shift(12))
    oi["oi_value_ratio96"] = (
        oi["sum_open_interest_value"] / oi["sum_open_interest_value"].shift(1).rolling(96, min_periods=12).median()
    ).replace([np.inf, -np.inf], np.nan)
    funding = fetch_funding_rate(symbol, start_at=start_at - pd.Timedelta(days=2), end_at=end_at)
    if funding.empty:
        oi["funding_rate"] = 0.0
    else:
        funding = funding.sort_values("funding_time").copy()
        merged = pd.merge_asof(
            oi.sort_values("bar_time"),
            funding.sort_values("funding_time"),
            left_on="bar_time",
            right_on="funding_time",
            direction="backward",
            tolerance=pd.Timedelta(hours=16),
        )
        oi["funding_rate"] = pd.to_numeric(merged.get("funding_rate"), errors="coerce").fillna(0.0)
    columns = [
        "bar_time",
        "sum_open_interest",
        "sum_open_interest_value",
        *POSITIONING_COLUMNS,
    ]
    for col in columns:
        if col not in oi:
            oi[col] = np.nan
    return oi[columns].copy()


def fetch_open_interest_hist(symbol: str, *, start_at: pd.Timestamp, end_at: pd.Timestamp) -> pd.DataFrame:
    clean = clean_symbol(symbol)
    start_ts = pd.Timestamp(start_at).floor("min")
    end_ts = pd.Timestamp(end_at).ceil("min")
    cache_dir = Path("pickle_cache") / "weijie_positioning"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"oi_{clean}_5m_{start_ts:%Y%m%d%H%M}_{end_ts:%Y%m%d%H%M}.pkl"
    if cache_path.exists():
        try:
            cached = pd.read_pickle(cache_path)
            # 空结果常由临时网络故障或接口保留边界造成，不能永久缓存。
            if isinstance(cached, pd.DataFrame) and not cached.empty:
                return cached
        except Exception:
            pass
    rows: list[dict[str, Any]] = []
    cursor = shanghai_timestamp_ms(start_ts)
    end_ms = shanghai_timestamp_ms(end_ts)
    step_ms = 5 * 60 * 1000 * 500
    while cursor <= end_ms:
        chunk_end = int(min(end_ms, cursor + step_ms))
        params = {
            "symbol": clean,
            "period": "5m",
            "startTime": int(cursor),
            "endTime": chunk_end,
            "limit": 500,
        }
        url = "https://fapi.binance.com/futures/data/openInterestHist?" + urllib.parse.urlencode(params)
        try:
            with urllib.request.urlopen(url, timeout=15) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_payload = exc.read()
            try:
                error_body = error_payload.decode("utf-8")
            except UnicodeDecodeError as decode_exc:
                error_body = (
                    f"<HTTP 响应不是有效 UTF-8：{decode_exc}；"
                    f"前 64 字节十六进制={error_payload[:64].hex()}>"
                )
            if exc.code == 400 and "-1130" in error_body and "startTime" in error_body:
                # 超出 Binance 保留期的 startTime 会返回 -1130，而非空数组。
                # 只跳过这个明确错误；参数错误和网络错误仍然失败关闭。
                next_cursor = chunk_end if chunk_end > cursor else end_ms + 1
                if next_cursor <= cursor:
                    break
                cursor = next_cursor
                continue
            break
        except Exception:
            break
        if not isinstance(payload, list):
            break
        if not payload:
            # Binance 只保留有限历史。较早的空分片不能阻止继续请求仍在
            # 保留期内的后续分片，否则跨保留边界的回测会错误地全程缺失。
            next_cursor = chunk_end if chunk_end > cursor else end_ms + 1
            if next_cursor <= cursor:
                break
            cursor = next_cursor
            continue
        rows.extend(item for item in payload if isinstance(item, dict))
        last_ts = finite_float(payload[-1].get("timestamp"), cursor)
        next_cursor = int(last_ts) + 5 * 60 * 1000
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if cursor <= end_ms:
            time.sleep(0.03)
    frame = normalize_open_interest_rows(rows)
    if not frame.empty:
        frame = frame[(frame["bar_time"] >= start_ts) & (frame["bar_time"] <= end_ts)].copy()
    try:
        frame.to_pickle(cache_path)
    except Exception:
        pass
    return frame


def fetch_funding_rate(symbol: str, *, start_at: pd.Timestamp, end_at: pd.Timestamp) -> pd.DataFrame:
    clean = clean_symbol(symbol)
    start_ts = pd.Timestamp(start_at).floor("min")
    end_ts = pd.Timestamp(end_at).ceil("min")
    cache_dir = Path("pickle_cache") / "weijie_positioning"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"funding_{clean}_{start_ts:%Y%m%d%H%M}_{end_ts:%Y%m%d%H%M}.pkl"
    if cache_path.exists():
        try:
            cached = pd.read_pickle(cache_path)
            if isinstance(cached, pd.DataFrame) and not cached.empty:
                return cached
        except Exception:
            pass
    params = {
        "symbol": clean,
        "startTime": int(shanghai_timestamp_ms(start_ts)),
        "endTime": int(shanghai_timestamp_ms(end_ts)),
        "limit": 1000,
    }
    url = "https://fapi.binance.com/fapi/v1/fundingRate?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        payload = []
    rows = payload if isinstance(payload, list) else []
    frame = normalize_funding_rows(rows)
    try:
        frame.to_pickle(cache_path)
    except Exception:
        pass
    return frame


def normalize_open_interest_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    if "timestamp" not in frame:
        return pd.DataFrame()
    frame["bar_time"] = (
        pd.to_datetime(pd.to_numeric(frame["timestamp"], errors="coerce"), unit="ms", errors="coerce", utc=True)
        .dt.tz_convert("Asia/Shanghai")
        .dt.tz_localize(None)
    )
    frame["sum_open_interest"] = pd.to_numeric(frame.get("sumOpenInterest"), errors="coerce")
    frame["sum_open_interest_value"] = pd.to_numeric(frame.get("sumOpenInterestValue"), errors="coerce")
    return frame.dropna(subset=["bar_time"]).sort_values("bar_time").reset_index(drop=True)


def normalize_funding_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    if "fundingTime" not in frame:
        return pd.DataFrame()
    frame["funding_time"] = (
        pd.to_datetime(pd.to_numeric(frame["fundingTime"], errors="coerce"), unit="ms", errors="coerce", utc=True)
        .dt.tz_convert("Asia/Shanghai")
        .dt.tz_localize(None)
    )
    frame["funding_rate"] = pd.to_numeric(frame.get("fundingRate"), errors="coerce")
    return frame.dropna(subset=["funding_time"]).sort_values("funding_time").reset_index(drop=True)


__all__ = [
    "POSITIONING_COLUMNS",
    "enrich_market_positioning",
    "enrich_signal_positioning",
    "fetch_funding_rate",
    "fetch_open_interest_hist",
    "fetch_positioning_features",
    "normalize_funding_rows",
    "normalize_open_interest_rows",
]
