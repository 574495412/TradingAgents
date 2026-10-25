"""Explicit quote sources, immutable captures and exchange-session validation."""

from __future__ import annotations

import hashlib
import importlib
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

FIELDS = ("open", "high", "low", "close", "volume", "dividend", "split")
HK_CLOSURES = {
    "2023-09-01": "https://www.hkex.com.hk/News/Market-Communications/2023/2309012news?sc_lang=en",
    "2023-09-08": "https://www.hkex.com.hk/News/Market-Communications/2023/2309083news?sc_lang=en",
}
HK_DELAYED_OPEN = {
    "2023-10-09": (
        "06:00:00",
        "https://www.hkex.com.hk/News/Market-Communications/2023/2310092news?sc_lang=en",
    ),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tencent_prices(symbol, start, end, output, cache=None):
    """Capture the public Tencent unadjusted day feed, year by year; no qfq/hfq."""
    import requests

    security = "hk" + symbol.removesuffix(".HK").zfill(5)
    rows, identities = [], []
    for year in range(int(start[:4]), int(end[:4]) + 1):
        begin = max(start, f"{year}-01-01")
        finish = min(end, f"{year + 1}-01-01")
        if begin >= finish:
            continue
        file = output / f"{symbol}_{year}_tencent.json"
        cached = Path(cache) / file.name if cache else None
        params = {
            "param": f"{security},day,{begin},{(pd.Timestamp(finish) - pd.Timedelta(days=1)).date()},640,"
        }
        if cached and cached.is_file():
            raw = cached.read_bytes()
        else:
            response = requests.get(
                "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get", params=params, timeout=25
            )
            response.raise_for_status()
            raw = response.content
        file.write_bytes(raw)
        result = json.loads(raw)
        if result.get("code") != 0:
            raise ValueError("Tencent source rejected the daily request")
        payload = result["data"][security]
        if "day" not in payload:
            raise ValueError("unadjusted daily source missing; adjusted prices cannot substitute")
        for row in payload["day"]:
            if begin <= row[0] < finish:
                rows.append(row[:6])
        identities.append({"file": file.name, "sha256": digest(file), "request": params})
    frame = pd.DataFrame(rows, columns=["date", "open", "close", "high", "low", "volume"])
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.set_index("date").sort_index().astype(float)
    if frame.index.has_duplicates:
        raise ValueError("duplicate Tencent session rows")
    return frame, identities


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sessions(market, start, end):
    """Exchange calendars include holidays, early closes and US DST; end exclusive."""
    import exchange_calendars as xcals

    calendar = (
        xcals.get_calendar("XNYS" if market == "US" else "XHKG", start=start, end=end)
        .schedule.loc[start : pd.Timestamp(end) - pd.Timedelta(days=1)]
        .copy()
    )
    if market == "HK":
        calendar = calendar.drop(index=pd.to_datetime(list(HK_CLOSURES)), errors="ignore")
        for day, (opened, _) in HK_DELAYED_OPEN.items():
            if pd.Timestamp(day) in calendar.index:
                calendar.loc[day, "open"] = pd.Timestamp(f"{day} {opened}", tz="UTC")
    return calendar


def validate(frame, calendar):
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("duplicate or unordered session dates")
    if not set(FIELDS) <= set(frame):
        raise ValueError("missing OHLCV or explicit corporate-action columns")
    if not frame.index.as_unit("ns").equals(calendar.index.as_unit("ns")):
        missing = calendar.index.difference(frame.index).strftime("%Y-%m-%d").tolist()
        extra = frame.index.difference(calendar.index).strftime("%Y-%m-%d").tolist()
        raise ValueError(f"calendar mismatch; missing={missing[:8]}, extra={extra[:8]}")
    values = frame[list(FIELDS)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("unknown price, volume or corporate action; cannot fill gaps")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("nonpositive price")
    if (frame.volume <= 0).any() or (frame[["dividend", "split"]] < 0).any().any():
        raise ValueError("negative volume or action")
    if (
        (frame.high < frame[["open", "close", "low"]].max(axis=1))
        | (frame.low > frame[["open", "close", "high"]].min(axis=1))
    ).any():
        raise ValueError("inconsistent OHLC envelope")
    # Yahoo OHLC is split-adjusted even with auto_adjust=False. Never combine
    # such a series with nominal historic board lots or apply splits twice.
    if (frame.split != 0).any():
        raise ValueError("split detected; nominal-price and lot-history reconstruction required")


def daily_from_hourly(hourly, calendar, day):
    """Explicit vendor-hour aggregation; reject partial grids instead of filling."""
    day = pd.Timestamp(day)
    session = calendar.loc[day]
    expected = pd.date_range(session["open"], session["close"], freq="1h", inclusive="left")
    if hourly.index.tz is None or not hourly.index.tz_convert("UTC").as_unit("ns").equals(
        expected.as_unit("ns")
    ):
        raise ValueError("hourly reconstruction lacks the complete vendor session grid")
    if not np.isfinite(hourly[list(FIELDS)].to_numpy(dtype=float)).all():
        raise ValueError("unknown hourly reconstruction values")
    if (hourly[["dividend", "split"]] != 0).any().any():
        raise ValueError("cannot reconstruct corporate actions from hourly bars")
    return {
        "open": hourly.open.iloc[0],
        "high": hourly.high.max(),
        "low": hourly.low.min(),
        "close": hourly.close.iloc[-1],
        "volume": hourly.volume.sum(),
        "dividend": 0.0,
        "split": 0.0,
    }


def capture_yahoo(plan, output, vendor_cache=None, daily_overrides=None):
    """Public research source, explicitly distinct from a verified Longbridge feed."""
    import exchange_calendars
    import yfinance as yf

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "ta-equity-data.v1",
        "provider": "Yahoo US prices/actions; Tencent HK unadjusted prices + Yahoo actions",
        "broker_verified": False,
        "start": plan["data_start"],
        "end": plan["data_end"],
        "versions": {
            "yfinance": yf.__version__,
            "exchange_calendars": exchange_calendars.__version__,
        },
        "captured_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "symbols": {},
        "corporate_action_authority": "Yahoo reported events; not broker-verified",
        "calendar_overrides": {"closures": HK_CLOSURES, "delayed_open": HK_DELAYED_OPEN},
        "daily_reconstructions": [],
    }
    write_json(output / "plan.json", plan)
    for market, spec in plan["markets"].items():
        calendar = sessions(market, plan["data_start"], plan["data_end"])
        calendar.to_csv(output / f"{market}_sessions.csv", index_label="date")
        for symbol in spec["symbols"]:
            ticker = yf.Ticker(symbol["yahoo"])
            cached = Path(vendor_cache) / f"{symbol['symbol']}_vendor.csv" if vendor_cache else None
            if cached and cached.is_file():
                cached_plan = json.loads((Path(vendor_cache) / "plan.json").read_text())
                same_dates = all(
                    cached_plan.get(key) == plan.get(key) for key in ("data_start", "data_end")
                )
                cached_symbols = {
                    s["symbol"]: s["yahoo"]
                    for m in cached_plan["markets"].values()
                    for s in m["symbols"]
                }
                requested_symbols = {
                    s["symbol"]: s["yahoo"] for m in plan["markets"].values() for s in m["symbols"]
                }
                if not same_dates or any(
                    cached_symbols.get(s) != y for s, y in requested_symbols.items()
                ):
                    raise ValueError("vendor cache plan differs")
                raw = pd.read_csv(cached, index_col="vendor_timestamp")
                zone = "America/New_York" if market == "US" else "Asia/Hong_Kong"
                raw.index = pd.DatetimeIndex(pd.to_datetime(raw.index, utc=True).tz_convert(zone))
            else:
                raw = ticker.history(
                    start=plan["data_start"],
                    end=plan["data_end"],
                    interval="1d",
                    auto_adjust=False,
                    back_adjust=False,
                    actions=True,
                    repair=False,
                    prepost=False,
                    keepna=True,
                    timeout=25,
                    raise_errors=True,
                )
            if raw.empty:
                raise ValueError(f"empty history: {symbol['symbol']}")
            raw.to_csv(output / f"{symbol['symbol']}_vendor.csv", index_label="vendor_timestamp")
            metadata = ticker.get_history_metadata()
            if metadata.get("currency") != spec["currency"]:
                raise ValueError(f"currency mismatch: {symbol['symbol']}")
            frame = raw.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                    "Dividends": "dividend",
                    "Stock Splits": "split",
                }
            )
            frame.index = pd.DatetimeIndex(frame.index.tz_localize(None).normalize(), name="date")
            if "Capital Gains" in raw and (raw["Capital Gains"] != 0).any():
                raise ValueError("capital-gain distribution requires separate cash-event handling")
            frame = frame[list(FIELDS)]
            if market == "HK":
                prices, identities = tencent_prices(
                    symbol["symbol"], plan["data_start"], plan["data_end"], output, vendor_cache
                )
                extra = prices.index.difference(calendar.index)
                if len(extra):
                    rejected = prices.loc[extra]
                    if (rejected.volume != 0).any() or (
                        rejected[["open", "high", "low", "close"]].nunique(axis=1) != 1
                    ).any():
                        raise ValueError(
                            "non-session Tencent rows contain actual trading; calendar conflict"
                        )
                    manifest.setdefault("ignored_non_session_quotes", {})[symbol["symbol"]] = (
                        extra.strftime("%Y-%m-%d").tolist()
                    )
                    prices = prices.drop(index=extra)
                actions = frame[["dividend", "split"]].reindex(prices.index)
                frame = prices.join(actions)[list(FIELDS)]
                manifest.setdefault("hk_price_sources", {})[symbol["symbol"]] = identities
            for day, record in (daily_overrides or {}).get(symbol["symbol"], {}).items():
                source = Path(record["file"])
                if digest(source) != record["sha256"]:
                    raise ValueError("hourly source hash differs")
                hourly = pd.read_csv(source, index_col=0)
                hourly.index = pd.DatetimeIndex(pd.to_datetime(hourly.index, utc=True))
                hourly = hourly.rename(
                    columns={
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Volume": "volume",
                        "Dividends": "dividend",
                        "Stock Splits": "split",
                    }
                )
                if day not in frame.index or not frame.loc[day, list(FIELDS)].isna().any():
                    raise ValueError("only explicitly missing daily bars may be reconstructed")
                frame.loc[day, list(FIELDS)] = pd.Series(daily_from_hourly(hourly, calendar, day))
                frozen_source = output / f"{symbol['symbol']}_{day}_hourly_source.csv"
                frozen_source.write_bytes(source.read_bytes())
                manifest["daily_reconstructions"].append(
                    {
                        "symbol": symbol["symbol"],
                        "day": day,
                        "file": frozen_source.name,
                        "sha256": digest(frozen_source),
                        "method": "complete vendor 1h grid OHLCV aggregation",
                        "limitation": "same Yahoo source; closing-auction inclusion not independently verified",
                    }
                )
            validate(frame, calendar)
            path = output / f"{symbol['symbol']}.csv"
            frame.to_csv(path, index_label="date")
            manifest["symbols"][symbol["symbol"]] = {
                "file": path.name,
                "sha256": digest(path),
                "rows": len(frame),
                "vendor_file": f"{symbol['symbol']}_vendor.csv",
                "vendor_sha256": digest(output / f"{symbol['symbol']}_vendor.csv"),
                "dividend_events": int((frame.dividend != 0).sum()),
                "split_events": int((frame.split != 0).sum()),
                "currency": spec["currency"],
            }
    manifest["calendar_hashes"] = {
        market: digest(output / f"{market}_sessions.csv") for market in plan["markets"]
    }
    write_json(output / "manifest.json", manifest)
    return manifest


class LongbridgeQuotes:
    """Quote-only SDK adapter; caller supplies Config, credentials are never persisted."""

    def __init__(self, config, sdk="longport.openapi"):
        if sdk not in {"longport.openapi", "longbridge.openapi"}:
            raise ValueError("unsupported broker SDK")
        self.api = importlib.import_module(sdk)
        self.context = self.api.QuoteContext(config)

    def history(self, symbol, start, end):
        """Daily date queries are chunked below the documented 1000-bar ceiling."""
        start, end = date.fromisoformat(start), date.fromisoformat(end)
        if start >= end:
            raise ValueError("start must precede exclusive end")
        rows = []
        cursor = start
        while cursor < end:
            last = min(cursor + timedelta(days=180), end) - timedelta(days=1)
            batch = self.context.history_candlesticks_by_date(
                symbol,
                self.api.Period.Day,
                self.api.AdjustType.NoAdjust,
                cursor,
                last,
                self.api.TradeSessions.Intraday,
            )
            for bar in batch:
                stamp = pd.Timestamp(bar.timestamp)
                if stamp.tzinfo is None:
                    raise ValueError("broker returned a timezone-naive timestamp")
                zone = "America/New_York" if symbol.endswith(".US") else "Asia/Hong_Kong"
                rows.append(
                    {
                        "date": stamp.tz_convert(zone).date().isoformat(),
                        **{key: float(getattr(bar, key)) for key in FIELDS[:5]},
                    }
                )
            cursor = last + timedelta(days=1)
        frame = pd.DataFrame(rows)
        if frame.empty:
            raise ValueError("empty broker history")
        frame = frame.set_index("date").sort_index()
        if frame.index.has_duplicates:
            raise ValueError("duplicate broker history")
        # Quote bars have no dividend/split event ledger. Do not invent zeros.
        return frame


def load_dataset(path, plan):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["start"] != plan["data_start"] or manifest["end"] != plan["data_end"]:
        raise ValueError("dataset dates differ from frozen plan")
    frames, calendars = {}, {}
    for market, spec in plan["markets"].items():
        calendar_path = path / f"{market}_sessions.csv"
        if digest(calendar_path) != manifest["calendar_hashes"][market]:
            raise ValueError("session calendar hash mismatch")
        cal = pd.read_csv(calendar_path, index_col="date", parse_dates=["date"])
        cal.index = pd.DatetimeIndex(cal.index, name="date")
        for key in ("open", "close"):
            cal[key] = pd.to_datetime(cal[key], utc=True)
        calendars[market] = cal
        for spec_symbol in spec["symbols"]:
            symbol = spec_symbol["symbol"]
            record = manifest["symbols"][symbol]
            file = path / record["file"]
            if file.resolve().parent != path.resolve() or digest(file) != record["sha256"]:
                raise ValueError("dataset path or content hash mismatch")
            frame = pd.read_csv(file, index_col="date", parse_dates=["date"])
            validate(frame, cal)
            frames[symbol] = frame
    return frames, calendars, manifest
