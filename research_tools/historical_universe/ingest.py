"""Checksum-verified official daily histories, with gaps preserved and no accounts."""

import argparse
import fcntl
import hashlib
import io
import json
import re
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from research_tools.factor_lab.io import read, save, sha


def parse_archive(payload, symbol, month):
    """Parse source bytes; never repair, trim tails or insert missing candles."""
    expected = f"{symbol}-1d-{month}.csv"
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        if z.namelist() != [expected] or z.getinfo(expected).file_size > 2_000_000 or z.testzip():
            raise ValueError("Archive members, uncompressed size or CRC differ")
        raw = pd.read_csv(io.BytesIO(z.read(expected)), header=None, dtype=str)
    if len(raw) and raw.iloc[0, 0] == "open_time":
        raw = raw.iloc[1:].reset_index(drop=True)
    if raw.shape[1] != 12 or not 1 <= len(raw) <= 31:
        raise ValueError("Unexpected daily schema or monthly row count")
    values = raw.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite official candle")
    epoch = values[0].to_numpy(dtype=np.int64)
    unit = "us" if epoch.min() > 100_000_000_000_000 else "ms"
    if not np.array_equal(values[0].to_numpy(), epoch):
        raise ValueError("Noninteger candle timestamp")
    dates = pd.DatetimeIndex(pd.to_datetime(epoch, unit=unit, utc=True), name="date")
    begin = pd.Timestamp(month + "-01", tz="UTC")
    end = begin + pd.offsets.MonthBegin(1)
    if (
        dates.has_duplicates
        or not dates.is_monotonic_increasing
        or not dates.floor("D").equals(dates)
        or dates[0] < begin
        or dates[-1] >= end
    ):
        raise ValueError("Duplicate, unordered, non-UTC-daily or wrong-month candle")
    out = pd.DataFrame(
        {
            name: values[i].to_numpy()
            for i, name in [
                (1, "open"),
                (2, "high"),
                (3, "low"),
                (4, "close"),
                (5, "volume"),
                (7, "quote_volume"),
                (8, "trades"),
                (9, "taker_base_volume"),
                (10, "taker_quote_volume"),
            ]
        },
        index=dates,
    )
    if (
        (out[["open", "high", "low", "close"]] <= 0).any().any()
        or (out.high < out[["open", "low", "close"]].max(axis=1)).any()
        or (out.low > out[["open", "high", "close"]].min(axis=1)).any()
        or (
            out[["volume", "quote_volume", "trades", "taker_base_volume", "taker_quote_volume"]] < 0
        )
        .any()
        .any()
    ):
        raise ValueError("Invalid official OHLC or nonnegative volume")
    if (
        (out.taker_base_volume > out.volume + 1e-6).any()
        or (out.taker_quote_volume > out.quote_volume + 1e-4).any()
        or (out.trades != np.floor(out.trades)).any()
    ):
        raise ValueError("Trade count or taker volume exceeds candle total")
    closing_epoch = values[6].to_numpy(dtype=np.int64)
    if not np.array_equal(values[6].to_numpy(), closing_epoch):
        raise ValueError("Noninteger close timestamp")
    out["close_time"] = pd.to_datetime(closing_epoch, unit=unit, utc=True)
    if (out.close_time < dates).any() or (out.close_time >= dates + pd.Timedelta(days=1)).any():
        raise ValueError("Candle close outside its daily interval")
    # Source has millisecond closing precision even when opens use microseconds.
    out["full_daily_duration"] = (
        out.close_time.to_numpy()
        == (dates + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)).to_numpy()
    )
    missing = pd.date_range(dates[0], dates[-1], freq="D").difference(dates)
    return out, {
        "rows": len(out),
        "first": dates[0].isoformat(),
        "last": dates[-1].isoformat(),
        "timestamp_unit": unit,
        "interior_missing_days": [t.isoformat() for t in missing],
        "zero_quote_volume_days": int(out.quote_volume.eq(0).sum()),
        "partial_candles": int((~out.full_daily_duration).sum()),
    }


class Downloader:
    def __init__(self, root, plan):
        self.root, self.plan = root, plan
        self.lock, self.local = threading.Lock(), threading.local()
        self.requests, self.bytes = 0, 0

    def get(self, key):
        if not key.startswith(self.plan["prefix"]) or ".." in key.split("/"):
            raise ValueError("Out-of-scope archive key")
        if not hasattr(self.local, "session"):
            self.local.session = requests.Session()
        for attempt in range(self.plan["budget"]["retry_attempts"]):
            with self.lock:
                self.requests += 1
                if self.requests > self.plan["budget"]["maximum_download_http_requests"]:
                    raise ValueError("HTTP download budget exhausted")
            try:
                r = self.local.session.get(self.plan["download_origin"] + "/" + key, timeout=30)
                r.raise_for_status()
                with self.lock:
                    self.bytes += len(r.content)
                    if self.bytes > self.plan["budget"]["maximum_download_bytes"]:
                        raise ValueError("Downloaded byte budget exhausted")
                return r.content
            except requests.RequestException:
                if attempt + 1 == self.plan["budget"]["retry_attempts"]:
                    raise
                time.sleep(min(2**attempt, 4))
        raise AssertionError("Unreachable retry state")

    def one(self, row):
        name = row["key"].rsplit("/", 1)[-1]
        folder = self.root / "raw" / row["symbol"]
        folder.mkdir(parents=True, exist_ok=True)
        payload = self.get(row["key"])
        checksum = self.get(row["checksum_key"])
        check = re.fullmatch(r"([0-9a-fA-F]{64})\s+\*?([^\r\n]+)\s*", checksum.decode().strip())
        digest = hashlib.sha256(payload).hexdigest()
        if (
            check is None
            or check[1].lower() != digest
            or check[2].strip() != name
            or len(payload) != row["size"]
        ):
            raise ValueError("Published checksum, basename or listed ZIP size differs: " + name)
        _, quality = parse_archive(payload, row["symbol"], row["month"])
        # Both network records validated before any usable raw archive appears.
        for dest, content in [(folder / name, payload), (folder / (name + ".CHECKSUM"), checksum)]:
            with dest.open("xb") as stream:
                stream.write(content)
        return {
            "symbol": row["symbol"],
            "month": row["month"],
            "key": row["key"],
            "zip": str((folder / name).relative_to(self.root)),
            "zip_sha256": digest,
            "checksum_sha256": hashlib.sha256(checksum).hexdigest(),
            "passed": True,
            **quality,
        }


def ingest(catalog, output):
    output.mkdir(parents=True, exist_ok=False)
    lock = (output / ".download.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan_path = Path(__file__).with_name("plan.json")
    plan = read(plan_path)
    for name, digest in read(catalog / "completion.json")["manifest"].items():
        if (
            catalog.resolve() not in (catalog / name).resolve().parents
            or sha(catalog / name) != digest
        ):
            raise ValueError("Official catalog changed")
    records = read(catalog / "archives.json")
    save(
        output / "plan.json",
        {
            **plan,
            "catalog": str(catalog.resolve()),
            "catalog_completion_sha256": sha(catalog / "completion.json"),
            "source_sha256": sha(Path(__file__)),
            "plan_source_sha256": sha(plan_path),
            "registered_at_utc": datetime.now(UTC).isoformat(),
            "archive_count": len(records),
        },
    )
    downloads = Downloader(output, plan)
    reports, errors = [], []
    journal = (output / "downloads.jsonl").open("x")
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=plan["budget"]["workers"]) as pool:
        futures = {pool.submit(downloads.one, r): r for r in records}
        for i, future in enumerate(as_completed(futures)):
            source = futures[future]
            try:
                row = future.result()
                reports.append(row)
            except Exception as exc:
                row = {
                    "key": source["key"],
                    "symbol": source["symbol"],
                    "month": source["month"],
                    "passed": False,
                    "error": type(exc).__name__ + ": " + str(exc),
                }
                errors.append(row)
            journal.write(json.dumps(row, allow_nan=False) + "\n")
            journal.flush()
            if (i + 1) % 250 == 0:
                print(
                    {
                        "finished_archives": i + 1,
                        "of": len(records),
                        "passed": len(reports),
                        "errors": len(errors),
                        "http_requests": downloads.requests,
                        "seconds": round(time.monotonic() - started, 1),
                    },
                    flush=True,
                )
    journal.close()
    save(
        output / "download_summary.json",
        {
            "passed": not errors,
            "archives": len(reports),
            "errors": errors,
            "http_requests": downloads.requests,
            "download_bytes": downloads.bytes,
            "journal_sha256": sha(output / "downloads.jsonl"),
        },
    )
    if errors:
        raise ValueError("Archive ingestion has explicit quarantined failures; preserve this run")
    symbols = sorted({r["symbol"] for r in reports})
    (output / "symbols").mkdir()
    quality = []
    for symbol in symbols:
        selected = sorted([r for r in reports if r["symbol"] == symbol], key=lambda r: r["month"])
        parts = []
        for row in selected:
            path = output / row["zip"]
            if (
                sha(path) != row["zip_sha256"]
                or sha(path.with_name(path.name + ".CHECKSUM")) != row["checksum_sha256"]
            ):
                raise ValueError("Raw archive changed before merge")
            frame, _ = parse_archive(path.read_bytes(), symbol, row["month"])
            parts.append(frame)
        combined = pd.concat(parts)
        if combined.index.has_duplicates or not combined.index.is_monotonic_increasing:
            raise ValueError("Cross-month duplicate or reversed daily calendar")
        gaps = pd.date_range(combined.index[0], combined.index[-1], freq="D").difference(
            combined.index
        )
        path = output / "symbols" / (symbol + "_1d.parquet")
        combined.to_parquet(path)
        quality.append(
            {
                "symbol": symbol,
                "rows": len(combined),
                "first": combined.index[0].isoformat(),
                "last": combined.index[-1].isoformat(),
                "interior_missing_days": [t.isoformat() for t in gaps],
                "zero_quote_volume_days": int(combined.quote_volume.eq(0).sum()),
                "partial_candles": int((~combined.full_daily_duration).sum()),
                "parquet_sha256": sha(path),
            }
        )
    summary = {
        "schema": "historical-perpetual-daily.v1",
        "symbols": len(symbols),
        "rows": sum(r["rows"] for r in quality),
        "archives": len(reports),
        "http_requests": downloads.requests,
        "download_bytes": downloads.bytes,
        "quality": quality,
        "gap_days": sum(len(r["interior_missing_days"]) for r in quality),
        "partial_candles": sum(r["partial_candles"] for r in quality),
        "missing_days_imputed": False,
        "returns_clipped": False,
        "financial_accounts": 0,
        "profit_goal_met": False,
    }
    save(output / "summary.json", summary)
    save(
        output / "completion.json",
        {
            "manifest": {
                str(p.relative_to(output)): sha(p)
                for p in output.rglob("*")
                if p.is_file() and p.name != ".download.lock"
            },
            "raw_ingestion_complete": True,
            "financial_accounts": 0,
        },
    )
    print({k: v for k, v in summary.items() if k != "quality"}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    ingest(a.catalog, a.output)
