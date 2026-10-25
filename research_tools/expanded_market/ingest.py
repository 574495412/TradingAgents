"""Bounded checksum ingestion of sparse five-minute source calendars, without repair."""

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
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest


def flags(frame):
    b = frame[["open", "high", "low", "close"]]
    prices = np.isfinite(b.to_numpy()).all(axis=1) & b.gt(0).all(axis=1).to_numpy()
    prices &= frame.high.ge(b.max(axis=1)).to_numpy() & frame.low.le(b.min(axis=1)).to_numpy()
    finite = np.isfinite(frame[["volume", "quote_volume"]].to_numpy()).all(axis=1)
    tolerance = 1e-4 + 1e-6 * (frame.high * frame.volume).abs()
    liquidity = (
        prices & finite & frame.volume.gt(0).to_numpy() & frame.quote_volume.gt(0).to_numpy()
    )
    liquidity &= frame.quote_volume.le(frame.high * frame.volume + tolerance).to_numpy()
    liquidity &= frame.quote_volume.ge(frame.low * frame.volume - tolerance).to_numpy()
    out = frame.copy()
    out["price_usable"] = prices
    out["execution_liquidity_usable"] = liquidity
    return out


def parse(payload, symbol, month):
    name = f"{symbol}-5m-{month}.csv"
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        if (
            z.namelist() != [name]
            or z.getinfo(name).file_size > 15_000_000
            or z.testzip() is not None
        ):
            raise ValueError("Archive member, CRC or uncompressed byte limit differs")
        raw = pd.read_csv(io.BytesIO(z.read(name)), header=None, dtype=str)
    if len(raw) and raw.iloc[0, 0] == "open_time":
        raw = raw.iloc[1:].reset_index(drop=True)
    if raw.shape[1] != 12 or not 0 < len(raw) <= 31 * 288:
        raise ValueError("Unexpected5m source schema or row count")
    values = raw.apply(pd.to_numeric, errors="raise")
    epoch = values[0].to_numpy(dtype="int64")
    unit = "us" if epoch.min() > 100_000_000_000_000 else "ms"
    index = pd.DatetimeIndex(pd.to_datetime(epoch, unit=unit, utc=True), name="date")
    start = pd.Timestamp(month + "-01", tz="UTC")
    end = start + pd.offsets.MonthBegin(1)
    if (
        index.has_duplicates
        or not index.is_monotonic_increasing
        or not index.floor("5min").equals(index)
        or index[0] < start
        or index[-1] >= end
    ):
        raise ValueError("Duplicate, non5m, unordered or wrong-month source time")
    frame = pd.DataFrame(
        {
            k: values[i].to_numpy()
            for k, i in [
                ("open", 1),
                ("high", 2),
                ("low", 3),
                ("close", 4),
                ("volume", 5),
                ("quote_volume", 7),
                ("trades", 8),
                ("taker_base_volume", 9),
                ("taker_quote_volume", 10),
            ]
        },
        index=index,
    )
    frame["close_time"] = pd.to_datetime(values[6].to_numpy(dtype="int64"), unit=unit, utc=True)
    frame = flags(frame)
    full = frame.close_time.ge(
        index + pd.Timedelta(minutes=5) - pd.Timedelta(milliseconds=1)
    ) & frame.close_time.lt(index + pd.Timedelta(minutes=5))
    frame["full_source_candle_duration"] = full
    frame["execution_liquidity_usable"] &= full
    frame["auxiliary_usable"] = (
        np.isfinite(frame[["trades", "taker_base_volume", "taker_quote_volume"]].to_numpy()).all(
            axis=1
        )
        & frame.trades.ge(0)
        & frame.trades.eq(np.floor(frame.trades))
        & frame.taker_base_volume.ge(0)
        & frame.taker_base_volume.le(frame.volume + 1e-6)
        & frame.taker_quote_volume.ge(0)
        & frame.taker_quote_volume.le(frame.quote_volume + 1e-4)
    )
    return frame, {
        "rows": len(frame),
        "timestamp_unit": unit,
        "unusable_price_bars": int((~frame.price_usable).sum()),
        "unusable_execution_bars": int((~frame.execution_liquidity_usable).sum()),
        "unusable_auxiliary_bars": int((~frame.auxiliary_usable).sum()),
    }


class Download:
    def __init__(self, root, plan):
        self.root, self.plan = root, plan
        self.lock = threading.Lock()
        self.local = threading.local()
        self.requests = 0
        self.bytes = 0
        self.journal = (root / "transport.jsonl").open("x")

    def get(self, key):
        if not key.startswith(self.plan["prefix"]) or ".." in key.split("/"):
            raise ValueError("Unregistered key path")
        if not hasattr(self.local, "session"):
            self.local.session = requests.Session()
        for attempt in range(self.plan["budget"]["retry_attempts"]):
            with self.lock:
                self.requests += 1
                sequence = self.requests
                if sequence > self.plan["budget"]["maximum_download_http_requests"]:
                    raise ValueError("HTTP budget exceeded")
            try:
                r = self.local.session.get(self.plan["download_origin"] + "/" + key, timeout=25)
                body = r.content
                with self.lock:
                    self.bytes += len(body)
                    self.journal.write(
                        json.dumps(
                            {
                                "sequence": sequence,
                                "key": key,
                                "attempt": attempt + 1,
                                "status": r.status_code,
                                "bytes": len(body),
                                "sha256": hashlib.sha256(body).hexdigest(),
                            }
                        )
                        + "\n"
                    )
                    self.journal.flush()
                    if (
                        self.bytes > self.plan["budget"]["maximum_total_bytes"]
                        or len(body) > self.plan["budget"]["maximum_file_bytes"]
                    ):
                        raise ValueError("Fixed byte budget exceeded")
                r.raise_for_status()
                return body
            except requests.RequestException as exc:
                with self.lock:
                    self.journal.write(
                        json.dumps(
                            {
                                "sequence": sequence,
                                "key": key,
                                "attempt": attempt + 1,
                                "error_type": type(exc).__name__,
                            }
                        )
                        + "\n"
                    )
                    self.journal.flush()
                if attempt + 1 == self.plan["budget"]["retry_attempts"]:
                    raise
                time.sleep(min(2**attempt, 4))
        raise AssertionError("Unreachable retry state")

    def symbol(self, symbol, records):
        folder = self.root / "raw" / symbol
        folder.mkdir()
        parsed = []
        sources = []
        for row in sorted(records, key=lambda r: r["month"]):
            payload = self.get(row["key"])
            checksum = self.get(row["checksum_key"])
            name = Path(row["key"]).name
            archive = folder / name
            checkfile = folder / (name + ".CHECKSUM")
            archive.write_bytes(payload)
            checkfile.write_bytes(checksum)
            match = re.fullmatch(r"([0-9a-fA-F]{64})\s+\*?([^\r\n]+)\s*", checksum.decode().strip())
            if (
                not match
                or match[1].lower() != sha(archive)
                or match[2].strip() != name
                or len(payload) != row["size"]
            ):
                raise ValueError("Official checksum, basename or listed size differs")
            frame, quality = parse(payload, symbol, row["month"])
            sources.append(
                {
                    **row,
                    "path": str(archive),
                    "sha256": sha(archive),
                    "checksum_sha256": sha(checkfile),
                    **quality,
                }
            )
            begin, end = (
                pd.Timestamp(self.plan["bars_start"], tz="UTC"),
                pd.Timestamp(self.plan["end_exclusive"], tz="UTC"),
            )
            parsed.append(frame.loc[(frame.index >= begin) & (frame.index < end)])
        if not parsed or not sum(len(p) for p in parsed):
            raise ValueError("No actual listed price source in scope; not zero data")
        frame = pd.concat(parsed)
        if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
            raise ValueError("Cross-month source timestamps differ")
        output = self.root / "symbols" / (symbol + "_5m.parquet")
        frame.to_parquet(output)
        report = profile(symbol, frame, self.plan)
        save(
            self.root / "reports" / (symbol + ".json"),
            {"symbol": symbol, "sources": sources, "output_sha256": sha(output), **report},
        )
        return {
            "symbol": symbol,
            "passed": True,
            "source_kind": "new_official_archives",
            "archives": len(sources),
            "output_sha256": sha(output),
            **report,
        }


def profile(symbol, frame, plan):
    inside = pd.date_range(frame.index[0], frame.index[-1], freq="5min")
    full = pd.date_range(
        plan["bars_start"], plan["end_exclusive"], freq="5min", tz="UTC", inclusive="left"
    )
    return {
        "rows": len(frame),
        "first": frame.index[0].isoformat(),
        "last": frame.index[-1].isoformat(),
        "interior_missing_bars": len(inside.difference(frame.index)),
        "outside_source_range_bars": len(full.difference(inside)),
        "unusable_price_bars": int((~frame.price_usable).sum()),
        "unusable_execution_bars": int((~frame.execution_liquidity_usable).sum()),
        "missing_calendars_filled": False,
        "historical_tradability_certified": False,
    }


def ingest(catalog, output):
    output.mkdir(parents=True, exist_ok=False)
    lock = (output / ".collection.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify_manifest(catalog)
    plan = read(catalog / "plan.json")
    paths = [
        Path(__file__),
        Path("research_tools/expanded_market/plan.json"),
        catalog / "completion.json",
    ]
    save(
        output / "plan.json",
        {
            **plan,
            "source_hashes": {str(p): sha(p) for p in paths},
            "catalog": str(catalog),
            "before_archive_download": True,
        },
    )
    for name in ["raw", "symbols", "reports"]:
        (output / name).mkdir()
    refs = read(catalog / "references.json")
    rows = []
    errors = []
    for r in refs:
        source = Path(r["path"])
        if sha(source) != r["sha256"]:
            raise ValueError("Sealed reference changed")
        frame = pd.read_parquet(source).loc[plan["bars_start"] :]
        frame = frame.loc[frame.index < pd.Timestamp(plan["end_exclusive"], tz="UTC")]
        frame = flags(frame)
        path = output / "symbols" / (r["symbol"] + "_5m.parquet")
        frame.to_parquet(path)
        report = {
            **r,
            "source_kind": "sealed_six_column_reference",
            "auxiliary_source_check_in_this_phase": None,
            "passed": True,
            "output_sha256": sha(path),
            **profile(r["symbol"], frame, plan),
        }
        save(output / "reports" / (r["symbol"] + ".json"), report)
        rows.append(report)
    records = read(catalog / "archives.json")
    by = {}
    for r in records:
        by.setdefault(r["symbol"], []).append(r)
    missing = [r["symbol"] for r in read(catalog / "summary.json")["reports"] if not r["archives"]]
    for s in missing:
        errors.append(
            {
                "symbol": s,
                "passed": False,
                "error_type": "NoListedArchive",
                "zero_data_substitute": False,
            }
        )
    download = Download(output, plan)
    started = time.monotonic()
    journal = (output / "symbols.jsonl").open("x")
    with ThreadPoolExecutor(max_workers=plan["budget"]["workers"]) as pool:
        futures = {pool.submit(download.symbol, s, v): s for s, v in by.items()}
        for i, future in enumerate(as_completed(futures)):
            symbol = futures[future]
            try:
                r = future.result()
                rows.append(r)
            except Exception as exc:
                r = {"symbol": symbol, "passed": False, "error_type": type(exc).__name__}
                errors.append(r)
            journal.write(json.dumps(r, allow_nan=False) + "\n")
            journal.flush()
            print(
                {
                    "finished_new_symbols": i + 1,
                    "of": len(by),
                    "passed": sum(r["passed"] for r in rows),
                    "errors": len(errors),
                    "http_requests": download.requests,
                    "bytes": download.bytes,
                    "seconds": round(time.monotonic() - started, 1),
                },
                flush=True,
            )
    journal.close()
    download.journal.close()
    save(
        output / "download_summary.json",
        {
            "schema": "expanded5m-ingestion.v1",
            "passed": not errors,
            "symbols": len(rows),
            "reused_reference_symbols": len(refs),
            "new_source_symbols": len(rows) - len(refs),
            "rows": sum(r["rows"] for r in rows),
            "records": rows,
            "errors": errors,
            "http_requests": download.requests,
            "bytes": download.bytes,
            "new_financial_accounts": 0,
            "profit_goal_met": False,
            "missing_prices_imputed": False,
            "historical_status_certified": False,
        },
    )
    if errors:
        raise ValueError("Incomplete input ingestion; preserve explicit failed capture")
    save(
        output / "completion.json",
        {
            "manifest": {
                str(p.relative_to(output)): sha(p)
                for p in output.rglob("*")
                if p.is_file() and p.name != ".collection.lock"
            },
            "input_capture_complete": True,
            "financial_accounts": 0,
            "historical_certification_passed": False,
        },
    )
    print(
        {"input_capture_complete": True, "symbols": len(rows), "new_financial_accounts": 0},
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    ingest(a.catalog, a.output)
