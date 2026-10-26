"""Collect real spot daily archives for frozen slots, retaining absence and gaps."""

import argparse
import hashlib
import io
import threading
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest


class Meter:
    def __init__(self, plan):
        self.calls = 0
        self.bytes = 0
        self.plan = plan
        self.lock = threading.Lock()

    def get(self, url, params=None):
        error = None
        for _ in range(2):
            with self.lock:
                if self.calls >= self.plan["maximum_daily_archive_http_requests"]:
                    raise ValueError("Registered HTTP budget exhausted")
                self.calls += 1
            try:
                r = requests.get(url, params=params, timeout=(10, 20))
                with self.lock:
                    self.bytes += len(r.content)
                    if self.bytes > self.plan["maximum_daily_source_bytes"]:
                        raise ValueError("Registered source byte budget exhausted")
                if r.status_code >= 500:
                    raise requests.HTTPError("Upstream server error", response=r)
                return r.content, {
                    "url": r.url,
                    "status": r.status_code,
                    "bytes": len(r.content),
                    "sha256": hashlib.sha256(r.content).hexdigest(),
                    "observed_at": pd.Timestamp.now(tz="UTC").isoformat(),
                }
            except requests.RequestException as exc:
                error = type(exc).__name__
        raise RuntimeError("Two read-only source attempts failed: " + str(error))


def parse(body, month):
    with zipfile.ZipFile(io.BytesIO(body)) as z:
        names = z.namelist()
        if len(names) != 1 or not names[0].endswith(".csv"):
            raise ValueError("Unexpected spot ZIP members")
        f = pd.read_csv(z.open(names[0]), header=None)
    if f.shape[1] != 12 or f.empty:
        raise ValueError("Unexpected spot daily schema")
    unit = "us" if month >= "2025-01" else "ms"
    index = pd.DatetimeIndex(pd.to_datetime(f[0], unit=unit, utc=True), name="date").as_unit("ns")
    start = pd.Timestamp(month + "-01", tz="UTC")
    expected = pd.date_range(start, start + pd.offsets.MonthBegin(), freq="D", inclusive="left")
    if not index.is_monotonic_increasing or index.has_duplicates or len(index.difference(expected)):
        raise ValueError("Spot daily clocks are duplicated, out of month or out of order")
    closing = pd.DatetimeIndex(pd.to_datetime(f[6], unit=unit, utc=True)).as_unit("ns")
    if not closing.equals(index + pd.Timedelta(days=1) - pd.Timedelta(1, unit=unit)):
        raise ValueError("Spot daily close time or archive unit differs")
    out = f[[1, 2, 3, 4, 5, 7]].astype(float)
    out.columns = ["open", "high", "low", "close", "volume", "quote_volume"]
    out.index = index
    finite = np.isfinite(out.to_numpy()).all(axis=1)
    valid = (
        finite
        & out.iloc[:, :4].gt(0).all(axis=1)
        & out.volume.gt(0)
        & out.quote_volume.gt(0)
        & out.high.ge(out[["open", "close", "low"]].max(axis=1))
        & out.low.le(out[["open", "close", "high"]].min(axis=1))
    )
    tolerance = 0.01 + 1e-6 * (out.high * out.volume).abs()
    valid &= out.quote_volume.ge(out.low * out.volume - tolerance) & out.quote_volume.le(
        out.high * out.volume + tolerance
    )
    out["feature_usable"] = valid
    return out, {
        "rows": len(out),
        "invalid_or_nonpositive_volume_rows": int((~valid).sum()),
        "absent_daily_labels": [t.isoformat() for t in expected.difference(index)],
        "unit": unit,
    }


def catalog(symbol, output, plan, meter):
    prefix = f"data/spot/monthly/klines/{symbol}/1d/"
    body, receipt = meter.get(plan["spot_listing"], {"prefix": prefix, "max-keys": 1000})
    path = output / "catalogs" / (symbol + ".xml")
    path.write_bytes(body)
    if receipt["status"] != 200:
        return {"symbol": symbol, "receipt": receipt, "error": "listing_http_error", "keys": []}
    root = ET.fromstring(body)
    for node in root.iter():
        node.tag = node.tag.split("}")[-1]
    if root.findtext("IsTruncated") != "false":
        raise ValueError("Source catalog truncated; cannot infer archive absence")
    keys = [node.text for node in root.iter("Key")]
    if any(not k.startswith(prefix) for k in keys):
        raise ValueError("Source catalog escaped requested prefix")
    return {"symbol": symbol, "receipt": receipt, "error": None, "keys": keys}


def archive_job(request, catalogs, output, plan, meter):
    symbol, month = request["symbol"], request["month"]
    url = plan["spot_archive"].format(symbol=symbol, month=month)
    key = url.split("data.binance.vision/", 1)[1]
    cat = catalogs[symbol]
    base = {**request, "url": url}
    if cat["error"]:
        return {**base, "state": "unknown_catalog_error"}
    if key not in cat["keys"] or key + ".CHECKSUM" not in cat["keys"]:
        return {**base, "state": "unknown_archive_absent_from_current_catalog"}
    checksum, check_receipt = meter.get(url + ".CHECKSUM")
    check_path = output / "archives" / (url.rsplit("/", 1)[1] + ".CHECKSUM")
    check_path.write_bytes(checksum)
    if check_receipt["status"] != 200:
        return {**base, "state": "unknown_checksum_http_error", "checksum_receipt": check_receipt}
    expected = checksum.decode().split()[0]
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("Official archive checksum format invalid")
    body, receipt = meter.get(url)
    path = output / "archives" / url.rsplit("/", 1)[1]
    path.write_bytes(body)
    if receipt["status"] != 200:
        return {**base, "state": "unknown_archive_http_error", "receipt": receipt}
    if hashlib.sha256(body).hexdigest() != expected:
        raise ValueError("Official spot archive checksum mismatch")
    frame, info = parse(body, month)
    frame.to_parquet(output / "months" / (symbol + "_" + month + ".parquet"))
    return {
        **base,
        "state": "verified_source",
        **info,
        "receipt": receipt,
        "checksum_receipt": check_receipt,
        "official_sha256": expected,
    }


def collect(registration, output):
    verify_manifest(registration)
    plan = read(registration / "plan.json")
    frozen = read(registration / "registration.json")
    if any(sha(p) != h for p, h in frozen["source_hashes"].items()):
        raise ValueError("Registered source changed before acquisition")
    jobs = read(registration / "daily_requests.json")
    symbols = sorted({r["symbol"] for r in jobs})
    if 4 * len(jobs) + 2 * len(symbols) + 2 > plan["budget"]["maximum_daily_archive_http_requests"]:
        raise ValueError("Worst-case two-attempt collection exceeds registered budget")
    output.mkdir(parents=True, exist_ok=False)
    for name in ["sources", "catalogs", "archives", "months", "symbols", "records"]:
        (output / name).mkdir()
    save(
        output / "registration.json",
        {
            "parent_completion_sha256": sha(registration / "completion.json"),
            "source_hashes": frozen["source_hashes"],
            "before_new_spot_financial_results": True,
        },
    )
    meter = Meter(plan["budget"])
    body, receipt = meter.get(plan["source_readme"])
    if receipt["status"] != 200:
        raise ValueError("Pinned primary source unavailable")
    (output / "sources" / "binance-public-data-README.md").write_bytes(body)
    save(output / "sources" / "receipt.json", receipt)
    catalogs = {}
    with ThreadPoolExecutor(max_workers=plan["budget"]["maximum_daily_workers"]) as pool:
        futures = {pool.submit(catalog, s, output, plan, meter): s for s in symbols}
        for future in as_completed(futures):
            s = futures[future]
            try:
                catalogs[s] = future.result()
            except Exception as exc:
                catalogs[s] = {
                    "symbol": s,
                    "keys": [],
                    "error": type(exc).__name__ + ": " + str(exc),
                }
            save(output / "catalogs" / (s + ".json"), catalogs[s])
    print({"catalogs": len(catalogs), "jobs": len(jobs), "http": meter.calls}, flush=True)
    records = []
    with ThreadPoolExecutor(max_workers=plan["budget"]["maximum_daily_workers"]) as pool:
        futures = {pool.submit(archive_job, r, catalogs, output, plan, meter): r for r in jobs}
        for future in as_completed(futures):
            request = futures[future]
            try:
                record = future.result()
            except Exception as exc:
                record = {
                    **request,
                    "state": "unknown_source_error",
                    "error": type(exc).__name__ + ": " + str(exc),
                }
            save(
                output / "records" / (request["symbol"] + "_" + request["month"] + ".json"), record
            )
            records.append(record)
            if len(records) % 100 == 0:
                print(
                    {
                        "completed": len(records),
                        "total": len(jobs),
                        "http": meter.calls,
                        "source_bytes": meter.bytes,
                    },
                    flush=True,
                )
    for s in symbols:
        paths = sorted((output / "months").glob(s + "_*.parquet"))
        if paths:
            frame = pd.concat([pd.read_parquet(p) for p in paths]).sort_index()
            if frame.index.has_duplicates:
                raise ValueError("Combined monthly source duplicated an actual observation")
            frame.to_parquet(output / "symbols" / (s + "_1d.parquet"))
    records.sort(key=lambda r: (r["symbol"], r["month"]))
    summary = {
        "schema": "matched-carry-spot-daily-sources.v1",
        "symbols": len(symbols),
        "requested_months": len(jobs),
        "verified_months": sum(r["state"] == "verified_source" for r in records),
        "unknown_months": sum(r["state"] != "verified_source" for r in records),
        "unknown_source_errors": sum(r["state"] == "unknown_source_error" for r in records),
        "source_rows": sum(r.get("rows", 0) for r in records),
        "http_requests": meter.calls,
        "response_bytes": meter.bytes,
        "records": records,
        "financial_accounts": 0,
        "spot_perpetual_identity_certified": False,
        "historical_tradability_certified": False,
        "profit_goal_met": False,
        "portfolio_actual": None,
    }
    save(output / "summary.json", summary)
    save(
        output / "completion.json",
        {
            "manifest": {
                str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()
            },
            "financial_accounts": 0,
        },
    )
    print({k: v for k, v in summary.items() if k != "records"}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--registration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    collect(args.registration, args.output)
