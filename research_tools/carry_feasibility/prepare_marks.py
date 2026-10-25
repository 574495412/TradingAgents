"""Prepare actual mark OHLC, then independently re-read all raw CSV values."""

import argparse
import csv
import hashlib
import io
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from research_tools.carry_feasibility.prepare import expected_clock
from research_tools.factor_lab.io import read, save, sha


def fetch(url):
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=(10, 30))
            response.raise_for_status()
            return response.content
        except requests.RequestException:
            if attempt == 2:
                raise


def job(root, plan, symbol, month, cache):
    url = plan["archive"].format(symbol=symbol, month=month)
    old = cache / "archives" / url.rsplit("/", 1)[1] if cache else None
    if old and old.is_file() and old.with_suffix(".zip.CHECKSUM").is_file():
        checksum = old.with_suffix(".zip.CHECKSUM").read_bytes()
        body = old.read_bytes()
    else:
        checksum = fetch(url + ".CHECKSUM")
        body = fetch(url)
    expected = checksum.decode().split()[0]
    if hashlib.sha256(body).hexdigest() != expected:
        raise ValueError("Official mark checksum mismatch")
    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        if len(archive.namelist()) != 1:
            raise ValueError("Unexpected mark archive members")
        frame = pd.read_csv(archive.open(archive.namelist()[0]), header=None)
    if frame.iloc[0, 0] == "open_time":
        frame = frame.iloc[1:].reset_index(drop=True)
    raw_count = len(frame)
    start = pd.Timestamp(month + "-01", tz="UTC")
    index = pd.DatetimeIndex(
        pd.to_datetime(frame[0].astype("int64"), unit="ms", utc=True), name="date"
    )
    expected_index = expected_clock(start, start + pd.offsets.MonthBegin())
    repair_count = 0
    if not index.equals(expected_index) and month == "2026-06":
        gap = plan["known_gap"]
        gap_clock = expected_clock(gap["start"], gap["end_exclusive"])
        if (
            index.has_duplicates
            or len(index.difference(expected_index))
            or not expected_index.difference(index).equals(gap_clock)
        ):
            raise ValueError("Mark gap differs from registered real API repair")
        raw_repair = read(root / "repair_connector.json")
        page = next(p for p in raw_repair["pages"] if p["request"]["symbol"] == symbol)
        patch = pd.DataFrame(page["response"]["result"])
        if not pd.DatetimeIndex(pd.to_datetime(patch[0], unit="ms", utc=True)).equals(gap_clock):
            raise ValueError("Actual API repair does not cover exact missing clock")
        frame = (
            pd.concat([frame, patch])
            .sort_values(0, key=lambda c: c.astype("int64"))
            .reset_index(drop=True)
        )
        index = pd.DatetimeIndex(
            pd.to_datetime(frame[0].astype("int64"), unit="ms", utc=True), name="date"
        )
        repair_count = len(patch)
    if frame.shape[1] != 12 or not index.equals(expected_index):
        raise ValueError("Mark calendar or schema differs")
    if not np.array_equal(
        pd.to_datetime(frame[6].astype("int64"), unit="ms", utc=True).array,
        index + pd.Timedelta(minutes=5) - pd.Timedelta(milliseconds=1),
    ):
        raise ValueError("Mark interval close differs")
    out = frame[[1, 2, 3, 4]].astype(float)
    out.columns = ["open", "high", "low", "close"]
    out.index = index
    if (
        not np.isfinite(out.to_numpy()).all()
        or (out <= 0).any().any()
        or (out.high < out.max(axis=1)).any()
        or (out.low > out.min(axis=1)).any()
    ):
        raise ValueError("Invalid mark OHLC")
    path = root / "archives" / url.rsplit("/", 1)[1]
    with path.open("xb") as stream:
        stream.write(body)
    path.with_suffix(".zip.CHECKSUM").write_bytes(checksum)
    return out, {
        "symbol": symbol,
        "month": month,
        "url": url,
        "sha256": expected,
        "rows": len(out),
        "archive_rows": raw_count,
        "actual_api_repair_rows": repair_count,
    }


def prepare(plan_path, root, cache=None):
    if (root / "plan.json").exists():
        raise FileExistsError("Preserve a prior or partial mark collection")
    plan = read(plan_path)
    probe = read(root / "connector.json")
    (root / "archives").mkdir(exist_ok=False)
    plan["source_hashes"] = {
        str(p): sha(p)
        for p in [
            plan_path,
            Path(__file__),
            Path("research_tools/carry_feasibility/prepare.py"),
            Path("research_tools/factor_lab/io.py"),
        ]
    }
    plan["connector_sha256"] = sha(root / "connector.json")
    plan["repair_connector_sha256"] = sha(root / "repair_connector.json")
    plan["preserved_partial_cache"] = str(cache) if cache else None
    plan["spot_completion_sha256"] = sha(Path(plan["spot_data"]) / "completion.json")
    save(root / "plan.json", plan)
    months = pd.date_range(
        plan["start"], plan["end_exclusive"], freq="MS", inclusive="left"
    ).strftime("%Y-%m")
    requests = [(s, m) for s in plan["symbols"] for m in months]
    if len(requests) != plan["budget_monthly_files"]:
        raise ValueError("Mark budget differs")
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(job, root, plan, s, m, cache) for s, m in requests]
        results = []
        for i, future in enumerate(futures, 1):
            results.append(future.result())
            print(f"verified actual mark archive {i}/{len(futures)}", flush=True)
    counts = []
    max_error = 0.0
    for symbol in plan["symbols"]:
        frame = pd.concat([f for f, r in results if r["symbol"] == symbol])
        if not frame.index.equals(expected_clock(plan["start"], plan["end_exclusive"])):
            raise ValueError("Whole mark clock differs")
        frame.to_parquet(root / f"{symbol}_mark_5m.parquet")
        independent_times, independent_prices = [], []
        for _, receipt in results:
            if receipt["symbol"] != symbol:
                continue
            path = root / "archives" / receipt["url"].rsplit("/", 1)[1]
            if sha(path) != path.with_suffix(".zip.CHECKSUM").read_text().split()[0]:
                raise ValueError("Independent mark checksum differs")
            with zipfile.ZipFile(path) as archive:
                rows = list(csv.reader(io.TextIOWrapper(archive.open(archive.namelist()[0]))))
            if rows[0][0] == "open_time":
                rows = rows[1:]
            independent_times.extend(int(r[0]) for r in rows)
            independent_prices.extend([[float(r[i]) for i in [1, 2, 3, 4]] for r in rows])
        page = next(
            p
            for p in read(root / "repair_connector.json")["pages"]
            if p["request"]["symbol"] == symbol
        )
        independent_times.extend(int(r[0]) for r in page["response"]["result"])
        independent_prices.extend(
            [[float(r[i]) for i in [1, 2, 3, 4]] for r in page["response"]["result"]]
        )
        order = np.argsort(independent_times)
        independent_times = np.array(independent_times)[order]
        array = np.array(independent_prices)[order]
        if not np.array_equal(frame.index.as_unit("ms").asi8, independent_times) or not np.allclose(
            frame.to_numpy(), array, rtol=1e-14, atol=1e-9
        ):
            raise ValueError("Independent raw mark reconstruction differs")
        max_error = max(max_error, float(np.max(np.abs(frame.to_numpy() - array))))
        counts.append(
            {"symbol": symbol, "rows": len(frame), "raw_price_values": array.size, "passed": True}
        )
    frame = pd.read_parquet(root / "BTCUSDT_mark_5m.parquet")
    for row in probe["response"]["result"]:
        when = pd.Timestamp(row[0], unit="ms", tz="UTC")
        if not np.allclose(
            frame.loc[when].to_numpy(), np.array(row[1:5], dtype=float), rtol=0, atol=1e-8
        ):
            raise ValueError("Mark connector sample differs from archive")
    summary = {
        "schema": "actual-mark-path-data.v1",
        "archives": [r for _, r in results],
        "counts": counts,
        "independent_csv_audit_passed": True,
        "maximum_price_error": max_error,
        "raw_api_sample_bars": 2,
        "actual_api_gap_repair_bars": sum(r["actual_api_repair_rows"] for _, r in results),
        "start": plan["start"],
        "end_exclusive": plan["end_exclusive"],
        "financial_accounts": 0,
        "profit_proven": False,
    }
    save(root / "summary.json", summary)
    files = [
        root / n for n in ["plan.json", "connector.json", "repair_connector.json", "summary.json"]
    ] + list(root.glob("*.parquet"))
    save(
        root / "completion.json",
        {
            "manifest": {p.name: sha(p) for p in files},
            "audit_passed": True,
            "archives": len(results),
        },
    )
    print(
        {
            "mark_data_complete": True,
            "counts": counts,
            "maximum_independent_price_error": max_error,
        },
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan", type=Path, default=Path("research_tools/carry_feasibility/mark_plan.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    prepare(args.plan, args.output, args.cache)
