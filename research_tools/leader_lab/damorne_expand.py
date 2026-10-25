"""Add independently registered liquid contracts to the checksummed market dataset."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from .damorne_data import archive_job, get
from .evidence import save, sha


def cached_archive(root, symbol, kind, period, frequency):
    path = root / kind / symbol / f"{symbol}-1m-{period}.zip"
    checksum = path.with_name(path.name + ".CHECKSUM")
    if path.exists() and checksum.exists():
        expected = checksum.read_text().split()[0]
        if sha(path) != expected:
            raise ValueError("Retained official archive checksum changed")
        return {
            "symbol": symbol,
            "kind": kind,
            "period": period,
            "frequency": frequency,
            "path": str(path),
            "sha256": expected,
            "url": f"https://data.binance.vision/data/futures/um/{frequency}/{kind}/{symbol}/1m/{path.name}",
        }
    return archive_job(root, symbol, kind, period, frequency)


def expand(old, root):
    output = root / "combined_data"
    registration = json.loads((root / "liquid_universe_registration.json").read_text())
    protocol = json.loads((old / "protocol.json").read_text())
    extra = sorted(set(registration["symbols"]) - set(protocol["symbols"]))
    protocol.update(
        symbols=sorted(set(protocol["symbols"]) | set(extra)),
        training_symbols=registration["symbols"],
        liquidity_universe_source_sha256=sha(root / "liquid_universe_registration.json"),
        scope="primary fixed top20 liquid public-market universe; original leader instruments are transfer diagnostics only",
    )
    if (output / "protocol.json").exists():
        if json.loads((output / "protocol.json").read_text()) != protocol:
            raise ValueError("Expansion protocol changed")
        (output / "recovery_expand_source.py").write_bytes(Path(__file__).read_bytes())
    else:
        save(output / "protocol.json", protocol)
        (output / "executed_expand_source.py").write_bytes(Path(__file__).read_bytes())
    sources = json.loads((old / "market_sources.json").read_text())
    backfill = json.loads((old / "rest_backfill_hashes.json").read_text())
    funding = json.loads((old / "funding_sources.json").read_text())
    jobs = [
        (symbol, kind, period, frequency)
        for symbol in extra
        for kind in ["klines", "markPriceKlines"]
        for period, frequency in [
            ("2026-09", "monthly"),
            *[(f"2026-10-{day:02d}", "daily") for day in range(1, 24)],
        ]
    ]
    failures = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        tasks = {pool.submit(cached_archive, output / "market_raw", *job): job for job in jobs}
        for task in as_completed(tasks):
            try:
                sources.append(task.result())
            except Exception as exc:
                failures.append({"job": tasks[task], "error": type(exc).__name__ + ":" + str(exc)})
            if (len(sources) - 920 + len(failures)) % 100 == 0:
                print({"new_archives": len(sources) - 920, "failures": len(failures)}, flush=True)
    failure_file = output / "archive_failures.json"
    suffix = 1
    while failure_file.exists():
        failure_file = output / f"archive_failures_recovery_{suffix:03d}.json"
        suffix += 1
    save(failure_file, failures)
    for failure in failures:
        symbol, kind, period, frequency = failure["job"]
        if frequency != "daily" or "404" not in failure["error"]:
            raise ValueError("Unexpected archive source gap; no zero fill")
        endpoint = "klines" if kind == "klines" else "markPriceKlines"
        lower = int(datetime.fromisoformat(period + "T00:00:00+00:00").timestamp() * 1000)
        upper = lower + 86400000
        url = f"https://fapi.binance.com/fapi/v1/{endpoint}?symbol={symbol}&interval=1m&startTime={lower}&endTime={upper - 1}&limit=1500"
        body = get(url)
        rows = json.loads(body)
        if (
            not isinstance(rows, list)
            or len(rows) != 1440
            or rows[0][0] != lower
            or rows[-1][0] != upper - 60000
        ):
            raise ValueError("Incomplete endpoint backfill")
        path = output / "rest_backfill" / kind / f"{symbol}-{period}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        backfill.append(
            {
                "symbol": symbol,
                "kind": kind,
                "path": str(path),
                "sha256": sha(path),
                "format": "rest_json",
                "url": url,
            }
        )
    for symbol in extra:
        url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&startTime=1789084800000&endTime=1792799999999&limit=1000"
        body = get(url)
        rows = json.loads(body)
        if not isinstance(rows, list) or not rows or len(rows) == 1000:
            raise ValueError("Funding requires complete source")
        path = output / "funding_raw" / f"{symbol}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        funding.append(
            {
                "symbol": symbol,
                "path": str(path),
                "sha256": sha(path),
                "rows": len(rows),
                "url": url,
            }
        )
    save(output / "market_sources.json", sources)
    save(output / "rest_backfill_hashes.json", backfill)
    save(output / "funding_sources.json", funding)
    # True report-end marks, never used for training or a decision signal.
    old_marks = json.loads((root / "evaluation_marks.json").read_text())
    for symbol in extra:
        body = get(
            f"https://fapi.binance.com/fapi/v1/markPriceKlines?symbol={symbol}&interval=1m&startTime=1792800000000&endTime=1792800059999&limit=1"
        )
        row = json.loads(body)
        if not isinstance(row, list) or len(row) != 1 or row[0][0] != 1792800000000:
            raise ValueError("Missing exact terminal mark")
        old_marks["marks"][symbol] = float(row[0][1])
    save(root / "combined_evaluation_marks.json", old_marks)
    save(
        output / "capture_receipt.json",
        {
            "recorded_utc": datetime.now(UTC).isoformat(),
            "additional_symbols": extra,
            "official_archive_count": len(sources),
            "backfill_count": len(backfill),
            "funding_symbols": len(funding),
            "source_sha256": sha(Path(__file__)),
            "credentials_used": False,
        },
    )
    print(
        {"additional_symbols": extra, "archives": len(sources), "funding": len(funding)}, flush=True
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("old", type=Path)
    p.add_argument("root", type=Path)
    a = p.parse_args()
    expand(a.old.absolute(), a.root.absolute())
