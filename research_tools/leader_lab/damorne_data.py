"""Bounded, anonymous leader-order capture and checksummed official market archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from .evidence import PUBLIC_BASE, save, sha

LEADER = "5219911688676500225"
START = "2026-09-11T00:00:00+00:00"
END = "2026-10-24T00:00:00+00:00"
DAY = 86_400_000


def get(url, payload=None, limit=20_000_000):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"User-Agent": "LeaderLab-read-only/1.0", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=25) as response:
        body = response.read(limit + 1)
    if len(body) > limit:
        raise ValueError("Response exceeds explicit size bound")
    return body


def capture_orders(output, start_ms, end_ms):
    output.mkdir(exist_ok=False)
    requests, records = [], []

    def window(lower, upper, depth=0):
        if len(requests) >= 128 or depth > 16:
            raise ValueError("Public-order subdivision bound reached")
        payload = {
            "portfolioId": LEADER,
            "startTime": lower,
            "endTime": upper,
            "pageSize": 200,
            "pageNumber": 1,
        }
        for attempt in range(3):
            body = get(PUBLIC_BASE + "lead-portfolio/order-history", payload, 4_000_000)
            name = f"orders_{len(requests):03d}.json"
            (output / name).write_bytes(body)
            data = json.loads(body)
            receipt = {
                "path": name,
                "sha256": sha(output / name),
                "request": payload,
                "code": data.get("code"),
                "captured_utc": datetime.now(UTC).isoformat(),
            }
            requests.append(receipt)
            if data.get("code") == "000000":
                break
            if data.get("code") != "11012005":
                break
            time.sleep(0.3 * (attempt + 1))
        if data.get("code") != "000000" or not isinstance(data.get("data"), dict):
            receipt["unavailable_window"] = True
            return
        rows, total = data["data"].get("list"), data["data"].get("total")
        if not isinstance(rows, list) or not isinstance(total, int) or total < len(rows):
            raise ValueError("Malformed public-order page")
        receipt.update(total=total, returned=len(rows))
        if total >= 200:
            if upper - lower < 2:
                raise ValueError("Cannot uniquely capture this dense millisecond without IDs")
            middle = (lower + upper) // 2
            window(lower, middle, depth + 1)
            window(middle + 1, upper, depth + 1)
            receipt["superseded_by_disjoint_subwindows"] = True
            return
        if total != len(rows):
            raise ValueError("Incomplete order window")
        for row in rows:
            # API filtering may use order creation or update; enforce both coverage interpretations.
            creation, update = int(row["orderTime"]), int(row["orderUpdateTime"])
            if not (lower <= creation <= upper or lower <= update <= upper):
                raise ValueError("API returned an order outside both requested clocks")
            records.append(row)

    try:
        lower = start_ms
        while lower < end_ms:
            upper = min(end_ms - 1, lower + 3 * DAY - 1)
            window(lower, upper)
            lower = upper + 1
        # No order IDs are published. Identical rows are retained as an explicit ambiguity.
        fingerprints = [json.dumps(x, sort_keys=True, separators=(",", ":")) for x in records]
        repeated = len(fingerprints) - len(set(fingerprints))
        if repeated:
            raise ValueError(f"{repeated} identical no-ID rows require source reconciliation")
        records.sort(
            key=lambda r: (int(r["orderUpdateTime"]), r["symbol"], r["positionSide"], r["side"])
        )
        save(
            output / "orders.json",
            {
                "leader_id": LEADER,
                "start_ms": start_ms,
                "end_ms_exclusive": end_ms,
                "rows": records,
                "requests": requests,
                "scope": "executed-order summaries as declared for disjoint public windows, unavailable windows retained; not unfilled/cancelled orders or individual fills",
                "unavailable_windows": [
                    r["request"] for r in requests if r.get("unavailable_window")
                ],
                "orders_have_unique_ids": False,
                "credentials_used": False,
            },
        )
        return len(records)
    finally:
        save(output / "requests.json", requests)


def archive_job(root, symbol, kind, period, frequency):
    name = f"{symbol}-1m-{period}.zip"
    base = f"https://data.binance.vision/data/futures/um/{frequency}/{kind}/{symbol}/1m/{name}"
    directory = root / kind / symbol
    directory.mkdir(parents=True, exist_ok=True)
    checksum = get(base + ".CHECKSUM", limit=1024)
    expected = checksum.decode().split()[0]
    body = get(base)
    if hashlib.sha256(body).hexdigest() != expected:
        raise ValueError("Official archive checksum mismatch")
    target = directory / name
    with target.open("xb") as stream:
        stream.write(body)
    (directory / (name + ".CHECKSUM")).write_bytes(checksum)
    return {
        "symbol": symbol,
        "kind": kind,
        "period": period,
        "frequency": frequency,
        "path": str(target),
        "sha256": expected,
        "url": base,
    }


def capture_order_cursor(output, start_ms, end_ms):
    """Public website uses an order-time cursor; require declared count reconciliation."""
    output.mkdir(exist_ok=False)
    rows, receipts, cursor, declared = [], [], "", None
    for page in range(20):
        payload = {
            "portfolioId": LEADER,
            "startTime": start_ms,
            "endTime": end_ms - 1,
            "pageSize": 200,
            "indexValue": cursor,
        }
        for attempt in range(4):
            body = get(PUBLIC_BASE + "lead-portfolio/order-history", payload, 4_000_000)
            path = output / f"page_{page:03d}_attempt_{attempt}.json"
            path.write_bytes(body)
            raw = json.loads(body)
            receipts.append(
                {
                    "request": payload,
                    "path": str(path),
                    "sha256": sha(path),
                    "captured_utc": datetime.now(UTC).isoformat(),
                    "code": raw.get("code"),
                }
            )
            if raw.get("code") == "000000" or raw.get("code") != "11012005":
                break
            time.sleep(0.4 * (attempt + 1))
        if raw.get("code") != "000000" or not isinstance(raw.get("data"), dict):
            save(output / "failed_requests.json", receipts)
            raise ValueError("Public cursor page unavailable")
        data = raw["data"]
        if declared is None:
            declared = data["total"]
        if declared != data["total"] or not isinstance(data.get("list"), list):
            raise ValueError("Declared order history changed during pagination")
        rows.extend(data["list"])
        if len(rows) == declared:
            signatures = [json.dumps(row, sort_keys=True) for row in rows]
            if len(set(signatures)) != len(rows):
                raise ValueError("No-ID order duplicates cannot silently be removed")
            if any(not start_ms <= int(r["orderTime"]) < end_ms for r in rows):
                raise ValueError("Order creation time outside requested interval")
            rows.sort(
                key=lambda r: (int(r["orderUpdateTime"]), r["symbol"], r["positionSide"], r["side"])
            )
            save(
                output / "orders.json",
                {
                    "leader_id": LEADER,
                    "start_ms": start_ms,
                    "end_ms_exclusive": end_ms,
                    "rows": rows,
                    "requests": receipts,
                    "unavailable_windows": [],
                    "declared_total": declared,
                    "scope": "complete executed-order summaries as declared; filtering uses order creation time; no unfilled/cancelled intent or individual fills",
                    "orders_have_unique_ids": False,
                    "credentials_used": False,
                },
            )
            return len(rows)
        new_cursor = data.get("indexValue")
        if not data["list"] or not new_cursor or new_cursor == cursor or len(rows) > declared:
            raise ValueError("Stalled or inconsistent order pagination")
        cursor = new_cursor
    raise ValueError("Cursor pagination exceeds capture bound")


def capture(root):
    protocol = json.loads((root / "protocol.json").read_text())
    start_ms = int(datetime.fromisoformat(protocol["start"]).timestamp() * 1000)
    end_ms = int(datetime.fromisoformat(protocol["end"]).timestamp() * 1000)
    count = capture_orders(root / "orders", start_ms, end_ms)
    print({"public_orders": count}, flush=True)
    jobs = []
    for symbol in protocol["symbols"]:
        for kind in ["klines", "markPriceKlines"]:
            jobs.append((symbol, kind, "2026-09", "monthly"))
            for day in range(1, 24):
                jobs.append((symbol, kind, f"2026-10-{day:02d}", "daily"))
    successes, failures = [], []
    raw = root / "market_raw"
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(archive_job, raw, *job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                successes.append(future.result())
            except Exception as exc:
                failures.append({"job": job, "error": type(exc).__name__ + ":" + str(exc)})
            if (len(successes) + len(failures)) % 80 == 0:
                print(
                    {
                        "market_archives": len(successes),
                        "failures": len(failures),
                        "jobs": len(jobs),
                    },
                    flush=True,
                )
    save(
        root / "market_sources.json",
        sorted(successes, key=lambda x: (x["symbol"], x["kind"], x["period"])),
    )
    save(root / "market_failures.json", failures)
    # Public REST funding is a different source, retained independently from OHLC archives.
    funding = root / "funding_raw"
    funding.mkdir()
    receipts = []
    for symbol in protocol["symbols"]:
        url = (
            "https://fapi.binance.com/fapi/v1/fundingRate?"
            f"symbol={symbol}&startTime={start_ms}&endTime={end_ms - 1}&limit=1000"
        )
        body = get(url, limit=2_000_000)
        target = funding / f"{symbol}.json"
        target.write_bytes(body)
        rows = json.loads(body)
        if not isinstance(rows, list) or len(rows) == 1000:
            raise ValueError("Funding capture invalid or needs pagination")
        receipts.append(
            {
                "symbol": symbol,
                "path": str(target),
                "sha256": sha(target),
                "rows": len(rows),
                "url": url,
            }
        )
    save(root / "funding_sources.json", receipts)
    print(
        {
            "orders": count,
            "archives": len(successes),
            "failures": len(failures),
            "funding_symbols": len(receipts),
        },
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    capture(args.root.absolute())
