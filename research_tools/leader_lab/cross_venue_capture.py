"""Read-only expanded carry data for the previously frozen volume universe."""

from __future__ import annotations

import argparse
import json
import shutil
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from .evidence import save, sha


def capture(root, output, reuse=None):
    output.mkdir(exist_ok=False)
    target = output / "cross_venue_sources"
    target.mkdir()
    old = reuse or root / "cross_venue_sources"
    universe = json.loads((root / "carry_expansion_universe.json").read_text())
    start, end = 1790294400000, 1792800000000
    save(
        output / "capture_protocol.json",
        {
            "registered_utc": datetime.now(UTC).isoformat(),
            "universe": universe,
            "source_sha256": sha(Path(__file__)),
            "start_ms": start,
            "end_ms": end,
            "reuse_only_digest_verified_original_sources": True,
            "reuse_source": str(old),
            "request_weight_budget_per_minute": 800,
            "missing_data_is_not_zero": True,
        },
    )
    (output / "executed_capture_source.py").write_bytes(Path(__file__).read_bytes())
    shutil.copyfile(old / "USDCUSDT_hourly.json", target / "USDCUSDT_hourly.json")
    original = json.loads((old / "capture_reconciliation_v002.json").read_text())
    budget_start, weight = time.monotonic(), 0

    def public(payload):
        nonlocal budget_start, weight
        estimate = 60 if payload["type"] == "fundingHistory" else 40
        elapsed = time.monotonic() - budget_start
        if elapsed >= 60:
            budget_start, weight = time.monotonic(), 0
        elif weight + estimate > 800:
            time.sleep(max(0, 60 - elapsed))
            budget_start, weight = time.monotonic(), 0
        request = urllib.request.Request(
            "https://api.hyperliquid.xyz/info",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "User-Agent": "read-only-quant-research"},
        )
        with urllib.request.urlopen(request, timeout=22) as response:
            data = json.load(response)
        weight += estimate
        return data

    complete, failures = [], []
    for coin in universe["coins"]:
        previous = next((x for x in original["records"] if x["coin"] == coin), None)
        if previous:
            for suffix, field in [("candles", "candles_sha256"), ("funding", "funding_sha256")]:
                p = old / f"{coin}_{suffix}.json"
                if sha(p) != previous[field]:
                    raise ValueError("Original carry source changed")
                shutil.copyfile(p, target / p.name)
        else:
            try:
                candles = public(
                    {
                        "type": "candleSnapshot",
                        "req": {"coin": coin, "interval": "1h", "startTime": start, "endTime": end},
                    }
                )
                save(target / f"{coin}_candles.json", candles)
                rates, cursor = [], start
                for page in range(5):
                    payload = {
                        "type": "fundingHistory",
                        "coin": coin,
                        "startTime": cursor,
                        "endTime": end,
                    }
                    batch = public(payload)
                    save(
                        target / f"{coin}_funding_page_{page}.json",
                        {"request": payload, "response": batch},
                    )
                    if not isinstance(batch, list) or not batch:
                        raise ValueError("Funding response missing")
                    rates.extend(x for x in batch if start <= int(x["time"]) < end)
                    last = max(int(x["time"]) for x in batch)
                    if last < cursor:
                        raise ValueError("Funding cursor did not advance")
                    cursor = last + 1
                    if last >= end - 3600000:
                        break
                save(target / f"{coin}_funding.json", rates)
            except Exception as e:
                failures.append({"coin": coin, "error": type(e).__name__ + ":" + str(e)})
                continue
        bars = json.loads((target / f"{coin}_candles.json").read_text())
        rates = json.loads((target / f"{coin}_funding.json").read_text())
        if (
            [int(x["t"]) for x in bars] != list(range(start, end + 1, 3600000))
            or [int(x["time"]) // 3600000 * 3600000 for x in rates]
            != list(range(start, end, 3600000))
            or max(int(x["time"]) % 3600000 for x in rates) > 1000
        ):
            failures.append({"coin": coin, "error": "Incomplete hourly price or funding coverage"})
            continue
        complete.append(
            {
                "coin": coin,
                "hourly_candles_including_endpoint": len(bars),
                "hourly_funding": len(rates),
                "original_ms_preserved": True,
                "candles_sha256": sha(target / f"{coin}_candles.json"),
                "funding_sha256": sha(target / f"{coin}_funding.json"),
            }
        )
        save(
            output / f"progress_{len(complete) + len(failures):02d}.json",
            {"complete": len(complete), "failed": len(failures)},
        )
    save(
        target / "capture_reconciliation_v002.json",
        {
            "source": "https://api.hyperliquid.xyz/info",
            "start_ms": start,
            "end_ms_exclusive": end,
            "records": complete,
            "captured_utc": datetime.now(UTC).isoformat(),
            "failed_symbols_excluded": failures,
            "settlement_timestamp_semantics": "hour floor used only for coverage; original timestamp retained",
        },
    )
    print({"complete_symbols": len(complete), "failed": failures}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse", type=Path)
    args = parser.parse_args()
    capture(
        args.root.absolute(), args.output.absolute(), args.reuse.absolute() if args.reuse else None
    )
