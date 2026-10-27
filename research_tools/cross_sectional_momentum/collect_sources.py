"""Bounded original-method and public historical-input availability capture."""

import argparse
import gzip
import io
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import requests

from research_tools.factor_lab.io import read, save, sha


def collect(output):
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).with_name("source_plan.json")
    plan = read(source)
    save(output / "plan.json", plan)
    save(
        output / "registration.json",
        {
            "registered_at_utc": datetime.now(UTC).isoformat(),
            "wall_ns": time.time_ns(),
            "source_hashes": {str(p): sha(p) for p in [source, Path(__file__)]},
            "before_new_public_historical_cap_query": True,
            "published_methods_already_read_in_browser": True,
            "economic_returns_not_evaluated": True,
        },
    )
    assert len(plan["sources"]) <= plan["budget"]["maximum_requests"]
    receipts = []
    for item in plan["sources"]:
        r = {
            "id": item["id"],
            "url": item["url"],
            "sent_wall_ns": time.time_ns(),
            "sent_monotonic_ns": time.monotonic_ns(),
            "status": None,
        }
        try:
            with requests.get(
                item["url"],
                params=item.get("params"),
                timeout=(10, 15),
                stream=True,
                allow_redirects=False,
                headers={"Accept-Encoding": "gzip"},
            ) as response:
                r.update(
                    url=response.url,
                    status=response.status_code,
                    headers={
                        k: response.headers[k]
                        for k in [
                            "Date",
                            "Age",
                            "Content-Type",
                            "Content-Encoding",
                            "ETag",
                            "Last-Modified",
                        ]
                        if k in response.headers
                    },
                )
                body = response.raw.read(
                    plan["budget"]["maximum_raw_response_bytes_each"] + 1, decode_content=False
                )
            if len(body) > plan["budget"]["maximum_raw_response_bytes_each"]:
                raise ValueError("Response byte limit reached")
            path = output / (item["id"] + ".wire")
            path.write_bytes(body)
            r.update(raw_bytes=len(body), raw_sha256=sha(path))
            if (
                r["headers"].get("Content-Encoding", "").lower() == "gzip"
                or body[:2] == b"\x1f\x8b"
            ):
                with gzip.GzipFile(fileobj=io.BytesIO(body)) as zipped:
                    body = zipped.read(plan["budget"]["maximum_decoded_bytes_each"] + 1)
            if len(body) > plan["budget"]["maximum_decoded_bytes_each"]:
                raise ValueError("Decoded byte limit reached")
            decoded = output / (item["id"] + "." + item["extension"])
            decoded.write_bytes(body)
            r["decoded_sha256"] = sha(decoded)
            if item["extension"] == "json":
                try:
                    data = json.loads(body)
                    r["provider_error_code"] = (
                        data.get("error", {}).get("status", {}).get("error_code")
                        if isinstance(data.get("error"), dict)
                        else data.get("status", {}).get("error_code")
                    )
                    r["valid_price_market_cap_volume_record"] = bool(
                        r["status"] == 200
                        and data.get("id") == "bitcoin"
                        and all(
                            isinstance(
                                data.get("market_data", {}).get(k, {}).get("usd"), (int, float)
                            )
                            and data["market_data"][k]["usd"] > 0
                            for k in ["current_price", "market_cap", "total_volume"]
                        )
                    )
                except (ValueError, AttributeError, TypeError):
                    r["valid_price_market_cap_volume_record"] = False
        except Exception as error:
            r.update(error_type=type(error).__name__, error=str(error)[:200])
        r.update(received_wall_ns=time.time_ns(), received_monotonic_ns=time.monotonic_ns())
        save(output / (item["id"] + ".receipt.json"), r)
        receipts.append(r)
        print(
            {
                k: r.get(k)
                for k in [
                    "id",
                    "status",
                    "raw_bytes",
                    "provider_error_code",
                    "valid_price_market_cap_volume_record",
                    "error_type",
                ]
            },
            flush=True,
        )
    save(
        output / "summary.json",
        {
            "requests": len(receipts),
            "receipts": receipts,
            "new_financial_accounts": 0,
            "profit_goal_met": False,
        },
    )
    save(
        output / "completion.json",
        {
            "manifest": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
            "financial_result_generated": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    collect(parser.parse_args().output)
