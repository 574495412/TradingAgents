"""Capture a bounded, anonymous second-recorder witness of frozen entry windows."""

import argparse
import gzip
import io
import json
from datetime import UTC, datetime
from pathlib import Path

import requests

from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest


def collect(output):
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).with_name("execution_witness_plan.json")
    plan = read(source)
    inputs = Path(plan["prepared_inputs"])
    original = Path(plan["original_accounts"])
    verify_manifest(inputs)
    verify_manifest(original)
    save(
        output / "plan.json",
        {
            **plan,
            "source_hashes": {
                str(p): sha(p)
                for p in [
                    source,
                    Path(__file__),
                    inputs / "completion.json",
                    original / "completion.json",
                ]
            },
        },
    )
    save(
        output / "registration.json",
        {
            "registered_at_utc": datetime.now(UTC).isoformat(),
            "plan_sha256": sha(output / "plan.json"),
            "before_new_native_trade_and_quote_access": True,
            "new_financial_accounts": 0,
            "known_history_only": True,
        },
    )
    requests_count, total_bytes, receipts = 0, 0, []
    session = requests.Session()
    session.headers.update({"Accept-Encoding": "gzip", "User-Agent": "QuantResearchEvidence/1.0"})

    def get(url, stem, params=None):
        nonlocal requests_count, total_bytes
        assert requests_count < plan["budget"]["read_only_HTTP_requests_max"]
        requests_count += 1
        receipt = {
            "url": url,
            "stem": stem,
            "requested_at_utc": datetime.now(UTC).isoformat(),
            "status": None,
        }
        raw = output / (stem + ".raw")
        try:
            response = session.get(
                url, params=params, timeout=(10, 25), stream=True, allow_redirects=False
            )
            receipt.update(
                url=response.url,
                status=response.status_code,
                headers={
                    k: response.headers[k]
                    for k in [
                        "Content-Type",
                        "Content-Encoding",
                        "x-slice-size",
                        "Last-Modified",
                        "ETag",
                    ]
                    if k in response.headers
                },
            )
            chunks, size = [], 0
            while chunk := response.raw.read(65536, decode_content=False):
                size += len(chunk)
                total_bytes += len(chunk)
                if (
                    size > plan["budget"]["compressed_bytes_per_response_max"]
                    or total_bytes > plan["budget"]["total_compressed_bytes_max"]
                ):
                    raise ValueError("Registered byte bound exceeded")
                chunks.append(chunk)
            response.close()
            body = b"".join(chunks)
            raw.write_bytes(body)
            receipt.update(compressed_bytes=len(body), raw_sha256=sha(raw))
            if (
                response.headers.get("Content-Encoding", "").lower() == "gzip"
                or body[:2] == b"\x1f\x8b"
            ):
                with gzip.GzipFile(fileobj=io.BytesIO(body)) as zipped:
                    body = zipped.read(plan["budget"]["decompressed_bytes_per_response_max"] + 1)
            assert len(body) <= plan["budget"]["decompressed_bytes_per_response_max"]
            decoded = output / (stem + ".decoded")
            decoded.write_bytes(body)
            receipt.update(decoded_bytes=len(body), decoded_sha256=sha(decoded))
        except Exception as e:
            receipt.update(error_type=type(e).__name__, error=str(e))
        receipt["finished_at_utc"] = datetime.now(UTC).isoformat()
        save(output / (stem + ".receipt.json"), receipt)
        receipts.append(receipt)
        print(
            {
                "stem": stem,
                "status": receipt["status"],
                "bytes": receipt.get("decoded_bytes"),
                "error": receipt.get("error"),
            },
            flush=True,
        )

    for name, url in plan["documents"].items():
        get(url, "doc_" + name)
    for symbol in plan["metadata_symbols"]:
        get("https://api.tardis.dev/v1/instruments/binance-futures/" + symbol, "metadata_" + symbol)
    for market, symbols in plan["markets"].items():
        get(
            "https://api.tardis.dev/v1/data-feeds/" + market,
            "feed_" + market,
            {
                **plan["capture_window"],
                "offset": 0,
                "filters": json.dumps(
                    [{"channel": c, "symbols": symbols} for c in plan["channels"]],
                    separators=(",", ":"),
                ),
            },
        )
    session.close()
    save(
        output / "summary.json",
        {
            "receipts": receipts,
            "requests": requests_count,
            "compressed_bytes": total_bytes,
            "new_financial_accounts": 0,
            "profit_goal_met": False,
        },
    )
    save(
        output / "completion.json",
        {
            "capture_finished": True,
            "manifest": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
            "profit_goal_met": False,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    collect(p.parse_args().output)
