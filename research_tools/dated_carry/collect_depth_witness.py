"""Capture the smallest full-depth followup to the frozen BBO contradictions."""

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
    source = Path(__file__).with_name("depth_witness_plan.json")
    plan = read(source)
    for key in ["parent", "parent_audit"]:
        verify_manifest(Path(plan[key]))
    save(
        output / "plan.json",
        {
            **plan,
            "source_hashes": {
                str(p): sha(p)
                for p in [
                    source,
                    Path(__file__),
                    Path(plan["parent"]) / "completion.json",
                    Path(plan["parent_audit"]) / "completion.json",
                ]
            },
        },
    )
    save(
        output / "registration.json",
        {
            "registered_at_utc": datetime.now(UTC).isoformat(),
            "plan_sha256": sha(output / "plan.json"),
            "before_native_depth_access": True,
            "new_financial_accounts": 0,
        },
    )
    jobs = [
        ("protocol", plan["document"], None),
        (
            "depth",
            "https://api.tardis.dev/v1/data-feeds/" + plan["market"],
            {
                **plan["capture_window"],
                "offset": 0,
                "filters": json.dumps(
                    [{"channel": c, "symbols": plan["symbols"]} for c in plan["channels"]],
                    separators=(",", ":"),
                ),
            },
        ),
    ]
    assert len(jobs) <= plan["budget"]["read_only_HTTP_requests_max"]
    receipts = []
    with requests.Session() as session:
        for stem, url, params in jobs:
            record = {"url": url, "status": None, "requested_at_utc": datetime.now(UTC).isoformat()}
            try:
                with session.get(
                    url,
                    params=params,
                    headers={"Accept-Encoding": "gzip"},
                    timeout=(10, 25),
                    stream=True,
                    allow_redirects=False,
                ) as response:
                    record.update(
                        status=response.status_code,
                        url=response.url,
                        encoding=response.headers.get("Content-Encoding"),
                    )
                    chunks, size = [], 0
                    while piece := response.raw.read(65536, decode_content=False):
                        size += len(piece)
                        assert size <= plan["budget"]["compressed_bytes_per_response_max"]
                        chunks.append(piece)
                    body = b"".join(chunks)
                (output / (stem + ".raw")).write_bytes(body)
                record.update(raw_bytes=len(body), raw_sha256=sha(output / (stem + ".raw")))
                if record["encoding"] == "gzip" or body[:2] == b"\x1f\x8b":
                    with gzip.GzipFile(fileobj=io.BytesIO(body)) as zipped:
                        body = zipped.read(
                            plan["budget"]["decompressed_bytes_per_response_max"] + 1
                        )
                assert len(body) <= plan["budget"]["decompressed_bytes_per_response_max"]
                (output / (stem + ".decoded")).write_bytes(body)
                record.update(
                    decoded_bytes=len(body), decoded_sha256=sha(output / (stem + ".decoded"))
                )
            except Exception as e:
                record.update(error_type=type(e).__name__, error=str(e))
            record["finished_at_utc"] = datetime.now(UTC).isoformat()
            save(output / (stem + ".receipt.json"), record)
            receipts.append(record)
            print(
                {
                    "stem": stem,
                    "status": record["status"],
                    "bytes": record.get("decoded_bytes"),
                    "error": record.get("error"),
                },
                flush=True,
            )
    save(
        output / "summary.json",
        {"receipts": receipts, "new_financial_accounts": 0, "profit_goal_met": False},
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
