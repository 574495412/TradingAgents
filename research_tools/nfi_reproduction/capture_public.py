"""Bounded anonymous source capture for exact NFI reproduction, no trading."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import requests


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def capture(root, name, url):
    if root.exists():
        plan = json.loads((root / "plan.json").read_text())
    else:
        root.mkdir(parents=True)
        plan = {
            "repository": "iterativv/NostalgiaForInfinity",
            "purpose": "Select one published profitable report with exact source, config and inputs before any reproduction or local adaptation.",
            "selection_rule": "First complete source/report/data/config contract found; never rank by maximum return.",
            "maximum_source_requests": 16,
            "maximum_bytes_each": 10000000,
            "allow_trading": False,
            "allow_factor_expansion": False,
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "registered_wall_ns": time.time_ns(),
        }
        write(root / "plan.json", plan)
    if plan["code_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("Registered source collector changed")
    if len(list(root.glob("*.receipt.json"))) >= plan["maximum_source_requests"]:
        raise ValueError("Bounded source request budget exhausted")
    receipt_path = root / (name + ".receipt.json")
    if receipt_path.exists() or (root / (name + ".body")).exists():
        raise FileExistsError("Preserve existing capture")
    receipt = {"url": url, "sent_wall_ns": time.time_ns()}
    try:
        with requests.get(
            url,
            timeout=(10, 25),
            stream=True,
            allow_redirects=False,
            headers={
                "Accept": "application/json,text/plain,text/html",
                "Accept-Encoding": "identity",
            },
        ) as response:
            body = response.raw.read(plan["maximum_bytes_each"] + 1, decode_content=False)
            receipt.update(
                status=response.status_code,
                headers={
                    key: response.headers[key]
                    for key in [
                        "Date",
                        "Content-Type",
                        "ETag",
                        "Link",
                        "Location",
                        "X-RateLimit-Remaining",
                    ]
                    if key in response.headers
                },
                bytes=len(body),
                oversize=len(body) > plan["maximum_bytes_each"],
            )
        (root / (name + ".body")).write_bytes(body)
        receipt["sha256"] = hashlib.sha256(body).hexdigest()
    except requests.RequestException as error:
        receipt.update(error_type=type(error).__name__, error=str(error)[:200])
    receipt["received_wall_ns"] = time.time_ns()
    write(receipt_path, receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k not in ["sha256"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--url", required=True)
    args = parser.parse_args()
    capture(args.root, args.name, args.url)
