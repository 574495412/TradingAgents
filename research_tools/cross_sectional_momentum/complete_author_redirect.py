"""Follow one inspected public Dropbox CDN redirect, preserving prior failures."""

import argparse
import time
from pathlib import Path
from urllib.parse import urlsplit

import requests

from research_tools.factor_lab.io import read, save, sha


def complete(output):
    output.mkdir(parents=True, exist_ok=False)
    parent = Path("data/research/cross_sectional_momentum_author_data_20261027_v003")
    original = parent / "workbook_0.receipt.json"
    record = read(original)
    url = record["headers"]["Location"]
    host = urlsplit(url)
    assert host.scheme == "https"
    assert host.hostname == "ucf519be3c4fa5a8ece5747a48f6.dl.dropboxusercontent.com"
    save(
        output / "plan.json",
        {
            "source": str(original),
            "sha256": sha(original),
            "expected_host": host.hostname,
            "maximum_requests": 1,
            "reason": "The already observed302 uses a named Dropbox CDN subdomain. Prior host check stopped before contacting it. Continue exactly this public redirect, without changing any prior artifact.",
            "new_financial_accounts": 0,
        },
    )
    save(
        output / "registration.json",
        {
            "wall_ns": time.time_ns(),
            "source_hash": sha(Path(__file__)),
            "plan_sha256": sha(output / "plan.json"),
            "before_CDN_download": True,
        },
    )
    receipt = {
        "sent_wall_ns": time.time_ns(),
        "status": None,
        "redirect_parent_sha256": sha(original),
    }
    with requests.get(
        url,
        timeout=(10, 15),
        stream=True,
        allow_redirects=False,
        headers={"Accept-Encoding": "identity"},
    ) as response:
        receipt.update(
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
        body = response.raw.read(8388609, decode_content=False)
    assert len(body) <= 8388608
    (output / "CDN.body").write_bytes(body)
    receipt.update(
        received_wall_ns=time.time_ns(), bytes=len(body), sha256=sha(output / "CDN.body")
    )
    save(output / "CDN.receipt.json", receipt)
    valid = receipt["status"] == 200 and body[:4] == b"PK\x03\x04"
    if valid:
        (output / "LTW_3factor.xlsx").write_bytes(body)
    save(
        output / "summary.json",
        {
            "workbook_downloaded": valid,
            "HTTP_status": receipt["status"],
            "requests_this_version": 1,
            "new_financial_accounts": 0,
            "profit_goal_met": False,
        },
    )
    save(
        output / "completion.json",
        {
            "manifest": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
            "new_financial_accounts": 0,
        },
    )
    print(
        {"status": receipt["status"], "bytes": len(body), "workbook_downloaded": valid}, flush=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    complete(parser.parse_args().output)
