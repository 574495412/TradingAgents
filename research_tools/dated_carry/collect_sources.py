"""Capture bounded original method/specification and archive-prefix metadata only."""

import argparse
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode, urljoin

import requests

from research_tools.factor_lab.io import read, save, sha


def collect(output):
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).with_name("source_plan.json")
    p = read(source)
    save(output / "plan.json", p)
    save(
        output / "registration.json",
        {
            "registered_at_utc": datetime.now(UTC).isoformat(),
            "before_source_capture_and_new_payoff_evaluation": True,
            "source_hashes": {str(q): sha(q) for q in [source, Path(__file__)]},
            "financial_accounts": 0,
        },
    )
    tasks = [("paper", p["paper"])]
    tasks += [(f"doc_{i}", u) for i, u in enumerate(p["official_documents"])]
    tasks += [
        (
            f"archive_{i}",
            "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision?"
            + urlencode({"prefix": prefix, "delimiter": "/", "max-keys": 1000}),
        )
        for i, prefix in enumerate(p["archive_prefixes"])
    ]
    session = requests.Session()
    session.headers["User-Agent"] = "QuantResearchSourceAudit/1.0 (public read-only)"
    records, responses, request_count, byte_count = [], {}, 0, 0
    for identity, original_url in tasks:
        url = original_url
        for hop in range(4):
            assert request_count < p["budget"]["maximum_read_only_http_requests"]
            request_count += 1
            begin = time.monotonic()
            receipt = {
                "id": identity,
                "redirect_hop": hop,
                "requested_url": url,
                "requested_at_utc": datetime.now(UTC).isoformat(),
            }
            try:
                response = session.get(url, timeout=(10, 25), allow_redirects=False)
                body = response.content
                byte_count += len(body)
                filename = f"{identity}_{hop}.raw"
                (output / filename).write_bytes(body)
                receipt.update(
                    status=response.status_code,
                    body_file=filename,
                    bytes=len(body),
                    sha256=sha(output / filename),
                    headers={
                        k: v
                        for k, v in response.headers.items()
                        if k.lower()
                        in ["content-type", "last-modified", "etag", "date", "content-encoding"]
                    },
                )
                assert byte_count <= p["budget"]["maximum_actual_data_bytes"]
            except requests.RequestException as e:
                receipt.update(status=None, error=str(e), error_type=type(e).__name__)
                response = None
            receipt["elapsed_seconds"] = time.monotonic() - begin
            save(output / f"{identity}_{hop}.receipt.json", receipt)
            records.append(receipt)
            if (
                response is not None
                and response.status_code in [301, 302, 303, 307, 308]
                and response.headers.get("Location")
            ):
                url = urljoin(url, response.headers["Location"])
                continue
            responses[identity] = receipt
            print(
                {"id": identity, "status": receipt["status"], "bytes": receipt.get("bytes", 0)},
                flush=True,
            )
            break
    session.close()
    inventories = []
    for i, prefix in enumerate(p["archive_prefixes"]):
        receipt = responses[f"archive_{i}"]
        item = {
            "prefix": prefix,
            "receipt": receipt["id"] + "_" + str(receipt["redirect_hop"]),
            "status": receipt["status"],
            "contracts": [],
            "complete_prefix_listing": False,
        }
        if receipt["status"] == 200:
            node = ET.fromstring((output / receipt["body_file"]).read_bytes())
            ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
            item["contracts"] = [n.text for n in node.findall("s:CommonPrefixes/s:Prefix", ns)]
            item["complete_prefix_listing"] = (
                node.findtext("s:IsTruncated", namespaces=ns) == "false"
            )
        inventories.append(item)
    save(output / "archive_inventories.json", inventories)
    save(
        output / "summary.json",
        {
            "schema": "dated-carry-source-capture.v1",
            "http_requests": request_count,
            "response_bytes": byte_count,
            "final_source_responses": responses,
            "archive_inventories": inventories,
            "new_market_archives": 0,
            "new_financial_accounts": 0,
            "profit_goal_met": False,
            "source_authenticity_or_historical_state_certified": False,
        },
    )
    manifest = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
    save(
        output / "completion.json",
        {"capture_finished": True, "manifest": manifest, "profit_goal_met": False},
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    collect(p.parse_args().output)
