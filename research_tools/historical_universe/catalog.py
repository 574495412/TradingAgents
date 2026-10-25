"""Enumerate official historical daily archives without a surviving-symbol filter."""

import argparse
import re
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import requests

from research_tools.factor_lab.io import read, save, sha

NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}


class Listing:
    def __init__(self, root, plan):
        self.root, self.plan = root, plan
        self.requests = 0
        self.lock = threading.Lock()
        self.local = threading.local()

    def page(self, prefix, marker):
        if not prefix.startswith(self.plan["prefix"]):
            raise ValueError("Out-of-scope archive prefix")
        if not hasattr(self.local, "session"):
            self.local.session = requests.Session()
        params = {"prefix": prefix, "delimiter": "/", "max-keys": 1000}
        if marker:
            params["marker"] = marker
        for attempt in range(self.plan["budget"]["retry_attempts"]):
            with self.lock:
                self.requests += 1
                sequence = self.requests
                if sequence > self.plan["budget"]["maximum_catalog_requests"]:
                    raise ValueError("Catalog request budget exhausted")
            try:
                response = self.local.session.get(
                    self.plan["listing_endpoint"], params=params, timeout=30
                )
                response.raise_for_status()
                if len(response.content) > 2_000_000:
                    raise ValueError("Oversize listing response")
                path = self.root / "raw" / f"page{sequence:05d}.xml"
                path.write_bytes(response.content)
                return ET.fromstring(response.content), {
                    "file": str(path.relative_to(self.root)),
                    "sha256": sha(path),
                    "request": params,
                    "status": response.status_code,
                }
            except (requests.RequestException, ET.ParseError):
                if attempt + 1 == self.plan["budget"]["retry_attempts"]:
                    raise
                time.sleep(min(2**attempt, 4))
        raise AssertionError("Unreachable retry state")

    def all(self, prefix):
        marker, seen, pages, contents, prefixes = "", set(), [], [], []
        while True:
            tree, receipt = self.page(prefix, marker)
            if tree.findtext("s:Prefix", namespaces=NS) != prefix:
                raise ValueError("Listing prefix differs")
            pages.append(receipt)
            prefixes += [v.text for v in tree.findall("s:CommonPrefixes/s:Prefix", NS)]
            contents += [
                {
                    "key": v.findtext("s:Key", namespaces=NS),
                    "size": int(v.findtext("s:Size", namespaces=NS)),
                    "etag": v.findtext("s:ETag", namespaces=NS),
                    "last_modified": v.findtext("s:LastModified", namespaces=NS),
                }
                for v in tree.findall("s:Contents", NS)
            ]
            if tree.findtext("s:IsTruncated", namespaces=NS) == "false":
                break
            marker = tree.findtext("s:NextMarker", namespaces=NS)
            if not marker or marker in seen:
                raise ValueError("Truncated listing without advancing NextMarker")
            seen.add(marker)
        return contents, prefixes, pages


def catalog(output):
    output.mkdir(parents=True, exist_ok=False)
    (output / "raw").mkdir()
    plan_path = Path(__file__).with_name("plan.json")
    plan = read(plan_path)
    save(
        output / "plan.json",
        {
            **plan,
            "source_sha256": sha(Path(__file__)),
            "plan_source_sha256": sha(plan_path),
            "registered_before_catalog": datetime.now(UTC).isoformat(),
        },
    )
    listing = Listing(output, plan)
    _, prefixes, pages = listing.all(plan["prefix"])
    if len(prefixes) > plan["budget"]["maximum_root_prefixes"] or len(prefixes) != len(
        set(prefixes)
    ):
        raise ValueError("Root universe oversized or duplicated")
    symbols = sorted(
        p[len(plan["prefix"]) : -1]
        for p in prefixes
        if re.fullmatch(r"[A-Z0-9]+USDT", p[len(plan["prefix"]) : -1])
    )
    if len(symbols) > plan["budget"]["maximum_usdt_symbols"]:
        raise ValueError("USDT universe exceeds registration")
    save(
        output / "root_universe.json",
        {"all_prefixes": prefixes, "usdt_symbols": symbols, "pages": pages},
    )
    records, reports = [], []

    def one(symbol):
        contents, children, receipt = listing.all(f"{plan['prefix']}{symbol}/1d/")
        if children:
            raise ValueError("Unexpected nested interval archive")
        by_key = {r["key"]: r for r in contents}
        if len(by_key) != len(contents):
            raise ValueError("Duplicate archive key across pages")
        found = []
        for key, row in by_key.items():
            name = key.rsplit("/", 1)[-1]
            m = re.fullmatch(re.escape(symbol) + r"-1d-(\d{4}-\d{2})\.zip", name)
            if m is None or not plan["start"][:7] <= m[1] < plan["end_exclusive"][:7]:
                continue
            if key + ".CHECKSUM" not in by_key:
                raise ValueError("Official checksum missing for " + key)
            found.append(
                {"symbol": symbol, "month": m[1], **row, "checksum_key": key + ".CHECKSUM"}
            )
        return symbol, sorted(found, key=lambda x: x["month"]), receipt

    with ThreadPoolExecutor(max_workers=plan["budget"]["workers"]) as pool:
        for i, (symbol, found, receipt) in enumerate(pool.map(one, symbols)):
            records.extend(found)
            pages.extend(receipt)
            reports.append(
                {
                    "symbol": symbol,
                    "archives": len(found),
                    "first_month": found[0]["month"] if found else None,
                    "last_month": found[-1]["month"] if found else None,
                }
            )
            if (i + 1) % 50 == 0:
                print(
                    {
                        "catalogued_symbols": i + 1,
                        "of": len(symbols),
                        "archives": len(records),
                        "requests": listing.requests,
                    },
                    flush=True,
                )
    if (
        not records
        or len(records) > plan["budget"]["maximum_archive_files"]
        or sum(r["size"] for r in records) > plan["budget"]["maximum_download_bytes"]
    ):
        raise ValueError("Archive data budget violated")
    metadata_path = Path(plan["source_directory"]) / "exchange_info.json"
    current = {r["symbol"]: r for r in read(metadata_path)["symbols"]}
    for r in reports:
        r["capture_time_current_metadata"] = current.get(r["symbol"])
        r["absent_from_current_metadata"] = r["symbol"] not in current
        r["historical_tradability_certified"] = False
    save(output / "archives.json", records)
    save(output / "symbols.json", reports)
    summary = {
        "schema": "historical-perpetual-catalog.v1",
        "root_prefixes": len(prefixes),
        "usdt_symbols": len(symbols),
        "symbols_with_in_scope_archives": sum(r["archives"] > 0 for r in reports),
        "archive_files": len(records),
        "archive_bytes": sum(r["size"] for r in records),
        "requests": listing.requests,
        "pages": pages,
        "current_metadata_sha256": sha(metadata_path),
        "archive_observed_not_complete_exchange_history": True,
        "financial_accounts": 0,
        "profit_goal_met": False,
    }
    save(output / "summary.json", summary)
    save(
        output / "completion.json",
        {
            "manifest": {
                str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()
            },
            "catalog_complete": True,
            "financial_accounts": 0,
        },
    )
    print({k: v for k, v in summary.items() if k != "pages"}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    catalog(a.output)
