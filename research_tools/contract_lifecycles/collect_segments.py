"""Targeted detail capture plus reported-trade archive crosschecks, no account replay."""

import argparse
import io
import json
import xml.etree.ElementTree as ET
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

from research_tools.contract_lifecycles.collect import request, text_nodes
from research_tools.factor_lab.io import read, save, sha

NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}


def collect(root):
    plan_path = Path(__file__).with_name("segment_plan.json")
    plan = read(plan_path)
    registered = read(root / "plan.json")
    if registered["registered_plan_sha256"] != sha(plan_path):
        raise ValueError("Targeted registration changed")
    save(
        root / "collection_registration.json",
        {
            "collector_sha256": sha(Path(__file__)),
            "plan_sha256": sha(root / "plan.json"),
            "maximum_file_bytes": 25_000_000,
            "before_archive_probes": True,
            "five_targeted_details_already_captured": True,
        },
    )
    (root / "collector_source.py").write_bytes(Path(__file__).read_bytes())
    details = []
    for r in plan["target_details"]:
        code = r["code"]
        data = request(
            root,
            code,
            "https://www.binance.com/bapi/composite/v1/public/cms/article/detail/query",
            {"articleCode": code},
        )
        text = text_nodes(json.loads(data["body"]))
        if (root / (code + ".txt")).read_text() != text:
            raise ValueError("Prior current-body text changed")
        details.append(
            {
                "code": code,
                "title": data["title"],
                "publishDate": data["publishDate"],
                "raw_sha256": sha(root / (code + ".raw")),
                "text_sha256": sha(root / (code + ".txt")),
            }
        )
    archives = []
    (root / "trades").mkdir()
    count = 0
    for i, prefix in enumerate(plan["archive_prefix_probes"]):
        response = requests.get(
            "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision",
            params={"prefix": prefix, "max-keys": 1000},
            timeout=25,
        )
        count += 1
        path = root / "trades" / f"listing{i}.xml"
        path.write_bytes(response.content)
        save(
            root / "trades" / f"listing{i}.receipt.json",
            {
                "url": response.url,
                "status": response.status_code,
                "sha256": sha(path),
                "captured_at_utc": datetime.now(UTC).isoformat(),
            },
        )
        response.raise_for_status()
        tree = ET.fromstring(response.content)
        if (
            tree.findtext("s:Prefix", namespaces=NS) != prefix
            or tree.findtext("s:IsTruncated", namespaces=NS) != "false"
        ):
            raise ValueError("Exact trade prefix not fully listed")
        keys = {
            v.findtext("s:Key", namespaces=NS): int(v.findtext("s:Size", namespaces=NS))
            for v in tree.findall("s:Contents", NS)
        }
        record = {
            "prefix": prefix,
            "listed_keys": keys,
            "archive_exists": prefix + ".zip" in keys,
            "listing_sha256": sha(path),
        }
        if record["archive_exists"]:
            required = [prefix + ".zip.CHECKSUM", prefix + ".zip"]
            for key in required:
                if key not in keys or keys[key] > 25_000_000:
                    raise ValueError("Trade archive absent checksum or beyond fixed byte limit")
                count += 1
                if count > plan["maximum_archive_requests"]:
                    raise ValueError("Archive request budget exceeded")
                r = requests.get("https://data.binance.vision/" + key, timeout=25)
                target = root / "trades" / Path(key).name
                target.write_bytes(r.content)
                save(
                    root / "trades" / (Path(key).name + ".receipt.json"),
                    {"url": r.url, "status": r.status_code, "sha256": sha(target)},
                )
                r.raise_for_status()
                if len(r.content) != keys[key] or len(r.content) > 25_000_000:
                    raise ValueError("Trade archive size differs")
            archive = root / "trades" / Path(required[1]).name
            checksum = root / "trades" / Path(required[0]).name
            if sha(archive) != checksum.read_text().split()[0].lower():
                raise ValueError("Official trade checksum differs")
            with zipfile.ZipFile(archive) as z:
                names = z.namelist()
                if names != [Path(prefix).name + ".csv"] or z.testzip() is not None:
                    raise ValueError("Trade ZIP integrity differs")
                frame = pd.read_csv(io.BytesIO(z.read(names[0])), header=None, dtype=str)
            if not str(frame.iloc[0, 0]).isdigit():
                frame = frame.iloc[1:]
            if frame.shape[1] != 7:
                raise ValueError("Unexpected reported-aggTrade schema")
            numeric = frame.iloc[:, :6].astype(float)
            ids = frame[0].astype("int64")
            times = pd.to_datetime(frame[5].astype("int64"), unit="ms", utc=True)
            day = pd.Timestamp(prefix[-10:], tz="UTC")
            if (
                ids.duplicated().any()
                or not ids.is_monotonic_increasing
                or not times.is_monotonic_increasing
                or (numeric.iloc[:, 1:3] <= 0).any().any()
                or not ((times >= day) & (times < day + pd.Timedelta(days=1))).all()
                or (numeric[3] > numeric[4]).any()
            ):
                raise ValueError("Reported trade ID/time/value validation failed")
            record.update(
                rows=len(frame),
                first_reported_trade_at=times.iloc[0].isoformat(),
                last_reported_trade_at=times.iloc[-1].isoformat(),
                first_aggregate_id=int(ids.iloc[0]),
                last_aggregate_id=int(ids.iloc[-1]),
                quote_turnover=float((numeric[1] * numeric[2]).sum()),
                archive_sha256=sha(archive),
                checksum_passed=True,
            )
        archives.append(record)
        print(record, flush=True)
    save(
        root / "summary.json",
        {
            "schema": "contract-segment-sources.v1",
            "details": details,
            "archive_probes": archives,
            "detail_requests": len(details),
            "archive_requests": count,
            "total_new_requests": len(details) + count,
            "original_historical_versions_certified": False,
            "archive_existence_is_relisting_proof": False,
            "new_financial_accounts": 0,
            "profit_goal_met": False,
        },
    )
    save(
        root / "completion.json",
        {
            "manifest": {str(p.relative_to(root)): sha(p) for p in root.rglob("*") if p.is_file()},
            "collection_complete": True,
            "new_financial_accounts": 0,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    collect(p.parse_args().output)
