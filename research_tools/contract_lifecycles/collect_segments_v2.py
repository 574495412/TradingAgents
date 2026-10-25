"""Targeted detail capture plus reported-trade archive crosschecks, no account replay."""

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from research_tools.contract_lifecycles.collect import request, text_nodes
from research_tools.factor_lab.io import read, save, sha

NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}


def collect(root, parent):
    root.mkdir(parents=True, exist_ok=False)
    recovery_path = Path(__file__).with_name("segment_collection_recovery.json")
    recovery = read(recovery_path)
    for p in parent.iterdir():
        if p.is_file() and (
            p.suffix in [".raw", ".txt"]
            or p.name.endswith(".receipt.json")
            or p.name == "plan.json"
        ):
            shutil.copyfile(p, root / p.name)
    plan_path = Path(__file__).with_name("segment_plan.json")
    plan = read(plan_path)
    registered = read(root / "plan.json")
    if registered["registered_plan_sha256"] != sha(plan_path):
        raise ValueError("Targeted registration changed")
    save(
        root / "recovery_plan.json",
        {
            **recovery,
            "source_plan_sha256": sha(plan_path),
            "recovery_source_sha256": sha(recovery_path),
            "adopted_files": {p.name: sha(p) for p in root.iterdir() if p.is_file()},
            "parent_first_listing_sha256": sha(parent / "trades/listing0.xml"),
        },
    )
    save(
        root / "collection_registration.json",
        {
            "collector_sha256": sha(Path(__file__)),
            "plan_sha256": sha(root / "plan.json"),
            "maximum_file_bytes": recovery["maximum_file_bytes"],
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
    downloaded_bytes = 0
    for i, prefix in enumerate(plan["archive_prefix_probes"]):
        path = root / "trades" / f"listing{i}.xml"
        if i == 0:
            shutil.copyfile(parent / "trades/listing0.xml", path)
            shutil.copyfile(
                parent / "trades/listing0.receipt.json", root / "trades/listing0.receipt.json"
            )
        else:
            response = requests.get(
                "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision",
                params={"prefix": prefix, "max-keys": 1000},
                timeout=25,
            )
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
        count += 1
        tree = ET.fromstring(path.read_bytes())
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
                if key not in keys or keys[key] > recovery["maximum_file_bytes"]:
                    raise ValueError("Trade archive absent checksum or beyond fixed byte limit")
                downloaded_bytes += keys[key]
                if downloaded_bytes > recovery["maximum_total_archive_bytes"]:
                    raise ValueError("Total archive byte budget exceeded")
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
                if len(r.content) != keys[key] or len(r.content) > recovery["maximum_file_bytes"]:
                    raise ValueError("Trade archive size differs")
            archive = root / "trades" / Path(required[1]).name
            checksum = root / "trades" / Path(required[0]).name
            if sha(archive) != checksum.read_text().split()[0].lower():
                raise ValueError("Official trade checksum differs")
            with zipfile.ZipFile(archive) as z:
                names = z.namelist()
                if names != [Path(prefix).name + ".csv"] or z.testzip() is not None:
                    raise ValueError("Trade ZIP integrity differs")
                with z.open(names[0]) as stream:
                    first_line = stream.readline().decode("utf-8")
                header = not first_line.split(",")[0].strip().isdigit()
                rows, quote, first_time, last_time, first_id, last_id = (
                    0,
                    0.0,
                    None,
                    None,
                    None,
                    None,
                )
                with z.open(names[0]) as stream:
                    chunks = pd.read_csv(
                        stream,
                        header=None,
                        skiprows=int(header),
                        names=list(range(7)),
                        dtype={
                            0: "int64",
                            1: "float64",
                            2: "float64",
                            3: "int64",
                            4: "int64",
                            5: "int64",
                            6: str,
                        },
                        chunksize=100000,
                    )
                    for frame in chunks:
                        ids = frame[0]
                        times = pd.to_datetime(frame[5], unit="ms", utc=True)
                        day = pd.Timestamp(prefix[-10:], tz="UTC")
                        if (
                            not len(frame)
                            or not np.isfinite(frame.iloc[:, :6].to_numpy()).all()
                            or ids.duplicated().any()
                            or not ids.is_monotonic_increasing
                            or (last_id is not None and int(ids.iloc[0]) <= last_id)
                            or not times.is_monotonic_increasing
                            or (last_time is not None and times.iloc[0] < last_time)
                            or (frame[[1, 2]] <= 0).any().any()
                            or (frame[3] > frame[4]).any()
                            or not ((times >= day) & (times < day + pd.Timedelta(days=1))).all()
                        ):
                            raise ValueError("Reported trade ID/time/value validation failed")
                        if first_time is None:
                            first_time, first_id = times.iloc[0], int(ids.iloc[0])
                        last_time, last_id = times.iloc[-1], int(ids.iloc[-1])
                        rows += len(frame)
                        quote += float((frame[1] * frame[2]).sum())
                if not rows:
                    raise ValueError("Empty reported-trade archive")
                record.update(
                    rows=rows,
                    first_reported_trade_at=first_time.isoformat(),
                    last_reported_trade_at=last_time.isoformat(),
                    first_aggregate_id=first_id,
                    last_aggregate_id=last_id,
                    quote_turnover=quote,
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
            "parent_and_adopted_http_requests_counted_once": True,
            "downloaded_bytes": downloaded_bytes,
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
    p.add_argument("--parent", type=Path, required=True)
    a = p.parse_args()
    collect(a.output, a.parent)
