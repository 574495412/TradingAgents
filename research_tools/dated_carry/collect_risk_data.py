"""Bounded rule documents, actual dated marks and one-month warm-up prices."""

import argparse
import csv
import hashlib
import io
import json
import zipfile
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd
import requests

from research_tools.contract_lifecycles.collect import text_nodes
from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest


def collect(output):
    output.mkdir(parents=True, exist_ok=False)
    src = Path(__file__).with_name("risk_data_plan.json")
    p = read(src)
    for root in p["parents"]:
        verify_manifest(Path(root))
    paths = [src, Path(__file__), Path("research_tools/contract_lifecycles/collect.py")]
    save(
        output / "plan.json",
        {
            **p,
            "source_hashes": {str(q): sha(q) for q in paths},
            "parent_completion_hashes": {r: sha(Path(r) / "completion.json") for r in p["parents"]},
        },
    )
    save(
        output / "registration.json",
        {
            "registered_at_utc": datetime.now(UTC).isoformat(),
            "before_new_mark_and_rule_capture": True,
            "new_financial_accounts": 0,
            "plan_sha256": sha(output / "plan.json"),
        },
    )
    count, byte_count, session = 0, 0, requests.Session()

    def get(url, stem, params=None):
        nonlocal count, byte_count
        assert count < p["budget"]["maximum_read_only_HTTP_requests"]
        count += 1
        response = session.get(url, params=params, timeout=(10, 30), allow_redirects=False)
        data = response.content
        byte_count += len(data)
        assert byte_count <= p["budget"]["maximum_response_bytes"]
        (output / stem).write_bytes(data)
        save(
            output / (stem + ".receipt.json"),
            {
                "url": response.url,
                "status": response.status_code,
                "bytes": len(data),
                "sha256": sha(output / stem),
                "retrieved_at_utc": datetime.now(UTC).isoformat(),
            },
        )
        return response.status_code, data

    documents = []
    for d in p["documents"]:
        stem = d["id"]
        status, body = get(
            "https://www.binance.com/bapi/composite/v1/public/cms/article/detail/query",
            stem + ".raw",
            {"articleCode": stem},
        )
        result = {**d, "status": status, "body_available": False}
        if status == 200:
            decoded = json.loads(body)
            if decoded.get("success") and decoded.get("data"):
                data = decoded["data"]
                text = text_nodes(json.loads(data["body"]))
                (output / (stem + ".txt")).write_text(text)
                result.update(
                    body_available=bool(text),
                    publishDate=data.get("publishDate"),
                    title=data.get("title"),
                    text_sha256=sha(output / (stem + ".txt")),
                    original_historical_body_certified=False,
                )
        documents.append(result)
        print({"document": stem, "available": result["body_available"]}, flush=True)
    jobs = [(c, "trade", m) for c in p["contracts"] for m in p["additional_trade_months"]]
    jobs += [(c, "mark", m) for c in p["contracts"] for m in p["mark_months"]]
    receipts = []
    pieces = {}
    for contract, kind, month in jobs:
        filename = f"{contract}-{kind}-5m-{month}.zip"
        url = p[kind + "_template"].format(contract=contract, month=month)
        status, body = get(url, filename)
        check_status, checksum = get(url + ".CHECKSUM", filename + ".CHECKSUM")
        row = {
            "contract": contract,
            "kind": kind,
            "month": month,
            "status": status,
            "checksum_status": check_status,
            "available": False,
        }
        if status == check_status == 200:
            expected, name = checksum.decode().strip().split()
            assert name == url.rsplit("/", 1)[-1] and hashlib.sha256(body).hexdigest() == expected
            with zipfile.ZipFile(io.BytesIO(body)) as zipped:
                assert len(zipped.namelist()) == 1
                records = list(
                    csv.reader(
                        io.TextIOWrapper(zipped.open(zipped.namelist()[0]), encoding="utf-8-sig")
                    )
                )
            if records[0][0] == "open_time":
                records = records[1:]
            assert all(len(r) == 12 for r in records)
            times = pd.DatetimeIndex(
                pd.to_datetime([int(r[0]) for r in records], unit="ms", utc=True), name="time"
            )
            assert (
                times.is_monotonic_increasing
                and not times.has_duplicates
                and (times.strftime("%Y-%m") == month).all()
            )
            full = [int(r[6]) == int(r[0]) + 299999 for r in records]
            prices = pd.DataFrame(
                [[float(Decimal(x)) for x in r[1:5]] for r in records],
                columns=["open", "high", "low", "close"],
                index=times,
            )
            assert (prices > 0).all().all() and prices.notna().all().all()
            assert (
                prices.high.ge(prices.max(axis=1)).all() and prices.low.le(prices.min(axis=1)).all()
            )
            if kind == "trade":
                prices["volume"] = [float(Decimal(r[5])) for r in records]
                prices["quote_volume"] = [float(Decimal(r[7])) for r in records]
            prices["full_5m_bar"] = full
            pieces.setdefault((contract, kind), []).append(prices)
            row.update(
                available=True,
                rows=len(records),
                first_open=times[0].isoformat(),
                last_open=times[-1].isoformat(),
                partial_rows=full.count(False),
                zip_sha256=expected,
            )
        receipts.append(row)
        print(row, flush=True)
    session.close()
    for (contract, kind), frames in pieces.items():
        joined = pd.concat(frames)
        assert not joined.index.has_duplicates
        joined.to_parquet(output / f"{contract}_{kind}_5m.parquet")
    summary = {
        "documents": documents,
        "market_archives": receipts,
        "all_documents_available": all(d["body_available"] for d in documents),
        "all_market_archives_available": all(r["available"] for r in receipts),
        "read_only_HTTP_requests": count,
        "response_bytes": byte_count,
        "new_financial_accounts": 0,
        "profit_goal_met": False,
    }
    save(output / "summary.json", summary)
    manifest = {q.name: sha(q) for q in output.iterdir() if q.is_file()}
    save(
        output / "completion.json",
        {"capture_finished": True, "manifest": manifest, "profit_goal_met": False},
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    collect(p.parse_args().output)
