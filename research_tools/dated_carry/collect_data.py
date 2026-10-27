"""Six predeclared dated-contract raw archives; no strategy/account returns."""

import argparse
import hashlib
import io
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest


def collect(output):
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).with_name("data_plan.json")
    plan = read(source)
    parents = [Path(plan[k]) for k in ["spot_parent", "source_parent", "source_followup"]]
    for root in parents:
        verify_manifest(root)
    paths = [source, Path(__file__), *[p / "completion.json" for p in parents]]
    paths += [parents[0] / (s + "_spot_5m.parquet") for s in ["BTCUSDT", "ETHUSDT"]]
    save(output / "plan.json", {**plan, "source_hashes": {str(p): sha(p) for p in paths}})
    save(
        output / "registration.json",
        {
            "registered_at_utc": datetime.now(UTC).isoformat(),
            "before_new_dated_prices": True,
            "plan_sha256": sha(output / "plan.json"),
            "no_new_financial_accounts": True,
        },
    )
    session = requests.Session()
    count, byte_count = 0, 0

    def fetch(url, filename):
        nonlocal count, byte_count
        assert count < plan["budget"]["maximum_read_only_http_requests"]
        count += 1
        response = session.get(url, timeout=(10, 30), allow_redirects=False)
        body = response.content
        byte_count += len(body)
        assert byte_count <= plan["budget"]["maximum_response_bytes"]
        (output / filename).write_bytes(body)
        save(
            output / (filename + ".receipt.json"),
            {
                "url": url,
                "retrieved_at_utc": datetime.now(UTC).isoformat(),
                "status": response.status_code,
                "bytes": len(body),
                "sha256": sha(output / filename),
            },
        )
        return response.status_code, body

    receipts = []
    for contract in plan["contracts"]:
        pieces = []
        for month in plan["months"]:
            filename = f"{contract}-5m-{month}.zip"
            url = plan["archive_template"].format(contract=contract, month=month)
            status, body = fetch(url, filename)
            check_status, checksum = fetch(url + ".CHECKSUM", filename + ".CHECKSUM")
            assert status == check_status == 200
            advertised, checkname = checksum.decode().strip().split()
            assert checkname == filename and hashlib.sha256(body).hexdigest() == advertised
            with zipfile.ZipFile(io.BytesIO(body)) as zipped:
                assert len(zipped.namelist()) == 1
                frame = pd.read_csv(zipped.open(zipped.namelist()[0]))
            assert len(frame.columns) == 12 and "open_time" in frame.columns
            frame = frame.iloc[:, :12].copy()
            frame.columns = [
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "trades",
                "taker_base",
                "taker_quote",
                "ignore",
            ]
            clock = pd.to_datetime(frame.open_time, unit="ms", utc=True)
            frame.index = pd.DatetimeIndex(clock, name="time")
            assert frame.index.is_monotonic_increasing and not frame.index.has_duplicates
            assert all(frame.index.strftime("%Y-%m") == month)
            numeric = frame[["open", "high", "low", "close", "volume", "quote_volume"]]
            assert (
                numeric.notna().all().all()
                and (numeric[["open", "high", "low", "close"]] > 0).all().all()
            )
            assert frame.high.ge(frame[["open", "close", "low"]].max(axis=1)).all()
            assert frame.low.le(frame[["open", "close", "high"]].min(axis=1)).all()
            assert frame.volume.ge(0).all() and frame.quote_volume.ge(0).all()
            frame["full_5m_bar"] = frame.close_time.eq(frame.open_time + 299999)
            frame["positive_execution_volume"] = frame.volume.gt(0) & frame.quote_volume.gt(0)
            pieces.append(frame)
            r = {
                "contract": contract,
                "month": month,
                "rows": len(frame),
                "first_open": frame.index[0].isoformat(),
                "last_open": frame.index[-1].isoformat(),
                "partial_rows": int((~frame.full_5m_bar).sum()),
                "nonpositive_execution_volume": int((~frame.positive_execution_volume).sum()),
                "zip_sha256": advertised,
            }
            receipts.append(r)
            print(r, flush=True)
        combined = pd.concat(pieces)
        assert not combined.index.has_duplicates
        combined.to_parquet(output / (contract + "_5m.parquet"))
    settlements = []
    for i, url in enumerate(plan["settlement_endpoints"]):
        status, body = fetch(url, f"settlement_{i}.raw")
        settlements.append(
            {"url": url, "status": status, "bytes": len(body), "raw_file": f"settlement_{i}.raw"}
        )
    session.close()
    save(
        output / "summary.json",
        {
            "schema": "dated-carry-raw-data-pilot.v1",
            "archives": receipts,
            "settlement_responses": settlements,
            "http_requests": count,
            "response_bytes": byte_count,
            "new_market_archives": len(receipts),
            "new_financial_accounts": 0,
            "profit_goal_met": False,
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
