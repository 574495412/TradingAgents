"""Re-read official raw archives independently of the spot preparation parser."""

import argparse
import csv
import hashlib
import io
import json
import math
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root, output):
    if output.exists():
        raise FileExistsError("Do not overwrite an earlier independent audit")
    receipt = read(root / "completion.json")
    for name, digest in receipt["manifest"].items():
        path = root / name
        if root.resolve() not in path.resolve().parents or sha(path) != digest:
            raise ValueError("Feasibility manifest changed or path outside root")
    plan = read(root / "plan.json")
    for name, digest in plan["source_hashes"].items():
        if sha(Path(name)) != digest:
            raise ValueError("Preparation source changed")
    summary = read(root / "summary.json")
    connector = read(root / "connector.json")
    parent = Path(plan["parent_perpetual_data"])
    reports = []
    maximum_spot_relative_error = 0.0
    for info in summary["symbols"]:
        symbol = info["symbol"]
        values, clocks = [], []
        for archive in summary["archives"]:
            if archive["symbol"] != symbol:
                continue
            path = root / "archives" / archive["url"].rsplit("/", 1)[1]
            checksum = path.with_suffix(".zip.CHECKSUM").read_text().split()[0]
            if sha(path) != checksum or checksum != archive["sha256"]:
                raise ValueError("Official archive bytes/checksum changed")
            with zipfile.ZipFile(path) as stream:
                records = list(csv.reader(io.TextIOWrapper(stream.open(stream.namelist()[0]))))
            month = datetime.strptime(archive["month"], "%Y-%m").replace(tzinfo=UTC)
            units = 1000000 if month.year >= 2025 else 1000
            step = 300 * units
            opening = int(month.timestamp()) * units
            for i, row in enumerate(records):
                if (
                    len(row) != 12
                    or int(row[0]) != opening + i * step
                    or int(row[6]) != opening + (i + 1) * step - 1
                ):
                    raise ValueError("Raw archive time unit/calendar mismatch")
                clocks.append(int(row[0]) * (1000000000 // units))
                values.append([float(row[k]) for k in [1, 2, 3, 4, 5, 7]])
            if len(records) != archive["rows"]:
                raise ValueError("Raw archive row count differs")
        actual = pd.read_parquet(root / f"{symbol}_spot_5m.parquet")
        raw_values = np.array(values)
        if not np.array_equal(actual.index.as_unit("ns").asi8, clocks) or not np.allclose(
            actual.to_numpy(), raw_values, rtol=3e-15, atol=1e-9
        ):
            raise ValueError("Independent raw spot values differ from parquet")
        maximum_spot_relative_error = max(
            maximum_spot_relative_error,
            float(
                np.max(np.abs(actual.to_numpy() - raw_values) / np.maximum(1, np.abs(raw_values)))
            ),
        )
        raw_funding = [
            r
            for p in connector["funding_pages"]
            if p["request"]["symbol"] == symbol
            for r in p["rows"]
        ]
        funding = pd.read_parquet(root / f"{symbol}_funding_exact.parquet")
        if not np.array_equal(
            funding.index.as_unit("ms").asi8, [r["fundingTime"] for r in raw_funding]
        ):
            raise ValueError("Funding settlement times rounded or changed")
        if not np.array_equal(
            funding.funding_rate.to_numpy(), [float(r["fundingRate"]) for r in raw_funding]
        ) or not np.array_equal(
            funding.mark.to_numpy(), [float(r["markPrice"]) for r in raw_funding]
        ):
            raise ValueError("Actual funding rates/marks changed")
        for part in info["funding_rate_by_year"]:
            rows = [
                r
                for r in raw_funding
                if datetime.fromtimestamp(r["fundingTime"] / 1000, UTC).year == part["year"]
            ]
            total = math.fsum(float(r["fundingRate"]) for r in rows)
            if len(rows) != part["events"] or not math.isclose(
                total, part["sum_rates"], abs_tol=1e-12
            ):
                raise ValueError("Funding yearly totals differ")
        for name, digest in info["parent_hashes"].items():
            if sha(parent / name) != digest:
                raise ValueError("Parent market input changed")
        perpetual = pd.read_parquet(parent / f"{symbol}_5m.parquet").loc[actual.index]
        basis = perpetual.close.to_numpy() / raw_values[:, 3] - 1
        for quantile, expected in info["simultaneous_close_basis_quantiles"].items():
            if not math.isclose(
                float(np.quantile(basis, float(quantile))), expected, abs_tol=1e-12
            ):
                raise ValueError("Simultaneous close basis diagnostic differs")
        reports.append(
            {
                "symbol": symbol,
                "spot_rows": len(actual),
                "spot_values": raw_values.size,
                "exact_funding_events": len(funding),
                "passed": True,
            }
        )
    sample = next(p for p in connector["availability_probe"] if p["name"] == "funding")["response"][
        "result"
    ]
    btc = pd.read_parquet(root / "BTCUSDT_funding_exact.parquet")
    for row in sample:
        event = btc.loc[pd.Timestamp(row["fundingTime"], unit="ms", tz="UTC")]
        if event.funding_rate != float(row["fundingRate"]) or event.mark != float(row["markPrice"]):
            raise ValueError("Initial public funding sample differs from continuous dataset")
    budget = plan["illustrative_budget"]
    for scenario, reported in zip(plan["cost_scenarios"], summary["cost_arithmetic"], strict=True):
        spot_round = (
            2
            * budget["spot_leg_nav_fraction"]
            * (scenario["spot_commission_bps"] + scenario["slippage_bps_each_leg_each_side"])
            / 10000
        )
        futures_round = (
            2
            * budget["perpetual_leg_nav_fraction"]
            * (scenario["perpetual_commission_bps"] + scenario["slippage_bps_each_leg_each_side"])
            / 10000
        )
        if not math.isclose(
            spot_round + futures_round, reported["round_trip_cost_on_initial_nav"], abs_tol=1e-15
        ):
            raise ValueError("Both-leg round-trip cost arithmetic differs")
    result = {
        "schema": "spot-perpetual-independent-data-audit.v1",
        "passed": True,
        "completion_sha256": sha(root / "completion.json"),
        "audit_source_sha256": sha(Path(__file__)),
        "archives": len(summary["archives"]),
        "symbols": reports,
        "maximum_spot_relative_error": maximum_spot_relative_error,
        "actual_financial_accounts": 0,
        "profit_proven": False,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.data, args.output)
