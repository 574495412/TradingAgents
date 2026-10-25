"""Seal exact settlement events and independently compare every connector record."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from research_tools.factor_lab.io import read, save, sha


def prepare(root):
    if (root / "completion.json").exists():
        raise FileExistsError("Preserve funding snapshot")
    plan_path = Path(__file__).with_name("plan.json")
    plan = read(plan_path)
    collection = read(root / "collection.json")
    if collection["requests"] > plan["funding_request_budget"] or not all(
        r["passed"] for r in collection["outcomes"]
    ):
        raise ValueError("Funding collection incomplete or over budget")
    parent = Path(plan["data"])
    parent_receipt = read(parent / "completion.json")
    start, end = pd.Timestamp(plan["funding_start"]), pd.Timestamp(plan["funding_end_exclusive"])
    reports = []
    raw_files = []
    for symbol in collection["symbols"]:
        pages = sorted((root / "raw").glob(symbol + "_*.json"))
        raw_files.extend(pages)
        cursor = int(start.timestamp() * 1000)
        records = []
        for i, path in enumerate(pages):
            page = read(path)
            request = page["request"]
            rows = page["response"]["result"]
            if request != {
                "symbol": symbol,
                "startTime": cursor,
                "endTime": int(end.timestamp() * 1000) - 1,
                "limit": 1000,
            }:
                raise ValueError("Funding request scope/cursor changed")
            if i < len(pages) - 1 and len(rows) != 1000:
                raise ValueError("Early termination without page boundary")
            if i == len(pages) - 1 and len(rows) >= 1000:
                raise ValueError("Final funding page is truncated")
            if rows:
                cursor = rows[-1]["fundingTime"] + 1
            records.extend(rows)
        out = pd.DataFrame(
            {
                "funding_rate": [float(r["fundingRate"]) for r in records],
                "mark": [float(r["markPrice"]) for r in records],
            },
            index=pd.DatetimeIndex(
                pd.to_datetime([r["fundingTime"] for r in records], unit="ms", utc=True),
                name="date",
            ),
        )
        if (
            not len(out)
            or not out.index.is_monotonic_increasing
            or out.index.has_duplicates
            or not np.isfinite(out.to_numpy()).all()
            or (out.mark <= 0).any()
            or out.index[0] < start
            or out.index[-1] >= end
        ):
            raise ValueError("Funding record integrity failed")
        name = symbol + "_funding.parquet"
        if sha(parent / name) != parent_receipt["manifest"][name]:
            raise ValueError("Legacy funding source changed")
        old = pd.read_parquet(parent / name).loc[lambda f: (f.index >= start) & (f.index < end)]
        nominal = out.index.floor("h")
        if (
            nominal.has_duplicates
            or not nominal.equals(old.index)
            or not np.allclose(
                out.funding_rate.to_numpy(), old.funding_rate.to_numpy(), atol=1e-12, rtol=0
            )
        ):
            raise ValueError(
                "Actual versus sealed nominal funding calendar/rates differ: " + symbol
            )
        out.to_parquet(root / (symbol + "_exact.parquet"))
        # Reopen bytes and independently test raw timestamp/rate/mark values.
        restored = pd.read_parquet(root / (symbol + "_exact.parquet"))
        if (
            restored.index.as_unit("ms").asi8.tolist() != [r["fundingTime"] for r in records]
            or restored.funding_rate.tolist() != [float(r["fundingRate"]) for r in records]
            or restored.mark.tolist() != [float(r["markPrice"]) for r in records]
        ):
            raise ValueError("Raw funding reconstruction differs")
        reports.append(
            {
                "symbol": symbol,
                "pages": len(pages),
                "events": len(out),
                "nominal_calendar_rates_match": True,
                "exact_raw_times_rates_marks_match": True,
                "maximum_ms_after_nominal_hour": float(
                    (out.index - nominal).total_seconds().max() * 1000
                ),
                "negative_events": int((out.funding_rate < 0).sum()),
            }
        )
    if (
        sum(r["events"] for r in reports) != collection["records"]
        or len(raw_files) != collection["requests"]
    ):
        raise ValueError("Funding coverage counts differ")
    summary = {
        "schema": "max-effect-actual-funding.v1",
        "passed": True,
        "symbols": len(reports),
        "requests": len(raw_files),
        "events": collection["records"],
        "reports": reports,
        "first": "2025-01-01",
        "end_exclusive": "2026-09-01",
        "financial_accounts": 0,
        "proxy_marks_used": False,
        "source_hashes": {str(plan_path): sha(plan_path), str(Path(__file__)): sha(Path(__file__))},
    }
    save(root / "summary.json", summary)
    files = (
        [root / "collection.json", root / "summary.json"]
        + raw_files
        + list(root.glob("*_exact.parquet"))
    )
    save(
        root / "completion.json",
        {
            "manifest": {str(p.relative_to(root)): sha(p) for p in files},
            "audit_passed": True,
            "events": collection["records"],
        },
    )
    print(
        {
            "exact_funding_complete": True,
            "symbols": len(reports),
            "requests": len(raw_files),
            "events": collection["records"],
            "no_proxy_marks": True,
        },
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    a = p.parse_args()
    prepare(a.data)
