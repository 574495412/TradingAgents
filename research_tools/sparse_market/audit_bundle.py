"""Verify source adoption and independently decode new actual funding pages."""

import argparse
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest


def audit(bundle, cohort, output):
    if output.exists():
        raise FileExistsError("Preserve source-bundle audit")
    verify_manifest(bundle)
    verify_manifest(cohort)
    for path, digest in read(bundle / "registration.json")["source_hashes"].items():
        if sha(Path(path)) != digest:
            raise ValueError("Source bundle binding changed")
    for r in read(bundle / "references.json"):
        path = (
            bundle
            / ("symbols" if r["kind"] == "market" else "funding")
            / (r["symbol"] + ("_5m.parquet" if r["kind"] == "market" else "_exact.parquet"))
        )
        if sha(path) != r["sha256"] or sha(Path(r["source"])) != r["sha256"]:
            raise ValueError("Adopted source values changed")
    raw = Path("data/research/sparse_addition_funding_collection_20261026_v001")
    new_records = 0
    for s in read(raw / "plan.json")["symbols"]:
        rows = [
            r
            for p in sorted(raw.glob(s + "_page*.json"))
            for r in read(p)["response"]["structuredContent"]["result"]
        ]
        f = pd.read_parquet(bundle / "funding" / (s + "_exact.parquet"))
        clock = pd.DatetimeIndex(
            pd.to_datetime([r["fundingTime"] for r in rows], unit="ms", utc=True)
        )
        rates = np.array([float(Decimal(r["fundingRate"])) for r in rows])
        marks = np.array([float(Decimal(r["markPrice"])) for r in rows])
        if (
            not clock.equals(f.index)
            or not np.array_equal(rates, f.funding_rate)
            or not np.array_equal(marks, f.mark)
        ):
            raise ValueError("New actual raw-event time/rate/mark changed")
        new_records += len(f)
    first = {}
    for variant in read(cohort / "summary.json")["source_variants"]:
        for row in read(cohort / (variant + "_cohort.json")):
            for s in row["selected"]:
                at = pd.Timestamp(row["decision"])
                first[s] = min(first.get(s, at), at)
    count, unknown = 0, []
    for s in read(cohort / "symbols.json"):
        f = pd.read_parquet(bundle / "funding" / (s + "_exact.parquet"))
        if (
            f.index.has_duplicates
            or not f.index.is_monotonic_increasing
            or not np.isfinite(f.funding_rate).all()
        ):
            raise ValueError("Actual funding event grain differs")
        usable = np.isfinite(f.mark) & f.mark.gt(0)
        if not usable.equals(f.held_cash_mark_usable) or f.payable_cashflow_certified.any():
            raise ValueError("Unknown settlement qualification changed")
        for at in f.index[~usable]:
            if s not in first or at >= first[s] - pd.Timedelta(days=7):
                raise ValueError("Unknown mark overlaps candidate or first signal window")
            unknown.append(
                {"symbol": s, "at": at.isoformat(), "first_candidate": first[s].isoformat()}
            )
        count += len(f)
    result = {
        "schema": "sparse-bundle-independent-audit.v1",
        "passed": True,
        "source_sha256": sha(Path(__file__)),
        "bundle_completion_sha256": sha(bundle / "completion.json"),
        "cohort_completion_sha256": sha(cohort / "completion.json"),
        "symbols": len(first) + 1,
        "selected_source_funding_records": count,
        "new_raw_funding_records": new_records,
        "unqualified_marks": unknown,
        "new_raw_fields_preserved": True,
        "historical_schedule_certified": False,
        "payable_cashflows_certified": False,
        "profit_goal_met": False,
    }
    save(output, result)
    print({k: v for k, v in result.items() if k != "unqualified_marks"}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--cohort", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    audit(a.bundle, a.cohort, a.output)
