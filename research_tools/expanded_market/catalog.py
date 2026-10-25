"""Discover actually listed expanded5marchives, preserving sealed references."""

import argparse
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.historical_universe.catalog import Listing
from research_tools.max_effect.run_accounts import verify_manifest


def catalog(cohort, output):
    output.mkdir(parents=True, exist_ok=False)
    (output / "raw").mkdir()
    verify_manifest(cohort)
    for p, digest in read(cohort / "registration.json")["source_hashes"].items():
        if sha(Path(p)) != digest:
            raise ValueError("Cohort source changed")
    plan = read(cohort / "plan.json")
    existing = Path(plan["existing_market"])
    verify_manifest(existing)
    old = set(read(existing / "prepare_plan.json")["symbols"])
    names = read(cohort / "symbols.json")
    save(
        output / "plan.json",
        {
            **plan,
            "cohort": str(cohort),
            "cohort_completion_sha256": sha(cohort / "completion.json"),
            "catalog_source_sha256": sha(Path(__file__)),
            "listing_source_sha256": sha(Path("research_tools/historical_universe/catalog.py")),
            "existing_market_completion_sha256": sha(existing / "completion.json"),
            "before_market_requests": True,
        },
    )
    references = []
    for symbol in sorted(old & set(names)):
        source = existing / (symbol + "_5m.parquet")
        references.append(
            {
                "symbol": symbol,
                "path": str(source),
                "sha256": sha(source),
                "kind": "sealed_six_column_reference",
            }
        )
    listing = Listing(output, plan)
    begin = pd.Timestamp(plan["bars_start"], tz="UTC").strftime("%Y-%m")
    end = (pd.Timestamp(plan["end_exclusive"], tz="UTC") - pd.Timedelta(days=1)).strftime("%Y-%m")
    records = []
    reports = []

    def one(symbol):
        prefix = plan["prefix"] + symbol + "/5m/"
        contents, children, pages = listing.all(prefix)
        if children:
            raise ValueError("Unexpected nested5mdirectory")
        by = {r["key"]: r for r in contents}
        rows = []
        pattern = re.compile(re.escape(prefix + symbol + "-5m-") + r"(\d{4}-\d{2})\.zip")
        for key, r in sorted(by.items()):
            match = pattern.fullmatch(key)
            if not match or not begin <= match[1] <= end:
                continue
            if key + ".CHECKSUM" not in by:
                raise ValueError("ListedZIPwithoutCHECKSUM")
            rows.append(
                {
                    "symbol": symbol,
                    "month": match[1],
                    "key": key,
                    "size": r["size"],
                    "checksum_key": key + ".CHECKSUM",
                }
            )
        return rows, {
            "symbol": symbol,
            "archives": len(rows),
            "pages": pages,
            "no_listed_archive_is_not_zero_data": not rows,
        }

    with ThreadPoolExecutor(max_workers=plan["budget"]["workers"]) as pool:
        futures = {pool.submit(one, s): s for s in names if s not in old}
        for i, future in enumerate(as_completed(futures)):
            rows, report = future.result()
            records.extend(rows)
            reports.append(report)
            if (i + 1) % 25 == 0:
                print(
                    {
                        "listed_symbols": i + 1,
                        "archives": len(records),
                        "requests": listing.requests,
                    },
                    flush=True,
                )
    if len(records) > plan["budget"]["maximum_archives"]:
        raise ValueError("Archive budget exceeded")
    save(output / "archives.json", sorted(records, key=lambda r: (r["symbol"], r["month"])))
    save(output / "references.json", references)
    save(
        output / "summary.json",
        {
            "schema": "expanded5m-catalog.v1",
            "requested_symbols": len(names),
            "reused_reference_symbols": len(references),
            "new_symbols": len(reports),
            "listed_archives": len(records),
            "listing_requests": listing.requests,
            "reports": reports,
            "new_financial_accounts": 0,
            "profit_goal_met": False,
        },
    )
    save(
        output / "completion.json",
        {
            "manifest": {
                str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()
            },
            "new_financial_accounts": 0,
        },
    )
    print(
        {
            k: read(output / "summary.json")[k]
            for k in [
                "requested_symbols",
                "reused_reference_symbols",
                "new_symbols",
                "listed_archives",
                "listing_requests",
            ]
        },
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    catalog(a.cohort, a.output)
