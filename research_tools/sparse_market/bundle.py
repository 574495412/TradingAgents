"""Bind acquired calendars and actual funding, retaining immutable parent identities."""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from research_tools.expanded_market.funding_inputs import validate_page
from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    p = Path(__file__).with_name("plan.json")
    plan = read(p)
    market = Path(plan["parent_market"])
    funding = Path(plan["parent_funding"])
    additional = Path("data/research/sparse_addition_market_20261026_v001")
    collected = Path("data/research/sparse_addition_funding_collection_20261026_v001")
    for root in [market, funding, additional]:
        verify_manifest(root)
    source_bindings = {str(f): sha(f) for f in collected.iterdir() if f.is_file()}
    source_bindings.update(
        {
            str(f): sha(f)
            for f in [
                p,
                Path(__file__),
                Path("research_tools/expanded_market/funding_inputs.py"),
                *[r / "completion.json" for r in [market, funding, additional]],
            ]
        }
    )
    save(
        output / "registration.json",
        {
            "source_hashes": source_bindings,
            "hardlink_read_only_adoption": True,
            "before_financial_returns": True,
            "new_financial_accounts": 0,
        },
    )
    (output / "symbols").mkdir()
    (output / "funding").mkdir()
    references = []
    names = set()
    for root in [market, additional]:
        for file in (root / "symbols").glob("*_5m.parquet"):
            s = file.name.removesuffix("_5m.parquet")
            if s in names:
                raise ValueError("Duplicate market source")
            names.add(s)
            os.link(file, output / "symbols" / file.name)
            references.append(
                {"symbol": s, "kind": "market", "source": str(file), "sha256": sha(file)}
            )
    for file in (funding / "symbols").glob("*_exact.parquet"):
        os.link(file, output / "funding" / file.name)
        references.append(
            {
                "symbol": file.name.removesuffix("_exact.parquet"),
                "kind": "funding",
                "source": str(file),
                "sha256": sha(file),
            }
        )
    captured = read(collected / "plan.json")
    new_rows = []
    for s in captured["symbols"]:
        report = read(collected / ("report_" + s + ".json"))
        if not report["collection_complete"] or report["error"]:
            raise ValueError("New actual funding collection incomplete")
        start = captured["startTime"]
        rows, count = [], 1000
        for file in sorted(collected.glob(s + "_page*.json")):
            if count < 1000:
                raise ValueError("Unexpected page after short actual page")
            page = validate_page(read(file), s, start, captured["endTime"])
            rows.extend(page)
            count = len(page)
            if page:
                start = page[-1]["fundingTime"] + 1
        if count >= 1000 or len(rows) != report["records"]:
            raise ValueError("Funding page count differs")
        f = pd.DataFrame(
            {
                "funding_rate": [float(r["fundingRate"]) for r in rows],
                "mark": [float(r["markPrice"]) for r in rows],
            },
            index=pd.DatetimeIndex(
                pd.to_datetime([r["fundingTime"] for r in rows], unit="ms", utc=True), name="date"
            ),
        )
        f["rate_time_usable"] = True
        f["held_cash_mark_usable"] = np.isfinite(f.mark) & f.mark.gt(0)
        f["payable_cashflow_certified"] = False
        f.to_parquet(output / "funding" / (s + "_exact.parquet"))
        new_rows.append(
            {
                "symbol": s,
                "records": len(f),
                "unqualified_marks": int((~f.held_cash_mark_usable).sum()),
            }
        )
    funded = {f.name.removesuffix("_exact.parquet") for f in (output / "funding").iterdir()}
    if funded != names:
        raise ValueError("Funding and market acquisition rosters differ")
    save(output / "references.json", references)
    save(
        output / "plan.json",
        {
            **plan,
            "parent_market": str(output),
            "bundled_parent_market": str(market),
            "additional_market": str(additional),
        },
    )
    save(
        output / "summary.json",
        {
            "schema": "sparse-source-bundle.v1",
            "acquired_symbols": len(names),
            "additional_funding": new_rows,
            "historical_tradability_certified": False,
            "payable_cashflows_certified": False,
            "financial_accounts": 0,
        },
    )
    save(
        output / "completion.json",
        {
            "manifest": {
                str(f.relative_to(output)): sha(f) for f in output.rglob("*") if f.is_file()
            },
            "financial_accounts": 0,
            "historical_certification_passed": False,
        },
    )
    print(read(output / "summary.json"), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    prepare(p.parse_args().output)
