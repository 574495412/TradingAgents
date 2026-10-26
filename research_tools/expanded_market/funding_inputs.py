"""Seal rate/time-valid funding records without imputing unqualified settlement marks."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest


def validate_page(raw, symbol, expected_start, end):
    request = raw["request"]
    if request != {"symbol": symbol, "startTime": expected_start, "endTime": end, "limit": 1000}:
        raise ValueError("Funding page request differs")
    response = raw["response"]
    rows = response.get("structuredContent", {}).get("result")
    if response.get("isError") or not isinstance(rows, list) or len(rows) > 1000:
        raise ValueError("Funding page failed")
    previous = expected_start - 1
    for row in rows:
        at = row["fundingTime"]
        if (
            row["symbol"] != symbol
            or not isinstance(at, int)
            or not expected_start <= at <= end
            or at <= previous
            or not np.isfinite(float(row["fundingRate"]))
        ):
            raise ValueError("Funding time, identity or rate invalid")
        # A zero/unknown mark must remain archived. It is forbidden for held cashflows.
        float(row["markPrice"])
        previous = at
    return rows


def prepare(collection, recovery, cohort, output):
    output.mkdir(parents=True, exist_ok=False)
    plan = read(collection / "plan.json")
    if sha(cohort / "completion.json") != plan["cohort_completion_sha256"]:
        raise ValueError("Funding and cohort identity differ")
    recover = read(recovery / "plan.json")
    if sha(Path(__file__).with_name("funding_recovery.json")) != sha(recovery / "plan.json"):
        raise ValueError("Recovery source plan changed")
    summary = read(collection / "collection_summary.json")
    resumed = read(recovery / "collection_summary.json")
    calls = summary["api_calls"] + resumed["api_calls"]
    if calls > recover["maximum_combined_calls"]:
        raise ValueError("Original and continuation request budget exceeded")
    parent = Path(read(cohort / "plan.json")["existing_funding"])
    verify_manifest(parent)
    verify_manifest(cohort)
    bindings = {
        str(p): sha(p) for root in [collection, recovery] for p in root.rglob("*") if p.is_file()
    }
    binding_paths = [
        Path(__file__),
        Path(__file__).with_name("funding_recovery.json"),
        parent / "completion.json",
        cohort / "completion.json",
    ]
    bindings.update({str(p): sha(p) for p in binding_paths})
    save(
        output / "registration.json",
        {
            "source_hashes": bindings,
            "api_calls": calls,
            "new_financial_accounts": 0,
            "nonpositive_mark_substitution": False,
        },
    )
    first = int(pd.Timestamp("2024-12-25", tz="UTC").timestamp() * 1000)
    end = int(pd.Timestamp("2026-09-01", tz="UTC").timestamp() * 1000) - 1
    recovered = {r["symbol"]: r for r in recover["scope"]}
    first_candidate = {}
    for r in read(cohort / "cohort.json"):
        for s in r["selected"]:
            first_candidate.setdefault(s, pd.Timestamp(r["decision"]))
    reports, unknowns = [], []
    (output / "symbols").mkdir()
    for symbol in plan["symbols"]:
        pages = 0
        if symbol in plan["adopted_symbols"]:
            path = parent / (symbol + "_exact.parquet")
            frame = pd.read_parquet(path)
            if not (
                (frame.index >= pd.to_datetime(first, unit="ms", utc=True))
                & (frame.index <= pd.to_datetime(end, unit="ms", utc=True))
            ).all():
                raise ValueError("Adopted actual funding outside registered window")
            provenance = {"kind": "sealed_exact_reference", "path": str(path), "sha256": sha(path)}
        else:
            pages_paths = sorted((collection / "raw").glob(symbol + "_page*.json"))
            if symbol in recovered:
                if sha(pages_paths[0]) != recovered[symbol]["parent_sha256"]:
                    raise ValueError("Stopped parent page changed")
                pages_paths += sorted(recovery.glob(symbol + "_page*.json"))
                done = read(recovery / ("reports_" + symbol + ".json"))
            else:
                done = read(collection / ("reports_" + symbol + ".json"))
            if not done["collection_complete"] or done["error"]:
                raise ValueError("Funding acquisition still incomplete: " + symbol)
            start, records, last_count = first, [], 1000
            for path in pages_paths:
                if last_count < 1000:
                    raise ValueError("Unexpected page after completed short page")
                rows = validate_page(read(path), symbol, start, end)
                records += rows
                pages += 1
                last_count = len(rows)
                if rows:
                    start = rows[-1]["fundingTime"] + 1
            if last_count >= 1000 or not records:
                raise ValueError("Final funding page is incomplete or history unavailable")
            frame = pd.DataFrame(
                {
                    "funding_rate": [float(r["fundingRate"]) for r in records],
                    "mark": [float(r["markPrice"]) for r in records],
                },
                index=pd.DatetimeIndex(
                    pd.to_datetime([r["fundingTime"] for r in records], unit="ms", utc=True),
                    name="date",
                ),
            )
            provenance = {"kind": "paginated_actual_api", "pages": [str(p) for p in pages_paths]}
        if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
            raise ValueError("Funding event grain differs")
        frame["rate_time_usable"] = np.isfinite(frame.funding_rate)
        frame["held_cash_mark_usable"] = np.isfinite(frame.mark) & frame.mark.gt(0)
        frame["payable_cashflow_certified"] = False
        at = first_candidate.get(symbol)
        bad = frame.loc[~frame.held_cash_mark_usable]
        for clock, row in bad.iterrows():
            unknowns.append(
                {
                    "symbol": symbol,
                    "time": clock.isoformat(),
                    "mark": float(row.mark) if np.isfinite(row.mark) else None,
                    "rate": float(row.funding_rate),
                    "first_candidate": at.isoformat() if at is not None else None,
                    "before_first_candidate": at is not None and clock < at,
                    "inside_first_predictor_window": at is not None
                    and at - pd.Timedelta(days=7) <= clock < at,
                }
            )
        frame.to_parquet(output / "symbols" / (symbol + "_exact.parquet"))
        reports.append(
            {
                "symbol": symbol,
                "records": len(frame),
                "pages": pages,
                "source": provenance,
                "nonpositive_or_unknown_marks": len(bad),
                "first": frame.index[0].isoformat(),
                "last": frame.index[-1].isoformat(),
                "largest_observed_gap_hours": float(
                    frame.index.to_series().diff().max().total_seconds() / 3600
                ),
                "reported_event_enumeration_complete": True,
                "historical_funding_schedule_certified": False,
                "payable_cashflows_certified": False,
            }
        )
    save(output / "symbols.json", reports)
    save(output / "unqualified_marks.json", unknowns)
    result = {
        "schema": "expanded-exact-funding-inputs.v1",
        "symbols": len(reports),
        "reported_events": sum(r["records"] for r in reports),
        "api_calls": calls,
        "nonpositive_or_unknown_marks": len(unknowns),
        "unqualified_marks_at_or_after_first_candidate": sum(
            not r["before_first_candidate"] for r in unknowns
        ),
        "rate_time_inputs_enumerated": True,
        "all_settlement_marks_positive": not unknowns,
        "historical_schedule_and_payable_cashflows_certified": False,
        "no_mark_or_calendar_imputation": True,
        "new_financial_accounts": 0,
        "profit_goal_met": False,
        "production_eligible": False,
    }
    save(output / "summary.json", result)
    save(
        output / "completion.json",
        {
            "manifest": {
                str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()
            },
            "input_capture_complete": True,
            "financial_certification": False,
        },
    )
    print(result, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--collection", type=Path, required=True)
    p.add_argument("--recovery", type=Path, required=True)
    p.add_argument("--cohort", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    prepare(a.collection, a.recovery, a.cohort, a.output)
