"""Independent raw-event reconciliation and Decimal past-window checks."""

import argparse
import hashlib
import json
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def audit(inputs, cohort, output):
    if output.exists():
        raise FileExistsError("Preserve completed funding audit")
    for name, digest in read(inputs / "completion.json")["manifest"].items():
        if sha(inputs / name) != digest:
            raise ValueError("Funding artifact changed")
    for path, digest in read(inputs / "registration.json")["source_hashes"].items():
        if sha(Path(path)) != digest:
            raise ValueError("Funding source identity changed")
    first_clock = {}
    clocks = {}
    for r in read(cohort / "cohort.json"):
        for s in r["selected"]:
            at = pd.Timestamp(r["decision"])
            clocks.setdefault(s, []).append(at)
            first_clock.setdefault(s, at)
    records, counts, unknown, max_error, anchors = [], 0, 0, 0.0, 0
    for report in read(inputs / "symbols.json"):
        s, source = report["symbol"], report["source"]
        f = pd.read_parquet(inputs / "symbols" / (s + "_exact.parquet"))
        if source["kind"] == "sealed_exact_reference":
            p = Path(source["path"])
            if sha(p) != source["sha256"]:
                raise ValueError("Adopted reference changed")
            original = pd.read_parquet(p)
            times = original.index
            rates, marks = original.funding_rate.to_numpy(), original.mark.to_numpy()
            decimal_rates = [Decimal(str(v)) for v in rates]
        else:
            raw = []
            last_time, last_count = None, None
            for i, p in enumerate(source["pages"]):
                page = read(Path(p))
                req = page["request"]
                if last_time is not None and req["startTime"] != last_time + 1:
                    raise ValueError("API continuation gap or overlap")
                if i and last_count != 1000:
                    raise ValueError("Page after completed short page")
                rows = page["response"]["structuredContent"]["result"]
                if any(r["symbol"] != s for r in rows):
                    raise ValueError("Foreign funding symbol")
                if rows:
                    last_time = rows[-1]["fundingTime"]
                last_count = len(rows)
                raw.extend(rows)
            if last_count >= 1000:
                raise ValueError("No terminal funding page")
            times = pd.DatetimeIndex(
                pd.to_datetime([r["fundingTime"] for r in raw], unit="ms", utc=True)
            )
            decimal_rates = [Decimal(r["fundingRate"]) for r in raw]
            rates = np.array([float(r) for r in decimal_rates])
            marks = np.array([float(Decimal(r["markPrice"])) for r in raw])
        if not times.equals(f.index) or len(f) != report["records"]:
            raise ValueError("Dropped, rounded or extra event timestamps")
        if not np.array_equal(rates, f.funding_rate) or not np.array_equal(
            marks, f.mark, equal_nan=True
        ):
            raise ValueError("Funding rate or mark source value changed")
        usable = np.isfinite(marks) & (marks > 0)
        if (
            not np.array_equal(usable, f.held_cash_mark_usable)
            or f.payable_cashflow_certified.any()
        ):
            raise ValueError("Cash qualification differs")
        bad_times = times[~usable]
        if len(bad_times) and (s not in first_clock or not (bad_times < first_clock[s]).all()):
            raise ValueError("Unknown mark overlaps a candidate period; block financial inputs")
        # Compare floating implementation windows to a raw Decimal [t-7d,t) sum.
        for clock in sorted(set(clocks.get(s, []))):
            left = times.searchsorted(clock - pd.Timedelta(days=7), side="left")
            right = times.searchsorted(clock, side="left")
            exact = float(sum(decimal_rates[left:right], Decimal(0)))
            candidate = float(
                f.loc[
                    (f.index >= clock - pd.Timedelta(days=7)) & (f.index < clock), "funding_rate"
                ].sum()
            )
            error = abs(exact - candidate)
            max_error = max(max_error, error)
            anchors += 1
            if error > 1e-12:
                raise ValueError("Past-only Decimal funding window differs")
        counts += len(f)
        unknown += len(bad_times)
        records.append(
            {
                "symbol": s,
                "records": len(f),
                "raw_or_reference_reconciled": True,
                "unqualified_marks": len(bad_times),
                "first_candidate": first_clock.get(s).isoformat() if s in first_clock else None,
            }
        )
    result = {
        "schema": "expanded-funding-independent-audit.v1",
        "passed": True,
        "source_sha256": sha(Path(__file__)),
        "inputs_completion_sha256": sha(inputs / "completion.json"),
        "cohort_completion_sha256": sha(cohort / "completion.json"),
        "symbols": len(records),
        "records": counts,
        "unqualified_marks": unknown,
        "unqualified_marks_before_first_candidate_only": True,
        "past_seven_day_anchors": anchors,
        "max_decimal_sum_error": max_error,
        "actual_historical_schedule_certified": False,
        "payable_cashflows_certified": False,
        "new_financial_accounts": 0,
        "profit_goal_met": False,
        "symbol_reports": records,
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print({k: v for k, v in result.items() if k != "symbol_reports"}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--cohort", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    audit(a.inputs, a.cohort, a.output)
