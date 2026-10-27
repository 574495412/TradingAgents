"""Extend only source windows; retain the original exit instruction and all prices."""

import argparse
import os
from pathlib import Path

import pandas as pd

from research_tools.carry_case_review.audit_inputs_v2 import check_csv
from research_tools.carry_case_review.prepare_v3 import parse
from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest


def prepare(amendment, output, review):
    change = read(amendment)
    base = Path(change["base_source"])
    run = Path(change["base_run"])
    prior = Path(change["base_input_review"])
    for p in [base, run, prior]:
        verify_manifest(p)
    plan = read(base / "plan.json")
    cards = read(Path(plan["parent_boundaries"]) / "boundaries.json")
    output.mkdir(parents=True, exist_ok=False)
    for name in ["raw", "symbols", "funding"]:
        (output / name).mkdir()
    paths = [
        amendment,
        Path(__file__),
        Path(change["engine"]),
        Path("research_tools/carry_case_review/prepare_v3.py"),
        Path("research_tools/carry_case_review/audit_inputs_v2.py"),
        base / "completion.json",
        run / "completion.json",
        prior / "completion.json",
    ]
    save(
        output / "registration.json",
        {
            "source_hashes": {str(p): sha(p) for p in paths},
            "before_buffer_case_results": True,
            "new_http_requests": 0,
        },
    )
    for p in (base / "raw").rglob("*"):
        if p.is_file():
            target = output / "raw" / p.relative_to(base / "raw")
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(p, target)
    cases = []
    checks = []
    for original in plan["cases"]:
        s = original["symbol"]
        t = pd.Timestamp(original["decision"])
        card = next(r for r in cards if r["symbol"] == s)
        deadlines = []
        if pd.Timestamp(card["spot_available_at_assumed"]) <= t:
            deadlines.append(pd.Timestamp(card["spot_stop_at"]))
        if pd.Timestamp(card["perpetual_available_at_assumed"]) <= t:
            deadlines.append(pd.Timestamp(card["perpetual_stop_at"]))
        end = min(deadlines) - pd.Timedelta(minutes=5)
        case = {
            **original,
            "original_case_deadline": original["exit_deadline"],
            "exit_request_at": (
                pd.Timestamp(original["exit_deadline"]) - pd.Timedelta(minutes=5)
            ).isoformat(),
            "exit_deadline": end.isoformat(),
            "known_market_cessation": min(deadlines).isoformat(),
        }
        cases.append(case)
        counts = {}
        for market in ["spot", "perpetual", "mark"]:
            archive = output / "raw" / market / f"{s}-5m-{case['month']}.zip"
            f = parse(archive.read_bytes(), market, case)
            f.to_parquet(output / "symbols" / f"{s}_{market}_5m.parquet")
            counts[market] = check_csv(output, case, market)
            old = pd.read_parquet(base / "symbols" / f"{s}_{market}_5m.parquet")
            pd.testing.assert_frame_equal(
                f.loc[old.index], old, check_exact=True, check_dtype=False, check_freq=False
            )
        rates = pd.read_parquet(Path(plan["parent_bundle"]) / "funding" / (s + "_exact.parquet"))
        rates = rates.loc[(rates.index >= t - pd.Timedelta(days=7)) & (rates.index <= end)]
        raw = []
        for p in sorted((base / "funding").glob(s + "_page*.json")):
            os.link(p, output / "funding" / p.name)
            raw.extend(read(p)["response"]["structuredContent"]["result"])
        selected = [
            r
            for r in raw
            if t - pd.Timedelta(days=7)
            <= pd.Timestamp(r["fundingTime"], unit="ms", tz="UTC")
            <= end
        ]
        selected.sort(key=lambda r: r["fundingTime"])
        clock = pd.DatetimeIndex(
            pd.to_datetime([r["fundingTime"] for r in selected], unit="ms", utc=True)
        )
        if not clock.equals(rates.index) or any(
            float(r["fundingRate"]) != q.funding_rate or float(r["markPrice"]) != q.mark
            for r, (_, q) in zip(selected, rates.iterrows(), strict=True)
        ):
            raise ValueError("Extended funding differs from retained raw responses")
        rates.to_parquet(output / "funding" / (s + "_exact.parquet"))
        checks.append(
            {
                "symbol": s,
                "raw_source_rows": counts,
                "original_prefix_unchanged": True,
                "original_exit_request_at": case["exit_request_at"],
                "buffer_end": end.isoformat(),
                "funding_events_including_past7": len(rates),
                "case_identity_reused_from_review": sha(prior / "completion.json"),
            }
        )
    plan = {**plan, "cases": cases, "engine": change["engine"], "buffer_counterfactual": change}
    save(output / "plan.json", plan)
    save(
        output / "completion.json",
        {
            "manifest": {
                str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()
            },
            "new_http_requests": 0,
            "financial_case_paths": 0,
        },
    )
    review.mkdir(parents=True, exist_ok=False)
    save(
        review / "verification.json",
        {
            "schema": "carry-case-buffer-input-review.v1",
            "passed": True,
            "source_completion_sha256": sha(output / "completion.json"),
            "source_sha256": sha(Path(__file__)),
            "checks": checks,
            "baseline_input_review_sha256": sha(prior / "completion.json"),
            "external_third_party_review": False,
            "financial_case_paths": 0,
            "profit_goal_met": False,
        },
    )
    save(
        review / "completion.json",
        {
            "manifest": {"verification.json": sha(review / "verification.json")},
            "financial_case_paths": 0,
        },
    )
    print({"cases": checks, "new_http_requests": 0, "financial_case_paths": 0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.plan, args.output, args.review)
