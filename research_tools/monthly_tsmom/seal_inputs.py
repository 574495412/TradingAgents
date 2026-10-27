"""Seal captured raw funding and build diagnostic monthly economic-index observations."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from factors.common.monthly_tsmom import monthly_observation
from factors.crypto.perpetual_returns import daily_long_index_returns
from research_tools.expanded_market.funding_inputs import validate_page
from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest
from research_tools.sparse_market.audit_final_cohort import prepare_raw


def seal(readiness, capture, output):
    output.mkdir(parents=True, exist_ok=False)
    verify_manifest(readiness)
    plan = read(readiness / "plan.json")
    reg = read(capture / "registration.json")
    assert reg["readiness_completion_sha256"] == sha(readiness / "completion.json")
    collection = read(capture / "collection_summary.json")
    assert collection["api_calls"] <= plan["raw_capture"]["maximum_api_calls"]
    assert reg["tasks"] == read(readiness / "capture_tasks.json")
    bindings = read(Path(plan["existing_bindings"]))
    reports = {r["symbol"]: r for r in collection["reports"]}
    all_funding, source_checks = {}, []
    (output / "funding").mkdir()
    source_paths = [
        Path(__file__),
        Path(__file__).with_name("plan.json"),
        Path("factors/common/monthly_tsmom.py"),
        Path("factors/crypto/perpetual_returns.py"),
        Path("research_tools/expanded_market/funding_inputs.py"),
        readiness / "completion.json",
        capture / "registration.json",
        capture / "collection_summary.json",
    ]
    for row in read(readiness / "funding_coverage.json"):
        s = row["symbol"]
        old_path = Path(bindings[s]["funding_path"])
        assert sha(old_path) == bindings[s]["funding_sha256"]
        old = pd.read_parquet(old_path)[["funding_rate", "mark"]]
        task = next((t for t in reg["tasks"] if t["symbol"] == s), None)
        records = []
        if task:
            report = reports[s]
            assert report["collection_complete"] and not report["error"]
            cursor = int(pd.Timestamp(task["start"]).timestamp() * 1000)
            end = int(pd.Timestamp(task["end_exclusive"]).timestamp() * 1000) - 1
            paths = sorted((capture / "raw").glob(s + "_page*.json"))
            assert len(paths) == report["pages"]
            for n, path in enumerate(paths):
                rows = validate_page(read(path), s, cursor, end)
                assert len(rows) == 1000 if n < len(paths) - 1 else len(rows) < 1000
                records += rows
                if rows:
                    cursor = rows[-1]["fundingTime"] + 1
                source_paths.append(path)
            assert len(records) == report["records"]
            early = pd.DataFrame(
                {
                    "funding_rate": [float(r["fundingRate"]) for r in records],
                    "mark": [float(r["markPrice"]) for r in records],
                },
                index=pd.DatetimeIndex(
                    pd.to_datetime([r["fundingTime"] for r in records], unit="ms", utc=True),
                    name="date",
                ),
            )
            f = pd.concat([early, old])
        else:
            f = old.copy()
        assert f.index.is_monotonic_increasing and not f.index.has_duplicates
        f["rate_time_usable"] = np.isfinite(f.funding_rate)
        f["held_cash_mark_usable"] = np.isfinite(f.mark) & f.mark.gt(0)
        f["payable_cashflow_certified"] = False
        f.to_parquet(output / "funding" / (s + "_exact.parquet"))
        all_funding[s] = f
        source_paths.append(old_path)
        source_checks.append(
            {
                "symbol": s,
                "new_records": len(records),
                "combined_events": len(f),
                "bad_marks": int((~f.held_cash_mark_usable).sum()),
                "first": f.index[0].isoformat(),
                "last": f.index[-1].isoformat(),
                "observed_max_gap_hours": float(
                    f.index.to_series().diff().max().total_seconds() / 3600
                ),
                "historical_schedule_certified": False,
            }
        )
    assert sum(r["pages"] for r in collection["reports"]) == collection["api_calls"]
    cohort = Path(plan["cohort"])
    old_plan = read(cohort / "plan.json")
    versions = prepare_raw(old_plan, cohort)
    observations, unknowns, causes = [], [], []
    indexes = {}
    (output / "daily_indexes").mkdir()
    for variant, raw in versions.items():
        indexes[variant] = {}
        for s, funding in all_funding.items():
            idx = daily_long_index_returns(raw[s], funding)
            indexes[variant][s] = idx
            idx.to_parquet(output / "daily_indexes" / f"{variant}_{s}.parquet")
        for m in read(readiness / "monthly_membership.json"):
            if m["variant"] != variant:
                continue
            for s in m["selected"]:
                try:
                    obs = monthly_observation(indexes[variant][s].economic_return, m["decision"])
                except ValueError as error:
                    unknowns.append(
                        {
                            "variant": variant,
                            "symbol": s,
                            "decision": m["decision"],
                            "reason": str(error),
                        }
                    )
                    continue
                price = m["price_only_12month_return_reference"][s]
                observations.append(
                    {
                        **obs,
                        "variant": variant,
                        "symbol": s,
                        "price_only_return_reference": price,
                        "direction_differs_from_price": obs["direction"] != int(np.sign(price)),
                    }
                )
        for cut in [pd.Timestamp("2025-07-01T00:00:00Z"), pd.Timestamp("2026-04-01T00:00:00Z")]:
            changed_funding = all_funding["BTCUSDT"].copy()
            changed_funding.loc[changed_funding.index >= cut, "funding_rate"] *= -100
            changed_daily = raw["BTCUSDT"].copy()
            changed_daily.loc[changed_daily.index >= cut, ["open", "high", "low", "close"]] *= 19
            actual = daily_long_index_returns(changed_daily, changed_funding)
            pd.testing.assert_frame_equal(
                indexes[variant]["BTCUSDT"].loc[lambda f, at=cut: f.index < at],
                actual.loc[lambda f, at=cut: f.index < at],
            )
            assert monthly_observation(actual.economic_return, cut) == monthly_observation(
                indexes[variant]["BTCUSDT"].economic_return, cut
            )
            causes.append(
                {
                    "variant": variant,
                    "cut": cut.isoformat(),
                    "future_price_and_funding_do_not_change_past": True,
                }
            )
    save(output / "plan.json", plan)
    save(output / "source_checks.json", source_checks)
    save(output / "observations.json", observations)
    save(output / "unknown_observations.json", unknowns)
    save(output / "causality.json", causes)
    save(
        output / "registration.json",
        {"source_hashes": {str(p): sha(p) for p in source_paths}, "new_financial_accounts": 0},
    )
    summary = {
        "schema": "monthly-tsmom-economic-inputs.v1",
        "selected_symbols": len(all_funding),
        "new_api_calls": collection["api_calls"],
        "new_raw_events": sum(r["records"] for r in collection["reports"]),
        "monthly_source_observations": len(observations),
        "unknown_observations": len(unknowns),
        "price_vs_economic_direction_differences": sum(
            r["direction_differs_from_price"] for r in observations
        ),
        "source_variants_are_independent": False,
        "internal_formula_review_performed": False,
        "new_financial_accounts": 0,
        "profit_goal_met": False,
        "historical_schedule_certified": False,
        "customer_cash_certified": False,
        "independent_oos": False,
        "external_review": False,
    }
    save(output / "summary.json", summary)
    save(
        output / "completion.json",
        {
            "passed": True,
            "manifest": {
                str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()
            },
            "new_financial_accounts": 0,
        },
    )
    print(summary, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--readiness", type=Path, required=True)
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    seal(a.readiness, a.capture, a.output)
