"""Freeze and evaluate visible deadline screens without computing account returns."""

import argparse
from pathlib import Path

import pandas as pd

from factors.crypto.matched_spot_carry_horizon import covered_cost, horizon
from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest


def prepare(plan_path, output):
    plan = read(plan_path)
    for key in [
        "parent_registration",
        "daily_sources",
        "daily_audit",
        "boundaries",
        "perpetual_registry",
    ]:
        verify_manifest(Path(plan[key]))
    output.mkdir(parents=True, exist_ok=False)
    paths = [
        plan_path,
        Path(__file__),
        Path("factors/crypto/matched_spot_carry_horizon.py"),
        *[
            Path(plan[k]) / "completion.json"
            for k in [
                "parent_registration",
                "daily_sources",
                "daily_audit",
                "boundaries",
                "perpetual_registry",
            ]
        ],
    ]
    save(
        output / "registration.json",
        {
            "source_hashes": {str(p): sha(p) for p in paths},
            "before_new_account_returns": True,
            "parameter_grid": False,
        },
    )
    save(output / "plan.json", plan)
    parent = read(Path(plan["parent_plan"]))
    rows = read(Path(plan["daily_audit"]) / "slot_data_feasibility.json")
    perp = read(Path(plan["perpetual_registry"]) / "registry.json")["cards"]
    spot = read(Path(plan["boundaries"]) / "boundaries.json")
    clocks = {}
    for r in rows:
        clocks.setdefault((r["variant"], r["window"]), set()).add(pd.Timestamp(r["decision"]))
    ends = {w["id"]: pd.Timestamp(w["end"], tz="UTC") for w in parent["windows"]}
    result, checks = [], []
    for r in rows:
        t = pd.Timestamp(r["decision"])
        following = sorted(x for x in clocks[(r["variant"], r["window"])] if x > t)
        next_clock = following[0] if following else ends[r["window"]]
        info = horizon(r["symbol"], t, next_clock, ends[r["window"]], perp, spot)
        covered = covered_cost(r.get("known_rate_sum"), info["remaining_hours"])
        result.append(
            {
                **r,
                **info,
                "horizon_forecast_rate": covered["forecast_rate"],
                "horizon_guarded_data_only_candidate": bool(
                    r["data_eligible"] and covered["cost_guard_passed"]
                ),
                "actual_target": None,
            }
        )
        # Physical removal of every not-yet-visible card must not change this decision.
        filtered_perp = [x for x in perp if pd.Timestamp(x["available_at_assumed"]) <= t]
        filtered_spot = [x for x in spot if pd.Timestamp(x["spot_available_at_assumed"]) <= t]
        if (
            horizon(r["symbol"], t, next_clock, ends[r["window"]], filtered_perp, filtered_spot)
            != info
        ):
            raise ValueError("Future notice changes a past horizon")
    # Explicitly assert a real knownBNX deadline is invisible before its publication.
    bnx = next(x for x in perp if x["symbol"] == "BNXUSDT")
    known = pd.Timestamp(bnx["available_at_assumed"])
    for t in [known - pd.Timedelta(seconds=1), known]:
        info = horizon(
            "BNXUSDT", t, known + pd.Timedelta(days=7), known + pd.Timedelta(days=30), [bnx], []
        )
        checks.append({"decision": t.isoformat(), "visible_deadlines": info["visible_deadlines"]})
    primary = [
        r
        for r in result
        if r["variant"] == "daily_archive" and r["window"] == "continuous2025_2026"
    ]
    old = [r for r in primary if r["guarded_data_only_candidate"]]
    save(output / "slots.json", result)
    save(
        output / "summary.json",
        {
            "schema": "matched-carry-horizon-inputs.v1",
            "slot_rows": len(result),
            "causal_notice_removal_checks": len(result),
            "notice_boundary_checks": checks,
            "primary_old_guarded_slots": len(old),
            "primary_horizon_guarded_slots": sum(
                r["horizon_guarded_data_only_candidate"] for r in primary
            ),
            "all_horizon_guarded_slots": sum(
                r["horizon_guarded_data_only_candidate"] for r in result
            ),
            "old_guarded_details": [
                {
                    k: r[k]
                    for k in [
                        "symbol",
                        "decision",
                        "known_rate_sum",
                        "remaining_hours",
                        "horizon_forecast_rate",
                        "horizon_guarded_data_only_candidate",
                    ]
                }
                for r in old
            ],
            "financial_accounts": 0,
            "profit_goal_met": False,
            "portfolio_actual": None,
        },
    )
    save(
        output / "completion.json",
        {
            "manifest": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
            "financial_accounts": 0,
        },
    )
    print(read(output / "summary.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.plan, args.output)
