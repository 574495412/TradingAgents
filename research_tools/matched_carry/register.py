"""Register matched short slots and daily data requirements before new spot returns."""

import argparse
from pathlib import Path

import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest

EXCLUDED = {"1000BONKUSDT", "1000PEPEUSDT", "1000SHIBUSDT"}


def register(plan_path, output):
    plan = read(plan_path)
    parents = [
        Path(plan[k])
        for k in ["parent_targets", "parent_cohort", "parent_bundle", "failure_report"]
    ]
    for p in parents:
        verify_manifest(p)
    output.mkdir(parents=True, exist_ok=False)
    anchors = [r for r in read(parents[0] / "anchors.json") if r["candidate"] == "funding7_raw"]
    slots, clocks = [], {}
    for r in anchors:
        names = sorted(s for s, w in r["book"].items() if w < 0)
        if len(names) not in [0, 4]:
            raise ValueError("Frozen funding source does not contain zero/four short slots")
        slots.append({k: r[k] for k in ["variant", "window", "decision"]} | {"slots": names})
        for s in names:
            clocks.setdefault(s, []).append(pd.Timestamp(r["decision"]))
    end = max(pd.Timestamp(w["end"], tz="UTC") for w in plan["windows"])
    requests = []
    for s, times in sorted(clocks.items()):
        if s in EXCLUDED:
            continue
        first = (min(times) - pd.Timedelta(days=29)).strftime("%Y-%m")
        # Retain real prices until the full end in case an exit is delayed.
        months = pd.period_range(first, (end - pd.Timedelta(days=1)).strftime("%Y-%m"), freq="M")
        requests.extend({"symbol": s, "month": str(m)} for m in months)
    paths = [
        plan_path,
        Path(__file__),
        Path(__file__).with_name("collect_daily.py"),
        Path(plan["baseline_carry_plan"]),
        Path("research_tools/factor_lab/io.py"),
        *[p / "completion.json" for p in parents],
    ]
    save(output / "plan.json", plan)
    save(output / "slots.json", slots)
    save(output / "daily_requests.json", requests)
    save(
        output / "registration.json",
        {
            "source_hashes": {str(p): sha(p) for p in paths},
            "slot_anchors": len(slots),
            "all_slot_symbols": sorted(clocks),
            "excluded_unit_mismatch_symbols": sorted(EXCLUDED),
            "daily_symbols": len({r["symbol"] for r in requests}),
            "daily_requests": len(requests),
            "financial_returns_computed": False,
            "spot_identity_certified": False,
            "profit_goal_met": False,
        },
    )
    save(
        output / "completion.json",
        {
            "manifest": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
            "financial_returns_computed": False,
        },
    )
    print(read(output / "registration.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    register(args.plan, args.output)
