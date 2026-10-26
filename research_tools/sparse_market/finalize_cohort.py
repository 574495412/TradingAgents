"""Finalize both feature-source cohorts after bounded additions and type closure."""

import argparse
from pathlib import Path

import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest
from research_tools.sparse_market.cohort import daily_versions, select
from research_tools.sparse_market.identity import directives


def prepare(bundle, identity, output):
    output.mkdir(parents=True, exist_ok=False)
    plan_path = Path(__file__).with_name("plan.json")
    plan = {**read(plan_path), "parent_market": str(bundle)}
    for p in [bundle, identity, Path(plan["parent_daily"]), Path(plan["parent_cohort"])]:
        verify_manifest(p)
    paths = [
        plan_path,
        Path(__file__),
        Path(__file__).with_name("cohort.py"),
        Path(__file__).with_name("identity.py"),
        Path("factors/common/max_effect.py"),
        bundle / "completion.json",
        identity / "completion.json",
    ]
    save(
        output / "registration.json",
        {
            "source_hashes": {str(p): sha(p) for p in paths},
            "first_daily_label_policy": "exclude_first_observed_daily_row_conservatively",
            "final_source_roster_before_returns": True,
            "financial_accounts": 0,
        },
    )
    save(output / "plan.json", plan)
    registry = read(identity / "registry.json")
    parent = read(Path(plan["parent_cohort"]) / "cohort.json")
    clocks = [pd.Timestamp(r["decision"]) for r in parent]
    versions, replacements = daily_versions(plan)
    union, comparisons = set(), []
    for variant, raw in versions.items():
        rows, names = select(raw, registry, clocks)
        union |= names
        save(output / (variant + "_cohort.json"), rows)
        for old, row in zip(parent, rows, strict=True):
            removed = sorted(set(old["selected"]) - set(row["selected"]))
            added = sorted(set(row["selected"]) - set(old["selected"]))
            if removed or added:
                comparisons.append(
                    {
                        "variant": variant,
                        "decision": row["decision"],
                        "removed": removed,
                        "added": added,
                    }
                )
        print({"variant": variant, "symbols": len(names), "decisions": len(rows)}, flush=True)
    available = {
        p.name.removesuffix("_5m.parquet") for p in (bundle / "symbols").glob("*_5m.parquet")
    }
    if (
        not union.issubset(available)
        or len(union) > plan["budget"]["maximum_total_acquired_symbols"]
    ):
        raise ValueError(
            "Final source-directed universe still has missing sources or exceeds budget"
        )
    save(output / "symbols.json", sorted(union))
    save(output / "registry.json", registry)
    save(output / "lifecycle_directives.json", directives(registry, sorted(union)))
    save(output / "reconciliation_days.json", replacements)
    save(output / "cohort_changes.json", comparisons)
    summary = {
        "schema": "sparse-identity-cohort-final.v1",
        "symbols": len(union),
        "decisions_per_variant": len(clocks),
        "source_variants": list(versions),
        "changed_clock_source_cells": len(comparisons),
        "complete_day_replacements": len(replacements),
        "type_restrictions": len(registry["type_cards"]),
        "end_cards": len(registry["cards"]),
        "launch_cards": len(registry["launch_cards"]),
        "known_metadata_correction": "prior v002 summary retained legacy type_restrictions=5; actual sealed registry contained12 and selection used all12. This final summary derives count from registry.",
        "financial_accounts": 0,
        "historical_original_versions_certified": False,
        "profit_goal_met": False,
    }
    save(output / "summary.json", summary)
    save(
        output / "completion.json",
        {
            "manifest": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
            "financial_accounts": 0,
        },
    )
    print(summary, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--identity", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    prepare(a.bundle, a.identity, a.output)
