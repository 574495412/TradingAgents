"""Freeze fixed-family weekly targets on the audited identity-aware source cohorts."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from factors.common.max_effect import book as max_book
from factors.crypto.funding_cross_section import accrued7, book as funding_book
from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest
from research_tools.sparse_market.cohort import daily_versions, feature_views
from research_tools.sparse_market.identity import crypto_context


def current_rows(raw, registry, clock, members):
    subset = {s: raw[s] for s in ["BTCUSDT", *members]}
    local_registry = {
        **registry,
        "launch_cards": [r for r in registry["launch_cards"] if r["symbol"] in subset],
    }
    base, episodes = feature_views(subset, local_registry)
    return {
        s: (
            episodes[(s, c["feature_start"])].loc[clock]
            if c["feature_start"]
            else base[s].loc[clock]
        )
        for s in subset
        for c in [crypto_context(registry, s, clock)]
    }


def weights(rows, rates, clock, parameters, candidate):
    if candidate.startswith("funding7"):
        values = accrued7({s: rates[s] for s in rows if s != "BTCUSDT"}, clock)
        return funding_book(rows, values, parameters, candidate)
    return max_book(rows, parameters, candidate)


def prepare(cohort, bundle, output):
    output.mkdir(parents=True, exist_ok=False)
    for p in [cohort, bundle]:
        verify_manifest(p)
    checked = cohort.with_name(cohort.name + "_independent_audit.json")
    if not read(checked)["passed"] or read(checked)["cohort_completion_sha256"] != sha(
        cohort / "completion.json"
    ):
        raise ValueError("Final cohort independent audit missing or changed")
    plan = read(cohort / "plan.json")
    registry = read(cohort / "registry.json")
    symbols = read(cohort / "symbols.json")
    parameters_path = Path("results/research/max_effect_inputs_20261026_v002/plan.json")
    parameters = read(parameters_path)["parameters"]
    fq = Path("../freqtrade").resolve()
    paths = [
        Path(__file__),
        Path(__file__).with_name("cohort.py"),
        Path(__file__).with_name("identity.py"),
        Path("factors/common/max_effect.py"),
        Path("factors/crypto/funding_cross_section.py"),
        fq / "research/crypto/sparse_calendar.py",
        fq / "research/crypto/sparse_sources.py",
        cohort / "completion.json",
        bundle / "completion.json",
        checked,
        parameters_path,
    ]
    plan = {
        **plan,
        "cohort": str(cohort),
        "bundle": str(bundle),
        "parameters": parameters,
        "engine": str(fq / "research/crypto/sparse_calendar.py"),
        "source_hashes": {str(p): sha(p) for p in paths},
        "before_targets_and_returns": True,
    }
    save(output / "plan.json", plan)
    save(
        output / "registration.json",
        {"plan_sha256": sha(output / "plan.json"), "financial_returns_computed": False},
    )
    rates = {s: pd.read_parquet(bundle / "funding" / (s + "_exact.parquet")) for s in symbols}
    versions, replaced = daily_versions(plan)
    anchors, checks = [], []
    for variant, raw in versions.items():
        records = {
            pd.Timestamp(r["decision"]): r for r in read(cohort / (variant + "_cohort.json"))
        }
        all_base, all_episodes = feature_views(raw, registry)
        for window in plan["windows"]:
            start, end = [pd.Timestamp(window[k], tz="UTC") for k in ["start", "end"]]
            calendar = pd.date_range(start, end, freq="h", inclusive="left")
            clocks = sorted({start, *pd.date_range(start, end, freq="W-MON", inclusive="left")})
            target_books = {c: [] for c in plan["candidates"] if c != "cash"}
            for clock in clocks:
                members = records[clock]["selected"]
                feature_rows = {}
                for s in ["BTCUSDT", *members]:
                    context = crypto_context(registry, s, clock)
                    if context["blocked"]:
                        raise ValueError(
                            "Frozen member now has a visible identity/type restriction"
                        )
                    feature_rows[s] = (
                        all_episodes[(s, context["feature_start"])].loc[clock]
                        if context["feature_start"]
                        else all_base[s].loc[clock]
                    )
                for candidate, books in target_books.items():
                    w, meta = weights(feature_rows, rates, clock, parameters, candidate)
                    if set(w.index[w.ne(0)]) - set(members):
                        raise ValueError("Target escapes the past-information candidate set")
                    books.append(w.reindex(symbols, fill_value=0.0))
                    anchors.append(
                        {
                            "variant": variant,
                            "window": window["id"],
                            "decision": clock.isoformat(),
                            "candidate": candidate,
                            "selected": members,
                            "book": {s: float(v) for s, v in w.items() if v},
                            "details": meta,
                        }
                    )
            for candidate, books in target_books.items():
                frame = pd.DataFrame(books, index=clocks).reindex(calendar).ffill()
                frame.columns = ["target_" + s for s in frame.columns]
                frame["risk_scale"], frame["rebalance"] = 1.0, frame.index.isin(clocks)
                frame.to_parquet(output / f"{variant}_{window['id']}_{candidate}_targets.parquet")
        for clock in [
            pd.Timestamp(t, tz="UTC") for t in ["2025-01-01", "2025-06-02", "2026-07-06"]
        ]:
            members = records[clock]["selected"]
            chosen = {s: raw[s] for s in ["BTCUSDT", *members]}
            changed = {s: f.copy() for s, f in chosen.items()}
            for f in changed.values():
                f.loc[f.index >= clock, "close"] *= 3.0
                f.loc[f.index >= clock, "quote_volume"] *= 17.0
            baseline = current_rows(chosen, registry, clock, members)
            perturbed = current_rows(changed, registry, clock, members)
            for s in baseline:
                np.testing.assert_allclose(
                    baseline[s], perturbed[s], rtol=0, atol=0, equal_nan=True
                )
            future_rates = {s: f.copy() for s, f in rates.items()}
            for f in future_rates.values():
                f.loc[f.index >= clock, "funding_rate"] *= -13.0
            for candidate in [c for c in plan["candidates"] if c != "cash"]:
                left, _ = weights(baseline, rates, clock, parameters, candidate)
                right, _ = weights(perturbed, future_rates, clock, parameters, candidate)
                pd.testing.assert_series_equal(left, right)
            checks.append(
                {
                    "variant": variant,
                    "clock": clock.isoformat(),
                    "members": len(members),
                    "future_price_quote_funding_perturbation": True,
                    "all_four_books_unchanged": True,
                }
            )
        print({"variant_targets_frozen": variant}, flush=True)
    for window in plan["windows"]:
        calendar = pd.date_range(
            window["start"], window["end"], freq="h", inclusive="left", tz="UTC"
        )
        cash = pd.DataFrame(0.0, index=calendar, columns=["target_" + s for s in symbols])
        cash["risk_scale"], cash["rebalance"] = 1.0, False
        cash.iloc[0, cash.columns.get_loc("rebalance")] = True
        cash.to_parquet(output / ("cash_" + window["id"] + "_targets.parquet"))
    save(output / "anchors.json", anchors)
    save(output / "causality_checks.json", checks)
    save(output / "lifecycle_directives.json", read(cohort / "lifecycle_directives.json"))
    save(
        output / "summary.json",
        {
            "schema": "sparse-weekly-targets.v1",
            "symbols": len(symbols),
            "anchor_source_candidate_cells": len(anchors),
            "future_perturbation_cases": len(checks),
            "source_reconciliation_days": len(replaced),
            "maximum_base_accounts": plan["budget"]["maximum_financial_base_accounts"],
            "cash_accounts": 6,
            "financial_accounts_computed": 0,
            "profit_goal_met": False,
        },
    )
    save(
        output / "completion.json",
        {
            "manifest": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
            "financial_accounts": 0,
        },
    )
    print(read(output / "summary.json"), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    prepare(a.cohort, a.bundle, a.output)
