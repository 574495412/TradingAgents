"""Freeze one bounded refinement round, keep selection ahead of diagnostic evaluation."""

import argparse
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from factors.portfolio.recipes import BALANCED_REVIEW_20261025
from research_tools.factor_lab.io import read, save, sha
from research_tools.factor_lab.study import (
    account,
    compose_score,
    executor,
    load_market,
    output_account,
)
from research_tools.factor_lab_audit import audit_account

from .compare import aggregate, compare


def selection(rows, root, plan):
    assessments = []
    for candidate in plan["candidates"]:
        cases = []
        reasons = []
        for cost in plan["costs_slippage_bps"]:
            sample = [
                r
                for r in rows
                if r["candidate"] == candidate
                and r["slippage_bps"] == cost
                and r["window"] in plan["validation_windows"]
            ]
            if len(sample) != 4:
                raise ValueError("Incomplete validation grid")
            summary = aggregate(sample, root)
            cases.append({"slippage_bps": cost, **summary})
            if summary["mean_window_return"] <= 0:
                reasons.append(f"{cost}:nonpositive_mean_return")
            if summary["positive_windows"] < 3:
                reasons.append(f"{cost}:insufficient_positive_windows")
            if summary["worst_observed_drawdown"] > 0.08:
                reasons.append(f"{cost}:drawdown_exceeded")
            if summary["complete_trades"] < 30:
                reasons.append(f"{cost}:insufficient_complete_positions")
            if (
                summary["pooled_complete_profit_factor"] is None
                or summary["pooled_complete_profit_factor"] < 1.1
            ):
                reasons.append(f"{cost}:closed_profit_factor_below_target")
        assessments.append(
            {"candidate": candidate, "cases": cases, "passed": not reasons, "reasons": reasons}
        )
    eligible = [r for r in assessments if r["passed"]]
    chosen = (
        max(
            eligible,
            key=lambda r: (
                min(c["mean_window_return"] for c in r["cases"]),
                -max(c["worst_observed_drawdown"] for c in r["cases"]),
            ),
        )["candidate"]
        if eligible
        else "cash"
    )
    return {
        "selected_candidate": chosen,
        "assessments": assessments,
        "selection_inputs": "2025validation only; hypotheses themselves are post-review",
        "additional_or_2026_results_included": False,
        "production_eligible": False,
    }


def run(root):  # noqa: C901
    started = time.monotonic()
    root.mkdir(parents=True, exist_ok=False)
    plan = read(Path(__file__).with_name("plan.json"))
    parent = Path(plan["parent"])
    original = read(parent / "plan.json")
    parent_audit = read(Path(plan["parent_audit"]))
    if not parent_audit["passed"] or parent_audit["source_catalog_sha256"] != sha(
        parent / "catalog.json"
    ):
        raise ValueError("Parent independent audit changed")
    for name, expected in original["source_identities"].items():
        if sha(Path(name)) != expected:
            raise ValueError("Parent source changed")
    plan["registered_at"] = datetime.now(UTC).isoformat()
    plan["source_hashes"] = {str(p.resolve()): sha(p) for p in Path(__file__).parent.glob("*.py")}
    for p in [
        Path(plan["execution_source"]),
        Path("factors/portfolio/recipes.py"),
        parent / "catalog.json",
        parent / "predictions.parquet",
        parent / "standardized.parquet",
        parent / "features.parquet",
        Path(plan["parent_audit"]),
    ]:
        plan["source_hashes"][str(p.resolve())] = sha(p)
    save(root / "plan.json", plan)
    save(
        root / "registration.json",
        {
            "plan_sha256": sha(root / "plan.json"),
            "parent_catalog_sha256": sha(parent / "catalog.json"),
        },
    )
    snapshot = root / "source_snapshot"
    snapshot.mkdir()
    for source in plan["source_hashes"]:
        p = Path(source)
        if p.suffix == ".py":
            shutil.copyfile(p, snapshot / p.name)
    old = compare(parent)
    save(root / "original_comparison.json", old)
    predictions = pd.read_parquet(parent / "predictions.parquet")
    z = pd.read_parquet(parent / "standardized.parquet")
    features = pd.read_parquet(parent / "features.parquet")
    norm = read(parent / "normalization.json")
    score_full = predictions.full_24h

    def centered(score):
        frame = score.unstack("symbol")
        return frame.sub(frame.mean(axis=1), axis=0).stack(future_stack=True).rename("score")

    balanced = compose_score(z, BALANCED_REVIEW_20261025, norm["label_scale"])
    scores = {
        "full_budget40": score_full,
        "full_hurdle40": score_full,
        "full_trend40": score_full.where(np.sign(score_full) == np.sign(features.mom_168), 0),
        "full_neutral40": centered(score_full),
        "balanced6_neutral40": centered(balanced),
    }
    for factor, direction in BALANCED_REVIEW_20261025.items():
        scores["single_" + factor] = centered(z[factor] * direction * norm["label_scale"])
    data, rates = load_market(parent)
    engine = executor(plan["execution_source"])
    rows = []

    def one(candidate, window, cost, capital=10000, phase=None):
        config = {
            **original["default_account"],
            "gross": 0.4,
            "capital": capital,
            "slippage_bps": cost,
            "neutral_target": candidate.endswith("neutral40") or candidate.startswith("single_"),
        }
        if candidate == "full_hurdle40":
            config["threshold"] = 0.0032
        a = account(parent, engine, data, rates, scores[candidate], window, config)
        q = a["curve"]
        a["metrics"]["mean_absolute_net_exposure"] = float((q.net_exposure / q.equity).abs().mean())
        a["metrics"]["maximum_absolute_net_exposure"] = float(
            (q.net_exposure / q.equity).abs().max()
        )
        key = f"{phase or window['phase']}_{window['id']}_{candidate}_cap{capital}_slip{cost}"
        result = output_account(root / "accounts" / key, a, config, window)
        row = {
            "id": key,
            "candidate": candidate,
            "preset": candidate,
            "window": window["id"],
            "phase": phase or window["phase"],
            "capital": capital,
            "slippage_bps": cost,
            "metrics": a["metrics"],
            "capacity": a["capacity"],
            "result_sha256": sha(root / "accounts" / key / "result.json"),
        }
        rows.append(row)
        print(
            candidate, window["id"], cost, capital, round(a["metrics"]["net_return"], 6), flush=True
        )
        return result

    for w in original["test_windows"]:
        if w["id"] not in plan["validation_windows"]:
            continue
        for candidate in plan["candidates"]:
            for cost in plan["costs_slippage_bps"]:
                one(candidate, w, cost)
    choice = selection(rows, root, plan)
    save(root / "selection.json", choice)
    # Fixed single-factor experiments do not influence selection of the candidates above.
    for factor in plan["single_factor_diagnostics"]:
        for w in original["test_windows"]:
            if w["id"] in plan["validation_windows"]:
                one("single_" + factor, w, 3, phase="single_factor_diagnostic")
    for w in original["test_windows"]:
        if w["id"] in plan["validation_windows"]:
            continue
        for candidate in plan["candidates"]:
            for cost in plan["costs_slippage_bps"]:
                one(candidate, w, cost, phase="reused_2026_diagnostic")
    selected = choice["selected_candidate"]
    diagnostic = "full_neutral40" if selected == "cash" else selected
    follow = sorted({"full_budget40", diagnostic})
    for w in plan["additional_windows"]:
        for candidate in follow:
            for cost in plan["costs_slippage_bps"]:
                one(candidate, w, cost)
    stress = next(w for w in original["test_windows"] if w["id"] == "2025Q2")
    for candidate in follow:
        for capital in (1000000, 10000000):
            for cost in plan["costs_slippage_bps"]:
                one(candidate, stress, cost, capital, phase="capacity")
    save(root / "accounts.json", rows)
    audits = [audit_account(root, row, data, rates) for row in rows]
    groups = []
    for stage in (
        "validation",
        "reused_2026_diagnostic",
        "additional_evaluation",
        "single_factor_diagnostic",
        "capacity",
    ):
        candidates = sorted({r["candidate"] for r in rows if r["phase"] == stage})
        for candidate in candidates:
            for cost in (3, 10):
                subset = [
                    r
                    for r in rows
                    if r["phase"] == stage
                    and r["candidate"] == candidate
                    and r["slippage_bps"] == cost
                ]
                if not subset:
                    continue
                groups.append(
                    {
                        "phase": stage,
                        "candidate": candidate,
                        "slippage_bps": cost,
                        **aggregate(subset, root),
                    }
                )
    summary = {
        "schema": "factor-lab-refinement-results.v1",
        "plan_sha256": sha(root / "plan.json"),
        "original": old,
        "refinements": rows,
        "refinement_groups": groups,
        "selection": choice,
        "audit": {
            "passed": True,
            "accounts": audits,
            "max_nav_error": max(a["max_nav_error"] for a in audits),
        },
        "single_factor_scope": "post-hoc hypothesis, four original2025windows only, cost8bps, target-neutral gross0.4",
        "additional_windows": plan["additional_windows"],
        "target_neutrality_is_not_actual_fill_neutrality": True,
        "production_eligible": False,
        "orders_enabled": False,
        "independent_oos": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    save(root / "summary.json", summary)
    save(
        root / "completion.json",
        {
            "summary_sha256": sha(root / "summary.json"),
            "account_count": len(rows),
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    return {
        "selected": selected,
        "accounts": len(rows),
        "seconds": time.monotonic() - started,
        "audit": True,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(run(a.output.resolve()))
