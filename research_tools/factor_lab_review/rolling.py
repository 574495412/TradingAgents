"""Fixed 180-day training experiment; matured labels and unpenalized intercept."""

import argparse
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.factor_lab.metrics import prediction_metrics
from research_tools.factor_lab.study import account, executor, load_market, mask, output_account
from research_tools.factor_lab_audit import audit_account
from research_tools.factor_lab_review.compare import aggregate
from research_tools.factor_lab_review.run import selection


def fit_predict(x, labels, start, end, columns, penalty=100):
    """No labels at or after start enter fit, imputation, or scale estimation."""
    clock = pd.Timestamp(start, tz="UTC")
    t = x.index.get_level_values("time")
    train = (t >= clock - pd.Timedelta(days=180)) & (t + pd.Timedelta(hours=24) < clock)
    train &= labels.notna().to_numpy()
    tx = x.loc[train, columns]
    y = labels.loc[train].to_numpy()
    med = tx.median()
    if med.isna().any() or not len(y):
        raise ValueError("Missing training evidence")
    scale = tx.std(ddof=1).replace(0, 1).fillna(1)
    z = ((tx.fillna(med) - med) / scale).clip(-5, 5).to_numpy()
    center = z.mean(axis=0)
    centered = z - center
    beta = np.linalg.solve(
        centered.T @ centered + penalty * np.eye(len(columns)), centered.T @ (y - y.mean())
    )
    intercept = float(y.mean() - center @ beta)
    evaluate = mask(x.index, start, end)
    ex = ((x.loc[evaluate, columns].fillna(med) - med) / scale).clip(-5, 5)
    scores = pd.Series(ex.to_numpy() @ beta + intercept, index=ex.index, name="score")
    return scores, {
        "columns": columns,
        "median": med.to_dict(),
        "scale": scale.to_dict(),
        "coefficients": beta.tolist(),
        "intercept": intercept,
        "ridge_penalty": penalty,
        "training_rows": len(y),
        "training_first_clock": t[train].min().isoformat(),
        "training_last_label_clock": (t[train].max() + pd.Timedelta(hours=24)).isoformat(),
        "evaluation_start": clock.isoformat(),
        "training_max_feature_clock": t[train].max().isoformat(),
    }


def run(root):
    started = time.monotonic()
    root.mkdir(parents=True, exist_ok=False)
    template = read(Path(__file__).with_name("plan.json"))
    parent = Path(template["parent"])
    original = read(parent / "plan.json")
    plan = {
        "schema": "factor-lab-rolling-refinement.v1",
        "registered_at": datetime.now(UTC).isoformat(),
        "parent": str(parent),
        "prior_review": "factor_lab_review_20261025_v001",
        "candidates": ["rolling_full40", "rolling_technical40"],
        "validation_windows": template["validation_windows"],
        "costs_slippage_bps": [3, 10],
        "training": "previous180calendar days;24h labels must mature strictly before window start; fit once per window; train-only median/std, +/-5 clipping; centered Ridge and unpenalized intercept",
        "ridge_penalty": 100,
        "gross": 0.4,
        "selection": template["selection"],
        "additional_windows": template["additional_windows"],
        "search_budget": "2models x10windows x2costs=40; no grid; stop after this round regardless of outcome",
        "motivation": "2024 fixed normalization and coefficients may drift; compare with existing full_budget40 at equal risk; centered intercept also changes so this is a package comparison, not isolated causal attribution",
        "hypothesis_origin": "post-review of original64 and first114; all2026 history already available, no untouched OOS claim",
        "reference": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html",
        "production_eligible": False,
        "orders_enabled": False,
        "independent_oos": False,
    }
    paths = [
        Path(__file__),
        Path(template["execution_source"]),
        parent / "features.parquet",
        parent / "labels.parquet",
        parent / "catalog.json",
        parent / "models.json",
    ]
    paths += [Path(__file__).with_name(n) for n in ("compare.py", "run.py")]
    plan["source_hashes"] = {str(p.resolve()): sha(p) for p in paths}
    save(root / "plan.json", plan)
    save(root / "registration.json", {"plan_sha256": sha(root / "plan.json")})
    snapshot = root / "source_snapshot"
    snapshot.mkdir()
    for p in paths:
        if p.suffix == ".py":
            shutil.copyfile(p, snapshot / p.name)
    x = pd.read_parquet(parent / "features.parquet")
    labels = pd.read_parquet(parent / "labels.parquet")["24"]
    oldmodels = read(parent / "models.json")
    bars, rates = load_market(parent)
    engine = executor(template["execution_source"])
    rows, evaluations, audits = [], [], []
    windows = [
        {
            **w,
            "phase": "validation"
            if w["id"] in plan["validation_windows"]
            else "reused_2026_diagnostic",
        }
        for w in original["test_windows"]
    ] + plan["additional_windows"]
    choice = None
    for wi, window in enumerate(windows):
        if wi == 4:
            choice = selection(rows, root, plan)
            save(root / "selection.json", choice)
        for candidate in plan["candidates"]:
            family = candidate.removeprefix("rolling_").removesuffix("40")
            score, model = fit_predict(
                x, labels, window["start"], window["end"], oldmodels[family + "_24h"]["columns"]
            )
            save(root / f"model_{window['id']}_{candidate}.json", model)
            pmask = mask(score.index, window["start"], window["end"], 24)
            evaluations.append(
                {
                    "window": window["id"],
                    "candidate": candidate,
                    **prediction_metrics(score.loc[pmask], labels.reindex(score.index).loc[pmask]),
                }
            )
            for cost in (3, 10):
                config = {**original["default_account"], "gross": 0.4, "slippage_bps": cost}
                a = account(parent, engine, bars, rates, score, window, config)
                key = f"{window['id']}_{candidate}_slip{cost}"
                output_account(root / "accounts" / key, a, config, window)
                row = {
                    "id": key,
                    "candidate": candidate,
                    "preset": candidate,
                    "window": window["id"],
                    "phase": window["phase"],
                    "capital": 10000,
                    "slippage_bps": cost,
                    "metrics": a["metrics"],
                    "capacity": a["capacity"],
                    "result_sha256": sha(root / "accounts" / key / "result.json"),
                }
                rows.append(row)
                audits.append(audit_account(root, row, bars, rates))
                print(
                    candidate, window["id"], cost, round(a["metrics"]["net_return"], 6), flush=True
                )
    groups = []
    for phase in ("validation", "reused_2026_diagnostic", "additional_evaluation"):
        for candidate in plan["candidates"]:
            for cost in (3, 10):
                subset = [
                    r
                    for r in rows
                    if r["phase"] == phase
                    and r["candidate"] == candidate
                    and r["slippage_bps"] == cost
                ]
                groups.append(
                    {
                        "phase": phase,
                        "candidate": candidate,
                        "slippage_bps": cost,
                        **aggregate(subset, root),
                    }
                )
    summary = {
        "plan_sha256": sha(root / "plan.json"),
        "refinements": rows,
        "refinement_groups": groups,
        "prediction_evaluations": evaluations,
        "selection": choice,
        "audit": {"passed": True, "accounts": audits},
        "production_eligible": False,
        "independent_oos": False,
        "orders_enabled": False,
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
    print(
        {
            "selected": choice["selected_candidate"],
            "accounts": len(rows),
            "seconds": time.monotonic() - started,
        }
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args().output.resolve())
