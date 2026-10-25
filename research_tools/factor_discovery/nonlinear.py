"""Two fixed shallow tree candidates; preserve all rejected first-round evidence."""

import argparse
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

from factors.common.mechanisms import FLOW_COLUMNS, PRICE_COLUMNS, lease_model_score
from factors.common.regime import REGIME_COLUMNS, condition_market
from research_tools.factor_discovery.run import select, write_account
from research_tools.factor_lab.io import read, save, sha
from research_tools.factor_lab.metrics import portfolio_metrics
from research_tools.factor_lab.study import executor
from research_tools.factor_lab_audit import audit_account
from research_tools.factor_lab_review.compare import aggregate

PARAMS = {
    "max_iter": 80,
    "learning_rate": 0.05,
    "max_leaf_nodes": 7,
    "max_depth": 3,
    "min_samples_leaf": 80,
    "l2_regularization": 10.0,
    "max_bins": 32,
    "early_stopping": False,
    "random_state": 25,
}


def fit(x, y, start, end, columns):
    boundary = pd.Timestamp(start, tz="UTC")
    t = x.index.get_level_values("time")
    train = (
        (t.hour == 0)
        & (t >= boundary - pd.Timedelta(days=180))
        & (t + pd.Timedelta(hours=72) < boundary)
        & y.notna().to_numpy()
    )
    target = y.loc[train]
    bounds = target.quantile([0.01, 0.99]).to_numpy()
    if len(target) < 500:
        raise ValueError("Insufficient daily training cohort")
    model = HistGradientBoostingRegressor(**PARAMS)
    model.fit(x.loc[train, columns], target.clip(*bounds))
    evaluate = (t.hour == 0) & (t >= boundary) & (t < pd.Timestamp(end, tz="UTC"))
    score = pd.Series(model.predict(x.loc[evaluate, columns]), index=x.index[evaluate])
    return (
        score,
        model,
        {
            "columns": columns,
            "training_rows": len(target),
            "training_sample": "daily decision clocks only, not repeated hourly72h labels",
            "target_clip_training_quantiles": bounds.tolist(),
            "latest_matured_label": (t[train].max() + pd.Timedelta(hours=72)).isoformat(),
            "evaluation_start": boundary.isoformat(),
            "parameters": PARAMS,
        },
    )


def run(root):
    started = time.monotonic()
    root.mkdir(parents=True, exist_ok=False)
    parent = Path("results/research/factor_discovery_20261025_v001")
    original = read(parent / "plan.json")
    if sha(parent / "summary.json") != read(parent / "completion.json")["summary_sha256"]:
        raise ValueError("Parent completion changed")
    candidates = ["regime_flow_hgb72", "regime_price_hgb72"]
    paths = [
        Path(__file__),
        Path("factors/common/regime.py"),
        Path("factors/common/mechanisms.py"),
        Path("research_tools/factor_discovery/run.py"),
        Path(original["engine"]),
        parent / "features.parquet",
        parent / "labels.parquet",
        parent / "summary.json",
    ]
    plan = {
        "schema": "factor-discovery-nonlinear.v1",
        "registered_at": datetime.now(UTC).isoformat(),
        "parent": str(parent.resolve()),
        "candidates": candidates,
        "control": "regime_price_hgb72",
        "motivation": "First-round real-flow rules had isolated gains but did not generalize; test regime-conditioned nonlinearity versus same-method price-only control",
        "parameters": PARAMS,
        "training": "180day rolling, monthly refit, daily clocks only,72hour matured targets, train-only1/99percent target winsorization, native missing values; random early-stopping split disabled",
        "account": original["account"],
        "admission": original["admission"],
        "costs_slippage_bps": [3, 10],
        "budget": "2fixed candidates x4screen x2cost=16;both x3continuous x2cost=12;both x2capital x2cost=8;total36; no parameter search",
        "followup": "Both candidates followed after screen is frozen; neither later returns nor capacity changes admission",
        "source_hashes": {str(p.resolve()): sha(p) for p in paths},
        "runtime": {"sklearn": sklearn.__version__, "numpy": np.__version__},
        "reference": "https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html",
        "independent_oos": False,
        "production_eligible": False,
        "orders_enabled": False,
    }
    save(root / "plan.json", plan)
    save(
        root / "registration.json",
        {"plan_sha256": sha(root / "plan.json"), "registered_before_fitting": True},
    )
    snapshot = root / "source_snapshot"
    snapshot.mkdir()
    for p in paths:
        if p.suffix == ".py":
            (snapshot / p.name).write_bytes(p.read_bytes())
    x = condition_market(pd.read_parquet(parent / "features.parquet"))
    y = pd.read_parquet(parent / "labels.parquet").label72
    scores = {}
    for candidate in candidates:
        columns = (
            PRICE_COLUMNS
            + REGIME_COLUMNS
            + (FLOW_COLUMNS if candidate == "regime_flow_hgb72" else [])
        )
        parts = []
        for month in pd.period_range("2025-01", "2026-09", freq="M"):
            prediction, model, metadata = fit(
                x, y, str(month.start_time.date()), str((month + 1).start_time.date()), columns
            )
            model_path = root / f"model_{candidate}_{month}.joblib"
            joblib.dump(model, model_path)
            save(model_path.with_suffix(".json"), {**metadata, "model_sha256": sha(model_path)})
            parts.append(prediction)
        daily = pd.concat(parts).unstack("symbol")
        hourly = daily.reindex(
            pd.date_range("2025-01-01", "2026-10-01", freq="h", inclusive="left", tz="UTC"),
            method="ffill",
        )
        score = pd.DataFrame({s: lease_model_score(hourly[s]) for s in original["symbols"]})
        for s in original["symbols"]:
            score["vol_" + s] = x.xs(s, level="symbol").vol24
        score.to_parquet(root / f"scores_{candidate}.parquet")
        scores[candidate] = score
    data = Path(original["data"])
    receipt = read(data / "completion.json")
    for name, value in receipt["manifest"].items():
        if sha(data / name) != value:
            raise ValueError("Sealed data changed")
    bars = {s: pd.read_parquet(data / f"{s}_5m.parquet") for s in original["symbols"]}
    rates = {s: pd.read_parquet(data / f"{s}_funding.parquet") for s in original["symbols"]}
    engine = executor(original["engine"])
    rows, audits = [], []

    def one(candidate, window, cost, phase, capital=10000):
        config = {**plan["account"], "capital": capital, "slippage_bps": cost}
        a = engine.replay(bars, rates, scores[candidate], config, window["start"], window["end"])
        boundary = pd.date_range(window["start"], window["end"], freq="D", tz="UTC")
        benchmark = bars["BTCUSDT"].close.reindex(boundary - pd.Timedelta(minutes=5))
        benchmark.index = boundary
        a["metrics"] = portfolio_metrics(a["curve"], a["trades"], a["events"], capital, benchmark)
        key = f"{phase}_{window['id']}_{candidate}_cap{capital}_slip{cost}"
        write_account(root / "accounts" / key, a, config, window)
        row = {
            "id": key,
            "candidate": candidate,
            "window": window["id"],
            "phase": phase,
            "start": window["start"],
            "end": window["end"],
            "capital": capital,
            "slippage_bps": cost,
            "metrics": a["metrics"],
            "capacity": a["capacity"],
            "result_sha256": sha(root / "accounts" / key / "result.json"),
        }
        rows.append(row)
        audits.append(audit_account(root, row, bars, rates))
        print(
            phase,
            window["id"],
            candidate,
            cost,
            capital,
            round(a["metrics"]["net_return"], 6),
            flush=True,
        )

    for w in original["screen_windows"]:
        for c in candidates:
            for cost in (3, 10):
                one(c, w, cost, "screen")
    choice = select(rows, root, plan)
    save(root / "selection.json", choice)
    for w in original["continuous_windows"]:
        for c in candidates:
            for cost in (3, 10):
                one(c, w, cost, "continuous" if w["id"].startswith("continuous") else "extension")
    capacity_window = next(w for w in original["continuous_windows"] if w["id"] == "continuous2026")
    for c in candidates:
        for capital in (100000, 1000000):
            for cost in (3, 10):
                one(c, capacity_window, cost, "capacity", capital)
    groups = [
        {
            "candidate": c,
            "slippage_bps": cost,
            **aggregate(
                [
                    r
                    for r in rows
                    if r["phase"] == "screen" and r["candidate"] == c and r["slippage_bps"] == cost
                ],
                root,
            ),
        }
        for c in candidates
        for cost in (3, 10)
    ]
    summary = {
        "plan_sha256": sha(root / "plan.json"),
        "selection": choice,
        "account_rows": rows,
        "screen_groups": groups,
        "audit": {
            "passed": all(a["passed"] for a in audits),
            "account_count": len(audits),
            "maximum_nav_error": max(a["max_nav_error"] for a in audits),
        },
        "audit_accounts": audits,
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
            "accounts": len(rows),
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(
        {
            "admitted": choice["admitted_candidates"],
            "accounts": len(rows),
            "seconds": time.monotonic() - started,
        },
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output.resolve())
