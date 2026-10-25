"""Freeze a finite mechanism experiment and audit every resulting account."""

import argparse
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from factors.common.mechanisms import (
    CATALOG,
    FLOW_COLUMNS,
    PRICE_COLUMNS,
    RULES,
    features,
    lease_model_score,
    rule_scores,
)
from research_tools.factor_lab.io import read, save, sha
from research_tools.factor_lab.metrics import portfolio_metrics, prediction_metrics
from research_tools.factor_lab.study import executor
from research_tools.factor_lab_audit import audit_account
from research_tools.factor_lab_review.compare import aggregate


def fit_month(x, y, start, end, columns):
    clock = pd.Timestamp(start, tz="UTC")
    t = x.index.get_level_values("time")
    train = (
        (t >= clock - pd.Timedelta(days=180))
        & (t + pd.Timedelta(hours=72) < clock)
        & y.notna().to_numpy()
    )
    tx = x.loc[train, columns]
    target = y.loc[train].to_numpy()
    med = tx.median()
    if med.isna().any() or len(target) < 1000:
        raise ValueError("Insufficient training-only evidence")
    scale = tx.std(ddof=1).replace(0, 1).fillna(1)
    z = ((tx.fillna(med) - med) / scale).clip(-5, 5).to_numpy()
    center = z.mean(axis=0)
    centered = z - center
    beta = np.linalg.solve(
        centered.T @ centered + 100 * np.eye(len(columns)), centered.T @ (target - target.mean())
    )
    intercept = float(target.mean() - center @ beta)
    chosen = (t >= clock) & (t < pd.Timestamp(end, tz="UTC"))
    ex = ((x.loc[chosen, columns].fillna(med) - med) / scale).clip(-5, 5)
    score = pd.Series(ex.to_numpy() @ beta + intercept, index=ex.index)
    return score, {
        "columns": columns,
        "median": med.to_dict(),
        "scale": scale.to_dict(),
        "coefficients": beta.tolist(),
        "intercept": intercept,
        "training_rows": len(target),
        "latest_matured_label": (t[train].max() + pd.Timedelta(hours=72)).isoformat(),
        "evaluation_start": clock.isoformat(),
        "penalty": 100,
    }


def write_account(folder, a, config, window):
    folder.mkdir(parents=True, exist_ok=False)
    a["curve"].to_parquet(folder / "equity_5m.parquet", index=False)
    for name in ("events", "trades", "requests", "positions", "pending_exits"):
        save(folder / (name + ".json"), a[name])
    nav = a["curve"].set_index("date").equity
    dd = 1 - nav / nav.cummax()
    samples = nav.loc[nav.index.minute == 0]
    result = {
        "metrics": a["metrics"],
        "capacity": a["capacity"],
        "config": config,
        "window": window,
        "curve": [
            {"time": t.isoformat(), "equity": float(v), "drawdown": float(dd.loc[t])}
            for t, v in samples.items()
        ],
        "trades": a["trades"],
        "open_positions": a["positions"],
        "risk_halted": a["risk_halted"],
        "engine": a["engine"],
        "production_eligible": False,
        "orders_enabled": False,
        "independent_oos": False,
        "source_hashes": {p.name: sha(p) for p in folder.iterdir()},
    }
    save(folder / "result.json", result)


def select(rows, root, plan):
    assessments = []
    limits = plan["admission"]
    for candidate in plan["candidates"]:
        cases = []
        reasons = []
        for cost in plan["costs_slippage_bps"]:
            sample = [
                r
                for r in rows
                if r["candidate"] == candidate
                and r["slippage_bps"] == cost
                and r["phase"] == "screen"
            ]
            if len(sample) != 4:
                raise ValueError("Incomplete frozen screen")
            g = aggregate(sample, root)
            cases.append({"slippage_bps": cost, **g})
            tests = {
                "return_below3percent_target": g["mean_window_return"]
                >= limits["minimum_mean_28day_return_each_cost"],
                "positive_windows": g["positive_windows"]
                >= limits["minimum_positive_windows_each_cost"],
                "drawdown": g["worst_observed_drawdown"] <= limits["maximum_observed_drawdown"],
                "complete_trades": g["complete_trades"]
                >= limits["minimum_complete_trades_each_cost"],
                "closed_profit_factor": g["pooled_complete_profit_factor"] is not None
                and g["pooled_complete_profit_factor"]
                >= limits["minimum_closed_profit_factor_each_cost"],
                "median_sharpe": g["median_window_sharpe"] is not None
                and g["median_window_sharpe"] >= limits["minimum_median_daily_sharpe_each_cost"],
            }
            reasons.extend(f"{cost}:{k}" for k, v in tests.items() if not v)
        assessments.append(
            {"candidate": candidate, "cases": cases, "passed": not reasons, "reasons": reasons}
        )
    eligible = [g for g in assessments if g["candidate"] != plan["control"]]
    ranked = sorted(
        eligible,
        key=lambda g: (
            -min(c["mean_window_return"] for c in g["cases"]),
            max(c["worst_observed_drawdown"] for c in g["cases"]),
            g["candidate"],
        ),
    )
    return {
        "admitted_candidates": [g["candidate"] for g in eligible if g["passed"]],
        "research_followups": [g["candidate"] for g in ranked[:2]],
        "assessments": assessments,
        "later_results_included": False,
        "production_eligible": False,
    }


def run(root):
    started = time.monotonic()
    root.mkdir(parents=True, exist_ok=False)
    plan = read(Path(__file__).with_name("plan.json"))
    data = Path(plan["data"])
    receipt = read(data / "completion.json")
    for name, value in receipt["manifest"].items():
        if sha(data / name) != value:
            raise ValueError("Dataset manifest changed")
    plan["registered_at"] = datetime.now(UTC).isoformat()
    paths = [
        Path(__file__),
        Path(__file__).with_name("plan.json"),
        Path("factors/common/mechanisms.py"),
        Path(plan["engine"]),
        Path("research_tools/factor_lab/metrics.py"),
        Path("research_tools/factor_lab_audit.py"),
        Path("research_tools/factor_lab_review/compare.py"),
        data / "completion.json",
    ]
    plan["source_hashes"] = {str(p.resolve()): sha(p) for p in paths}
    plan["data_manifest"] = receipt["manifest"]
    save(root / "plan.json", plan)
    save(
        root / "registration.json",
        {
            "plan_sha256": sha(root / "plan.json"),
            "registered_before_model_fit_and_economic_screen": True,
        },
    )
    snap = root / "source_snapshot"
    snap.mkdir()
    for p in paths:
        if p.suffix == ".py":
            shutil.copyfile(p, snap / p.name)
    bars = {s: pd.read_parquet(data / f"{s}_5m.parquet") for s in plan["symbols"]}
    rates = {s: pd.read_parquet(data / f"{s}_funding.parquet") for s in plan["symbols"]}
    panels, opens = {}, {}
    for s in plan["symbols"]:
        panels[s], opens[s] = features(bars[s], rates[s])
    x = pd.concat(panels, names=["symbol", "time"]).swaplevel().sort_index()
    y = (
        pd.concat({s: op.shift(-72) / op - 1 for s, op in opens.items()}, names=["symbol", "time"])
        .swaplevel()
        .sort_index()
    )
    x.to_parquet(root / "features.parquet")
    y.rename("label72").to_frame().to_parquet(root / "labels.parquet")
    scores = {
        rule: pd.DataFrame({s: rule_scores(panels[s], rule) for s in panels}) for rule in RULES
    }
    for candidate, columns in (
        ("mechanism_ridge72", PRICE_COLUMNS + FLOW_COLUMNS),
        ("price_ridge72", PRICE_COLUMNS),
    ):
        parts = []
        for month in pd.period_range("2025-01", "2026-09", freq="M"):
            score, model = fit_month(
                x, y, str(month.start_time.date()), str((month + 1).start_time.date()), columns
            )
            save(root / f"model_{candidate}_{month}.json", model)
            parts.append(score)
        fitted = pd.concat(parts).unstack("symbol")
        scores[candidate] = pd.DataFrame({s: lease_model_score(fitted[s]) for s in fitted})
    for candidate, score in scores.items():
        for s in panels:
            score["vol_" + s] = panels[s].vol24
        score.to_parquet(root / f"scores_{candidate}.parquet")
    save(root / "factor_catalog.json", CATALOG)
    engine = executor(plan["engine"])
    rows = []
    audits = []

    def one(candidate, w, cost, phase, capital=10000):
        config = {**plan["account"], "capital": capital, "slippage_bps": cost}
        a = engine.replay(bars, rates, scores[candidate], config, w["start"], w["end"])
        boundary = pd.date_range(w["start"], w["end"], freq="D", tz="UTC")
        benchmark = bars["BTCUSDT"].close.reindex(boundary - pd.Timedelta(minutes=5))
        benchmark.index = boundary
        a["metrics"] = portfolio_metrics(a["curve"], a["trades"], a["events"], capital, benchmark)
        a["metrics"]["mean_absolute_net_exposure"] = float(
            (a["curve"].net_exposure / a["curve"].equity).abs().mean()
        )
        key = f"{phase}_{w['id']}_{candidate}_cap{capital}_slip{cost}"
        write_account(root / "accounts" / key, a, config, w)
        row = {
            "id": key,
            "candidate": candidate,
            "window": w["id"],
            "phase": phase,
            "start": w["start"],
            "end": w["end"],
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
            w["id"],
            candidate,
            cost,
            capital,
            round(a["metrics"]["net_return"], 6),
            flush=True,
        )

    for window in plan["screen_windows"]:
        for candidate in plan["candidates"]:
            for cost in plan["costs_slippage_bps"]:
                one(candidate, window, cost, "screen")
    selection = select(rows, root, plan)
    save(root / "selection.json", selection)
    follow = selection["research_followups"] + [plan["control"]]
    for window in plan["continuous_windows"]:
        for candidate in follow:
            for cost in plan["costs_slippage_bps"]:
                one(
                    candidate,
                    window,
                    cost,
                    "continuous" if window["id"].startswith("continuous") else "extension",
                )
    capacity_window = next(w for w in plan["continuous_windows"] if w["id"] == "continuous2026")
    for candidate in selection["research_followups"]:
        for capital in (100000, 1000000):
            for cost in plan["costs_slippage_bps"]:
                one(candidate, capacity_window, cost, "capacity", capital)
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
        for c in plan["candidates"]
        for cost in plan["costs_slippage_bps"]
    ]
    predictions = []
    for window in plan["screen_windows"] + plan["continuous_windows"]:
        t = x.index.get_level_values("time")
        selected = (t >= pd.Timestamp(window["start"], tz="UTC")) & (
            t + pd.Timedelta(hours=72) < pd.Timestamp(window["end"], tz="UTC")
        )
        for candidate in ("mechanism_ridge72", "price_ridge72"):
            fitted = scores[candidate][plan["symbols"]].stack(future_stack=True)
            fitted.index = fitted.index.set_names(["time", "symbol"])
            # This is the executable daily leased score, not a claim about raw model accuracy.
            predictions.append(
                {
                    "window": window["id"],
                    "candidate": candidate,
                    **prediction_metrics(fitted.reindex(x.index).loc[selected], y.loc[selected]),
                }
            )
    summary = {
        "schema": "factor-discovery-results.v1",
        "plan_sha256": sha(root / "plan.json"),
        "selection": selection,
        "account_rows": rows,
        "screen_groups": groups,
        "factor_catalog": CATALOG,
        "prediction_diagnostics": predictions,
        "limitations": plan["limitations"],
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
            "admitted": selection["admitted_candidates"],
            "followups": selection["research_followups"],
            "accounts": len(rows),
            "seconds": time.monotonic() - started,
        },
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args().output.resolve())
