"""Registered fast factor study: frozen training, representative quarters, full costs."""

import importlib.metadata
import importlib.util
import math
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .features import build
from .io import read, save, sha
from .metrics import METRIC_DEFINITIONS, portfolio_metrics, prediction_metrics


def executor(path):
    spec = importlib.util.spec_from_file_location("crypto_factor_execution", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stacked(frames):
    return pd.concat(frames, names=["symbol", "time"]).swaplevel().sort_index()


def mask(index, start, end, horizon=0):
    t = index.get_level_values("time")
    return (t >= pd.Timestamp(start, tz="UTC")) & (
        t + pd.Timedelta(hours=horizon) < pd.Timestamp(end, tz="UTC")
    )


def compose_score(z, weights, label_scale):
    keys = list(weights)
    denominator = math.fsum(abs(float(weights[k])) for k in keys)
    if not keys or denominator <= 0:
        raise ValueError("Select at least one nonzero factor")
    return pd.Series(
        z[keys].to_numpy() @ np.array([weights[k] for k in keys]) / denominator * label_scale,
        index=z.index,
    )


def load_market(run):
    plan = read(run / "plan.json")
    source = Path(plan["source_run"]) / "data"
    return (
        {s: pd.read_parquet(source / f"{s}_5m.parquet") for s in plan["symbols"]},
        {s: pd.read_parquet(source / f"{s}_funding.parquet") for s in plan["symbols"]},
    )


def account(run, engine, bars, funding, score, window, config):
    # All decisions at clock t use a score generated exclusively from information available at t.
    scores = score.unstack("symbol")
    features = pd.read_parquet(run / "features.parquet", columns=["vol_24"])
    vol = features.vol_24.unstack("symbol")
    for s in bars:
        scores["vol_" + s] = vol[s]
    a = engine.replay(bars, funding, scores, config, window["start"], window["end"])
    raw = bars["BTCUSDT"]["close"]
    boundary = pd.date_range(window["start"], window["end"], freq="D", tz="UTC")
    benchmark = raw.reindex(boundary - pd.Timedelta(minutes=5))
    benchmark.index = boundary
    m = portfolio_metrics(a["curve"], a["trades"], a["events"], config["capital"], benchmark)
    a["metrics"] = m
    return a


def output_account(folder, a, config, window):
    folder.mkdir(parents=True, exist_ok=False)
    a["curve"].to_parquet(folder / "equity_5m.parquet", index=False)
    for name in ["events", "trades", "requests", "positions", "pending_exits"]:
        save(folder / (name + ".json"), a[name])
    nav = a["curve"].set_index("date").equity
    # Every hourly sample plus every new drawdown extreme; the full5m curve remains available locally.
    sampled = nav.loc[(nav.index.minute == 0)]
    curve = [
        {"time": t.isoformat(), "equity": float(v), "drawdown": float(1 - v / nav.loc[:t].max())}
        for t, v in sampled.items()
    ]
    result = {
        "metrics": a["metrics"],
        "prediction": a.get("prediction"),
        "capacity": a["capacity"],
        "config": config,
        "window": window,
        "curve": curve,
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
    return result


def train(output):  # noqa: C901
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    template = Path(__file__).with_name("plan.json")
    plan = read(template)
    plan["registered_at"] = datetime.now(UTC).isoformat()
    plan["account_presets"] = ["full", "technical", "training_factor8", "long_only"]
    plan["account_grid"] = (
        "7 fixed28day windows x4presets x2costs=56base accounts; Q2_2025 full and training_factor8 x2larger capitals x2costs=8capacity cases"
    )
    source = Path(plan["source_run"]) / "data"
    news = Path(plan["news_source"])
    identities = {
        str(source / "receipts.json"): sha(source / "receipts.json"),
        str(news / "events.json"): sha(news / "events.json"),
        str(news / "coverage.json"): sha(news / "coverage.json"),
        plan["execution_source"]: sha(Path(plan["execution_source"])),
    }
    for name, value in read(source / "receipts.json").items():
        if sha(source / name) != value:
            raise ValueError("Sealed data changed")
    for item in read(news / "events.json"):
        if sha(news / item["raw_file"]) != item["sha256"]:
            raise ValueError("Official news source changed")
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for p in sorted(Path(__file__).parent.glob("*.py")):
        shutil.copyfile(p, snapshot / p.name)
        identities[str(p.resolve())] = sha(p)
    canonical_root = Path(__file__).resolve().parents[2] / "factors"
    for p in sorted(canonical_root.rglob("*.py")):
        if "legacy" in p.relative_to(canonical_root).parts:
            continue
        relative = p.relative_to(canonical_root)
        target = snapshot / "factors" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
        identities[str(p.resolve())] = sha(p)
    shutil.copyfile(plan["execution_source"], snapshot / "factor_portfolio.py")
    plan["source_identities"] = identities
    save(output / "plan.json", plan)
    save(
        output / "registration.json",
        {
            "plan_sha256": sha(output / "plan.json"),
            "runtime": {
                "python": sys.version,
                "packages": {
                    p: importlib.metadata.version(p)
                    for p in ["numpy", "pandas", "pyarrow", "TA-Lib"]
                },
            },
            "orders_enabled": False,
        },
    )
    events = read(news / "events.json")
    frames, raw, catalog = build(source, events, plan["symbols"])
    if len(catalog) != plan["study_search_budget"]["factor_columns_expected"]:
        raise ValueError("Declared factor budget differs")
    x = stacked(frames)
    x.to_parquet(output / "features.parquet")
    labels = stacked(
        {
            s: pd.DataFrame({str(h): r.open.shift(-h) / r.open - 1 for h in plan["horizons_hours"]})
            for s, r in raw.items()
        }
    )
    labels.to_parquet(output / "labels.parquet")
    train_mask = mask(
        x.index, plan["training_start"], plan["training_end"], max(plan["horizons_hours"])
    )
    tx = x.loc[train_mask]
    med = tx.median()
    scale = tx.std(ddof=1).replace(0, 1).fillna(1)
    if med.isna().any():
        raise ValueError("All-missing training factor, cannot impute evidence")
    z = ((x.fillna(med) - med) / scale).clip(-5, 5)
    z.to_parquet(output / "standardized.parquet")
    save(
        output / "normalization.json",
        {
            "median": med.to_dict(),
            "scale": scale.to_dict(),
            "label_scale": float(labels.loc[train_mask, "24"].std(ddof=1)),
            "label_scale_by_horizon": {
                str(h): float(labels.loc[train_mask, str(h)].std(ddof=1))
                for h in plan["horizons_hours"]
            },
        },
    )
    training_stats = []
    for item in catalog:
        k = item["id"]
        m = prediction_metrics(x.loc[train_mask, k], labels.loc[train_mask, "24"])
        training_stats.append(
            {
                **item,
                "training_rank_ic": m["rank_ic"],
                "training_ic": m["ic"],
                "training_coverage": float(tx[k].notna().mean()),
            }
        )
    ordered = sorted(training_stats, key=lambda c: (-abs(c["training_rank_ic"] or 0), c["id"]))
    chosen = {}
    corr = tx.corr().fillna(0)
    for c in ordered:
        k = c["id"]
        if c["training_rank_ic"] is None or any(abs(corr.loc[k, j]) >= 0.85 for j in chosen):
            continue
        chosen[k] = 1 if c["training_rank_ic"] >= 0 else -1
        if len(chosen) == 8:
            break
    save(
        output / "selection.json",
        {
            "weights": chosen,
            "selection_basis": "2024training24hRankIC only; correlation screen. Subsequent diagnostics never replace this selection",
            "future_results_included": False,
            "production_eligible": False,
        },
    )
    preds = {}
    models = {}
    families = {i["id"]: i["family"] for i in catalog}
    for family in plan["model_families"]:
        columns = [
            k
            for k in x.columns
            if (
                families[k] in ["trend", "price", "technical"]
                if family == "price_baseline"
                else (
                    families[k] not in ["macro_news", "interaction"]
                    or k == "funding_macd_interaction"
                )
                if family == "technical"
                else families[k] == "macro_news"
                if family == "macro_news"
                else True
            )
        ]
        xx = z.loc[train_mask, columns].to_numpy()
        for horizon in plan["horizons_hours"]:
            y = labels.loc[train_mask, str(horizon)].to_numpy()
            valid = np.isfinite(y)
            mean = float(y[valid].mean())
            beta = np.linalg.solve(
                xx[valid].T @ xx[valid] + plan["ridge_penalty"] * np.eye(len(columns)),
                xx[valid].T @ (y[valid] - mean),
            )
            key = f"{family}_{horizon}h"
            preds[key] = pd.Series(z[columns].to_numpy() @ beta + mean, index=z.index)
            models[key] = {
                "family": family,
                "horizon": horizon,
                "columns": columns,
                "coefficients": dict(zip(columns, beta.tolist(), strict=True)),
                "intercept": mean,
                "training_rows": int(valid.sum()),
                "ridge_penalty": plan["ridge_penalty"],
            }
    prediction = pd.DataFrame(preds)
    normalization = read(output / "normalization.json")
    prediction["training_factor8"] = compose_score(z, chosen, normalization["label_scale"])
    prediction.to_parquet(output / "predictions.parquet")
    save(output / "models.json", models)
    evaluations = []
    factor_rows = {c["id"]: dict(c) for c in training_stats}
    validation_masks = []
    for w in plan["test_windows"]:
        wm = mask(x.index, w["start"], w["end"], 24)
        if w["phase"] == "validation":
            validation_masks.append(wm)
        btc = (
            raw["BTCUSDT"]
            .loc[
                (raw["BTCUSDT"].index >= pd.Timestamp(w["start"], tz="UTC"))
                & (raw["BTCUSDT"].index < pd.Timestamp(w["end"], tz="UTC"))
            ]
            .open
        )
        ret = float(btc.iloc[-1] / btc.iloc[0] - 1)
        w["btc_return"] = ret
        w["regime"] = "上涨" if ret > 0.05 else "下跌" if ret < -0.05 else "区间"
        w["btc_hourly_volatility"] = float(btc.pct_change(fill_method=None).std())
        for key, model in models.items():
            hm = mask(x.index, w["start"], w["end"], model["horizon"])
            m = prediction_metrics(prediction.loc[hm, key], labels.loc[hm, str(model["horizon"])])
            evaluations.append({"window": w["id"], "phase": w["phase"], "model": key, **m})
    vm = np.logical_or.reduce(validation_masks)
    for k, row in factor_rows.items():
        m = prediction_metrics(x.loc[vm, k], labels.loc[vm, "24"])
        row.update(
            validation_ic=m["ic"],
            validation_rank_ic=m["rank_ic"],
            validation_ic_ir=m["ic_ir"],
            validation_coverage=float(x.loc[vm, k].notna().mean()),
            selected_training_weight=chosen.get(k, 0),
            coefficient_full24=models["full_24h"]["coefficients"].get(k, 0),
        )
    save(output / "factor_catalog.json", list(factor_rows.values()))
    save(output / "prediction_evaluation.json", evaluations)
    engine = executor(plan["execution_source"])
    bars, funding = load_market(output)
    accounts = []
    for w in plan["test_windows"]:
        for preset in plan["account_presets"]:
            score = (
                prediction["training_factor8"]
                if preset == "training_factor8"
                else prediction[f"{preset}_24h"]
                if preset != "long_only"
                else prediction["full_24h"] * 0
            )
            for cost in plan["cost_cases_slippage_bps"]:
                config = {**plan["default_account"], "slippage_bps": cost}
                if preset == "long_only":
                    config["control"] = "long_only"
                a = account(output, engine, bars, funding, score, w, config)
                horizon_mask = mask(score.index, w["start"], w["end"], 24)
                a["prediction"] = (
                    prediction_metrics(score.loc[horizon_mask], labels.loc[horizon_mask, "24"])
                    if preset != "long_only"
                    else None
                )
                key = f"{w['id']}_{preset}_capital10000_slip{cost}"
                output_account(output / "accounts" / key, a, config, w)
                accounts.append(
                    {
                        "id": key,
                        "preset": preset,
                        "window": w["id"],
                        "phase": w["phase"],
                        "capital": config["capital"],
                        "slippage_bps": cost,
                        "metrics": a["metrics"],
                        "capacity": a["capacity"],
                    }
                )
                print(f"Account {key}: net {a['metrics']['net_return']:.6f}", flush=True)
    # Prespecified stressed calendar window, not the best/worst candidate window.
    w = next(w for w in plan["test_windows"] if w["id"] == "2025Q2")
    for preset in ["full", "training_factor8"]:
        score = prediction["training_factor8" if preset == "training_factor8" else "full_24h"]
        for capital in plan["capacity_capitals"][1:]:
            for cost in plan["cost_cases_slippage_bps"]:
                config = {**plan["default_account"], "capital": capital, "slippage_bps": cost}
                a = account(output, engine, bars, funding, score, w, config)
                key = f"capacity_{w['id']}_{preset}_capital{capital}_slip{cost}"
                output_account(output / "accounts" / key, a, config, w)
                accounts.append(
                    {
                        "id": key,
                        "preset": preset,
                        "window": w["id"],
                        "phase": "capacity",
                        "capital": capital,
                        "slippage_bps": cost,
                        "metrics": a["metrics"],
                        "capacity": a["capacity"],
                    }
                )
    save(output / "accounts.json", accounts)
    save(
        output / "catalog.json",
        {
            "schema": "factor-lab-catalog.v1",
            "factor_count": len(catalog),
            "factors": list(factor_rows.values()),
            "windows": plan["test_windows"],
            "models": list(models),
            "prediction_evaluations": evaluations,
            "accounts": accounts,
            "training_selection": chosen,
            "default_config": plan["default_account"],
            "metric_definitions": METRIC_DEFINITIONS,
            "news": events,
            "news_coverage": read(news / "coverage.json"),
            "limitations": plan["limitations"],
            "metric_references": [
                "https://ta-lib.org/functions/",
                "https://github.com/ranaroussi/quantstats",
                "https://qlib.readthedocs.io/en/latest/component/report.html",
            ],
            "independent_oos": False,
            "production_eligible": False,
            "orders_enabled": False,
        },
    )
    save(
        output / "completion.json",
        {
            "factor_count": len(catalog),
            "model_fits": len(models),
            "account_count": len(accounts),
            "elapsed_seconds": time.monotonic() - started,
            "plan_sha256": sha(output / "plan.json"),
            "catalog_sha256": sha(output / "catalog.json"),
        },
    )
    return {
        "factors": len(catalog),
        "models": len(models),
        "accounts": len(accounts),
        "seconds": time.monotonic() - started,
    }
