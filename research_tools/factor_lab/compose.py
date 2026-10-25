"""Validate a UI composition and run a real, short historical research account."""

import time
from pathlib import Path

import pandas as pd

from .io import read, save, sha
from .metrics import prediction_metrics
from .study import account, compose_score, executor, load_market, mask, output_account

PRESETS = {
    "full",
    "technical",
    "price_baseline",
    "macro_news",
    "training_factor8",
    "custom",
    "cash",
    "long_only",
}


def validate(config, plan, factors):
    allowed = {
        "preset",
        "weights",
        "window",
        "horizon",
        "capital",
        "fee_bps",
        "slippage_bps",
        "participation",
        "gross",
        "max_positions",
        "rebalance_hours",
        "threshold",
        "stop_loss",
        "drawdown_halt",
    }
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError("Unknown composition fields")
    preset = config.get("preset", "training_factor8")
    if preset not in PRESETS:
        raise ValueError("Unknown composition preset")
    windows = {w["id"]: w for w in plan["test_windows"]}
    if config.get("window") not in windows:
        raise ValueError("Select a registered representative window")
    horizon = config.get("horizon", 24)
    if type(horizon) is not int or horizon not in (1, 4, 24):
        raise ValueError("Unsupported prediction horizon")
    weights = config.get("weights", {})
    if not isinstance(weights, dict) or set(weights) - set(factors) or len(weights) > 100:
        raise ValueError("Unknown or excess factor IDs")
    for value in weights.values():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not -10 <= value <= 10:
            raise ValueError("Factor weights must be finite numbers in[-10,10]")
    if preset == "custom" and not any(abs(v) > 0 for v in weights.values()):
        raise ValueError("Select at least one nonzero factor")
    out = dict(plan["default_account"])
    limits = {
        "capital": (100, 10000000),
        "fee_bps": (0, 50),
        "slippage_bps": (0, 100),
        "participation": (0.0001, 0.01),
        "gross": (0.05, 1.0),
        "threshold": (0, 0.1),
        "stop_loss": (0.01, 0.5),
        "drawdown_halt": (0.01, 0.5),
    }
    for k, (lo, hi) in limits.items():
        if k in config:
            value = config[k]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not lo <= value <= hi
            ):
                raise ValueError("Invalid " + k)
            out[k] = float(value)
    for k, values in [("max_positions", range(1, 7)), ("rebalance_hours", (1, 4, 12, 24))]:
        if k in config:
            if type(config[k]) is not int or config[k] not in values:
                raise ValueError("Invalid " + k)
            out[k] = config[k]
    if preset in ("cash", "long_only"):
        out["control"] = preset
    return preset, weights, horizon, windows[config["window"]], out


def compose(run, config, output):
    started = time.monotonic()
    plan = read(run / "plan.json")
    if sha(run / "plan.json") != read(run / "registration.json")["plan_sha256"]:
        raise ValueError("Registered study plan changed")
    # Current code and engine must match the frozen experiment, avoiding silent UI drift.
    for name, checksum in plan["source_identities"].items():
        if sha(Path(name)) != checksum:
            raise ValueError("Research source changed; register a new study version")
    catalog = read(run / "factor_catalog.json")
    preset, weights, horizon, window, engine_config = validate(
        config, plan, [c["id"] for c in catalog]
    )
    if preset == "custom" or preset == "training_factor8" and horizon != 24:
        if preset == "training_factor8":
            weights = read(run / "selection.json")["weights"]
        z = pd.read_parquet(run / "standardized.parquet", columns=list(weights))
        norm = read(run / "normalization.json")
        score = compose_score(z, weights, norm["label_scale_by_horizon"][str(horizon)])
    else:
        key = (
            "training_factor8"
            if preset == "training_factor8"
            else f"{preset}_{horizon}h"
            if preset not in ("cash", "long_only")
            else "full_24h"
        )
        score = pd.read_parquet(run / "predictions.parquet", columns=[key])[key]
    labels = pd.read_parquet(run / "labels.parquet", columns=[str(horizon)])[str(horizon)]
    prediction = (
        None
        if preset in ("cash", "long_only")
        else prediction_metrics(
            score.loc[mask(score.index, window["start"], window["end"], horizon)],
            labels.loc[mask(labels.index, window["start"], window["end"], horizon)],
        )
    )
    bars, funding = load_market(run)
    engine = executor(plan["execution_source"])
    a = account(run, engine, bars, funding, score, window, engine_config)
    result = output_account(output, a, engine_config, window)
    result.update(
        prediction=prediction,
        preset=preset,
        weights=weights,
        horizon=horizon,
        metric_definitions=read(run / "catalog.json")["metric_definitions"],
        limitations=plan["limitations"],
        elapsed_seconds=time.monotonic() - started,
        source_catalog_sha256=sha(run / "catalog.json"),
        holdout_consumed=window["phase"] == "holdout",
        research_only=True,
    )
    # Append a distinct UI receipt; source result and all historical accounts remain immutable.
    save(output / "composition.json", result)
    return result
