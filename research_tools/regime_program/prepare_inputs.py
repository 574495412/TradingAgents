"""Freeze strategy intents and monthly model forecasts before account evaluation."""

import argparse
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from factors.common.github_formulas import daily
from factors.common.regime_books import intents, source_features
from factors.portfolio.regime_allocation import allocate
from research_tools.factor_lab.io import read, save, sha
from research_tools.github_factor_study.run import snapshot_sources
from research_tools.regime_program.models import predictions


def prepare(root, state_root):
    started = time.monotonic()
    root.mkdir(parents=True, exist_ok=False)
    plan_path = Path(__file__).with_name("plan.json")
    plan = read(plan_path)
    data = Path(plan["data"])
    for name, digest in read(state_root / "completion.json")["manifest"].items():
        if sha(state_root / name) != digest:
            raise ValueError("Causal state evidence changed")
    if read(state_root / "plan.json")["state_parameters"] != plan["state_parameters"]:
        raise ValueError("State parameters differ from fixed preparation")
    sources = Path(plan["source_pack"])
    if sha(sources / "source_catalog.json") != read(sources / "completion.json")["catalog_sha256"]:
        raise ValueError("Source catalog changed")
    for record in read(sources / "source_catalog.json")["records"]:
        if sha(Path(record["local"])) != record["sha256"]:
            raise ValueError("Reference content changed")
    paths = [
        Path(__file__),
        plan_path,
        Path("research_tools/regime_program/models.py"),
        Path("factors/common/regime_books.py"),
        Path("factors/common/github_formulas.py"),
        Path("factors/portfolio/regime_allocation.py"),
        Path("research_tools/factor_lab/io.py"),
        Path("research_tools/github_factor_study/run.py"),
        state_root / "completion.json",
        sources / "source_catalog.json",
        data / "completion.json",
    ]
    plan["registered_at"] = datetime.now(UTC).isoformat()
    plan["state_root"] = str(state_root)
    plan["source_hashes"] = {str(p.resolve()): sha(p) for p in paths}
    save(root / "plan.json", plan)
    save(
        root / "registration.json",
        {"plan_sha256": sha(root / "plan.json"), "before_forecasts_and_account_returns": True},
    )
    (root / "source_snapshot").mkdir()
    snapshot_sources(paths, root / "source_snapshot")
    for name, digest in read(data / "completion.json")["manifest"].items():
        if Path(name).name != name or sha(data / name) != digest:
            raise ValueError("Official market data changed")
    symbols = read(data / "prepare_plan.json")["symbols"]
    frames = {s: daily(pd.read_parquet(data / (s + "_5m.parquet"))) for s in symbols}
    rates = {s: pd.read_parquet(data / (s + "_funding.parquet")) for s in symbols}
    states = pd.read_parquet(state_root / "states_daily.parquet")
    eligible = pd.read_parquet(state_root / "eligible_daily.parquet")
    f = {s: source_features(d) for s, d in frames.items()}
    normalization = {}
    for key in ("pst_ewmac32", "pst_breakout10", "mr_wings4"):
        sample = pd.DataFrame({s: x[key].shift(1) for s, x in f.items()})
        sample = sample.loc[
            (sample.index >= pd.Timestamp("2024-01-01", tz="UTC"))
            & (sample.index < pd.Timestamp("2025-01-01", tz="UTC"))
        ]
        magnitude = float(np.nanmean(np.abs(sample.to_numpy())))
        normalization[key] = magnitude if np.isfinite(magnitude) and magnitude > 0 else None
    raw, _ = intents(frames, rates, eligible, normalization)
    forecasts, fits = predictions(states, frames["BTCUSDT"])
    targets = allocate(raw, states, forecasts, plan["routing"])
    if set(targets) != set(plan["candidates"]):
        raise ValueError("Candidate population differs")
    calendar = pd.date_range("2025-01-01", "2026-09-01", freq="h", inclusive="left", tz="UTC")
    flags = pd.read_parquet(state_root / "event_risk_hourly.parquet").reindex(calendar)
    for key, frame in targets.items():
        if not np.isfinite(frame).all().all() or (frame.abs().sum(axis=1) > 1 + 1e-10).any():
            raise ValueError("Invalid portfolio intentions")
        score = frame.reindex(calendar, method="ffill").rename(columns=lambda s: "target_" + s)
        score["risk_scale"] = (
            np.where(flags.covered_fomc_post_release, 0.5, 1.0) if key == "event_router" else 1.0
        )
        score.to_parquet(root / (key + "_targets.parquet"))
    forecasts.to_parquet(root / "model_forecasts.parquet")
    save(root / "fits.json", fits)
    save(
        root / "normalization.json",
        {
            "training_start": "2024-01-01",
            "training_end_exclusive": "2025-01-01",
            "values": normalization,
        },
    )
    summary = {
        "schema": "multi-regime-intent-preparation.v1",
        "candidates": list(targets),
        "daily_forecasts": len(forecasts),
        "monthly_fits": len(fits),
        "forecast_availability": {
            c: int(forecasts[c].notna().sum()) for c in ["linear_probability", "mlp_probability"]
        },
        "hmm_known_decisions": int(forecasts.hmm_state.ne("unknown").sum()),
        "per_family_active_days": {k: int(v.abs().sum(axis=1).gt(0).sum()) for k, v in raw.items()},
        "strict_training_maturity": all(
            pd.Timestamp(f["maximum_label_available"]) < pd.Timestamp(f["fit_at"])
            for f in fits
            if f["status"] == "fitted"
        ),
        "model_warning_count": sum(len(f.get("warnings", [])) for f in fits),
        "source_only_profit_claims": False,
        "account_returns_computed": False,
        "orders_enabled": False,
        "production_eligible": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    save(root / "summary.json", summary)
    names = [p.name for p in root.iterdir() if p.is_file()]
    save(
        root / "completion.json",
        {
            "manifest": {name: sha(root / name) for name in names},
            "summary_sha256": sha(root / "summary.json"),
        },
    )
    print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--states", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.output.resolve(), args.states.resolve())
