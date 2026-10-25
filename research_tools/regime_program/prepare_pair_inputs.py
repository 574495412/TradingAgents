"""Frozen causal four-hour signals from already audited formation models."""

import argparse
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.github_factor_study.run import snapshot_sources


def observation(raw, rates, model, clock, parameters):
    clock = pd.Timestamp(clock)
    prices = {}
    for s in (model["x"], model["y"]):
        window = raw[s].loc[
            (raw[s].index >= clock - pd.Timedelta(hours=4)) & (raw[s].index < clock)
        ]
        prices[s] = float(window.close.iloc[-1]) if len(window) == 48 else None
    valid = all(p is not None and np.isfinite(p) and p > 0 for p in prices.values())
    z = (
        (
            np.log(prices[model["y"]])
            - model["alpha"]
            - model["beta"] * np.log(prices[model["x"]])
            - model["mean"]
        )
        / model["sigma"]
        if valid
        else None
    )
    estimated = abs(z) * model["sigma"] / (1 + model["beta"]) if valid else None
    daily = {}
    max_funding = None
    floor = clock.floor("D")
    days = pd.date_range(floor - pd.Timedelta(days=5), floor, freq="D", inclusive="left")
    for s in prices:
        past = rates[s].loc[
            (rates[s].index >= days[0])
            & (rates[s].index < floor)
            & (rates[s].index + pd.Timedelta(minutes=5) <= clock)
        ]
        sums = past.funding_rate.resample("D").sum(min_count=1).reindex(days)
        daily[s] = float(sums.mean()) if sums.notna().all() else None
        if len(past):
            last = past.index[-1].isoformat()
            max_funding = max(max_funding, last) if max_funding else last
    known = valid and all(r is not None and np.isfinite(r) for r in daily.values())
    projected = (
        max(
            0.0,
            7
            * np.sign(z)
            * (model["beta"] * daily[model["x"]] - daily[model["y"]])
            / (1 + model["beta"]),
        )
        if known
        else None
    )
    roundtrip = 2 * (parameters["stress_fee_bps"] + parameters["stress_slippage_bps"]) / 10000
    hurdle = max(2 * roundtrip, roundtrip + projected) if known else None
    eligible = bool(
        known and parameters["z_entry"] <= abs(z) < parameters["z_stop"] and estimated > hurdle
    )
    return {
        "pair_key": model["x"] + "/" + model["y"],
        "x": model["x"],
        "y": model["y"],
        "beta": model["beta"],
        "alpha": model["alpha"],
        "mean": model["mean"],
        "sigma": model["sigma"],
        "z": float(z) if valid else None,
        "signal_valid": bool(valid),
        "entry_eligible": eligible,
        "reference_prices": prices,
        "estimated_convergence_unit_gross": float(estimated) if valid else None,
        "funding_daily_means": daily,
        "projected_funding_cost_unit_gross": float(projected) if known else None,
        "hurdle_unit_gross": float(hurdle) if known else None,
        "maximum_funding_observation": max_funding,
        "maximum_price_observation": (clock - pd.Timedelta(minutes=5)).isoformat()
        if valid
        else None,
    }


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    plan_path = Path(__file__).with_name("pair_execution_plan.json")
    plan = read(plan_path)
    original = read(Path(plan["formation_plan"]))
    formation = Path(plan["formation"])
    data = Path(plan["data"])
    receipt = read(formation / "completion.json")
    for name, digest in receipt["manifest"].items():
        if Path(name).name != name or sha(formation / name) != digest:
            raise ValueError("Formation changed")
    audit = read(Path(plan["formation_audit"]))
    if not audit["passed"] or audit["completion_sha256"] != sha(formation / "completion.json"):
        raise ValueError("Formation not independently verified")
    if (
        sha(Path(plan["formation_plan"]))
        != read(formation / "plan.json")["source_hashes"][
            str(Path(plan["formation_plan"]).resolve())
        ]
    ):
        raise ValueError("Original parameters changed after formation")
    paths = [
        Path(__file__),
        plan_path,
        Path(plan["formation_plan"]),
        formation / "completion.json",
        Path(plan["formation_audit"]),
        data / "completion.json",
        Path("research_tools/factor_lab/io.py"),
        Path("research_tools/github_factor_study/run.py"),
    ]
    plan["registered_at"] = datetime.now(UTC).isoformat()
    plan["source_hashes"] = {str(p.resolve()): sha(p) for p in paths}
    save(root / "plan.json", plan)
    save(
        root / "registration.json",
        {"plan_sha256": sha(root / "plan.json"), "before_signals_and_account_returns": True},
    )
    (root / "source_snapshot").mkdir()
    snapshot_sources(paths, root / "source_snapshot")
    for name, digest in read(data / "completion.json")["manifest"].items():
        if sha(data / name) != digest:
            raise ValueError("Sealed data changed")
    fits = {p.name: read(p) for p in formation.glob("rolling_*.json")}
    fixed = read(formation / "fixed_formation.json")
    model_hashes = {name: sha(formation / name) for name in ["fixed_formation.json", *fits]}
    models = [m for f in [fixed, *fits.values()] for m in f["selected"]]
    symbols = sorted({m[k] for m in models for k in ("x", "y")})
    raw = {s: pd.read_parquet(data / (s + "_5m.parquet")) for s in symbols}
    rates = {s: pd.read_parquet(data / (s + "_funding.parquet")) for s in symbols}
    clocks = pd.date_range("2025-01-01", "2026-09-01", freq="4h", inclusive="left", tz="UTC")
    output = {k: [] for k in original["candidates"]}
    counts = Counter()
    checks = []
    for clock in clocks:
        for key in output:
            name = (
                "fixed_formation.json" if key == "pairs_fixed" else f"rolling_{clock:%Y%m}01.json"
            )
            fit = fixed if key == "pairs_fixed" else fits[name]
            model_id = model_hashes[name]
            pairs = []
            for model in fit["selected"] if key != "cash" else []:
                row = observation(raw, rates, model, clock, original["signal"])
                row.update(model_id=model_id, fit_at=fit["fit_at"])
                pairs.append(row)
                counts[key + "_pair_observations"] += 1
                counts[key + "_entry_eligible_observations"] += row["entry_eligible"]
            output[key].append({"time": clock.isoformat(), "pairs": pairs})
    for cut in ("2025-01-05", "2026-03-15", "2026-05-20"):
        clock = pd.Timestamp(cut, tz="UTC")
        model = (fixed if clock.year == 2025 else fits[f"rolling_{clock:%Y%m}01.json"])["selected"][
            0
        ]
        expected = observation(raw, rates, model, clock, original["signal"])
        prefix = observation(
            {s: f.loc[f.index < clock] for s, f in raw.items()},
            {s: f.loc[f.index < clock] for s, f in rates.items()},
            model,
            clock,
            original["signal"],
        )
        changed = {s: f.copy() for s, f in raw.items()}
        new_rates = {s: f.copy() for s, f in rates.items()}
        for f in changed.values():
            f.loc[clock:, ["open", "high", "low", "close"]] *= 1000
        for f in new_rates.values():
            f.loc[clock:, "funding_rate"] = 1
        future = observation(changed, new_rates, model, clock, original["signal"])
        if expected != prefix or expected != future:
            raise ValueError("Future data changed pair signal")
        checks.append(
            {"clock": clock.isoformat(), "prefix_equal": True, "future_perturbation_equal": True}
        )
    for key, rows in output.items():
        save(root / (key + ".json"), rows)
    save(
        root / "summary.json",
        {
            "candidates": list(output),
            "decision_rows_per_candidate": len(clocks),
            "symbol_universe": symbols,
            "counts": dict(counts),
            "causality_checks": checks,
            "account_returns_computed": False,
            "independent_oos": False,
            "orders_enabled": False,
        },
    )
    files = ["plan.json", "registration.json", "summary.json", *[k + ".json" for k in output]]
    save(
        root / "completion.json",
        {"manifest": {n: sha(root / n) for n in files}, "account_returns_computed": False},
    )
    print({"root": str(root), "clocks": len(clocks), **dict(counts)}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args().output.resolve())
