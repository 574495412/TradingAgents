"""Freeze source-sizing inputs and independent scalar checks before returns."""

import argparse
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from factors.common.github_formulas import daily
from factors.common.regime_books import source_features
from factors.portfolio.regime_allocation import bounded_mix
from factors.portfolio.source_sizing import sized_intent
from research_tools.factor_lab.io import read, save, sha
from research_tools.github_factor_study.run import snapshot_sources

RULES = {"trend": "pst_ewmac32", "breakout": "pst_breakout10", "reversion": "mr_wings4"}


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    plan_path = Path(__file__).with_name("sizing_plan.json")
    plan = read(plan_path)
    p = plan["parameters"]
    sources = Path(plan["source_pack"])
    parent = Path(plan["parent_inputs"])
    data = Path(plan["data"])
    if (
        sha(sources / "source_review.json")
        != read(sources / "completion.json")["source_review_sha256"]
    ):
        raise ValueError("Pinned sizing source receipt changed")
    for record in read(sources / "source_review.json")["records"]:
        if sha(Path(record["local"])) != record["sha256"]:
            raise ValueError("Sizing source changed")
    for name, digest in read(parent / "completion.json")["manifest"].items():
        if Path(name).name != name or sha(parent / name) != digest:
            raise ValueError("Parent inputs changed")
    paths = [
        Path(__file__),
        plan_path,
        Path("factors/portfolio/source_sizing.py"),
        Path("factors/common/regime_books.py"),
        Path("factors/common/github_formulas.py"),
        Path("factors/portfolio/regime_allocation.py"),
        Path("research_tools/factor_lab/io.py"),
        Path("research_tools/github_factor_study/run.py"),
        parent / "completion.json",
        sources / "source_review.json",
        data / "completion.json",
    ]
    plan["registered_at"] = datetime.now(UTC).isoformat()
    plan["source_hashes"] = {str(s.resolve()): sha(s) for s in paths}
    save(root / "plan.json", plan)
    save(
        root / "registration.json",
        {"plan_sha256": sha(root / "plan.json"), "before_new_targets_and_returns": True},
    )
    (root / "source_snapshot").mkdir()
    snapshot_sources(paths, root / "source_snapshot")
    for name, digest in read(data / "completion.json")["manifest"].items():
        if Path(name).name != name or sha(data / name) != digest:
            raise ValueError("Official data changed")
    symbols = read(data / "prepare_plan.json")["symbols"]
    features = {
        s: source_features(daily(pd.read_parquet(data / (s + "_5m.parquet")))).shift(1)
        for s in symbols
    }
    eligible = pd.read_parquet(Path(plan["states"]) / "eligible_daily.parquet")
    normalization = read(parent / "normalization.json")["values"]
    dates = pd.date_range("2025-01-01", "2026-09-01", freq="D", inclusive="left", tz="UTC")
    calendar = pd.date_range("2025-01-01", "2026-09-01", freq="h", inclusive="left", tz="UTC")
    keys = [
        "trend_conviction",
        "trend_risk16",
        "trend_risk16_buffer10",
        "breakout_risk16",
        "reversion_risk16",
    ]
    target = {k: pd.DataFrame(0.0, index=dates, columns=symbols) for k in keys}
    buffer = target["trend_risk16"].copy()
    maximum_error = 0.0
    checked = 0
    for clock in dates:
        names = [s for s in sorted(symbols) if eligible.loc[clock, s]]
        vol = pd.Series({s: features[s].loc[clock, "relative_vol"] for s in names})
        for family, factor in RULES.items():
            raw = pd.Series({s: features[s].loc[clock, factor] for s in names})
            sized = sized_intent(raw, vol, normalization[factor], p)
            key = family + "_risk16"
            target[key].loc[clock, sized["risk"].index] = sized["risk"]
            if family == "trend":
                target["trend_conviction"].loc[clock, sized["conviction"].index] = sized[
                    "conviction"
                ]
                target["trend_risk16_buffer10"].loc[clock] = target[key].loc[clock]
                buffer.loc[clock, sized["buffer"].index] = sized["buffer"]
            # Scalar reference follows source cash-risk/forecast equations, without
            # calling the sizing helper; check every selected symbol/day/rule.
            ranked = sorted(
                names, key=lambda s: (-abs(np.clip(raw[s] / normalization[factor], -3, 3)), s)
            )
            selected = [
                s
                for s in ranked
                if np.isfinite(raw[s])
                and np.isfinite(vol[s])
                and abs(np.clip(raw[s] / normalization[factor], -3, 3))
                > p["existing_selection_threshold"]
            ][:10]
            base = {
                s: p["annual_risk_target"]
                / (p["annualization"] ** 0.5)
                / max(vol[s], p["relative_volatility_floor"])
                / len(selected)
                for s in selected
            }
            desired = {
                s: base[s] * min(20, max(-20, raw[s] / normalization[factor] * 10)) / 10
                for s in selected
            }
            gross = sum(abs(v) for v in desired.values())
            cap = min(1.0, 0.8 / gross) if gross else 1.0
            reference = {s: desired.get(s, 0) * cap / 0.8 for s in names}
            error = max((abs(sized["risk"][s] - reference[s]) for s in names), default=0)
            maximum_error = max(maximum_error, error)
            checked += len(selected)
            if error > 1e-12:
                raise ValueError("Independent source sizing differs")
    target["equal_mix"] = target["trend_risk16"] * 0.0
    for clock in dates:
        books = {
            k: target[k].loc[clock] for k in ("trend_risk16", "breakout_risk16", "reversion_risk16")
        }
        target["equal_mix"].loc[clock] = bounded_mix(books, dict.fromkeys(books, 1 / 3))
    files = []
    for candidate in plan["candidates"]:
        if candidate in ("trend", "breakout", "mean_reversion", "cash"):
            frame = pd.read_parquet(parent / (candidate + "_targets.parquet"))
        else:
            frame = (
                target[candidate]
                .reindex(calendar, method="ffill")
                .rename(columns=lambda s: "target_" + s)
            )
            frame["risk_scale"] = 1.0
            if candidate == "trend_risk16_buffer10":
                for symbol in symbols:
                    frame["buffer_" + symbol] = buffer[symbol].reindex(calendar, method="ffill")
        weights = frame.filter(like="target_")
        if (
            not np.isfinite(weights).all().all()
            or (weights.abs().sum(axis=1) > 1 + 1e-10).any()
            or (weights.ne(0).sum(axis=1) > 10).any()
        ):
            raise ValueError("Sizing input budget or name limit invalid")
        name = candidate + "_targets.parquet"
        frame.to_parquet(root / name)
        files.append(name)
    save(
        root / "summary.json",
        {
            "schema": "source-forecast-sizing-preparation.v1",
            "candidates": plan["candidates"],
            "source_files": 6,
            "decision_days": len(dates),
            "independent_scalar_positions": checked,
            "maximum_scalar_error": maximum_error,
            "account_returns_computed": False,
            "mean_unit_gross": {k: float(v.abs().sum(axis=1).mean()) for k, v in target.items()},
            "orders_enabled": False,
            "production_eligible": False,
        },
    )
    files += ["summary.json", "plan.json", "registration.json"]
    save(
        root / "completion.json",
        {
            "manifest": {name: sha(root / name) for name in files},
            "summary_sha256": sha(root / "summary.json"),
        },
    )
    audit_path = root.with_name(root.name + "_audit.json")
    save(
        audit_path,
        {
            "passed": True,
            "input_completion_sha256": sha(root / "completion.json"),
            "maximum_scalar_error": maximum_error,
            "independent_positions_checked": checked,
            "source_files_hash_checked": 6,
            "controls_reused_from_immutable_inputs": True,
            "profit_verified": False,
            "orders_enabled": False,
        },
    )
    print(read(root / "summary.json"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args().output.resolve())
