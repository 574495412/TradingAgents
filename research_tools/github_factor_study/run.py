"""Source-pinned GitHub formulas, shared account economics and immutable continuous results."""

import argparse
import hashlib
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from factors.common.github_formulas import PRICE_RULES, QLIB_RULES, books, daily, formulas
from research_tools.factor_discovery.run import write_account
from research_tools.factor_lab.io import read, save, sha
from research_tools.factor_lab.metrics import portfolio_metrics, prediction_metrics
from research_tools.factor_lab.study import executor
from research_tools.factor_lab_audit import audit_account


def snapshot_sources(paths, folder):
    manifest = {}
    for path in paths:
        if path.suffix != ".py":
            continue
        identity = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:12]
        name = identity + "_" + path.name
        shutil.copyfile(path, folder / name)
        manifest[str(path.resolve())] = {"snapshot": name, "sha256": sha(path)}
    save(folder / "manifest.json", manifest)
    return manifest


def select(rows, plan):
    out = []
    for c in plan["candidates"]:
        cases = [r for r in rows if r["candidate"] == c]
        if len(cases) != 2:
            raise ValueError("Incomplete two-cost screen")
        reasons = []
        for r in cases:
            m = r["metrics"]
            checks = {
                "return_below20percent": m["net_return"] >= 0.20,
                "drawdown": m["max_drawdown"] <= 0.15,
                "complete_trades": m["closed_trades"] >= 30,
                "closed_profit_factor": m["profit_factor"] is not None
                and m["profit_factor"] >= 1.1,
                "sharpe": m["sharpe"] is not None and m["sharpe"] >= 0.5,
            }
            reasons.extend(f"{r['slippage_bps']}:{k}" for k, v in checks.items() if not v)
        out.append({"candidate": c, "passed": not reasons, "reasons": reasons, "cases": cases})
    rank = sorted(
        [r for r in out if r["candidate"] != "liquid_long_control"],
        key=lambda a: (
            -min(r["metrics"]["net_return"] for r in a["cases"]),
            max(r["metrics"]["max_drawdown"] for r in a["cases"]),
            a["candidate"],
        ),
    )
    return {
        "admitted_candidates": [a["candidate"] for a in rank if a["passed"]],
        "research_followups": [a["candidate"] for a in rank[:3]],
        "assessments": out,
        "later_returns_used": False,
        "production_eligible": False,
    }


def run(root):
    started = time.monotonic()
    root.mkdir(parents=True, exist_ok=False)
    plan = read(Path(__file__).with_name("plan.json"))
    data = Path(plan["data"])
    receipt = read(data / "completion.json")
    for name, digest in receipt["manifest"].items():
        if sha(data / name) != digest:
            raise ValueError("Sealed data changed")
    sources = Path(plan["source_pack"])
    for record in read(sources / "file_receipts.json"):
        if sha(Path(record["local_path"])) != record["sha256"]:
            raise ValueError("Pinned upstream source changed")
    paths = [
        Path(__file__),
        Path(__file__).with_name("plan.json"),
        Path("factors/common/github_formulas.py"),
        Path(plan["engine"]),
        Path("research_tools/factor_discovery/run.py"),
        Path("research_tools/factor_lab/metrics.py"),
        Path("research_tools/factor_lab_audit.py"),
        data / "completion.json",
        sources / "file_receipts.json",
    ] + list(sources.glob("*/metadata.json"))
    plan["registered_at"] = datetime.now(UTC).isoformat()
    plan["source_hashes"] = {str(p.resolve()): sha(p) for p in paths}
    plan["data_manifest"] = receipt["manifest"]
    save(root / "plan.json", plan)
    save(
        root / "registration.json",
        {
            "plan_sha256": sha(root / "plan.json"),
            "registered_before_calibration_and_accounts": True,
        },
    )
    snap = root / "source_snapshot"
    snap.mkdir()
    snapshot_sources(paths, snap)
    symbols = read(data / "prepare_plan.json")["symbols"]
    bars = {s: pd.read_parquet(data / (s + "_5m.parquet")) for s in symbols}
    rates = {s: pd.read_parquet(data / (s + "_funding.parquet")) for s in symbols}
    ds = {s: daily(bars[s]) for s in symbols}
    fs = {s: formulas(ds[s]) for s in symbols}
    dates = ds[symbols[0]].index
    train = (dates >= pd.Timestamp("2024-01-01", tz="UTC")) & (
        dates + pd.Timedelta(days=1) < pd.Timestamp("2025-01-01", tz="UTC")
    )
    normalization = {
        k: float(
            np.nanmean(
                np.abs(
                    pd.DataFrame({s: f[k].shift(1) for s, f in fs.items()}).loc[train].to_numpy()
                )
            )
        )
        for k in PRICE_RULES
    }
    if any(not np.isfinite(v) or v <= 0 for v in normalization.values()):
        raise ValueError("No training-only forecast scale")
    target = (
        pd.concat(
            {s: (d.open.shift(-1) / d.open - 1) for s, d in ds.items()}, names=["symbol", "time"]
        )
        .swaplevel()
        .sort_index()
    )
    directions = {}
    diagnostics = {}
    for k in QLIB_RULES:
        factor = (
            pd.concat({s: f[k].shift(1) for s, f in fs.items()}, names=["symbol", "time"])
            .swaplevel()
            .sort_index()
        )
        times = factor.index.get_level_values("time")
        tm = (times >= pd.Timestamp("2024-01-01", tz="UTC")) & (
            times + pd.Timedelta(days=1) < pd.Timestamp("2025-01-01", tz="UTC")
        )
        m = prediction_metrics(factor.loc[tm], target.loc[tm])
        if m["rank_ic"] is None:
            raise ValueError("Cannot orient factor from training cohort")
        directions[k] = 1 if m["rank_ic"] >= 0 else -1
        diagnostics[k] = m
    save(
        root / "training_calibration.json",
        {
            "normalization": normalization,
            "directions": directions,
            "diagnostics": diagnostics,
            "end": "2025-01-01",
            "future_returns_used": False,
        },
    )
    raw_scores, vol, _, eligible = books(bars, rates, normalization, directions)
    eligible.to_parquet(root / "eligible_daily.parquet")
    calendar = pd.date_range("2024-01-01", "2026-09-01", freq="h", inclusive="left", tz="UTC")
    scores = {}
    for candidate in plan["candidates"]:
        score = raw_scores[candidate].reindex(calendar, method="ffill")
        for s in symbols:
            score["vol_" + s] = vol[candidate][s].reindex(calendar, method="ffill")
        score.to_parquet(root / (candidate + "_scores.parquet"))
        scores[candidate] = score
    engine = executor(plan["engine"])
    rows = []
    audits = []

    def one(candidate, window, cost, phase, capital=10000):
        config = {**plan["account"], "capital": capital, "slippage_bps": cost}
        if candidate in QLIB_RULES:
            config["neutral_target"] = True
        elif candidate == "sentinel_champion":
            config["max_positions"] = 5
        elif candidate == "sentinel_carry21":
            config.update(max_positions=10, neutral_target=True)
        elif candidate == "sentinel_carry14_buffer":
            config.update(max_positions=16, weight_mode="rank_buffer", side_count=8, keep_buffer=4)
        a = engine.replay(bars, rates, scores[candidate], config, window["start"], window["end"])
        boundary = pd.date_range(window["start"], window["end"], freq="D", tz="UTC")
        benchmark = bars["BTCUSDT"].close.reindex(boundary - pd.Timedelta(minutes=5))
        benchmark.index = boundary
        a["metrics"] = portfolio_metrics(a["curve"], a["trades"], a["events"], capital, benchmark)
        a["metrics"]["mean_absolute_net_exposure"] = float(
            (a["curve"].net_exposure / a["curve"].equity).abs().mean()
        )
        key = f"{phase}_{candidate}_cap{capital}_slip{cost}"
        write_account(root / "accounts" / key, a, config, window)
        row = {
            "id": key,
            "candidate": candidate,
            "window": window["id"],
            "phase": phase,
            "capital": capital,
            "slippage_bps": cost,
            "start": window["start"],
            "end": window["end"],
            "metrics": a["metrics"],
            "capacity": a["capacity"],
            "risk_halted": a["risk_halted"],
            "result_sha256": sha(root / "accounts" / key / "result.json"),
        }
        rows.append(row)
        audits.append(audit_account(root, row, bars, rates))
        print(phase, candidate, cost, capital, round(a["metrics"]["net_return"], 6), flush=True)

    for c in plan["candidates"]:
        for cost in (3, 10):
            one(c, plan["screen"], cost, "screen2025")
    selection = select(rows, plan)
    save(root / "selection.json", selection)
    follow = selection["research_followups"] + ["liquid_long_control"]
    for w in plan["diagnostics"]:
        for c in follow:
            for cost in (3, 10):
                one(c, w, cost, w["id"])
    w = plan["diagnostics"][-1]
    for c in selection["research_followups"]:
        for capital in (100000, 1000000):
            for cost in (3, 10):
                one(c, w, cost, "capacity", capital)
    summary = {
        "schema": "github-factor-local-results.v1",
        "plan_sha256": sha(root / "plan.json"),
        "selection": selection,
        "account_rows": rows,
        "audit": {
            "passed": all(a["passed"] for a in audits),
            "accounts": len(audits),
            "maximum_nav_error": max(a["max_nav_error"] for a in audits),
        },
        "audit_accounts": audits,
        "calibration": read(root / "training_calibration.json"),
        "universe": plan["universe"],
        "source_deviations": plan["source_deviations"],
        "elapsed_seconds": time.monotonic() - started,
        "independent_oos": False,
        "orders_enabled": False,
        "production_eligible": False,
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
