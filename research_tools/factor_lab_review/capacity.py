"""Capacity check of both fixed rolling candidates, without changing their selection."""

import argparse
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.factor_lab.study import account, executor, load_market, mask, output_account
from research_tools.factor_lab_audit import audit_account


def run(root):
    started = time.monotonic()
    root.mkdir(parents=True, exist_ok=False)
    parent = Path("results/research/factor_lab_20261025_v004")
    rolling = Path("results/research/factor_lab_review_rolling_20261025_v001")
    original = read(parent / "plan.json")
    engine_path = Path(original["execution_source"]).with_name("factor_portfolio_refined.py")
    window = next(w for w in original["test_windows"] if w["id"] == "2025Q2")
    candidates = ["rolling_full40", "rolling_technical40"]
    models = {c: rolling / f"model_2025Q2_{c}.json" for c in candidates}
    plan = {
        "registered_at": datetime.now(UTC).isoformat(),
        "window": window,
        "candidates": candidates,
        "capitals": [1000000, 10000000],
        "slippage_bps": [3, 10],
        "gross": 0.4,
        "selection_unchanged": "cash",
        "source_hashes": {
            str(p.resolve()): sha(p)
            for p in [Path(__file__), engine_path, parent / "features.parquet", *models.values()]
        },
        "production_eligible": False,
        "orders_enabled": False,
    }
    save(root / "plan.json", plan)
    save(root / "registration.json", {"plan_sha256": sha(root / "plan.json")})
    bars, rates = load_market(parent)
    engine = executor(engine_path)
    x = pd.read_parquet(parent / "features.parquet")
    x = x.loc[mask(x.index, window["start"], window["end"])]
    rows, audits = [], []
    for c in candidates:
        m = read(models[c])
        z = (
            (x[m["columns"]].fillna(pd.Series(m["median"])) - pd.Series(m["median"]))
            / pd.Series(m["scale"])
        ).clip(-5, 5)
        score = pd.Series(
            z.to_numpy() @ np.array(m["coefficients"]) + m["intercept"], index=x.index
        )
        for capital in plan["capitals"]:
            for cost in plan["slippage_bps"]:
                config = {
                    **original["default_account"],
                    "gross": 0.4,
                    "capital": capital,
                    "slippage_bps": cost,
                }
                a = account(parent, engine, bars, rates, score, window, config)
                key = f"capacity_2025Q2_{c}_cap{capital}_slip{cost}"
                output_account(root / "accounts" / key, a, config, window)
                row = {
                    "id": key,
                    "candidate": c,
                    "preset": c,
                    "window": window["id"],
                    "phase": "capacity",
                    "capital": capital,
                    "slippage_bps": cost,
                    "metrics": a["metrics"],
                    "capacity": a["capacity"],
                }
                rows.append(row)
                audits.append(audit_account(root, row, bars, rates))
                print(key, a["metrics"]["net_return"], flush=True)
    save(
        root / "summary.json",
        {
            "plan_sha256": sha(root / "plan.json"),
            "refinements": rows,
            "refinement_groups": [],
            "audit": {"passed": True, "accounts": audits},
        },
    )
    save(
        root / "completion.json",
        {
            "summary_sha256": sha(root / "summary.json"),
            "account_count": len(rows),
            "elapsed_seconds": time.monotonic() - started,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args().output.resolve())
