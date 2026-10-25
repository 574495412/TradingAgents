"""Freeze weekly MAX28 and matched momentum controls before any return scoring."""

import argparse
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from factors.common.github_formulas import daily
from factors.common.max_effect import book, features
from research_tools.factor_lab.io import read, save, sha
from research_tools.github_factor_study.run import snapshot_sources


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    (root / "source_snapshot").mkdir()
    p = Path(__file__).with_name("plan.json")
    plan = read(p)
    data, funding = Path(plan["data"]), Path(plan["funding_data"])
    source = Path("data/research/max_effect_sources_20261026_v001")
    for record in read(source / "catalog.json")["records"]:
        if sha(source / record["file"]) != record["sha256"]:
            raise ValueError("Primary source bytes changed")
    for parent in [data, funding]:
        for name, digest in read(parent / "completion.json")["manifest"].items():
            if (
                parent.resolve() not in (parent / name).resolve().parents
                or sha(parent / name) != digest
            ):
                raise ValueError("Input source changed")
    paths = [
        Path(__file__),
        p,
        Path("factors/common/max_effect.py"),
        Path("factors/common/github_formulas.py"),
        Path("research_tools/factor_lab/io.py"),
        Path("research_tools/github_factor_study/run.py"),
        source / "catalog.json",
        data / "completion.json",
        funding / "completion.json",
        Path("../freqtrade/research/crypto/calendar_portfolio.py"),
    ]
    plan["source_hashes"] = {str(x): sha(x) for x in paths}
    plan["registered_at_utc"] = datetime.now(UTC).isoformat()
    save(root / "plan.json", plan)
    save(
        root / "registration.json",
        {"plan_sha256": sha(root / "plan.json"), "before_targets_and_account_returns": True},
    )
    snapshot_sources(paths, root / "source_snapshot")
    symbols = read(data / "prepare_plan.json")["symbols"]
    frames = {s: daily(pd.read_parquet(data / (s + "_5m.parquet"))) for s in symbols}
    f = features(frames)
    pd.concat(f, names=["symbol", "date"]).to_parquet(root / "features_daily.parquet")
    anchors = []
    for window in plan["windows"]:
        first, end = pd.Timestamp(window["start"], tz="UTC"), pd.Timestamp(window["end"], tz="UTC")
        calendar = pd.date_range(first, end, freq="h", inclusive="left")
        clocks = sorted(
            set([first] + list(pd.date_range(first, end, freq="W-MON", inclusive="left")))
        )
        for candidate in plan["candidates"]:
            rows = []
            for t in clocks:
                weights, info = book(
                    {s: x.loc[t] for s, x in f.items()}, plan["parameters"], candidate
                )
                rows.append(weights.rename(t))
                anchors.append(
                    {"window": window["id"], "candidate": candidate, "clock": t.isoformat(), **info}
                )
            targets = (
                pd.DataFrame(rows)
                .reindex(calendar, method="ffill")
                .rename(columns=lambda s: "target_" + s)
            )
            targets["risk_scale"] = 1.0
            targets["rebalance"] = calendar.isin(clocks)
            if (
                not np.isfinite(targets.to_numpy(dtype=float)).all()
                or (targets.filter(like="target_").abs().sum(axis=1) > 1 + 1e-12).any()
            ):
                raise ValueError("Invalid source-directed weights")
            targets.to_parquet(root / f"{window['id']}_{candidate}_targets.parquet")
    checks = []
    for cut in ["2025-03-01", "2025-12-01", "2026-04-01"]:
        t = pd.Timestamp(cut, tz="UTC")
        modified = {s: d.copy() for s, d in frames.items()}
        for d in modified.values():
            d.loc[d.index >= t, "close"] *= 17.0
            d.loc[d.index >= t, "quote_volume"] *= 0.001
        changed = features(modified)
        for s in symbols:
            pd.testing.assert_frame_equal(f[s].loc[:t], changed[s].loc[:t])
        checks.append({"cut": t.isoformat(), "real_future_price_volume_perturbation_passed": True})
    save(root / "anchors.json", anchors)
    save(root / "causality_checks.json", checks)
    summary = {
        "schema": "max-effect-weekly-inputs.v1",
        "anchor_cases": len(anchors),
        "formed": sum(r["status"] == "formed" for r in anchors),
        "unknown_universe": sum(r["status"] != "formed" for r in anchors),
        "symbols": len(symbols),
        "prefix_checks": checks,
        "account_returns_computed": False,
        "independent_oos": False,
        "direction_from_source": "long_highMAX28_short_lowMAX28",
    }
    save(root / "summary.json", summary)
    save(
        root / "completion.json",
        {
            "manifest": {x.name: sha(x) for x in root.iterdir() if x.is_file()},
            "account_returns_computed": False,
        },
    )
    print(summary, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    prepare(a.output)
