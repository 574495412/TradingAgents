"""Seal source-paired weekly funding targets before financial execution."""

import argparse
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from factors.common.github_formulas import daily
from factors.common.max_effect import features
from factors.crypto.funding_cross_section import accrued7, book
from research_tools.factor_lab.io import read, save, sha
from research_tools.github_factor_study.run import snapshot_sources
from research_tools.max_effect.run_accounts import verify_manifest


def load_bars(plan, variant):
    parent, supplement = Path(plan["data"]), Path(plan["funding_data"])
    symbols = read(parent / "prepare_plan.json")["symbols"]
    frames = {}
    for s in symbols:
        frame = pd.read_parquet(parent / (s + "_5m.parquet"))
        if variant == "api_revision":
            revision = pd.read_parquet(supplement / (s + "_api_revision_5m.parquet"))
            if not frame.index.is_unique or not revision.index.isin(frame.index).all():
                raise ValueError("Revision cannot align with sealed market")
            frame.loc[revision.index, revision.columns] = revision
        elif variant != "archive":
            raise ValueError("Unregistered source variant")
        frames[s] = frame
    return frames


def prepare(root, supplement):
    root.mkdir(parents=True, exist_ok=False)
    p = Path(__file__).with_name("plan.json")
    plan = read(p)
    parent = Path(plan["data"])
    old_inputs = Path("results/research/max_effect_inputs_20261026_v002")
    for source in [parent, supplement, old_inputs]:
        verify_manifest(source)
    if read(supplement / "plan.json")["study_plan_sha256"] != sha(p):
        raise ValueError("Supplement belongs to another registered study")
    for path, digest in read(supplement / "raw_source_bindings.json").items():
        if sha(Path(path)) != digest:
            raise ValueError("Official response bytes changed")
    paths = [
        p,
        Path(__file__),
        Path("factors/crypto/funding_cross_section.py"),
        Path("factors/common/max_effect.py"),
        Path("factors/common/github_formulas.py"),
        Path("research_tools/factor_lab/io.py"),
        Path("research_tools/github_factor_study/run.py"),
        Path("research_tools/max_effect/run_accounts.py"),
        Path("../freqtrade/research/crypto/calendar_portfolio.py"),
        parent / "completion.json",
        supplement / "completion.json",
        old_inputs / "completion.json",
    ]
    plan = {
        **plan,
        "funding_data": str(supplement),
        "parameters": read(old_inputs / "plan.json")["parameters"],
        "source_variants": ["archive", "api_revision"],
        "registered_at_utc": datetime.now(UTC).isoformat(),
        "implementation_notes": [
            "Accrued rates rounded to 12 decimals solely to remove floating summation noise; fixed before targets or PnL.",
            "Only one registered day of API bars is overlaid; every other row and source remains unchanged.",
        ],
        "source_hashes": {str(x): sha(x) for x in paths},
    }
    save(root / "plan.json", plan)
    save(
        root / "registration.json",
        {"plan_sha256": sha(root / "plan.json"), "before_targets_and_returns": True},
    )
    (root / "source_snapshot").mkdir()
    snapshot_sources(paths, root / "source_snapshot")
    rates = {
        s: pd.read_parquet(supplement / (s + "_exact.parquet"))
        for s in read(parent / "prepare_plan.json")["symbols"]
    }
    anchors, checks, reuse = [], [], []
    for variant in plan["source_variants"]:
        bars = load_bars(plan, variant)
        frames = {s: daily(b) for s, b in bars.items()}
        f = features(frames)
        pd.concat(f, names=["symbol", "date"]).to_parquet(
            root / (variant + "_features_daily.parquet")
        )
        for window in plan["windows"]:
            first, end = (
                pd.Timestamp(window["start"], tz="UTC"),
                pd.Timestamp(window["end"], tz="UTC"),
            )
            calendar = pd.date_range(first, end, freq="h", inclusive="left")
            clocks = sorted(
                set([first] + list(pd.date_range(first, end, freq="W-MON", inclusive="left")))
            )
            signals = {t: accrued7(rates, t) for t in clocks}
            for candidate in plan["candidates"]:
                values = []
                for t in clocks:
                    w, info = book(
                        {s: x.loc[t] for s, x in f.items()},
                        signals[t],
                        plan["parameters"],
                        candidate,
                    )
                    values.append(w.rename(t))
                    anchors.append(
                        {
                            "variant": variant,
                            "window": window["id"],
                            "candidate": candidate,
                            "clock": t.isoformat(),
                            "funding7": signals[t].to_dict(),
                            **info,
                        }
                    )
                targets = (
                    pd.DataFrame(values)
                    .reindex(calendar, method="ffill")
                    .rename(columns=lambda s: "target_" + s)
                )
                targets["risk_scale"], targets["rebalance"] = 1.0, calendar.isin(clocks)
                if (
                    not np.isfinite(targets.to_numpy(dtype=float)).all()
                    or (targets.filter(like="target_").abs().sum(axis=1) > 1 + 1e-12).any()
                ):
                    raise ValueError("Invalid source-paired target")
                targets.to_parquet(root / f"{variant}_{window['id']}_{candidate}_targets.parquet")
                if variant == "archive" and candidate == "momentum28_control":
                    old = pd.read_parquet(
                        old_inputs / f"{window['id']}_{candidate}_targets.parquet"
                    )
                    pd.testing.assert_frame_equal(targets, old, check_freq=False)
                    reuse.append(
                        {
                            "window": window["id"],
                            "targets_identical": True,
                            "old_target_sha256": sha(
                                old_inputs / f"{window['id']}_{candidate}_targets.parquet"
                            ),
                        }
                    )
        for cut in ["2025-03-01", "2025-12-01", "2026-04-01"]:
            t = pd.Timestamp(cut, tz="UTC")
            altered = {s: d.copy() for s, d in frames.items()}
            altered_rates = {s: d.copy() for s, d in rates.items()}
            for d in altered.values():
                d.loc[d.index >= t, "close"] *= 17
                d.loc[d.index >= t, "quote_volume"] *= 0.001
            for d in altered_rates.values():
                d.loc[d.index >= t, "funding_rate"] *= 100
            changed = features(altered)
            for s in f:
                pd.testing.assert_frame_equal(f[s].loc[:t], changed[s].loc[:t])
            pd.testing.assert_series_equal(accrued7(rates, t), accrued7(altered_rates, t))
            for candidate in plan["candidates"]:
                a, _ = book(
                    {s: x.loc[t] for s, x in f.items()},
                    accrued7(rates, t),
                    plan["parameters"],
                    candidate,
                )
                b, _ = book(
                    {s: x.loc[t] for s, x in changed.items()},
                    accrued7(altered_rates, t),
                    plan["parameters"],
                    candidate,
                )
                pd.testing.assert_series_equal(a, b)
            checks.append(
                {
                    "variant": variant,
                    "cut": cut,
                    "actual_future_price_quote_funding_perturbation_passed": True,
                }
            )
    save(root / "anchors.json", anchors)
    save(root / "causality_checks.json", checks)
    save(root / "control_reuse_inputs.json", reuse)
    save(
        root / "summary.json",
        {
            "anchor_cases": len(anchors),
            "source_variants": 2,
            "statuses": {
                k: sum(a["status"] == k for a in anchors)
                for k in sorted({a["status"] for a in anchors})
            },
            "prefix_checks": checks,
            "returns_computed": False,
            "independent_oos": False,
        },
    )
    save(
        root / "completion.json",
        {
            "manifest": {x.name: sha(x) for x in root.iterdir() if x.is_file()},
            "returns_computed": False,
        },
    )
    print(read(root / "summary.json"), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    a = p.parse_args()
    prepare(a.output, a.data)
