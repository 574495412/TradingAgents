"""Copy frozen family intentions only in their preregistered observable domains."""

import argparse
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.github_factor_study.run import snapshot_sources


def gate_frame(frame, states, allowed):
    if not set(allowed) <= {"trend_up", "trend_down", "range", "volatility_breakout"}:
        raise ValueError("Unknown/liquidity-stress states must never be enabled")
    for index in (frame.index, states.index):
        if (
            not isinstance(index, pd.DatetimeIndex)
            or index.tz is None
            or not index.is_unique
            or not index.is_monotonic_increasing
        ):
            raise ValueError("Ordered unique timezone-aware clocks required")
    known = states.state.reindex(frame.index, method="ffill").fillna("unknown")
    mask = known.isin(allowed)
    out = frame.copy()
    columns = [c for c in frame if c.startswith("target_")]
    if not columns or "risk_scale" not in frame:
        raise ValueError("Missing target/risk columns")
    out.loc[~mask, columns] = 0.0
    return out, mask


def sealed(root):
    for name, digest in read(root / "completion.json")["manifest"].items():
        if Path(name).name != name or sha(root / name) != digest:
            raise ValueError("Sealed input changed: " + str(root / name))


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    registered = Path("docs/research/regime_program/module_domains_20261026")
    sealed(registered)
    plan = read(registered / "plan.json")
    parent, states_root = Path(plan["parent_inputs"]), Path(plan["states"])
    sealed(parent)
    sealed(states_root)
    cards = read(registered / "cards.json")
    for path, digest in read(registered / "registration.json")["source_hashes"].items():
        if sha(Path(path)) != digest:
            raise ValueError("Domain registration source changed")
    paths = [
        Path(__file__),
        registered / "completion.json",
        registered / "plan.json",
        registered / "cards.json",
        registered / "protocol.json",
        parent / "completion.json",
        states_root / "completion.json",
        Path("research_tools/factor_lab/io.py"),
        Path("research_tools/github_factor_study/run.py"),
    ]
    plan.update(
        registered_at=datetime.now(UTC).isoformat(),
        source_hashes={str(p.resolve()): sha(p) for p in paths},
    )
    save(root / "plan.json", plan)
    save(
        root / "registration.json",
        {
            "plan_sha256": sha(root / "plan.json"),
            "before_domain_targets_and_returns": True,
            "independent_oos": False,
        },
    )
    (root / "source_snapshot").mkdir()
    snapshot_sources(paths, root / "source_snapshot")
    states = pd.read_parquet(states_root / "states_daily.parquet")
    profiles = {c["id"]: c for c in cards["cards"]}
    checks = []
    activity = []
    for key in plan["screen_modules"]:
        card = profiles[key]
        frame = pd.read_parquet(parent / (card["family"] + "_targets.parquet"))
        target, mask = gate_frame(frame, states, card["allowed_states"])
        columns = [c for c in frame if c.startswith("target_")]
        if (
            not np.isfinite(target.to_numpy()).all()
            or (target[columns].abs().sum(axis=1) > 1 + 1e-10).any()
        ):
            raise ValueError("Invalid masked target budgets")
        pd.testing.assert_frame_equal(target.loc[mask, columns], frame.loc[mask, columns])
        pd.testing.assert_series_equal(target.risk_scale, frame.risk_scale)
        if target.loc[~mask, columns].ne(0).any().any():
            raise ValueError("Off-domain intentions retained")
        for cut in ("2025-04-01", "2026-01-01", "2026-07-01"):
            clock = pd.Timestamp(cut, tz="UTC")
            prefix, _ = gate_frame(
                frame.loc[frame.index < clock],
                states.loc[states.index < clock],
                card["allowed_states"],
            )
            pd.testing.assert_frame_equal(prefix, target.loc[target.index < clock])
            future = frame.copy()
            future.loc[clock:, columns] *= 1000
            changed = states.copy()
            changed.loc[clock:, "state"] = "unknown"
            perturbed, _ = gate_frame(future, changed, card["allowed_states"])
            pd.testing.assert_frame_equal(perturbed.loc[perturbed.index < clock], prefix)
            checks.append(
                {"module": key, "cut": cut, "prefix_equal": True, "future_mutation_equal": True}
            )
        target.to_parquet(root / (key + "_targets.parquet"))
        screen = (frame.index >= pd.Timestamp(plan["screen"]["start"], tz="UTC")) & (
            frame.index < pd.Timestamp(plan["screen"]["end"], tz="UTC")
        )
        activity.append(
            {
                "module": key,
                "family": card["family"],
                "allowed_states": card["allowed_states"],
                "screen_hours": int(screen.sum()),
                "screen_domain_hours": int(mask.loc[screen].sum()),
                "screen_nonzero_intent_hours": int(
                    target.loc[screen, columns].ne(0).any(axis=1).sum()
                ),
            }
        )
    save(
        root / "summary.json",
        {
            "schema": "causal-domain-targets.v1",
            "profiles": activity,
            "causality_checks": checks,
            "account_returns_computed": False,
            "independent_oos": False,
            "orders_enabled": False,
        },
    )
    names = [
        "plan.json",
        "registration.json",
        "summary.json",
        *[k + "_targets.parquet" for k in plan["screen_modules"]],
    ]
    save(root / "completion.json", {"manifest": {n: sha(root / n) for n in names}})
    print({"root": str(root), "profiles": activity, "causal_checks": len(checks)}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args().output.resolve())
