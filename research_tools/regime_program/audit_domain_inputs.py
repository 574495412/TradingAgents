"""Independent timestamp alignment and direct source-column checks for domain masks."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from research_tools.factor_lab.io import read, save, sha


def audit(root, output):
    for name, digest in read(root / "completion.json")["manifest"].items():
        if Path(name).name != name or sha(root / name) != digest:
            raise ValueError("Domain inputs changed")
    plan = read(root / "plan.json")
    for path, digest in plan["source_hashes"].items():
        if sha(Path(path)) != digest:
            raise ValueError("Domain input source changed")
    states = pd.read_parquet(Path(plan["states"]) / "states_daily.parquet")
    cards = {c["id"]: c for c in read(Path(plan["registry_snapshot"]))["cards"]}
    checked = 0
    allowed = 0
    for key in plan["screen_modules"]:
        card = cards[key]
        base = pd.read_parquet(Path(plan["parent_inputs"]) / (card["family"] + "_targets.parquet"))
        out = pd.read_parquet(root / (key + "_targets.parquet"))
        if not base.index.equals(out.index) or not base.columns.equals(out.columns):
            raise ValueError("Mask changed clock or universe")
        prior = states.index.get_indexer(base.index, method="pad")
        eligible = np.array(
            [i >= 0 and states.iloc[i].state in card["allowed_states"] for i in prior]
        )
        columns = [c for c in base if c.startswith("target_")]
        expected = np.where(eligible[:, None], base[columns].to_numpy(), 0.0)
        if not np.array_equal(expected, out[columns].to_numpy()):
            raise ValueError("Independent domain mask differs")
        pd.testing.assert_frame_equal(base.drop(columns=columns), out.drop(columns=columns))
        if any(i >= 0 and states.index[i] > t for t, i in zip(base.index, prior, strict=True)):
            raise ValueError("Future state alignment")
        checked += len(base) * len(columns)
        allowed += int(eligible.sum())
    save(
        output,
        {
            "schema": "independent-domain-input-audit.v1",
            "passed": True,
            "input_completion_sha256": sha(root / "completion.json"),
            "target_values_checked": checked,
            "allowed_hourly_rows": allowed,
            "maximum_error": 0.0,
            "auditor_sha256": sha(Path(__file__)),
            "account_returns_computed": False,
        },
    )
    print({"passed": True, "target_values_checked": checked}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.root.resolve(), args.output.resolve())
