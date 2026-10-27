"""Freeze one monthly candidate and matching absolute-intent long control."""

import argparse
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest
from research_tools.sparse_market.identity import directives


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    path = Path(__file__).with_name("account_plan.json")
    plan = read(path)
    source = Path(plan["inputs"])
    readiness = Path(plan["readiness"])
    for root in [source, readiness]:
        verify_manifest(root)
    checked = read(Path(plan["input_audit"]))
    assert checked["passed"] and checked["inputs_completion_sha256"] == sha(
        source / "completion.json"
    )
    assert checked["source_observations_equal"]
    observations = read(source / "observations.json")
    members = read(readiness / "monthly_membership.json")
    symbols = read(readiness / "summary.json")["selected_union"]
    paths = [
        path,
        Path(__file__),
        Path(plan["input_audit"]),
        source / "completion.json",
        readiness / "completion.json",
        Path(plan["existing_bindings"]),
        Path("factors/common/monthly_tsmom.py"),
        Path("factors/crypto/perpetual_returns.py"),
        Path("research_tools/sparse_market/identity.py"),
    ]
    plan = {**plan, "symbols": symbols, "source_hashes": {str(p): sha(p) for p in paths}}
    save(output / "plan.json", plan)
    save(
        output / "registration.json",
        {
            "registered_at_utc": datetime.now(UTC).isoformat(),
            "before_monthly_targets_and_returns": True,
            "plan_sha256": sha(output / "plan.json"),
        },
    )
    bindings = read(Path(plan["existing_bindings"]))
    save(output / "execution_bindings.json", {s: bindings[s] for s in symbols})
    registry = read(Path(plan["cohort"]) / "registry.json")
    save(output / "lifecycle.json", directives(registry, symbols))
    a, b = [pd.Timestamp(plan["window"][k], tz="UTC") for k in ["start", "end"]]
    hours = pd.date_range(a, b, freq="h", inclusive="left")
    anchors, audits = [], []
    for variant in plan["variants"]:
        books = {}
        for m in members:
            if m["variant"] != variant:
                continue
            rows = {
                r["symbol"]: r
                for r in observations
                if r["variant"] == variant and r["decision"] == m["decision"]
            }
            assert set(rows) == set(m["selected"])
            book = {
                s: rows[s]["direction"] * 0.16 / len(rows) / rows[s]["annual_volatility"]
                for s in rows
            }
            scale = max(1.0, sum(abs(v) for v in book.values()))
            book = {s: v / scale for s, v in book.items()}
            t = pd.Timestamp(m["decision"])
            books[t] = pd.Series(book, dtype=float).reindex(symbols, fill_value=0.0)
            anchors.append(
                {
                    "variant": variant,
                    "decision": m["decision"],
                    "selected": m["selected"],
                    "unit_intents": book,
                }
            )
            vector = np.array(
                [rows[s]["direction"] / rows[s]["annual_volatility"] for s in m["selected"]]
            ) * (0.16 / len(rows))
            vector /= max(1.0, np.abs(vector).sum())
            np.testing.assert_allclose(vector, [book[s] for s in m["selected"]], atol=1e-15, rtol=0)
            audits.append(
                {"variant": variant, "decision": m["decision"], "scalar_vector_sizing_equal": True}
            )
        f = (
            pd.DataFrame.from_dict(books, orient="index")
            .reindex(hours, method="ffill")
            .rename(columns=lambda s: "target_" + s)
        )
        f["risk_scale"] = 1.0
        f["rebalance"] = f.index.isin(books)
        assert f.rebalance.sum() == 20 and np.isfinite(f.to_numpy(dtype=float)).all()
        for profile in plan["profiles"]:
            target = f.copy()
            if profile == "magnitude_long":
                target.loc[:, target.columns.str.startswith("target_")] = target.filter(
                    like="target_"
                ).abs()
            target.to_parquet(output / f"{variant}_{profile}_targets.parquet")
    equal = {}
    for profile in plan["profiles"]:
        left, right = [
            pd.read_parquet(output / f"{v}_{profile}_targets.parquet") for v in plan["variants"]
        ]
        pd.testing.assert_frame_equal(left, right, check_exact=True)
        equal[profile] = True
    save(output / "anchors.json", anchors)
    save(output / "sizing_checks.json", audits)
    save(
        output / "summary.json",
        {
            "source_target_equivalence": equal,
            "monthly_anchors_per_source": 20,
            "actual_financial_accounts": 0,
            "new_economic_candidates": 1,
        },
    )
    save(
        output / "completion.json",
        {"passed": True, "manifest": {p.name: sha(p) for p in output.iterdir() if p.is_file()}},
    )
    print(read(output / "summary.json"), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    prepare(p.parse_args().output)
