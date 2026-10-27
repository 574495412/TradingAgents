"""Independent BTC qualification, NumPy forecasts, weights and long-control audit."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest
from research_tools.sparse_market.audit_final_cohort import prepare_raw
from research_tools.sparse_market.identity import crypto_context
from research_tools.trend_pool_audit.audit_inputs_v2 import scalar_features, weights


def quote28(frame, t):
    rows = frame.loc[(frame.index >= t - pd.Timedelta(days=28)) & (frame.index < t)]
    assert len(rows) == 28 and rows.price_liquidity_feature_usable.eq(True).all()
    return float(np.mean(rows.quote_volume.to_numpy(dtype=float)))


def audit(inputs, output):
    verify_manifest(inputs)
    if output.exists():
        raise FileExistsError("Preserve input audit")
    plan = read(inputs / "plan.json")
    for path, digest in plan["source_hashes"].items():
        assert sha(Path(path)) == digest
    parent = Path(plan["inputs"]["parent_targets"])
    inherited_path = Path(str(parent) + "_independent_audit.json")
    inherited = read(inherited_path)
    assert inherited["passed"] and inherited["inputs_completion_sha256"] == sha(
        parent / "completion.json"
    )
    old_plan = read(parent / "plan.json")
    cohort = Path(old_plan["inputs"]["cohort"])
    raw_versions = prepare_raw(read(cohort / "plan.json"), cohort)
    registry = read(cohort / "registry.json")
    member_rows = read(inputs / "membership.json")
    anchors = read(inputs / "anchors.json")
    groups = {}
    maximum = 0.0
    for variant, raw in raw_versions.items():
        members = {}
        for r in [x for x in member_rows if x["variant"] == variant]:
            t = pd.Timestamp(r["clock"])
            btc = raw["BTCUSDT"].loc[raw["BTCUSDT"].index < t]
            valid = (
                btc.price_liquidity_feature_usable.eq(True)
                & btc.volume.gt(0)
                & btc.quote_volume.gt(0)
            )
            valid.iloc[0] = False
            history = btc.close.iloc[-61:].to_numpy(dtype=float)
            eligible = (
                int(valid.sum()) >= 180
                and len(history) == 61
                and valid.iloc[-61:].all()
                and np.var(history[1:] / history[:-1] - 1, ddof=1) > 1e-12
                and quote28(raw["BTCUSDT"], t) >= 50_000_000
            )
            assert bool(eligible) == r["BTC_eligible"]
            candidates = r["old_members"] + (["BTCUSDT"] if eligible else [])
            expected = sorted(candidates, key=lambda s: (-quote28(raw[s], t), s))[:30]
            assert expected == r["new_members"]
            members[t] = expected
        cache = {}
        for r in [x for x in anchors if x["variant"] == variant]:
            day = pd.Timestamp(r["decision"])
            assert r["new_members"] == members[pd.Timestamp(r["membership_at"])]
            rows = {}
            for s in r["new_members"]:
                c = crypto_context(registry, s, day)
                if c["blocked"]:
                    continue
                key = (s, c["feature_start"])
                if key not in cache:
                    cache[key] = scalar_features(raw[s], c["feature_start"])
                f = cache[key]
                if day in f.index and np.isfinite(f.loc[day].to_numpy(dtype=float)).all():
                    rows[s] = f.loc[day]
            assert set(rows) == set(r["qualified_forecasts"])
            expected = (
                weights(rows, plan["forecast_normalization"]) if len(r["new_members"]) >= 10 else {}
            )
            assert set(expected) == set(r["trend_book"])
            error = max([abs(expected[s] - r["trend_book"][s]) for s in expected] or [0.0])
            maximum = max(maximum, error)
            assert error < 1e-8
            assert r["long_book"] == {s: abs(v) for s, v in r["trend_book"].items()}
            for name, book in [
                ("btc_tradable", expected),
                ("magnitude_long", {s: abs(v) for s, v in expected.items()}),
            ]:
                groups.setdefault((variant, name), {})[day] = book
    for (variant, name), books in groups.items():
        actual = pd.read_parquet(inputs / f"{variant}_{name}_targets.parquet")
        expected = (
            pd.DataFrame.from_dict(books, orient="index")
            .reindex(columns=plan["symbols"])
            .fillna(0.0)
        )
        expected = expected.reindex(actual.index, method="ffill")
        assert np.max(abs(actual.filter(like="target_").to_numpy() - expected.to_numpy())) < 1e-8
        assert actual.risk_scale.eq(1.0).all() and actual.rebalance.eq(actual.index.hour == 0).all()
    for variant in plan["variants"]:
        a = pd.read_parquet(inputs / f"{variant}_btc_tradable_targets.parquet")
        b = pd.read_parquet(inputs / f"{variant}_magnitude_long_targets.parquet")
        pd.testing.assert_frame_equal(
            a.filter(like="target_").abs(), b.filter(like="target_"), check_exact=True
        )
    equality = read(inputs / "summary.json")["source_target_equivalence"]
    for name, same in equality.items():
        assert same == pd.read_parquet(inputs / f"daily_archive_{name}_targets.parquet").equals(
            pd.read_parquet(
                inputs / f"daily_with_complete_5m_reconciliation_{name}_targets.parquet"
            )
        )
    save(
        output,
        {
            "schema": "BTC-role-independent-input-audit.v1",
            "passed": True,
            "source_sha256": sha(Path(__file__)),
            "inputs_completion_sha256": sha(inputs / "completion.json"),
            "inherited_input_audit_sha256": sha(inherited_path),
            "weekly_source_cases": len(member_rows),
            "source_day_cases": len(anchors),
            "maximum_independent_weight_error": maximum,
            "old_PIT_targets_rebuilt_exact": True,
            "matched_long_absolute_weights_exact": True,
            "source_target_equivalence": equality,
            "new_financial_accounts_computed": 0,
            "profit_goal_met": False,
        },
    )
    print(read(output), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    audit(a.inputs, a.output)
