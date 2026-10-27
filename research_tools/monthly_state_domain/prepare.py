"""Freeze source-aligned causal states and one pre-existing trend-domain gate."""

import argparse
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from factors.crypto.market_state_sessions import classify_with_decision_domain
from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest
from research_tools.sparse_market.audit_final_cohort import prepare_raw
from research_tools.sparse_market.cohort import feature_views
from research_tools.sparse_market.identity import crypto_context


def state_views(raw, registry, clocks, parameters):
    base, episodes = feature_views(raw, registry)
    possible = {t: [] for t in clocks}
    for s, f in base.items():
        f = f.reindex(clocks)
        good = (
            f.observations.ge(180)
            & f.quote28.ge(parameters["minimum_daily_quote"])
            & np.isfinite(f[["max28", "momentum28", "beta60"]]).all(axis=1)
        )
        for t in clocks[good]:
            possible[t].append(s)
    members, records = {}, []
    for clock in clocks:
        candidates = []
        for s in possible[clock]:
            context = crypto_context(registry, s, clock)
            if context["blocked"]:
                continue
            r = (
                episodes[(s, context["feature_start"])].loc[clock]
                if context["feature_start"]
                else base[s].loc[clock]
            )
            if (
                r.observations >= 180
                and r.quote28 >= parameters["minimum_daily_quote"]
                and np.isfinite(r[["max28", "momentum28", "beta60"]].to_numpy(dtype=float)).all()
            ):
                candidates.append((s, float(r.quote28)))
        chosen = sorted(candidates, key=lambda r: (-r[1], r[0]))[: parameters["maximum_eligible"]]
        members[clock] = [s for s, _ in chosen]
        records.append(
            {
                "decision": clock.isoformat(),
                "eligible_before_rank": len(candidates),
                "selected": members[clock],
                "prior_quote28": dict(chosen),
            }
        )
    frames = {}
    for s, f in raw.items():
        cal = pd.date_range(f.index[0], f.index[-1] + pd.Timedelta(days=1), freq="D")
        q = f.reindex(cal)
        valid = q.price_liquidity_feature_usable.eq(True) & q.volume.gt(0) & q.quote_volume.gt(0)
        valid.iloc[0] = False
        frames[s] = q[["close", "quote_volume"]].where(valid, axis=0)
    states, eligible = classify_with_decision_domain(frames, parameters, members)
    return states.reindex(clocks), eligible.reindex(clocks), records, frames


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    p = Path(__file__).with_name("plan.json")
    plan = read(p)
    reference = Path(plan["state_reference"])
    cohort = Path(plan["cohort"])
    targets = Path(plan["targets"])
    for parent in [reference, cohort, targets, Path(plan["unconditional_finalization"])]:
        verify_manifest(parent)
    params = read(Path(plan["state_parameter_source"]))["state_parameters"]
    old_plan = read(reference / "plan.json")
    assert old_plan["state_parameters"] == params
    for name, digest in old_plan["source_hashes"].items():
        if Path(name).name == "market_states.py":
            assert sha(Path(name)) == digest
    paths = [
        p,
        Path(__file__),
        Path("factors/common/market_states.py"),
        Path("factors/crypto/market_state_sessions.py"),
        Path(plan["state_parameter_source"]),
        Path("research_tools/sparse_market/audit_final_cohort.py"),
        Path("research_tools/sparse_market/cohort.py"),
        Path("research_tools/sparse_market/identity.py"),
        reference / "completion.json",
        cohort / "completion.json",
        targets / "completion.json",
        Path(plan["unconditional_run"]) / "completion.json",
        Path(plan["unconditional_finalization"]) / "completion.json",
        Path(plan["engine"]),
    ]
    plan = {**plan, "state_parameters": params, "source_hashes": {str(q): sha(q) for q in paths}}
    save(output / "plan.json", plan)
    save(
        output / "registration.json",
        {
            "registered_at_utc": datetime.now(UTC).isoformat(),
            "before_new_state_rebuild_targets_and_attribution": True,
            "plan_sha256": sha(output / "plan.json"),
            "no_new_state_thresholds": True,
        },
    )
    old_cohort = read(cohort / "plan.json")
    registry = read(cohort / "registry.json")
    versions = prepare_raw(old_cohort, cohort)
    a, b = [pd.Timestamp(plan["window"][k], tz="UTC") for k in ["start", "end"]]
    days = pd.date_range(a, b, freq="D", inclusive="left")
    hours = pd.date_range(a, b, freq="h", inclusive="left")
    monthlies = pd.date_range(a, b, freq="MS", inclusive="left")
    prior = pd.read_parquet(reference / "states_daily.parquet").reindex(days)
    source_summaries, months, causal = [], [], []
    for variant, raw in versions.items():
        states, eligible, members, frames = state_views(raw, registry, days, params)
        states.to_parquet(output / (variant + "_states.parquet"))
        eligible.to_parquet(output / (variant + "_eligible.parquet"))
        save(output / (variant + "_members.json"), members)
        original = pd.read_parquet(targets / (variant + "_monthly_tsmom_targets.parquet"))
        target_columns = list(original.filter(like="target_").columns)
        gate = {t: states.at[t, "state"] in plan["gate"]["active_states"] for t in monthlies}
        monthly_gate = pd.Series(gate, dtype=bool).reindex(hours, method="ffill")
        for profile in plan["profiles"]:
            frame = original.copy()
            frame.loc[~monthly_gate, target_columns] = 0.0
            if profile == "conditional_long":
                frame.loc[:, target_columns] = frame[target_columns].abs()
            assert frame.rebalance.equals(original.rebalance)
            assert frame.risk_scale.equals(original.risk_scale)
            assert np.isfinite(frame.to_numpy(dtype=float)).all()
            frame.to_parquet(output / f"{variant}_{profile}_targets.parquet")
        for t in monthlies:
            months.append(
                {
                    "variant": variant,
                    "decision": t.isoformat(),
                    "state": states.at[t, "state"],
                    "gate_active": bool(gate[t]),
                    "features": {
                        k: float(states.at[t, k])
                        for k in [
                            "trend_z",
                            "efficiency",
                            "volatility_ratio",
                            "breadth",
                            "eligible_count",
                        ]
                    },
                    "market_members": next(
                        r["selected"] for r in members if r["decision"] == t.isoformat()
                    ),
                }
            )
        for cut in [pd.Timestamp("2025-07-01T00:00:00Z"), pd.Timestamp("2026-04-01T00:00:00Z")]:
            subset = {s: f.loc[f.index < cut] for s, f in raw.items() if (f.index < cut).any()}
            sub_clocks = days[days <= cut]
            small, _, small_members, _ = state_views(subset, registry, sub_clocks, params)
            pd.testing.assert_frame_equal(states.loc[:cut], small, check_exact=True)
            assert [r for r in members if pd.Timestamp(r["decision"]) <= cut] == small_members
            changed_frames = {s: f.copy() for s, f in frames.items()}
            changed_frames["BTCUSDT"].loc[lambda f, at=cut: f.index >= at, "close"] *= 17
            changed_frames["BTCUSDT"].loc[lambda f, at=cut: f.index >= at, "quote_volume"] *= 0.001
            changed, _ = classify_with_decision_domain(
                changed_frames,
                params,
                {pd.Timestamp(r["decision"]): r["selected"] for r in members},
            )
            pd.testing.assert_frame_equal(
                states.loc[:cut], changed.reindex(days).loc[:cut], check_exact=True
            )
            causal.append(
                {
                    "variant": variant,
                    "cut": cut.isoformat(),
                    "all_source_prefix_members_and_states_equal": True,
                    "BTC_future_prices_and_quotes_do_not_change_past_states": True,
                }
            )
        feature_diffs = {
            k: float((states[k] - prior[k]).abs().max())
            for k in ["trend_z", "efficiency", "volatility_ratio", "breadth", "eligible_count"]
        }
        source_summaries.append(
            {
                "variant": variant,
                "days": len(states),
                "state_counts": states.state.value_counts().to_dict(),
                "active_monthly_anchors": sum(gate.values()),
                "old_state_label_differences": int(states.state.ne(prior.state).sum()),
                "old_feature_max_abs_differences": feature_diffs,
                "state_parameters_exactly_equal": True,
                "market_members_from_full_historical_roster": True,
            }
        )
    equivalence = {}
    for profile in plan["profiles"]:
        left, right = [pd.read_parquet(output / f"{v}_{profile}_targets.parquet") for v in versions]
        equivalence[profile] = left.equals(right)
    cross_profile = {
        v: pd.read_parquet(output / f"{v}_conditional_trend_targets.parquet").equals(
            pd.read_parquet(output / f"{v}_conditional_long_targets.parquet")
        )
        for v in versions
    }
    save(output / "monthly_decisions.json", months)
    save(output / "causality_checks.json", causal)
    save(
        output / "summary.json",
        {
            "schema": "monthly-causal-state-gate-inputs.v1",
            "sources": source_summaries,
            "source_target_equivalence": equivalence,
            "cross_profile_target_equivalence": cross_profile,
            "causal_source_cut_checks": len(causal),
            "new_financial_accounts": 0,
            "new_domain_variants": 1,
            "new_independent_mechanisms": 0,
            "profit_goal_met": False,
            "independent_oos": False,
        },
    )
    save(
        output / "completion.json",
        {
            "passed": True,
            "manifest": {q.name: sha(q) for q in output.iterdir() if q.is_file()},
            "profit_goal_met": False,
        },
    )
    print(read(output / "summary.json"), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    prepare(p.parse_args().output)
