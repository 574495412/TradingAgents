"""Freeze one benchmark-eligibility change and an explicit magnitude-long control."""

import argparse
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from factors.crypto.trend_sessions import qualified_features
from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest
from research_tools.sparse_market.audit_final_cohort import prepare_raw
from research_tools.sparse_market.cohort import feature_views
from research_tools.sparse_market.identity import crypto_context
from research_tools.trend_pool_audit.prepare_v2 import WARMUP, hourly, sized


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    p = Path(__file__).with_name("plan.json")
    plan = read(p)
    parent = Path(plan["inputs"]["parent_targets"])
    parent_run = Path(plan["inputs"]["parent_accounts"])
    for root in [parent, parent_run, Path(plan["inputs"]["method_source"])]:
        verify_manifest(root)
    check_path = Path(str(parent) + "_independent_audit.json")
    checked = read(check_path)
    assert checked["passed"] and checked["inputs_completion_sha256"] == sha(
        parent / "completion.json"
    )
    old_plan = read(parent / "plan.json")
    for path, digest in old_plan["source_hashes"].items():
        assert sha(Path(path)) == digest
    symbols = old_plan["symbols"]
    paths = [
        p,
        Path(__file__),
        Path("factors/crypto/trend_sessions.py"),
        Path("factors/common/github_formulas.py"),
        Path("factors/portfolio/source_sizing.py"),
        Path("research_tools/trend_pool_audit/prepare_v2.py"),
        Path("research_tools/sparse_market/cohort.py"),
        Path("research_tools/sparse_market/identity.py"),
        Path("research_tools/sparse_market/audit_final_cohort.py"),
        parent / "completion.json",
        check_path,
        parent_run / "completion.json",
        Path(plan["inputs"]["method_source"]) / "completion.json",
        Path("../freqtrade/research/crypto/sparse_calendar_v2.py"),
        Path("../freqtrade/research/crypto/sparse_sources_v2.py"),
    ]
    plan = {
        **plan,
        "registered_at_utc": datetime.now(UTC).isoformat(),
        "symbols": symbols,
        "account": old_plan["account"],
        "parameters": old_plan["parameters"],
        "forecast_normalization": old_plan["forecast_normalization"],
        "source_hashes": {str(q): sha(q) for q in paths},
    }
    save(output / "plan.json", plan)
    save(
        output / "registration.json",
        {"before_new_targets_and_returns": True, "plan_sha256": sha(output / "plan.json")},
    )
    for name in ["execution_bindings.json", "lifecycle.json"]:
        shutil.copyfile(parent / name, output / name)
    shutil.copyfile("docs/research/regime_program/targets.json", output / "protocol_snapshot.json")
    cohort = Path(old_plan["inputs"]["cohort"])
    registry = read(cohort / "registry.json")
    versions = prepare_raw(read(cohort / "plan.json"), cohort)
    a, b = [pd.Timestamp(plan["window"][k], tz="UTC") for k in ["start", "end"]]
    days = pd.date_range(a, b, freq="D", inclusive="left")
    hours = pd.date_range(a, b, freq="h", inclusive="left")
    weekly = sorted({a, *pd.date_range(a, b, freq="W-MON", inclusive="left")})
    anchors, membership, causes, source_equivalence = [], [], [], {}
    for variant, raw in versions.items():
        base, episode_base = feature_views(raw, registry)
        refs = {
            pd.Timestamp(r["decision"]): r["selected"]
            for r in read(cohort / (variant + "_cohort.json"))
        }
        feature_names = sorted({"BTCUSDT", *(s for t in weekly for s in refs[t])})
        features = {s: qualified_features(raw[s], WARMUP) for s in feature_names}
        episodes = {}
        for card in registry["launch_cards"]:
            if card["symbol"] in features:
                start = pd.Timestamp(card["launch_at"]).ceil("D").isoformat()
                episodes[(card["symbol"], start)] = qualified_features(
                    raw[card["symbol"]], WARMUP, episode_start=start
                )
        domains = {}
        for clock in weekly:
            row = base["BTCUSDT"].loc[clock]
            context = crypto_context(registry, "BTCUSDT", clock)
            eligible = (
                not context["blocked"]
                and row.observations >= 180
                and row.quote28 >= 50_000_000
                and np.isfinite(row[["max28", "momentum28", "beta60"]].to_numpy(dtype=float)).all()
            )
            candidates = refs[clock] + (["BTCUSDT"] if eligible else [])
            quotes = {}
            for s in candidates:
                c = crypto_context(registry, s, clock)
                r = (
                    episode_base[(s, c["feature_start"])].loc[clock]
                    if c["feature_start"]
                    else base[s].loc[clock]
                )
                quotes[s] = float(r.quote28)
            selected = sorted(candidates, key=lambda s: (-quotes[s], s))[:30]
            domains[clock] = selected
            membership.append(
                {
                    "variant": variant,
                    "clock": clock.isoformat(),
                    "BTC_eligible": bool(eligible),
                    "BTC_quote28": float(row.quote28),
                    "old_members": refs[clock],
                    "new_members": selected,
                    "displaced": sorted(set(refs[clock]) - set(selected)),
                }
            )
        books = {"excluded": {}, "btc_tradable": {}, "magnitude_long": {}}
        last = None
        for day in days:
            if day in domains:
                last = day

            def rows_for(members, decision=day, feature_map=features, episode_map=episodes):
                rows = {}
                for s in members:
                    c = crypto_context(registry, s, decision)
                    if c["blocked"]:
                        continue
                    f = (
                        episode_map[(s, c["feature_start"])]
                        if c["feature_start"]
                        else feature_map[s]
                    )
                    if decision in f.index:
                        r = f.loc[decision]
                        if (
                            np.isfinite(
                                r[["pst_ewmac32", "relative_vol"]].to_numpy(dtype=float)
                            ).all()
                            and r.consecutive_observations >= 180
                        ):
                            rows[s] = r
                return rows

            old_rows = rows_for(refs[last])
            new_rows = rows_for(domains[last])
            old = (
                sized(
                    old_rows, plan["forecast_normalization"], plan["parameters"], sorted(old_rows)
                )
                if len(refs[last]) >= 10
                else pd.Series(dtype=float)
            )
            new = (
                sized(
                    new_rows, plan["forecast_normalization"], plan["parameters"], sorted(new_rows)
                )
                if len(domains[last]) >= 10
                else pd.Series(dtype=float)
            )
            books["excluded"][day] = old.reindex(symbols, fill_value=0.0)
            books["btc_tradable"][day] = new.reindex(symbols, fill_value=0.0)
            books["magnitude_long"][day] = new.abs().reindex(symbols, fill_value=0.0)
            anchors.append(
                {
                    "variant": variant,
                    "decision": day.isoformat(),
                    "membership_at": last.isoformat(),
                    "old_members": refs[last],
                    "new_members": domains[last],
                    "qualified_forecasts": list(new_rows),
                    "trend_book": {s: float(v) for s, v in new.items() if v},
                    "long_book": {s: float(abs(v)) for s, v in new.items() if v},
                    "BTC_target": float(new.get("BTCUSDT", 0.0)),
                }
            )
        rebuilt = hourly(books["excluded"], hours, symbols)
        old_target = pd.read_parquet(parent / (variant + "_pit_targets.parquet"))
        pd.testing.assert_frame_equal(rebuilt, old_target, check_exact=True, check_freq=False)
        for name in ["btc_tradable", "magnitude_long"]:
            frame = hourly(books[name], hours, symbols)
            assert np.isfinite(frame.to_numpy(dtype=float)).all()
            assert frame.filter(like="target_").abs().sum(axis=1).le(1 + 1e-10).all()
            frame.to_parquet(output / f"{variant}_{name}_targets.parquet")
        for cut in [pd.Timestamp("2025-07-01T00:00:00Z"), pd.Timestamp("2026-04-01T00:00:00Z")]:
            altered = raw["BTCUSDT"].copy()
            altered.loc[altered.index >= cut, ["open", "high", "low", "close"]] *= 17
            altered.loc[altered.index >= cut, "quote_volume"] *= 0.01
            changed = qualified_features(altered, WARMUP)
            pd.testing.assert_frame_equal(
                features["BTCUSDT"].loc[:cut], changed.loc[:cut], check_freq=False
            )
            causes.append(
                {
                    "variant": variant,
                    "cut": cut.isoformat(),
                    "actual_future_BTC_price_quote_perturbation_same_past": True,
                }
            )
    for name in ["btc_tradable", "magnitude_long"]:
        left = pd.read_parquet(output / f"daily_archive_{name}_targets.parquet")
        right = pd.read_parquet(
            output / f"daily_with_complete_5m_reconciliation_{name}_targets.parquet"
        )
        source_equivalence[name] = left.equals(right)
    save(output / "anchors.json", anchors)
    save(output / "membership.json", membership)
    save(output / "causality_checks.json", causes)
    save(
        output / "summary.json",
        {
            "old_PIT_targets_exactly_reproduced": True,
            "weekly_source_cases": len(membership),
            "source_day_cases": len(anchors),
            "BTC_nonzero_source_day_targets": sum(r["BTC_target"] != 0 for r in anchors),
            "source_target_equivalence": source_equivalence,
            "new_financial_accounts_computed": 0,
            "new_economic_mechanisms": 0,
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
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    prepare(p.parse_args().output)
