"""Freeze matched weekly domains, contiguous daily forecasts and source pairs."""

import argparse
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from factors.common.github_formulas import daily, formulas
from factors.crypto.trend_sessions import qualified_features
from factors.portfolio.source_sizing import sized_intent
from research_tools.factor_lab.io import read, save, sha
from research_tools.funding_pool_bridge.prepare import difference, old_bars
from research_tools.max_effect.run_accounts import verify_manifest
from research_tools.sparse_market.audit_final_cohort import prepare_raw
from research_tools.sparse_market.cohort import feature_views
from research_tools.sparse_market.identity import crypto_context

WARMUP = pd.Timestamp("2023-08-01T00:00:00Z")


def candidates(base, episodes, registry, names, clock):
    selected = []
    for s in names:
        if s == "BTCUSDT" or clock not in base[s].index:
            continue
        c = crypto_context(registry, s, clock)
        if c["blocked"]:
            continue
        row = (
            episodes[(s, c["feature_start"])].loc[clock]
            if c["feature_start"]
            else base[s].loc[clock]
        )
        if (
            row.observations >= 180
            and row.quote28 >= 50_000_000
            and np.isfinite(row[["max28", "momentum28", "beta60"]].to_numpy(dtype=float)).all()
        ):
            selected.append((s, float(row.quote28)))
    return [s for s, _ in sorted(selected, key=lambda x: (-x[1], x[0]))[:30]]


def sized(rows, scale, parameters, names):
    raw = pd.Series({s: rows[s].pst_ewmac32 for s in names}, dtype=float)
    vol = pd.Series({s: rows[s].relative_vol for s in names}, dtype=float)
    return (
        sized_intent(raw, vol, scale, parameters)["risk"] if len(names) else pd.Series(dtype=float)
    )


def hourly(books, calendar, symbols):
    frame = pd.DataFrame.from_dict(books, orient="index").reindex(columns=symbols, fill_value=0.0)
    frame = frame.reindex(calendar, method="ffill").rename(columns=lambda s: "target_" + s)
    frame["risk_scale"], frame["rebalance"] = 1.0, frame.index.hour == 0
    return frame


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    p = Path(__file__).with_name("plan.json")
    plan = read(p)
    inputs = plan["inputs"]
    for k in [
        "legacy_inputs",
        "legacy_market",
        "legacy_funding",
        "daily",
        "cohort",
        "bundle",
        "reference_run",
    ]:
        verify_manifest(Path(inputs[k]))
    old_symbols = read(Path(inputs["legacy_market"]) / "prepare_plan.json")["symbols"]
    pit_symbols = read(Path(inputs["cohort"]) / "symbols.json")
    symbols = sorted(set(old_symbols) | set(pit_symbols))
    parameters = read(Path(inputs["legacy_inputs"]) / "plan.json")["parameters"]
    scale = read(Path(inputs["normalization"]))["values"]["pst_ewmac32"]
    registry = read(Path(inputs["cohort"]) / "registry.json")
    source_paths = [
        p,
        Path(__file__),
        Path("factors/crypto/trend_sessions.py"),
        Path("factors/common/github_formulas.py"),
        Path("factors/portfolio/source_sizing.py"),
        Path("research_tools/sparse_market/cohort.py"),
        Path("research_tools/sparse_market/identity.py"),
        Path("research_tools/sparse_market/audit_final_cohort.py"),
        Path("research_tools/funding_pool_bridge/prepare.py"),
        Path("../freqtrade/research/crypto/sparse_calendar_v2.py"),
        Path("../freqtrade/research/crypto/sparse_sources_v2.py"),
        Path(inputs["normalization"]),
        Path(inputs["legacy_eligibility"]),
        *[
            Path(inputs[k]) / "completion.json"
            for k in [
                "legacy_inputs",
                "legacy_market",
                "legacy_funding",
                "daily",
                "cohort",
                "bundle",
                "reference_run",
            ]
        ],
    ]
    shutil.copyfile("docs/research/regime_program/targets.json", output / "protocol_snapshot.json")
    plan = {
        **plan,
        "registered_at_utc": datetime.now(UTC).isoformat(),
        "old_symbols": old_symbols,
        "pit_symbols": pit_symbols,
        "symbols": symbols,
        "parameters": parameters,
        "forecast_normalization": scale,
        "source_hashes": {str(q): sha(q) for q in source_paths},
    }
    save(output / "plan.json", plan)
    save(
        output / "registration.json",
        {"before_new_targets_and_returns": True, "plan_sha256": sha(output / "plan.json")},
    )
    a, b = [pd.Timestamp(plan["window"][k], tz="UTC") for k in ["start", "end"]]
    days = pd.date_range(a, b, freq="D", inclusive="left")
    calendar = pd.date_range(a, b, freq="h", inclusive="left")
    weekly = sorted({a, *pd.date_range(a, b, freq="W-MON", inclusive="left")})
    legacy_elig = pd.read_parquet(inputs["legacy_eligibility"])
    old_daily = {
        s: daily(pd.read_parquet(Path(inputs["legacy_market"]) / (s + "_5m.parquet")))
        for s in old_symbols
    }
    old_features = {s: formulas(f).shift(1) for s, f in old_daily.items()}
    old_books = {
        d: sized(
            {s: old_features[s].loc[d] for s in old_symbols},
            scale,
            parameters,
            [s for s in sorted(old_symbols) if legacy_elig.loc[d, s]],
        ).reindex(symbols, fill_value=0.0)
        for d in days
    }
    rebuilt = hourly(old_books, calendar, symbols)
    sealed = pd.read_parquet(Path(inputs["legacy_inputs"]) / "trend_risk16_targets.parquet")
    old_target = sealed.filter(like="target_").reindex(
        columns=rebuilt.filter(like="target_").columns, fill_value=0.0
    )
    error = float(np.max(np.abs(old_target.to_numpy() - rebuilt.filter(like="target_").to_numpy())))
    if error > 1e-12:
        raise ValueError("Original trend mathematical reconstruction differs")
    sealed0 = sealed.reindex(
        columns=["target_" + s for s in symbols] + ["risk_scale"], fill_value=0.0
    )
    sealed0["rebalance"] = sealed0.index.hour == 0
    sealed0.to_parquet(output / "legacy_targets.parquet")
    comparisons = []
    bindings = {}
    for s in symbols:
        bp = Path(inputs["bundle"]) / "symbols" / (s + "_5m.parquet")
        fp = Path(inputs["bundle"]) / "funding" / (s + "_exact.parquet")
        shared = bp.exists()
        if not shared:
            bp = Path(inputs["legacy_market"]) / (s + "_5m.parquet")
            fp = Path(inputs["legacy_funding"]) / (s + "_exact.parquet")
        bindings[s] = {
            "bar_path": str(bp),
            "funding_path": str(fp),
            "from_bundle": shared,
            "bar_sha256": sha(bp),
            "funding_sha256": sha(fp),
        }
        if s in old_symbols:
            left = old_bars(
                Path(inputs["legacy_market"]) / (s + "_5m.parquet"),
                a - pd.Timedelta(hours=1),
                b - pd.Timedelta(minutes=5),
            )
            right = (
                pd.read_parquet(bp).loc[a - pd.Timedelta(hours=1) : b - pd.Timedelta(minutes=5)]
                if shared
                else left
            )
            lf = pd.read_parquet(Path(inputs["legacy_funding"]) / (s + "_exact.parquet")).loc[a:b]
            rf = pd.read_parquet(fp).loc[a:b]
            comparisons.append(
                {
                    "symbol": s,
                    "bars": difference(
                        left,
                        right,
                        [
                            "open",
                            "high",
                            "low",
                            "close",
                            "volume",
                            "quote_volume",
                            "price_usable",
                            "execution_liquidity_usable",
                        ],
                    ),
                    "funding": difference(lf, rf, ["mark", "funding_rate"]),
                }
            )
    save(output / "execution_bindings.json", bindings)
    save(output / "execution_source_comparison.json", comparisons)
    source_changed = any(
        any(r[k][v] for v in ["different_rows", "left_only", "right_only"])
        for r in comparisons
        for k in ["bars", "funding"]
    )
    daily_plan = read(Path(inputs["cohort"]) / "plan.json")
    versions = prepare_raw(daily_plan, Path(inputs["cohort"]))
    anchors = []
    diagnostics = []
    causality = []
    feature_counts = {}
    for variant, raw in versions.items():
        base, episodes = feature_views(raw, registry)
        ref = {
            pd.Timestamp(r["decision"]): r["selected"]
            for r in read(Path(inputs["cohort"]) / (variant + "_cohort.json"))
        }
        old_members = {t: candidates(base, episodes, registry, old_symbols, t) for t in weekly}
        for t in weekly:
            chosen = candidates(base, episodes, registry, base, t)
            if chosen != ref[t]:
                raise ValueError("PIT raw membership reconstruction differs at " + str(t))
        features = {s: qualified_features(raw[s], WARMUP) for s in symbols}
        per_episode = {}
        for card in registry["launch_cards"]:
            s = card["symbol"]
            if s in features:
                start = pd.Timestamp(card["launch_at"]).ceil("D").isoformat()
                per_episode[(s, start)] = qualified_features(raw[s], WARMUP, episode_start=start)
        source_raw = {
            s: formulas(
                raw[s].loc[WARMUP:, ["open", "high", "low", "close", "volume", "quote_volume"]]
            ).shift(1)
            for s in old_symbols
        }
        diagnostic_books = {
            k: {}
            for k in [
                "raw_daily_legacy_eligibility",
                "qualified_daily_domain",
                "qualified_weekly_domain",
            ]
        }
        main = {"fixed40": {}, "pit": {}}
        last = None
        counts = {"fixed40_forecast_rejections": 0, "pit_forecast_rejections": 0}
        for day in days:
            if day in old_members:
                last = day

            def current(s, decision=day, episode_view=per_episode, feature_view=features):
                c = crypto_context(registry, s, decision)
                f = episode_view[(s, c["feature_start"])] if c["feature_start"] else feature_view[s]
                if c["blocked"] or decision not in f.index:
                    return None
                row = f.loc[decision]
                return (
                    row
                    if np.isfinite(row[["pst_ewmac32", "relative_vol"]].to_numpy(dtype=float)).all()
                    and row.consecutive_observations >= 180
                    else None
                )

            for domain, members in [("fixed40", old_members[last]), ("pit", ref[last])]:
                rows = {s: r for s in members if (r := current(s)) is not None}
                counts[domain + "_forecast_rejections"] += len(members) - len(rows)
                w = (
                    sized(rows, scale, parameters, sorted(rows))
                    if len(members) >= 10
                    else pd.Series(dtype=float)
                )
                main[domain][day] = w.reindex(symbols, fill_value=0.0)
                anchors.append(
                    {
                        "variant": variant,
                        "domain": domain,
                        "decision": day.isoformat(),
                        "membership_at": last.isoformat(),
                        "members": members,
                        "qualified_forecasts": list(rows),
                        "book": {s: float(v) for s, v in w.items() if v},
                    }
                )
            names = [s for s in sorted(old_symbols) if legacy_elig.loc[day, s]]
            diagnostic_books["raw_daily_legacy_eligibility"][day] = sized(
                {s: source_raw[s].loc[day] for s in names}, scale, parameters, names
            ).reindex(symbols, fill_value=0.0)
            dm = candidates(base, episodes, registry, old_symbols, day)
            dr = {s: r for s in dm if (r := current(s)) is not None}
            diagnostic_books["qualified_daily_domain"][day] = (
                sized(dr, scale, parameters, sorted(dr))
                if len(dm) >= 10
                else pd.Series(dtype=float)
            ).reindex(symbols, fill_value=0.0)
            diagnostic_books["qualified_weekly_domain"][day] = main["fixed40"][day]
        prior = rebuilt
        for name, books in diagnostic_books.items():
            frame = hourly(books, calendar, symbols)
            changed = frame.filter(like="target_").ne(prior.filter(like="target_")).any(axis=1)
            diagnostics.append(
                {
                    "variant": variant,
                    "step": name,
                    "hourly_rows_changed": int(changed.sum()),
                    "max_target_change": float(
                        np.max(
                            np.abs(
                                frame.filter(like="target_").to_numpy()
                                - prior.filter(like="target_").to_numpy()
                            )
                        )
                    ),
                }
            )
            prior = frame
        for domain, books in main.items():
            frame = hourly(books, calendar, symbols)
            if (
                not np.isfinite(frame.to_numpy(dtype=float)).all()
                or (frame.filter(like="target_").abs().sum(axis=1) > 1 + 1e-10).any()
                or (frame.filter(like="target_").ne(0).sum(axis=1) > 10).any()
            ):
                raise ValueError("Invalid trend unit target")
            frame.to_parquet(output / f"{variant}_{domain}_targets.parquet")
        feature_counts[variant] = counts
        for cut in [pd.Timestamp("2025-07-01T00:00:00Z"), pd.Timestamp("2026-04-01T00:00:00Z")]:
            for s in ["BTCUSDT", "ETHUSDT", "PUMPUSDT"]:
                changed = raw[s].copy()
                changed.loc[changed.index >= cut, ["open", "high", "low", "close"]] *= 17
                changed.loc[changed.index >= cut, "quote_volume"] *= 0.01
                altered = qualified_features(changed, WARMUP)
                pd.testing.assert_frame_equal(
                    features[s].loc[:cut], altered.loc[:cut], check_freq=False
                )
                causality.append(
                    {
                        "variant": variant,
                        "symbol": s,
                        "decision": cut.isoformat(),
                        "actual_future_price_quote_perturbation_same_past_forecasts": True,
                    }
                )
    save(output / "anchors.json", anchors)
    save(output / "input_diagnostics.json", diagnostics)
    save(output / "causality_checks.json", causality)
    directives = read(
        Path("results/research/sparse_weekly_targets_20261026_v001/lifecycle_directives.json")
    )
    save(output / "lifecycle.json", directives)
    save(
        output / "summary.json",
        {
            "schema": "trend-pool-inputs.v1",
            "old_target_reconstruction_max_error": error,
            "old_execution_sources_changed": source_changed,
            "conditional_source_bridge_required": source_changed,
            "source_variants": plan["source_variants"],
            "daily_trend_decisions": len(days),
            "hourly_rows_per_account": len(calendar),
            "frozen_membership_cases_reconstructed": len(weekly) * 2,
            "trend_input_source_domain_cases": len(anchors),
            "contiguous_forecast_rejections": feature_counts,
            "causality_cases": len(causality),
            "new_financial_accounts_computed": 0,
            "orders_enabled": False,
        },
    )
    save(
        output / "completion.json",
        {
            "passed": True,
            "manifest": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
            "profit_goal_met": False,
        },
    )
    print(read(output / "summary.json"), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    prepare(p.parse_args().output)
