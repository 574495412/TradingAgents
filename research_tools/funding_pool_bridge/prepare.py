"""Register a finite historical bridge and reconstruct matched opportunity books."""

import argparse
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from factors.crypto.funding_cross_section import accrued7, book
from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest
from research_tools.sparse_market.cohort import feature_views
from research_tools.sparse_market.identity import crypto_context

COLUMNS = ["open", "high", "low", "close", "volume", "quote_volume"]


def old_bars(path, start, end):
    frame = pd.read_parquet(path).loc[start:end].copy()
    finite = np.isfinite(frame[COLUMNS]).all(axis=1)
    prices = frame[["open", "high", "low", "close"]]
    valid = (
        finite
        & prices.gt(0).all(axis=1)
        & frame.high.ge(prices[["open", "close"]].max(axis=1))
        & frame.low.le(prices[["open", "close"]].min(axis=1))
    )
    frame["price_usable"] = valid
    frame["execution_liquidity_usable"] = valid & frame.quote_volume.gt(0) & frame.volume.gt(0)
    return frame


def load_market(plan, mode, symbols):
    sources = plan["inputs"]
    begin = pd.Timestamp(plan["window"]["start"], tz="UTC") - pd.Timedelta(days=7)
    finish = pd.Timestamp(plan["window"]["end"], tz="UTC") - pd.Timedelta(minutes=5)
    bars, rates, bindings = {}, {}, {}
    for s in symbols:
        shared = Path(sources["bundle"]) / "symbols" / (s + "_5m.parquet")
        use_shared = mode == "shared" and shared.exists()
        bp = shared if use_shared else Path(sources["old_market"]) / (s + "_5m.parquet")
        fp = (
            Path(sources["bundle"]) / "funding" / (s + "_exact.parquet")
            if use_shared
            else Path(sources["old_funding"]) / (s + "_exact.parquet")
        )
        bars[s] = (
            pd.read_parquet(bp).loc[begin:finish] if use_shared else old_bars(bp, begin, finish)
        )
        rates[s] = pd.read_parquet(fp)
        bindings[s] = {str(p): sha(p) for p in [bp, fp]}
    return bars, rates, bindings


def difference(left, right, columns):
    common = left.index.intersection(right.index)
    a, b = left.loc[common, columns], right.loc[common, columns]
    unequal = ~(a.eq(b) | (a.isna() & b.isna()))
    return {
        "left_rows": len(left),
        "right_rows": len(right),
        "left_only": len(left.index.difference(right.index)),
        "right_only": len(right.index.difference(left.index)),
        "different_rows": int(unequal.any(axis=1).sum()),
        "different_cells": int(unequal.to_numpy().sum()),
    }


def rows_at(base, episodes, registry, symbols, clock):
    result = {}
    for s in symbols:
        if s not in base or clock not in base[s].index:
            continue
        c = crypto_context(registry, s, clock)
        if c["blocked"]:
            continue
        result[s] = (
            episodes[(s, c["feature_start"])].loc[clock]
            if c["feature_start"]
            else base[s].loc[clock]
        )
    return result


def hourly(books, calendar, clocks, symbols):
    frame = pd.DataFrame(books, index=clocks).reindex(columns=symbols, fill_value=0.0)
    frame = frame.reindex(calendar).ffill().rename(columns=lambda s: "target_" + s)
    frame["risk_scale"], frame["rebalance"] = 1.0, frame.index.isin(clocks)
    return frame


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).with_name("plan.json")
    plan = read(source)
    inputs = plan["inputs"]
    for key in [
        "old_inputs",
        "pit_inputs",
        "cohort",
        "bundle",
        "old_market",
        "old_funding",
        "daily",
    ]:
        verify_manifest(Path(inputs[key]))
    fixed = read(Path(inputs["old_market"]) / "prepare_plan.json")["symbols"]
    pit_symbols = read(Path(inputs["cohort"]) / "symbols.json")
    paths = [
        source,
        Path(__file__),
        Path("factors/crypto/funding_cross_section.py"),
        Path("factors/common/max_effect.py"),
        Path("research_tools/sparse_market/cohort.py"),
        Path("research_tools/sparse_market/identity.py"),
        Path("../freqtrade/research/crypto/sparse_calendar_v2.py"),
        Path("../freqtrade/research/crypto/sparse_sources_v2.py"),
        Path("../freqtrade/research/crypto/calendar_portfolio.py"),
        *[Path(v) / "completion.json" for v in inputs.values()],
        Path("docs/research/regime_program/targets.json"),
    ]
    plan = {
        **plan,
        "registered_at_utc": datetime.now(UTC).isoformat(),
        "old_symbols": fixed,
        "pit_symbols": pit_symbols,
        "source_hashes": {str(p): sha(p) for p in paths},
    }
    save(output / "plan.json", plan)
    save(
        output / "registration.json",
        {"plan_sha256": sha(output / "plan.json"), "before_new_targets_and_accounts": True},
    )
    save(output / "protocol_snapshot.json", read(Path("docs/research/regime_program/targets.json")))
    old, old_rates, old_bindings = load_market(plan, "old", fixed)
    shared, shared_rates, shared_bindings = load_market(
        plan, "shared", sorted(set(fixed) | set(pit_symbols))
    )
    comparisons = []
    a = pd.Timestamp(plan["window"]["start"], tz="UTC")
    b = pd.Timestamp(plan["window"]["end"], tz="UTC")
    for s in fixed:
        comparisons.append(
            {
                "symbol": s,
                "shared_from_bundle": s in pit_symbols,
                "price_quote": difference(old[s], shared[s], COLUMNS),
                "quality": difference(
                    old[s], shared[s], ["price_usable", "execution_liquidity_usable"]
                ),
                "funding": difference(
                    old_rates[s].loc[a - pd.Timedelta(days=7) : b],
                    shared_rates[s].loc[a - pd.Timedelta(days=7) : b],
                    ["mark", "funding_rate"],
                ),
            }
        )
    save(output / "source_comparison.json", comparisons)
    save(output / "market_bindings.json", {"old": old_bindings, "shared": shared_bindings})
    del old, old_rates, shared
    registry = read(Path(inputs["cohort"]) / "registry.json")
    raw_paths = sorted((Path(inputs["daily"]) / "symbols").glob("*_1d.parquet"))
    raw = {p.name.removesuffix("_1d.parquet"): pd.read_parquet(p) for p in raw_paths}
    save(output / "daily_bindings.json", {str(p): sha(p) for p in raw_paths})
    base, episodes = feature_views(raw, registry)
    parameters = read(Path(inputs["old_inputs"]) / "plan.json")["parameters"]
    calendar = pd.date_range(a, b, freq="h", inclusive="left")
    clocks = sorted({a, *pd.date_range(a, b, freq="W-MON", inclusive="left")})
    reference = {
        pd.Timestamp(r["decision"]): r
        for r in read(Path(inputs["cohort"]) / "daily_archive_cohort.json")
    }
    matched, pit, anchors = [], [], []
    for clock in clocks:
        all_rows = rows_at(base, episodes, registry, base, clock)
        candidates = [
            s
            for s, r in all_rows.items()
            if s != "BTCUSDT"
            and r.observations >= parameters["minimum_listing_observations"]
            and r.quote28 >= parameters["minimum_daily_quote_volume"]
            and np.isfinite(r[["max28", "momentum28", "beta60"]].to_numpy(dtype=float)).all()
        ]
        selected = sorted(candidates, key=lambda s: (-float(all_rows[s].quote28), s))[
            : parameters["maximum_eligible"]
        ]
        if selected != reference[clock]["selected"]:
            raise ValueError("PIT membership reconstruction differs at " + str(clock))
        old_rows = {s: r for s, r in all_rows.items() if s in fixed}
        pit_rows = {s: all_rows[s] for s in ["BTCUSDT", *selected]}
        cases = []
        for rows in [old_rows, pit_rows]:
            funding = accrued7({s: shared_rates[s] for s in rows if s != "BTCUSDT"}, clock)
            w, info = book(rows, funding, parameters, plan["candidate"])
            cases.append((w, info))
        matched.append(cases[0][0].reindex(fixed, fill_value=0.0))
        pit.append(cases[1][0].reindex(pit_symbols, fill_value=0.0))
        anchors.append(
            {
                "clock": clock.isoformat(),
                "matched40": cases[0][1],
                "pit": cases[1][1],
                "matched40_book": {s: float(v) for s, v in cases[0][0].items() if v},
                "pit_book": {s: float(v) for s, v in cases[1][0].items() if v},
            }
        )
    rebuilt = hourly(pit, calendar, clocks, pit_symbols)
    sealed = pd.read_parquet(
        Path(inputs["pit_inputs"]) / "daily_archive_diagnostic2026_funding7_raw_targets.parquet"
    )
    pd.testing.assert_frame_equal(rebuilt, sealed, check_freq=False)
    matched_frame = hourly(matched, calendar, clocks, fixed)
    matched_frame.to_parquet(output / "matched40_targets.parquet")
    old_target = Path(inputs["old_inputs"]) / "archive_diagnostic2026_funding7_raw_targets.parquet"
    sealed_old = pd.read_parquet(old_target)
    matched_targets = matched_frame.reindex(columns=sealed_old.columns)
    target_diff = difference(sealed_old, matched_targets, list(sealed_old.columns))
    save(output / "anchors.json", anchors)
    save(
        output / "target_comparison.json",
        {
            "pit_hourly_exactly_reconstructed": True,
            "hourly_rows": len(calendar),
            "decision_clocks": len(clocks),
            "old_vs_matched40": target_diff,
            "old_target_sha256": sha(old_target),
        },
    )
    directives_path = Path(inputs["pit_inputs"]) / "lifecycle_directives.json"
    directives = read(directives_path)
    save(output / "lifecycle40.json", {s: directives[s] for s in fixed if s in directives})
    checks = {
        "all_new_PIT_hourly_books_exact": True,
        "new_financial_accounts_computed": 0,
        "raw_daily_assets": len(raw),
        "old_only_names": sorted(set(fixed) - set(pit_symbols)),
        "source_different_names": [
            r["symbol"]
            for r in comparisons
            if any(
                r[k]["different_rows"] or r[k]["left_only"] or r[k]["right_only"]
                for k in ["price_quote", "quality", "funding"]
            )
        ],
    }
    save(output / "checks.json", checks)
    manifest = {
        str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()
    }
    save(
        output / "completion.json", {"manifest": manifest, "passed": True, "orders_enabled": False}
    )
    print(checks, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args().output)
