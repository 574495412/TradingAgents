"""Freeze one weekly lottery adaptation and matched reversal control before cash."""

import argparse
import math
import time
from pathlib import Path

import pandas as pd

from factors.common.weekly_lottery import reversal_book, weekly_features
from research_tools.cross_sectional_momentum.offline_coverage import source_versions
from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest
from research_tools.sparse_market.identity import crypto_context


def views(raw, registry):
    base, episodes = {}, {}
    for symbol, frame in raw.items():
        f = frame.reindex(
            pd.date_range(frame.index[0], frame.index[-1] + pd.Timedelta(days=1), freq="D")
        )
        valid = f.price_liquidity_feature_usable.eq(True) & f.volume.gt(0) & f.quote_volume.gt(0)
        valid.iloc[0] = False
        close = f.close.where(valid)
        base[symbol] = weekly_features(close)
        for card in registry["launch_cards"]:
            if card["symbol"] == symbol:
                start = pd.Timestamp(card["launch_at"]).ceil("D")
                episodes[(symbol, start.isoformat())] = weekly_features(
                    close.where(close.index >= start)
                )
    return base, episodes


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    path = Path(__file__).with_name("economic_plan.json")
    plan = read(path)
    cohort, bundle, daily, source = (
        Path(plan[k]) for k in ["cohort", "bundle", "daily", "source_capture"]
    )
    for folder in [cohort, bundle, daily, source]:
        verify_manifest(folder)
    inputs_audit = bundle.with_name(bundle.name + "_independent_audit.json")
    assert read(inputs_audit)["passed"]
    assert read(inputs_audit)["bundle_completion_sha256"] == sha(bundle / "completion.json")
    assert read(source / "GJ2021.receipt.json")["pdf_magic"] is True
    paths = [
        path,
        Path(__file__),
        Path("factors/common/weekly_lottery.py"),
        Path("research_tools/cross_sectional_momentum/offline_coverage.py"),
        Path("research_tools/sparse_market/identity.py"),
        Path("research_tools/contract_lifecycles/segments.py"),
        Path("research_tools/contract_lifecycles/timeline.py"),
        Path("research_tools/sparse_market/account_audit.py"),
        Path(plan["engine"]),
        Path(plan["engine_helper"]),
        cohort / "completion.json",
        bundle / "completion.json",
        daily / "completion.json",
        source / "completion.json",
        inputs_audit,
    ]
    hashes = {str(p): sha(p) for p in paths}
    save(output / "plan.json", {**plan, "source_hashes": hashes})
    save(
        output / "registration.json",
        {
            "wall_ns": time.time_ns(),
            "source_hashes": hashes,
            "before_new_targets_and_financial_evaluation": True,
            "new_economic_variants": 1,
        },
    )
    registry = read(cohort / "registry.json")
    symbols = read(cohort / "symbols.json")
    rawpaths = sorted((daily / "symbols").glob("*_1d.parquet"))
    variants = source_versions(plan, rawpaths)
    start, end = (pd.Timestamp(plan["window"][k], tz="UTC") for k in ["start", "end"])
    clocks = sorted({start, *pd.date_range(start, end, freq="W-MON", inclusive="left")})
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    anchors, audit_count, futures = [], 0, []
    frozen_frames = {}
    for variant, raw in variants.items():
        records = {
            pd.Timestamp(r["decision"]): r for r in read(cohort / (variant + "_cohort.json"))
        }
        base, episodes = views({s: raw[s] for s in symbols}, registry)
        books = {c: [] for c in plan["candidates"]}
        for clock in clocks:
            members = records[clock]["selected"]
            rows, independent = {}, {}
            for symbol in members:
                context = crypto_context(registry, symbol, clock)
                assert not context["blocked"]
                feature = (
                    episodes[(symbol, context["feature_start"])]
                    if context["feature_start"]
                    else base[symbol]
                )
                rows[symbol] = feature.loc[clock].to_dict()
                # Separate eight-date raw-prefix arithmetic, not the factor function.
                dates = pd.date_range(
                    clock - pd.Timedelta(days=8), clock, freq="D", inclusive="left"
                )
                f = raw[symbol].reindex(dates)
                assert (
                    f.price_liquidity_feature_usable.eq(True).all()
                    and f.volume.gt(0).all()
                    and f.quote_volume.gt(0).all()
                )
                assert context["feature_start"] is None or dates[0] >= pd.Timestamp(
                    context["feature_start"]
                )
                prices = list(f.close)
                independent[symbol] = {
                    "max7": max(math.log(prices[i] / prices[i - 1]) for i in range(1, 8)),
                    "reversal7": math.log(prices[-1] / prices[0]),
                }
                for key in ["max7", "reversal7"]:
                    assert abs(independent[symbol][key] - rows[symbol][key]) <= 1e-12
            for candidate in plan["candidates"]:
                book, meta = reversal_book(rows, candidate)
                key = "max7" if candidate.startswith("max7") else "reversal7"
                ranked = sorted(members, key=lambda s: (independent[s][key], s))
                n = len(ranked)
                low = [s for rank, s in enumerate(ranked) if rank * 5 // n == 0]
                high = [s for rank, s in enumerate(ranked) if rank * 5 // n == 4]
                expected = {s: 0.5 / len(low) for s in low}
                expected.update({s: -0.5 / len(high) for s in high})
                assert book == expected
                audit_count += 1
                books[candidate].append(pd.Series(book).reindex(symbols, fill_value=0.0))
                anchors.append(
                    {
                        "variant": variant,
                        "candidate": candidate,
                        "decision": clock.isoformat(),
                        "selected": members,
                        "book": book,
                        "details": meta,
                    }
                )
        for candidate, rows in books.items():
            frame = pd.DataFrame(rows, index=clocks).reindex(hours).ffill()
            frame.columns = ["target_" + s for s in frame.columns]
            frame["risk_scale"], frame["rebalance"] = 1.0, frame.index.isin(clocks)
            frame.to_parquet(output / f"{variant}_{candidate}_targets.parquet")
            frozen_frames[(variant, candidate)] = frame
        for clock in [
            pd.Timestamp(t, tz="UTC") for t in ["2025-01-01", "2025-06-02", "2026-07-06"]
        ]:
            members = records[clock]["selected"]
            copied = {s: raw[s].copy() for s in members}
            for frame in copied.values():
                frame.loc[frame.index >= clock, "close"] *= 3.0
                frame.loc[frame.index >= clock, "quote_volume"] *= 17.0
            changed, changed_episodes = views(copied, registry)
            for symbol in members:
                context = crypto_context(registry, symbol, clock)
                old = (
                    episodes[(symbol, context["feature_start"])]
                    if context["feature_start"]
                    else base[symbol]
                )
                new = (
                    changed_episodes[(symbol, context["feature_start"])]
                    if context["feature_start"]
                    else changed[symbol]
                )
                pd.testing.assert_series_equal(old.loc[clock], new.loc[clock])
            futures.append(
                {
                    "variant": variant,
                    "clock": clock.isoformat(),
                    "past_max_and_control_unchanged": True,
                }
            )
        print({"frozen_variant": variant, "clocks": len(clocks)}, flush=True)
    aliases = []
    for candidate in plan["candidates"]:
        frames = [frozen_frames[(variant, candidate)] for variant in plan["source_variants"]]
        equal = frames[0].equals(frames[1])
        aliases.append({"candidate": candidate, "source_variants_exact_targets_equal": equal})
    save(output / "anchors.json", anchors)
    save(output / "source_aliases.json", aliases)
    save(output / "causality_checks.json", futures)
    save(output / "lifecycle_directives.json", read(cohort / "lifecycle_directives.json"))
    save(
        output / "summary.json",
        {
            "formation_clocks": len(clocks),
            "source_candidate_books": len(anchors),
            "raw_eight_date_book_checks": audit_count,
            "future_perturbation_checks": len(futures),
            "symbols": len(symbols),
            "new_economic_variants": 1,
            "new_financial_accounts": 0,
            "source_aliases": aliases,
        },
    )
    save(
        output / "completion.json",
        {
            "manifest": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
            "financial_results_computed": False,
        },
    )
    print(
        {"inputs_complete": str(output), "book_checks": audit_count, "aliases": aliases}, flush=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args().output)
