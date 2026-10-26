"""Freeze identity-aware positive-liquidity cohorts before expanded account returns."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from factors.common.max_effect import features
from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest
from research_tools.sparse_market.identity import crypto_context, directives


def daily_versions(plan):
    raw = {
        p.name.removesuffix("_1d.parquet"): pd.read_parquet(p)
        for p in (Path(plan["parent_daily"]) / "symbols").glob("*_1d.parquet")
    }
    reconciled = {s: f.copy() for s, f in raw.items()}
    replacements = []
    for path in sorted((Path(plan["parent_market"]) / "symbols").glob("*_5m.parquet")):
        s = path.name.removesuffix("_5m.parquet")
        bars = pd.read_parquet(
            path, columns=["open", "high", "low", "close", "volume", "quote_volume", "price_usable"]
        )
        aggregate = bars.resample("D").agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            quote_volume=("quote_volume", "sum"),
            rows=("close", "size"),
            price_valid=("price_usable", "all"),
        )
        days = aggregate.index[aggregate.rows.eq(288) & aggregate.price_valid].intersection(
            raw[s].index
        )
        different = pd.Series(False, index=days)
        for col in ["open", "high", "low", "close", "quote_volume"]:
            tol = (0.01 if col == "quote_volume" else 1e-10) + 1e-9 * raw[s].loc[days, col].abs()
            different |= (raw[s].loc[days, col] - aggregate.loc[days, col]).abs() > tol
        columns = ["open", "high", "low", "close", "volume", "quote_volume"]
        for day in days[different]:
            a = aggregate.loc[day]
            valid = bool(
                a.volume > 0
                and a.quote_volume > 0
                and a.low > 0
                and a.high >= max(a.open, a.close)
                and a.low <= min(a.open, a.close)
                and a.low * a.volume - 0.01 - 1e-6 * abs(a.high * a.volume)
                <= a.quote_volume
                <= a.high * a.volume + 0.01 + 1e-6 * abs(a.high * a.volume)
            )
            reconciled[s].loc[day, columns] = a[columns]
            reconciled[s].at[day, "price_liquidity_feature_usable"] = valid
            replacements.append({"symbol": s, "day": day.isoformat(), "reconciled_usable": valid})
    return {"daily_archive": raw, "daily_with_complete_5m_reconciliation": reconciled}, replacements


def feature_views(raw, registry):
    frames = {}
    for s, f in raw.items():
        calendar = pd.date_range(f.index[0], f.index[-1] + pd.Timedelta(days=1), freq="D")
        q = f.reindex(calendar)
        valid = q.price_liquidity_feature_usable.eq(True) & q.volume.gt(0) & q.quote_volume.gt(0)
        # A first source daily label alone does not prove an entire trading day.
        valid.iloc[0] = False
        frames[s] = q[["close", "quote_volume"]].where(valid, axis=0)
    base = features(frames)
    per_episode = {}
    for launch in registry["launch_cards"]:
        s = launch["symbol"]
        q = frames[s].copy()
        start = pd.Timestamp(launch["launch_at"]).ceil("D")
        q.loc[q.index < start, :] = np.nan
        per_episode[(s, start.isoformat())] = features({"BTCUSDT": frames["BTCUSDT"], s: q})[s]
    return base, per_episode


def select(raw, registry, clocks):
    base, episodes = feature_views(raw, registry)
    records, chosen = [], {"BTCUSDT"}
    for clock in clocks:
        eligible = []
        for s, f in base.items():
            if s == "BTCUSDT" or clock not in f.index:
                continue
            context = crypto_context(registry, s, clock)
            if context["blocked"]:
                continue
            row = (
                episodes[(s, context["feature_start"])].loc[clock]
                if context["feature_start"]
                else f.loc[clock]
            )
            if (
                row.observations >= 180
                and row.quote28 >= 50_000_000
                and np.isfinite(row[["max28", "momentum28", "beta60"]].to_numpy(dtype=float)).all()
            ):
                eligible.append((s, float(row.quote28)))
        selected = [s for s, _ in sorted(eligible, key=lambda r: (-r[1], r[0]))[:30]]
        records.append(
            {
                "decision": clock.isoformat(),
                "selected": selected,
                "eligible": len(eligible),
                "historical_tradability_certified": False,
            }
        )
        chosen.update(selected)
    return records, chosen


def prepare(identity, output):
    output.mkdir(parents=True, exist_ok=False)
    p = Path(__file__).with_name("plan.json")
    plan = read(p)
    verify_manifest(identity)
    for root in [plan["parent_cohort"], plan["parent_market"], plan["parent_daily"]]:
        verify_manifest(Path(root))
    save(
        output / "registration.json",
        {
            "source_hashes": {
                str(p): sha(p)
                for p in [
                    p,
                    Path(__file__),
                    Path(__file__).with_name("identity.py"),
                    Path("factors/common/max_effect.py"),
                    identity / "completion.json",
                    *[
                        Path(plan[k]) / "completion.json"
                        for k in ["parent_cohort", "parent_market", "parent_daily"]
                    ],
                ]
            },
            "first_daily_label_without_complete_trading_day_evidence": "exclude_first_observed_daily_row_conservatively",
            "before_financial_accounts": True,
        },
    )
    registry = read(identity / "registry.json")
    parent = read(Path(plan["parent_cohort"]) / "cohort.json")
    clocks = [pd.Timestamp(r["decision"]) for r in parent]
    versions, replacements = daily_versions(plan)
    rows, union = {}, set()
    for variant, raw in versions.items():
        rows[variant], chosen = select(raw, registry, clocks)
        union |= chosen
        print({"variant": variant, "symbols": len(chosen), "decisions": len(clocks)}, flush=True)
    if len(union) > plan["budget"]["maximum_total_acquired_symbols"]:
        raise ValueError("Combined source-cohort acquisition budget exceeded")
    old = set(read(Path(plan["parent_cohort"]) / "symbols.json"))
    new = sorted(union - old)
    if len(new) > plan["budget"]["maximum_additional_symbols"]:
        raise ValueError("Additional source roster budget exceeded")
    comparisons = []
    for variant, data in rows.items():
        for oldrow, row in zip(parent, data, strict=True):
            removed, added = (
                sorted(set(oldrow["selected"]) - set(row["selected"])),
                sorted(set(row["selected"]) - set(oldrow["selected"])),
            )
            if removed or added:
                comparisons.append(
                    {
                        "variant": variant,
                        "decision": row["decision"],
                        "removed": removed,
                        "added": added,
                    }
                )
        save(output / (variant + "_cohort.json"), data)
    save(output / "symbols.json", sorted(union))
    save(output / "new_symbols.json", new)
    save(output / "cohort_changes.json", comparisons)
    save(output / "reconciliation_days.json", replacements)
    save(output / "lifecycle_directives.json", directives(registry, sorted(union)))
    save(output / "registry.json", registry)
    result = {
        "schema": "identity-aware-cohort.v1",
        "decisions_per_variant": len(clocks),
        "source_variants": list(rows),
        "symbols": len(union),
        "new_symbols": new,
        "prior_symbols_retained": len(union & old),
        "acquisition_only_removed_names": sorted(old - union),
        "complete_day_replacements": len(replacements),
        "changed_clock_source_cells": len(comparisons),
        "reused_ticker_symbols": ["LITUSDT", "PUMPUSDT"],
        "type_restrictions": 5,
        "new_financial_accounts": 0,
        "historical_original_versions_certified": False,
        "profit_goal_met": False,
    }
    save(output / "summary.json", result)
    save(
        output / "completion.json",
        {
            "manifest": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
            "financial_accounts": 0,
        },
    )
    print(result, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--identity", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    prepare(a.identity, a.output)
