"""Freeze monthly raw-data requirements without observing strategy-account returns."""

import argparse
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest
from research_tools.sparse_market.audit_final_cohort import prepare_raw
from research_tools.sparse_market.identity import crypto_context


def qualify(output):
    output.mkdir(parents=True, exist_ok=False)
    plan_path = Path(__file__).with_name("plan.json")
    plan = read(plan_path)
    cohort = Path(plan["cohort"])
    old_plan = read(cohort / "plan.json")
    daily = Path(old_plan["parent_daily"])
    for parent in [cohort, daily, Path(plan["source_pack"])]:
        verify_manifest(parent)
    paths = [
        plan_path,
        Path(__file__),
        Path("research_tools/sparse_market/audit_final_cohort.py"),
        Path("research_tools/sparse_market/identity.py"),
        Path("research_tools/contract_lifecycles/segments.py"),
        cohort / "completion.json",
        daily / "completion.json",
        Path(plan["source_pack"]) / "completion.json",
        Path(plan["existing_bindings"]),
    ]
    save(output / "plan.json", plan)
    save(
        output / "registration.json",
        {
            "registered_at_utc": datetime.now(UTC).isoformat(),
            "before_new_monthly_selection_and_raw_capture": True,
            "source_hashes": {str(p): sha(p) for p in paths},
            "known_history_only": True,
            "financial_accounts": 0,
        },
    )
    registry = read(cohort / "registry.json")
    versions = prepare_raw(old_plan, cohort)
    start, end = [pd.Timestamp(plan["window"][k], tz="UTC") for k in ["start", "end"]]
    clocks = pd.date_range(start, end, freq="MS", inclusive="left")
    records, union, need = [], set(), {}
    duplicate_keys, bad_dates = [], []
    for variant, raw in versions.items():
        vectors = {}
        for symbol, f in raw.items():
            if f.index.has_duplicates:
                duplicate_keys.append({"variant": variant, "symbol": symbol})
            if not f.index.is_monotonic_increasing or not f.index.equals(f.index.normalize()):
                bad_dates.append({"variant": variant, "symbol": symbol})
            if duplicate_keys or bad_dates:
                raise ValueError("Daily grain is not unique normalized UTC days")
            dates = pd.date_range(f.index[0], f.index[-1], freq="D")
            q = f.reindex(dates)
            valid = (
                q.price_liquidity_feature_usable.eq(True)
                & q.volume.gt(0)
                & q.quote_volume.gt(0)
                & np.isfinite(q.close)
                & q.close.gt(0)
            ).to_numpy(copy=True)
            valid[0] = False
            vectors[symbol] = (dates, q, np.r_[0, valid.cumsum()])
        for clock in clocks:
            left = clock - pd.DateOffset(months=12)
            first = left - pd.Timedelta(days=1)
            count = (clock - first).days
            eligible, reasons = [], {"identity": 0, "history_or_quality": 0, "liquidity": 0}
            for symbol, (dates, q, cumulative) in vectors.items():
                c = crypto_context(registry, symbol, clock)
                if c["blocked"] or (
                    c["feature_start"] and pd.Timestamp(c["feature_start"]) > first
                ):
                    reasons["identity"] += 1
                    continue
                a, b = dates.searchsorted(first), dates.searchsorted(clock)
                if first < dates[0] or b - a != count or cumulative[b] - cumulative[a] != count:
                    reasons["history_or_quality"] += 1
                    continue
                quote = float(q.quote_volume.iloc[b - 28 : b].mean())
                if not np.isfinite(quote) or quote < 50_000_000:
                    reasons["liquidity"] += 1
                    continue
                eligible.append((symbol, quote, float(q.close.iloc[b - 1] / q.close.iloc[a] - 1)))
            chosen = sorted(eligible, key=lambda r: (-r[1], r[0]))[:10]
            selected = [s for s, _, _ in chosen]
            assert len(eligible) + sum(reasons.values()) == len(raw)
            records.append(
                {
                    "variant": variant,
                    "decision": clock.isoformat(),
                    "return_start": left.isoformat(),
                    "return_days": count - 1,
                    "universe_size": len(raw),
                    "eligible": len(eligible),
                    "exclusion_counts": reasons,
                    "selected": selected,
                    "prior_quote28": {s: q for s, q, _ in chosen},
                    "price_only_12month_return_reference": {s: r for s, _, r in chosen},
                    "economic_return_signal_status": "not_yet_constructed",
                }
            )
            union.update(selected)
            for s in selected:
                need[s] = min(need.get(s, left), left)
    bindings = read(Path(plan["existing_bindings"]))
    tasks, coverage = [], []
    boundary = pd.Timestamp(plan["raw_capture"]["end_exclusive"])
    for s in sorted(union):
        old = bindings.get(s)
        item = {
            "symbol": s,
            "earliest_signal_history": need[s].isoformat(),
            "execution_binding_present": old is not None,
        }
        if old:
            path = Path(old["funding_path"])
            assert sha(path) == old["funding_sha256"]
            f = pd.read_parquet(path)
            item.update(
                {
                    "existing_first_event": f.index[0].isoformat(),
                    "existing_last_event": f.index[-1].isoformat(),
                    "existing_bad_marks": int((~f.held_cash_mark_usable).sum()),
                    "existing_funding_path": str(path),
                    "existing_funding_sha256": sha(path),
                }
            )
            if need[s] < boundary:
                tasks.append(
                    {
                        "symbol": s,
                        "start": need[s].isoformat(),
                        "end_exclusive": boundary.isoformat(),
                    }
                )
        coverage.append(item)
    assert len(tasks) <= plan["raw_capture"]["maximum_symbols"]
    same = all(
        a["selected"] == b["selected"]
        for a, b in zip(records[: len(clocks)], records[len(clocks) :], strict=True)
    )
    save(output / "monthly_membership.json", records)
    save(output / "funding_coverage.json", coverage)
    save(output / "capture_tasks.json", tasks)
    save(
        output / "summary.json",
        {
            "schema": "monthly-tsmom-readiness.v1",
            "months_per_variant": len(clocks),
            "source_variants": len(versions),
            "monthly_source_cases": len(records),
            "selected_union": sorted(union),
            "selected_symbols": len(union),
            "source_selected_names_equal": same,
            "required_early_funding_symbols": len(tasks),
            "missing_execution_bindings": [
                r["symbol"] for r in coverage if not r["execution_binding_present"]
            ],
            "duplicate_daily_keys": duplicate_keys,
            "bad_daily_dates": bad_dates,
            "new_financial_accounts": 0,
            "profit_goal_met": False,
            "historical_identity_certified": False,
            "independent_oos": False,
        },
    )
    save(
        output / "completion.json",
        {
            "passed": True,
            "manifest": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
            "financial_accounts": 0,
        },
    )
    print(read(output / "summary.json"), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    qualify(p.parse_args().output)
