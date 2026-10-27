"""Bounded raw-prefix denominator and unapproved provider-link audit; no networking."""

import argparse
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from factors.common.year52_momentum import momentum_labels, week_end_labels
from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest
from research_tools.sparse_market.identity import crypto_context


def qualification(frame, formation_label, context, parameters):
    """Full prefix only, calendar-contiguous quote window, episode-specific age."""
    end = pd.Timestamp(formation_label, tz="UTC") + pd.Timedelta(days=1)
    start = pd.Timestamp(context["feature_start"]) if context["feature_start"] else frame.index[0]
    prefix = frame.loc[(frame.index < end) & (frame.index >= start)]
    valid = (
        prefix.price_liquidity_feature_usable.eq(True)
        & prefix.full_daily_duration.eq(True)
        & prefix.volume.gt(0)
        & prefix.quote_volume.gt(0)
        & np.isfinite(prefix.quote_volume)
    )
    # Excluding the archive's first label is independent of the current episode.
    if len(prefix) and prefix.index[0] == frame.index[0]:
        valid.iloc[0] = False
    observations = int(valid.sum())
    days = parameters["trailing_calendar_days"]
    window = prefix.loc[prefix.index >= end - pd.Timedelta(days=days)]
    expected = pd.date_range(end - pd.Timedelta(days=days), end, freq="D", inclusive="left")
    complete = window.index.equals(expected) and bool(valid.reindex(expected).eq(True).all())
    quote = float(window.quote_volume.mean()) if complete else None
    eligible = (
        not context["blocked"]
        and observations >= parameters["minimum_valid_observed_days"]
        and quote is not None
        and quote >= parameters["minimum_mean_daily_quote_usdt"]
    )
    return {"eligible": bool(eligible), "observations": observations, "quote28": quote}


def source_versions(plan, paths):
    raw = {p.name.removesuffix("_1d.parquet"): pd.read_parquet(p) for p in paths}
    alternate = {s: f.copy() for s, f in raw.items()}
    cohort = Path(plan["cohort"])
    market = Path(read(cohort / "plan.json")["parent_market"])
    rows = read(cohort / "reconciliation_days.json")
    for row in rows:
        symbol, day = row["symbol"], pd.Timestamp(row["day"])
        bars = pd.read_parquet(
            market / "symbols" / (symbol + "_5m.parquet"),
            columns=["open", "high", "low", "close", "volume", "quote_volume", "price_usable"],
            filters=[("date", ">=", day), ("date", "<", day + pd.Timedelta(days=1))],
        )
        if len(bars) != 288 or not bars.price_usable.all():
            raise ValueError("Sealed reconciliation no longer has a complete raw day")
        o, high, low, close, volume, quote = (
            bars.open.iloc[0],
            bars.high.max(),
            bars.low.min(),
            bars.close.iloc[-1],
            bars.volume.sum(),
            bars.quote_volume.sum(),
        )
        valid = bool(
            volume > 0
            and quote > 0
            and low > 0
            and high >= max(o, close)
            and low <= min(o, close)
            and low * volume - 0.01 - 1e-6 * abs(high * volume)
            <= quote
            <= high * volume + 0.01 + 1e-6 * abs(high * volume)
        )
        if valid != row["reconciled_usable"]:
            raise ValueError("Reconstructed usability differs from frozen reconciliation")
        alternate[symbol].loc[day, ["open", "high", "low", "close", "volume", "quote_volume"]] = (
            o,
            high,
            low,
            close,
            volume,
            quote,
        )
        alternate[symbol].at[day, "price_liquidity_feature_usable"] = valid
    return {"daily_archive": raw, "daily_with_complete_5m_reconciliation": alternate}


def catalog_intervals(rows, metric):
    result = {}
    for row in rows:
        if row["asset"] in result:
            raise ValueError("Duplicate provider asset catalog")
        matches = [
            f
            for m in row["metrics"]
            if m["metric"] == metric
            for f in m["frequencies"]
            if f["frequency"] == "1d" and f.get("community") is True
        ]
        if len(matches) > 1:
            raise ValueError("Ambiguous Community daily interval")
        if matches:
            result[row["asset"]] = (
                date.fromisoformat(matches[0]["min_time"][:10]),
                date.fromisoformat(matches[0]["max_time"][:10]),
            )
    return result


def candidate_link(symbol, catalog):
    name = symbol.removesuffix("USDT").lower()
    return {
        "native_symbol": symbol,
        "provider_candidate": name if name in catalog else None,
        "provider_current_name": catalog[name]["full_name"] if name in catalog else None,
        "link_status": "exact_ticker_candidate_unverified"
        if name in catalog
        else "no_exact_ticker_candidate",
        "known_ticker_reuse": symbol in {"PUMPUSDT", "LITUSDT"},
        "historical_identity_verified": False,
    }


def interval_status(link, label, price, cap):
    provider = link["provider_candidate"]
    reasons = []
    if provider is None:
        reasons.append("no_exact_ticker_candidate")
    else:
        prices = momentum_labels(label)
        if provider not in price:
            reasons.append("no_community_price_interval")
        elif not all(price[provider][0] <= d <= price[provider][1] for d in prices):
            reasons.append("required_price_labels_outside_catalog_interval")
        if provider not in cap:
            reasons.append("no_community_estimated_cap_interval")
        elif not cap[provider][0] <= label <= cap[provider][1]:
            reasons.append("formation_cap_label_outside_catalog_interval")
    return {"candidate_catalog_interval_possible": not reasons, "interval_failure_reasons": reasons}


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    path = Path(__file__).with_name("offline_coverage_plan.json")
    plan = read(path)
    cohort, daily, catalog = (Path(plan[k]) for k in ["cohort", "daily", "catalog"])
    for folder in [cohort, daily, catalog]:
        verify_manifest(folder)
    paths = sorted((daily / "symbols").glob("*_1d.parquet"))
    assert len(paths) <= plan["budget"]["maximum_native_symbols"]
    assert len(paths) == 860
    replacements = read(cohort / "reconciliation_days.json")
    market = Path(read(cohort / "plan.json")["parent_market"])
    market_manifest = read(market / "completion.json")["manifest"]
    used_market = sorted({market / "symbols" / (r["symbol"] + "_5m.parquet") for r in replacements})
    for p in used_market:
        if sha(p) != market_manifest[str(p.relative_to(market))]:
            raise ValueError("Raw five-minute source differs")
    bindings = [
        path,
        Path(__file__),
        Path("factors/common/year52_momentum.py"),
        Path("research_tools/sparse_market/identity.py"),
        Path("research_tools/contract_lifecycles/segments.py"),
        Path("research_tools/contract_lifecycles/timeline.py"),
        cohort / "completion.json",
        daily / "completion.json",
        catalog / "completion.json",
        market / "completion.json",
        *paths,
        *used_market,
    ]
    save(output / "plan.json", plan)
    save(
        output / "registration.json",
        {
            "registered_wall_ns": time.time_ns(),
            "source_hashes": {str(p): sha(p) for p in bindings},
            "before_new_membership_or_candidate_results": True,
            "network_requests_allowed": 0,
            "economic_evaluation_allowed": False,
        },
    )
    registry = read(cohort / "registry.json")
    reference = {r["asset"]: r for r in read(catalog / "asset_reference.json")["data"]}
    price = catalog_intervals(read(catalog / "price_catalog.json")["data"], "PriceUSD")
    cap = catalog_intervals(read(catalog / "estimated_cap_catalog.json")["data"], "CapMrktEstUSD")
    links = {
        p.name.removesuffix("_1d.parquet"): candidate_link(
            p.name.removesuffix("_1d.parquet"), reference
        )
        for p in paths
    }
    save(output / "candidate_links.json", list(links.values()))
    begin, end = (
        date.fromisoformat(plan["formation_labels_start"]),
        date.fromisoformat(plan["formation_labels_end_exclusive"]),
    )
    labels = [
        d for y in range(begin.year, end.year + 1) for d in week_end_labels(y) if begin <= d < end
    ]
    versions = source_versions(plan, paths)
    rows, membership, summaries = [], {}, []
    for variant, frames in versions.items():
        formation = []
        for number, label in enumerate(labels):
            decision = pd.Timestamp(label + timedelta(days=1), tz="UTC") + pd.Timedelta(minutes=5)
            eligible = []
            for symbol, frame in frames.items():
                context = crypto_context(registry, symbol, decision)
                if context["blocked"]:
                    continue
                result = qualification(frame, label, context, plan["eligibility"])
                if result["eligible"]:
                    eligible.append(
                        {"symbol": symbol, **result, "episode": context["active_contract_key"]}
                    )
            eligible.sort(key=lambda r: (-r["quote28"], r["symbol"]))
            selected = eligible[: plan["eligibility"]["maximum_selected_names"]]
            formation.append(
                {
                    "formation_label": label.isoformat(),
                    "decision_at_assumed": decision.isoformat(),
                    "all_native_eligible": eligible,
                    "selected": [r["symbol"] for r in selected],
                    "historical_tradability_certified": False,
                }
            )
            for rank, row in enumerate(eligible, 1):
                link = links[row["symbol"]]
                rows.append(
                    {
                        "variant": variant,
                        "formation_label": label.isoformat(),
                        "decision_at_assumed": decision.isoformat(),
                        "native_symbol": row["symbol"],
                        "native_episode": row["episode"],
                        "quote_rank": rank,
                        "selected_top30": rank <= 30,
                        **link,
                        **interval_status(link, label, price, cap),
                        "actual_required_date_rows_verified": False,
                    }
                )
            if (number + 1) % 20 == 0:
                print({"variant": variant, "formations_done": number + 1}, flush=True)
        membership[variant] = formation
        save(output / (variant + "_membership.json"), formation)
        for year in [2025, 2026]:
            subset = [
                r
                for r in rows
                if r["variant"] == variant and r["formation_label"].startswith(str(year))
            ]
            for scope in ["all_native_eligible", "selected_top30"]:
                cells = (
                    subset
                    if scope == "all_native_eligible"
                    else [r for r in subset if r["selected_top30"]]
                )
                n = len(cells)
                summaries.append(
                    {
                        "variant": variant,
                        "year": year,
                        "scope": scope,
                        "formations": sum(
                            r["formation_label"].startswith(str(year)) for r in formation
                        ),
                        "slots": n,
                        "unique_native_symbols": len({r["native_symbol"] for r in cells}),
                        "exact_ticker_candidate_slots": sum(
                            r["provider_candidate"] is not None for r in cells
                        ),
                        "candidate_catalog_interval_possible_slots": sum(
                            r["candidate_catalog_interval_possible"] for r in cells
                        ),
                        "historical_identity_certified_slots": 0,
                        "actual_date_row_coverage": None,
                    }
                )
        print({"variant_complete": variant, "formations": len(formation)}, flush=True)
    save(output / "slots.json", rows)
    changes = [
        {
            "formation_label": a["formation_label"],
            "archive_selected": a["selected"],
            "reconciled_selected": b["selected"],
        }
        for a, b in zip(*membership.values(), strict=True)
        if a["selected"] != b["selected"]
    ]
    save(output / "source_membership_differences.json", changes)
    save(
        output / "summary.json",
        {
            "schema": "momentum-offline-coverage-decision.v1",
            "decision": "unknown_with_missing_evidence",
            "native_symbols": len(paths),
            "formations_per_variant": len(labels),
            "source_variant_membership_changes": len(changes),
            "summaries": summaries,
            "new_financial_accounts": 0,
            "network_requests": 0,
            "historical_identity_verified": False,
            "actual_date_row_coverage": None,
            "original_author_replication": False,
            "profit_goal_met": False,
            "data_branch_closed_unknown": True,
            "next_condition": "Only a specific raw-source identity and date-row acquisition contract can reopen this data branch; no provider-covered-only economic evaluation.",
        },
    )
    save(
        output / "completion.json",
        {
            "manifest": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
            "new_financial_accounts": 0,
        },
    )
    print(
        {"complete": str(output), "slots": len(rows), "source_membership_changes": len(changes)},
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
