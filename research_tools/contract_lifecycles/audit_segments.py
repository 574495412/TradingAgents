"""Audit source-bound schedules and recompute a provisional history-qualified preview."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from research_tools.contract_lifecycles.segments import age_and_quote, context_at
from research_tools.contract_lifecycles.timeline import schedules_at, validate
from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest


def audit(root, output):
    output.mkdir(parents=True, exist_ok=False)
    verify_manifest(root)
    registered = read(root / "plan.json")
    for path, digest in registered["source_hashes"].items():
        if sha(Path(path)) != digest:
            raise ValueError("Prepared segment source changed")
    daily = Path(registered["data"])
    verify_manifest(daily)
    registry = read(root / "registry.json")
    cards = registry["cards"]
    validate(cards)
    card_path = root / "registry.json"
    bindings = {}
    for c in cards:
        for key in ["source_raw", "source_text"]:
            path = Path(c[key])
            if sha(path) != c[key + "_sha256"]:
                raise ValueError("Card source bytes changed")
            bindings[str(path)] = sha(path)
        text = Path(c["source_text"]).read_text()
        for key in ["product_literal", "settlement_literal", "entry_literal"]:
            if c[key] and c[key] not in text:
                raise ValueError("Manual clause does not exist in bound source")
        raw = read(Path(c["source_raw"]))["data"]
        if raw["code"] != c["source_code"] or pd.Timestamp(
            raw["publishDate"], unit="ms", tz="UTC"
        ) != pd.Timestamp(c["publish_at_claimed"]):
            raise ValueError("Notice identity/publication changed")
        if (
            c["original_historical_content_certified"]
            or c["terminal_settlement_price"] is not None
            or c["terminal_settlement_fee"] is not None
        ):
            raise ValueError("Unverified terminal or original version promoted")
    paths = [
        Path(__file__),
        Path(__file__).with_name("segments.py"),
        card_path,
        root / "completion.json",
        daily / "completion.json",
    ]
    save(
        output / "registration.json",
        {
            "source_hashes": {str(p): sha(p) for p in paths},
            "before_preview_recomputation": True,
            "prior_timeline_helper_sha256": sha(Path(__file__).with_name("timeline.py")),
            "scenario": "Current clauses at claimed publication +5min, no reopening inferred from a later end notice; provisional only, not point-in-time certification",
            "financial_accounts": 0,
        },
    )
    (output / "source_snapshot").mkdir()
    for p in [Path(__file__), Path(__file__).with_name("segments.py"), card_path]:
        (output / "source_snapshot" / p.name).write_bytes(p.read_bytes())
    preview = read(daily / "eligibility_preview.json")
    frames = {
        p.name.removesuffix("_1d.parquet"): pd.read_parquet(p)
        for p in (daily / "symbols").glob("*_1d.parquet")
    }
    arrays = {
        s: (
            f.index.as_unit("ns").asi8,
            f.price_liquidity_feature_usable.to_numpy(),
            f.quote_volume.to_numpy(),
        )
        for s, f in frames.items()
    }
    revised, changes, original_removed = [], [], []
    symbols = sorted({c["symbol"] for c in cards})
    for row in preview:
        now = pd.Timestamp(row["decision"])
        constraints = {s: context_at(registry, s, now) for s in symbols}
        eligible = []
        for s, (times, valid, q) in arrays.items():
            if constraints.get(s, {}).get("blocked"):
                continue
            lo = int(np.searchsorted(times, (now - pd.Timedelta(days=28)).value))
            hi = int(np.searchsorted(times, now.value))
            feature_start = constraints.get(s, {}).get("feature_start")
            segment_lo = (
                int(np.searchsorted(times, pd.Timestamp(feature_start).value))
                if feature_start
                else 0
            )
            if (
                hi - lo != 28
                or lo < segment_lo
                or int(valid[segment_lo:hi].sum()) < 180
                or not valid[lo:hi].all()
            ):
                continue
            if not np.array_equal(
                times[lo:hi],
                np.arange(
                    (now - pd.Timedelta(days=28)).value, now.value, 86400 * 10**9, dtype=np.int64
                ),
            ):
                continue
            mean = float(q[lo:hi].mean())
            if mean >= 50_000_000:
                eligible.append((s, mean))
        ordered = sorted(eligible, key=lambda r: (-r[1], r[0]))[:30]
        new = [s for s, _ in ordered]
        revised.append(
            {
                "decision": row["decision"],
                "selected": new,
                "eligible": len(eligible),
                "historical_tradability_certified": False,
                "scenario_only": True,
            }
        )
        removed = [s for s in row["selected"] if s not in new]
        for s in removed:
            original_removed.append(
                {
                    "decision": row["decision"],
                    "symbol": s,
                    "reason": constraints.get(s, {}).get("reason", "feature_spans_known_end"),
                }
            )
        if new != row["selected"]:
            changes.append(
                {
                    "decision": row["decision"],
                    "removed": removed,
                    "added": [s for s in new if s not in row["selected"]],
                }
            )
    checks = []
    for cut in ["2022-11-13T00:00Z", "2024-12-15T00:00Z", "2026-04-01T00:00Z"]:
        known = {
            **registry,
            "cards": [
                c for c in cards if pd.Timestamp(c["available_at_assumed"]) <= pd.Timestamp(cut)
            ],
            "launch_cards": [
                c
                for c in registry["launch_cards"]
                if pd.Timestamp(c["available_at_assumed"]) <= pd.Timestamp(cut)
            ],
        }
        for symbol in symbols:
            if context_at(registry, symbol, cut) != context_at(known, symbol, cut):
                raise ValueError("Future launch or terminal notice changes past context")
        checks.append({"cut": cut, "future_notice_removal_invariant_passed": True})
    current = {
        r["symbol"]: r
        for r in read(
            Path("data/research/historical_universe_sources_20261026_v001/exchange_info.json")
        )["symbols"]
    }
    priorities = []
    end = pd.Timestamp("2026-09-01", tz="UTC")
    for s in read(Path("data/research/contract_lifecycle_notices_20261026_v001/plan.json"))[
        "priority_symbols"
    ]:
        schedules = schedules_at(cards, s, end)
        dates = [pd.Timestamp(c["settlement_at"]) for c in schedules]
        earliest = min(dates) if dates else None
        f = frames[s]
        priorities.append(
            {
                "symbol": s,
                "current_status_at_capture": current.get(s, {}).get("status", "ABSENT"),
                "schedule_cards": sum(c["symbol"] == s for c in cards),
                "source_end_clauses": sorted(
                    {c["settlement_at"] for c in cards if c["symbol"] == s}
                ),
                "rows_after_known_end_requiring_segment_proof": int((f.index >= earliest).sum())
                if earliest
                else None,
                "original_selected_slots": sum(s in r["selected"] for r in preview),
                "removed_original_slots": sum(r["symbol"] == s for r in original_removed),
                "historical_status_certified": False,
                "relisting_start_certified": False,
                "terminal_price_and_fee_verified": False,
            }
        )
    initial = {s for row in preview for s in row["selected"]}
    coverage = read(Path("data/research/contract_lifecycle_notices_20261026_v001/summary.json"))
    supplement = read(Path("data/research/contract_segment_followup_20261026_v002/summary.json"))
    profiles = []
    for row in preview:
        if "AERGOUSDT" in row["selected"]:
            now = pd.Timestamp(row["decision"])
            context = context_at(registry, "AERGOUSDT", now)
            profiles.append(
                {
                    "decision": row["decision"],
                    **context,
                    **age_and_quote(frames["AERGOUSDT"], context, now),
                }
            )
    new_valid = frames["AERGOUSDT"].loc["2025-04-17":].price_liquidity_feature_usable.eq(True)
    first180 = (new_valid.index[new_valid.cumsum() >= 180][0] + pd.Timedelta(days=1)).isoformat()
    result = {
        "schema": "historical-contract-segment-audit.v1",
        "source_capture_complete": True,
        "source_capture_requests": coverage["total_requests"] + supplement["total_new_requests"],
        "supplement_requests": supplement["total_new_requests"],
        "launch_cards": registry["launch_cards"],
        "launch_profiles": profiles,
        "AERGO_first_180_valid_full_observation_clock": first180,
        "source_conflicts": registry["source_conflicts"],
        "reported_trade_probes": supplement["archive_probes"],
        "unresolved_episode_starts": ["BLZUSDT"],
        "generic_scope_symbols": sorted(c["symbol"] for c in cards if c.get("scope_indirect")),
        "catalog_articles": coverage["catalog_articles"],
        "catalog_traversals": coverage["traversals"],
        "article_details": coverage["details"] + len(supplement["details"]),
        "unfetched_title_candidates": coverage["unfetched_title_candidates"],
        "cards": cards,
        "card_count": len(cards),
        "covered_symbols": len(symbols),
        "priority_symbols": priorities,
        "priority_count": len(priorities),
        "unresolved_priority_symbols": [r["symbol"] for r in priorities if not r["schedule_cards"]],
        "postponement_revisions": sum(c["supersedes"] is not None for c in cards),
        "multi_episode_symbols": sorted(
            s for s in symbols if len({c["episode"] for c in cards if c["symbol"] == s}) > 1
        ),
        "preview_decisions": len(preview),
        "changed_decisions": len(changes),
        "changes": changes,
        "removed_original_slots": len(original_removed),
        "removed_slots": original_removed,
        "new_names_outside_initial_union": sorted(
            {s for r in revised for s in r["selected"]} - initial
        ),
        "future_notice_checks": checks,
        "independent_schedule_audit_pending": True,
        "original_historical_content_certified": False,
        "all_contract_segments_certified": False,
        "terminal_settlement_values_verified": False,
        "original_datasets_modified": False,
        "new_financial_accounts": 0,
        "portfolio_actual": None,
        "profit_goal_met": False,
        "production_eligible": False,
        "orders_enabled": False,
        "note": "Partial current-body scenario. Additional generic futures clauses require named-token scope binding and current product-type corroboration; current metadata is not historical status. Explicit AERGO launch resets full-day features only. Old inventory and total-account highwater never reset. BLZ notice/trade-source conflict remains unresolved.",
    }
    save(output / "preview.json", revised)
    save(output / "source_bindings.json", bindings)
    save(output / "summary.json", result)
    save(
        output / "completion.json",
        {
            "manifest": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
            "scenario_recomputation_complete": True,
            "historical_certification_passed": False,
            "financial_accounts": 0,
        },
    )
    print(
        {
            k: result[k]
            for k in [
                "card_count",
                "covered_symbols",
                "unresolved_priority_symbols",
                "changed_decisions",
                "removed_original_slots",
                "new_names_outside_initial_union",
                "multi_episode_symbols",
            ]
        },
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    audit(a.source, a.output)
