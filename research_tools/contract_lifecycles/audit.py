"""Audit source-bound schedules and recompute a provisional history-qualified preview."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from research_tools.contract_lifecycles.timeline import restriction_at, schedules_at, validate
from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest


def audit(root, output):
    output.mkdir(parents=True, exist_ok=False)
    verify_manifest(root)
    registered = read(root / "plan.json")
    daily = Path(registered["parent_preview"])
    verify_manifest(daily)
    parent = Path(registered["parent_known_events"])
    verify_manifest(parent)
    card_path = Path(__file__).with_name("event_cards.json")
    cards = read(card_path)["cards"]
    validate(cards)
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
        Path(__file__).with_name("timeline.py"),
        card_path,
        root / "completion.json",
        daily / "completion.json",
        parent / "completion.json",
    ]
    save(
        output / "registration.json",
        {
            "source_hashes": {str(p): sha(p) for p in paths},
            "before_preview_recomputation": True,
            "scenario": "Current clauses at claimed publication +5min, no reopening inferred from a later end notice; provisional only, not point-in-time certification",
            "financial_accounts": 0,
        },
    )
    (output / "source_snapshot").mkdir()
    for p in [Path(__file__), Path(__file__).with_name("timeline.py"), card_path]:
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
        constraints = {s: restriction_at(cards, s, now) for s in symbols}
        ends = {
            s: min(
                (pd.Timestamp(c["settlement_at"]).value for c in schedules_at(cards, s, now)),
                default=None,
            )
            for s in symbols
        }
        eligible = []
        for s, (times, valid, q) in arrays.items():
            if constraints.get(s, {}).get("blocked"):
                continue
            lo = int(np.searchsorted(times, (now - pd.Timedelta(days=28)).value))
            hi = int(np.searchsorted(times, now.value))
            if hi - lo != 28 or int(valid[:hi].sum()) < 180 or not valid[lo:hi].all():
                continue
            if not np.array_equal(
                times[lo:hi],
                np.arange(
                    (now - pd.Timedelta(days=28)).value, now.value, 86400 * 10**9, dtype=np.int64
                ),
            ):
                continue
            if ends.get(s) is not None and np.any(times[lo:hi] + 86400 * 10**9 > ends[s]):
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
        known = [c for c in cards if pd.Timestamp(c["available_at_assumed"]) <= pd.Timestamp(cut)]
        for s in symbols:
            if restriction_at(cards, s, cut) != restriction_at(known, s, cut):
                raise ValueError("A future notice changes past restriction")
        checks.append({"cut": cut, "future_notice_removal_invariant_passed": True})
    current = {r["symbol"]: r for r in read(Path(registered["parent_metadata"]))["symbols"]}
    priorities = []
    end = pd.Timestamp(registered["input_end_exclusive"], tz="UTC")
    for s in registered["priority_symbols"]:
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
                    set(c["settlement_at"] for c in cards if c["symbol"] == s)
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
    initial = set(registered["selected_symbols"])
    coverage = read(root / "summary.json")
    result = {
        "schema": "historical-contract-timeline-audit.v1",
        "source_capture_complete": True,
        "source_capture_requests": coverage["total_requests"],
        "catalog_articles": coverage["catalog_articles"],
        "catalog_traversals": coverage["traversals"],
        "article_details": coverage["details"],
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
        "note": "Partial current-body scenario only; absence of a known restriction does not prove historical trading. No spot/BUSD/COIN-M event borrowed as a USD-M USDT boundary; no arbitrary +30min entry cutoff or latest-deadline overwrite of another episode.",
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
