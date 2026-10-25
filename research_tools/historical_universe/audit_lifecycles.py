"""Audit known terminal clauses and recompute the preview without postterminal entries."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.historical_universe.lifecycle import (
    blocked_new_entry,
    validate,
    whole_daily_feature_allowed,
)


def audit(output):
    output.mkdir(parents=True, exist_ok=False)
    plan_path = Path(__file__).with_name("lifecycle_plan.json")
    plan = read(plan_path)
    root = Path(plan["source_directory"])
    daily = Path(plan["parent_daily"])
    for name, h in read(root / "completion.json")["manifest"].items():
        if root.resolve() not in (root / name).resolve().parents or sha(root / name) != h:
            raise ValueError("Notice snapshot changed")
    events = read(root / "lifecycles.json")
    by = validate(events)
    save(
        output / "plan.json",
        {
            "source_code_sha256": sha(Path(__file__)),
            "lifecycle_code_sha256": sha(Path(__file__).with_name("lifecycle.py")),
            "notice_completion_sha256": sha(root / "completion.json"),
            "daily_completion_sha256": sha(daily / "completion.json"),
            "before_preview_comparison": True,
            "event_policy": "No price/fee imputation; current capture publication dates are provisional historical availability, not certified real-time versions",
        },
    )
    for name, h in read(daily / "completion.json")["manifest"].items():
        if daily.resolve() not in (daily / name).resolve().parents or sha(daily / name) != h:
            raise ValueError("Daily data changed")
    preview = read(daily / "eligibility_preview.json")
    frames = {
        p.name.removesuffix("_1d.parquet"): pd.read_parquet(p)
        for p in (daily / "symbols").glob("*_1d.parquet")
    }
    metadata = read(
        Path("data/research/historical_universe_sources_20261026_v001/exchange_info.json")
    )
    current = {r["symbol"]: r for r in metadata["symbols"]}
    claims = []
    for s, f in frames.items():
        if s not in current:
            continue
        r = current[s]
        opening = pd.Timestamp(r["onboardDate"], unit="ms", tz="UTC")
        ending = pd.Timestamp(r["deliveryDate"], unit="ms", tz="UTC")
        claims.append(
            {
                "symbol": s,
                "status_at_capture": r["status"],
                "current_onboard_at": opening.isoformat(),
                "current_delivery_at": ending.isoformat(),
                "rows_before_current_onboard": int((f.index < opening.floor("D")).sum()),
                "rows_after_current_delivery": int((f.index >= ending).sum()),
                "historical_earlier_contract_segment_certified": False,
            }
        )
    oldslots = [
        {"symbol": s, "decision": r["decision"]}
        for r in preview
        for s in r["selected"]
        if s in by and blocked_new_entry(by[s], r["decision"])
    ]
    arrays = {
        s: (
            f.index.as_unit("ns").asi8,
            f.price_liquidity_feature_usable.to_numpy(),
            f.quote_volume.to_numpy(),
        )
        for s, f in frames.items()
    }
    revised = []
    deltas = []
    for p in preview:
        now = pd.Timestamp(p["decision"])
        eligible = []
        for s, (times, valid, q) in arrays.items():
            event = by.get(s)
            if event and blocked_new_entry(event, now):
                continue
            hi = int(np.searchsorted(times, now.value))
            lo = int(np.searchsorted(times, (now - pd.Timedelta(days=28)).value))
            if hi - lo != 28 or int(valid[:hi].sum()) < 180 or not valid[lo:hi].all():
                continue
            if not np.array_equal(
                times[lo:hi],
                np.arange(
                    (now - pd.Timedelta(days=28)).value,
                    now.value,
                    86400 * 1_000_000_000,
                    dtype=np.int64,
                ),
            ):
                continue
            if event and not all(
                whole_daily_feature_allowed(event, pd.Timestamp(int(t), tz="UTC"), now)
                for t in times[lo:hi]
            ):
                continue
            mean = float(q[lo:hi].sum() / 28)
            if mean >= 50_000_000:
                eligible.append((s, mean))
        selected = [s for s, v in sorted(eligible, key=lambda r: (-r[1], r[0]))[:30]]
        revised.append(
            {
                "decision": p["decision"],
                "selected": selected,
                "eligible_names": len(eligible),
                "historical_status_fully_verified": False,
            }
        )
        if selected != p["selected"]:
            deltas.append(
                {
                    "decision": p["decision"],
                    "removed": sorted(set(p["selected"]) - set(selected)),
                    "added": sorted(set(selected) - set(p["selected"])),
                }
            )
    for e in events:
        t = pd.Timestamp(e["available_at_assumed"])
        for p, r in zip(preview, revised, strict=True):
            if pd.Timestamp(p["decision"]) < t and p["selected"] != r["selected"]:
                # A different, already-known event may legitimately alter that date.
                prior = [
                    x
                    for x in events
                    if pd.Timestamp(x["available_at_assumed"]) <= pd.Timestamp(p["decision"])
                ]
                if not prior:
                    raise ValueError("Future notice changed earlier preview")
    summary = {
        "schema": "historical-lifecycle-audit.v1",
        "notice_cases": len(events),
        "events": events,
        "original_selected_slots_after_known_entry_cutoff": oldslots,
        "removed_postterminal_slots": len(oldslots),
        "changed_decisions": len(deltas),
        "preview_decisions": len(preview),
        "deltas": deltas,
        "current_metadata_claims": claims,
        "notice_availability_uses_claimed_publication_plus5m": True,
        "historical_article_versions_certified": False,
        "all268historical_status_certified": False,
        "terminal_settlement_values_verified": False,
        "preview_is_live_tradable": False,
        "financial_accounts": 0,
        "net_return": None,
        "profit_goal_met": False,
        "source_hashes": {
            str(root / "completion.json"): sha(root / "completion.json"),
            str(daily / "completion.json"): sha(daily / "completion.json"),
        },
    }
    save(output / "preview.json", revised)
    save(output / "summary.json", summary)
    save(
        output / "completion.json",
        {
            "manifest": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
            "known_boundary_audit_complete": True,
            "financial_accounts": 0,
        },
    )
    print(
        {
            "notice_cases": len(events),
            "postterminal_selected_slots": len(oldslots),
            "changed_decisions": len(deltas),
            "financial_accounts": 0,
        },
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    audit(a.output)
