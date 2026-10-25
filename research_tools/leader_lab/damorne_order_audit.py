"""Reconcile public order chains and study execution mechanisms without runtime labels."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from .damorne_research import FEATURES, MINUTE, load_market, millis, reconstruct_orders
from .evidence import save, sha


def quantiles(values):
    return (
        {str(q): float(np.quantile(values, q)) for q in [0.1, 0.25, 0.5, 0.75, 0.9]}
        if values
        else None
    )


def audit(root, closed_source, output):
    output.mkdir(exist_ok=False)
    orders_path = root / "orders_cursor_v002/orders.json"
    document = json.loads(orders_path.read_text())
    events, chains, unknown = reconstruct_orders(document)
    summaries = json.loads(closed_source.read_text())
    summaries = [
        p
        for p in summaries
        if document["start_ms"] <= p["opened_ms"] and p["closed_ms"] < document["end_ms_exclusive"]
    ]
    matched, mismatches = [], []
    for p in chains:
        candidates = [
            x
            for x in summaries
            if x["symbol"] == p["symbol"]
            and x["side"] == p["side"]
            and abs(x["closed_ms"] - p["closed_ms"]) <= 2000
        ]
        if len(candidates) != 1:
            mismatches.append({"chain": p, "reason": "ambiguous identity or clock"})
            continue
        x = candidates[0]
        errors = {
            "entry_relative": abs(p["entry"] / x["entry_price"] - 1),
            "exit_relative": abs(p["exit"] / x["exit_price"] - 1),
            "quantity_relative": abs(p["closed_quantity"] / x["quantity"] - 1),
        }
        if max(errors.values()) > 1e-6:
            mismatches.append({"position_id": x["position_id"], "errors": errors, "chain": p})
            continue
        matched.append(
            {**p, "position_id": x["position_id"], "reported_closing_pnl": x["reported_pnl"]}
        )
    protocol, market, quality = load_market(root)
    train_end = millis(protocol["train_end"])
    catalog = json.loads((root / "discovery/catalog.json").read_text())
    crypto = {r["symbol"] for r in catalog["symbols"] if "Crypto" in r.get("underlyingSubType", [])}
    threshold = json.loads((root / "study_v003/order_findings.json").read_text())[
        "training_only_size_split_notional"
    ]
    cohorts = {}
    for asset in ["crypto", "tradfi"]:
        for size in ["small", "large"]:
            rows = [
                p
                for p in matched
                if (p["symbol"] in crypto) == (asset == "crypto")
                and (p["notional"] <= threshold) == (size == "small")
            ]
            cohorts[f"{asset}_{size}"] = {
                "positions": len(rows),
                "gross_price_return": quantiles([p["gross_price_return"] for p in rows]),
                "duration_minutes": quantiles([p["duration_minutes"] for p in rows]),
                "sum_reported_closing_pnl": math.fsum(p["reported_closing_pnl"] for p in rows),
                "reported_position_win_rate": sum(p["reported_closing_pnl"] > 0 for p in rows)
                / len(rows)
                if rows
                else None,
                "gross_price_move_below_research_16bps_cost": sum(
                    p["gross_price_return"] <= 0.0016 for p in rows
                ),
            }
    additions, reductions, close_submission_delays, train_entry_features = [], [], [], []
    for p in matched:
        chain_events = [
            e
            for e in events
            if e["symbol"] == p["symbol"]
            and e["positionSide"] == p["side"]
            and p["opened_ms"] <= e["clock"] <= p["closed_ms"]
        ]
        qty, average = 0.0, 0.0
        sign = 1 if p["side"] == "LONG" else -1
        for event in chain_events:
            amount, price = float(event["executedQty"]), float(event["avgPrice"])
            if event["event"] == "ADD":
                additions.append(
                    {
                        "position_id": p["position_id"],
                        "own_price_return_before_add": sign * (price / average - 1),
                        "new_quantity_over_existing": amount / qty,
                        "age_minutes": (event["clock"] - p["opened_ms"]) / MINUTE,
                    }
                )
            if event["side"] == ("BUY" if sign == 1 else "SELL"):
                average = (qty * average + amount * price) / (qty + amount)
                qty += amount
            else:
                if event["event"] == "REDUCE":
                    reductions.append(
                        {
                            "position_id": p["position_id"],
                            "own_price_return_before_reduce": sign * (price / average - 1),
                            "fraction_removed": amount / qty,
                        }
                    )
                if event["type"] == "LIMIT":
                    close_submission_delays.append(
                        (int(event["orderTime"]) - p["opened_ms"]) / 1000
                    )
                qty -= amount
        opening = chain_events[0]
        if p["symbol"] in market and p["symbol"] in crypto and p["closed_ms"] < train_end:
            clock = int(opening["orderTime"]) // MINUTE * MINUTE - MINUTE
            if clock in market[p["symbol"]].index:
                f = market[p["symbol"]].loc[clock, FEATURES]
                if np.isfinite(f.to_numpy()).all():
                    train_entry_features.append(
                        {"sign": sign, **{k: float(f[k]) for k in FEATURES}}
                    )
    train_effects = {"cases": len(train_entry_features)}
    for key in ["return5", "return15", "return60", "flow"]:
        vals = [r["sign"] * r[key] for r in train_entry_features]
        train_effects[key] = {
            "directional_value": quantiles(vals),
            "same_sign_fraction": sum(v > 0 for v in vals) / len(vals) if vals else None,
        }
    report = {
        "source_orders_sha256": sha(orders_path),
        "closed_summaries_sha256": sha(closed_source),
        "public_order_summaries": len(events),
        "eligible_closed_summaries": len(summaries),
        "observed_chains": len(chains),
        "strictly_reconciled_chains": len(matched),
        "mismatches": mismatches,
        "unknown_inventory_closes": unknown,
        "all_event_counts": dict(Counter(r["event"] for r in events)),
        "reconciled_chain_additions": additions,
        "reconciled_chain_reductions": reductions,
        "close_limit_submission_after_observed_entry_seconds": quantiles(close_submission_delays),
        "cohorts": cohorts,
        "train_only_entry_features": train_effects,
        "scope": "Descriptive public-record findings. LIMIT is not proof of maker execution; orderTime to last update is not first-fill delay; recorded PNL is not audited net account return.",
        "full_market_selection_recovered": False,
        "leader_private_equity_or_protection_orders_recovered": False,
        "production_eligible": False,
    }
    save(output / "audit.json", report)
    save(output / "reconciled_positions.json", matched)
    save(
        output / "source_freeze.json",
        {
            "audit_source_sha256": sha(Path(__file__)),
            "study_source_sha256": sha(Path(__file__).with_name("damorne_research.py")),
        },
    )
    print(
        json.dumps(
            {
                "reconciled": len(matched),
                "additions": len(additions),
                "reductions": len(reductions),
                "cohorts": cohorts,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--closed-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.root.absolute(), args.closed_source.absolute(), args.output.absolute())
