"""Separate submission decisions from later fills, lifecycle roles and loss tails."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .damorne_research import MINUTE, load_market
from .evidence import save, sha


def quantiles(values):
    return (
        {str(q): float(np.quantile(values, q)) for q in [0.1, 0.25, 0.5, 0.75, 0.9]}
        if values
        else None
    )


def market_cache(source, output):
    cache = output / "market_cache"
    manifest = cache / "manifest.json"
    if manifest.exists():
        bindings = json.loads((cache / "source_bindings.json").read_text())
        if any(sha(source / name) != digest for name, digest in bindings.items()):
            raise ValueError("Market cache provenance differs from source manifest")
        rows = json.loads(manifest.read_text())
        market = {}
        for row in rows:
            path = cache / row["file"]
            if sha(path) != row["sha256"]:
                raise ValueError("Market cache changed")
            with np.load(path, allow_pickle=False) as data:
                market[row["symbol"]] = pd.DataFrame(
                    data["values"], index=data["index"], columns=row["columns"]
                )
        return market
    _, market, quality = load_market(source)
    cache.mkdir(parents=True)
    rows = []
    for symbol, f in market.items():
        target = cache / f"{symbol}.npz"
        np.savez_compressed(target, index=f.index.to_numpy(), values=f.to_numpy(dtype=float))
        rows.append(
            {
                "symbol": symbol,
                "columns": list(f.columns),
                "file": target.name,
                "sha256": sha(target),
            }
        )
    save(manifest, rows)
    save(cache / "quality.json", quality)
    save(
        cache / "source_bindings.json",
        {
            name: sha(source / name)
            for name in ["market_sources.json", "rest_backfill_hashes.json", "funding_sources.json"]
        },
    )
    return market


def diagnose(source, output, cache_owner=None):
    output.mkdir(exist_ok=False)
    market = market_cache(source, cache_owner or output)
    positions = json.loads((source / "audit_v001/reconciled_positions.json").read_text())
    full_summaries = json.loads(
        (
            source.parent
            / "leader_lab_20261024/damorne_public_profile_v003/positions_normalized.json"
        ).read_text()
    )
    events = json.loads((source / "study_v003/order_events.json").read_text())
    catalog = json.loads((source / "discovery/catalog.json").read_text())
    crypto = {r["symbol"] for r in catalog["symbols"] if "Crypto" in r.get("underlyingSubType", [])}
    records, add_records = [], []
    for p in sorted(positions, key=lambda r: r["opened_ms"]):
        chain = [
            e
            for e in events
            if e["symbol"] == p["symbol"]
            and e["positionSide"] == p["side"]
            and p["opened_ms"] <= e["clock"] <= p["closed_ms"]
        ]
        first = chain[0]
        symbol, sign = p["symbol"], 1 if p["side"] == "LONG" else -1
        prior_rows = [
            x
            for x in full_summaries
            if x["symbol"] == symbol and x["side"] == p["side"] and x["closed_ms"] < p["opened_ms"]
        ]
        prior_raw = max(prior_rows, key=lambda x: x["closed_ms"]) if prior_rows else None
        prior = (
            {"closed_ms": prior_raw["closed_ms"], "reported_closing_pnl": prior_raw["reported_pnl"]}
            if prior_raw
            else None
        )
        gap = p["opened_ms"] - prior["closed_ms"] if prior else None
        role = (
            "first_or_new_episode"
            if not prior or gap > 12 * 3600000
            else "after_profit"
            if prior["reported_closing_pnl"] > 0
            else "after_loss"
        )
        submitted = int(first["orderTime"])
        available = submitted // MINUTE * MINUTE - MINUTE
        f = market.get(symbol)
        features = f.loc[available].to_dict() if f is not None and available in f.index else None
        record = {
            "position_id": p["position_id"],
            "symbol": symbol,
            "sign": sign,
            "asset_class": "crypto" if symbol in crypto else "tradfi",
            "role_at_fill": role,
            "opened_ms": p["opened_ms"],
            "submitted_ms": submitted,
            "closed_ms": p["closed_ms"],
            "notional": p["notional"],
            "reported_closing_pnl": p["reported_closing_pnl"],
            "gross_price_return": p["gross_price_return"],
            "entry_last_update_delay_seconds": (p["opened_ms"] - submitted) / 1000,
            "submitted_before_previous_closed": bool(prior and submitted < prior["closed_ms"]),
            "previous_close_to_submission_seconds": (submitted - prior["closed_ms"]) / 1000
            if prior
            else None,
            "proxy_entry_offset_from_submission_market": sign * (p["entry"] / features["close"] - 1)
            if features
            else None,
            "directional_return5_at_submission": sign * features["return5"]
            if features and math.isfinite(features["return5"])
            else None,
            "directional_return60_at_submission": sign * features["return60"]
            if features and math.isfinite(features["return60"])
            else None,
        }
        records.append(record)
        average, qty = 0.0, 0.0
        for e in chain:
            amount, price = float(e["executedQty"]), float(e["avgPrice"])
            opening = e["side"] == ("BUY" if sign == 1 else "SELL")
            if opening:
                if qty > 0:
                    add_records.append(
                        {
                            "position_id": p["position_id"],
                            "symbol": symbol,
                            "order_created_ms": e["orderTime"],
                            "filled_last_update_ms": e["clock"],
                            "created_before_observed_entry": int(e["orderTime"]) < p["opened_ms"],
                            "price_return_at_observed_add": sign * (price / average - 1),
                            "added_quantity_ratio": amount / qty,
                        }
                    )
                average = (average * qty + price * amount) / (qty + amount)
                qty += amount
            else:
                qty -= amount
    groups = {}
    for role in ["first_or_new_episode", "after_profit", "after_loss"]:
        rows = [r for r in records if r["role_at_fill"] == role and r["asset_class"] == "crypto"]
        groups[role] = {
            "count": len(rows),
            "preposted_before_previous_close": sum(
                r["submitted_before_previous_closed"] for r in rows
            ),
            "entry_delay_seconds": quantiles([r["entry_last_update_delay_seconds"] for r in rows]),
            "entry_proxy_offset": quantiles(
                [
                    r["proxy_entry_offset_from_submission_market"]
                    for r in rows
                    if r["proxy_entry_offset_from_submission_market"] is not None
                ]
            ),
            "same_direction_return5_fraction": sum(
                r["directional_return5_at_submission"] > 0
                for r in rows
                if r["directional_return5_at_submission"] is not None
            )
            / max(1, sum(r["directional_return5_at_submission"] is not None for r in rows)),
        }
    tails = {}
    for name, state_file in [
        ("first_batch", "study_v003/holdout_state.json"),
        (
            "resting_target",
            "resting_target_v002/development_previously_seen_holdout_dates_state.json",
        ),
    ]:
        state = json.loads((source / state_file).read_text())
        tails[name] = {}
        for sleeve in ["small", "large"]:
            rows = [p for p in state["closed"] if p["sleeve"] == sleeve]
            wins = [p["net_pnl"] for p in rows if p["net_pnl"] > 0]
            losses = [-p["net_pnl"] for p in rows if p["net_pnl"] < 0]
            total_loss = math.fsum(losses)
            tails[name][sleeve] = {
                "whole_positions": len(rows),
                "mean_win": statistics_mean(wins),
                "mean_loss": statistics_mean(losses),
                "loss_quantiles": quantiles(losses),
                "worst10_loss_share": math.fsum(sorted(losses, reverse=True)[:10]) / total_loss
                if total_loss
                else None,
                "largest_losses": sorted(rows, key=lambda p: p["net_pnl"])[:10],
            }
    save(output / "entry_roles.json", records)
    save(output / "addition_submission_audit.json", add_records)
    save(output / "tail_attribution.json", tails)
    save(
        output / "mechanism_summary.json",
        {
            "groups_crypto": groups,
            "reconciled_additions": len(add_records),
            "preposted_additions": sum(r["created_before_observed_entry"] for r in add_records),
            "scope": "12hour episode gap is a declared assumption; final average fill is a quote proxy only; public reported PNL is not own net account return",
            "previous_position_summary_sha256": sha(
                source.parent
                / "leader_lab_20261024/damorne_public_profile_v003/positions_normalized.json"
            ),
            "input_hashes": {
                p: sha(source / p)
                for p in ["audit_v001/reconciled_positions.json", "study_v003/order_events.json"]
            },
            "source_sha256": sha(Path(__file__)),
        },
    )
    print(
        json.dumps({"groups_crypto": groups, "adds": add_records}, ensure_ascii=False), flush=True
    )


def statistics_mean(values):
    return math.fsum(values) / len(values) if values else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-owner", type=Path)
    args = parser.parse_args()
    diagnose(
        args.source.absolute(),
        args.output.absolute(),
        args.cache_owner.absolute() if args.cache_owner else None,
    )
