"""Independently audit recorded current rules, clocks and receipt-causal books."""

import argparse
import hashlib
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path

import numpy as np

from research_tools.dated_carry.audit_depth_witness import Book
from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest


class SpotBook(Book):
    """Spot uses lastUpdateId+1 continuity, not futures' pu rule."""

    def update(self, data):
        previous = self.snapshot_id if self.last_id is None else self.last_id
        if data["u"] <= previous:
            return False
        if not data["U"] <= previous + 1 <= data["u"]:
            raise ValueError("Spot U/u chain has a gap")
        for side, key in [(self.bids, "b"), (self.asks, "a")]:
            for p, q in data[key]:
                price, quantity = Decimal(p), Decimal(q)
                if price <= 0 or quantity < 0:
                    raise ValueError("Invalid spot book level")
                if quantity:
                    side[price] = quantity
                else:
                    side.pop(price, None)
        if not self.bids or not self.asks or max(self.bids) >= min(self.asks):
            raise ValueError("Empty/crossed spot book")
        self.last_id, self.applied = data["u"], self.applied + 1
        return True


def reconstruct(records, snapshot, snapshot_received_ns, watermark, market):
    if snapshot_received_ns > watermark:
        raise ValueError("Snapshot was received after common watermark")
    selected = [r for r in records if r["received_wall_ns"] <= watermark]
    book = SpotBook(snapshot) if market == "spot" else Book(snapshot)
    # All records after the snapshot's sequence, including buffered earlier
    # arrivals, are available only once the snapshot has actually arrived.
    last_received = None
    for record in selected:
        if book.update(record["data"]):
            last_received = record["received_wall_ns"]
    if book.last_id is None:
        raise ValueError("No overlapping usable delta after snapshot")
    return book, max(snapshot_received_ns, last_received)


def rule_fingerprint(symbol):
    value = {**symbol, "filters": sorted(symbol["filters"], key=lambda f: f["filterType"])}
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def audit(source, output):
    output.mkdir(parents=True, exist_ok=False)
    verify_manifest(source)
    plan = read(source / "plan.json")
    for path, digest in plan["source_hashes"].items():
        assert sha(Path(path)) == digest
    registration, selection = read(source / "registration.json"), read(source / "selection.json")
    save(
        output / "registration.json",
        {
            "before_new_clock_book_and_rule_comparisons": True,
            "source_sha256": sha(Path(__file__)),
            "futures_book_checker_sha256": sha(
                Path("research_tools/dated_carry/audit_depth_witness.py")
            ),
            "capture_completion_sha256": sha(source / "completion.json"),
            "new_financial_accounts": 0,
            "watermark": "Minimum last received frame time across both streams; no price-based selection.",
            "BBO_alignment": "Last received BBO by native update ID <= reconstructed depth ID, all received <= watermark; no future-received quote.",
        },
    )
    clocks, rules, failures, stream_records = [], [], [], {}
    pre_meta, post_meta = {}, {}
    limit = plan["clock"]["maximum_server_vs_client_interval_error_ms"]
    for market in ["spot", "future"]:
        for phase in ["pre", "post"]:
            r = read(source / f"{market}_time_{phase}.receipt.json")
            d = json.loads((source / f"{market}_time_{phase}.decoded").read_bytes())
            low, high = r["sent_wall_ns"] / 1e6, r["received_wall_ns"] / 1e6
            valid = (
                r["status"] == 200
                and r["rtt_ms"] <= plan["clock"]["maximum_request_rtt_ms"]
                and low - limit <= d["serverTime"] <= high + limit
            )
            clocks.append(
                {
                    "market": market,
                    "phase": phase,
                    "valid": valid,
                    "rtt_ms": r["rtt_ms"],
                    "server_minus_client_offset_interval_ms": [
                        d["serverTime"] - high,
                        d["serverTime"] - low,
                    ],
                    "dedicated_time_endpoint": True,
                    "third_party_clock_certification": False,
                }
            )
        pre_meta[market] = json.loads((source / f"{market}_metadata_pre.decoded").read_bytes())
        post_meta[market] = json.loads((source / f"{market}_metadata_post.decoded").read_bytes())
        assert pre_meta[market].get("exchangeFilters") == post_meta[market].get("exchangeFilters")
        pre = {s["symbol"]: s for s in pre_meta[market]["symbols"]}
        post = {s["symbol"]: s for s in post_meta[market]["symbols"]}
        meta_receipt = read(source / f"{market}_metadata_pre.receipt.json")
        ws = read(source / (market + "_stream.receipt.json"))
        assert (
            registration["registered_wall_ns"]
            <= meta_receipt["received_wall_ns"]
            <= selection["selected_at_wall_ns"]
            <= ws["attempted_wall_ns"]
        )
        assert ws["connected"] and not ws.get("error_type")
        for asset, pair in selection["symbols"].items():
            s = pair[market]["symbol"]
            assert pre[s] == pair[market]
            unchanged = rule_fingerprint(pre[s]) == rule_fingerprint(post[s])
            rules.append(
                {
                    "asset": asset,
                    "market": market,
                    "symbol": s,
                    "native_filters": pre[s]["filters"],
                    "pre_rule_sha256": rule_fingerprint(pre[s]),
                    "post_rule_sha256": rule_fingerprint(post[s]),
                    "before_quotes": True,
                    "unchanged_through_probe": unchanged,
                    "actual_client_fees_or_order_acceptance_certified": False,
                }
            )
        rows, counts, previous_wall, previous_mono = [], Counter(), None, None
        allowed_symbols = {pair[market]["symbol"].lower() for pair in selection["symbols"].values()}
        with (source / (market + "_native.ndjson")).open() as file:
            for number, line in enumerate(file, 1):
                raw = json.loads(line)
                native = json.loads(raw["native_raw"])
                parts = native["stream"].split("@")
                symbol, channel = parts[:2]
                assert symbol in allowed_symbols and channel in ["depth", "bookTicker"]
                assert native["data"]["s"].lower() == symbol
                wall, mono = raw["received_wall_ns"], raw["received_monotonic_ns"]
                assert wall >= ws["connected_wall_ns"] >= selection["selected_at_wall_ns"]
                assert previous_wall is None or wall >= previous_wall
                assert previous_mono is None or mono >= previous_mono
                previous_wall, previous_mono = wall, mono
                rows.append(
                    {
                        "line": number,
                        "received_wall_ns": wall,
                        "received_monotonic_ns": mono,
                        "symbol": symbol,
                        "channel": channel,
                        "data": native["data"],
                    }
                )
                counts[native["stream"]] += 1
        assert rows and len(counts) == 4 and len(rows) == ws["messages"]
        stream_records[market] = {"rows": rows, "counts": dict(counts)}
    watermark = min(r["rows"][-1]["received_wall_ns"] for r in stream_records.values())
    books, ages, asof = [], [], []
    for market, captured in stream_records.items():
        rows = [r for r in captured["rows"] if r["received_wall_ns"] <= watermark]
        timed = [r for r in rows if isinstance(r["data"].get("E"), int)]
        apparent = np.array([r["received_wall_ns"] / 1e6 - r["data"]["E"] for r in timed])
        duration_wall = (rows[-1]["received_wall_ns"] - rows[0]["received_wall_ns"]) / 1e6
        duration_mono = (rows[-1]["received_monotonic_ns"] - rows[0]["received_monotonic_ns"]) / 1e6
        ages.append(
            {
                "market": market,
                "observed_messages": len(captured["rows"]),
                "messages_before_common_watermark": len(rows),
                "native_channel_counts": captured["counts"],
                "event_time_known_messages": len(timed),
                "apparent_event_to_receive_ms": {
                    "min": float(apparent.min()),
                    "p50": float(np.median(apparent)),
                    "p99": float(np.quantile(apparent, 0.99)),
                    "max": float(apparent.max()),
                },
                "within_registered_coarse_clock_limit": bool((np.abs(apparent) <= limit).all()),
                "wall_minus_monotonic_elapsed_ms": duration_wall - duration_mono,
                "spot_bookTicker_event_time_unknown_not_inferred": market == "spot",
            }
        )
        for asset, pair in selection["symbols"].items():
            symbol = pair[market]["symbol"].lower()
            delta = [r for r in rows if r["symbol"] == symbol and r["channel"] == "depth"]
            receipt = read(source / f"{market}_{asset}_snapshot.receipt.json")
            assert receipt["status"] == 200
            assert receipt["sent_wall_ns"] >= max(
                read(source / (m + "_stream.receipt.json"))["connected_wall_ns"]
                for m in ["spot", "future"]
            )
            snapshot = json.loads((source / f"{market}_{asset}_snapshot.decoded").read_bytes())
            try:
                book, known_at = reconstruct(
                    delta, snapshot, receipt["received_wall_ns"], watermark, market
                )
                compatible = [
                    r
                    for r in rows
                    if r["symbol"] == symbol
                    and r["channel"] == "bookTicker"
                    and r["data"]["u"] <= book.last_id
                ]
                bbo = (
                    max(compatible, key=lambda r: (r["data"]["u"], r["received_wall_ns"]))
                    if compatible
                    else None
                )
                bid, ask = max(book.bids), min(book.asks)
                same = bool(
                    bbo
                    and (bid, book.bids[bid], ask, book.asks[ask])
                    == tuple(Decimal(bbo["data"][k]) for k in ["b", "B", "a", "A"])
                )
                books.append(
                    {
                        "asset": asset,
                        "market": market,
                        "symbol": symbol,
                        "snapshot_id": book.snapshot_id,
                        "depth_update_id": book.last_id,
                        "deltas_applied": book.applied,
                        "sequence_valid": True,
                        "common_watermark_ns": watermark,
                        "state_known_at_ns": known_at,
                        "book_age_at_watermark_ms": (watermark - known_at) / 1e6,
                        "update_id_aligned_BBO_exact": same,
                        "BBO_update_id": bbo["data"]["u"] if bbo else None,
                        "different_channels_same_collector_not_independent_venue": True,
                    }
                )
                asof.append(
                    {
                        "asset": asset,
                        "market": market,
                        "symbol": symbol,
                        "bid": str(bid),
                        "bid_quantity": str(book.bids[bid]),
                        "ask": str(ask),
                        "ask_quantity": str(book.asks[ask]),
                        "watermark_ns": watermark,
                        "source_is_not_a_fill_or_signal": True,
                    }
                )
            except Exception as e:
                failures.append(
                    {"asset": asset, "market": market, "error": str(e), "type": type(e).__name__}
                )
    ready = (
        all(c["valid"] for c in clocks)
        and all(r["unchanged_through_probe"] for r in rules)
        and len(books) == 4
        and not failures
        and all(b["update_id_aligned_BBO_exact"] for b in books)
        and all(a["within_registered_coarse_clock_limit"] for a in ages)
    )
    for name, value in [
        ("clock_proofs", clocks),
        ("reported_rule_epochs", rules),
        ("native_ages", ages),
        ("book_proofs", books),
        ("asof_quotes", asof),
        ("failures", failures),
    ]:
        save(output / (name + ".json"), value)
    save(
        output / "summary.json",
        {
            "source_observability_passed": ready,
            "captured_native_messages": sum(len(s["rows"]) for s in stream_records.values()),
            "reported_rule_epochs": len(rules),
            "valid_book_sequences": len(books),
            "BBO_exact_comparisons": sum(b["update_id_aligned_BBO_exact"] for b in books),
            "selected_contracts": {
                a: v["future"]["symbol"] for a, v in selection["symbols"].items()
            },
            "common_watermark_ns": watermark,
            "source_failures": failures,
            "new_financial_accounts": 0,
            "forward_strategy_validation_completed": False,
            "historical_2025_rules_certified": False,
            "real_execution_or_capacity_certified": False,
            "external_review_performed": False,
            "profit_goal_met": False,
            "orders_enabled": False,
        },
    )
    save(
        output / "completion.json",
        {
            "passed": ready,
            "manifest": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
            "profit_goal_met": False,
        },
    )
    print(read(output / "summary.json"), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    audit(a.source, a.output)
