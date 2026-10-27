"""Receipt-causal native futures book reconstruction; indicative sweeps only."""

import argparse
import json
from decimal import Decimal
from pathlib import Path

import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest


class Book:
    def __init__(self, snapshot):
        self.bids = {Decimal(p): Decimal(q) for p, q in snapshot["bids"] if Decimal(q)}
        self.asks = {Decimal(p): Decimal(q) for p, q in snapshot["asks"] if Decimal(q)}
        self.snapshot_id = snapshot["lastUpdateId"]
        self.last_id, self.applied = None, 0

    def update(self, data):
        if data["u"] < self.snapshot_id or (self.last_id is not None and data["u"] <= self.last_id):
            return False
        if self.last_id is None:
            if not data["U"] <= self.snapshot_id <= data["u"]:
                raise ValueError("Initial snapshot has no overlapping native delta")
        elif data["pu"] != self.last_id:
            raise ValueError("Native pu/u chain has a gap")
        for side, key in [(self.bids, "b"), (self.asks, "a")]:
            for p, q in data[key]:
                price, quantity = Decimal(p), Decimal(q)
                if price <= 0 or quantity < 0:
                    raise ValueError("Invalid depth level")
                if quantity:
                    side[price] = quantity
                else:
                    side.pop(price, None)
        if not self.bids or not self.asks or max(self.bids) >= min(self.asks):
            raise ValueError("Empty or crossed reconstructed book")
        self.last_id, self.applied = data["u"], self.applied + 1
        return True


def reconstruct(records, clock_ns):
    book, buffered, snapshot_received, last_received, last_line = None, [], None, None, None
    for rec in records:
        if rec["received_ns"] > clock_ns:
            continue
        if rec["channel"] == "depthSnapshot":
            if not rec.get("generated"):
                raise ValueError("Snapshot provenance missing")
            book = Book(rec["data"])
            snapshot_received = rec["received_ns"]
            for pending in buffered:
                if book.update(pending["data"]):
                    last_received, last_line = pending["received_ns"], pending["line"]
            buffered = []
        else:
            if book is None:
                buffered.append(rec)
            elif book.update(rec["data"]):
                last_received, last_line = rec["received_ns"], rec["line"]
    if book is None or book.last_id is None:
        raise ValueError("No receipt-available overlapping snapshot and delta")
    return book, {
        "snapshot_received_ns": snapshot_received,
        "last_update_received_ns": last_received,
        "last_line": last_line,
    }


def sell_sweep(book, quantity):
    remaining, cash, levels = Decimal(quantity), Decimal(0), []
    if remaining <= 0:
        raise ValueError("Nonpositive requested quantity")
    for price in sorted(book.bids, reverse=True):
        take = min(remaining, book.bids[price])
        cash += take * price
        levels.append({"price": str(price), "available": str(book.bids[price]), "taken": str(take)})
        remaining -= take
        if not remaining:
            break
    return {
        "quantity": str(quantity),
        "unfilled": str(remaining),
        "displayed_depth_sufficient": remaining == 0,
        "vwap": str(cash / Decimal(quantity)) if not remaining else None,
        "levels": levels,
    }


def run(source, output):
    output.mkdir(parents=True, exist_ok=False)
    verify_manifest(source)
    plan = read(source / "plan.json")
    for key in ["parent", "parent_audit"]:
        verify_manifest(Path(plan[key]))
    for path, digest in plan["source_hashes"].items():
        assert sha(Path(path)) == digest
    save(
        output / "registration.json",
        {
            "source_sha256": sha(Path(__file__)),
            "raw_completion_sha256": sha(source / "completion.json"),
            "before_new_depth_reconstruction": True,
            "new_financial_accounts": 0,
            "checks": plan["checks"],
        },
    )
    clock = pd.Timestamp("2025-01-01T00:05:00Z").value
    records, counts, last = {s: [] for s in plan["symbols"]}, {}, None
    assert read(source / "depth.receipt.json")["status"] == 200
    for number, line in enumerate((source / "depth.decoded").open(), 1):
        if not line.strip():
            raise ValueError("Captured disconnect requires a separate reset audit")
        stamp, raw = line.rstrip().split(" ", 1)
        ns = pd.Timestamp(stamp).value
        assert last is None or ns >= last
        last = ns
        message = json.loads(raw)
        parts = message["stream"].split("@")
        symbol, channel = parts[:2]
        assert symbol in records and channel in plan["channels"]
        if channel == "depth":
            assert message["data"]["s"].lower() == symbol
        records[symbol].append(
            {
                "line": number,
                "received_ns": ns,
                "channel": channel,
                "data": message["data"],
                "generated": message.get("generated", False),
            }
        )
        counts[message["stream"]] = counts.get(message["stream"], 0) + 1
    witnesses = read(Path(plan["parent_audit"]) / "quote_witnesses.json")
    proofs, sweeps, failures = [], [], []
    for symbol, rs in records.items():
        try:
            book, timing = reconstruct(rs, clock)
        except ValueError as e:
            failures.append({"symbol": symbol, "error": str(e)})
            continue
        reference = next(q for q in witnesses if q["symbol"] == symbol)
        bbo = reference["quote_before"]
        best_bid, best_ask = max(book.bids), min(book.asks)
        match = (
            best_bid == Decimal(bbo["b"])
            and book.bids[best_bid] == Decimal(bbo["B"])
            and best_ask == Decimal(bbo["a"])
            and book.asks[best_ask] == Decimal(bbo["A"])
        )
        proofs.append(
            {
                "symbol": symbol,
                "snapshot_id": book.snapshot_id,
                "last_update_id": book.last_id,
                "applied_deltas": book.applied,
                "native_chain_valid": True,
                "clock_ns": clock,
                **timing,
                "depth_update_age_ms": (clock - timing["last_update_received_ns"]) / 10**6,
                "best_bid": str(best_bid),
                "best_bid_quantity": str(book.bids[best_bid]),
                "best_ask": str(best_ask),
                "best_ask_quantity": str(book.asks[best_ask]),
                "independent_channel_bbo_exact": match,
                "bbo_received_ns": bbo["received_ns"],
                "bbo_update_id": bbo["u"],
            }
        )
        save(
            output / (symbol + "_asof_book.json"),
            {
                "bids": [[str(p), str(q)] for p, q in sorted(book.bids.items(), reverse=True)],
                "asks": [[str(p), str(q)] for p, q in sorted(book.asks.items())],
            },
        )
        for witness in witnesses:
            if witness["symbol"] != symbol:
                continue
            e = witness["original_event"]
            sweep = sell_sweep(book, str(e["quantity"]))
            vwap = Decimal(sweep["vwap"]) if sweep["vwap"] else None
            model = Decimal(str(e["price"]))
            sweeps.append(
                {
                    "symbol": symbol,
                    "slippage_bps": witness["slippage_bps"],
                    "case_aliases": witness["case_aliases"],
                    "original_model_price": str(model),
                    **sweep,
                    "model_price_compatible_with_displayed_sweep": model <= vwap if vwap else None,
                    "optimistic_shortfall_bps": float((model - vwap) / vwap * 10000)
                    if vwap
                    else None,
                    "optimistic_gross_cash_difference": float(
                        (model - vwap) * Decimal(str(e["quantity"]))
                    )
                    if vwap
                    else None,
                    "indicative_fractional_snapshot_only_not_actual_execution": True,
                }
            )
    save(output / "chain_proofs.json", proofs)
    save(output / "sweeps.json", sweeps)
    save(output / "failures.json", failures)
    save(
        output / "verification.json",
        {
            "feed_counts": counts,
            "successful_native_chains": len(proofs),
            "chain_failures": failures,
            "both_bbo_channels_match": len(proofs) == 2
            and all(p["independent_channel_bbo_exact"] for p in proofs),
            "sweeps": len(sweeps),
            "optimistic_model_cases": sum(
                s["model_price_compatible_with_displayed_sweep"] is False for s in sweeps
            ),
            "new_financial_accounts": 0,
            "actual_execution_or_capacity_certified": False,
            "external_review_performed": False,
            "independent_oos": False,
            "profit_goal_met": False,
            "orders_enabled": False,
        },
    )
    save(
        output / "completion.json",
        {
            "audit_finished": True,
            "manifest": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
            "profit_goal_met": False,
        },
    )
    print(read(output / "verification.json"), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.source, a.output)
