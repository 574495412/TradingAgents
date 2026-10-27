"""Audit a bounded raw historical book/quote window; no simulated fills or profit."""

import argparse
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path

import pandas as pd

from factors.common.option_units import premium_cash_value, underlying_notional_fits
from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest


def apply_book(levels, updates):
    result = dict(levels)
    for action, price, amount in updates:
        price, amount = Decimal(str(price)), Decimal(str(amount))
        if price <= 0 or amount < 0:
            raise ValueError("Invalid native book level")
        if action == "new":
            if price in result or amount == 0:
                raise ValueError("Duplicate/zero new book level")
            result[price] = amount
        elif action == "change":
            if price not in result or amount == 0:
                raise ValueError("Change without owned book level")
            result[price] = amount
        elif action == "delete":
            if price not in result:
                raise ValueError("Delete without owned book level")
            del result[price]
        else:
            raise ValueError("Unknown native book action")
    return result


def bbo(bids, asks):
    if not bids or not asks:
        return None
    bid, ask = max(bids), min(asks)
    if bid > ask:
        return None
    return (bid, bids[bid], ask, asks[ask])


def plain(value):
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def audit(source, output):
    output.mkdir(exist_ok=False, parents=True)
    verify_manifest(source)
    parent = Path(read(source / "plan.json")["parent"])
    verify_manifest(parent)
    text = (source / "one_minute_option_witness.decoded").read_text()
    receipt = read(source / "one_minute_option_witness.receipt.json")
    if receipt["status"] != 200 or receipt.get("error_type"):
        raise ValueError("No complete native witness")
    save(
        output / "registration.json",
        {
            "source_completion_sha256": sha(source / "completion.json"),
            "audit_source_sha256": sha(Path(__file__)),
            "option_unit_math_sha256": sha(Path("factors/common/option_units.py")),
            "after_source_access_before_source_audit": True,
            "economic_evaluation_allowed": False,
        },
    )
    bids, asks, change_id, last_book_time = {}, {}, None, None
    counts = Counter()
    states = []
    crosschecks = []
    trios = {}
    previous_local = None
    disconnects = 0
    gaps = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            disconnects += 1
            bids, asks, change_id, last_book_time = {}, {}, None, None
            continue
        local, message = line.split(" ", 1)
        clock = pd.Timestamp(local)
        if previous_local is not None and clock < previous_local:
            raise ValueError("Collector timestamps out of order")
        if not pd.Timestamp("2025-01-01T00:00:00Z") <= clock < pd.Timestamp("2025-01-01T00:01:00Z"):
            raise ValueError("Witness exceeds registered local-minute window")
        previous_local = clock
        packet = json.loads(message, parse_float=Decimal)
        if packet.get("method") != "subscription":
            raise ValueError("Unexpected raw native message")
        data = packet["params"]["data"]
        channel = packet["params"]["channel"].split(".")[0]
        if data["instrument_name"] != "BTC-31JAN25-100000-C":
            raise ValueError("Mixed option identity")
        counts[channel] += 1
        stamp = data["timestamp"]
        trios.setdefault(stamp, {})
        if channel == "book":
            if data["type"] == "snapshot":
                bids, asks, change_id = {}, {}, None
                counts["snapshots"] += 1
            elif data["type"] == "change":
                if change_id is None or data["prev_change_id"] != change_id:
                    gaps.append(
                        {
                            "line": number,
                            "expected": change_id,
                            "observed_previous": data.get("prev_change_id"),
                        }
                    )
                    bids, asks, change_id = {}, {}, None
                    continue
            else:
                raise ValueError("Unknown native book event type")
            bids = apply_book(bids, data["bids"])
            asks = apply_book(asks, data["asks"])
            change_id = data["change_id"]
            last_book_time = stamp
            value = bbo(bids, asks)
            if value is None:
                counts["unusable_book_states"] += 1
            else:
                state = {
                    "line": number,
                    "local_timestamp": local,
                    "exchange_timestamp_ms": stamp,
                    "change_id": change_id,
                    "bbo": value,
                }
                states.append(state)
                trios[stamp]["book"] = state
        elif channel in {"quote", "ticker"}:
            native = (
                data["best_bid_price"],
                data["best_bid_amount"],
                data["best_ask_price"],
                data["best_ask_amount"],
            )
            value = bbo(bids, asks)
            comparable = value is not None and last_book_time <= stamp
            crosschecks.append(
                {
                    "line": number,
                    "channel": channel,
                    "exchange_timestamp_ms": stamp,
                    "book_exchange_timestamp_ms": last_book_time,
                    "local_timestamp": local,
                    "comparable_past_received_book": comparable,
                    "exact_bbo_match": native == value if comparable else None,
                    "native_bbo": native,
                    "book_bbo": value,
                }
            )
            trios[stamp][channel] = {"line": number, "local_timestamp": local, "data": data}
        else:
            raise ValueError("Undeclared channel")
    aligned = [
        (t, row)
        for t, row in sorted(trios.items())
        if all(k in row for k in ["book", "quote", "ticker"])
    ]
    if not aligned:
        raise ValueError("No common native-timestamp book/quote/ticker observation")
    stamp, first = aligned[0]
    ticker = first["ticker"]["data"]
    quote = first["quote"]["data"]
    reference = read(parent / "historical_contract_reference.json")["result"]
    assert (
        reference["quote_currency"] == "BTC"
        and reference["settlement_currency"] == "BTC"
        and reference["min_trade_amount"] == 0.1
        and reference["contract_size"] == 1
    )
    native_index = str(ticker["index_price"])
    q = str(reference["min_trade_amount"])
    bid = str(quote["best_bid_price"])
    ask = str(quote["best_ask_price"])
    mark = str(ticker["mark_price"])
    bid_cash = premium_cash_value(q, bid, native_index)
    ask_cash = premium_cash_value(q, ask, native_index)
    mark_cash = premium_cash_value(q, mark, native_index)
    illustration = {
        "exchange_timestamp_ms": stamp,
        "first_raw_trio": plain(first),
        "native_index_price_USD_per_BTC": native_index,
        "distinct_forward_underlying_price_USD": str(ticker["underlying_price"]),
        "native_amount_BTC_current_archived_reference": q,
        "bid_premium_cash_USD": float(bid_cash),
        "ask_premium_cash_USD": float(ask_cash),
        "model_mark_cash_USD": float(mark_cash),
        "mark_minus_bid_USD": float(mark_cash - bid_cash),
        "spread_USD": float(ask_cash - bid_cash),
        "minimum_underlying_notional_USD": float(Decimal(q) * Decimal(native_index)),
        "illustrative_cash_budget_USD": 10000,
        "illustrative_underlying_gross_cap": 0.8,
        "underlying_notional_necessary_check_pass": underlying_notional_fits(
            q, native_index, 10000, ".8"
        ),
        "USD_USDT_conversion_assumption_for_comparison_only": "1:1; no funded account or verified FX conversion",
        "historical_original_minimum_size_certified": False,
        "fee_or_margin_history_certified": False,
        "order_submitted": False,
        "no_strategy_price_or_underlying_notional_definition_adopted": True,
    }
    save(output / "first_observation_units.json", illustration)
    save(output / "book_states.json", plain(states))
    save(output / "channel_crosschecks.json", plain(crosschecks))
    save(output / "sequence_gaps.json", gaps)
    summary = {
        "messages": sum(counts[c] for c in ["book", "quote", "ticker"]),
        "counts": dict(counts),
        "disconnects": disconnects,
        "sequence_gaps": len(gaps),
        "valid_book_states": len(states),
        "causally_comparable_bbo_checks": sum(
            c["comparable_past_received_book"] for c in crosschecks
        ),
        "exact_bbo_matches": sum(c["exact_bbo_match"] is True for c in crosschecks),
        "bbo_mismatches": sum(c["exact_bbo_match"] is False for c in crosschecks),
        "same_exchange_timestamp_complete_trios": len(aligned),
        "source_only": True,
        "independent_collector_same_exchange": True,
        "third_party_strategy_review": False,
        "full_lifecycle_or_option_chain_qualified": False,
        "new_financial_accounts": 0,
        "economic_evaluation_allowed": False,
        "profit_goal_met": False,
    }
    save(output / "summary.json", summary)
    save(
        output / "completion.json",
        {
            "manifest": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
            "new_financial_accounts": 0,
        },
    )
    print(summary, flush=True)
    print({k: v for k, v in illustration.items() if k != "first_raw_trio"}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    audit(a.source, a.output)
