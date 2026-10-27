"""Challenge frozen modeled entries with native second-recorder trades and BBO.

No financial replay, parameter selection, or replacement fill. Native receipt time
controls quote availability; trade time controls exchange-bar reconstruction.
Observed quantity gcd is a sample lattice, not proof of historical order filters.
"""

import argparse
import csv
import io
import json
import math
import zipfile
from collections import Counter
from decimal import Decimal
from functools import reduce
from pathlib import Path

import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest


def archive_bar(path, begin_ms):
    with zipfile.ZipFile(path) as zipped:
        for row in csv.reader(io.TextIOWrapper(zipped.open(zipped.namelist()[0]))):
            if not row[0].isdigit():
                continue
            time = int(row[0])
            # Official spot timestamps changed to microseconds from Jan2025.
            unit = 1000 if time > 10**14 else 1
            if time // unit == begin_ms:
                return {
                    "open": row[1],
                    "high": row[2],
                    "low": row[3],
                    "close": row[4],
                    "volume": row[5],
                    "quote_volume": row[7],
                    "count": int(row[8]),
                    "archive_timestamp_unit": "us" if unit == 1000 else "ms",
                    "raw": row,
                }
    raise ValueError("Registered entry bar missing from original archive")


def quantity_lattice(values):
    amounts = [Decimal(v) for v in values]
    scale = max(-v.as_tuple().exponent for v in amounts)
    denominator = 10**scale
    return Decimal(reduce(math.gcd, (int(a * denominator) for a in amounts))) / denominator


def latest_known_quote(quotes, clock_ns):
    known = [q for q in quotes if q["received_ns"] <= clock_ns]
    return known[-1] if known else None


def run(source, output):
    output.mkdir(parents=True, exist_ok=False)
    verify_manifest(source)
    plan = read(source / "plan.json")
    original = Path(plan["original_accounts"])
    verify_manifest(original)
    for path, digest in plan["source_hashes"].items():
        assert sha(Path(path)) == digest, path
    save(
        output / "registration.json",
        {
            "before_new_trade_aggregation_and_quote_comparisons": True,
            "source_sha256": sha(Path(__file__)),
            "raw_completion_sha256": sha(source / "completion.json"),
            "checks": plan["comparisons"],
            "price_quote_policy": "latest collector-received quote <= exact old modeled fill clock; freshness reported, never future filled",
            "no_account_or_replacement_fill": True,
        },
    )
    start = pd.Timestamp(plan["capture_window"]["from"])
    start_ms = start.value // 10**6
    end_ms = start_ms + 300000
    clock_ns = end_ms * 10**6
    markets, all_counts, warnings = {}, [], []
    for market, symbols in plan["markets"].items():
        receipt = read(source / f"feed_{market}.receipt.json")
        if receipt["status"] != 200 or receipt.get("error"):
            warnings.append(
                {"market": market, "reason": "native feed unavailable", "receipt": receipt}
            )
            continue
        trades, quotes = {s: [] for s in symbols}, {s: [] for s in symbols}
        previous_receipt, disconnects = None, 0
        for number, line in enumerate((source / f"feed_{market}.decoded").open(), 1):
            if not line.strip():
                disconnects += 1
                continue
            stamp, raw = line.rstrip().split(" ", 1)
            received = pd.Timestamp(stamp).value
            assert previous_receipt is None or received >= previous_receipt
            previous_receipt = received
            m = json.loads(raw)
            symbol, channel = m["stream"].split("@")
            assert symbol in symbols and channel in plan["channels"]
            data = m["data"]
            assert data["s"].lower() == symbol
            rec = {"line": number, "received_at": stamp, "received_ns": received, **data}
            (trades if channel == "trade" else quotes)[symbol].append(rec)
        for symbol in symbols:
            all_counts.append(
                {
                    "market": market,
                    "symbol": symbol,
                    "trades": len(trades[symbol]),
                    "quotes": len(quotes[symbol]),
                    "disconnect_markers": disconnects,
                }
            )
            markets[(market, symbol)] = {"trades": trades[symbol], "quotes": quotes[symbol]}
    aggregations, quote_witnesses, concentration = [], [], []
    for (market, symbol), records in markets.items():
        selected = [r for r in records["trades"] if start_ms <= r["T"] < end_ms]
        if not selected:
            warnings.append(
                {"market": market, "symbol": symbol, "reason": "no witnessed trades in bar"}
            )
            continue
        selected.sort(key=lambda r: (r["T"], r["t"]))
        identifiers = [t["t"] for t in selected]
        assert len(identifiers) == len(set(identifiers))
        base = symbol.upper().split("_")[0]
        path = (
            Path("data/research/spot_perpetual_feasibility_20261026_v001/archives")
            / f"{base}-5m-2025-01.zip"
            if market == "binance"
            else Path("data/research/dated_carry_pilot_20261027_v001")
            / f"{symbol.upper()}-5m-2025-01.zip"
        )
        expected = archive_bar(path, start_ms)
        ps, qs = [Decimal(t["p"]) for t in selected], [Decimal(t["q"]) for t in selected]
        native = {
            "open": ps[0],
            "high": max(ps),
            "low": min(ps),
            "close": ps[-1],
            "volume": sum(qs),
            "quote_volume": sum(p * q for p, q in zip(ps, qs, strict=True)),
            "count": len(selected),
        }
        deltas = {k: str(Decimal(native[k]) - Decimal(expected[k])) for k in native}
        latencies = pd.Series([(t["received_ns"] - t["T"] * 10**6) / 10**6 for t in selected])
        lattice = quantity_lattice(t["q"] for t in selected)
        record = {
            "market": market,
            "symbol": symbol,
            "archive": str(path),
            "archive_sha256": sha(path),
            "native": {k: str(v) if isinstance(v, Decimal) else v for k, v in native.items()},
            "official": expected,
            "native_minus_official": deltas,
            "all_fields_decimal_exact": all(Decimal(d) == 0 for d in deltas.values()),
            "first_trade_id": identifiers[0],
            "last_trade_id": identifiers[-1],
            "missing_interior_trade_ids": sum(
                max(0, b - a - 1) for a, b in zip(identifiers[:-1], identifiers[1:], strict=True)
            ),
            "trade_type_counts": dict(Counter(t.get("X", "not_provided") for t in selected)),
            "sample_quantity_gcd": str(lattice),
            "sample_gcd_is_certified_filter": False,
            "receipt_minus_trade_ms": {
                "min": float(latencies.min()),
                "p50": float(latencies.median()),
                "p99": float(latencies.quantile(0.99)),
                "max": float(latencies.max()),
            },
            "last_bar_trade_received_at": max(t["received_at"] for t in selected),
            "bar_complete_by_modeled_fill_clock": all(
                t["received_ns"] <= clock_ns for t in selected
            ),
        }
        aggregations.append(record)
    unique = {}
    for case in read(original / "registration.json")["cases"]:
        folder = original / "accounts" / case["id"]
        for e in read(folder / "events.json"):
            if e["type"] != "entry":
                continue
            market = "binance" if e["market"] == "spot" else "binance-futures"
            symbol = e["symbol"].lower() + ("_250328" if market == "binance-futures" else "")
            key = (market, symbol, e["time"], e["price"], e["quantity"], e["gross_quantity"])
            unique.setdefault(key, {"cases": [], "event": e, "slippage_bps": case["slippage_bps"]})[
                "cases"
            ].append(case["id"])
    for (market, symbol, when, _, _, _), row in unique.items():
        e = row["event"]
        quotes = markets.get((market, symbol), {}).get("quotes", [])
        now = pd.Timestamp(when).value
        before = latest_known_quote(quotes, now)
        later = next((q for q in quotes if q["received_ns"] > now), None)
        record = {
            "case_aliases": row["cases"],
            "market": market,
            "symbol": symbol,
            "slippage_bps": row["slippage_bps"],
            "original_event": e,
            "quote_before": before,
            "first_quote_after_diagnostic_only": later,
            "quote_is_execution_proof": False,
        }
        if before:
            buy = e["market"] == "spot"
            p = Decimal(before["a" if buy else "b"])
            q = Decimal(before["A" if buy else "B"])
            needed = Decimal(str(e["gross_quantity"]))
            modeled = Decimal(str(e["price"]))
            optimistic = (p - modeled) / p if buy else (modeled - p) / p
            record.update(
                reference_side="ask" if buy else "bid",
                quote_age_ms=(now - before["received_ns"]) / 10**6,
                visible_quantity=str(q),
                requested_gross_quantity=str(needed),
                displayed_top_quantity_sufficient=needed <= q,
                optimistic_price_shortfall_bps=float(optimistic * 10000),
                model_price_compatible_with_observed_side=optimistic <= 0,
            )
            agg = next(
                (a for a in aggregations if a["market"] == market and a["symbol"] == symbol), None
            )
            if agg:
                record["model_quantity_mod_observed_trade_gcd"] = str(
                    needed % Decimal(agg["sample_quantity_gcd"])
                )
        quote_witnesses.append(record)
    for row in read(Path("results/research/dated_carry_accounts_20261027_v002") / "summary.json")[
        "account_rows"
    ]:
        concentration.append(
            {
                "id": row["id"],
                "original_net_pnl": row["metrics"]["net_pnl"],
                "returns_recomputed": False,
            }
        )
    for name, value in [
        ("trade_aggregation", aggregations),
        ("quote_witnesses", quote_witnesses),
        ("feed_counts", all_counts),
        ("warnings", warnings),
        ("unchanged_return_references", concentration),
    ]:
        save(output / (name + ".json"), value)
    save(
        output / "verification.json",
        {
            "native_markets_audited": len(aggregations),
            "unique_model_entry_cases": len(quote_witnesses),
            "case_aliases_total": sum(len(r["case_aliases"]) for r in quote_witnesses),
            "all_native_bars_exactly_match_official": len(aggregations) == 4
            and all(a["all_fields_decimal_exact"] for a in aggregations),
            "unfavorable_quote_comparisons": sum(
                r.get("optimistic_price_shortfall_bps", 0) > 0 for r in quote_witnesses
            ),
            "insufficient_displayed_top_quantity": sum(
                not r.get("displayed_top_quantity_sufficient", False) for r in quote_witnesses
            ),
            "missing_before_quotes": sum(r["quote_before"] is None for r in quote_witnesses),
            "historical_order_filters_certified": False,
            "real_execution_certified": False,
            "third_party_review_performed": False,
            "independent_oos": False,
            "new_financial_accounts": 0,
            "new_return_source_admitted": False,
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
