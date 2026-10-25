"""Fixed momentum/resting-target hypothesis; intrabar fills are explicit research proxies."""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .damorne_research import (
    FEATURES,
    MINUTE,
    ContractBook,
    load_market,
    metadata_for,
    millis,
    score,
)
from .evidence import save, sha


def account(config, market, metadata, start_ms, end_ms, state=None, break_ms=None):
    book = ContractBook(config, metadata)
    if state:
        book.restore(state)
    symbols = sorted(market)
    index = next(iter(market.values())).index
    arrays = {
        s: market[s][["open", "mark", "funding", "high", "low", *FEATURES]].to_numpy(dtype=float)
        for s in symbols
    }
    for i in np.where((index >= start_ms) & (index < end_ms))[0]:
        when = int(index[i])
        if break_ms is not None and when >= break_ms:
            return None, book.snapshot()
        rows = {s: arrays[s][i] for s in symbols}
        marks = {s: r[1] for s, r in rows.items()}
        for (symbol, sign), p in sorted(book.positions.items()):
            value = -sign * p["qty"] * marks[symbol] * rows[symbol][2]
            book.wallet += value
            book.funding += value
            p["funding"] += value
        for order in book.pending:
            if order["due_ms"] != when:
                raise ValueError("Missing exact paper execution clock")
            book.transact(
                "OPEN",
                order["symbol"],
                order["sign"],
                rows[order["symbol"]][0],
                when,
                marks,
                order["sleeve"],
            )
        book.pending = []
        equity = book.equity(marks)
        if equity <= 0:
            raise ValueError("Research account insolvent")
        book.curve.append(
            {
                "timestamp": when,
                "equity": equity,
                "wallet": book.wallet,
                "margin": book.reserve(),
                "positions": len(book.positions),
            }
        )
        if i == 0:
            continue
        for symbol in symbols:
            f = dict(zip(FEATURES, arrays[symbol][i - 1][5:], strict=True))
            if not all(math.isfinite(v) for v in f.values()) or f["quote60"] < 1_000_000:
                continue
            for sign in [1, -1]:
                key = (symbol, sign)
                if key in book.positions:
                    continue
                previous = book.last_close.get(key)
                cooldown = 2 * MINUTE if previous and previous["net_pnl"] > 0 else 60 * MINUTE
                if previous and when - previous["when"] < cooldown:
                    continue
                if not (
                    sign * f["return60"] >= 0.01
                    and sign * f["return15"] > 0
                    and sign * f["return5"] > 0
                    and sign * f["flow"] > 0
                ):
                    continue
                sleeve = (
                    "large"
                    if f["volume_ratio"] >= 2
                    and f["quote60"] >= 5_000_000
                    and sign * f["flow"] > 0.1
                    else "small"
                )
                order = {
                    "decision_ms": when,
                    "due_ms": when + MINUTE,
                    "symbol": symbol,
                    "sign": sign,
                    "action": "OPEN",
                    "sleeve": sleeve,
                }
                book.pending.append(order)
                book.decisions.append(order)
        # Stop has priority when both barriers are seen inside the same bar.
        # Resting target activates after one full minute: no unknown first10seconds fill.
        for (symbol, sign), p in sorted(book.positions.items()):
            row = rows[symbol]
            target = p["entry"] * (1 + sign * config[f"{p['sleeve']}_target"])
            stop = p["entry"] * (1 - sign * config[f"{p['sleeve']}_stop"])
            stop_hit = row[4] <= stop if sign == 1 else row[3] >= stop
            tick = metadata[symbol].get("price_tick", 0)
            target_hit = row[3] > target + tick if sign == 1 else row[4] < target - tick
            active = when >= p["opened_ms"] + MINUTE
            expired = when - p["opened_ms"] >= config[f"{p['sleeve']}_max_minutes"] * MINUTE
            if stop_hit or expired or (active and target_hit):
                price = (
                    (min(row[0], stop) if sign == 1 else max(row[0], stop))
                    if stop_hit
                    else row[0]
                    if expired
                    else target
                )
                # Fees and adverse slippage also charged on target fills; LIMIT never implies maker fee.
                book.transact("CLOSE", symbol, sign, price, when + MINUTE - 1, marks)
    result = score(book, marks)
    result.update(
        resting_target_fill="trade through target by more than one tick; no queue evidence",
        intrabar_ambiguous_stop_first=True,
        target_active_after_one_minute=True,
        hypothetical_barrier_fill_times_not_actual_execution_timestamps=True,
    )
    return result, book.snapshot()


def study(root, output):
    output.mkdir(exist_ok=False)
    protocol, market, quality = load_market(root)
    catalog = json.loads((root / "discovery/catalog.json").read_text())
    crypto = {r["symbol"] for r in catalog["symbols"] if "Crypto" in r.get("underlyingSubType", [])}
    market = {s: f for s, f in market.items() if s in crypto}
    prior = json.loads((root / "study_v003/selection.json").read_text())["candidate"]
    policy = {
        **prior,
        "id": "momentum-resting-target-v1",
        "engine": "resting_target_v1",
        "admission": "momentum_5_15_60",
        "tree": None,
        "small_stop": 0.02,
        "large_stop": 0.02,
        "small_max_minutes": 720,
        "large_max_minutes": 2160,
        "max_additions": 0,
        "continuation": True,
        "evidence_scope": "new hypothesis after first holdout inspected; ALL historical results development-only",
    }
    sources = {
        str(p): sha(p)
        for p in [
            Path(__file__),
            Path(__file__).with_name("damorne_research.py"),
            Path(__file__).with_name("evidence.py"),
        ]
    }
    save(
        output / "candidate_freeze.json",
        {
            "registered_utc": datetime.now(UTC).isoformat(),
            "study_source_sha256": sha(Path(__file__).with_name("damorne_research.py")),
            "additional_source_hashes": sources,
            "market_symbols": sorted(market),
            "candidates": [policy],
            "historical_holdout_already_inspected": True,
            "no_historical_promotion": True,
        },
    )
    save(
        output / "selection.json",
        {
            "candidate": policy,
            "eligible_on_validation": False,
            "scope": "one fixed mechanistic hypothesis; diagnostic paper only; no selection from later outcomes",
        },
    )
    metadata = metadata_for(catalog)
    for row in catalog["symbols"]:
        tick = next(f for f in row["filters"] if f["filterType"] == "PRICE_FILTER")
        metadata[row["symbol"]]["price_tick"] = float(tick["tickSize"])
    scores = []
    for label, lower, upper in [
        ("development_validation_dates", protocol["train_end"], protocol["validation_end"]),
        ("development_previously_seen_holdout_dates", protocol["validation_end"], protocol["end"]),
    ]:
        summary, state = account(policy, market, metadata, millis(lower), millis(upper))
        save(output / f"{label}_score.json", summary)
        save(output / f"{label}_state.json", state)
        scores.append(summary)
        print(
            {
                "period": label,
                "net_return": summary["net_return"],
                "small": summary["sleeves"]["small"]["net_pnl"],
                "large": summary["sleeves"]["large"]["net_pnl"],
            },
            flush=True,
        )
    last_start, end = millis(protocol["validation_end"]), millis(protocol["end"])
    middle = last_start + (end - last_start) // (2 * MINUTE) * MINUTE
    _, prefix = account(policy, market, metadata, last_start, end, break_ms=middle)
    resumed, state = account(
        policy, market, metadata, middle, end, state=json.loads(json.dumps(prefix))
    )
    if resumed != scores[-1] or state != json.loads(
        (output / "development_previously_seen_holdout_dates_state.json").read_text()
    ):
        raise ValueError("Resting-target restart mismatch")
    save(output / "restart_verification.json", {"exact_terminal_state_and_score_equal": True})
    stresses = []
    for extra in [2, 5]:
        summary, _ = account(
            {**policy, "slippage_bps": policy["slippage_bps"] + extra},
            market,
            metadata,
            last_start,
            end,
        )
        stresses.append({"extra_slippage_bps": extra, **summary})
    save(output / "cost_stress.json", stresses)
    if any(sha(Path(path)) != digest for path, digest in sources.items()):
        raise ValueError("Frozen implementation changed")
    save(
        output / "run_status.json",
        {
            "status": "completed",
            "production_eligible": False,
            "historical_results_development_only": True,
            "orders_enabled": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    study(args.root.absolute(), args.output.absolute())
