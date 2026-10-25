"""One live-public-data paper step using frozen rules and persistent owned inventory.

This process has no exchange order client. Re-run --once after the next minute to
advance the same paper account; a repeated minute cannot create duplicate fills.
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .damorne_data import get
from .damorne_research import MINUTE, account, feature_frame, metadata_for
from .evidence import save, sha

COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]


def check_collection_deadline(decision_clock, completed_ms):
    if not decision_clock <= completed_ms < decision_clock + MINUTE:
        raise ValueError(
            "Collection missed the future execution clock; no backdated paper forecast"
        )


def _capture(symbol, when, folder):
    values = {}
    for kind, endpoint in [("trade", "klines"), ("mark", "markPriceKlines")]:
        url = f"https://fapi.binance.com/fapi/v1/{endpoint}?symbol={symbol}&interval=1m&endTime={when + MINUTE - 1}&limit=1500"
        body = get(url)
        path = folder / f"{symbol}_{kind}.json"
        path.write_bytes(body)
        rows = json.loads(body)
        if not isinstance(rows, list) or len(rows) < 120:
            raise ValueError("Live public bars unavailable")
        values[kind] = (
            pd.DataFrame(rows, columns=COLUMNS).apply(pd.to_numeric).set_index("open_time")
        )
    trade, mark = values["trade"], values["mark"]
    if (
        not trade.index.equals(mark.index)
        or trade.index[-1] != when
        or not np.all(np.diff(trade.index) == MINUTE)
    ):
        raise ValueError("Live public market clock is incomplete; no fill or interpolation")
    f = feature_frame(trade, mark)
    url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&startTime={int(f.index[0])}&endTime={when + MINUTE - 1}&limit=1000"
    body = get(url)
    (folder / f"{symbol}_funding.json").write_bytes(body)
    rates = json.loads(body)
    if not isinstance(rates, list) or len(rates) >= 1000 or not rates:
        raise ValueError("Explicit live funding history required")
    f["funding"] = 0.0
    for row in rates:
        clock = int(row["fundingTime"]) // MINUTE * MINUTE
        if clock in f.index:
            f.loc[clock, "funding"] += float(row["fundingRate"])
    return symbol, f


def step(study, output):
    selection_path = study / "selection.json"
    selection = json.loads(selection_path.read_text())
    freeze = json.loads((study / "candidate_freeze.json").read_text())
    source = Path(__file__).with_name("damorne_research.py")
    if sha(source) != freeze["study_source_sha256"]:
        raise ValueError("Frozen strategy implementation changed; create a new paper cohort")
    policy = selection["candidate"]
    if not any(policy == c for c in freeze["candidates"]):
        raise ValueError("Selection differs from frozen candidate population")
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.json"
    if not manifest.exists():
        save(
            manifest,
            {
                "started_utc": datetime.now(UTC).isoformat(),
                "selection_sha256": sha(selection_path),
                "study_source_sha256": sha(source),
                "paper_source_sha256": sha(Path(__file__)),
                "candidate": policy,
                "capital_is_simulated": True,
                "orders_enabled": False,
                "scope": "DIAGNOSTIC_PAPER_ONLY"
                if not selection["eligible_on_validation"]
                else "RESEARCH_PAPER_ONLY",
            },
        )
    binding = json.loads(manifest.read_text())
    if binding["selection_sha256"] != sha(selection_path) or binding["paper_source_sha256"] != sha(
        Path(__file__)
    ):
        raise ValueError("Paper policy/source identity drift; do not merge cohorts")
    latest = output / "latest.json"
    previous = json.loads(latest.read_text()) if latest.exists() else None
    raw_time = get("https://fapi.binance.com/fapi/v1/time", limit=4096)
    now_ms = int(json.loads(raw_time)["serverTime"])
    when = now_ms // MINUTE * MINUTE
    if now_ms - when < 10_000:
        print(json.dumps({"status": "current_bar_publication_grace", "orders_enabled": False}))
        return
    if previous and when < previous["next_ms"]:
        print(
            json.dumps(
                {"status": "no_new_minute", "next_ms": previous["next_ms"], "orders_enabled": False}
            )
        )
        return
    start = previous["next_ms"] if previous else when
    if when - start > 120 * MINUTE:
        raise ValueError(
            "Paper gap exceeds2hours; stop and reconcile rather than invent live continuity"
        )
    cycle = output / (datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8])
    cycle.mkdir()
    (cycle / "server_time.json").write_bytes(raw_time)
    catalog_raw = get("https://fapi.binance.com/fapi/v1/exchangeInfo", limit=4_000_000)
    (cycle / "exchange_info.json").write_bytes(catalog_raw)
    catalog = json.loads(catalog_raw)
    symbols = freeze["market_symbols"]
    with ThreadPoolExecutor(max_workers=3) as pool:
        market = dict(pool.map(lambda symbol: _capture(symbol, when, cycle), symbols))
    for f in market.values():
        if start not in f.index:
            raise ValueError("Cannot resume missing paper clock")
    completed_time = get("https://fapi.binance.com/fapi/v1/time", limit=4096)
    (cycle / "collection_completed_time.json").write_bytes(completed_time)
    check_collection_deadline(when, int(json.loads(completed_time)["serverTime"]))
    # Last/current candle is unclosed: its OPEN is the paper fill proxy only.
    # All decision features come from the preceding, already-closed minute.
    state = None
    if previous:
        state = json.loads(Path(previous["state_path"]).read_text())
        if sha(Path(previous["state_path"])) != previous["state_sha256"]:
            raise ValueError("Paper checkpoint changed")
    summary, state = account(
        policy, market, metadata_for(catalog), start, when + MINUTE, state=state
    )
    save(cycle / "state.json", state)
    save(cycle / "score.json", summary)
    save(
        cycle / "source_receipts.json",
        [{"path": p.name, "sha256": sha(p)} for p in sorted(cycle.glob("*.json"))],
    )
    pointer = {
        "next_ms": when + MINUTE,
        "state_path": str((cycle / "state.json").absolute()),
        "state_sha256": sha(cycle / "state.json"),
        "last_cycle": str(cycle.absolute()),
        "production_eligible": False,
        "orders_enabled": False,
    }
    temporary = output / ("latest-" + uuid.uuid4().hex + ".tmp")
    save(temporary, pointer)
    os.replace(temporary, latest)
    print(
        json.dumps(
            {
                "status": "paper_step_completed",
                "output": str(cycle),
                "market_symbols": len(symbols),
                "first_cold_start_cycle": previous is None,
                "closed_positions": summary["complete_positions"],
                "pending_orders_are_paper_intents_only": len(state["pending"]),
                "production_eligible": False,
                "orders_enabled": False,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--once", action="store_true", required=True)
    args = parser.parse_args()
    step(args.study.absolute(), args.output.absolute())
