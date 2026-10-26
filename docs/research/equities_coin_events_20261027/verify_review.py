"""Original-source and frozen simulated-inventory checks; not external attestation."""

import csv
import hashlib
import json
import re
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "results/research/equity_coin_events_20261027_v002"
DATA = ROOT / "data/research/equity_coin_events_20261027_v001"
PIT = ROOT / "data/research/equity_pit_20261026_v001"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text())


reg = read(RUN / "registration.json")
assert all(sha(ROOT / f) == h for f, h in reg["inputs_sha256"].items())
assert all(sha(ROOT / f) == h for f, h in reg["source_sha256"].items())
assert (
    sha(RUN / "analysis_source.py")
    == reg["source_sha256"]["tradingagents/research/equities/coin_event_review.py"]
)
events = read(RUN / "events.json")
summary = read(RUN / "summary.json")
extractions = read(RUN / "extraction.json")
assert (
    len(events) == 7
    and summary["regular_earnings_releases"] == 6
    and summary["preliminary_financial_events"] == 1
)
assert all(
    x["absolute_first_public_at"] is None and not x["source_clock_for_intraday_signal_eligible"]
    for x in events
)
api = {x["accessionNumber"]: x for x in read(PIT / "events.json")}
roundtrips = 0
substantive = 0
for record in extractions:
    parent = ROOT / record["parent_file"]
    body = parent.read_bytes()
    assert sha(parent) == record["parent_sha256"]
    accepted = (
        datetime.strptime(
            re.search(rb"<ACCEPTANCE-DATETIME>(\d{14})", body).group(1).decode(), "%Y%m%d%H%M%S"
        )
        .replace(tzinfo=ZoneInfo("America/New_York"))
        .astimezone(UTC)
    )
    event = next(x for x in events if x["accession"] == record["accession"])
    assert (
        accepted
        == datetime.fromisoformat(api[record["accession"]]["acceptanceDateTime"])
        == datetime.fromisoformat(event["raw_sec_accepted_at_utc"])
    )
    for d in record["html_documents"]:
        child = RUN / record["accession"] / d["filename"]
        assert (
            child.read_bytes() == body[d["source_start"] : d["source_end"]]
            and sha(child) == d["sha256"]
        )
        roundtrips += 1
        substantive += d["role"] in ["primary_8k", "substantive_exhibit"]
    assert datetime.fromisoformat(
        event["research_available_at_with_existing_one_hour_assumption"]
    ) == accepted + timedelta(hours=1)
clock_checks = []
for event in events:
    if "quarter" not in event:
        continue
    raw = (DATA / "raw" / (event["quarter"] + "-release.htm")).read_text()
    original = json.loads(
        next(
            block
            for block in re.findall(
                r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>', raw, re.S
            )
            if "NewsArticle" in block
        )
    )
    assert event["ir_release_metadata"]["datePublished"] == original["datePublished"]
    assert event["publication_clock_quarantined"] and not event["metadata_automatically_shifted"]
    if "wire_original_page_clock" in event:
        w = event["wire_original_page_clock"]
        lo, hi = map(datetime.fromisoformat, w["reported_utc_interval"])
        assert (hi - lo).total_seconds() == 60
        nominal = (lo - datetime.fromisoformat(original["datePublished"])).total_seconds()
        assert nominal == w["nominal_IR_to_wire_start_seconds"]
        assert nominal in [18000, 21600]
        accepted = datetime.fromisoformat(event["raw_sec_accepted_at_utc"])
        order = (
            "SEC_before_reported_wire_minute"
            if accepted < lo
            else "reported_wire_minute_before_SEC"
            if hi <= accepted
            else "order_unknown_within_wire_display_minute"
        )
        clock_checks.append(
            {
                "quarter": event["quarter"],
                "IR_literal_clock": original["datePublished"],
                "wire_reported_minute_utc": [lo.isoformat(), hi.isoformat()],
                "nominal_offset_hours": nominal / 3600,
                "SEC_clock_order": order,
                "global_first_public_at": None,
            }
        )
assert len(clock_checks) == 4
calendar = {r["date"]: r for r in csv.DictReader((PIT / "prices/signal_sessions.csv").open())}
for e in events:
    accepted = datetime.fromisoformat(e["raw_sec_accepted_at_utc"])
    expected = next(
        day
        for day, row in calendar.items()
        if "2023-01-03" <= day <= "2024-06-26" and datetime.fromisoformat(row["close"]) > accepted
    )
    assert expected == e["filing_reaction_session"]
# Recompute COIN contribution directly from the original simulated fills and prices.
state_path = (
    ROOT
    / "results/research/equities_pit_stocks_20261026_v002/US_french_mom_12_2_10000_15bps/state.json"
)
state = read(state_path)
price_path = PIT / "prices/COIN.US.csv"
prices = {r["date"]: r for r in csv.DictReader(price_path.open())}
rows = [
    r
    for r in csv.DictReader(
        (
            ROOT / "results/research/equity_depth_attribution_20261026_v002/daily-stock-pnl.csv"
        ).open()
    )
    if r["account"] == "US_french_mom_12_2_10000_15bps" and r["symbol"] == "COIN.US"
]
qty = 0
previous = Decimal(prices["2022-12-30"]["close"])
total = Decimal(0)
selected = Decimal(0)
max_error = Decimal(0)
selected_days = {e["filing_reaction_session"] for e in events}
for row in rows:
    day = row["day"]
    p = prices[day]
    opened, close = Decimal(p["open"]), Decimal(p["close"])
    assert int(row["prior_quantity"]) == qty
    overnight = qty * (opened - previous)
    fills = [f for f in state["fills"] if f["symbol"] == "COIN.US" and f["day"] == day]
    for fill in fills:
        assert Decimal(str(fill["raw_open"])) == opened and fill["clock"] == int(
            datetime.fromisoformat(calendar[day]["open"]).timestamp() * 1000
        )
        qty += fill["quantity"] if fill["side"] == "BUY" else -fill["quantity"]
    assert qty == int(row["closing_quantity"]) and qty >= 0
    fees = sum((Decimal(str(f["fee"])) for f in fills), Decimal(0))
    slip = sum((Decimal(str(f["slippage"])) for f in fills), Decimal(0))
    # Fill costs are positive payments; the attribution CSV stores signed negative PnL components.
    assert abs(fees + Decimal(row["fees"])) < Decimal("1e-10") and abs(
        slip + Decimal(row["slippage"])
    ) < Decimal("1e-10")
    assert Decimal(p["dividend"]) == Decimal(p["split"]) == 0
    value = overnight + qty * (close - opened) - fees - slip
    error = abs(value - Decimal(row["net_marked_pnl"]))
    max_error = max(max_error, error)
    assert error < Decimal("1e-7")
    total += Decimal(row["net_marked_pnl"])
    if day in selected_days:
        selected += Decimal(row["net_marked_pnl"])
    previous = close
assert (
    len(rows) == 372
    and total == Decimal(summary["unchanged_coin_marked_pnl"])
    and selected == Decimal(summary["filing_reaction_session_coin_net_marked_pnl"])
)
assert total - selected == Decimal(summary["all_other_sessions_coin_net_marked_pnl"])
result = {
    "schema": "equity-coin-event-verification.v1",
    "role": "Internal source, precision and frozen simulated-inventory checks; no independent external attestation.",
    "source_hashes_matched": True,
    "seven_original_acceptance_clocks_matched": True,
    "original_HTML_byte_roundtrips": roundtrips,
    "substantive_documents": substantive,
    "auxiliary_render_documents": roundtrips - substantive,
    "IR_metadata_quarantined": 6,
    "original_wire_display_clock_crosschecks": clock_checks,
    "certified_absolute_first_public_clocks": 0,
    "unchanged_frozen_COIN_contribution_rows": len(rows),
    "max_original_simulated_fill_price_PnL_error": str(max_error),
    "source_inventory_price_inputs_sha256": {
        str(p.relative_to(ROOT)): sha(p) for p in [state_path, price_path]
    },
    "new_economic_accounts": 0,
    "external_independent_attestation": False,
    "profit_goal_met": False,
}
(ROOT / "docs/research/equities_coin_events_20261027/verification.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print(json.dumps(result))
