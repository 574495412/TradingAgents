"""Audit raw COIN earnings-event clocks and frozen account contributions.

Run directly. Source-reported website metadata is not certified public time.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import re
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from html.parser import HTMLParser
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[3]
PLAN = Path(__file__).with_name("coin_event_review_plan.json")
DATA = ROOT / "data/research/equity_coin_events_20261027_v001"
PIT = ROOT / "data/research/equity_pit_20261026_v001"
IR_KEYS = ["2022Q4", "2023Q1", "2023Q2", "2023Q3", "2023Q4", "2024Q1"]
IR_LABELS = [
    "Fourth Quarter and Full Year 2022",
    "First Quarter 2023",
    "Second Quarter 2023",
    "Third Quarter 2023",
    "Fourth Quarter and Full Year 2023",
    "First Quarter 2024",
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ArticleMetadata(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.active = False
        self.parts = []
        self.blocks = []

    def handle_starttag(self, tag, attrs):
        if tag == "script" and dict(attrs).get("type") == "application/ld+json":
            self.active = True
            self.parts = []

    def handle_data(self, data):
        if self.active:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if tag == "script" and self.active:
            self.blocks.append(json.loads("".join(self.parts)))
            self.active = False


def article_metadata(body):
    parser = ArticleMetadata()
    parser.feed(body.decode("utf-8"))
    articles = [block for block in parser.blocks if block.get("@type") == "NewsArticle"]
    if len(articles) != 1:
        raise ValueError("Missing or ambiguous issuer NewsArticle metadata")
    return {key: articles[0].get(key) for key in ["headline", "datePublished", "dateModified"]}


def first_reaction(accepted, calendar):
    if accepted.tzinfo is None:
        raise ValueError("SEC acceptance clock lacks timezone")
    for day, opened, close in calendar:
        if close > accepted:
            phase = "before_open" if accepted <= opened else "during_regular_session"
            return day, phase
    raise ValueError("No complete reaction session in the frozen window")


def wire_minute_interval(displayed):
    for suffix, daylight in [("Eastern Daylight Time", True), ("Eastern Standard Time", False)]:
        if displayed.endswith(suffix):
            value = datetime.strptime(
                displayed.removesuffix(suffix).strip(), "%b %d, %Y %I:%M %p"
            ).replace(tzinfo=ZoneInfo("America/New_York"))
            if bool(value.dst()) != daylight:
                raise ValueError("Publisher timezone label conflicts with date")
            lo = value.astimezone(UTC)
            return lo, lo + timedelta(minutes=1)
    raise ValueError("Unrecognized publisher clock label")


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    plan = json.loads(PLAN.read_text())
    identity_path = PIT / "identities.json"
    identity = next(x for x in json.loads(identity_path.read_text()) if x["symbol"] == "COIN.US")
    if (identity["cusip"], identity["cik"]) != ("19260Q107", "0001679788"):
        raise ValueError("Security and issuer identity mismatch")
    events_path = PIT / "events.json"
    events = [
        x
        for x in json.loads(events_path.read_text())
        if x["cik"] == identity["cik"]
        and "2023-01-01" <= x["filingDate"] < "2024-06-27"
        and "2.02" in x.get("items", "").split(",")
    ]
    if [x["accessionNumber"] for x in events] != plan["selected_accessions"]:
        raise ValueError("Frozen event selection changed")
    extractor_path = Path(__file__).with_name("arelle_review.py")
    projection_path = Path(__file__).with_name("lifecycle_review.py")
    extractor = load_file("coin_original_extractor", extractor_path)
    projection = load_file("coin_text_projection", projection_path)
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in [PLAN, events_path, identity_path]}
    all_receipts = []
    for receipt_path in sorted((DATA / "raw").glob("*.receipt.json")):
        receipt = json.loads(receipt_path.read_text())
        raw = receipt_path.with_name(receipt_path.name.removesuffix(".receipt.json"))
        if sha(raw) != receipt["sha256"]:
            raise ValueError("Original capture differs from source receipt")
        for p in [raw, receipt_path]:
            inputs[str(p.relative_to(ROOT))] = sha(p)
        all_receipts.append({"file": str(raw.relative_to(ROOT)), **receipt})
    cal_path = PIT / "prices/signal_sessions.csv"
    with cal_path.open(newline="") as handle:
        calendar = [
            (r["date"], datetime.fromisoformat(r["open"]), datetime.fromisoformat(r["close"]))
            for r in csv.DictReader(handle)
            if plan["window"][0] <= r["date"] <= plan["window"][1]
        ]
    pnl_path = ROOT / plan["fixed_account_diagnostic"]["source"]
    with pnl_path.open(newline="") as handle:
        pnl = [
            row
            for row in csv.DictReader(handle)
            if row["account"] == plan["fixed_account_diagnostic"]["account"]
        ]
    for p in [cal_path, pnl_path]:
        inputs[str(p.relative_to(ROOT))] = sha(p)
    wire_path = DATA / "businesswire-observed-clocks.json"
    wire_observations = json.loads(wire_path.read_text())
    wires = {x["quarter"]: x for x in wire_observations["records"]}
    if len(wires) != len(wire_observations["records"]):
        raise ValueError("Ambiguous duplicate publisher observations")
    inputs[str(wire_path.relative_to(ROOT))] = sha(wire_path)
    public_records, clock_issues, extraction_records = [], [], []
    quarter_index = 0
    for event in events:
        accession = event["accessionNumber"]
        parent_path = DATA / "raw" / (accession + ".txt")
        raw = parent_path.read_bytes()
        if re.search(rb"CENTRAL INDEX KEY:\s*0001679788\b", raw) is None:
            raise ValueError("Original filing belongs to a different issuer")
        stamp = re.search(rb"<ACCEPTANCE-DATETIME>(\d{14})", raw)
        if stamp is None:
            raise ValueError("Original acceptance clock missing")
        accepted = (
            datetime.strptime(stamp.group(1).decode(), "%Y%m%d%H%M%S")
            .replace(tzinfo=ZoneInfo("America/New_York"))
            .astimezone(UTC)
        )
        if accepted != datetime.fromisoformat(event["acceptanceDateTime"]):
            raise ValueError("Raw acceptance clock differs from original API")
        folder = output / accession
        folder.mkdir()
        docs = extractor.extract_documents(raw)
        doc_records = []
        for name, entry in docs.items():
            if not name.endswith((".htm", ".html")):
                continue
            body = entry.pop("bytes")
            target = folder / name
            target.write_bytes(body)
            role = (
                "primary_8k"
                if name == event["primaryDocument"]
                else "auxiliary_render"
                if name == "R1.htm"
                else "substantive_exhibit"
            )
            doc_records.append({**entry, "role": role, "sha256": sha(target)})
        primary = projection.text((folder / event["primaryDocument"]).read_bytes())
        preliminary = "preliminary unaudited financial and operating results" in primary.lower()
        regular = "shareholder letter" in primary.lower() and any(
            x["role"] == "substantive_exhibit" for x in doc_records
        )
        if preliminary == regular:
            raise ValueError(
                "Unclassified financial event; do not force all Item2.02 into earnings releases"
            )
        row = {
            "accession": accession,
            "event_kind": "preliminary_guidance_and_restructuring"
            if preliminary
            else "regular_earnings_with_shareholder_letter",
            "raw_sec_accepted_at_utc": accepted.isoformat(),
            "sec_api_acceptance_match": True,
            "research_available_at_with_existing_one_hour_assumption": (
                accepted + timedelta(hours=1)
            ).isoformat(),
            "absolute_first_public_at": None,
            "absolute_first_public_certified": False,
            "source_clock_for_intraday_signal_eligible": False,
        }
        if regular:
            key = IR_KEYS[quarter_index]
            quarter_index += 1
            release_path = DATA / "raw" / (key + "-release.htm")
            schedule_path = (
                DATA
                / "raw"
                / (key + ("-schedule-canonical.htm" if key == "2024Q1" else "-schedule.htm"))
            )
            release_meta, schedule_meta = (
                article_metadata(release_path.read_bytes()),
                article_metadata(schedule_path.read_bytes()),
            )
            label = IR_LABELS[quarter_index - 1]
            if (
                release_meta["headline"] != f"Coinbase Releases {label} Shareholder Letter"
                or schedule_meta["headline"]
                != f"Coinbase Announces Date of {label} Financial Results"
            ):
                raise ValueError("Issuer page title does not match its role")
            schedule_text = projection.text(schedule_path.read_bytes())
            match = re.search(
                r"on (?:Monday|Tuesday|Wednesday|Thursday|Friday), ([A-Za-z]+ \d{1,2}, \d{4}), after market close",
                schedule_text,
            )
            if match is None:
                raise ValueError("No original advance release date/slot")
            scheduled_day = datetime.strptime(match.group(1), "%B %d, %Y").date().isoformat()
            if scheduled_day != event["reportDate"]:
                raise ValueError("Planned quarter event does not match filing; no automatic join")
            planned_session = next(x for x in calendar if x[0] == scheduled_day)
            literal = datetime.fromisoformat(release_meta["datePublished"])
            if literal.date().isoformat() != event["reportDate"]:
                raise ValueError("Issuer release day and original event disagree")
            if literal.tzinfo is None:
                issue = "timezone_missing"
                comparison = None
            else:
                comparison = (literal.astimezone(UTC) - planned_session[2]).total_seconds()
                issue = (
                    "literal_metadata_precedes_planned_after_close_slot"
                    if comparison < 0
                    else "source_clock_not_independently_observed"
                )
            row.update(
                quarter=key,
                ir_release_metadata=release_meta,
                ir_schedule_metadata=schedule_meta,
                planned_release_day=scheduled_day,
                planned_slot="after_market_close",
                actual_release_slot_verified=False,
                metadata_clock_issue=issue,
                metadata_automatically_shifted=False,
                ir_literal_minus_same_day_market_close_seconds=comparison,
                publication_clock_quarantined=True,
            )
            clock_issues.append(
                {
                    "quarter": key,
                    "ir_release_source": str(release_path.relative_to(ROOT)),
                    "ir_schedule_source": str(schedule_path.relative_to(ROOT)),
                    "source_reported_clock": release_meta["datePublished"],
                    "literal_minus_sec_acceptance_seconds": (
                        literal.astimezone(UTC) - accepted
                    ).total_seconds()
                    if literal.tzinfo
                    else None,
                    "issue": issue,
                    "automatic_timezone_correction_applied": False,
                    "actual_first_public_at": None,
                }
            )
            if key in wires:
                observed = wires[key]
                lo, hi = wire_minute_interval(observed["visible_publication_text"])
                if lo.date().isoformat() != scheduled_day:
                    raise ValueError("Wire observation belongs to a different event day")
                source_clock = {
                    "source_url": observed["url"],
                    "source_displayed_publication": observed["visible_publication_text"],
                    "precision": "minute; seconds unknown",
                    "reported_utc_interval": [lo.isoformat(), hi.isoformat()],
                    "nominal_IR_to_wire_start_seconds": (
                        lo - literal.astimezone(UTC)
                    ).total_seconds()
                    if literal.tzinfo
                    else None,
                    "absolute_first_public_certified": False,
                }
                row["wire_original_page_clock"] = source_clock
                clock_issues[-1]["wire_original_page_clock"] = source_clock
        day, phase = first_reaction(accepted, calendar)
        coin_rows = [r for r in pnl if r["day"] == day and r["symbol"] == "COIN.US"]
        if len(coin_rows) != 1:
            raise ValueError("Missing/duplicate frozen stock contribution")
        contribution = coin_rows[0]
        row.update(
            filing_reaction_session=day,
            accepted_relative_to_reaction_session=phase,
            intraday_reaction_separable_from_daily_data=phase != "during_regular_session",
            coin_contribution={
                k: contribution[k]
                for k in [
                    "prior_quantity",
                    "closing_quantity",
                    "overnight_price_pnl",
                    "intraday_price_pnl",
                    "fees",
                    "slippage",
                    "net_marked_pnl",
                ]
            },
            total_account_marked_pnl_same_session=str(
                sum(Decimal(r["net_marked_pnl"]) for r in pnl if r["day"] == day)
            ),
            first_news_reaction_claimed=False,
            causality_identified=False,
        )
        public_records.append(row)
        extraction_records.append(
            {
                "accession": accession,
                "parent_file": str(parent_path.relative_to(ROOT)),
                "parent_sha256": sha(parent_path),
                "html_documents": doc_records,
            }
        )
    if quarter_index != len(IR_KEYS):
        raise ValueError("Unexpected count of regular releases")
    selected_days = {e["filing_reaction_session"] for e in public_records}
    all_coin = [r for r in pnl if r["symbol"] == "COIN.US"]
    total = sum(Decimal(r["net_marked_pnl"]) for r in all_coin)
    selected = sum(Decimal(r["net_marked_pnl"]) for r in all_coin if r["day"] in selected_days)
    summary = {
        "schema": "equity-coin-event-diagnostic.v1",
        "selected_filings": len(events),
        "regular_earnings_releases": quarter_index,
        "preliminary_financial_events": len(events) - quarter_index,
        "raw_acceptance_clocks_matched": len(events),
        "source_reported_ir_clock_anomalies": len(clock_issues),
        "original_wire_page_clock_crosschecks": len(wires),
        "certified_first_public_release_clocks": 0,
        "unchanged_coin_marked_pnl": str(total),
        "unique_filing_reaction_sessions": len(selected_days),
        "filing_reaction_session_coin_net_marked_pnl": str(selected),
        "all_other_sessions_coin_net_marked_pnl": str(total - selected),
        "zero_inventory_filing_reaction_sessions": [
            e["filing_reaction_session"]
            for e in public_records
            if int(e["coin_contribution"]["prior_quantity"])
            == int(e["coin_contribution"]["closing_quantity"])
            == 0
        ],
        "scope": "Accounting proximity to SEC filing only; not causal earnings PnL, global first-news reaction, net strategy counterfactual or executable event avoidance.",
        "historical_consensus_acquired": False,
        "PEAD_reproduced": False,
        "new_economic_accounts": 0,
        "external_independent_attestation": False,
        "profit_goal_met": False,
    }
    for name, value in [
        ("events.json", public_records),
        ("clock-issues.json", clock_issues),
        ("extraction.json", extraction_records),
        ("source-receipts.json", all_receipts),
        ("summary.json", summary),
    ]:
        write(output / name, value)
    write(
        output / "registration.json",
        {
            "schema": "equity-coin-event-registration.v1",
            "inputs_sha256": inputs,
            "source_sha256": {
                str(p.relative_to(ROOT)): sha(p)
                for p in [Path(__file__), extractor_path, projection_path]
            },
            "original_selection_is_post_result_focus": True,
            "new_economic_accounts": 0,
        },
    )
    (output / "analysis_source.py").write_bytes(Path(__file__).read_bytes())
    (output / "plan.json").write_bytes(PLAN.read_bytes())
    print(json.dumps(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args().output.absolute())
