"""Inspect the fixed EXAS lifecycle without producing new strategy returns.

Run directly, avoiding the application's automatic dotenv initializer.
This is an internal provenance review, not an independent attestation.
"""

import argparse
import csv
import hashlib
import html
import importlib.util
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "data/research/equity_exas_lifecycle_20261026_v001"
PIT = ROOT / "data/research/equity_pit_20261026_v001"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def dates(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = [row["date"] for row in rows]
    if len(result) != len(set(result)):
        raise ValueError(f"Duplicate daily dates: {path.name}")
    return set(result)


def text(body):
    # Analysis projection only. The original document is separately byte-bound.
    decoded = body.decode("utf-8")
    decoded = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", "", decoded, flags=re.I | re.S)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]*>", " ", decoded))).strip()


def anchor(projected, phrase, expected_occurrences=1):
    positions = [match.start() for match in re.finditer(re.escape(phrase), projected)]
    if len(positions) != expected_occurrences:
        raise ValueError(f"Source clause is missing or ambiguous: {phrase}")
    return {"phrase": phrase, "positions_in_text_projection": positions}


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    helper_path = Path(__file__).with_name("arelle_review.py")
    spec = importlib.util.spec_from_file_location("raw_sgml_review", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    api_path = PIT / "raw/CIK0001124140.json"
    recent = json.loads(api_path.read_text())["filings"]["recent"]
    api = dict(zip(recent["accessionNumber"], recent["acceptanceDateTime"], strict=True))
    inputs = {str(api_path.relative_to(ROOT)): sha(api_path)}
    captures = []
    for receipt_path in sorted((SOURCE / "raw").glob("*.receipt.json")):
        receipt = json.loads(receipt_path.read_text())
        raw = receipt_path.with_name(receipt_path.name.removesuffix(".receipt.json"))
        if receipt.get("sha256") is not None:
            if sha(raw) != receipt["sha256"]:
                raise ValueError(f"Raw capture changed: {raw.name}")
            inputs[str(raw.relative_to(ROOT))] = sha(raw)
        inputs[str(receipt_path.relative_to(ROOT))] = sha(receipt_path)
        captures.append({"receipt": str(receipt_path.relative_to(ROOT)), **receipt})
    filings = []
    originals = {}
    for filename, accession, document_names in [
        (
            "initial_8k.txt",
            "0001193125-25-288896",
            ["d38144d8k.htm", "d38144dex21.htm", "d38144dex991.htm"],
        ),
        ("preclose_8k.txt", "0001193125-26-116862", ["d92966d8k.htm"]),
        ("completed_8k.txt", "0001193125-26-118700", ["d128802d8k.htm"]),
    ]:
        raw_path = SOURCE / "raw" / filename
        raw = raw_path.read_bytes()
        stamp = re.search(rb"<ACCEPTANCE-DATETIME>(\d{14})", raw)
        if stamp is None:
            raise ValueError("Missing original SEC acceptance clock")
        accepted = (
            datetime.strptime(stamp.group(1).decode(), "%Y%m%d%H%M%S")
            .replace(tzinfo=ZoneInfo("America/New_York"))
            .astimezone(ZoneInfo("UTC"))
        )
        if accepted != datetime.fromisoformat(api[accession]):
            raise ValueError("Original SEC and submissions API clocks differ")
        extracted = helper.extract_documents(raw)
        docs = []
        for name in document_names:
            entry = extracted[name]
            body = entry.pop("bytes")
            target = output / name
            target.write_bytes(body)
            if body != raw[entry["source_start"] : entry["source_end"]]:
                raise ValueError("Original document byte boundaries differ")
            projection = text(body)
            target.with_suffix(".txt").write_text(projection + "\n")
            originals[name] = projection
            docs.append({**entry, "sha256": sha(target)})
        filings.append(
            {
                "accession": accession,
                "parent_file": str(raw_path.relative_to(ROOT)),
                "parent_sha256": sha(raw_path),
                "accepted_at_utc": accepted.isoformat(),
                "research_available_at_utc": (accepted + timedelta(hours=1)).isoformat(),
                "availability_assumption": "Existing one-hour research processing lag; not proof of first public release.",
                "original_api_acceptance_match": True,
                "documents": docs,
            }
        )
    agreement = originals["d38144dex21.htm"]
    clauses = [
        anchor(agreement, phrase)
        for phrase in [
            "no holder of Book-Entry Shares shall be required to deliver a Certificate or letter of transmittal",
            "as promptly as reasonably practicable after the Effective Time",
            "No interest will be paid or accrued",
            "no event later than the fifth Business Day thereafter",
        ]
    ]
    initial = originals["d38144dex991.htm"]
    preclose = originals["d92966d8k.htm"]
    completion = originals["d128802d8k.htm"]
    # These exact issuer clauses must be checked; a press forecast is not a close.
    timeline = {
        "initial_forecast": anchor(initial, "second quarter of 2026"),
        "preclose_expectation": anchor(preclose, "March 23, 2026"),
        "completion_cash_right": anchor(completion, "right to receive $105.00 in cash"),
        "trading_halted_before_open": anchor(completion, "halted prior to the opening of trading"),
    }
    nasdaq_path = PIT / "supplemental/EXAS_nasdaq_corporate_action.htm"
    nasdaq_receipt = nasdaq_path.with_suffix(".htm.receipt.json")
    saved = json.loads(nasdaq_receipt.read_text())
    if sha(nasdaq_path) != saved["sha256"]:
        raise ValueError("Frozen Nasdaq bulletin changed")
    bulletin = text(nasdaq_path.read_bytes())
    exchange_checks = {
        "revision_notice": anchor(bulletin, "UPDATED: Merger closed", expected_occurrences=2),
        "last_trading_day": anchor(bulletin, "Last Trading Date: March 20, 2026"),
        "suspension_day": anchor(
            bulletin, "Marketplace Effective Date for Suspension: March 24, 2026"
        ),
        "closed_before_open": anchor(bulletin, "closed prior to the market open on March 23, 2026"),
    }
    for path in [nasdaq_path, nasdaq_receipt]:
        inputs[str(path.relative_to(ROOT))] = sha(path)
    # Use an already frozen, actually observed trading calendar, not weekdays.
    calendar_path = PIT / "prices/TSLA.US.csv"
    old_path = PIT / "prices/EXAS.US.csv"
    later_path = PIT / "supplemental/EXAS_chartexchange_observed.csv"
    calendar = dates(calendar_path)
    covered = dates(old_path) | dates(later_path)
    interval = sorted(day for day in calendar if "2024-06-27" <= day <= "2024-09-26")
    missing = sorted(set(interval) - covered)
    for path in [calendar_path, old_path, later_path]:
        inputs[str(path.relative_to(ROOT))] = sha(path)
    coverage = json.loads((SOURCE / "raw/exas_vendor_coverage.json").read_text())
    stock_meta = coverage["stocks"]
    if stock_meta.get("unavailable") or not stock_meta.get("found"):
        raise ValueError("Vendor coverage is unavailable or a completed no-match")
    records = stock_meta["records"]
    offers = json.loads((SOURCE / "raw/vendor_catalog.json").read_text())["offers"]
    offer = next(item for item in offers["products"] if item["id"] == "stk-day")
    evidence = {
        "schema": "equity-lifecycle-source-review.v1",
        "security": {"historical_ticker": "EXAS", "cusip": "30063P105", "issuer_cik": "0001124140"},
        "filings": filings,
        "contract_clauses": clauses,
        "timeline_clauses": timeline,
        "exchange_clauses": exchange_checks,
        "book_entry_transmittal_required": False,
        "certificate_holders_require_surrender_or_loss_affidavit": True,
        "five_business_days_means": "Deadline to mail certificate-holder instructions, not a universal payment or broker availability deadline.",
        "cash_right": {
            "gross_usd_per_eligible_share": 105,
            "interest": False,
            "withholding_and_fees_verified": False,
        },
        "last_trading_day": "2026-03-20",
        "merger_effective_day": "2026-03-23",
        "suspension_effective_day": "2026-03-24",
        "exact_merger_effective_clock": None,
        "broker_cash_received_at": None,
        "broker_cash_spendable_at": None,
        "cash_receipt_inferred_from_contract": False,
        "nasdaq_current_page_is_updated": True,
        "nasdaq_page_first_published_at": None,
        "nasdaq_page_historical_revision_available_at": None,
        "updated_page_backdated_as_signal": False,
        "remaining_missing_sessions": missing,
        "missing_session_count": len(missing),
        "vendor_declaration": {
            "source": stock_meta["source"],
            "scope": stock_meta["scope"],
            "release_id": coverage.get("releaseId"),
            "records": records,
            "file_contents_acquired": False,
            "period_completeness_verified": False,
            "reported_offer": {key: offer[key] for key in ["id", "amount", "billing"]},
            "currency": offers["currency"],
            "price_basis": offers["priceBasis"],
            "purchase_authorized": False,
            "purchase_made": False,
        },
        "long_window_execution_allowed": False,
        "new_economic_accounts": 0,
        "external_independent_attestation": None,
        "profit_goal_met": False,
    }
    write(output / "lifecycle-evidence.json", evidence)
    write(output / "capture-attempts.json", captures)
    write(
        output / "registration.json",
        {
            "schema": "equity-lifecycle-registration.v1",
            "source_hashes": {
                str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)),
                str(helper_path.relative_to(ROOT)): sha(helper_path),
            },
            "inputs_sha256": inputs,
            "financial_account_evaluation": False,
            "scope": "A source audit of already fixed EXAS; no strategy selection, parameter search or return evaluation.",
        },
    )
    print(
        json.dumps(
            {
                "filings": len(filings),
                "raw_document_roundtrips": sum(len(x["documents"]) for x in filings),
                "missing_sessions": len(missing),
                "cash_available_at": None,
                "new_economic_accounts": 0,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args().output.absolute())
