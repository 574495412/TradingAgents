"""Current original-document and simulated-fill audit, not external acceptance.

Run with the bundled Python containing pypdf. Source PDFs remain unchanged and
local; their metadata describes the retrieved files, not certified first release.
"""

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data/research/equity_exas_market_semantics_20261027_v001"
FROZEN = ROOT / "results/research/equities_pit_stocks_20261026_v002"
PIT = ROOT / "data/research/equity_pit_20261026_v001"
RUN = ROOT / "results/research/equity_exas_market_semantics_20261027_v003"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    documents = []
    pages = {}
    for name, relevant in [
        ("utp-output-4.1a.pdf", [32, 33, 48, 49, 59, 60, 61]),
        ("utp-schedule-march2024.pdf", [1]),
        ("nasdaq-openclose-faq.pdf", [1, 2, 4]),
    ]:
        path = DATA / name
        receipt = read(path.with_suffix(".pdf.receipt.json"))
        assert receipt["status"] == 200 and receipt["pdf_magic_valid"]
        assert sha(path) == receipt["sha256"]
        reader = PdfReader(path)
        page_text = {n: re.sub(r"\s+", " ", reader.pages[n - 1].extract_text()) for n in relevant}
        pages[name] = page_text
        documents.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha(path),
                "url": receipt["url"],
                "observed_at": receipt["observed_at"],
                "pages": len(reader.pages),
                "inspected_text_pages": relevant,
                "pdf_declared_created_at": str(reader.metadata.get("/CreationDate")),
                "pdf_declared_modified_at": str(reader.metadata.get("/ModDate")),
                "metadata_is_verified_publication_or_effective_time": False,
                "historical_2024_rule_version_verified": False,
            }
        )
    assert "October 2026" in pages["utp-output-4.1a.pdf"][48]
    assert "3.14.2 UTP Sale Condition Matrix" in pages["utp-output-4.1a.pdf"][48]
    assert "I Odd Lot Trade No No No No Yes" in pages["utp-output-4.1a.pdf"][48]
    assert "Q Market Center Official Open3 No No Yes No No" in pages["utp-output-4.1a.pdf"][48]
    assert (
        "use the “Q” value as the official market center opening value"
        in pages["utp-output-4.1a.pdf"][49]
    )
    assert (
        "The Opening Cross sets the Nasdaq Official Opening Price"
        in pages["nasdaq-openclose-faq.pdf"][2]
    )
    assert "CTS Output Spec" in (DATA / "chartexchange-conditions-docs.ax.txt").read_text()
    assert "Nasdaq-listed securities Tape C" in (DATA / "utp-technical.ax.txt").read_text()
    # The calendar assigns a modeled execution time; it is not a trade receipt.
    with (PIT / "prices/US_sessions.csv").open(newline="") as handle:
        calendar_rows = list(csv.DictReader(handle))
    calendar = {row["date"]: row for row in calendar_rows}
    accounts, inputs = [], {}
    for state_path in sorted(FROZEN.glob("*/state.json")):
        state = read(state_path)
        fills = [f for f in state["fills"] if f["symbol"] == "EXAS.US"]
        assert all(f["day"] in calendar for f in fills)
        assert all(
            f["clock"] == int(datetime.fromisoformat(calendar[f["day"]]["open"]).timestamp() * 1000)
            for f in fills
        )
        accounts.append(
            {
                "account": state_path.parent.name,
                "EXAS_simulated_fills": len(fills),
                "fills_with_calendar_assigned_clock": len(fills),
                "actual_external_order_receipts": 0,
            }
        )
        inputs[str(state_path.relative_to(ROOT))] = sha(state_path)
    engine = ROOT / "tradingagents/research/equities/continuous.py"
    source = engine.read_text()
    assert 'raw = float(frames[s].at[day, "open"])' in source
    assert "stamp = int(session.open.timestamp() * 1000)" in source
    assert "f.volume.rolling(20).mean()" in source
    assert "float(volumes[s].at[prior]) * 0.001 / lot" in source
    inputs[str(engine.relative_to(ROOT))] = sha(engine)
    for path in sorted(DATA.iterdir()):
        if path.is_file():
            inputs[str(path.relative_to(ROOT))] = sha(path)
    inputs[str((PIT / "prices/US_sessions.csv").relative_to(ROOT))] = sha(
        PIT / "prices/US_sessions.csv"
    )
    result = {
        "schema": "equity-exas-market-semantics.v1",
        "review_kind": "Retrospective source/applicability audit; no new strategy or performance estimate",
        "security": {
            "ticker": "EXAS",
            "cik": "0001124140",
            "CUSIP": "30063P105",
            "listed_venue": "Nasdaq",
            "applicable_consolidated_plan": "UTP/Tape C",
        },
        "documents": documents,
        "current_document_observations": [
            "The vendor condition table explicitly follows CTS; do not assume it is a complete historical Tape-C implementation.",
            "Current UTP specification separates opening prints O from official market-center open Q and distinguishes consolidated/participant statistics.",
            "Current UTP odd-lot I and extended-hours T/U conditions can update volume without updating consolidated high/low/last; this does not identify the cause of the 20 observed differences.",
            "Current Nasdaq FAQ distinguishes opening-cross NOOP from the eligible-trade fallback when no cross exists; a vendor Open label alone does not identify either path.",
            "The file named March 2024 declares 2026 PDF creation metadata. This prevents treating its filename as a historical version certificate; metadata alone does not prove that its rules changed or when first released.",
        ],
        "frozen_engine_facts": {
            "raw_price": "vendor daily open",
            "fill_timestamp": "calendar session open",
            "liquidity_cap": "0.001 times previous 20-session mean full-day vendor volume",
            "auction_or_quote_evidence_consumed": False,
            "day_volume_proves_auction_capacity": False,
        },
        "accounts": accounts,
        "EXAS_simulated_fills": sum(a["EXAS_simulated_fills"] for a in accounts),
        "observed_price_or_volume_difference_causes_resolved": False,
        "historical_2024_tape_and_auction_records_acquired": False,
        "candidate_gap_rows_admitted": 0,
        "external_independent_attestation": False,
        "new_economic_accounts": 0,
        "profit_goal_met": False,
        "source_access_scope": "Official publicly linked documents retrieved without login; public availability does not grant redistribution. Schedule carries a restricted-distribution footer. PDF files remain local and are not published or committed.",
        "required_next_evidence": [
            "Historical rule edition and actual effective/revision dates for Tape C",
            "Source identifier, regular/extended session, price-type and volume/correction cutoff definitions",
            "EXAS trade/auction records with venue, sale conditions, exchange/SIP timestamps, size, cancellation/correction chain",
            "Broker order acceptance/cutoff, execution/partial/reject receipts for execution certification",
        ],
    }
    write(output / "evidence.json", result)
    write(
        output / "registration.json",
        {
            "inputs_sha256": inputs,
            "verifier_sha256": sha(Path(__file__)),
            "is_external_attestation": False,
        },
    )
    (output / "analysis_source.py").write_bytes(Path(__file__).read_bytes())
    print(
        json.dumps(
            {
                "PDFs_bound": len(documents),
                "simulated_EXAS_fills": result["EXAS_simulated_fills"],
                "causes_resolved": False,
                "admitted_gap_rows": 0,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RUN)
    main(parser.parse_args().output)
