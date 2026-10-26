"""Audit a fixed missing-price contract; never replace frozen research inputs.

Run this file directly to avoid the application's dotenv initializer. Public
rendered-table observations are vendor evidence, not exchange tape or an
independent financial attestation. No strategy is fitted or evaluated here.
"""

import argparse
import csv
import hashlib
import json
from datetime import datetime
from decimal import Decimal
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data/research/equity_exas_gap_20261027_v001"
PIT = ROOT / "data/research/equity_pit_20261026_v001"
FROZEN = ROOT / "results/research/equities_pit_stocks_20261026_v002"
CONTRACT = ROOT / "docs/research/equities_exas_lifecycle_20261027/missing-input-contract.json"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


class Table(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows, self.row, self.cell = [], None, None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        elif tag in ("td", "th"):
            self.cell = []

    def handle_data(self, value):
        if self.cell is not None:
            self.cell.append(value)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None:
            self.row.append("".join(self.cell).strip())
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None


def candidate_rows():
    result, captures = {}, []
    expected = ["Date", "Open", "High", "Low", "Close", "Adj. Close", "Change", "Volume"]
    for page in (4, 5):
        path = DATA / f"stockanalysis-page{page}-visible-rows.json"
        capture = read(path)
        rows = capture["rows"][0]
        if rows[0] != expected:
            raise ValueError("Unexpected vendor table schema")
        parser = Table()
        parser.feed((DATA / f"stockanalysis-page{page}-table.html").read_text())
        if parser.rows != rows:
            raise ValueError("Visible DOM and saved table HTML differ")
        captures.append({k: capture[k] for k in ("url", "observed_at", "method")})
        for original in rows[1:]:
            day = datetime.strptime(original[0], "%b %d, %Y").date().isoformat()
            values = [Decimal(x.replace(",", "")) for x in original[1:6]]
            volume = int(original[7].replace(",", ""))
            o, h, lo, c, adjusted = values
            if not all(x.is_finite() and x > 0 for x in values):
                raise ValueError("Invalid prices")
            if not lo <= min(o, c) <= max(o, c) <= h or volume < 0:
                raise ValueError("Invalid OHLC or volume")
            if day in result:
                raise ValueError("Duplicate session")
            result[day] = dict(
                zip(("open", "high", "low", "close", "adj_close"), values, strict=True)
            )
            result[day]["volume"] = volume
    return result, captures


def references(path):
    result = {}
    with path.open(newline="") as handle:
        for original in csv.DictReader(handle):
            row = {k.lower().replace(" ", "_"): v for k, v in original.items()}
            day = row.pop("date")
            if day in result:
                raise ValueError("Duplicate reference session")
            result[day] = {
                k: Decimal(v.replace(",", ""))
                for k, v in row.items()
                if k in ("open", "high", "low", "close", "volume")
            }
    return result


def compare(candidates, reference, days, tolerance):
    output = []
    for day in days:
        differences = {
            k: candidates[day][k] - reference[day][k] for k in ("open", "high", "low", "close")
        }
        output.append(
            {
                "date": day,
                "price_differences_candidate_minus_reference": {
                    k: str(v) for k, v in differences.items()
                },
                "price_within_preexisting_tolerance": all(
                    abs(v) <= tolerance for v in differences.values()
                ),
                "candidate_volume": candidates[day]["volume"],
                "reference_volume": int(reference[day]["volume"]),
                "volume_difference": candidates[day]["volume"] - int(reference[day]["volume"]),
                "volume_definition_or_revision_difference_explained": False,
            }
        )
    return output


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    candidates, captures = candidate_rows()
    contract = read(CONTRACT)["prices"]
    tol = Decimal(str(contract["price_overlap_absolute_usd_tolerance"]))
    left_file = PIT / "supplemental/EXAS_zenodo.csv"
    right_file = PIT / "supplemental/EXAS_chartexchange_observed.csv"
    left, right = references(left_file), references(right_file)
    fixed_left = compare(candidates, left, contract["left_overlap_sessions"], tol)
    fixed_right = compare(candidates, right, contract["right_overlap_sessions"], tol)
    # Additional overlap is descriptive, not an enlarged acceptance tolerance.
    extra = compare(candidates, left, sorted(candidates.keys() & left.keys()), tol)
    required = (
        contract["minimum_missing_sessions"]
        + contract["left_overlap_sessions"]
        + contract["right_overlap_sessions"]
    )
    absent = sorted(set(required) - candidates.keys())
    with (PIT / "prices/TSLA.US.csv").open(newline="") as handle:
        calendar = {row["date"] for row in csv.DictReader(handle)}
    off_calendar = sorted(candidates.keys() - calendar)
    if absent or off_calendar:
        raise ValueError("Missing required or off-calendar rows")
    fixed = fixed_left + fixed_right
    write(
        output / "overlap.json",
        {
            "left_fixed": fixed_left,
            "right_fixed": fixed_right,
            "additional_left_all_observed": extra,
        },
    )
    with (output / "candidate-contract-rows.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "open", "high", "low", "close", "adj_close", "volume"])
        for day in sorted(required):
            r = candidates[day]
            writer.writerow(
                [day, *[r[k] for k in ("open", "high", "low", "close", "adj_close", "volume")]]
            )
    inputs = [
        CONTRACT,
        left_file,
        right_file,
        PIT / "prices/TSLA.US.csv",
        PIT / "supplemental/EXAS_zenodo_record.json",
    ]
    inputs.extend(sorted(DATA.glob("*")))
    states = []
    for path in sorted(FROZEN.glob("*/state.json")):
        inputs.append(path)
        state = read(path)
        fills = [f for f in state["fills"] if f["symbol"] == "EXAS.US" and f["day"] == "2024-06-26"]
        prior = sum(
            f["quantity"] * (1 if f["side"] == "BUY" else -1)
            for f in state["fills"]
            if f["symbol"] == "EXAS.US" and f["day"] < "2024-06-26"
        )
        if fills:
            raise ValueError("Previously observed no-fill diagnostic is no longer true")
        states.append(
            {
                "account": path.parent.name,
                "prior_EXAS_quantity": prior,
                "fills_on_contested_open": fills,
                "scope": "No June 26 execution at the disputed open; this does not certify the daily source or its volume history.",
            }
        )
    write(output / "frozen-account-impact.json", states)
    record = read(PIT / "supplemental/EXAS_zenodo_record.json")
    result = {
        "schema": "equity-exas-gap-review.v1",
        "review_kind": "retrospective source-quality audit; not a new strategy or independent attestation",
        "observed_vendor_rows": len(candidates),
        "observed_range": [min(candidates), max(candidates)],
        "missing_session_candidates_acquired": len(contract["minimum_missing_sessions"]),
        "missing_sessions_without_candidate": [],
        "contract_rows_observed": len(required),
        "fixed_overlap_sessions": len(fixed),
        "fixed_overlap_price_failures": [
            r["date"] for r in fixed if not r["price_within_preexisting_tolerance"]
        ],
        "fixed_overlap_volume_differences": sum(r["volume_difference"] != 0 for r in fixed),
        "additional_left_overlap_sessions": len(extra),
        "additional_left_overlap_price_failures": [
            r["date"] for r in extra if not r["price_within_preexisting_tolerance"]
        ],
        "source_observations": captures,
        "vendor_footer_provider": "S&P Global Market Intelligence (page label; underlying feed independently unverified)",
        "exchange_original_trades_or_auction_records_acquired": False,
        "historical_source_release_or_correction_lineage_verified": False,
        "adjustment_and_session_volume_definitions_verified": False,
        "zenodo_record_created_at": record["created"],
        "zenodo_generation_time_or_partial_session_status": None,
        "publication_time_proves_generation_time": False,
        "data_access": "Public pages; Terms AI-agent exception checked against robots.txt; no login, subscription, paid download or redistribution permission assumed.",
        "price_candidate_admitted": False,
        "admitted_gap_rows": 0,
        "unresolved_admission_sessions": contract["minimum_missing_sessions"],
        "frozen_inputs_or_accounts_modified": False,
        "existing_result_scope": "Frozen numerical results retained as provisional diagnostics; June 26 has no simulated EXAS fills in eight accounts. Input conflict remains unresolved.",
        "merger_cash_spendable_at": None,
        "long_window_admitted": False,
        "new_economic_accounts": 0,
        "external_independent_attestation": False,
        "profit_goal_met": False,
        "remediation": "Obtain source-defined official opening/auction prints, volume-session/correction lineage and corporate-action identity evidence; explain all overlap differences within the unchanged contract before any input amendment. Do not widen tolerance, average sources, replace only the disagreeing day or infer merger cash availability.",
    }
    write(output / "summary.json", result)
    write(
        output / "registration.json",
        {
            "inputs_sha256": {str(p.relative_to(ROOT)): sha(p) for p in inputs},
            "source_sha256": {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))},
            "contract_predates_candidate_observation": True,
            "analysis_registration_is_retrospective": True,
            "external_attestation": False,
        },
    )
    (output / "analysis_source.py").write_bytes(Path(__file__).read_bytes())
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "missing_session_candidates_acquired",
                    "fixed_overlap_price_failures",
                    "fixed_overlap_volume_differences",
                    "price_candidate_admitted",
                    "new_economic_accounts",
                )
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
