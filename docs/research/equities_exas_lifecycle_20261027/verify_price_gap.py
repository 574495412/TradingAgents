"""Offline consistency checks; this is our code, not independent attestation."""

import csv
import hashlib
import json
import re
from datetime import datetime
from decimal import Decimal
from html import unescape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "results/research/equity_exas_gap_20261027_v001"
DATA = ROOT / "data/research/equity_exas_gap_20261027_v001"
PIT = ROOT / "data/research/equity_pit_20261026_v001/supplemental"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


registration = read(RUN / "registration.json")
for key in ("inputs_sha256", "source_sha256"):
    assert all(sha(ROOT / path) == digest for path, digest in registration[key].items())
candidate = {}
for page in (4, 5):
    body = (DATA / f"stockanalysis-page{page}-table.html").read_text()
    parsed = [
        [
            unescape(re.sub("<[^>]*>", "", cell)).strip()
            for cell in re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", row, re.S)
        ]
        for row in re.findall(r"<tr\b[^>]*>(.*?)</tr>", body, re.S)
    ]
    saved = read(DATA / f"stockanalysis-page{page}-visible-rows.json")["rows"][0]
    assert parsed == saved and len(parsed) == 101
    for row in parsed[1:]:
        day = datetime.strptime(row[0], "%b %d, %Y").date().isoformat()
        assert day not in candidate
        candidate[day] = [Decimal(v.replace(",", "")) for v in row[1:6]] + [
            Decimal(row[7].replace(",", ""))
        ]

contract_path = ROOT / "docs/research/equities_exas_lifecycle_20261027/missing-input-contract.json"
contract = read(contract_path)["prices"]
assert len(candidate) == 200
assert all(d in candidate for d in contract["minimum_missing_sessions"])
overlap = read(RUN / "overlap.json")
checked = 0
for side, filename in (("left", "EXAS_zenodo.csv"), ("right", "EXAS_chartexchange_observed.csv")):
    with (PIT / filename).open(newline="") as handle:
        reference = {}
        for row in csv.DictReader(handle):
            normalized = {k.lower().replace(" ", "_"): v for k, v in row.items()}
            reference[normalized["date"]] = normalized
    for record in overlap[side + "_fixed"]:
        day = record["date"]
        raw = reference[day]
        for i, field in enumerate(("open", "high", "low", "close")):
            assert candidate[day][i] - Decimal(raw[field]) == Decimal(
                record["price_differences_candidate_minus_reference"][field]
            )
        assert (
            candidate[day][-1] - Decimal(raw["volume"].replace(",", ""))
            == record["volume_difference"]
        )
        assert record["price_within_preexisting_tolerance"] == all(
            abs(Decimal(v)) <= Decimal(".01")
            for v in record["price_differences_candidate_minus_reference"].values()
        )
        checked += 1

summary = read(RUN / "summary.json")
assert checked == 20 and summary["fixed_overlap_price_failures"] == ["2024-06-26"]
assert summary["fixed_overlap_volume_differences"] == 20
assert not summary["price_candidate_admitted"] and summary["admitted_gap_rows"] == 0
assert not summary["profit_goal_met"] and not summary["external_independent_attestation"]
impact = read(RUN / "frozen-account-impact.json")
assert len(impact) == 8 and all(not r["fills_on_contested_open"] for r in impact)
assert [r["prior_EXAS_quantity"] for r in impact] == [12, 12, 0, 0, 0, 0, 0, 0]
assert sha(RUN / "analysis_source.py") == next(iter(registration["source_sha256"].values()))
receipt = {
    "internal_source_consistency_checks_passed": True,
    "original_table_rows_rebuilt": len(candidate),
    "fixed_overlap_sessions_recomputed": checked,
    "contract_sha256": sha(contract_path),
    "frozen_account_states_unchanged": 8,
    "candidate_data_admitted": False,
    "external_attestation": False,
    "scope": "Two parsers and Decimal checks are internal consistency only; source conflicts remain open.",
}
(RUN / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt))
