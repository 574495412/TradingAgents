"""Reconcile annual signal inputs to their original inline-XBRL submissions.

An independent parser reads tagged USD values, context dates, scale/sign and
entity-wide contexts. It never selects today's latest comparative fact.
"""

import argparse
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from .data import digest, write_json
from .pit_data import Capture


def inline_values(raw, report_end):
    soup = BeautifulSoup(raw, "html.parser")
    contexts, units = {}, {}
    for c in soup.find_all(re.compile(r"(^|:)context$", re.I)):
        if c.find(re.compile(r"(^|:)(explicitmember|typedmember)$", re.I)):
            continue
        start = c.find(re.compile(r"(^|:)startdate$", re.I))
        end = c.find(re.compile(r"(^|:)(enddate|instant)$", re.I))
        if end and end.get_text().strip() == report_end:
            first = start.get_text().strip() if start else None
            duration = (pd.Timestamp(report_end) - pd.Timestamp(first)).days + 1 if first else 0
            if not first or 330 <= duration <= 380:
                contexts[c.get("id")] = first
    for u in soup.find_all(re.compile(r"(^|:)unit$", re.I)):
        measures = u.find_all(re.compile(r"(^|:)measure$", re.I))
        units[u.get("id")] = (
            len(measures) == 1 and measures[0].get_text().strip().split(":")[-1] == "USD"
        )
    result = {}
    for n in soup.find_all(re.compile(r"(^|:)nonfraction$", re.I)):
        name = n.get("name", "")
        if (
            not name.startswith("us-gaap:")
            or n.get("contextref") not in contexts
            or not units.get(n.get("unitref"))
        ):
            continue
        tag = name.split(":")[-1]
        if (tag == "Assets") != (contexts[n.get("contextref")] is None):
            continue
        try:
            text = n.get_text().replace(",", "").replace("\u00a0", "").strip()
            transform = n.get("format", "").split(":")[-1].lower()
            if (transform == "numwordsen" and text.lower() in {"no", "none", "zero"}) or (
                transform in {"numdash", "zerodash"} and text in {"-", "—", "–"}
            ):
                text = "0"
            value = Decimal(text)
        except InvalidOperation:
            continue
        value *= Decimal(10) ** int(n.get("scale", "0"))
        if n.get("sign") == "-":
            value = -value
        result.setdefault(tag, set()).add(value)
    return result


def run(pit):
    annual = json.loads((pit / "annual.json").read_text())
    facts = json.loads((pit / "facts.json").read_text())
    filings = json.loads((pit / "filings.json").read_text())
    lookup = {(f["cik"], f["accessionNumber"]): f for f in filings}
    capture = Capture(pit / "originals")
    comparisons = []
    for record in annual:
        if record["status"] != "available":
            continue
        cik, acc = record["cik"], record["accession"]
        filing = lookup[(cik, acc)]
        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}/{filing['primaryDocument']}"
        file = capture.get(url, acc + ".htm")
        original = inline_values(file.read_text(errors="replace"), record["period_end"])
        values = {}
        for tag in record["tags"]:
            selected = [
                f
                for f in facts
                if f["accession"] == acc
                and f["tag"] == tag
                and f["end"] == record["period_end"]
                and (
                    f["start"] is None
                    if tag == "Assets"
                    else f["start"] is not None
                    and 330 <= (pd.Timestamp(f["end"]) - pd.Timestamp(f["start"])).days + 1 <= 380
                )
            ]
            vals = {Decimal(str(f["value"])) for f in selected}
            if len(vals) != 1 or not vals <= original.get(tag, set()):
                raise ValueError(
                    f"Original annual financial input differs {acc} {tag}: {vals} vs {original.get(tag)}"
                )
            values[tag] = float(next(iter(vals)))
        comparisons.append(
            {
                "cik": cik,
                "accession": acc,
                "period_end": record["period_end"],
                "available_at": record["available_at"],
                "tags_checked": values,
                "original_file": file.name,
                "original_sha256": digest(file),
                "passed": True,
            }
        )
        print("original annual reconciled", acc, len(comparisons), flush=True)
    result = {
        "schema": "equity-original-annual-audit.v1",
        "passed": True,
        "auditor_source_sha256": digest(__file__),
        "annual_signal_versions": len(comparisons),
        "tag_values_checked": sum(len(r["tags_checked"]) for r in comparisons),
        "scope": "All available annual US-GAAP signal inputs, not every companyfact/custom taxonomy/quarterly event or original public dissemination vintage.",
        "facts_sha256": digest(pit / "facts.json"),
        "annual_sha256": digest(pit / "annual.json"),
        "records": comparisons,
        "source_receipts": capture.receipts,
    }
    write_json(pit / "original-annual-audit.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pit", type=Path, required=True)
    run(p.parse_args().pit)
