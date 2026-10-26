"""Immutable public SEC disclosures and accession-level as-of financial facts.

SEC current tickers are lookup metadata, never the historical selection universe.
The pilot seed is fixed by a pre-2023 13F, including every selected identifier.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests

from .data import digest, write_json

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
UA = "CZ-Agent/EquityResearch (public financial research)"
# Identifier resolution, not selection. Historical cover-page symbols are
# captured separately and today's symbol is never used to pick a survivor.
IDENTITIES = {
    "88160R101": ("0001318605", "TSLA"),
    "98980L101": ("0001585521", "ZM"),
    "77543R102": ("0001428439", "ROKU"),
    "90364P105": ("0001734722", "PATH"),
    "30063P105": ("0001124140", "EXAS"),
    "45826J105": ("0001652130", "NTLA"),
    "H17182108": ("0001674416", "CRSP"),
    "87918A105": ("0001477449", "TDOC"),
    "852234103": ("0001512673", "XYZ"),
    "19260Q107": ("0001679788", "COIN"),
}


class Capture:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers["User-Agent"] = UA
        self.receipts = []

    def get(self, url, name):
        path = self.folder / name
        receipt = path.with_suffix(path.suffix + ".receipt.json")
        if path.is_file() and receipt.is_file():
            item = json.loads(receipt.read_text())
            if item["url"] != url or item["sha256"] != digest(path):
                raise ValueError("Cached source identity changed")
        else:
            if path.exists() or receipt.exists():
                raise ValueError("Incomplete immutable capture; inspect instead of overwrite")
            time.sleep(0.12)  # well below SEC's ten requests/second maximum
            response = self.session.get(url, timeout=60)
            response.raise_for_status()
            path.write_bytes(response.content)
            item = {
                "url": url,
                "file": name,
                "sha256": digest(path),
                "bytes": len(response.content),
                "captured_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "status": response.status_code,
            }
            write_json(receipt, item)
        self.receipts.append(item)
        return path

    def json(self, url, name):
        return json.loads(self.get(url, name).read_text())


def filing_rows(columns):
    lengths = {len(v) for v in columns.values()}
    if len(lengths) != 1:
        raise ValueError("SEC columnar filing lengths differ")
    return [
        dict(zip(columns, values, strict=True)) for values in zip(*columns.values(), strict=True)
    ]


def submissions(capture, cik, start="2020-01-01"):
    base = "https://data.sec.gov/submissions/"
    obj = capture.json(base + f"CIK{cik}.json", f"CIK{cik}.json")
    rows = filing_rows(obj["filings"]["recent"])
    for archive in obj["filings"]["files"]:
        if archive["filingTo"] >= start:
            cols = capture.json(base + archive["name"], archive["name"])
            rows.extend(filing_rows(cols))
    seen, normalized = {}, []
    for row in rows:
        if row["filingDate"] < start:
            continue
        acc = row["accessionNumber"]
        if acc in seen:
            if row != seen[acc]:
                raise ValueError("Conflicting SEC accession metadata")
            continue
        seen[acc] = row
        stamp = pd.Timestamp(row["acceptanceDateTime"])
        if stamp.tz is None:
            raise ValueError("Missing authoritative acceptance timezone")
        row = {**row, "cik": cik, "accepted_at_utc": stamp.tz_convert("UTC").isoformat()}
        normalized.append(row)
    return obj, sorted(normalized, key=lambda x: (x["accepted_at_utc"], x["accessionNumber"]))


def parse_13f(raw, filing):
    tree = ET.fromstring(raw)
    for node in tree.iter():
        node.tag = node.tag.split("}")[-1]
    result = []
    for i, node in enumerate(tree.findall("infoTable")):

        def value(tag, node=node):
            return node.findtext(tag, default="").strip()

        result.append(
            {
                "accession": filing["accessionNumber"],
                "report_date": filing["reportDate"],
                "available_at": filing["accepted_at_utc"],
                "row": i,
                "issuer": value("nameOfIssuer"),
                "class": value("titleOfClass"),
                "cusip": value("cusip"),
                "stated_value": float(value("value")),
                "value_unit": "thousands_USD" if filing["filingDate"] < "2023-01-03" else "USD",
                "shares": float(value("shrsOrPrnAmt/sshPrnamt")),
                "share_type": value("shrsOrPrnAmt/sshPrnamtType"),
                "put_call": value("putCall"),
                "investment_discretion": value("investmentDiscretion"),
            }
        )
    if not result:
        raise ValueError("No 13F information-table rows")
    return result


def seed_cohort(rows, plan):
    setting = plan["universe"]
    seed = [x for x in rows if x["accession"] == setting["seed_accession"]]
    if not seed or max(x["available_at"][:10] for x in seed) >= setting["known_before"]:
        raise ValueError("Universe was not known before the evaluation window")
    common = [
        x
        for x in seed
        if not x["put_call"] and x["share_type"] == "SH" and x["class"].lower() == "common stock"
    ]
    grouped = {}
    for x in common:
        key = x["cusip"]
        if key not in grouped:
            grouped[key] = {**x, "stated_value": 0, "shares": 0}
        grouped[key]["stated_value"] += x["stated_value"]
        grouped[key]["shares"] += x["shares"]
    selected = sorted(grouped.values(), key=lambda x: (-x["stated_value"], x["cusip"]))
    return [{**x, "seed_rank": i + 1} for i, x in enumerate(selected[: setting["size"]])]


def normalized_facts(companyfacts, filings):
    by_acc = {x["accessionNumber"]: x for x in filings}
    tags = {
        "Assets",
        "GrossProfit",
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "CostOfRevenue",
        "CostOfGoodsAndServicesSold",
        "NetIncomeLoss",
    }
    result, seen = [], set()
    for tag, concept in companyfacts.get("facts", {}).get("us-gaap", {}).items():
        if tag not in tags:
            continue
        for unit, facts in concept["units"].items():
            if unit != "USD":
                continue
            for fact in facts:
                filing = by_acc.get(fact["accn"])
                if not filing or filing["form"] not in {"10-K", "10-K/A", "10-Q", "10-Q/A"}:
                    continue
                item = {
                    "cik": filing["cik"],
                    "accession": fact["accn"],
                    "tag": tag,
                    "unit": unit,
                    "start": fact.get("start"),
                    "end": fact["end"],
                    "value": fact["val"],
                    "form": filing["form"],
                    "report_date": filing["reportDate"],
                    "filed": fact["filed"],
                    "accepted_at": filing["accepted_at_utc"],
                    "available_at": (
                        pd.Timestamp(filing["accepted_at_utc"]) + pd.Timedelta(hours=1)
                    ).isoformat(),
                }
                if item["end"] > filing["filingDate"]:
                    raise ValueError("Future reporting period in filing")
                key = json.dumps(item, sort_keys=True)
                if key not in seen:
                    result.append(item)
                    seen.add(key)
    return result


def annual_profitability(facts):
    """One exact-accession annual record; never borrow later comparative values."""
    by_acc, records = {}, []
    for fact in facts:
        if fact["form"] not in {"10-K", "10-K/A"} or fact["end"] != fact["report_date"]:
            continue
        duration = (
            (pd.Timestamp(fact["end"]) - pd.Timestamp(fact["start"])).days + 1
            if fact["start"]
            else 0
        )
        if (fact["tag"] == "Assets" and not fact["start"]) or 330 <= duration <= 380:
            by_acc.setdefault((fact["cik"], fact["accession"]), []).append(fact)
    for (cik, acc), rows in by_acc.items():
        values = {}
        for r in rows:
            values.setdefault(r["tag"], set()).add(r["value"])
        # Ambiguous same-tag contexts cannot silently pick the last API row.
        values = {k: next(iter(v)) for k, v in values.items() if len(v) == 1}
        gp, used = values.get("GrossProfit"), ["GrossProfit"]
        if gp is None:
            revenue = next(
                (
                    k
                    for k in [
                        "RevenueFromContractWithCustomerExcludingAssessedTax",
                        "Revenues",
                        "SalesRevenueNet",
                    ]
                    if k in values
                ),
                None,
            )
            cost = next(
                (k for k in ["CostOfRevenue", "CostOfGoodsAndServicesSold"] if k in values), None
            )
            if revenue and cost:
                gp, used = values[revenue] - values[cost], [revenue, cost]
        assets = values.get("Assets")
        records.append(
            {
                "cik": cik,
                "accession": acc,
                "available_at": rows[0]["available_at"],
                "period_end": rows[0]["report_date"],
                "gross_profit": gp,
                "assets": assets,
                "gross_profitability": gp / assets
                if gp is not None and assets and assets > 0
                else None,
                "tags": used + ["Assets"],
                "status": "available"
                if gp is not None and assets and assets > 0
                else "missing_or_ambiguous",
            }
        )
    return sorted(records, key=lambda x: (x["available_at"], x["cik"], x["accession"]))


def asof_annual(records, cik, decision_close):
    clock = pd.Timestamp(decision_close)
    eligible = [
        r
        for r in records
        if r["cik"] == cik
        and pd.Timestamp(r["available_at"]) <= clock
        and 0 <= (clock.tz_localize(None).normalize() - pd.Timestamp(r["period_end"])).days <= 550
    ]
    if not eligible:
        return None
    # Latest economic report period first; an amendment of an older year must
    # not displace a newer year. Latest version only within that period.
    latest = max(eligible, key=lambda r: (r["period_end"], r["available_at"], r["accession"]))
    return latest if latest["status"] == "available" else None


def capture(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plan = json.loads((HERE / "stock_protocol.json").read_text())
    registered = output / "protocol.json"
    if registered.exists() and json.loads(registered.read_text()) != plan:
        raise ValueError("Capture protocol changed; choose a new version")
    if not registered.exists():
        write_json(registered, plan)
        write_json(
            output / "registration.json",
            {
                "registered_at": pd.Timestamp.now(tz="UTC").isoformat(),
                "protocol_sha256": digest(registered),
                "price_returns_viewed": False,
            },
        )
    src = Capture(output / "raw")
    _, manager = submissions(src, plan["universe"]["manager_cik"], "2022-01-01")
    holdings, filing_index = [], []
    for filing in manager:
        if filing["form"] not in {"13F-HR", "13F-HR/A"} or not (
            "2022-09-30" <= filing["reportDate"] <= "2026-06-30"
        ):
            continue
        acc = filing["accessionNumber"]
        base = "https://www.sec.gov/Archives/edgar/data/1697748/" + acc.replace("-", "") + "/"
        index = src.json(base + "index.json", acc + "-index.json")
        candidates = [
            x["name"]
            for x in index["directory"]["item"]
            if x["name"].endswith(".xml") and x["name"] != "primary_doc.xml"
        ]
        if len(candidates) != 1:
            raise ValueError("Ambiguous information table")
        raw = src.get(base + candidates[0], acc + "-holdings.xml")
        holdings.extend(parse_13f(raw.read_bytes(), filing))
        filing_index.append(filing)
    seed = seed_cohort(holdings, plan)
    write_json(output / "historical-holdings.json", holdings)
    write_json(output / "manager-filings.json", filing_index)
    write_json(output / "seed.json", seed)
    facts, events, identities, issuer_filings = [], [], [], []
    for item in seed:
        if item["cusip"] not in IDENTITIES:
            raise ValueError("Selected security identity unresolved; no survivor substitution")
        cik, quote = IDENTITIES[item["cusip"]]
        metadata, filings = submissions(src, cik)
        issuer_filings.extend(filings)
        cf = src.json(
            f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json", f"CIK{cik}-facts.json"
        )
        if str(cf["cik"]).zfill(10) != cik:
            raise ValueError("Issuer CIK mismatch")
        facts.extend(normalized_facts(cf, filings))
        events.extend(
            [
                x
                for x in filings
                if x["form"] in {"8-K", "8-K/A", "6-K", "25", "25-NSE", "15-12B", "15-12G"}
            ]
        )
        annual = [
            x for x in filings if x["form"] in {"10-K", "20-F"} and x["filingDate"] < "2023-01-01"
        ]
        if not annual:
            raise ValueError("No pre-window issuer identity filing")
        historic = max(annual, key=lambda x: x["filingDate"])
        acc = historic["accessionNumber"]
        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}/{historic['primaryDocument']}"
        cover = src.get(url, f"{cik}-{acc}-identity.htm")
        # Save actual cover tags; do not treat today's submissions ticker as PIT.
        text = cover.read_text(errors="replace")
        symbols = re.findall(
            r'<ix:nonNumeric\b[^>]*name=["\x27]dei:TradingSymbol["\x27][^>]*>(.*?)</ix:nonNumeric>',
            text,
            re.I | re.S,
        )
        symbols = sorted({re.sub("<[^>]+>", "", s).strip() for s in symbols})
        if not symbols:
            raise ValueError("Historical trading symbol missing; identity needs review")
        identities.append(
            {
                **item,
                "cik": cik,
                "symbol": f"{quote}.US",
                "yahoo": quote,
                "lot_size": 1,
                "instrument": "equity",
                "issuer_name": metadata["name"],
                "historic_cover_symbols": symbols,
                "historic_identity_accession": acc,
                "identity_source": url,
                "quote_symbol_is_current_lookup": True,
            }
        )
        print("captured", cik, quote, "historical symbols", symbols, flush=True)
    annual = annual_profitability(facts)
    for file, obj in [
        ("identities.json", identities),
        ("filings.json", issuer_filings),
        ("facts.json", facts),
        ("events.json", events),
        ("annual.json", annual),
    ]:
        write_json(output / file, obj)
    write_json(output / "source-receipts.json", src.receipts)
    print(
        "PIT capture",
        len(holdings),
        "holdings",
        len(facts),
        "facts",
        len(events),
        "events",
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    capture(args.output)
