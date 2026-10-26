"""Read-only structural and temporal admission for the disclosed stock pilot."""

import json
from collections import Counter

import pandas as pd

from .data import digest, validate, write_json


def inspect(pit):
    def read(name):
        return json.loads((pit / name).read_text())

    receipts = read("source-receipts.json")
    for r in receipts:
        p = pit / "raw" / r["file"]
        if p.resolve().parent != (pit / "raw").resolve() or digest(p) != r["sha256"]:
            raise ValueError("Raw SEC evidence content/path changed")
    holdings, seed, ids = (
        read("historical-holdings.json"),
        read("seed.json"),
        read("identities.json"),
    )
    filings, facts, annual, events = (
        read("filings.json"),
        read("facts.json"),
        read("annual.json"),
        read("events.json"),
    )
    if len({(h["accession"], h["row"]) for h in holdings}) != len(holdings):
        raise ValueError("Duplicate holdings key")
    if len(ids) != 10 or {i["cusip"] for i in ids} != {s["cusip"] for s in seed}:
        raise ValueError("Historical universe silently changed")
    lookup = {(r["cik"], r["accessionNumber"]): r for r in filings}
    if len(lookup) != len(filings):
        raise ValueError("Duplicate filing key")
    for fact in facts:
        filing = lookup[(fact["cik"], fact["accession"])]
        if fact["accepted_at"] != filing["accepted_at_utc"] or pd.Timestamp(
            fact["available_at"]
        ) != pd.Timestamp(fact["accepted_at"]) + pd.Timedelta(hours=1):
            raise ValueError("Financial availability not bound to exact accession")
        if fact["end"] > filing["filingDate"] or fact["unit"] != "USD":
            raise ValueError("Fact period/unit violates contract")
    by_issuer = []
    for i in ids:
        local = [a for a in annual if a["cik"] == i["cik"]]
        by_issuer.append(
            {
                "symbol": i["symbol"],
                "cik": i["cik"],
                "annual_versions": len(local),
                "usable_profitability_versions": sum(a["status"] == "available" for a in local),
                "historical_symbols": i["historic_cover_symbols"],
            }
        )
    prices = pit / "prices"
    manifest = json.loads((prices / "manifest.json").read_text())
    cal = pd.read_csv(prices / "US_sessions.csv", index_col="date", parse_dates=["date"])
    if digest(prices / "US_sessions.csv") != manifest["calendar_hashes"]["US"]:
        raise ValueError("Account calendar changed")
    for key, r in manifest["symbols"].items():
        pairs = [("file", "sha256"), ("signal_file", "signal_sha256")]
        pairs += (
            [("vendor_file", "vendor_sha256"), ("metadata_file", "metadata_sha256")]
            if "vendor_file" in r
            else [("archive_file", "archive_sha256")]
        )
        for file_key, hash_key in pairs:
            if digest(prices / r[file_key]) != r[hash_key]:
                raise ValueError(f"Quote source changed {key}")
        f = pd.read_csv(prices / r["file"], index_col="date", parse_dates=["date"])
        validate(f, cal.loc[: f.index[-1]] if r.get("coverage_complete") is False else cal)
    if set(manifest["symbols"]) != {i["symbol"] for i in ids}:
        raise ValueError("Selected member missing from account quotes")
    periods = sorted({h["report_date"] for h in holdings})
    first = {h["cusip"] for h in holdings if h["report_date"] == periods[0]}
    last = {h["cusip"] for h in holdings if h["report_date"] == periods[-1]}
    result = {
        "schema": "equity-pit-quality.v1",
        "research_input_checks_passed": True,
        "raw_sec_files_verified": len(receipts),
        "filing_metadata_rows": len(filings),
        "historical_disclosure_rows": len(holdings),
        "disclosure_periods": periods,
        "distinct_historical_cusips": len({h["cusip"] for h in holdings}),
        "first_snapshot_not_in_last": len(first - last),
        "disappeared_is_not_delisting_evidence": True,
        "seed_members": len(seed),
        "quote_members": len(manifest["symbols"]),
        "full_window_available": manifest["full_window_available"],
        "full_window_admitted": False,
        "coverage_gap": manifest["coverage_gap"],
        "bounded_diagnostic_last_session": min(
            pd.read_csv(prices / r["file"], index_col="date").index.max()
            for r in manifest["symbols"].values()
        ),
        "company_fact_rows": len(facts),
        "annual_versions": len(annual),
        "annual_status": dict(Counter(a["status"] for a in annual)),
        "by_issuer": by_issuer,
        "filing_events": len(events),
        "item_202_filing_events": sum("2.02" in e.get("items", "") for e in events),
        "known_first_earnings_release_time": False,
        "earnings_expectation_history": False,
        "point_in_time_certified": False,
        "market_wide_historical_universe_complete": False,
        "delisted_price_coverage_certified": False,
        "corporate_actions_certified": False,
        "limitations": [
            "Pre-window disclosed ARK top10 is a narrow historical institution-selected cohort. All 16 disclosure snapshots retained; not a market-wide survivor-free security master. First-snapshot exits are not automatically delistings.",
            "SEC API is a current extraction with original accession and acceptance timestamps. Later filing versions/comparatives cannot leak backwards. Every available annual signal input is independently checked against its original inline XBRL before execution; quarterly facts, unavailable/custom tags and exact historical public dissemination vintages remain uncertified.",
            "Gross-profitability tags are unavailable for some issuers/versions. No zero fill, IFRS substitution or today's Yahoo fundamental fields. Latest unavailable annual version blocks that observation; unaffected securities remain eligible.",
            "8-K acceptance is a known filing event, not proof of first earnings release. No historical consensus expectations or certified release clock, so earnings-surprise/PEAD strategy is gated.",
            "Yahoo's current symbol is a quote lookup only. Historical cover tags and permanent issuer CIK retained; SQ/XYZ is one issuer. Full security-class/CUSIP/corporate-action history is not independently certified.",
            "Warmup split-adjusted prices are used only in return ratios. Account prices after 2022-12-30 must have no reported split through capture. Missing delisted bars or unhandled splits block the entire fixed cohort. Public action completeness and prices are not broker-certified.",
            "EXAS was acquired in March 2026 and disappeared from Yahoo. CC0 academic snapshot recovers every member through 2024-06-26 only. A separately frozen common-coverage pilot may run; the original 2023-2026 protocol remains blocked. No return attributed to missing periods or merger settlement.",
            "Same v2 cash/fees/settlement/participation assumptions apply. Three-session lock, daily-low drawdown and fixed slippage are conservative proxies, not synchronized intraday or market-depth proof. All evaluated history remains diagnostic.",
        ],
        "input_hashes": {
            name: digest(pit / name)
            for name in [
                "protocol.json",
                "seed.json",
                "identities.json",
                "historical-holdings.json",
                "filings.json",
                "facts.json",
                "annual.json",
                "events.json",
                "source-receipts.json",
            ]
        },
        "price_manifest_sha256": digest(prices / "manifest.json"),
    }
    write_json(pit / "data-quality.json", result)
    return result
