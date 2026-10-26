"""As-of disclosed membership, distinct from exchange listing/tradability."""

import pandas as pd


def disclosed_at(holdings, filings, clock):
    stamp = pd.Timestamp(clock)
    if stamp.tz is None:
        raise ValueError("Disclosed-universe clock needs an explicit timezone")
    visible = [f for f in filings if pd.Timestamp(f["accepted_at_utc"]) <= stamp]
    if not visible:
        return []
    period = max(f["reportDate"] for f in visible)
    versions = [f for f in visible if f["reportDate"] == period]
    latest = max(versions, key=lambda f: (f["accepted_at_utc"], f["accessionNumber"]))
    if latest["form"] != "13F-HR":
        raise ValueError(
            "13F amendment semantics need cover-page review before membership reconstruction"
        )
    return [r for r in holdings if r["accession"] == latest["accessionNumber"]]


def membership_intervals(holdings, filings):
    intervals, previous, opened = [], set(), {}
    for filing in sorted(filings, key=lambda f: (f["accepted_at_utc"], f["accessionNumber"])):
        stamp = filing["accepted_at_utc"]
        current = {r["cusip"] for r in disclosed_at(holdings, filings, stamp)}
        for cusip in previous - current:
            intervals.append({**opened.pop(cusip), "information_valid_to": stamp})
        for cusip in current - previous:
            opened[cusip] = {
                "cusip": cusip,
                "information_valid_from": stamp,
                "entry_accession": filing["accessionNumber"],
                "role": "Disclosed holdings membership only; not listing date, live holding or guaranteed tradability",
            }
        previous = current
    intervals.extend({**r, "information_valid_to": None} for r in opened.values())
    return sorted(intervals, key=lambda r: (r["cusip"], r["information_valid_from"]))
