import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from factors.equity.disclosed_stocks import targets
from tradingagents.research.equities.audit_v002 import expected_fee
from tradingagents.research.equities.fees import charge
from tradingagents.research.equities.pit_data import (
    annual_profitability,
    asof_annual,
    filing_rows,
    normalized_facts,
    seed_cohort,
)
from tradingagents.research.equities.pit_originals import inline_values
from tradingagents.research.equities.pit_universe import disclosed_at, membership_intervals


def annual(cik="1", acc="a", available="2023-03-01T22:00:00+00:00", end="2022-12-31", score=0.2):
    return {
        "cik": cik,
        "accession": acc,
        "available_at": available,
        "period_end": end,
        "gross_profitability": score,
        "status": "available",
    }


def test_filing_column_misalignment_rejected():
    with pytest.raises(ValueError):
        filing_rows({"a": [1], "b": [2, 3]})


def test_common_share_fees_are_reviewed_us_only():
    stock = {"minimum_fee": 1, "commission_bps": 1, "symbols": [{"instrument": "equity"}]}
    etf = {**stock, "symbols": [{"instrument": "ETF"}]}
    actual = charge("US", "2024-06-03", "SELL", 100, 55, stock)
    assert actual == charge("US", "2024-06-03", "SELL", 100, 55, etf)
    assert actual == {
        k: float(v) for k, v in expected_fee("US", "2024-06-03", "SELL", 100, 55, stock).items()
    }
    with pytest.raises(ValueError):
        charge("HK", "2024-06-03", "SELL", 100, 55, stock)
    with pytest.raises(ValueError):
        expected_fee("HK", "2024-06-03", "SELL", 100, 55, stock)


def test_asof_holdings_use_disclosure_clock_and_retain_exits():
    filings = [
        {
            "form": "13F-HR",
            "accessionNumber": "a",
            "reportDate": "2022-09-30",
            "accepted_at_utc": "2022-11-01T20:00:00Z",
        },
        {
            "form": "13F-HR",
            "accessionNumber": "b",
            "reportDate": "2022-12-31",
            "accepted_at_utc": "2023-02-01T20:00:00Z",
        },
    ]
    rows = [{"accession": "a", "cusip": "gone"}, {"accession": "b", "cusip": "new"}]
    assert disclosed_at(rows, filings, "2022-10-01T21:00:00Z") == []
    assert disclosed_at(rows, filings, "2023-02-01T19:59:59Z")[0]["cusip"] == "gone"
    assert disclosed_at(rows, filings, "2023-02-01T20:00:00Z")[0]["cusip"] == "new"
    intervals = membership_intervals(rows, filings)
    assert (
        intervals[0]["cusip"] == "gone"
        and intervals[0]["information_valid_to"] == "2023-02-01T20:00:00Z"
    )
    amended = [{**filings[-1], "form": "13F-HR/A"}]
    with pytest.raises(ValueError):
        disclosed_at(rows, amended, "2023-02-02T20:00:00Z")


def test_original_xbrl_context_scale_sign_and_explicit_zero_transform():
    raw = """<xbrli:context id="annual"><xbrli:period><xbrli:startDate>2022-01-01</xbrli:startDate><xbrli:endDate>2022-12-31</xbrli:endDate></xbrli:period></xbrli:context>
    <xbrli:context id="instant"><xbrli:period><xbrli:instant>2022-12-31</xbrli:instant></xbrli:period></xbrli:context>
    <xbrli:unit id="usd"><xbrli:measure>iso4217:USD</xbrli:measure></xbrli:unit>
    <ix:nonFraction name="us-gaap:GrossProfit" contextRef="annual" unitRef="usd" scale="3" sign="-">2</ix:nonFraction>
    <ix:nonFraction name="us-gaap:Assets" contextRef="instant" unitRef="usd" scale="3">10</ix:nonFraction>
    <ix:nonFraction name="us-gaap:CostOfRevenue" contextRef="annual" unitRef="usd" format="ixt-sec:numwordsen">no</ix:nonFraction>
    <ix:nonFraction name="us-gaap:Revenues" contextRef="annual" unitRef="usd">unknown</ix:nonFraction>"""
    values = inline_values(raw, "2022-12-31")
    assert values["GrossProfit"] == {-2000} and values["Assets"] == {10000}
    assert values["CostOfRevenue"] == {0} and "Revenues" not in values
    assert inline_values(raw, "2021-12-31") == {}


def test_exact_timestamp_lag_and_amendment():
    rows = [annual(), annual(acc="amend", available="2023-04-01T22:00:00Z", score=0.8)]
    assert asof_annual(rows, "1", "2023-03-01T21:59:59Z") is None
    assert asof_annual(rows, "1", "2023-03-01T22:00:00Z")["accession"] == "a"
    assert asof_annual(rows, "1", "2023-04-02T21:00:00Z")["accession"] == "amend"
    assert asof_annual(rows, "1", "2025-01-01T21:00:00Z") is None


def test_old_amendment_cannot_replace_newer_year_or_fill_newer_missing():
    rows = [
        annual(end="2021-12-31", acc="old-amend", available="2023-04-01T22:00:00Z", score=9),
        annual(),
    ]
    assert asof_annual(rows, "1", "2023-04-02T21:00:00Z")["accession"] == "a"
    rows.append(
        {
            **annual(acc="new-amend", available="2023-04-02T22:00:00Z"),
            "status": "missing_or_ambiguous",
        }
    )
    assert asof_annual(rows, "1", "2023-04-03T21:00:00Z") is None


def test_same_accession_annual_duration_and_period_only():
    base = {
        "cik": "1",
        "accession": "a",
        "form": "10-K",
        "report_date": "2022-12-31",
        "end": "2022-12-31",
        "start": None,
        "available_at": "2023-03-01T22:00:00Z",
    }
    facts = [
        {**base, "tag": "Assets", "value": 100},
        {**base, "tag": "GrossProfit", "value": 20, "start": "2022-01-01"},
        {**base, "tag": "GrossProfit", "value": 99, "start": "2022-10-01"},
        {**base, "tag": "GrossProfit", "value": 888, "end": "2021-12-31", "start": "2021-01-01"},
    ]
    assert annual_profitability(facts)[0]["gross_profitability"] == 0.2
    # A later accession's assets cannot supply missing assets in the original.
    facts[0]["accession"] = "b"
    assert all(r["gross_profitability"] is None for r in annual_profitability(facts))


def test_ambiguous_tag_and_quarter_are_not_latest_row_wins():
    b = {
        "cik": "1",
        "accession": "a",
        "form": "10-K",
        "report_date": "2022-12-31",
        "end": "2022-12-31",
        "available_at": "2023-03-01T22:00:00Z",
    }
    rows = [
        {**b, "tag": "Assets", "value": 100, "start": None},
        {**b, "tag": "GrossProfit", "value": 20, "start": "2022-01-01"},
        {**b, "tag": "GrossProfit", "value": 30, "start": "2022-01-01"},
    ]
    assert annual_profitability(rows)[0]["gross_profitability"] is None


def test_fact_joins_own_accession_not_filed_date_or_later_vintage():
    filing = {
        "cik": "1",
        "accessionNumber": "a",
        "form": "10-K",
        "reportDate": "2022-12-31",
        "filingDate": "2023-03-01",
        "accepted_at_utc": "2023-03-01T20:30:00Z",
    }
    facts = {
        "facts": {
            "us-gaap": {
                "Assets": {
                    "units": {
                        "USD": [
                            {"accn": "a", "end": "2022-12-31", "filed": "2023-03-01", "val": 100},
                            {
                                "accn": "later",
                                "end": "2022-12-31",
                                "filed": "2024-03-01",
                                "val": 999,
                            },
                        ]
                    }
                }
            }
        }
    }
    normalized = normalized_facts(facts, [filing])
    assert len(normalized) == 1 and normalized[0]["value"] == 100
    assert pd.Timestamp(normalized[0]["available_at"]) == pd.Timestamp("2023-03-01T21:30:00Z")


def fixture():
    index = pd.bdate_range("2021-01-01", "2024-12-31")
    cal = pd.DataFrame(
        {
            "open": pd.to_datetime(index, utc=True) + pd.Timedelta(hours=14),
            "close": pd.to_datetime(index, utc=True) + pd.Timedelta(hours=21),
        },
        index=index,
    )
    ids = [
        {"symbol": str(i), "cik": str(i), "available_at": "2022-10-17T17:20:29Z"} for i in range(10)
    ]
    frames = {
        r["symbol"]: pd.DataFrame(
            {
                "close": np.exp(np.arange(len(index)) * (0.001 + int(r["symbol"]) * 0.00001)),
                "dividend": 0.0,
            },
            index=index,
        )
        for r in ids
    }
    plan = json.loads(
        (
            Path(__file__).parents[1] / "tradingagents/research/equities/stock_protocol.json"
        ).read_text()
    )
    records = [annual(cik=str(i), score=0.01 * i) for i in range(10)]
    return frames, cal, ids, records, plan


def test_true_month_end_and_future_data_invariance():
    frames, cal, ids, records, plan = fixture()
    books, _, evidence = targets(frames, cal, ids, records, plan)
    # First filing was mid-month: no pre-disclosure/premature monthly trades.
    assert all(b.loc[:"2022-10-28"].sum().sum() == 0 for b in books.values())
    assert books["disclosed_equal"].loc["2022-10-31"].sum() == pytest.approx(0.8)
    changed, updated = copy.deepcopy(frames), copy.deepcopy(records)
    for f in changed.values():
        f.loc["2024-01-01":, "close"] *= 17
        f.loc["2024-01-01":, "dividend"] = 100
    updated.append(
        annual(cik="1", acc="future", end="2023-12-31", available="2024-04-01T20:00:00Z", score=100)
    )
    other, _, _ = targets(changed, cal, ids, updated, plan)
    for key in books:
        pd.testing.assert_frame_equal(books[key].loc[:"2023-12-31"], other[key].loc[:"2023-12-31"])
        assert (books[key].sum(axis=1) <= 0.800000001).all()
    assert all(
        pd.Timestamp(e["available_at"]) <= pd.Timestamp(e["decision_close"]) for e in evidence
    )
    # A truncated input prefix must not turn mid-month into month-end.
    prefix = {s: f.loc[:"2023-03-15"] for s, f in frames.items()}
    short, _, _ = targets(prefix, cal, ids, records, plan)
    for key in books:
        pd.testing.assert_frame_equal(short[key], books[key].loc[:"2023-03-15"])


def test_missing_selected_member_cannot_be_dropped():
    frames, cal, ids, records, plan = fixture()
    frames.pop("0")
    with pytest.raises(ValueError):
        targets(frames, cal, ids, records, plan)


def test_seed_is_before_window_and_not_current_survivors():
    plan = {"universe": {"seed_accession": "a", "known_before": "2023-01-01", "size": 2}}
    row = {
        "accession": "a",
        "available_at": "2022-10-01T12:00:00Z",
        "put_call": "",
        "share_type": "SH",
        "class": "Common Stock",
        "shares": 1,
    }
    rows = [
        {**row, "cusip": "gone", "stated_value": 5},
        {**row, "cusip": "live", "stated_value": 2},
        {**row, "accession": "b", "cusip": "later", "stated_value": 10000},
    ]
    assert [r["cusip"] for r in seed_cohort(rows, plan)] == ["gone", "live"]
    rows[0]["available_at"] = "2023-01-01T12:00:00Z"
    with pytest.raises(ValueError):
        seed_cohort(rows, plan)
