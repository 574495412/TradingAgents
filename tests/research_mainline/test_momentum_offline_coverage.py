"""Decisive missing-history, identity reuse and future-information counterexamples."""

from datetime import date

import pandas as pd

from research_tools.cross_sectional_momentum.offline_coverage import (
    candidate_link,
    interval_status,
    qualification,
)

PARAMETERS = {
    "trailing_calendar_days": 28,
    "minimum_valid_observed_days": 180,
    "minimum_mean_daily_quote_usdt": 50_000_000,
}


def frame():
    return pd.DataFrame(
        {
            "price_liquidity_feature_usable": True,
            "full_daily_duration": True,
            "volume": 1.0,
            "quote_volume": 60_000_000.0,
        },
        index=pd.date_range("2024-01-01", periods=400, freq="D", tz="UTC"),
    )


def context(start=None, blocked=False):
    return {"feature_start": start, "blocked": blocked}


def test_missing_calendar_day_is_not_replaced_by_28_available_observations():
    data = frame()
    label = date(2025, 1, 7)
    assert qualification(data, label, context(), PARAMETERS)["eligible"]
    data = data.drop(pd.Timestamp("2025-01-01", tz="UTC"))
    result = qualification(data, label, context(), PARAMETERS)
    assert not result["eligible"] and result["quote28"] is None


def test_new_identity_cannot_inherit_old_ticker_age():
    result = qualification(frame(), date(2025, 1, 7), context("2025-01-01T00:00:00Z"), PARAMETERS)
    assert result["observations"] == 7
    assert not result["eligible"]
    assert not qualification(frame(), date(2025, 1, 7), context(blocked=True), PARAMETERS)[
        "eligible"
    ]


def test_future_changes_cannot_fix_formation_window_or_change_past_age():
    data = frame()
    label = date(2025, 1, 7)
    expected = qualification(data, label, context(), PARAMETERS)
    data.loc[data.index >= "2025-01-08", "quote_volume"] = 1e30
    data.loc[data.index >= "2025-01-08", "price_liquidity_feature_usable"] = False
    assert qualification(data, label, context(), PARAMETERS) == expected


def test_partial_and_zero_volume_days_block_full_window_and_initial_age():
    data = frame()
    assert qualification(data, date(2024, 6, 28), context(), PARAMETERS)["observations"] == 179
    data.loc["2025-01-01", "full_daily_duration"] = False
    assert not qualification(data, date(2025, 1, 7), context(), PARAMETERS)["eligible"]
    data.loc["2025-01-01", "full_daily_duration"] = True
    data.loc["2025-01-01", "volume"] = 0
    assert not qualification(data, date(2025, 1, 7), context(), PARAMETERS)["eligible"]


def test_ticker_candidate_never_certifies_identity_or_strips_unit_prefix():
    catalog = {"bonk": {"full_name": "Bonk"}, "pump": {"full_name": "Pump.fun"}}
    assert candidate_link("1000BONKUSDT", catalog)["provider_candidate"] is None
    row = candidate_link("PUMPUSDT", catalog)
    assert row["known_ticker_reuse"] and not row["historical_identity_verified"]
    assert row["provider_candidate"] == "pump"


def test_catalog_interval_is_only_a_necessary_condition_and_checks_new_year_endpoints():
    link = {"provider_candidate": "btc"}
    price = {"btc": (date(2024, 12, 24), date(2026, 1, 1))}
    cap = {"btc": (date(2025, 1, 1), date(2026, 1, 1))}
    result = interval_status(link, date(2025, 1, 7), price, cap)
    assert not result["candidate_catalog_interval_possible"]  # Required Dec22 is outside.
    price["btc"] = (date(2024, 12, 22), date(2026, 1, 1))
    assert interval_status(link, date(2025, 1, 7), price, cap)[
        "candidate_catalog_interval_possible"
    ]
