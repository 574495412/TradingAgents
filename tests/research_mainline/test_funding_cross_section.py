"""Exact event cadence, known information, tied sets and predeclared cost coverage."""

import pandas as pd
import pytest

from factors.crypto.funding_cross_section import accrued7, book


def fixture():
    rows = {
        s: pd.Series(
            {
                "observations": 400,
                "quote28": 1e8,
                "max28": 0.1,
                "momentum28": i * 0.01,
                "beta60": 1.0,
            }
        )
        for i, s in enumerate(["BTCUSDT"] + [f"X{i:02}USDT" for i in range(12)])
    }
    p = {
        "minimum_listing_observations": 180,
        "minimum_daily_quote_volume": 5e7,
        "maximum_eligible": 30,
        "minimum_eligible": 10,
        "long_names": 4,
        "short_names": 4,
    }
    return rows, p


def test_accrual_uses_actual_cadence_and_excludes_current_or_future_settlement():
    rates = {
        "A": pd.DataFrame(
            {"funding_rate": [0.001, 0.002, 0.9]},
            index=pd.to_datetime(
                ["2024-12-25T00:00:00.010Z", "2024-12-31T20:00Z", "2025-01-01T00:00Z"],
                format="mixed",
                utc=True,
            ),
        )
    }
    assert accrued7(rates, "2025-01-01T00:00Z")["A"] == 0.003


def test_zero_rate_is_known_but_missing_calendar_is_not_zero():
    f = pd.DataFrame({"funding_rate": [0.0]}, index=pd.to_datetime(["2024-12-31"], utc=True))
    assert accrued7({"A": f}, "2025-01-01T00:00Z")["A"] == 0
    with pytest.raises(ValueError, match="Missing"):
        accrued7({"A": f.iloc[:0]}, "2025-01-01T00:00Z")


def test_equal_rate_ties_leave_cash_not_overwritten_short_portfolio():
    rows, p = fixture()
    w, r = book(rows, pd.Series(0.0021, index=rows), p, "funding7_raw")
    assert w.abs().sum() == 0 and r["status"] == "no_funding_dispersion"


def test_pure_carry_sign_disjointness_and_cost_floor_are_separate():
    rows, p = fixture()
    rates = pd.Series({s: (i - 6) * 0.0001 for i, s in enumerate(rows)})
    raw, r = book(rows, rates, p, "funding7_raw")
    guard, g = book(rows, rates, p, "funding7_cost_floor")
    assert set(r["long"]).isdisjoint(r["short"]) and raw.sum() == pytest.approx(0)
    assert max(rates[r["long"]]) <= min(rates[r["short"]])
    assert raw.abs().sum() == 1 and guard.abs().sum() == 0
    assert g["registered_unit_cost_floor"] == 0.00375
    strong, _ = book(rows, rates * 100, p, "funding7_cost_floor")
    assert strong.abs().sum() == 1
