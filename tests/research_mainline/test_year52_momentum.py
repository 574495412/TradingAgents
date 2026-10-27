"""Calendar, funding-denominator and source-availability counterexamples."""

from datetime import date
from fractions import Fraction

import pytest

from factors.common.year52_momentum import (
    formation_book,
    momentum_labels,
    two_week_momentum,
    week_end_labels,
)


def row(number, *, cap=1000000):
    return {
        "provider_id": f"id-{number:02}",
        "native_identity": f"chain:contract-{number}",
        "formation_label": "2025-01-07",
        "known_at": "2025-01-08T00:00:00Z",
        "identity_verified_asof": True,
        "eligible_asof": True,
        "is_stablecoin_asof": False,
        "is_wrapped_asof": False,
        "observed_days_asof": 90,
        "market_cap_usd": cap,
        "volume_usd": 10000000,
        "prices_by_label": {
            "2024-12-22": "100",
            "2024-12-31": "105",
            "2025-01-07": str(100 + number),
        },
    }


def test_52_periods_cover_leap_and_normal_year_without_erasing_extra_days():
    leap, normal = week_end_labels(2024), week_end_labels(2025)
    assert len(leap) == len(normal) == 52
    assert len(set(leap)) == len(set(normal)) == 52
    assert leap[0] == date(2024, 1, 7) and leap[-1] == date(2024, 12, 31)
    assert (leap[-1] - leap[-2]).days == 9
    assert (normal[-1] - normal[-2]).days == 8
    assert (momentum_labels("2025-01-07")[-1] - momentum_labels("2025-01-07")[0]).days == 16
    assert (momentum_labels("2026-01-07")[-1] - momentum_labels("2026-01-07")[0]).days == 15


def test_two_weeks_is_not_a_shift_of_fourteen_dates_at_new_year():
    prices = {"2024-12-22": "100", "2024-12-24": "200", "2024-12-31": "150", "2025-01-07": "110"}
    assert two_week_momentum(prices, "2025-01-07") == Fraction(1, 10)
    assert Fraction(110, 200) - 1 != two_week_momentum(prices, "2025-01-07")
    del prices["2024-12-31"]
    with pytest.raises(ValueError):
        two_week_momentum(prices, "2025-01-07")
    with pytest.raises(ValueError):
        momentum_labels("2025-01-06")


def test_market_cap_weights_and_capital_gross_are_distinct_from_equal_weights():
    rows = [row(n, cap=9000000 if n == 1 else 1000000) for n in range(10)]
    result = formation_book(rows, "2025-01-07", "2025-01-08T00:00:01Z")
    assert result["quintile_counts"] == [2] * 5
    assert result["spread_weights"]["id-00"] == Fraction(-1, 10)
    assert result["spread_weights"]["id-01"] == Fraction(-9, 10)
    assert sum(abs(v) for v in result["spread_weights"].values()) == 2
    assert sum(abs(v) for v in result["account_weights"].values()) == Fraction(4, 5)
    assert sum(result["account_weights"].values()) == 0
    for name, value in result["spread_weights"].items():
        assert result["account_weights"][name] == value * Fraction(2, 5)


def test_missing_cap_is_not_replaced_with_volume_or_current_active_membership():
    rows = [row(n) for n in range(10)]
    rows[0]["market_cap_usd"] = None
    rows[0]["volume_usd"] = 99999999999
    result = formation_book(rows, "2025-01-07", "2025-01-08T00:00:01Z")
    assert result["eligible"] == 9
    assert result["status"] == "insufficient_verified_inputs"
    assert result["account_weights"] == {}


def test_future_revision_cannot_change_past_weights_even_with_same_identity():
    rows = [row(n) for n in range(10)]
    before = formation_book(rows, "2025-01-07", "2025-01-08T00:00:01Z")
    later = row(0, cap=999999999999)
    later["known_at"] = "2025-02-01T00:00:00Z"
    later["prices_by_label"]["2025-01-07"] = "1000000"
    after = formation_book(rows + [later], "2025-01-07", "2025-01-08T00:00:01Z")
    assert after["account_weights"] == before["account_weights"]
    assert after["spread_weights"] == before["spread_weights"]


def test_ambiguous_identity_does_not_duplicate_capital_or_select_a_better_alias():
    rows = [row(n) for n in range(10)]
    duplicate = row(9)
    duplicate["provider_id"] = "different-symbol-same-native-asset"
    with pytest.raises(ValueError, match="duplicate identity"):
        formation_book(rows + [duplicate], "2025-01-07", "2025-01-08T00:00:01Z")


def test_unknown_classification_or_naive_clock_cannot_enter_book():
    rows = [row(n) for n in range(10)]
    rows[0]["is_stablecoin_asof"] = None
    result = formation_book(rows, "2025-01-07", "2025-01-08T00:00:01Z")
    assert result["status"] == "insufficient_verified_inputs"
    with pytest.raises(ValueError, match="Timezone"):
        formation_book(rows, "2025-01-07", "2025-01-08T00:00:01")
