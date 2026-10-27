"""Prevent date, reference-role and vendor-capitalization substitutions."""

from copy import deepcopy

import pytest

from factors.reference.crypto_risk_benchmark import (
    require_reference_use,
    week_label,
    window_coverage,
)
from research_tools.cross_sectional_momentum.audit_public_capitalization import validate_rows


def sample():
    return {
        "data": [
            {
                "asset": a,
                "time": "2025-01-07T00:00:00.000000000Z",
                "PriceUSD": "100.1234567890123456789",
                "SplyCur": "100000000",
                "CapMrktCurUSD": "10012345678.90123456789",
                "CapMrktEstUSD": "9000000000",
            }
            for a in ["btc", "eth"]
        ]
    }


def test_author_week_key_is_not_excel_serial_or_ISO_week():
    assert week_label(202314).isoformat() == "2023-04-08"
    assert week_label(202452).isoformat() == "2024-12-31"
    for value in [45024, "2023-14", 202353, 202300]:
        with pytest.raises(ValueError):
            week_label(value)


def test_old_reference_cannot_create_contemporary_overlap_or_missing_zero_returns():
    coverage = window_coverage([201404, 202252, 202314], "2025-01-01", "2026-09-01")
    assert coverage["overlap_rows"] == 0
    assert coverage["expected_nominal_week_labels"] == 86
    assert coverage["first_overlap_label"] is None
    assert not coverage["all_nominal_week_labels_covered"]
    with pytest.raises(ValueError, match="Duplicate"):
        window_coverage([202314, 202314], "2023-01-01", "2024-01-01")


def test_derived_returns_cannot_be_used_as_raw_capitalization_or_live_signal():
    require_reference_use("calendar_check")
    for role in ["live_signal", "capitalization_input", "identity_history"]:
        with pytest.raises(ValueError, match="Realized"):
            require_reference_use(role, units_verified=True)


def test_attribution_requires_both_units_and_complete_matching_window():
    coverage = window_coverage([202501], "2025-01-01", "2025-01-08")
    with pytest.raises(ValueError, match="units"):
        require_reference_use("ex_post_attribution", coverage=coverage)
    require_reference_use("ex_post_attribution", units_verified=True, coverage=coverage)
    with pytest.raises(ValueError, match="window"):
        require_reference_use(
            "ex_post_attribution",
            units_verified=True,
            coverage={"all_nominal_week_labels_covered": False},
        )


def test_capitalization_identity_uses_exact_high_precision_and_keeps_definitions_separate():
    checked = validate_rows(sample())
    assert len(checked) == 2
    assert all(not r["two_cap_definitions_equal"] for r in checked)
    assert all(not r["historical_publication_time_certified"] for r in checked)


def test_http200_null_cap_or_wrong_time_is_not_a_valid_observation():
    for bad in [None, "NaN", "0"]:
        data = sample()
        data["data"][0]["CapMrktEstUSD"] = bad
        with pytest.raises(ValueError):
            validate_rows(data)
    data = sample()
    data["data"][0]["time"] = "2025-01-08T00:00:00.000000000Z"
    with pytest.raises(ValueError, match="time"):
        validate_rows(data)


def test_duplicate_identity_or_inconsistent_supply_cannot_pass():
    data = sample()
    data["data"][1] = deepcopy(data["data"][0])
    with pytest.raises(ValueError, match="identity"):
        validate_rows(data)
    data = sample()
    data["data"][0]["SplyCur"] = "100000001"
    with pytest.raises(ValueError, match="reconcile"):
        validate_rows(data)
