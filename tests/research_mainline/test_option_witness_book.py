from decimal import Decimal

import pytest

from research_tools.options_vrp.audit_witness import apply_book, bbo


def test_native_updates_preserve_full_price_level_ownership():
    levels = apply_book({}, [["new", 0.0425, 50.9], ["new", 0.042, 2]])
    changed = apply_book(levels, [["change", 0.0425, 12], ["delete", 0.042, 0], ["new", 0.0415, 3]])
    assert levels[Decimal(".0425")] == Decimal("50.9")
    assert changed == {Decimal(".0425"): Decimal(12), Decimal(".0415"): Decimal(3)}


def test_missing_level_cannot_be_silently_deleted_or_created_by_change():
    for action in ["change", "delete"]:
        with pytest.raises(ValueError):
            apply_book({}, [[action, 0.04, 1]])
    with pytest.raises(ValueError):
        apply_book({Decimal(".04"): Decimal(1)}, [["new", 0.04, 1]])


def test_crossed_or_missing_side_has_no_executable_bbo():
    assert bbo({Decimal(".04"): Decimal(1)}, {Decimal(".039"): Decimal(1)}) is None
    assert bbo({}, {Decimal(".04"): Decimal(1)}) is None
    assert bbo({Decimal(".04"): Decimal(2)}, {Decimal(".041"): Decimal(3)}) == (
        Decimal(".04"),
        Decimal(2),
        Decimal(".041"),
        Decimal(3),
    )
