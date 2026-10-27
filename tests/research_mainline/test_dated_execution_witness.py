"""Prevent future quotes and timestamp/lot assumptions from certifying old fills."""

import zipfile
from decimal import Decimal

from research_tools.dated_carry.audit_execution_witness import (
    archive_bar,
    latest_known_quote,
    quantity_lattice,
)


def test_future_received_quote_cannot_repair_missing_historical_quote():
    quotes = [{"received_ns": 101, "a": "1"}, {"received_ns": 102, "a": "2"}]
    assert latest_known_quote(quotes, 100) is None
    assert latest_known_quote(quotes, 101) == quotes[0]


def test_last_available_quote_wins_and_does_not_choose_better_price():
    quotes = [
        {"received_ns": 99, "a": "1"},
        {"received_ns": 100, "a": "3"},
        {"received_ns": 101, "a": "2"},
    ]
    assert latest_known_quote(quotes, 100)["a"] == "3"


def test_trade_quantity_lattice_keeps_fractional_model_residue():
    lattice = quantity_lattice(["0.003", "0.007", "0.002"])
    assert lattice == Decimal("0.001")
    assert Decimal("0.013331436452749612") % lattice == Decimal("0.000331436452749612")


def test_january_spot_microseconds_and_future_milliseconds_are_same_bar(tmp_path):
    for scale, unit in [(1, "ms"), (1000, "us")]:
        archive = tmp_path / (unit + ".zip")
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.writestr(
                "table.csv", "open_time,open\n" + f"{1735689600000 * scale},1,2,1,2,3,0,4,5,0,0,0\n"
            )
        row = archive_bar(archive, 1735689600000)
        assert row["archive_timestamp_unit"] == unit and row["count"] == 5
