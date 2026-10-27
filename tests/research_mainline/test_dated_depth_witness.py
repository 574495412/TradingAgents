"""Economically relevant failure cases for as-of futures depth evidence."""

from decimal import Decimal

import pytest

from research_tools.dated_carry.audit_depth_witness import Book, reconstruct, sell_sweep


def snapshot():
    return {"lastUpdateId": 10, "bids": [["100", "0.010"], ["99", "0.020"]], "asks": [["101", "1"]]}


def delta(**changes):
    return {"U": 9, "u": 11, "pu": 8, "b": [], "a": [], **changes}


def test_rejects_snapshot_without_overlap_and_later_native_gap():
    book = Book(snapshot())
    with pytest.raises(ValueError, match="overlapping"):
        book.update(delta(U=11))
    assert book.update(delta())
    with pytest.raises(ValueError, match="gap"):
        book.update(delta(U=12, u=14, pu=12))


def test_absolute_quantity_and_zero_delete_are_not_additive():
    book = Book(snapshot())
    book.update(delta(b=[["100", "0.005"], ["98", "0"]]))
    assert book.bids[Decimal("100")] == Decimal("0.005")
    book.update(delta(U=12, u=13, pu=11, b=[["100", "0"]]))
    assert max(book.bids) == 99


def test_sell_sweep_crosses_two_levels_and_retains_unfilled_depth():
    book = Book(snapshot())
    filled = sell_sweep(book, "0.013")
    assert len(filled["levels"]) == 2
    assert Decimal(filled["vwap"]) == Decimal("1.297") / Decimal("0.013")
    partial = sell_sweep(book, "0.031")
    assert partial["unfilled"] == "0.001" and partial["vwap"] is None


def test_future_received_snapshot_cannot_backfill_past_state():
    records = [
        {"received_ns": 99, "channel": "depth", "data": delta(), "line": 1},
        {
            "received_ns": 101,
            "channel": "depthSnapshot",
            "data": snapshot(),
            "generated": True,
            "line": 2,
        },
    ]
    with pytest.raises(ValueError, match="No receipt-available"):
        reconstruct(records, 100)
    book, proof = reconstruct(records, 101)
    assert book.last_id == 11 and proof["snapshot_received_ns"] == 101
