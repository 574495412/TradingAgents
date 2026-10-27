"""No future snapshots, hidden update gaps or favorable rule-version selection."""

import pytest

from research_tools.dated_carry.audit_prospective_probe import (
    SpotBook,
    reconstruct,
    rule_fingerprint,
)


def snapshot():
    return {"lastUpdateId": 10, "bids": [["100", "1"]], "asks": [["101", "1"]]}


def test_spot_next_id_rule_differs_from_future_overlap_and_rejects_gap():
    book = SpotBook(snapshot())
    assert book.update({"U": 11, "u": 12, "b": [["100", "2"]], "a": []})
    with pytest.raises(ValueError, match="gap"):
        book.update({"U": 14, "u": 15, "b": [], "a": []})


def test_future_received_snapshot_or_delta_cannot_repair_an_earlier_state():
    rows = [
        {"received_wall_ns": 100, "data": {"U": 11, "u": 11, "b": [["100", "2"]], "a": []}},
        {"received_wall_ns": 102, "data": {"U": 13, "u": 14, "b": [], "a": []}},
    ]
    with pytest.raises(ValueError, match="after common"):
        reconstruct(rows, snapshot(), 101, 100, "spot")
    book, known = reconstruct(rows, snapshot(), 99, 101, "spot")
    assert book.last_id == 11 and known == 100
    with pytest.raises(ValueError, match="gap"):
        reconstruct(rows, snapshot(), 99, 103, "spot")


def test_filter_order_is_not_a_rule_change_but_quantity_or_identity_is():
    old = {
        "symbol": "BTCUSDT",
        "baseAsset": "BTC",
        "filters": [
            {"filterType": "LOT_SIZE", "stepSize": "0.001"},
            {"filterType": "MIN_NOTIONAL", "notional": "5"},
        ],
    }
    reordered = {**old, "filters": list(reversed(old["filters"]))}
    assert rule_fingerprint(old) == rule_fingerprint(reordered)
    changed = {**old, "filters": [{"filterType": "LOT_SIZE", "stepSize": "0.01"}]}
    assert rule_fingerprint(old) != rule_fingerprint(changed)
    assert rule_fingerprint(old) != rule_fingerprint({**old, "baseAsset": "OTHER"})
