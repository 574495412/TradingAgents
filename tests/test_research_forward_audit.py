"""用被破坏的原文与未来接收时钟反证输入审计的关键边界。"""

import json

import pytest

from tradingagents.dataflows.research.stream import received_record
from tradingagents.research.quant.backtesting.forward_audit import (
    verify_chain,
    verify_consensus,
    verify_prefix,
)
from tradingagents.research.quant.backtesting.forward_pilot import Journal


def test_journal_chain_corruption_is_rejected(tmp_path):
    journal = Journal(tmp_path)
    journal.add({"received_ms": 1000})
    journal.close()
    path = tmp_path / "public_events.jsonl"
    records, heads = verify_chain(path)
    assert len(records) == 1 and len(heads) == 2
    value = json.loads(path.read_text())
    value["previous_sha256"] = "other"
    path.write_text(json.dumps(value) + "\n")
    with pytest.raises(ValueError, match="不连续"):
        verify_chain(path)


def test_future_received_data_cannot_support_an_earlier_decision(tmp_path):
    journal = Journal(tmp_path)
    journal.add({"received_ms": 65_001})
    prefix = journal.seal()
    journal.close()
    records, heads = verify_chain(tmp_path / "public_events.jsonl")
    decision = {
        "source_prefix": prefix,
        "actual_decision_ms": 65_000,
        "canonical_close_ms": 60_000,
        "within_fixed_one_minute_deadline": True,
        "emitted": [],
        "orders_enabled": False,
    }
    with pytest.raises(ValueError, match="尚未收到"):
        verify_prefix(decision, records, heads)


def test_waiting_one_minute_from_bar_close_is_insufficient(tmp_path):
    journal = Journal(tmp_path)
    journal.add({"received_ms": 64_000})
    prefix = journal.seal()
    journal.close()
    records, heads = verify_chain(tmp_path / "public_events.jsonl")
    decision = {
        "source_prefix": prefix,
        "actual_decision_ms": 65_000,
        "canonical_close_ms": 60_000,
        "within_fixed_one_minute_deadline": True,
        "price_ready": ["PUBLIC"],
        "emitted": [
            {
                "actual_decision_ms": 65_000,
                "earliest_execution_ms": 120_000,
                "symbol": "PUBLIC",
                "orders_enabled": False,
            }
        ],
        "orders_enabled": False,
    }
    with pytest.raises(ValueError, match="时间"):
        verify_prefix(decision, records, heads)


def test_conflict_needs_actual_later_exact_confirmation():
    row = [0, "100", "101", "99", "100", "1", 59999, "100", 1, "0.5", "50", "0"]
    first = received_record(json.dumps([row]), received_ms=62_000) | {
        "kind": "klines",
        "symbol": "PUBLIC",
        "status": 200,
    }
    event = {
        "kind": "startup_public_value_conflict",
        "received_ms": 150_000,
        "conflicts": [
            {"symbol": "PUBLIC", "closing_ms": 60_000, "first_source_sha256": first["body_sha256"]}
        ],
    }
    with pytest.raises(ValueError, match="未被"):
        verify_consensus([first, event])
    confirmation = received_record(json.dumps([row]), received_ms=152_000) | {
        "kind": "klines",
        "symbol": "PUBLIC",
        "status": 200,
        "purpose": "STARTUP_CONSENSUS_RECHECK_ONLY",
        "requested_ms": 151_000,
    }
    assert verify_consensus([first, event, confirmation]) == 1
    with pytest.raises(ValueError, match="没有原文依据"):
        verify_consensus([event, confirmation])
