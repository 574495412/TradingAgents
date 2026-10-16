"""用完整前缀作对照，反证流式审计的未来权限、冲突与窗口边界。"""

from contextlib import contextmanager

import pytest

from tests.test_research_forward_inputs import candle
from tests.test_research_forward_stream import message
from tradingagents.dataflows.research.live_capture import MINUTE, canonical, digest
from tradingagents.dataflows.research.stream import received_record
from tradingagents.research.quant.backtesting.forward_audit import verify_consensus
from tradingagents.research.quant.backtesting.paper_live_audit import (
    received_bars,
    require_original,
)
from tradingagents.research.quant.backtesting.paper_store import Chain, entries
from tradingagents.research.quant.backtesting.source_cursor import SourceCursor


def history(closing, received=None, price=100):
    received = received or closing + 3000
    return received_record(canonical([candle(closing, price)]), received_ms=received) | {
        "kind": "klines",
        "symbol": "PUBLIC",
        "status": 200,
        "requested_ms": received - 100,
        "local_clock_consistent": True,
    }


@contextmanager
def study(tmp_path, records):
    chain = Chain(tmp_path / "raw.jsonl")
    for record in records:
        chain.add(record)
    chain.close()
    heads = [None] + [head for _, _, head in entries(chain.path)]
    cursor = SourceCursor(entries(chain.path), ["PUBLIC"], tmp_path / "originals.db")
    try:
        yield cursor, heads
    finally:
        cursor.close()


def prefix(heads, count):
    return {"event_count": count, "event_chain_sha256": heads[count]}


def test_longer_prefix_keeps_only_three_hundred_bars_and_preserves_original_membership(tmp_path):
    records = [history(i * MINUTE) for i in range(1, 702)]
    with study(tmp_path, records) as (cursor, heads):
        for count in (1, 100, 300, 500, 700):
            now, closing = count * MINUTE + 4000, count * MINUTE
            cursor.advance(prefix(heads, count), now=now, closing=closing)
            full = received_bars(records[:count], ["PUBLIC"], now)
            full.prune(closing)
            assert cursor.bars.values == full.values
            assert len(cursor.bars.values["PUBLIC"]) <= 300
        require_original(records[0], cursor)
        with pytest.raises(ValueError, match="不在已提交前缀"):
            require_original(records[-1], cursor)
        with pytest.raises(ValueError, match="不在已提交前缀"):
            require_original(records[0] | {"received_ms": 1}, cursor)
        cursor.finish()
        assert cursor.count == len(records) and cursor.head == heads[-1]


@pytest.mark.parametrize("failure", ["future_receipt", "wrong_head", "absent_prefix"])
def test_future_receipt_changed_identity_or_absent_prefix_cannot_advance(tmp_path, failure):
    with study(tmp_path, [history(MINUTE)]) as (cursor, heads):
        value = prefix(heads, 1)
        now = MINUTE + 4000
        if failure == "future_receipt":
            now = MINUTE + 2000
        elif failure == "wrong_head":
            value["event_chain_sha256"] = "另一身份"
        else:
            value["event_count"] += 1
        with pytest.raises(ValueError, match="尚未收到|身份错误|尚不存在"):
            cursor.advance(value, now=now, closing=MINUTE)


@pytest.mark.parametrize(
    "failure", [None, "missing_confirmation", "wrong_confirmation", "false_first_source"]
)
def test_each_conflict_requires_prior_raw_value_and_later_exact_confirmation(tmp_path, failure):
    first, wrong = history(MINUTE), history(MINUTE, MINUTE + 5000, price=101)
    declared = {
        "kind": "startup_public_value_conflict",
        "received_ms": MINUTE + 6000,
        "conflicts": [
            {
                "symbol": "PUBLIC",
                "closing_ms": MINUTE,
                "first_source_sha256": first["body_sha256"]
                if failure != "false_first_source"
                else "不存在的原文",
                "conflicting_source_sha256": wrong["body_sha256"],
            }
        ],
    }
    confirmed = history(
        MINUTE, 2 * MINUTE + 3000, price=102 if failure == "wrong_confirmation" else 100
    )
    confirmed["purpose"] = "STARTUP_CONSENSUS_RECHECK_ONLY"
    records = [first, wrong, declared] + ([] if failure == "missing_confirmation" else [confirmed])
    with study(tmp_path, records) as (cursor, heads):
        if failure:
            with pytest.raises(ValueError, match="冲突|两份原文"):
                cursor.advance(
                    prefix(heads, len(records)), now=2 * MINUTE + 4000, closing=2 * MINUTE
                )
        else:
            cursor.advance(prefix(heads, len(records)), now=2 * MINUTE + 4000, closing=2 * MINUTE)
            assert verify_consensus(records) == cursor.confirmed_conflicts == 1
            assert (
                cursor.bars.values == received_bars(records, ["PUBLIC"], 2 * MINUTE + 4000).values
            )


def test_session_reset_cannot_promote_an_old_pending_push(tmp_path):
    pending = received_record(canonical(message()), received_ms=120_200)
    session = {"kind": "paper_process_session", "received_ms": 120_600}
    records = [pending, session, history(MINUTE, 123_000)]
    with study(tmp_path, records) as (cursor, heads):
        cursor.advance(prefix(heads, 1), now=120_500, closing=MINUTE)
        assert cursor.pending and cursor.bars.at("PUBLIC", 2 * MINUTE, 120_500) is None
        cursor.advance(prefix(heads, 3), now=123_500, closing=2 * MINUTE)
        assert not cursor.pending and cursor.sessions == 1
        assert cursor.bars.values == received_bars(records, ["PUBLIC"], 123_500).values
        assert digest(pending) in cursor


def test_excessive_single_frame_gap_fails_before_caching_an_unbounded_history(tmp_path):
    records = [{"kind": "fixture", "received_ms": i} for i in range(10_001)]
    with study(tmp_path, records) as (cursor, heads):
        with pytest.raises(ValueError, match="超过固定审计容量"):
            cursor.advance(prefix(heads, len(records)), now=20_000, closing=0)
        assert cursor.maximum_prefix_events == 0
