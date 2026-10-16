"""坏JSON是原文质量问题，不能导致完整接收字节丢失或变成有效行情。"""

import pytest

from tradingagents.dataflows.research.live_capture import body
from tradingagents.dataflows.research.stream import received_record
from tradingagents.research.quant.backtesting.raw_chain import raw_entries
from tradingagents.research.quant.backtesting.segmented_chain import SegmentedChain, segment_entries


@pytest.mark.parametrize("raw", [b"<html>not json</html>", b'{"price":NaN}'])
def test_invalid_success_payload_remains_byte_verifiable_but_cannot_be_consumed(raw, tmp_path):
    record = received_record(raw, received_ms=10) | {"kind": "clock", "status": 200}
    chain = SegmentedChain(tmp_path / "source")
    chain.add(record)
    prefix = chain.seal()
    chain.close()
    received = list(raw_entries(tmp_path / "source"))
    assert received == [(0, record, prefix["event_chain_sha256"])]
    with pytest.raises(ValueError):
        body(record)
    with pytest.raises(ValueError):
        list(segment_entries(tmp_path / "source"))


def test_byte_reader_still_rejects_a_false_original_hash(tmp_path):
    record = received_record(b"[]", received_ms=10) | {"body_sha256": "错误的原文身份"}
    chain = SegmentedChain(tmp_path / "source")
    chain.add(record)
    chain.close()
    with pytest.raises(ValueError, match="原文字节与接收身份"):
        list(raw_entries(tmp_path / "source"))
