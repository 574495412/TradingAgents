"""验证压缩不改变事件、账户状态和失败原文，并反证不完整提交。"""

import gzip
import hashlib

import pytest

from tests.test_research_paper_account import frame
from tradingagents.dataflows.research.stream import received_record
from tradingagents.research.quant.backtesting.paper_store import Chain, StudyStore, entries
from tradingagents.research.quant.backtesting.segmented_chain import (
    FORMAT,
    SegmentedChain,
    segment_entries,
    segments,
)


def test_compressed_and_flat_logs_have_identical_records_prefixes_and_decoded_bytes(tmp_path):
    flat, packed = (
        Chain(tmp_path / "flat.jsonl"),
        SegmentedChain(tmp_path / "packed", maximum_events=2),
    )
    values = [{"kind": "fixture", "received_ms": i, "中文": "完整原文"} for i in range(5)]
    values.append(received_record("", received_ms=10) | {"kind": "index", "status": 0})
    for value in values:
        flat.add(value)
        packed.add(value)
    assert flat.seal() == packed.seal()
    flat.close()
    packed.close()
    assert list(entries(tmp_path / "flat.jsonl")) == list(segment_entries(tmp_path / "packed"))
    decoded = b"".join(
        gzip.decompress((tmp_path / "packed" / s["file"]).read_bytes())
        for s in segments(tmp_path / "packed")
    )
    assert (
        hashlib.sha256(decoded).digest()
        == hashlib.sha256((tmp_path / "flat.jsonl").read_bytes()).digest()
    )
    assert len(list(segments(tmp_path / "packed"))) == 3


def test_unindexed_complete_or_partial_segments_are_preserved_but_not_replayed(tmp_path):
    root = tmp_path / "packed"
    packed = SegmentedChain(root)
    packed.add({"kind": "fixture", "received_ms": 1})
    expected = packed.seal()
    packed.close()
    group = root / "segments" / "000000"
    (group / "999999999.jsonl.gz").write_bytes(gzip.compress("未提交的完整段".encode()))
    (group / "unfinished.tmp").write_bytes("未提交的半段".encode())
    restored = SegmentedChain(root, recover=True)
    try:
        assert restored.seal() == expected
        assert restored.recovered_tail
        assert len(list((root / restored.recovered_tail).iterdir())) == 2
        assert len(list(restored.iter_entries())) == 1
    finally:
        restored.close()


def test_compressed_byte_corruption_cannot_be_accepted_from_matching_index_fields(tmp_path):
    root = tmp_path / "packed"
    packed = SegmentedChain(root)
    packed.add({"kind": "fixture", "received_ms": 1})
    packed.close()
    segment = next(segments(root))
    path = root / segment["file"]
    raw = bytearray(path.read_bytes())
    raw[len(raw) // 2] ^= 1
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="压缩段内容身份错误"):
        SegmentedChain(root, recover=True)


def test_segmented_study_restores_open_inventory_identically_to_flat_study(tmp_path):
    states = []
    for storage in ("flat-jsonl", FORMAT):
        root = tmp_path / storage
        store = StudyStore(
            root,
            {
                "symbols": ["A"],
                "simulation_end_ms": 12_000_000,
                "storage": storage,
                "fixtures_only": True,
            },
        )
        for minute in range(100, 106):
            value = frame(minute, direction=1 if minute >= 103 else 0)
            store.source.add({"kind": "fixture", "received_ms": value["actual_ms"]})
            store.commit(value)
        states.append(store.states())
        store.close()
        restored = StudyStore(root, resume=True)
        try:
            assert restored.states() == states[-1]
            assert len(restored.accounts["3"].positions) == 1
        finally:
            restored.close()
    assert states[0] == states[1]


def test_unknown_storage_version_cannot_fall_back_to_a_different_log(tmp_path):
    with pytest.raises(ValueError, match="未知研究存储"):
        StudyStore(
            tmp_path / "unknown", {"symbols": ["A"], "simulation_end_ms": 1, "storage": "unknown"}
        )
