"""活动进程尾部、检查点身份和已提交段边界的复制反证。"""

from pathlib import Path

import pytest

from tradingagents.research.quant.backtesting.prefix_snapshot import copy_prefix, snapshot
from tradingagents.research.quant.backtesting.raw_chain import raw_entries
from tradingagents.research.quant.backtesting.segmented_chain import SegmentedChain
from tradingagents.research.quant.strategy_audit import save, sha


@pytest.mark.parametrize(
    "failure", [None, "bad_checkpoint", "inside_source", "middle_of_segment", "corrupt_segment"]
)
def test_snapshot_copies_exact_checkpoint_prefix_and_excludes_later_committed_tail(
    tmp_path, failure
):
    root, output = tmp_path / "原研究", tmp_path / "前缀"
    root.mkdir()
    source = root / "source"
    source.mkdir()
    file = Path(__file__)
    (source / file.name).write_bytes(file.read_bytes())
    save(
        root / "protocol.json",
        {
            "orders_enabled": False,
            "storage": "segmented-gzip-v1",
            "simulation_end_ms": 999999,
            "source_sha256": {str(file): sha(file)},
        },
    )
    original = SegmentedChain(root / "public_events")
    original.add({"kind": "人工行情", "received_ms": 1000})
    prefix = original.seal()
    frames = SegmentedChain(root / "account_frames")
    frames.add({"frame": {"actual_ms": 1001}, "source_prefix": prefix})
    head = frames.seal()
    save(
        root / "checkpoint.json",
        {
            "frame_count": 1,
            "frame_chain_sha256": "错误"
            if failure == "bad_checkpoint"
            else head["event_chain_sha256"],
            "accounts": {},
        },
    )
    original.add({"kind": "稍后的行情", "received_ms": 2000})
    if failure == "middle_of_segment":
        original.add({"kind": "同段第二条行情", "received_ms": 2001})
    original.seal()
    frames.add({"frame": {"actual_ms": 2001}, "source_prefix": original.seal()})
    frames.seal()
    if failure == "corrupt_segment":
        next((root / "public_events").rglob("*.gz")).write_bytes("坏段".encode())
    try:
        if failure == "middle_of_segment":
            with pytest.raises(ValueError):
                copy_prefix(root / "public_events", output, count=2)
        elif failure:
            with pytest.raises(ValueError):
                snapshot(root, root / "内部快照" if failure == "inside_source" else output)
        else:
            result = snapshot(root, output)
            assert result["source_prefix"] == prefix and result["frame_prefix"] == head
            assert len(list(raw_entries(output / "public_events"))) == 1
            assert len(list(raw_entries(output / "account_frames"))) == 1
            assert result["original_study_terminally_sealed"] is False
            assert (root / "checkpoint.json").read_bytes() == (
                output / "checkpoint.json"
            ).read_bytes()
            assert len(list(raw_entries(root / "public_events"))) == 2
    finally:
        original.close()
        frames.close()
