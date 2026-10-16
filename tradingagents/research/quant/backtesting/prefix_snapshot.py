"""复制已提交前缀供离线审计，绝不停止、锁定或改写正在运行的研究。"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from tradingagents.dataflows.research.live_capture import digest
from tradingagents.research.quant.backtesting.raw_chain import raw_entries
from tradingagents.research.quant.strategy_audit import save, sha


def copy_prefix(source, destination, count=None):
    """先固定索引字节，再复制不可变段；指定终点必须是已提交段的完整边界。"""
    source, destination = Path(source), Path(destination)
    raw = (source / "index.jsonl").read_bytes()
    lines = raw.splitlines(keepends=True)
    if lines and not lines[-1].endswith(b"\n"):
        lines.pop()
    selected, index_head, event_count, event_head = [], None, 0, None
    for sequence, line in enumerate(lines):
        entry = json.loads(line)
        if entry["sequence"] != sequence or entry["previous_sha256"] != index_head:
            raise ValueError("前缀索引链不连续")
        segment = entry["record"]
        if segment["start_count"] != event_count or segment["previous_event_sha256"] != event_head:
            raise ValueError("前缀段范围不连续")
        if count is not None and segment["end_count"] > count:
            raise ValueError("指定前缀不是已提交完整段的边界")
        selected.append(line)
        index_head = digest(entry)
        event_count, event_head = segment["end_count"], segment["event_chain_sha256"]
        if count is not None and event_count == count:
            break
    if not selected or (count is not None and event_count != count):
        raise ValueError("没有可复制的完整已提交前缀")
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "index.jsonl").write_bytes(b"".join(selected))
    for line in selected:
        segment = json.loads(line)["record"]
        relative = Path(segment["file"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or len(relative.parts) != 3
            or relative.parts[0] != "segments"
        ):
            raise ValueError("前缀段路径越界")
        origin, target = source / relative, destination / relative
        if (
            not origin.resolve().is_relative_to(source.resolve())
            or sha(origin) != segment["compressed_sha256"]
        ):
            raise ValueError("已提交原段的字节身份错误")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, target)
    actual_count, actual_head, maximum_received = 0, None, 0
    for sequence, record, current_head in raw_entries(destination):
        actual_head = current_head
        actual_count = sequence + 1
        maximum_received = max(maximum_received, record.get("received_ms", 0))
    if (actual_count, actual_head) != (event_count, event_head):
        raise ValueError("复制段的完整事件终点不同")
    return {"event_count": event_count, "event_chain_sha256": event_head}, maximum_received


def snapshot(root, output, *, observer=False):
    """快照保留原始结束时间；封存前缀不代表原研究已结束或经济条件通过。"""
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.is_relative_to(root):
        raise ValueError("快照不得写入原研究目录")
    protocol_bytes = (root / "protocol.json").read_bytes()
    protocol = json.loads(protocol_bytes)
    if protocol["orders_enabled"] is not False or protocol["storage"] != "segmented-gzip-v1":
        raise ValueError("只复制禁用订单的分段公开研究")
    checkpoint_bytes = None if observer else (root / "checkpoint.json").read_bytes()
    checkpoint = json.loads(checkpoint_bytes) if checkpoint_bytes is not None else None
    output.mkdir(parents=True, exist_ok=False)
    (output / "protocol.json").write_bytes(protocol_bytes)
    shutil.copytree(root / "source", output / "source")
    for origin, expected in protocol["source_sha256"].items():
        parts = Path(origin).parts
        namespace = "tradingagents" if "tradingagents" in parts else "cz"
        relative = Path(*parts[parts.index(namespace) :]) if observer else Path(origin).name
        if sha(output / "source" / relative) != expected:
            raise ValueError("复制的运行源码档案不同")
    frame_prefix = None
    if checkpoint is not None:
        frame_prefix, _ = copy_prefix(
            root / "account_frames", output / "account_frames", checkpoint["frame_count"]
        )
        if frame_prefix["event_chain_sha256"] != checkpoint["frame_chain_sha256"]:
            raise ValueError("检查点与账户帧终点不同")
        last = None
        for _, record, _ in raw_entries(output / "account_frames"):
            last = record
        wanted = last["source_prefix"]
        prefix, maximum = copy_prefix(
            root / "public_events", output / "public_events", wanted["event_count"]
        )
        if prefix != wanted:
            raise ValueError("账户帧引用的公开前缀身份不同")
        (output / "checkpoint.json").write_bytes(checkpoint_bytes)
    else:
        prefix, maximum = copy_prefix(root / "public_events", output / "public_events")
    result = {
        "schema": "cz-public-research-prefix-snapshot.v1",
        "kind": "observer" if observer else "financial",
        "original_root": str(root),
        "observed_ms": time.time_ns() // 1_000_000,
        "protocol_sha256": sha(output / "protocol.json"),
        "source_prefix": prefix,
        "maximum_record_received_ms": maximum,
        "frame_prefix": frame_prefix,
        "original_study_terminally_sealed": False,
        "capture_end_reached": False,
        "account_state_mutated": False,
        "strategy_profitability_evaluated": False,
        "orders_enabled": False,
        "production_eligible": False,
    }
    (output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    save(output / "prefix_snapshot.json", result)
    save(
        output / "receipts.json",
        {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()},
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observer", action="store_true")
    args = parser.parse_args()
    print(snapshot(args.source, args.output, observer=args.observer))


if __name__ == "__main__":
    main()
