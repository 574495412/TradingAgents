"""分段压缩原有事件链，压缩只改变存储，不改变事件身份或接收时间。"""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
import uuid
from pathlib import Path

from tradingagents.dataflows.research.live_capture import body, canonical, digest
from tradingagents.research.quant.backtesting.paper_store import Chain, entries
from tradingagents.research.quant.strategy_audit import sha

FORMAT = "segmented-gzip-v1"


def sync_directory(path):
    """段文件的改名也必须同步到目录，不能只同步压缩内容。"""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def segments(root):
    """索引本身有校验链，段路径仅限研究目录内的固定分组。"""
    root = Path(root)
    count, head = 0, None
    for _, segment, _ in entries(root / "index.jsonl"):
        path = Path(segment["file"])
        if (
            segment["format"] != FORMAT
            or path.is_absolute()
            or ".." in path.parts
            or len(path.parts) != 3
            or path.parts[0] != "segments"
            or segment["start_count"] != count
            or segment["previous_event_sha256"] != head
            or segment["end_count"] <= count
        ):
            raise ValueError("压缩段索引的范围、路径或前缀身份错误")
        target = root / path
        if (
            not target.resolve().is_relative_to(root.resolve())
            or sha(target) != segment["compressed_sha256"]
        ):
            raise ValueError("压缩段内容身份错误")
        yield segment
        count, head = segment["end_count"], segment["event_chain_sha256"]


def segment_entries(root):
    """流式还原全部原始行，并逐条验证与平面日志相同的事件哈希。"""
    root = Path(root)
    count, head = 0, None
    for segment in segments(root):
        decoded = hashlib.sha256()
        decoded_bytes = 0
        with gzip.open(root / segment["file"], "rb") as stream:
            for line in stream:
                decoded.update(line)
                decoded_bytes += len(line)
                if not line.endswith(b"\n"):
                    raise ValueError("压缩段内存在未提交的半行")
                entry = json.loads(line)
                if entry["sequence"] != count or entry["previous_sha256"] != head:
                    raise ValueError("解压后的事件校验链不连续")
                record = entry["record"]
                if "body_base64" in record:
                    raw = base64.b64decode(record["body_base64"], validate=True)
                    if hashlib.sha256(raw).hexdigest() != record["body_sha256"]:
                        raise ValueError("压缩原文的接收字节校验失败")
                    if record.get("status") == 200 or record.get("kind") == "websocket":
                        body(record)
                head = digest(entry)
                yield count, record, head
                count += 1
        if (
            count != segment["end_count"]
            or head != segment["event_chain_sha256"]
            or decoded.hexdigest() != segment["decoded_sha256"]
            or decoded_bytes != segment["decoded_bytes"]
        ):
            raise ValueError("解压段的终点或完整字节身份错误")


class SegmentedChain:
    """决策前封闭压缩段并同步索引，未提交段单独保留，绝不接到旧前缀。"""

    def __init__(self, path, *, recover=False, maximum_bytes=4 * 1024 * 1024, maximum_events=1024):
        self.path = Path(path)
        self.maximum_bytes, self.maximum_events = maximum_bytes, maximum_events
        if maximum_bytes <= 0 or maximum_events <= 0:
            raise ValueError("压缩段上限必须为正")
        self.path.mkdir(parents=True, exist_ok=recover)
        sync_directory(self.path.parent)
        self.index = Chain(self.path / "index.jsonl", recover=recover)
        self.count, self.head, self.recovered_tail = 0, None, self.index.recovered_tail
        self.active = self.compressor = None
        self.segment_number = self.index.count
        try:
            for count, _, head in segment_entries(self.path):
                self.count, self.head = count + 1, head
            referenced = {s["file"] for s in segments(self.path)}
            orphans = [
                p
                for p in self.path.rglob("*.gz")
                if str(p.relative_to(self.path)) not in referenced
            ]
            orphans.extend(self.path.rglob("*.tmp"))
            if orphans:
                quarantine = self.path / ("uncommitted-" + uuid.uuid4().hex)
                quarantine.mkdir()
                for number, orphan in enumerate(orphans):
                    os.replace(orphan, quarantine / f"{number}-{orphan.name}.uncommitted")
                sync_directory(quarantine)
                sync_directory(self.path)
                self.recovered_tail = quarantine.name
        except BaseException:
            self.index.close()
            raise

    def _open(self):
        directory = self.path / "segments"
        directory.mkdir(exist_ok=True)
        sync_directory(self.path)
        group = directory / f"{self.segment_number // 1000:06d}"
        group.mkdir(parents=True, exist_ok=True)
        sync_directory(directory)
        self.final = group / f"{self.segment_number:09d}.jsonl.gz"
        self.temporary = group / (self.final.name + "." + uuid.uuid4().hex + ".tmp")
        self.active = self.temporary.open("xb")
        self.compressor = gzip.GzipFile(
            filename="", mode="wb", fileobj=self.active, compresslevel=6, mtime=0
        )
        self.start_count, self.start_head = self.count, self.head
        self.written_bytes, self.decoded = 0, hashlib.sha256()

    def add(self, record):
        if self.active is None:
            self._open()
        entry = {"sequence": self.count, "previous_sha256": self.head, "record": record}
        line = canonical(entry) + b"\n"
        self.compressor.write(line)
        self.decoded.update(line)
        self.written_bytes += len(line)
        self.count, self.head = self.count + 1, digest(entry)
        if (
            self.written_bytes >= self.maximum_bytes
            or self.count - self.start_count >= self.maximum_events
        ):
            self.seal()

    def seal(self):
        if self.active is not None:
            self.compressor.close()
            self.active.flush()
            os.fsync(self.active.fileno())
            self.active.close()
            os.replace(self.temporary, self.final)
            sync_directory(self.final.parent)
            self.index.add(
                {
                    "format": FORMAT,
                    "file": str(self.final.relative_to(self.path)),
                    "start_count": self.start_count,
                    "end_count": self.count,
                    "previous_event_sha256": self.start_head,
                    "event_chain_sha256": self.head,
                    "decoded_sha256": self.decoded.hexdigest(),
                    "compressed_sha256": sha(self.final),
                    "decoded_bytes": self.written_bytes,
                }
            )
            self.index.seal()
            sync_directory(self.path)
            self.segment_number += 1
            self.active = self.compressor = None
        return {"event_count": self.count, "event_chain_sha256": self.head}

    def iter_entries(self):
        return segment_entries(self.path)

    def close(self):
        self.seal()
        self.index.close()


def load_chain(root, name, storage):
    """审计统一返回原事件，不根据存储方式修改任何模型输入。"""
    if storage not in {FORMAT, "flat-jsonl"}:
        raise ValueError("未知研究存储版本")
    iterator = (
        segment_entries(Path(root) / name)
        if storage == FORMAT
        else entries(Path(root) / f"{name}.jsonl")
    )
    records, heads = [], [None]
    for _, record, head in iterator:
        records.append(record)
        heads.append(head)
    return records, heads
