"""公开模拟进程的文件锁、双日志和检查点恢复，拒绝覆盖已提交前缀。"""

from __future__ import annotations

import copy
import fcntl
import json
import os
import time
import uuid
from pathlib import Path

from tradingagents.dataflows.research.live_capture import body, canonical, digest
from tradingagents.research.quant.backtesting.paper_account import PaperAccount


def atomic(path, value):
    """先同步临时文件，再原子替换并同步目录。"""
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def entries(path):
    """流式验证完整日志；只有不带换行的最后一段允许交给恢复器处理。"""
    head = None
    if not Path(path).exists():
        return
    with Path(path).open("rb") as stream:
        for sequence, line in enumerate(stream):
            if not line.endswith(b"\n"):
                raise ValueError("日志尾部尚未完整提交")
            entry = json.loads(line)
            if entry["sequence"] != sequence or entry["previous_sha256"] != head:
                raise ValueError("持久日志校验链不连续")
            if "body_base64" in entry["record"]:
                import base64
                import hashlib

                record = entry["record"]
                raw = base64.b64decode(record["body_base64"], validate=True)
                if hashlib.sha256(raw).hexdigest() != record["body_sha256"]:
                    raise ValueError("失败响应的原文字节身份不一致")
                if record.get("status") == 200 or record.get("kind") == "websocket":
                    body(record)
            head = digest(entry)
            yield sequence, entry["record"], head


class Chain:
    """日志只追加；断电留下的半行先另存现场，完整行从不静默截断。"""

    def __init__(self, path, *, recover=False):
        self.path = Path(path)
        self.count, self.head, self.recovered_tail = 0, None, None
        if self.path.exists() and not recover:
            raise ValueError("新日志不能覆盖已有文件")
        if self.path.exists():
            last_end = 0
            with self.path.open("rb") as stream:
                for line in stream:
                    if not line.endswith(b"\n"):
                        tail = self.path.with_name(
                            self.path.name + ".uncommitted-" + uuid.uuid4().hex
                        )
                        with tail.open("xb") as target:
                            target.write(line)
                            target.flush()
                            os.fsync(target.fileno())
                        self.recovered_tail = tail.name
                        break
                    last_end += len(line)
            if self.recovered_tail:
                with self.path.open("r+b") as stream:
                    stream.truncate(last_end)
                    stream.flush()
                    os.fsync(stream.fileno())
            for sequence, _, head in entries(self.path):
                self.count, self.head = sequence + 1, head
        self.file = self.path.open("ab" if recover else "xb")

    def add(self, record):
        value = {"sequence": self.count, "previous_sha256": self.head, "record": record}
        self.file.write(canonical(value) + b"\n")
        self.count, self.head = self.count + 1, digest(value)

    def seal(self):
        self.file.flush()
        os.fsync(self.file.fileno())
        return {"event_count": self.count, "event_chain_sha256": self.head}

    def close(self):
        self.file.close()

    def iter_entries(self):
        return entries(self.path)


class StudyStore:
    """两成本账户由同一已封存帧推进，检查点落后时重放已提交尾部。"""

    def __init__(self, root, protocol=None, *, resume=False):
        self.root = Path(root)
        if not resume:
            self.root.mkdir(parents=True, exist_ok=False)
        self.lock = (self.root / ".research.lock").open("a+b")
        try:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise ValueError("同一研究目录已有进程持有文件锁") from None
        self.source = self.frames = None
        try:
            if resume:
                self.protocol = json.loads((self.root / "protocol.json").read_text())
                if protocol is not None and protocol != self.protocol:
                    raise ValueError("恢复参数不能更改原研究协议")
            else:
                self.protocol = protocol
                atomic(self.root / "protocol.json", protocol)
            if self.protocol.get("storage", "flat-jsonl") == "segmented-gzip-v1":
                from tradingagents.research.quant.backtesting.segmented_chain import SegmentedChain

                self.source = SegmentedChain(self.root / "public_events", recover=resume)
                self.frames = SegmentedChain(self.root / "account_frames", recover=resume)
            elif self.protocol.get("storage", "flat-jsonl") == "flat-jsonl":
                self.source = Chain(self.root / "public_events.jsonl", recover=resume)
                self.frames = Chain(self.root / "account_frames.jsonl", recover=resume)
            else:
                raise ValueError("未知研究存储版本")
            self.accounts = {
                str(cost): PaperAccount(
                    self.protocol["symbols"], cost, period_end_ms=self.protocol["simulation_end_ms"]
                )
                for cost in (3, 10)
            }
            self.replayed_committed_tail = 0
            if resume:
                self._restore()
            self.session = {
                "id": uuid.uuid4().hex,
                "pid": os.getpid(),
                "started_ms": time.time_ns() // 1_000_000,
                "resume": resume,
                "initial_frame_count": self.frames.count,
                "committed_tail_replayed": self.replayed_committed_tail,
                "recovered_uncommitted_tails": [
                    v for v in (self.source.recovered_tail, self.frames.recovered_tail) if v
                ],
            }
            self.source.add(
                {
                    "kind": "paper_process_session",
                    "received_ms": self.session["started_ms"],
                    **self.session,
                }
            )
            self.source.seal()
            atomic(self.root / "current_process.json", self.session)
        except BaseException:
            self.close()
            raise

    def _restore(self):
        checkpoint_path = self.root / "checkpoint.json"
        checkpoint = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else None
        count = checkpoint["frame_count"] if checkpoint else 0
        if count > self.frames.count:
            raise ValueError("检查点引用了不存在的已提交账户帧")
        prefixes = {}
        if checkpoint and count == 0 and checkpoint["accounts"] != self.states():
            raise ValueError("空前缀检查点被修改")
        for sequence, record, frame_head in self.frames.iter_entries():
            prefix = record["source_prefix"]
            if (
                prefix["event_count"] in prefixes
                and prefixes[prefix["event_count"]] != prefix["event_chain_sha256"]
            ):
                raise ValueError("同一公开前缀出现不同身份")
            prefixes[prefix["event_count"]] = prefix["event_chain_sha256"]
            for account in self.accounts.values():
                account.apply(record["frame"])
            if sequence + 1 == count and (
                checkpoint["frame_chain_sha256"] != frame_head
                or checkpoint["accounts"] != self.states()
            ):
                raise ValueError("持久检查点不能由原账户帧精确重建")
        if prefixes.get(0) is not None:
            raise ValueError("空公开前缀身份错误")
        verified = {0} if 0 in prefixes else set()
        for sequence, _, source_head in self.source.iter_entries():
            if sequence + 1 in prefixes:
                if prefixes[sequence + 1] != source_head:
                    raise ValueError("账户引用的公开原文前缀已改变")
                verified.add(sequence + 1)
        if verified != set(prefixes):
            raise ValueError("账户帧引用的公开原文尚未持久提交")
        self.replayed_committed_tail = self.frames.count - count
        self.write_checkpoint()

    def states(self):
        return {name: account.checkpoint() for name, account in self.accounts.items()}

    def write_checkpoint(self):
        atomic(
            self.root / "checkpoint.json",
            {
                "frame_count": self.frames.count,
                "frame_chain_sha256": self.frames.head,
                "accounts": self.states(),
            },
        )

    def commit(self, frame):
        """先验证两账户，再同步输入帧，最后共同发布账户状态和检查点。"""
        staged = {name: copy.deepcopy(value) for name, value in self.accounts.items()}
        for value in staged.values():
            value.apply(frame)
        prefix = self.source.seal()
        self.frames.add(
            {
                "frame": frame,
                "source_prefix": prefix,
                "processing_completed_ms": time.time_ns() // 1_000_000,
            }
        )
        self.frames.seal()
        self.accounts = staged
        self.write_checkpoint()

    def close(self):
        for chain in (self.source, self.frames):
            if chain is not None:
                chain.close()
        if not self.lock.closed:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_UN)
            self.lock.close()
