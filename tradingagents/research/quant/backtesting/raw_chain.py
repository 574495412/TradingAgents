"""只核验封存原文字节；提供者返回坏JSON也必须能够原样保留。"""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
from pathlib import Path

from tradingagents.dataflows.research.live_capture import digest
from tradingagents.research.quant.backtesting.segmented_chain import segments

VERSION = "raw-byte-integrity-v1"


def raw_entries(root):
    """不把行情有效性混入日志完整性；消费原文时仍需单独验证JSON和字段。"""
    root = Path(root)
    count, head = 0, None
    for segment in segments(root):
        decoded, decoded_bytes = hashlib.sha256(), 0
        with gzip.open(root / segment["file"], "rb") as stream:
            for line in stream:
                decoded.update(line)
                decoded_bytes += len(line)
                if not line.endswith(b"\n"):
                    raise ValueError("原文段存在未提交的半行")
                entry = json.loads(line)
                if entry["sequence"] != count or entry["previous_sha256"] != head:
                    raise ValueError("原文事件校验链不连续")
                record = entry["record"]
                if "body_base64" in record:
                    raw = base64.b64decode(record["body_base64"], validate=True)
                    if hashlib.sha256(raw).hexdigest() != record["body_sha256"]:
                        raise ValueError("原文字节与接收身份不一致")
                head = digest(entry)
                yield count, record, head
                count += 1
        if (
            count != segment["end_count"]
            or head != segment["event_chain_sha256"]
            or decoded.hexdigest() != segment["decoded_sha256"]
            or decoded_bytes != segment["decoded_bytes"]
        ):
            raise ValueError("原文段完整字节或终点身份错误")
