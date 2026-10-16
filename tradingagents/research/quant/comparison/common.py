"""研究文件的规范编码、哈希及只写一次保存。"""

import hashlib
import json
from pathlib import Path


def encoded(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(
        Path(path).read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("JSON 不允许非有限数值")),
    )


def save(path, value):
    """使用排他创建，已有证据文件不可被重写。"""
    with Path(path).open("xb") as stream:
        stream.write(encoded(value) + b"\n")


def seal(path, value):
    """恢复时只接受内容完全相同的已有文件，不覆盖旧结果。"""
    path = Path(path)
    if path.exists():
        if digest(read(path)) != digest(value):
            raise ValueError("恢复产生了不同的封存内容")
    else:
        save(path, value)


def source_identity():
    root = Path(__file__).parent
    return digest({p.name: file_hash(p) for p in sorted(root.glob("*.py"))})
