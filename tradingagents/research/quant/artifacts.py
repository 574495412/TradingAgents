"""量化研究的独立文件工具；来源只读，结果只写一次。"""

import hashlib
import json
from pathlib import Path


def read(path):
    """未知与非有限数值不能悄悄变成有效研究数据。"""

    def invalid(value):
        raise ValueError(f"JSON 不允许非有限数值：{value}")

    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=invalid)


def sha(path):
    """流式读取原始字节身份，避免把大归档一次装入内存。"""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    """排他创建结果，已有实验或失败记录不得被覆盖。"""
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2)


def verify_receipts(root):
    """摘要覆盖整个目录，拒绝目录逃逸、额外文件、软链接和字节变化。"""
    root = Path(root).resolve()
    receipts = read(root / "receipts.json")
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()} - {"receipts.json"}
    if not isinstance(receipts, dict) or actual != set(receipts):
        raise ValueError("研究输入的封存文件集合发生变化")
    for name, checksum in receipts.items():
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root) or sha(path) != checksum:
            raise ValueError("研究输入的封存身份发生变化")
