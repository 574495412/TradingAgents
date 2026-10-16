"""迁移研究存储的所有权，旧路径仅保留别名，历史文件字节不重写。"""

import os
from pathlib import Path


def transfer_roots(entries):
    """在同一文件系统移动整目录；任一步失败就恢复本次已移动的目录。"""
    pairs = [(Path(old).absolute(), Path(new).absolute()) for old, new in entries]
    if len({old for old, _ in pairs}) != len(pairs) or len({new for _, new in pairs}) != len(pairs):
        raise ValueError("研究目录迁移存在重复来源或目标")
    for old, new in pairs:
        if old.is_symlink() or not old.is_dir() or new.exists() or new.is_symlink():
            raise ValueError("来源须为真实目录，目标须不存在")
        if new.is_relative_to(old) or old.is_relative_to(new):
            raise ValueError("迁移来源与目标不能相互包含")
        if any(
            old != other and (old.is_relative_to(other) or other.is_relative_to(old))
            for other, _ in pairs
        ):
            raise ValueError("不能同时迁移互相嵌套的来源")
        new.parent.mkdir(parents=True, exist_ok=True)
        if old.stat().st_dev != new.parent.stat().st_dev:
            raise ValueError("研究存储转交要求同一文件系统，避免复制一半后切换")
    moved, receipts = [], []
    try:
        for old, new in pairs:
            identity = old.stat()
            os.rename(old, new)
            moved.append((old, new))
            old.symlink_to(new, target_is_directory=True)
            after = new.stat()
            if (identity.st_dev, identity.st_ino) != (
                after.st_dev,
                after.st_ino,
            ) or old.resolve() != new.resolve():
                raise ValueError("迁移后的目录身份或旧路径别名不一致")
            receipts.append(
                {
                    "old_alias": str(old),
                    "owner_path": str(new),
                    "device": after.st_dev,
                    "inode": after.st_ino,
                    "directory_inode_preserved": True,
                    "file_bytes_rewritten": False,
                }
            )
    except BaseException:
        for old, new in reversed(moved):
            if old.is_symlink() and old.resolve() == new.resolve():
                old.unlink()
            if old.exists() or old.is_symlink():
                raise RuntimeError("旧路径已被其他写入占用；保留新目录，停止自动恢复") from None
            os.rename(new, old)
        raise
    return receipts
