"""研究存储转交须保持数据身份，并能在建立别名失败时恢复。"""

from pathlib import Path

import pytest

from tradingagents.research.storage import transfer_roots


def test_transfer_preserves_inode_bytes_and_future_legacy_writes(tmp_path):
    old, new = tmp_path / "cz" / "study", tmp_path / "ta" / "study"
    old.mkdir(parents=True)
    source = old / "raw.bin"
    source.write_bytes("原始研究数据".encode())
    identity = source.stat().st_ino
    receipt = transfer_roots([(old, new)])
    assert old.is_symlink() and old.resolve() == new
    assert (new / "raw.bin").stat().st_ino == identity
    assert (new / "raw.bin").read_bytes() == "原始研究数据".encode()
    (old / "new_event.json").write_text("{}")
    assert (new / "new_event.json").read_text() == "{}"
    assert receipt[0]["file_bytes_rewritten"] is False


def test_alias_failure_rolls_back_every_moved_directory(tmp_path, monkeypatch):
    old_a, old_b = tmp_path / "old_a", tmp_path / "old_b"
    for p in (old_a, old_b):
        p.mkdir()
        (p / "raw").write_text(p.name)
    original = Path.symlink_to

    def fail_second(path, *args, **kwargs):
        if path == old_b:
            raise OSError("人工别名创建失败")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "symlink_to", fail_second)
    new_a, new_b = tmp_path / "new_a", tmp_path / "new_b"
    with pytest.raises(OSError, match="人工别名"):
        transfer_roots([(old_a, new_a), (old_b, new_b)])
    assert all(
        p.is_dir() and not p.is_symlink() and (p / "raw").read_text() == p.name
        for p in (old_a, old_b)
    )
    assert not new_a.exists() and not new_b.exists()


@pytest.mark.parametrize("damage", ["occupied", "nested", "duplicate"])
def test_preflight_rejects_ambiguous_or_destructive_targets(tmp_path, damage):
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    entries = [(old, new)]
    if damage == "occupied":
        new.mkdir()
    elif damage == "nested":
        entries = [(old, old / "inside")]
    else:
        entries *= 2
    with pytest.raises(ValueError):
        transfer_roots(entries)
    assert old.is_dir() and not old.is_symlink()
