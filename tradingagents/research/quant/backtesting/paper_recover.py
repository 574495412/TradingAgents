"""仅迁移失败响应的日志读取校验，不改变冻结策略、账户或原结束时间。"""

from __future__ import annotations

import argparse
import ast
import asyncio
import time
import uuid
from pathlib import Path

from tradingagents.dataflows.research.public_http import PublicHTTP
from tradingagents.research.quant.backtesting.paper_live import live
from tradingagents.research.quant.backtesting.paper_store import StudyStore
from tradingagents.research.quant.strategy_audit import read, save, sha


def transport_patch(root):
    """除日志读取函数体以外源码必须相同，不能借恢复修改交易语义。"""
    protocol = read(root / "protocol.json")
    changes = []
    for filename, expected in protocol["source_sha256"].items():
        current, archived = Path(filename), root / "source" / Path(filename).name
        if sha(archived) != expected:
            raise ValueError("原源码归档已经改变")
        if sha(current) != expected:
            changes.append(current)
    if len(changes) != 1 or changes[0].name != "paper_store.py":
        raise ValueError("技术恢复只允许日志读取边界的单文件修复")
    current = changes[0]
    old = ast.parse((root / "source" / current.name).read_text())
    new = ast.parse(current.read_text())
    for module in (old, new):
        functions = [
            n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "entries"
        ]
        if len(functions) != 1:
            raise ValueError("日志读取函数身份不同")
        functions[0].body = [ast.Pass()]
    if ast.dump(old) != ast.dump(new):
        raise ValueError("修复改动超出了日志读取函数体")
    return {
        "original_store_sha256": protocol["source_sha256"][str(current)],
        "recovered_store_sha256": sha(current),
        "only_transport_reader_body_changed": True,
        "all_policy_and_account_sources_unchanged": True,
    }


def run(root):
    """原协议不覆盖，以追加修复身份恢复同一账本，并记录精确前缀核验。"""
    patch, protocol = transport_patch(root), read(root / "protocol.json")
    before = read(root / "checkpoint.json")
    identity = uuid.uuid4().hex
    directory = root / f"transport_recovery_{identity}"
    directory.mkdir(exist_ok=False)
    (directory / "reader_source.py").write_bytes(
        Path(StudyStore.commit.__code__.co_filename).read_bytes()
    )
    (directory / "recovery_source.py").write_bytes(Path(__file__).read_bytes())
    manifest = {
        "schema": "cz-paper-transport-reader-recovery.v1",
        "created_ms": time.time_ns() // 1_000_000,
        "original_protocol_sha256": sha(root / "protocol.json"),
        "simulation_end_ms_unchanged": protocol["simulation_end_ms"],
        "recovery_entry_source_sha256": sha(Path(__file__)),
        "before_checkpoint_sha256": sha(root / "checkpoint.json"),
        **patch,
        "orders_enabled": False,
    }
    save(directory / "manifest.json", manifest)
    store = StudyStore(root, protocol, resume=True)
    client = PublicHTTP()
    try:
        save(
            directory / "restoration.json",
            {
                "original_checkpoint_prefix_verified": True,
                "full_account_states_equal_before_new_inputs": store.states() == before["accounts"],
                "committed_tail_replayed": store.replayed_committed_tail,
                "restored_frame_count": store.frames.count,
                "simulation_end_ms": store.protocol["simulation_end_ms"],
                "new_pid": store.session["pid"],
            },
        )
        store.source.add(
            {
                "kind": "transport_reader_recovery",
                "received_ms": time.time_ns() // 1_000_000,
                "recovery_manifest_sha256": sha(directory / "manifest.json"),
                **patch,
            }
        )
        result = asyncio.run(live(store, client))
        save(root / f"result_{store.session['id']}.json", result)
        store.write_checkpoint()
        save(
            directory / "receipts.json",
            {p.name: sha(p) for p in directory.iterdir() if p.is_file()},
        )
        print({k: v for k, v in result.items() if k not in ("accounts", "stream")}, flush=True)
        return result
    finally:
        client.close()
        store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.source.resolve())
    if result["failure"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
