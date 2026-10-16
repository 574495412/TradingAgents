"""将已登记旧缓存复制为独立来源层，不改原全量采集或任何失败记录。"""

from __future__ import annotations

import argparse
import shutil
import time
from collections import Counter
from pathlib import Path

import pandas as pd

from tradingagents.dataflows.research.uniform_inputs import (
    dependency,
    profile,
    resolve_reuse_archive,
)
from tradingagents.research.quant.artifacts import read, save, sha
from tradingagents.research.quant.paths import source_path


def validate_dependencies(items):
    """原全集必须是固定748合约、17日、两类型的完整笛卡尔积。"""
    symbols = {r["symbol"] for r in items}
    days = pd.date_range("2026-09-28", "2026-10-14").strftime("%Y-%m-%d")
    expected = {
        (kind, symbol, day): dependency(kind, symbol, day)
        for symbol in symbols
        for day in days
        for kind in ("metrics", "indexPriceKlines")
    }
    if len(symbols) != 748 or len(items) != 25432 or len(expected) != 25432:
        raise ValueError("缓存层不能缩减原完整总体")
    observed = {(r["kind"], r["symbol"], r["day"]): r for r in items}
    if observed != expected:
        raise ValueError("完整依赖重复或日期、类型、公开URL身份不同")
    return {r["url"]: r for r in items}


def snapshot_archive(root, item, record, output, inputs):
    """核对旧来源后复制实际字节，复制时间不冒充原收据时间。"""
    old = resolve_reuse_archive(root, record, item["kind"])
    receipt = old.with_suffix(".zip.receipt.json")
    check = old.with_suffix(".zip.CHECKSUM")
    for path in (old, receipt, check):
        inputs[str(path)] = sha(path)
    saved = read(receipt)
    if (
        saved["status"] != 200
        or saved["url"] != item["url"]
        or saved["sha256"] != record["sha256"]
        or inputs[str(old)] != record["sha256"]
        or check.read_text().split()[0] != record["sha256"]
    ):
        raise ValueError("旧缓存字节、来源或官方CHECKSUM不同")
    target = output / item["file"]
    target.parent.mkdir(parents=True, exist_ok=True)
    for path, destination in (
        (old, target),
        (receipt, target.with_suffix(".zip.receipt.json")),
        (check, target.with_suffix(".zip.CHECKSUM")),
    ):
        with destination.open("xb") as stream, path.open("rb") as source:
            shutil.copyfileobj(source, stream)
        if sha(destination) != inputs[str(path)]:
            raise ValueError("缓存快照复制期间字节不同")
    checksum_receipt = old.with_suffix(".zip.CHECKSUM.receipt.json")
    if checksum_receipt.exists():
        inputs[str(checksum_receipt)] = sha(checksum_receipt)
        checked = read(checksum_receipt)
        if (
            checked["url"] != item["url"] + ".CHECKSUM"
            or checked["status"] != 200
            or checked["sha256"] != inputs[str(check)]
        ):
            raise ValueError("旧官方CHECKSUM下载收据不同")
        shutil.copyfile(checksum_receipt, target.with_suffix(".zip.CHECKSUM.receipt.json"))
    return item | {
        "status": "VERIFIED_BYTES_AND_PROFILE",
        "sha256": sha(target),
        "path": str(target),
        "profile": profile(target.read_bytes(), item),
        "original_archive": str(old),
        "original_receipt_sha256": inputs[str(receipt)],
        "official_checksum_receipt_present": checksum_receipt.exists(),
        "snapshot_ms": time.time_ns() // 1_000_000,
        "historical_first_publication_time_proven": False,
    }


def run(primary, metrics_reuse, index_reuse, output):
    """只复制原协议登记的完整缓存层，全局未知项依然由原计划约束。"""
    primary, output = Path(primary).resolve(), Path(output).resolve()
    protocol_path = primary / "protocol.json"
    primary_protocol = read(protocol_path)
    if any(
        primary_protocol[key] is not False
        for key in ("leader_labels_read", "orders_enabled", "returns_evaluated")
    ):
        raise ValueError("缓存层只接受原纯公开采集协议")
    lookup = validate_dependencies(primary_protocol["dependencies"])
    manifests = {
        "metrics": Path(metrics_reuse).resolve() / "metrics_manifest.json",
        "indexPriceKlines": Path(index_reuse).resolve() / "manifest.json",
    }
    if any(
        output.is_relative_to(root) for root in (primary, *(p.parent for p in manifests.values()))
    ):
        raise ValueError("快照不能写进原采集或旧缓存目录")
    registered = primary_protocol["reuse_manifest_sha256"]
    if {str(p): sha(p) for p in manifests.values()} != registered:
        raise ValueError("不能添加未登记缓存来源或换掉原缓存清单")
    candidates, urls = [], set()
    for kind, manifest in manifests.items():
        for record in read(manifest)["records"]:
            if record["status"] != "VERIFIED":
                continue
            item = dependency(kind, record["symbol"], record["day"])
            if lookup.get(item["url"]) != item or item["url"] in urls:
                raise ValueError("缓存候选不属于完整总体或发生重复")
            urls.add(item["url"])
            candidates.append({"item": item, "record": record, "root": str(manifest.parent)})
    output.mkdir(parents=True, exist_ok=False)
    (output / "records").mkdir()
    (output / "frozen_inputs").mkdir()
    sources = [
        Path(__file__).resolve(),
        source_path("profit_factor_cache_snapshot_plan.md"),
        Path(dependency.__code__.co_filename).resolve(),
        Path(save.__code__.co_filename).resolve(),
    ]
    frozen = {str(p): sha(p) for p in sources}
    inputs = {str(protocol_path): sha(protocol_path)} | registered
    save(
        output / "protocol.json",
        {
            "schema": "cz-registered-cache-snapshot.v1",
            "created_ms": time.time_ns() // 1_000_000,
            "primary_protocol_sha256": sha(protocol_path),
            "planned_dependencies": len(lookup),
            "candidate_cache_archives": len(candidates),
            "source_sha256": frozen,
            "registered_manifest_sha256": registered,
            "network_requests_enabled": False,
            "returns_evaluated": False,
            "leader_labels_read": False,
            "orders_enabled": False,
        },
    )
    save(output / "candidate_plan.json", candidates)
    for path in [protocol_path, *manifests.values()]:
        shutil.copyfile(
            path,
            output
            / "frozen_inputs"
            / (
                "primary_protocol.json"
                if path == protocol_path
                else path.parent.name + "_" + path.name
            ),
        )
    (output / "source").mkdir()
    for path in sources:
        shutil.copyfile(path, output / "source" / path.name)
    records = []
    for candidate in candidates:
        item = candidate["item"]
        try:
            result = snapshot_archive(candidate["root"], item, candidate["record"], output, inputs)
        except Exception as exc:
            result = item | {
                "status": "FAILED_OR_UNKNOWN",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        save(output / "records" / (str(len(records)).zfill(6) + ".json"), result)
        records.append(result)
    if any(sha(p) != expected for p, expected in (inputs | frozen).items()):
        raise ValueError("快照期间原输入或源码发生变化")
    save(output / "input_identities.json", inputs)
    result = {
        "schema": "cz-registered-cache-snapshot-result.v1",
        "planned_dependencies": len(lookup),
        "candidate_cache_archives": len(candidates),
        "recorded_cache_archives": len(records),
        "statuses": dict(Counter(r["status"] for r in records)),
        "records": records,
        "network_requests": 0,
        "snapshot_finished_ms": time.time_ns() // 1_000_000,
        "primary_collection_modified": False,
        "whole_population_collected": False,
        "historical_publication_times_proven": False,
        "returns_evaluated": False,
        "leader_labels_read": False,
        "orders_enabled": False,
        "production_eligible": False,
    }
    save(output / "result.json", result)
    save(
        output / "receipts.json",
        {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()},
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--metrics-reuse", type=Path, required=True)
    parser.add_argument("--index-reuse", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.primary, args.metrics_reuse, args.index_reuse, args.output)
    print({k: v for k, v in result.items() if k != "records"})


if __name__ == "__main__":
    main()
