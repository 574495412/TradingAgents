"""全量计划不变地合并主采集与已登记缓存，保留原失败、等待和冲突。"""

from __future__ import annotations

import argparse
import hashlib
import time
from collections import Counter
from pathlib import Path

from tradingagents.dataflows.research.cache_snapshot import validate_dependencies
from tradingagents.dataflows.research.uniform_inputs import dependency
from tradingagents.research.quant.artifacts import read, save, sha
from tradingagents.research.quant.paths import source_path

VERIFIED = "VERIFIED_BYTES_AND_PROFILE"
CACHE_PATH_ERROR = "旧归档未通过官方字节和来源校验"


def choose_source(primary, cache):
    """仅按预定来源状态选择，字节冲突和404矛盾不能用缓存悄悄覆盖。"""
    old_status = primary["status"] if primary else "PRIMARY_NOT_YET_RECORDED"
    known_cache = cache is not None and cache["status"] == VERIFIED
    if primary and old_status == VERIFIED:
        if known_cache and primary["sha256"] != cache["sha256"]:
            return "CONFLICTING_SUCCESSFUL_BYTES", None
        return VERIFIED, "primary"
    if primary and old_status == "PUBLIC_404":
        return (
            ("PUBLIC_404_CONFLICTS_WITH_CACHE", None) if known_cache else ("PUBLIC_404", "primary")
        )
    eligible = (
        primary is None
        or old_status == "NOT_FETCHED_AFTER_STOP"
        or (old_status == "FAILED_OR_UNKNOWN" and primary.get("error") == CACHE_PATH_ERROR)
    )
    if known_cache and eligible:
        return VERIFIED, "registered_cache"
    return old_status, None


def verify_seal(root):
    """封存来源必须覆盖实际文件集合，不能依靠存在一个摘要文件来宣称完成。"""
    receipts = read(root / "receipts.json")
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()} - {"receipts.json"}
    if actual != set(receipts) or any(
        sha(root / name) != expected for name, expected in receipts.items()
    ):
        raise ValueError("合并来源的封存文件集合或字节不同")


def check_record(root, record, inputs):
    """实际原文与收据必须同URL、同字节；官方404也需要其真实下载收据。"""
    path = root / record["file"]
    receipt = path.with_suffix(".zip.receipt.json")
    inputs[str(receipt)] = sha(receipt)
    saved = read(receipt)
    if saved["url"] != record["url"]:
        raise ValueError("来源记录的公开URL与收据不同")
    if record["status"] == "PUBLIC_404":
        if saved["status"] != 404 or path.exists():
            raise ValueError("404状态缺少对应收据或却有成功档案")
        return
    checksum = path.with_suffix(".zip.CHECKSUM")
    inputs.update({str(p): sha(p) for p in (path, checksum)})
    if (
        saved["status"] != 200
        or inputs[str(path)] != record["sha256"]
        or saved["sha256"] != record["sha256"]
        or checksum.read_text().split()[0] != record["sha256"]
    ):
        raise ValueError("成功来源的字节或官方CHECKSUM不同")


def run(primary, cache, output, allow_incomplete=False):
    """输出每项固定依赖及其来源，主采集前缀不能因为缓存已封存而冒充终态。"""
    primary, cache, output = (Path(p).resolve() for p in (primary, cache, output))
    if output.is_relative_to(primary) or output.is_relative_to(cache):
        raise ValueError("合并视图不能写进主采集或缓存快照")
    primary_protocol, cache_protocol = (
        read(primary / "protocol.json"),
        read(cache / "protocol.json"),
    )
    if any(
        p[key] is not False
        for p in (primary_protocol, cache_protocol)
        for key in ("orders_enabled", "leader_labels_read", "returns_evaluated")
    ):
        raise ValueError("只合并纯公开且不计算收益的来源")
    lookup = validate_dependencies(primary_protocol["dependencies"])
    if cache_protocol["primary_protocol_sha256"] != sha(primary / "protocol.json"):
        raise ValueError("缓存快照不属于同一完整采集协议")
    verify_seal(cache)
    terminal = (primary / "receipts.json").exists() and (primary / "result.json").exists()
    if terminal:
        verify_seal(primary)
    elif not allow_incomplete:
        raise ValueError("原主采集尚未终态，必须显式选择前缀诊断")
    record_paths = sorted((primary / "records").glob("*.json"))
    primary_rows = [read(p) for p in record_paths]
    if any(
        path.stem != hashlib.sha256(row["url"].encode()).hexdigest()
        for path, row in zip(record_paths, primary_rows, strict=True)
    ):
        raise ValueError("主采集记录文件名与URL身份不同")
    cache_result = read(cache / "result.json")
    cache_rows = cache_result["records"]
    if (
        len(cache_rows) != cache_result["recorded_cache_archives"]
        or len(cache_rows) != cache_protocol["candidate_cache_archives"]
        or cache_rows != [read(p) for p in sorted((cache / "records").glob("*.json"))]
    ):
        raise ValueError("缓存快照结果与全部逐项文件不同")
    maps = []
    for rows in (primary_rows, cache_rows):
        values = {r["url"]: r for r in rows}
        if len(values) != len(rows) or any(
            any(record.get(key) != value for key, value in lookup[url].items())
            for url, record in values.items()
        ):
            raise ValueError("来源项重复或不属于固定总体")
        maps.append(values)
    primary_by_url, cache_by_url = maps
    if terminal and (
        len(primary_rows) != 25432
        or read(primary / "result.json")["records"]
        != sorted(primary_rows, key=lambda r: (r["symbol"], r["day"], r["kind"]))
    ):
        raise ValueError("原主采集终态结果与实际逐项记录不同")
    inputs = {str(p): sha(p) for p in record_paths}
    inputs.update(
        {
            str(p): sha(p)
            for p in (
                primary / "protocol.json",
                cache / "protocol.json",
                cache / "result.json",
                cache / "receipts.json",
            )
        }
    )
    sources = [
        Path(__file__).resolve(),
        Path(validate_dependencies.__code__.co_filename).resolve(),
        Path(dependency.__code__.co_filename).resolve(),
        Path(save.__code__.co_filename).resolve(),
        source_path("profit_factor_cache_snapshot_plan.md"),
    ]
    frozen = {str(p): sha(p) for p in sources}
    output.mkdir(parents=True, exist_ok=False)
    (output / "source").mkdir()
    for path in sources:
        (output / "source" / path.name).write_bytes(path.read_bytes())
    save(
        output / "protocol.json",
        {
            "schema": "cz-uniform-public-input-view.v1",
            "created_ms": time.time_ns() // 1_000_000,
            "primary_protocol_sha256": sha(primary / "protocol.json"),
            "cache_receipts_sha256": sha(cache / "receipts.json"),
            "primary_collection_terminal": terminal,
            "explicit_prefix_diagnostic": allow_incomplete,
            "planned_dependencies": 25432,
            "snapshotted_primary_records": len(primary_rows),
            "selection_policy": "EXACT_SOURCE_STATUS_ONLY_CONFLICTS_RETAINED_V1",
            "source_sha256": frozen,
            "returns_evaluated": False,
            "orders_enabled": False,
            "leader_labels_read": False,
        },
    )
    for root, rows in ((primary, primary_rows), (cache, cache_rows)):
        for record in rows:
            if record["status"] in {VERIFIED, "PUBLIC_404"}:
                check_record(root, record, inputs)
    records = []
    for url, item in lookup.items():
        old, cached = primary_by_url.get(url), cache_by_url.get(url)
        status, selected = choose_source(old, cached)
        used = old if selected == "primary" else cached if selected == "registered_cache" else None
        record = item | {
            "status": status,
            "selected_source": selected,
            "original_primary_status": old["status"] if old else "PRIMARY_NOT_YET_RECORDED",
            "original_primary_error": old.get("error") if old else None,
            "registered_cache_status": cached["status"] if cached else "NOT_IN_REGISTERED_CACHE",
        }
        if used and status == VERIFIED:
            record.update(
                sha256=used["sha256"],
                archive_path=str((primary if selected == "primary" else cache) / item["file"]),
                profile=used["profile"],
            )
        records.append(record)
    if any(sha(path) != expected for path, expected in (inputs | frozen).items()):
        raise ValueError("合并期间原来源或源码发生变化")
    save(output / "input_identities.json", inputs)
    result = {
        "schema": "cz-uniform-public-input-view-result.v1",
        "planned_dependencies": 25432,
        "primary_collection_terminal": terminal,
        "snapshotted_primary_records": len(primary_rows),
        "effective_statuses": dict(Counter(r["status"] for r in records)),
        "selected_sources": dict(Counter(r["selected_source"] or "none" for r in records)),
        "cache_repairs_of_original_path_failures": sum(
            r["selected_source"] == "registered_cache"
            and r["original_primary_status"] == "FAILED_OR_UNKNOWN"
            for r in records
        ),
        "cache_while_primary_not_yet_recorded": sum(
            r["selected_source"] == "registered_cache"
            and r["original_primary_status"] == "PRIMARY_NOT_YET_RECORDED"
            for r in records
        ),
        "records": records,
        "network_requests": 0,
        "whole_population_coverage_approved": False,
        "returns_evaluated": False,
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
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    result = run(args.primary, args.cache, args.output, args.allow_incomplete)
    print({k: v for k, v in result.items() if k != "records"})


if __name__ == "__main__":
    main()
