"""用封存分钟基础量复核metrics流量时钟，空值和冲突保留，不计算收益。"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import math
import time
import zipfile
from collections import Counter
from decimal import Decimal
from pathlib import Path

import numpy as np

from tradingagents.dataflows.research.cache_snapshot import validate_dependencies
from tradingagents.dataflows.research.uniform_view import verify_seal
from tradingagents.research.quant.artifacts import read, save, sha
from tradingagents.research.quant.paths import source_path

MINUTE = 60_000
FLOW = "sum_taker_long_short_vol_ratio"
OFFSETS = {"preceding_five_minutes": -5, "following_five_minutes": 0}


def source_records(source, allow_incomplete):
    """封存视图沿用原主采集的终态；不能因为视图有收据便宣称全量完成。"""
    source = Path(source)
    if not (source / "protocol.json").exists() and not allow_incomplete:
        raise ValueError("采集尚未终态，前缀诊断必须明确授权选项")
    declared = read(source / "protocol.json")
    if declared.get("schema") not in {
        "cz-uniform-public-input-view.v1",
        "cz-unconditional-public-factor-inputs.v1",
    } or any(
        declared[key] is not False
        for key in ("leader_labels_read", "orders_enabled", "returns_evaluated")
    ):
        raise ValueError("审计只接受已登记纯公开输入协议")
    is_view = declared.get("schema") == "cz-uniform-public-input-view.v1"
    complete = (
        declared["primary_collection_terminal"]
        if is_view
        else ((source / "receipts.json").exists() and (source / "result.json").exists())
    )
    if type(complete) is not bool:
        raise ValueError("主采集终态必须为明确布尔值")
    if not complete and not allow_incomplete:
        raise ValueError("采集尚未终态，前缀诊断必须明确授权选项")
    files = [source / "protocol.json"]
    if is_view:
        verify_seal(source)
        result = read(source / "result.json")
        if result["primary_collection_terminal"] is not complete:
            raise ValueError("视图协议与结果的主采集终态不同")
        parent = read(source / "input_identities.json")
        candidates = [
            Path(p)
            for p, expected in parent.items()
            if Path(p).name == "protocol.json" and expected == declared["primary_protocol_sha256"]
        ]
        if len(candidates) != 1 or sha(candidates[0]) != declared["primary_protocol_sha256"]:
            raise ValueError("视图不能定位同一完整采集协议")
        protocol = read(candidates[0])
        records = result["records"]
        if len(records) != 25432 or (
            complete
            and any(r["original_primary_status"] == "PRIMARY_NOT_YET_RECORDED" for r in records)
        ):
            raise ValueError("视图遗漏原总体或把未记录项冒充终态")
        files.extend(
            [
                source / "result.json",
                source / "receipts.json",
                source / "input_identities.json",
                candidates[0],
            ]
        )
        if complete:
            primary_root = candidates[0].parent
            verify_seal(primary_root)
            original = {r["url"]: r for r in read(primary_root / "result.json")["records"]}
            if len(original) != 25432 or any(
                original[r["url"]]["status"] != r["original_primary_status"] for r in records
            ):
                raise ValueError("视图终态与原主采集逐项状态不同")
            files.extend([primary_root / "result.json", primary_root / "receipts.json"])
        for record in records:
            if record["status"] == "VERIFIED_BYTES_AND_PROFILE":
                archive = record["archive_path"]
                if parent.get(archive) != record["sha256"]:
                    raise ValueError("视图成功原文不属于原封存输入")
    else:
        protocol = declared
        paths = sorted((source / "records").glob("*.json"))
        records = [read(path) for path in paths]
        files.extend(paths)
        if complete:
            verify_seal(source)
            files.extend([source / "receipts.json", source / "result.json"])
            if len(records) != 25432 or read(source / "result.json")["records"] != sorted(
                records, key=lambda r: (r["symbol"], r["day"], r["kind"])
            ):
                raise ValueError("终态记录不完整或与逐项原文不同")
    lookup = validate_dependencies(protocol["dependencies"])
    if len({r["url"] for r in records}) != len(records) or any(
        any(row.get(k) != value for k, value in lookup[row["url"]].items()) for row in records
    ):
        raise ValueError("输入记录重复或与完整依赖身份不同")
    return protocol, records, complete, {str(p): sha(p) for p in files}


def timestamp(value):
    """无时区档案明确按UTC解释，已有偏移必须为UTC，不能静默抹去。"""
    value = dt.datetime.fromisoformat(value)
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.UTC)
    if value.utcoffset() != dt.timedelta(0) or value.microsecond:
        raise ValueError("持仓档案时钟不是整数秒UTC")
    return int(value.timestamp()) * 1000


def parse_metrics(payload, symbol, day):
    """保留原顺序和缺值；不把只有零持仓、空比率的日档案当成完整特征。"""
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if len(archive.namelist()) != 1 or archive.infolist()[0].file_size > 64 * 1024**2:
            raise ValueError("持仓档案大小或文件数异常")
        rows = list(
            csv.reader(io.StringIO(archive.read(archive.namelist()[0]).decode("utf-8-sig")))
        )
    header = rows[0]
    if len(set(header)) != len(header) or not {
        "create_time",
        "symbol",
        "sum_open_interest",
        FLOW,
    } <= set(header):
        raise ValueError("持仓字段身份异常")
    first, parsed, missing = timestamp(day), [], Counter()
    for fields in rows[1:]:
        row = dict(zip(header, fields, strict=True))
        clock = timestamp(row.pop("create_time"))
        if row.pop("symbol") != symbol or not first <= clock <= first + 86_400_000:
            raise ValueError("持仓合约或日期不同")
        values = {}
        for name, value in row.items():
            values[name] = Decimal(value) if value else None
            missing[name] += int(not value)
            if values[name] is not None and (not math.isfinite(values[name]) or values[name] < 0):
                raise ValueError("持仓或比率存在非有限、负值")
        if values["sum_open_interest"] is None:
            raise ValueError("基础持仓量缺失")
        parsed.append((clock, values))
    stamps = [clock for clock, _ in parsed]
    unique = set(stamps)
    return parsed, {
        "rows": len(parsed),
        "missing_by_column": dict(missing),
        "zero_open_interest_rows": sum(values["sum_open_interest"] == 0 for _, values in parsed),
        "duplicate_timestamp_rows": len(stamps) - len(unique),
        "nonincreasing_steps_in_original_order": sum(
            b <= a for a, b in zip(stamps, stamps[1:], strict=False)
        ),
        "off_five_minute_grid_rows": sum(clock % (5 * MINUTE) != 0 for clock in stamps),
        "missing_points_on_five_minute_reference_grid": len(
            set(range(first, first + 86_400_000, 5 * MINUTE)) - unique
        ),
        "next_day_boundary_rows": sum(clock == first + 86_400_000 for clock in stamps),
    }


def parse_index(payload, day):
    """指数收盘按完整分钟边界记键，高低开收必须守恒，重复仍保留。"""
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if len(archive.namelist()) != 1 or archive.infolist()[0].file_size > 64 * 1024**2:
            raise ValueError("指数档案大小或文件数异常")
        rows = list(
            csv.reader(io.StringIO(archive.read(archive.namelist()[0]).decode("utf-8-sig")))
        )
    if not rows:
        raise ValueError("指数档案为空")
    if rows[0][0] in {"open_time", "Open time"}:
        rows = rows[1:]
    first, parsed = timestamp(day), []
    for fields in rows:
        if len(fields) != 12 or not fields[0].isdigit():
            raise ValueError("指数时钟或列数异常")
        opening = int(fields[0])
        if (
            opening % MINUTE
            or int(fields[6]) != opening + MINUTE - 1
            or not first <= opening < first + 86_400_000
        ):
            raise ValueError("指数分钟结束时钟或日期异常")
        o, h, low, c = (Decimal(fields[k]) for k in (1, 2, 3, 4))
        if (
            any(not math.isfinite(v) or v <= 0 for v in (o, h, low, c))
            or not low <= min(o, c) <= max(o, c) <= h
        ):
            raise ValueError("指数价格或高低开收约束异常")
        parsed.append((opening + MINUTE, (o, h, low, c)))
    stamps = [clock for clock, _ in parsed]
    return parsed, {
        "rows": len(parsed),
        "duplicate_timestamp_rows": len(stamps) - len(set(stamps)),
        "nonincreasing_steps_in_original_order": sum(
            b <= a for a, b in zip(stamps, stamps[1:], strict=False)
        ),
        "missing_points_on_minute_reference_grid": len(
            set(range(first + MINUTE, first + 86_400_000 + MINUTE, MINUTE)) - set(stamps)
        ),
    }


def minute_volumes(raw, symbol):
    """只取公开基础成交量，不读取价格目标或用缺失分钟拼接窗口。"""
    if raw["symbol"] != symbol:
        raise ValueError("分钟基础量合约身份不同")
    result = {}
    for row in raw["rows"]:
        opening = row[0]
        if (
            len(row) != 12
            or type(opening) is not int
            or opening % MINUTE
            or row[6] != opening + MINUTE - 1
        ):
            raise ValueError("分钟基础量时钟异常")
        total, buy = float(row[5]), float(row[9])
        if not math.isfinite(total) or not math.isfinite(buy) or not 0 <= buy <= total:
            raise ValueError("分钟买卖基础量不守恒")
        if opening in result:
            raise ValueError("分钟基础量时间键重复")
        result[opening] = total, buy
    return result


def ratio(volumes, clock, offset):
    """按原开盘键精确取五分钟，卖量为零或有任一分钟缺失均未知。"""
    keys = [clock + (offset + k) * MINUTE for k in range(5)]
    if not all(key in volumes for key in keys):
        return None
    buy = math.fsum(volumes[key][1] for key in keys)
    sell = math.fsum(volumes[key][0] - volumes[key][1] for key in keys)
    return buy / sell if sell > 0 else None


def summarize(pairs):
    """所有解释使用相同已知配对人口，不因某一窗口缺失扩大另一解释覆盖。"""
    common = [
        p
        for p in pairs
        if p["archive_ratio"] is not None and all(p[name] is not None for name in OFFSETS)
    ]
    summaries = {}
    for name in OFFSETS:
        observed = np.asarray([p["archive_ratio"] for p in common])
        rebuilt = np.asarray([p[name] for p in common])
        summaries[name] = {
            "common_rows": len(common),
            "close_rows": sum(
                math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-8)
                for a, b in zip(observed, rebuilt, strict=True)
            ),
            "median_absolute_error": float(np.median(np.abs(observed - rebuilt)))
            if len(common)
            else None,
            "pearson_correlation": float(np.corrcoef(observed, rebuilt)[0, 1])
            if len(common) > 1 and observed.std() > 0 and rebuilt.std() > 0
            else None,
        }
    return {"scheduled_rows": len(pairs), "common_rows": len(common), "interpretations": summaries}


def run(source, minute_study, output, allow_incomplete=False):
    """固定已落盘项清单；完整终态或显式前缀均单独封存，绝不改原采集。"""
    source, minute_study, output = (Path(p).resolve() for p in (source, minute_study, output))
    protocol, records, complete, inputs = source_records(source, allow_incomplete)
    if protocol["leader_labels_read"] is not False or protocol["orders_enabled"] is not False:
        raise ValueError("不能读取带单或交易输入")
    identities = read(minute_study / "input_identities.json")
    minute_files = {(Path(p).parent.name, Path(p).stem): Path(p) for p in identities}
    plan = source_path("profit_factor_metrics_clock_plan.md")
    output.mkdir(parents=True, exist_ok=False)
    frozen = {
        str(p): sha(p)
        for p in (
            Path(__file__).resolve(),
            plan,
            Path(save.__code__.co_filename).resolve(),
            Path(validate_dependencies.__code__.co_filename).resolve(),
            Path(verify_seal.__code__.co_filename).resolve(),
            Path(__file__).with_name("uniform_inputs.py"),
        )
    }
    inputs.update(
        {
            str(source / "protocol.json"): sha(source / "protocol.json"),
            str(minute_study / "input_identities.json"): sha(
                minute_study / "input_identities.json"
            ),
        }
    )
    save(
        output / "protocol.json",
        {
            "created_ms": time.time_ns() // 1_000_000,
            "schema": "cz-metrics-clock-audit.v1",
            "collection_terminal": complete,
            "partial_prefix_explicit": allow_incomplete,
            "planned_dependencies": 25432,
            "snapshotted_record_count": len(records),
            "source_sha256": frozen,
            "fixed_clock_offsets_minutes": OFFSETS,
            "relative_tolerance": 1e-6,
            "absolute_tolerance": 1e-8,
            "returns_evaluated": False,
            "leader_labels_read": False,
            "orders_enabled": False,
        },
    )
    for name in frozen:
        (output / Path(name).name).write_bytes(Path(name).read_bytes())
    pairs, profiles, conflicts, seen, active_symbol = [], [], [], {}, None
    metrics = sorted(
        (
            r
            for r in records
            if r["kind"] == "metrics" and r["status"] == "VERIFIED_BYTES_AND_PROFILE"
        ),
        key=lambda r: (r["symbol"], r["day"]),
    )
    for record in metrics:
        symbol, day = record["symbol"], record["day"]
        if symbol != active_symbol:
            seen, active_symbol = {}, symbol
        path = Path(record.get("archive_path", source / record["file"]))
        check = path.with_suffix(".zip.CHECKSUM")
        receipt = path.with_suffix(".zip.receipt.json")
        if sha(path) != record["sha256"] or check.read_text().split()[0] != record["sha256"]:
            raise ValueError("持仓档案字节与官方CHECKSUM不同")
        saved = read(receipt)
        if (
            saved["status"] != 200
            or saved["url"] != record["url"]
            or saved["sha256"] != record["sha256"]
        ):
            raise ValueError("持仓下载来源不同")
        inputs.update({str(p): sha(p) for p in (path, check, receipt)})
        parsed, quality = parse_metrics(path.read_bytes(), symbol, day)
        profiles.append({"symbol": symbol, "day": day, **quality})
        volumes = {}
        for date in ((dt.date.fromisoformat(day) - dt.timedelta(days=1)).isoformat(), day):
            raw = minute_files.get((symbol, date))
            if raw is None:
                continue
            if sha(raw) != identities[str(raw)]:
                raise ValueError("封存分钟字节不同")
            values = minute_volumes(read(raw), symbol)
            if set(volumes) & set(values):
                raise ValueError("跨日分钟时间键重叠")
            volumes.update(values)
            inputs[str(raw)] = identities[str(raw)]
        for clock, values in parsed:
            if clock in seen:
                previous_day, previous_values = seen[clock]
                conflicts.append(
                    {
                        "symbol": symbol,
                        "clock_ms": clock,
                        "first_day": previous_day,
                        "other_day": day,
                        "same_values": previous_values == values,
                    }
                )
            else:
                seen[clock] = day, values
            pairs.append(
                {
                    "symbol": symbol,
                    "day": day,
                    "clock_ms": clock,
                    "archive_ratio": float(values[FLOW]) if values[FLOW] is not None else None,
                    **{name: ratio(volumes, clock, offset) for name, offset in OFFSETS.items()},
                }
            )
    index_profiles, index_conflicts, seen, active_symbol = [], [], {}, None
    indices = sorted(
        (
            r
            for r in records
            if r["kind"] == "indexPriceKlines" and r["status"] == "VERIFIED_BYTES_AND_PROFILE"
        ),
        key=lambda r: (r["symbol"], r["day"]),
    )
    for record in indices:
        symbol, day = record["symbol"], record["day"]
        if symbol != active_symbol:
            seen, active_symbol = {}, symbol
        path = Path(record.get("archive_path", source / record["file"]))
        check, receipt = path.with_suffix(".zip.CHECKSUM"), path.with_suffix(".zip.receipt.json")
        saved = read(receipt)
        if (
            sha(path) != record["sha256"]
            or saved["status"] != 200
            or saved["url"] != record["url"]
            or saved["sha256"] != record["sha256"]
            or check.read_text().split()[0] != record["sha256"]
        ):
            raise ValueError("指数档案字节、来源或官方CHECKSUM不同")
        inputs.update({str(p): sha(p) for p in (path, check, receipt)})
        parsed, quality = parse_index(path.read_bytes(), day)
        index_profiles.append({"symbol": symbol, "day": day, **quality})
        for clock, values in parsed:
            if clock in seen:
                previous_day, previous_values = seen[clock]
                index_conflicts.append(
                    {
                        "symbol": symbol,
                        "clock_ms": clock,
                        "first_day": previous_day,
                        "other_day": day,
                        "same_values": previous_values == values,
                    }
                )
            else:
                seen[clock] = day, values
    if any(sha(path) != expected for path, expected in (inputs | frozen).items()):
        raise ValueError("核验期间输入或源码发生变化")
    summary = summarize(pairs)
    result = {
        "schema": "cz-metrics-clock-quality-result.v1",
        "collection_terminal": complete,
        "snapshotted_records": len(records),
        "record_statuses": dict(Counter(r["status"] for r in records)),
        "planned_dependencies": 25432,
        "unrecorded_dependencies": 25432
        - len(records)
        + sum(r["status"] == "PRIMARY_NOT_YET_RECORDED" for r in records),
        "metrics_files_checked": len(metrics),
        "symbols_checked": len({r["symbol"] for r in metrics}),
        "file_row_count_histogram": dict(Counter(p["rows"] for p in profiles)),
        "files_missing_five_minute_reference_points": sum(
            p["missing_points_on_five_minute_reference_grid"] > 0 for p in profiles
        ),
        "files_with_original_nonincreasing_steps": sum(
            p["nonincreasing_steps_in_original_order"] > 0 for p in profiles
        ),
        "zero_open_interest_rows": sum(p["zero_open_interest_rows"] for p in profiles),
        "missing_fields": dict(sum((Counter(p["missing_by_column"]) for p in profiles), Counter())),
        "duplicate_keys": len(conflicts),
        "conflicting_duplicate_keys": sum(not p["same_values"] for p in conflicts),
        "index_files_checked": len(indices),
        "index_symbols_checked": len({r["symbol"] for r in indices}),
        "index_rows_checked": sum(p["rows"] for p in index_profiles),
        "index_files_missing_minute_reference_points": sum(
            p["missing_points_on_minute_reference_grid"] > 0 for p in index_profiles
        ),
        "index_files_with_original_nonincreasing_steps": sum(
            p["nonincreasing_steps_in_original_order"] > 0 for p in index_profiles
        ),
        "index_duplicate_keys": len(index_conflicts),
        "index_conflicting_duplicate_keys": sum(not p["same_values"] for p in index_conflicts),
        "clock_comparison": summary,
        "whole_population_coverage_approved": False,
        "open_interest_clock_proven": False,
        "historical_publication_times_proven": False,
        "returns_evaluated": False,
        "orders_enabled": False,
        "production_eligible": False,
    }
    save(output / "input_identities.json", inputs)
    save(output / "file_quality.json", profiles)
    save(output / "duplicate_keys.json", conflicts)
    save(output / "index_file_quality.json", index_profiles)
    save(output / "index_duplicate_keys.json", index_conflicts)
    save(output / "clock_pairs.json", pairs)
    save(output / "result.json", result)
    save(
        output / "receipts.json",
        {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()},
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--minute-study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    result = run(args.source, args.minute_study, args.output, args.allow_incomplete)
    print(result)


if __name__ == "__main__":
    main()
