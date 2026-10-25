"""将完整指数与持仓原文整理为保留未知的知识快照，不生成因子或标签。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import resource
import shutil
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tradingagents.dataflows.research import basis_oi_sources as parsing
from tradingagents.dataflows.research.cache_snapshot import validate_dependencies
from tradingagents.dataflows.research.uniform_inputs import dependency
from tradingagents.research.quant.artifacts import read, save, sha, verify_receipts

MINUTE = 60_000
DAY = 86_400_000
START = 1_790_553_600_000
END = 1_792_022_400_000
SYMBOLS = 748
ROWS_5M = 3_662_208
ROWS_HOURLY = 305_184
VERIFIED = "VERIFIED_BYTES_AND_PROFILE"
STATUSES = {VERIFIED, "PUBLIC_404", "FAILED_OR_UNKNOWN"}
SOURCE_RECEIPTS = "659674a5d2b6e8be62ef1ba9594bdcf6dbfd54be87e87dedffd0062174110d20"
SOURCE_PROTOCOL = "fe78a80bfb00c3ca98b824d67b6f6e3801217732e453570222964af39836407c"
FLAGS = {
    "labels_loaded": False,
    "models_loaded": False,
    "returns_evaluated": False,
    "orders_enabled": False,
    "leader_labels_read": False,
    "basis_generated": False,
}
KINDS = ("indexPriceKlines", "metrics")
RAW_SCHEMA = pa.schema(
    [
        ("symbol", pa.string()),
        ("kind", pa.string()),
        ("day", pa.string()),
        ("archive_file", pa.string()),
        ("archive_sha256", pa.string()),
        ("member", pa.string()),
        ("member_sha256", pa.string()),
        ("record_json", pa.string()),
    ]
)


def utc_day(clock):
    """原钟以UTC日归属，完成边界不能误归入下一档案。"""
    return datetime.fromtimestamp(clock / 1000, UTC).strftime("%Y-%m-%d")


def safe_source_path(root, relative):
    """来源必须是根目录内的普通文件，拒绝绝对路径、逃逸及软链接。"""
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("来源相对路径不合法")
    path = root / relative
    if not path.resolve().is_relative_to(root):
        raise ValueError("来源路径越出封存根目录")
    if any(p.is_symlink() for p in (path, *path.parents) if p != root and p.is_relative_to(root)):
        raise ValueError("来源路径包含软链接")
    return path


def validate_population(protocol, result):
    """完整身份及纯来源协议先验校验，不能以成功交集代替总体。"""
    if protocol.get("schema") != "cz-unconditional-public-factor-inputs.v1" or any(
        protocol.get(k) is not False
        for k in ("orders_enabled", "leader_labels_read", "returns_evaluated")
    ):
        raise ValueError("只接受原完整纯公开采集协议")
    lookup = validate_dependencies(protocol["dependencies"])
    records = result.get("records", [])
    if (
        result.get("schema") != "cz-unconditional-public-factor-inputs-result.v1"
        or result.get("planned_dependencies") != len(lookup)
        or result.get("recorded_dependencies") != len(lookup)
        or len(records) != len(lookup)
        or any(result.get(k) is not False for k in ("orders_enabled", "returns_evaluated"))
    ):
        raise ValueError("来源尚未完整终态或结果范围不同")
    mapped = {r["url"]: r for r in records}
    if len(mapped) != len(records) or set(mapped) != set(lookup):
        raise ValueError("来源记录重复或遗漏原依赖")
    for url, expected in lookup.items():
        row = mapped[url]
        if (
            any(row.get(k) != value for k, value in expected.items())
            or row.get("status") not in STATUSES
        ):
            raise ValueError("来源记录身份或状态不同")
    counts = Counter(r["status"] for r in records)
    if dict(counts) != result.get("statuses"):
        raise ValueError("来源状态统计与完整记录不一致")
    return sorted({r["symbol"] for r in records}), records


class BoundSource:
    """绑定唯一终态来源；每份实际读取原文在消费前后都核验。"""

    def __init__(self, root):
        self.root = Path(root).resolve()
        if sha(self.root / "receipts.json") != SOURCE_RECEIPTS:
            raise ValueError("来源不是本计划固定的封存版本")
        self.expected = read(self.root / "receipts.json")
        self.inputs = {str(self.root / "receipts.json"): SOURCE_RECEIPTS}
        verify_receipts(self.root)
        if sha(self.add("protocol.json")) != SOURCE_PROTOCOL:
            raise ValueError("来源协议不是事前固定身份")
        self.protocol = read(self.add("protocol.json"))
        self.result = read(self.add("result.json"))
        self.symbols, records = validate_population(self.protocol, self.result)
        self.records = {}
        for row in records:
            rel = "records/" + hashlib.sha256(row["url"].encode()).hexdigest() + ".json"
            if read(self.add(rel)) != row:
                raise ValueError("终态结果与逐项原始记录不同")
            self.records[row["symbol"], row["day"], row["kind"]] = row
        self.source_status_counts = dict(Counter(r["status"] for r in records))

    def add(self, relative):
        """单文件必须同时属于父收据及当前实际字节。"""
        path = safe_source_path(self.root, relative)
        expected = self.expected.get(str(Path(relative)))
        if expected is None or not path.is_file() or sha(path) != expected:
            raise ValueError("实际来源不在原收据或字节已改变")
        self.inputs[str(path)] = expected
        return path

    def archive(self, row):
        """已失败项不重新解读成功；404亦核对其公开下载收据。"""
        if row["status"] == "FAILED_OR_UNKNOWN":
            return None
        receipt = read(self.add(row["file"] + ".receipt.json"))
        if receipt.get("url") != row["url"]:
            raise ValueError("下载收据与公开URL不同")
        if row["status"] == "PUBLIC_404":
            if receipt.get("status") != 404 or safe_source_path(self.root, row["file"]).exists():
                raise ValueError("404来源与实际原文冲突")
            return None
        path = self.add(row["file"])
        expected = row.get("sha256")
        checksum = self.add(row["file"] + ".CHECKSUM").read_text().split()
        if (
            receipt.get("status") != 200
            or receipt.get("sha256") != expected
            or self.inputs[str(path)] != expected
            or not checksum
            or checksum[0] != expected
        ):
            raise ValueError("原文、HTTP收据与官方校验值不同")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError("读取期间原始档案发生变化")
        return payload

    def verify_after(self):
        """终态既核消费身份，也核原来源完整文件集合。"""
        for path, expected in self.inputs.items():
            if sha(Path(path)) != expected:
                raise ValueError("构建期间消费来源发生变化")
        verify_receipts(self.root)
        if sha(self.root / "receipts.json") != SOURCE_RECEIPTS:
            raise ValueError("构建期间原来源收据发生变化")


def panel_schema(rows):
    """固定文本数值及时间类型，首符号全空也不能推断为Arrow null列。"""
    names = sorted(rows[0])
    if any(set(row) != set(names) for row in rows):
        raise ValueError("规范面板行字段不一致")
    fields = []
    for name in names:
        if name.endswith("source_line_nos"):
            dtype = pa.list_(pa.int64())
        elif name.endswith("_ms") or name.endswith("duplicate_count"):
            dtype = pa.int64()
        else:
            dtype = pa.string()
        fields.append((name, dtype))
    return pa.schema(fields)


class Coverage:
    """全格覆盖守恒；未知理由留在实表，不以共同已知成员重选人口。"""

    def __init__(self, clocks, fields):
        self.clocks, self.fields = np.asarray(clocks), tuple(fields)
        self.known_by_clock = np.zeros((len(clocks), len(fields)), dtype=np.int64)
        self.complete_by_clock = np.zeros(len(clocks), dtype=np.int64)
        self.statuses = {f: Counter() for f in fields}
        self.by_symbol, self.by_day, self.by_hour = {}, {}, {}
        self.rows = 0

    def add(self, symbol, rows):
        """每个原符号必须恰好覆盖相同知识日历。"""
        if symbol in self.by_symbol or [r["clock_ms"] for r in rows] != list(self.clocks):
            raise ValueError("来源面板遗漏、重复或压缩原日历")
        all_known = np.ones(len(rows), dtype=bool)
        summary = {}
        day_keys = [utc_day(int(t)) for t in self.clocks[::288]]
        prepared = []
        for field in self.fields:
            states = np.asarray([r[field + "_value_status"] for r in rows], dtype=object)
            known = states == "KNOWN_RAW_VALUE"
            if any((r[field] is not None) != bool(k) for r, k in zip(rows, known, strict=True)):
                raise ValueError("字段状态与已知数值不一致")
            prepared.append((states, known))
        for j, (field, (states, known)) in enumerate(zip(self.fields, prepared, strict=True)):
            self.known_by_clock[:, j] += known
            all_known &= known
            counts = Counter(states.tolist())
            self.statuses[field].update(counts)
            summary[field] = dict(counts)
            for d, day in enumerate(day_keys):
                target = self.by_day.setdefault(day, {}).setdefault(field, Counter())
                target.update(states[d * 288 : (d + 1) * 288].tolist())
            for hour in range(24):
                target = self.by_hour.setdefault(str(hour), {}).setdefault(field, Counter())
                sliced = states.reshape(-1, 24, 12)[:, hour, :]
                target.update(sliced.reshape(-1).tolist())
        self.complete_by_clock += all_known
        self.by_symbol[symbol] = summary
        self.rows += len(rows)

    def result(self):
        """原生审查只获得原输入覆盖事实，不获得任何预测或收益。"""
        fields = {}
        for field in self.fields:
            index = field.startswith("index_1m_")
            known = self.statuses[field]["KNOWN_RAW_VALUE"]
            if sum(self.statuses[field].values()) != self.rows:
                raise ValueError("字段覆盖统计未守恒")
            fields[field] = {
                "source_column": field,
                "known_rows": known,
                "unknown_rows": self.rows - known,
                "time_basis": "completed_boundary_assumption"
                if index
                else "raw_T_plus_600000_ms_assumption",
                "lag_ms": 0 if index else 600_000,
                "scope": "panel_5m",
                "value_status_counts": dict(self.statuses[field]),
            }
        return {
            "schema": "ta-native-source-input-coverage.v1",
            "rows_5m": self.rows,
            "fields": fields,
            "by_symbol": self.by_symbol,
            "by_day": self.by_day,
            "by_utc_hour": self.by_hour,
            "by_clock": [
                {
                    "clock_ms": int(clock),
                    "complete_fields_rows": int(self.complete_by_clock[i]),
                    "known_rows": {
                        f: int(self.known_by_clock[i, j]) for j, f in enumerate(self.fields)
                    },
                }
                for i, clock in enumerate(self.clocks)
            ],
        }


def source_context(row, parsed=None):
    """原行通过明确档案与member身份回溯，字段名不隐含价格或交易含义。"""
    return {
        "symbol": row["symbol"],
        "kind": row["kind"],
        "day": row["day"],
        "archive_file": row["file"],
        "archive_sha256": row.get("sha256"),
        "member": parsed.get("member") if parsed else None,
        "member_sha256": parsed.get("member_sha256") if parsed else None,
    }


def write_raw(writer, context, records):
    """逐档案落全部原行，不把日历抽样误称完整原文保存。"""
    for start in range(0, len(records), 2048):
        rows = [
            context
            | {
                "record_json": json.dumps(
                    r, ensure_ascii=False, allow_nan=False, separators=(",", ":")
                )
            }
            for r in records[start : start + 2048]
        ]
        writer.write_table(pa.Table.from_pylist(rows, schema=RAW_SCHEMA))


def execution_paths():
    """仅冻结实际执行依赖，后续其他研究开发不改写本轮身份。"""
    return [
        Path(__file__).resolve(),
        Path(parsing.__file__).resolve(),
        Path(validate_dependencies.__code__.co_filename).resolve(),
        Path(dependency.__code__.co_filename).resolve(),
        Path(sha.__code__.co_filename).resolve(),
        Path(__file__).with_name("basis_oi_panel_plan.md"),
    ]


def verify_execution(frozen):
    """运行前后代码与计划同一身份，不能把途中修复重命名成原实验。"""
    if any(sha(Path(path)) != value for path, value in frozen.items()):
        raise ValueError("实际执行依赖或计划在构建期间发生变化")


def check_resources(output):
    """登记资源上限实际执行，不能靠分块意图冒称已有限制。"""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak if sys.platform == "darwin" else peak * 1024
    if peak_bytes > 2 * 1024**3:
        raise ValueError("内存峰值超过登记的2GiB")
    if shutil.disk_usage(output).free < 2 * 1024**3:
        raise ValueError("磁盘余量低于登记的2GiB")


def _run(source, output):
    """先冻结协议，再有界构建完整人口；异常和失败均在新目录封存。"""
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("新输出不能嵌套或覆盖原来源")
    frozen = {str(p): sha(p) for p in execution_paths()}
    bound = BoundSource(source)
    verify_execution(frozen)
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source"
    snapshot.mkdir()
    for i, (path, expected) in enumerate(frozen.items()):
        target = snapshot / f"{i:02d}_{Path(path).name}"
        target.write_bytes(Path(path).read_bytes())
        if sha(target) != expected:
            raise ValueError("执行依赖复制期间发生变化")
    clocks = np.arange(START, END, 5 * MINUTE, dtype=np.int64)
    days = [utc_day(t) for t in range(START, END, DAY)]
    fields = tuple(parsing.INDEX_FIELDS) + tuple(parsing.METRICS_FIELDS)
    protocol = {
        "schema": "ta-basis-oi-source-panel.v1",
        "created_ms": time.time_ns() // 1_000_000,
        "symbols": bound.symbols,
        "start_ms": START,
        "end_exclusive_ms": END,
        "rows_5m": ROWS_5M,
        "rows_hourly": ROWS_HOURLY,
        "source_root": str(source),
        "source_receipts_sha256": SOURCE_RECEIPTS,
        "source_protocol_sha256": SOURCE_PROTOCOL,
        "source_status_counts": bound.source_status_counts,
        "source_sha256": frozen,
        "columns": fields,
        "metrics_visibility": "raw_T_plus_600000_ms_assumption",
        "metrics_delay_ms": 600_000,
        "index_visibility": "completed_boundary_assumption",
        "unknowns_preserved": True,
        "raw_decimal_text_preserved": True,
        "numeric_or_historical_publication_complete": False,
        "historical_point_in_time_rules_proven": False,
        "independent_oos": False,
        "production_eligible": False,
        "network_requests_enabled": False,
        "memory_budget_bytes": 2 * 1024**3,
        "minimum_free_disk_bytes": 2 * 1024**3,
        "python": sys.version,
        "libraries": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "pyarrow")},
        **FLAGS,
    }
    save(output / "protocol.json", protocol)
    coverage = Coverage(clocks, fields)
    completed, raw_rows, anomaly_rows, parse_failed = [], 0, 0, []
    writer_5m = writer_hour = None
    started = time.monotonic()
    failure = None
    hourly_rows = 0
    try:
        with (
            pq.ParquetWriter(
                output / "source_records.parquet", RAW_SCHEMA, compression="zstd"
            ) as raw_writer,
            pq.ParquetWriter(
                output / "anomalies.parquet", RAW_SCHEMA, compression="zstd"
            ) as anomaly_writer,
            (output / "archive_profiles.jsonl").open("x", encoding="utf-8") as profiles,
        ):
            schema = None
            for number, symbol in enumerate(bound.symbols, 1):
                check_resources(output)
                normalized = {kind: {} for kind in KINDS}
                statuses = {kind: {} for kind in KINDS}
                contexts = {kind: {} for kind in KINDS}
                for day in days:
                    for kind in KINDS:
                        row = bound.records[symbol, day, kind]
                        statuses[kind][day] = row
                        context = source_context(row)
                        payload = bound.archive(row)
                        info = {"source_record": row, "parse_status": "NOT_PARSED_ORIGINAL_UNKNOWN"}
                        if payload is not None:
                            try:
                                parsed = parsing.parse_archive(payload, kind, symbol, day)
                            except parsing.ArchiveFormatError as exc:
                                failure_row = context | {
                                    "reason": str(exc),
                                    "parse_status": "ARCHIVE_PARSE_FAILED",
                                }
                                parse_failed.append(failure_row)
                                write_raw(anomaly_writer, context, [failure_row])
                                anomaly_rows += 1
                                statuses[kind][day] = row | {
                                    "parse_status": "ARCHIVE_PARSE_FAILED",
                                    "parse_error": str(exc),
                                }
                                info |= {"parse_status": "ARCHIVE_PARSE_FAILED", "error": str(exc)}
                            else:
                                context = source_context(row, parsed)
                                write_raw(raw_writer, context, parsed["raw_records"])
                                write_raw(anomaly_writer, context, parsed["anomalies"])
                                raw_rows += len(parsed["raw_records"])
                                anomaly_rows += len(parsed["anomalies"])
                                for key, value in parsed["normalized"].items():
                                    if key in normalized[kind]:
                                        raise ValueError("不同日档案产生同一规范键，不能选择覆盖")
                                    normalized[kind][key] = value
                                info |= {
                                    "parse_status": "PARSED",
                                    "member": parsed["member"],
                                    "member_sha256": parsed["member_sha256"],
                                    "profile": parsed["profile"],
                                }
                                if sha(bound.add(row["file"])) != row["sha256"]:
                                    raise ValueError("档案解析后字节发生变化")
                        contexts[kind][day] = context | {"parse_status": info["parse_status"]}
                        profiles.write(
                            json.dumps(
                                info, ensure_ascii=False, allow_nan=False, separators=(",", ":")
                            )
                            + "\n"
                        )
                rows = parsing.source_values_for_clock(
                    clocks.tolist(),
                    normalized["indexPriceKlines"],
                    normalized["metrics"],
                    statuses["indexPriceKlines"],
                    statuses["metrics"],
                )
                for row in rows:
                    row["symbol"] = symbol
                    for kind, raw_clock in (
                        ("indexPriceKlines", row["clock_ms"] - MINUTE),
                        ("metrics", row["clock_ms"] - 10 * MINUTE),
                    ):
                        context = contexts[kind].get(utc_day(raw_clock), {})
                        for key in ("archive_file", "archive_sha256", "member", "member_sha256"):
                            row[kind + "_" + key] = context.get(key)
                        row[kind + "_parse_status"] = context.get(
                            "parse_status", "OUTSIDE_SOURCE_RANGE"
                        )
                coverage.add(symbol, rows)
                if schema is None:
                    schema = panel_schema(rows)
                    writer_5m = pq.ParquetWriter(
                        output / "panel_5m.parquet", schema, compression="zstd"
                    )
                    writer_hour = pq.ParquetWriter(
                        output / "panel_hourly.parquet", schema, compression="zstd"
                    )
                if set(rows[0]) != set(schema.names):
                    raise ValueError("符号间面板schema改变")
                table = pa.Table.from_pylist(rows, schema=schema)
                writer_5m.write_table(table)
                hours = table.filter(pa.array([r["clock_ms"] % (60 * MINUTE) == 0 for r in rows]))
                writer_hour.write_table(hours)
                hourly_rows += hours.num_rows
                completed.append(symbol)
                check_resources(output)
                if number % 25 == 0 or number == len(bound.symbols):
                    print(
                        json.dumps(
                            {
                                "completed_symbols": number,
                                "symbols": len(bound.symbols),
                                "rows_5m": coverage.rows,
                                "raw_rows": raw_rows,
                                "elapsed_seconds": round(time.monotonic() - started, 2),
                            }
                        ),
                        flush=True,
                    )
        if coverage.rows != ROWS_5M or hourly_rows != ROWS_HOURLY or len(completed) != SYMBOLS:
            raise ValueError("完整人口或输出格点数量不守恒")
    except Exception as exc:
        failure = {"type": type(exc).__name__, "reason": str(exc)}
    finally:
        for writer in (writer_5m, writer_hour):
            if writer is not None:
                writer.close()
    try:
        bound.verify_after()
        verify_execution(frozen)
        if failure is None:
            for name, expected in (
                ("panel_5m.parquet", ROWS_5M),
                ("panel_hourly.parquet", ROWS_HOURLY),
                ("source_records.parquet", raw_rows),
                ("anomalies.parquet", anomaly_rows),
            ):
                if pq.read_metadata(output / name).num_rows != expected:
                    raise ValueError("实际Parquet行数与构建记录不一致")
    except Exception as exc:
        integrity = {"type": type(exc).__name__, "reason": str(exc)}
        failure = integrity if failure is None else failure | {"final_integrity_error": integrity}
    covered = coverage.result()
    covered["source_status_counts"] = bound.source_status_counts
    covered["original_unknown_dependencies"] = [
        r for r in bound.records.values() if r["status"] != VERIFIED
    ]
    covered["archive_parse_failures"] = parse_failed
    save(output / "coverage.json", covered)
    save(output / "consumed_inputs.json", bound.inputs)
    result = {
        "schema": "ta-basis-oi-source-panel-result.v1",
        "status": "FAILED" if failure else "COMPLETED_WITH_UNKNOWN",
        "failure": failure,
        "protocol_sha256": sha(output / "protocol.json"),
        "rows_5m": coverage.rows,
        "rows_hourly": hourly_rows,
        "completed_symbols": completed,
        "raw_source_rows": raw_rows,
        "anomaly_rows": anomaly_rows,
        "parse_failed_archives": len(parse_failed),
        "elapsed_seconds": time.monotonic() - started,
        "production_eligible": False,
        **FLAGS,
    }
    save(output / "result.json", result)
    save(
        output / "receipts.json",
        {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()},
    )
    verify_receipts(output)
    if failure:
        raise ValueError("来源面板构建失败，原错误与部分产物已封存：" + failure["reason"])
    return result


def run(source, output):
    """新目录创建后连准备阶段失败也封存，现有目录绝不覆盖。"""
    target = Path(output).resolve()
    existed = target.exists()
    try:
        return _run(source, output)
    except Exception as exc:
        if not existed and target.exists() and not (target / "receipts.json").exists():
            failure_path = target / "construction_failure.json"
            if not failure_path.exists():
                save(failure_path, {"type": type(exc).__name__, "reason": str(exc), **FLAGS})
            save(
                target / "receipts.json",
                {
                    str(p.relative_to(target)): sha(p)
                    for p in sorted(target.rglob("*"))
                    if p.is_file()
                },
            )
        raise


def main(argv=None):
    """命令只接受完整封存来源与新的本地输出，不提供成功子集选项。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = run(args.source, args.output)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "rows_5m",
                    "rows_hourly",
                    "raw_source_rows",
                    "anomaly_rows",
                    "elapsed_seconds",
                )
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
