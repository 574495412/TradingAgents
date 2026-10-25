"""由封存原文构建显式Regular/Special资金费视图，保留全人口和未知。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import sys
import urllib.parse
import zipfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

if __package__:
    from .historical_rest import capture_pages
else:
    from historical_rest import capture_pages

HERE = Path(__file__).resolve().parent
MONTHS = ("2026-08", "2026-09")
EXPECTED_SYMBOLS = 909
FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
ARCHIVE_URL = "https://data.binance.vision/"
ARCHIVE_FIELDS = ("calc_time", "funding_interval_hours", "last_funding_rate")


def sha(path):
    """以流式SHA绑定原文，避免整包读入内存。"""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    """NaN/Infinity不能成为有效规范JSON。"""

    def invalid(value):
        raise ValueError("NONFINITE_JSON:" + value)

    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=invalid)


def save(path, value):
    """新目录排他写入，绝不覆盖原结果。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)


def canonical(value):
    """比较完整原字段及数值类型，不只比较(symbol,time)。"""
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )


def decimal(value, label):
    """有限十进制精确比较，不经float或normalize舍入。"""
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(label + "_INVALID_DECIMAL")
    try:
        number = Decimal(str(value))
    except (ValueError, InvalidOperation) as error:
        raise ValueError(label + "_INVALID_DECIMAL") from error
    if not number.is_finite():
        raise ValueError(label + "_NONFINITE")
    return number


def bounds(month):
    """所有月份采用UTC右开原毫秒，不取整结算时钟。"""
    if month not in MONTHS:
        raise ValueError("MONTH_OUTSIDE_PLAN")
    year, number = map(int, month.split("-"))
    return tuple(
        int(datetime(year, m, 1, tzinfo=UTC).timestamp() * 1000) for m in (number, number + 1)
    )


class BoundSource:
    """收据内路径和实际使用输入的极小边界，不负责网络或调度。"""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.receipts_sha256 = sha(self.root / "receipts.json")
        self.expected = read(self.root / "receipts.json")
        self.used = {}
        self.integrity_changed = False
        self.attempts = defaultdict(list)
        for name in self.expected:
            leaf = Path(name)
            if re.fullmatch(r"attempt_[0-9]+\.json", leaf.name):
                self.attempts[leaf.parent.as_posix()].append(name)

    def path(self, relative):
        """路径必须逐字属于封存清单，不能逃逸或借符号链接换源。"""
        if not isinstance(relative, str) or "\\" in relative:
            raise ValueError("SOURCE_PATH_INVALID")
        leaf = Path(relative)
        if (
            leaf.is_absolute()
            or not leaf.parts
            or ".." in leaf.parts
            or relative not in self.expected
        ):
            raise ValueError("SOURCE_PATH_OUTSIDE_RECEIPTS")
        path = self.root / leaf
        if not path.resolve().is_relative_to(self.root) or any(
            p.is_symlink() for p in (path, *path.parents) if p.is_relative_to(self.root)
        ):
            raise ValueError("SOURCE_PATH_ESCAPE_OR_SYMLINK")
        try:
            actual = sha(path)
        except OSError:
            self.integrity_changed = True
            raise
        if actual != self.expected[relative]:
            self.integrity_changed = True
            raise ValueError("SOURCE_BYTES_CHANGED:" + relative)
        self.used[relative] = self.expected[relative]
        return path

    def load(self, relative):
        """仅从已绑定路径解码JSON。"""
        return read(self.path(relative))

    def verify_all(self):
        """集合与所有原文字节均通过后，才开始语义视图计算。"""
        actual = {str(p.relative_to(self.root)) for p in self.root.rglob("*") if p.is_file()} - {
            "receipts.json"
        }
        if actual != set(self.expected):
            raise ValueError("SOURCE_FILE_SET_CHANGED")
        for name in self.expected:
            self.path(name)
        self.used = {}
        self.verify_used()

    def verify_used(self):
        """末验真正用过的输入和来源收据，保留并发变化为失败。"""
        if self.integrity_changed or sha(self.root / "receipts.json") != self.receipts_sha256:
            raise ValueError("SOURCE_RECEIPTS_CHANGED")
        for name, expected in self.used.items():
            if sha(self.root / name) != expected:
                raise ValueError("USED_SOURCE_BYTES_CHANGED:" + name)

    def latest(self, folder):
        """只取本任务已封存的最后不可变attempt。"""
        names = self.attempts.get(folder, [])
        if not names:
            raise ValueError("TASK_RECORD_MISSING:" + folder)
        name = max(names, key=lambda p: int(Path(p).stem.split("_")[-1]))
        return self.load(name)

    def record_files(self, record, folder):
        """任务记录不可引用别的符号、目录或不相符的文件摘要。"""
        files = record.get("files")
        if not isinstance(files, dict):
            raise ValueError("TASK_FILES_INVALID")
        for name, expected in files.items():
            if not Path(name).is_relative_to(folder) or self.expected.get(name) != expected:
                raise ValueError("TASK_FILE_WRONG_SOURCE")
            self.path(name)


def validate_plan(plan):
    """保留原909并集和全部任务，不按覆盖筛人口。"""
    symbols = plan.get("symbols", [])
    if (
        plan.get("schema") != "ta-fixed-historical-public-capture.v1"
        or len(symbols) != EXPECTED_SYMBOLS
        or len(set(symbols)) != len(symbols)
    ):
        raise ValueError("FIXED_CAPTURE_POPULATION_INVALID")
    if any(
        not isinstance(s, str)
        or not s.endswith("USDT")
        or not all(c.isalnum() or c == "_" for c in s)
        for s in symbols
    ):
        raise ValueError("SYMBOL_INVALID")
    if (
        tuple(plan["months"]) != MONTHS
        or set(plan["rest_symbols"]) != set(symbols)
        or not set(plan["original_universe"]) <= set(symbols)
    ):
        raise ValueError("FIXED_CAPTURE_SCOPE_CHANGED")
    if (plan["start_ms"], plan["end_exclusive_ms"]) != (
        bounds(MONTHS[0])[0],
        bounds(MONTHS[-1])[1],
    ):
        raise ValueError("FIXED_CAPTURE_CLOCK_CHANGED")
    tasks = plan["archive_tasks"]
    expected = {(s, m, k) for s in symbols for m in MONTHS for k in ("1m", "fundingRate")}
    if (
        len(tasks) != len(expected)
        or {(t["symbol"], t["month"], t["kind"]) for t in tasks} != expected
    ):
        raise ValueError("FIXED_ARCHIVE_TASKS_INCOMPLETE")
    for task in tasks:
        symbol, month, kind = task["symbol"], task["month"], task["kind"]
        suffix = (
            f"klines/{symbol}/1m/{symbol}-1m-{month}.zip"
            if kind == "1m"
            else f"fundingRate/{symbol}/{symbol}-fundingRate-{month}.zip"
        )
        if (
            task["id"] != f"{kind}_{symbol}_{month}"
            or task["key"] != "data/futures/um/monthly/" + suffix
            or type(task["listed"]) is not bool
        ):
            raise ValueError("FIXED_ARCHIVE_IDENTITY_CHANGED")
    if any(
        plan[k] is not False for k in ("orders_enabled", "leader_labels_read", "returns_evaluated")
    ):
        raise ValueError("NOT_PUBLIC_READONLY_CAPTURE")


def record_identity(record, plan_sha256, symbol, task=None):
    """计划、符号和任务时钟均须对应同一封存来源。"""
    if record.get("symbol") != symbol or record.get("plan_sha256") != plan_sha256:
        raise ValueError("TASK_PLAN_OR_SYMBOL_MISMATCH")
    if task and any(record.get(k) != task[k] for k in ("month", "kind")):
        raise ValueError("ARCHIVE_TASK_MISMATCH")
    if task and record.get("task_id") != task["id"]:
        raise ValueError("ARCHIVE_TASK_MISMATCH")


def event_conflicts(events):
    """同类型同钟冲突不覆盖，异类型同钟保留两笔。"""
    grouped = defaultdict(list)
    for index, event in enumerate(events):
        grouped[event["symbol"], event["fundingTime"], event["rateType"]].append(index)
    return {index for indices in grouped.values() if len(indices) > 1 for index in indices}


def rejected_source(source, symbol, detail):
    """失败规范事件只入隔离记录，原文指针及原字段保留且不宣称可信。"""
    folder = "rest_jobs/" + symbol
    value = {
        **detail,
        "source_capture_receipts_sha256": source.receipts_sha256,
        "source_task_folder": folder,
        "trusted_typed_fields": False,
    }
    try:
        record = source.latest(folder)
        name = record.get("events")
        if (
            not isinstance(name, str)
            or not Path(name).is_relative_to(folder)
            or source.expected.get(name) != record.get("files", {}).get(name)
        ):
            raise ValueError("REJECTED_EVENT_REFERENCE_NOT_BOUND")
        value.update(
            events_path=name,
            events_sha256=source.expected[name],
            observed_normalized_events=source.load(name),
        )
    except Exception as error:
        value["source_reference_error"] = type(error).__name__ + ":" + str(error)[:300]
    return value


def replay_rest(source, plan, plan_sha256, symbol):
    """从收据绑定成功页离线重放，不相信单独的规范事件JSON。"""
    folder = "rest_jobs/" + symbol
    record = source.latest(folder)
    record_identity(record, plan_sha256, symbol)
    source.record_files(record, folder)
    if record["status"] not in {"API_RANGE_VERIFIED", "API_EMPTY_RANGE"}:
        raise ValueError("REST_CAPTURE_NOT_VERIFIED")
    event_path = record["events"]
    if event_path not in record["files"]:
        raise ValueError("EVENTS_NOT_BOUND_TO_TASK")
    normalized = source.load(event_path)
    pages = []
    for name in record["files"]:
        path = Path(name)
        if path.parent.as_posix() == folder + "/requests" and path.suffix == ".json":
            receipt = source.load(name)
            body_name = path.with_suffix(".body").as_posix()
            if (
                receipt.get("status") != 200
                or receipt.get("error") is not None
                or receipt.get("method") != "GET"
                or receipt.get("credentials_used") is not False
                or receipt.get("body") != body_name
                or record["files"].get(body_name) != receipt.get("sha256")
            ):
                raise ValueError("REST_RAW_RECEIPT_MISMATCH")
            parsed = urllib.parse.urlsplit(receipt["url"])
            query = urllib.parse.parse_qs(parsed.query, strict_parsing=True)
            if (
                parsed.scheme + "://" + parsed.netloc + parsed.path != FUNDING_URL
                or set(query) != {"symbol", "startTime", "endTime", "limit"}
                or any(len(v) != 1 for v in query.values())
                or query["symbol"] != [symbol]
                or query["limit"] != ["1000"]
                or parsed.fragment
            ):
                raise ValueError("REST_QUERY_WRONG_SOURCE")
            first, last = int(query["startTime"][0]), int(query["endTime"][0])
            if not plan["start_ms"] <= first <= last < plan["end_exclusive_ms"]:
                raise ValueError("REST_QUERY_OUTSIDE_PLAN")
            pages.append((int(path.stem), first, last, body_name, source.load(body_name)))
    pages.sort()
    if not pages:
        raise ValueError("REST_RAW_PAGES_MISSING")
    cursor = 0

    def fetch(first, last):
        nonlocal cursor
        if cursor >= len(pages) or pages[cursor][1:3] != (first, last):
            raise ValueError("REST_RAW_QUERY_SEQUENCE_CHANGED")
        data = pages[cursor][4]
        cursor += 1
        return data

    replay = capture_pages(
        symbol, plan["start_ms"], plan["end_exclusive_ms"], fetch, max_pages=len(pages) + 1
    )
    if (
        cursor != len(pages)
        or replay["status"] != record["status"]
        or canonical(replay["events"]) != canonical(normalized)
    ):
        raise ValueError("REST_RAW_NORMALIZED_MISMATCH")
    if "pager" in record and canonical(record["pager"]["checkpoint"]["closed_events"]) != canonical(
        normalized
    ):
        raise ValueError("REST_CHECKPOINT_NORMALIZED_MISMATCH")
    return {
        "record": record,
        "events": normalized,
        "conflicts": event_conflicts(normalized),
        "source_consistent": True,
        "raw_response_files": [{"path": p[3], "sha256": source.expected[p[3]]} for p in pages],
    }


def archive_rows(source, task, plan_sha256):
    """UTF-8官方摘要、三字段原CSV和规范JSON逐字段相符才可比较。"""
    folder = "archive_jobs/" + task["id"]
    record = source.latest(folder)
    record_identity(record, plan_sha256, task["symbol"], task)
    source.record_files(record, folder)
    if not task["listed"]:
        if record["status"] != "NOT_LISTED":
            raise ValueError("UNLISTED_ARCHIVE_STATE_CHANGED")
        return {
            "record": record,
            "rows": None,
            "source_consistent": False,
            "state": "UNKNOWN_NOT_LISTED",
        }
    if record["status"] != "VERIFIED":
        raise ValueError("ARCHIVE_CAPTURE_NOT_VERIFIED")
    bodies = []
    for key, suffix in (("zip_receipt", ""), ("checksum_receipt", ".CHECKSUM")):
        receipt = record[key]
        name = receipt["body"]
        if (
            name not in record["files"]
            or receipt.get("sha256") != record["files"][name]
            or receipt.get("status") != 200
            or receipt.get("method") != "GET"
            or receipt.get("credentials_used") is not False
            or receipt.get("url")
            != ARCHIVE_URL + urllib.parse.quote(task["key"] + suffix, safe="/")
        ):
            raise ValueError("ARCHIVE_RAW_RECEIPT_MISMATCH")
        external = Path(name).with_suffix(".json").as_posix()
        if external not in record["files"] or canonical(source.load(external)) != canonical(
            receipt
        ):
            raise ValueError("ARCHIVE_EXTERNAL_RECEIPT_MISMATCH")
        bodies.append(source.path(name))
    tokens = bodies[1].read_text(encoding="utf-8", errors="strict").split()
    if (
        len(tokens) != 2
        or tokens[1].removeprefix("*") != Path(task["key"]).name
        or tokens[0] != sha(bodies[0])
    ):
        raise ValueError("OFFICIAL_CHECKSUM_MISMATCH")
    first, last = bounds(task["month"])
    rows, prior = [], None
    with zipfile.ZipFile(bodies[0]) as archive:
        members = archive.infolist()
        if (
            len(members) != 1
            or members[0].filename != Path(task["key"]).name[:-4] + ".csv"
            or members[0].file_size > 128 * 1024**2
        ):
            raise ValueError("ARCHIVE_CSV_MEMBER_INVALID")
        with archive.open(members[0]) as stream:
            reader = csv.reader(io.TextIOWrapper(stream, encoding="utf-8-sig", newline=""))
            if tuple(next(reader, ())) != ARCHIVE_FIELDS:
                raise ValueError("ARCHIVE_COLUMNS_CHANGED")
            for row in reader:
                if len(row) != 3 or not row[0].isdigit():
                    raise ValueError("ARCHIVE_ROW_INVALID")
                stamp = int(row[0])
                if (
                    not first <= stamp < last
                    or (prior is not None and stamp <= prior)
                    or decimal(row[1], "INTERVAL") <= 0
                ):
                    raise ValueError("ARCHIVE_TIME_OR_INTERVAL_INVALID")
                decimal(row[2], "RATE")
                rows.append(dict(zip(ARCHIVE_FIELDS, row, strict=True)))
                prior = stamp
    normalized = record["funding_rows"]
    if normalized not in record["files"] or canonical(rows) != canonical(source.load(normalized)):
        raise ValueError("ARCHIVE_RAW_NORMALIZED_MISMATCH")
    return {"record": record, "rows": rows, "source_consistent": True, "state": "VERIFIED"}


def compare_month(symbol, month, original, archive, rest, errors):
    """保留原Regular对账，另报完整Regular+Special；未知不变成零现金。"""
    first, last = bounds(month)
    events = [e for e in rest.get("events", []) if first <= e["fundingTime"] < last]
    regular = Counter(
        (e["fundingTime"], decimal(e["fundingRate"], "RATE"))
        for e in events
        if e["rateType"] == "Regular"
    )
    special = Counter(
        (e["fundingTime"], decimal(e["fundingRate"], "RATE"))
        for e in events
        if e["rateType"] == "Special"
    )
    archived = Counter(
        (int(e["calc_time"]), decimal(e["last_funding_rate"], "RATE"))
        for e in (archive.get("rows") or [])
    )
    conflicts = sum(
        first <= rest["events"][i]["fundingTime"] < last for i in rest.get("conflicts", set())
    )
    consistent = bool(archive.get("source_consistent") and rest.get("source_consistent"))
    status = (
        "UNKNOWN_MISSING_SOURCE"
        if not consistent
        else "UNKNOWN_EMPTY_SOURCE"
        if not archived or not events
        else "MATCHED"
        if archived == regular + special
        else "MISMATCH"
    )
    proven = status == "MATCHED" and not conflicts
    api_status = rest.get("record", {}).get("status", "UNKNOWN")
    return {
        "symbol": symbol,
        "month": month,
        "start_ms": first,
        "end_exclusive_ms": last,
        "original_comparison": original,
        "rest_source_consistent": bool(rest.get("source_consistent")),
        "archive_source_consistent": bool(archive.get("source_consistent")),
        "source_consistent": consistent,
        "typed_events_consistent": bool(rest.get("source_consistent") and not conflicts),
        "regular_plus_special_status": status,
        "regular_plus_special_correspondence_proven": proven,
        "month_ready_under_source_assumptions": bool(proven and api_status == "API_RANGE_VERIFIED"),
        "api_range_verified": api_status == "API_RANGE_VERIFIED",
        "api_empty_unknown": api_status == "API_EMPTY_RANGE",
        "observed_regular_events": sum(regular.values()),
        "observed_special_events": sum(special.values()),
        "observed_archive_rows": sum(archived.values())
        if archive.get("source_consistent")
        else None,
        "identity_conflict_count": conflicts,
        "archive_only_multiset_count": sum((archived - regular - special).values())
        if consistent
        else None,
        "rest_only_multiset_count": sum((regular + special - archived).values())
        if consistent
        else None,
        "original_regular_mismatch_explained_by_special": bool(
            original["status"] == "MISMATCH" and proven and special
        ),
        "event_absence_proven": False,
        "known_zero_cash": False,
        "api_range_assumption_only": True,
        "independent_completeness_proven": False,
        "historical_eligibility_proven": False,
        "errors": errors,
    }


def source_paths():
    """只绑定实际用到的模块、纯分页helper和本计划。"""
    return {
        "historical_funding.py": Path(__file__).resolve(),
        "historical_rest.py": Path(capture_pages.__code__.co_filename).resolve(),
        "historical_funding_plan.md": HERE / "historical_funding_plan.md",
    }


def assert_stable(paths, expected):
    """并发源码修改必须显式失败，不能依赖整包背景漂移。"""
    for name, path in paths.items():
        if sha(path) != expected[name]:
            raise ValueError("EXECUTED_SOURCE_CHANGED:" + name)


def run(source, output):
    """新目录冻结后计算；单项未知保留，完整性失败则封存失败及部分输出。"""
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("OUTPUT_OVERLAPS_SOURCE")
    output.mkdir(parents=True, exist_ok=False)
    errors, coverage, event_counts = [], [], Counter()
    protocol = None
    try:
        bound = BoundSource(source)
        paths = source_paths()
        executed = {name: sha(path) for name, path in paths.items()}
        for name, path in paths.items():
            target = output / "source" / name
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(path, target)
            if sha(target) != executed[name]:
                raise ValueError("EXECUTED_SOURCE_COPY_CHANGED")
        inputs = ("receipts.json", "capture_plan.json", "funding_month_comparisons.json")
        input_ids = {str(source / name): sha(source / name) for name in inputs}
        for name in inputs:
            target = output / "inputs" / name
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(source / name, target)
            if sha(target) != input_ids[str(source / name)]:
                raise ValueError("INPUT_SNAPSHOT_CHANGED")
        protocol = {
            "schema": "ta-historical-typed-funding-view.v1",
            "source_root": str(source),
            "source_receipts_sha256": bound.receipts_sha256,
            "input_identities": input_ids,
            "executed_source": executed,
            "source_sha256": {str(paths[name]): value for name, value in executed.items()},
            "used_input_identities_file": "used_input_identities.json",
            "runtime_python": sys.version,
            "executable_sha256": sha(sys.executable),
            "type_source": "REST_explicit_rateType_only",
            "months": MONTHS,
            "api_range_assumption_only": True,
            "independent_completeness_proven": False,
            "historical_eligibility_proven": False,
            "returns_evaluated": False,
            "models_called": False,
            "orders_enabled": False,
            "network_requests": False,
        }
        save(output / "protocol.json", protocol)
        save(
            output / "freeze_receipts.json",
            {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()},
        )
        bound.verify_all()
        plan = bound.load("capture_plan.json")
        validate_plan(plan)
        plan_sha256 = bound.expected["capture_plan.json"]
        if sha(paths["historical_rest.py"]) != plan["executed_source"]["historical_rest.py"]:
            raise ValueError("REPLAY_HELPER_DIFFERS_FROM_CAPTURE")
        original_rows = bound.load("funding_month_comparisons.json")
        original = {(r["symbol"], r["month"]): r for r in original_rows}
        if len(original_rows) != len(plan["symbols"]) * 2 or set(original) != {
            (s, m) for s in plan["symbols"] for m in MONTHS
        }:
            raise ValueError("ORIGINAL_MONTH_COVERAGE_INCOMPLETE")
        tasks = {
            (t["symbol"], t["month"]): t
            for t in plan["archive_tasks"]
            if t["kind"] == "fundingRate"
        }
        with (
            (output / "typed_cash_events.jsonl").open("x", encoding="utf-8") as event_stream,
            (output / "rejected_events.jsonl").open("x", encoding="utf-8") as rejected,
        ):
            for symbol in plan["symbols"]:
                symbol_errors = []
                try:
                    rest = replay_rest(bound, plan, plan_sha256, symbol)
                except Exception as error:
                    detail = {
                        "symbol": symbol,
                        "stage": "REST",
                        "error": type(error).__name__ + ":" + str(error)[:500],
                    }
                    errors.append(detail)
                    symbol_errors.append(detail)
                    rest = {"events": [], "source_consistent": False, "conflicts": set()}
                    rejected.write(canonical(rejected_source(bound, symbol, detail)) + "\n")
                month_coverage = {}
                for month in MONTHS:
                    local_errors = list(symbol_errors)
                    try:
                        archive = archive_rows(bound, tasks[symbol, month], plan_sha256)
                    except Exception as error:
                        detail = {
                            "symbol": symbol,
                            "month": month,
                            "stage": "ARCHIVE",
                            "error": type(error).__name__ + ":" + str(error)[:500],
                        }
                        errors.append(detail)
                        local_errors.append(detail)
                        archive = {"rows": None, "source_consistent": False}
                    row = compare_month(
                        symbol, month, original[symbol, month], archive, rest, local_errors
                    )
                    coverage.append(row)
                    month_coverage[month] = row
                for index, event in enumerate(rest["events"]):
                    month = next(
                        m for m in MONTHS if bounds(m)[0] <= event["fundingTime"] < bounds(m)[1]
                    )
                    row = {
                        "symbol": symbol,
                        "time_ms": event["fundingTime"],
                        "rate": str(decimal(event["fundingRate"], "RATE")),
                        "mark": str(decimal(event["markPrice"], "MARK")),
                        "type": event["rateType"],
                        "raw": event,
                        "source": {
                            "capture_receipts_sha256": bound.receipts_sha256,
                            "events_path": rest["record"]["events"],
                            "events_sha256": bound.expected[rest["record"]["events"]],
                            "event_index": index,
                            "raw_response_files": rest["raw_response_files"],
                        },
                        "quality": {
                            "explicit_type_verified": True,
                            "trusted_typed_fields": index not in rest["conflicts"],
                            "identity_conflict": index in rest["conflicts"],
                            "month_ready_under_source_assumptions": month_coverage[month][
                                "month_ready_under_source_assumptions"
                            ],
                            "independent_completeness_proven": False,
                            "historical_eligibility_proven": False,
                        },
                    }
                    event_stream.write(canonical(row) + "\n")
                    event_counts[event["rateType"]] += 1
        bound.verify_used()
        for name, expected in input_ids.items():
            if sha(name) != expected:
                raise ValueError("INPUT_HEADER_CHANGED")
        assert_stable(paths, executed)
        for relative, expected in read(output / "freeze_receipts.json").items():
            if sha(output / relative) != expected:
                raise ValueError("VIEW_FROZEN_BYTES_CHANGED:" + relative)
        save(
            output / "used_input_identities.json",
            {str(source / name): value for name, value in bound.used.items()},
        )
        ready = sum(row["month_ready_under_source_assumptions"] for row in coverage)
        result = {
            "schema": "ta-historical-typed-funding-result.v1",
            "status": "COMPLETED"
            if ready == len(coverage) and not errors
            else "COMPLETED_WITH_UNKNOWN",
            "symbols": len(plan["symbols"]),
            "symbol_months": len(coverage),
            "month_ready_under_source_assumptions": ready,
            "original_status_counts": dict(
                Counter(row["original_comparison"]["status"] for row in coverage)
            ),
            "regular_plus_special_status_counts": dict(
                Counter(row["regular_plus_special_status"] for row in coverage)
            ),
            "original_regular_mismatch_explained_by_special": sum(
                row["original_regular_mismatch_explained_by_special"] for row in coverage
            ),
            "typed_event_counts": dict(event_counts),
            "error_count": len(errors),
            "source_files_verified": len(bound.expected),
            "actually_used_input_count": len(bound.used),
            "independent_completeness_proven": False,
            "historical_eligibility_proven": False,
            "cash_history_all_symbols_proven": False,
            "returns_evaluated": False,
            "models_called": False,
            "orders_enabled": False,
        }
        save(output / "coverage.json", coverage)
    except Exception as error:
        errors.append(
            {"stage": "GLOBAL_INTEGRITY", "error": type(error).__name__ + ":" + str(error)[:500]}
        )
        for name in ("typed_cash_events.jsonl", "coverage.json"):
            path = output / name
            if path.exists():
                path.rename(output / ("partial_" + name))
        result = {
            "schema": "ta-historical-typed-funding-result.v1",
            "status": "FAILED_INTEGRITY",
            "error_count": len(errors),
            "independent_completeness_proven": False,
            "historical_eligibility_proven": False,
            "cash_history_all_symbols_proven": False,
            "returns_evaluated": False,
            "models_called": False,
            "orders_enabled": False,
        }
        if coverage and not (output / "partial_coverage.json").exists():
            save(output / "partial_coverage.json", coverage)
    save(output / "errors.json", errors)
    save(output / "result.json", result)
    save(
        output / "receipts.json",
        {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()},
    )
    return result


def main():
    """显式新输出入口；只做本地封存数据转换。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.output), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
