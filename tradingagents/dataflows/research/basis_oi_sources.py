"""单包index／metrics纯解析与精确时钟连接，不生成因子或交易输入。"""

from __future__ import annotations

import csv
import hashlib
import io
import re
import zipfile
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from numbers import Integral
from pathlib import PurePosixPath

MINUTE = 60_000
METRICS_DELAY_MS = 10 * MINUTE
INDEX_FIELDS = tuple("index_1m_" + name for name in ("open", "high", "low", "close"))
METRICS_FIELDS = (
    "sum_open_interest",
    "sum_open_interest_value",
    "count_toptrader_long_short_ratio",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
)
TIME_STATUS = {
    "indexPriceKlines": "INDEX_COMPLETION_ASSUMPTION",
    "metrics": "METRICS_T_PLUS_10M_ASSUMPTION",
}
VERIFIED = {"VERIFIED", "VERIFIED_BYTES_AND_PROFILE"}
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


class ArchiveFormatError(ValueError):
    """已绑定原文无法按该档案格式读取；由driver明确封存PARSE_FAILED。"""

    def __init__(self, message, *, member=None, member_sha256=None):
        super().__init__(message)
        self.member, self.member_sha256 = member, member_sha256


def _day_bounds(day):
    if not isinstance(day, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
        raise ValueError("来源日期必须为完整UTC日")
    parsed = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=UTC)
    first = (parsed - EPOCH).days * 86_400_000
    return first, first + 86_400_000


def _integer(text):
    value = text.strip().lstrip("\ufeff")
    if not re.fullmatch(r"\d+", value):
        return None
    try:
        result = int(value)
    except ValueError:
        return None
    return result if result <= 2**63 - 1 else None


def _decimal(text, *, positive=False):
    """精确规范化等值Decimal文本，不调用受上下文精度影响的normalize。"""
    if text is None or not text.strip():
        return None, "EMPTY", ["EMPTY_OR_MISSING_FIELD"]
    if not re.fullmatch(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", text.strip()):
        return None, "INVALID", ["INVALID_DECIMAL"]
    try:
        value = Decimal(text.strip())
    except (InvalidOperation, ValueError):
        return None, "INVALID", ["INVALID_DECIMAL"]
    if not value.is_finite() or (positive and value <= 0):
        return (
            None,
            "INVALID",
            ["NONFINITE_OR_NONPOSITIVE_PRICE" if positive else "NONFINITE_DECIMAL"],
        )
    if not positive and value < 0:
        return None, "INVALID", ["NEGATIVE_METRICS_VALUE"]
    if value == 0:
        return "0", "KNOWN_RAW_VALUE", []
    sign, digits, exponent = value.as_tuple()
    while digits[-1] == 0:
        digits, exponent = digits[:-1], exponent + 1
    return str(Decimal((sign, digits, exponent))), "KNOWN_RAW_VALUE", []


def _metrics_time(text):
    """无时区文本只按明确UTC假设解释；不舍入毫秒或五分钟钟点。"""
    value = text.strip()
    if not re.match(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}", value):
        return None, "INVALID_TIMESTAMP", None
    fractions = re.finditer(r"\d{2}:\d{2}:\d{2}[.,](\d+)", value)
    if any(char != "0" for fraction in fractions for char in fraction[1][3:]):
        return None, "NON_INTEGER_MILLISECOND", None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        assumption = "UTC" if parsed.tzinfo is None else None
        parsed = parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)
        delta = parsed - EPOCH
        if delta.microseconds % 1000:
            return None, "NON_INTEGER_MILLISECOND", None
        stamp = delta.days * 86_400_000 + delta.seconds * 1000 + delta.microseconds // 1000
    except (ValueError, OverflowError):
        return None, "INVALID_TIMESTAMP", None
    return stamp, None, assumption


def _csv_rows(content, member, member_sha):
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ArchiveFormatError(
            "CSV不是UTF-8，不能替换编码猜符号", member=member, member_sha256=member_sha
        ) from error
    lines = text.splitlines(keepends=True)
    reader = csv.reader(lines, strict=True)
    result, prior = [], 0
    try:
        for fields in reader:
            result.append(
                {
                    "source_line_no": prior + 1,
                    "source_line_end_no": reader.line_num,
                    "fields": fields,
                    "raw_text": "".join(lines[prior : reader.line_num]),
                }
            )
            prior = reader.line_num
    except csv.Error as error:
        raise ArchiveFormatError(
            "CSV结构损坏：" + str(error), member=member, member_sha256=member_sha
        ) from error
    if not result or not any(row["fields"] for row in result):
        raise ArchiveFormatError(
            "CSV为空，无法确定字段结构", member=member, member_sha256=member_sha
        )
    return result


def _field(fields, index):
    return fields[index] if index is not None and index < len(fields) else None


def _header(rows, kind, member, member_sha):
    first = rows[0]["fields"]
    names = [name.strip().lstrip("\ufeff").lower().replace(" ", "_") for name in first]
    if kind == "metrics":
        expected = {"create_time", "symbol", *METRICS_FIELDS}
        if len(names) != len(set(names)) or not expected <= set(names):
            raise ArchiveFormatError(
                "metrics头缺少唯一的时钟、符号或六原字段", member=member, member_sha256=member_sha
            )
        return rows[1:], names, {name: names.index(name) for name in expected}
    if names and names[0] == "open_time":
        required = {0: "open_time", 1: "open", 2: "high", 3: "low", 4: "close", 6: "close_time"}
        if len(names) != 12 or any(names[index] != name for index, name in required.items()):
            raise ArchiveFormatError(
                "index头不匹配固定12位置及价格／时钟字段", member=member, member_sha256=member_sha
            )
        return rows[1:], names, {}
    if names and (
        names[0] in {"timestamp", "time", "date"}
        or any(name in {"open", "high", "low", "close"} for name in names[1:5])
    ):
        raise ArchiveFormatError(
            "index存在无法辨认的头，不能按数据行猜列", member=member, member_sha256=member_sha
        )
    return rows, None, {}


def _member_identity(info, expected):
    """只接受逐字身份或可逆原UTF-8字节的唯一身份，不猜替代符号。"""
    original = info.orig_filename
    if "\x00" in original:
        raise ValueError("ZIP成员名包含NUL，不能降格为缺数据")
    raw_name = original.encode("utf-8" if info.flag_bits & 2048 else "cp437")
    interpretation = "UTF8_FLAG" if info.flag_bits & 2048 else "ORIGINAL_NAME_EXACT"
    member = original
    if member != expected and not info.flag_bits & 2048:
        try:
            candidate = raw_name.decode("utf-8")
        except UnicodeDecodeError:
            candidate = None
        if candidate == expected:
            member, interpretation = candidate, "UTF8_BYTES_WITHOUT_FLAG_EXACT_EXPECTED"
    path = PurePosixPath(member)
    if path.is_absolute() or ".." in path.parts or "\\" in member:
        raise ValueError("ZIP成员路径逃逸，不能降格为缺数据")
    if member != expected:
        raise ArchiveFormatError("ZIP成员不匹配原符号、类型及日期", member=original)
    return member, {
        "member_original_filename": original,
        "member_original_name_bytes_hex": raw_name.hex(),
        "member_name_interpretation": interpretation,
        "member_utf8_flag": bool(info.flag_bits & 2048),
    }


def parse_archive(payload, kind, symbol, day):
    """读取一个固定身份ZIP；逐字段未知和异常行保留，正常经济键稳定排序。"""
    if (
        kind not in TIME_STATUS
        or not isinstance(symbol, str)
        or not symbol
        or "/" in symbol
        or "\\" in symbol
    ):
        raise ValueError("档案kind或符号身份不合法")
    first, last = _day_bounds(day)
    expected = (
        f"{symbol}-1m-{day}.csv" if kind == "indexPriceKlines" else f"{symbol}-metrics-{day}.csv"
    )
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = archive.infolist()
            if len(members) != 1 or members[0].is_dir() or members[0].file_size > 64 * 1024 * 1024:
                raise ArchiveFormatError("ZIP必须恰为一个有界CSV成员")
            member, member_identity = _member_identity(members[0], expected)
            content = archive.read(members[0])
    except (zipfile.BadZipFile, RuntimeError, OSError) as error:
        raise ArchiveFormatError("ZIP结构或CRC无法读取：" + str(error)) from error
    member_sha = hashlib.sha256(content).hexdigest()
    rows, header, positions = _header(
        _csv_rows(content, member, member_sha), kind, member, member_sha
    )
    columns = INDEX_FIELDS if kind == "indexPriceKlines" else METRICS_FIELDS
    raw_records, anomalies, groups = [], [], defaultdict(list)
    previous, inversions = None, 0
    for raw in rows:
        fields = raw["fields"]
        codes = []
        if len(fields) != (12 if kind == "indexPriceKlines" else len(header)):
            codes.append("FIELD_COUNT_MISMATCH")
        values, statuses, reasons = {}, {}, {}
        for offset, name in enumerate(columns, 1):
            index = offset if kind == "indexPriceKlines" else positions[name]
            values[name], statuses[name], reasons[name] = _decimal(
                _field(fields, index), positive=kind == "indexPriceKlines"
            )
        if kind == "indexPriceKlines":
            known = {name: Decimal(value) for name, value in values.items() if value is not None}
            # 不能凭矛盾的OHLC猜哪一个价正确，因此四价均未知，原fields仍保留。
            pairs = (
                ("high", "open"),
                ("high", "low"),
                ("high", "close"),
                ("open", "low"),
                ("close", "low"),
            )
            if any(
                "index_1m_" + upper in known
                and "index_1m_" + lower in known
                and known["index_1m_" + upper] < known["index_1m_" + lower]
                for upper, lower in pairs
            ):
                codes.append("INVALID_OHLC_RELATIONSHIP")
                for name in columns:
                    values[name], statuses[name] = None, "INVALID"
                    reasons[name] = sorted({*reasons[name], "INVALID_OHLC_RELATIONSHIP"})
        if kind == "indexPriceKlines":
            text = _field(fields, 0)
            stamp = _integer(text) if text is not None else None
            assumption = None
            if stamp is None:
                codes.append("INVALID_TIMESTAMP")
            else:
                if not first <= stamp < last:
                    codes.append("OUTSIDE_FILE_DAY")
                if stamp % MINUTE:
                    codes.append("OFFGRID_TIMESTAMP")
                close_text = _field(fields, 6)
                if close_text is None or _integer(close_text) != stamp + MINUTE - 1:
                    codes.append("INVALID_CLOSE_TIME")
            key = stamp + MINUTE if stamp is not None else None
            actual_symbol = None
        else:
            text = _field(fields, positions["create_time"])
            stamp, error, assumption = _metrics_time(text or "")
            if error:
                codes.append(error)
            elif not first <= stamp < last:
                codes.append("OUTSIDE_FILE_DAY")
            elif stamp % (5 * MINUTE):
                codes.append("OFFGRID_TIMESTAMP")
            actual_symbol = _field(fields, positions["symbol"])
            if actual_symbol != symbol:
                codes.append("WRONG_SYMBOL")
            key = stamp
        if stamp is not None:
            inversions += previous is not None and stamp < previous
            previous = stamp
        identity_codes = [
            code
            for code in codes
            if code not in {"FIELD_COUNT_MISMATCH", "INVALID_OHLC_RELATIONSHIP"}
        ]
        record = {
            **raw,
            "raw_timestamp_text": text,
            "parsed_timestamp_ms": stamp,
            "raw_symbol": actual_symbol,
            "identity_status": "INVALID_IDENTITY" if identity_codes else "ON_GRID",
            "timezone_assumption": assumption,
            "values": values,
            "value_status": statuses,
            "value_reasons": reasons,
            "anomaly_codes": codes,
        }
        raw_records.append(record)
        if codes:
            anomalies.append(
                {
                    "source_line_no": raw["source_line_no"],
                    "parsed_timestamp_ms": stamp,
                    "reasons": codes,
                    "fields": fields,
                    "raw_timestamp_text": text,
                }
            )
        if not identity_codes:
            groups[key].append(record)
    normalized = {}
    for key, group in sorted(groups.items()):
        values, statuses, reasons = {}, {}, {}
        for name in columns:
            numeric = {
                Decimal(row["values"][name])
                for row in group
                if row["value_status"][name] == "KNOWN_RAW_VALUE"
            }
            if (
                all(row["value_status"][name] == "KNOWN_RAW_VALUE" for row in group)
                and len(numeric) == 1
            ):
                values[name], statuses[name], reasons[name] = (
                    group[0]["values"][name],
                    "KNOWN_RAW_VALUE",
                    [],
                )
            elif len(group) == 1:
                values[name], statuses[name], reasons[name] = (
                    None,
                    group[0]["value_status"][name],
                    group[0]["value_reasons"][name],
                )
            else:
                values[name], statuses[name] = None, "DUPLICATE_CONFLICT"
                reasons[name] = sorted(
                    {
                        "DUPLICATE_FIELD_NOT_UNANIMOUS",
                        *(reason for row in group for reason in row["value_reasons"][name]),
                    }
                )
        normalized[key] = {
            "symbol": symbol,
            "kind": kind,
            "day": day,
            "key_ms": key,
            "source_line_nos": [row["source_line_no"] for row in group],
            "raw_timestamp_texts": [row["raw_timestamp_text"] for row in group],
            "duplicate_count": len(group),
            "values": values,
            "value_status": statuses,
            "value_reasons": reasons,
            "time_status": TIME_STATUS[kind],
            **(
                {"index_bar_open_ms": key - MINUTE, "index_completed_ms": key}
                if kind == "indexPriceKlines"
                else {
                    "raw_T_ms": key,
                    "research_visible_at_ms": key + METRICS_DELAY_MS,
                    "timezone_assumption": "UTC"
                    if any(row["timezone_assumption"] == "UTC" for row in group)
                    else None,
                }
            ),
        }
    return {
        "member": member,
        "member_sha256": member_sha,
        **member_identity,
        "raw_records": raw_records,
        "normalized": normalized,
        "anomalies": anomalies,
        "profile": {
            **member_identity,
            "raw_records": len(raw_records),
            "normalized_keys": len(normalized),
            "anomaly_rows": len(anomalies),
            "anomaly_reasons": dict(Counter(code for row in anomalies for code in row["reasons"])),
            "invalid_identity_rows": sum(
                row["identity_status"] == "INVALID_IDENTITY" for row in raw_records
            ),
            "duplicate_extra_rows": sum(len(group) - 1 for group in groups.values()),
            "inverse_time_steps_in_input": inversions,
            "field_status_counts": {
                name: dict(Counter(row["value_status"][name] for row in normalized.values()))
                for name in columns
            },
            "minimum_key_ms": min(normalized) if normalized else None,
            "maximum_key_ms": max(normalized) if normalized else None,
            "index_placeholder_fields_excluded": kind == "indexPriceKlines",
            "first_publication_time_proven": False,
        },
    }


def _source_status(statuses, day):
    source = statuses.get(day)
    if source is None:
        return "OUTSIDE_SOURCE_RANGE", None
    if isinstance(source, str):
        return source, None
    status = source["status"]
    return status, source.get("parse_error") or source.get("error") or source.get("reason")


def _date_for_ms(stamp):
    return (EPOCH + timedelta(milliseconds=stamp)).date().isoformat()


def source_values_for_clock(clocks, index_records, metrics_records, index_status, metrics_status):
    """只连接t完成index及原T=t−10m的metrics，按请求日历保留未知行。"""
    symbols = {
        row["symbol"] for records in (index_records, metrics_records) for row in records.values()
    }
    if len(symbols) > 1:
        raise ValueError("一次来源连接只能处理同一符号，不得跨合约补值")
    result, previous = [], None
    for value in clocks:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError("快照时钟须为整数毫秒")
        clock = int(value)
        if clock < 0 or clock % (5 * MINUTE) or (previous is not None and clock <= previous):
            raise ValueError("快照时钟须严格递增且为UTC五分钟原格点")
        previous = clock
        row = {
            "clock_ms": clock,
            "requested_index_completed_ms": clock,
            "requested_metrics_raw_T_ms": clock - METRICS_DELAY_MS,
        }
        for prefix, kind, key, records, statuses, fields in (
            ("index", "indexPriceKlines", clock, index_records, index_status, INDEX_FIELDS),
            (
                "metrics",
                "metrics",
                clock - METRICS_DELAY_MS,
                metrics_records,
                metrics_status,
                METRICS_FIELDS,
            ),
        ):
            day = _date_for_ms(key - MINUTE if prefix == "index" else key)
            status, reason = _source_status(statuses, day)
            record = records.get(key)
            if record is not None and (
                record["kind"] != kind
                or record["key_ms"] != key
                or record["day"] != day
                or status not in VERIFIED
            ):
                raise ValueError("源经济键、文件日或来源状态与已解析记录冲突")
            row.update(
                {
                    prefix + "_source_day": day,
                    prefix + "_source_status": status,
                    prefix + "_source_reason": reason,
                    prefix + "_time_status": TIME_STATUS[kind],
                    prefix + "_source_line_nos": record["source_line_nos"] if record else [],
                    prefix + "_duplicate_count": record["duplicate_count"] if record else 0,
                }
            )
            for name in fields:
                row[name] = record["values"][name] if record else None
                row[name + "_value_status"] = (
                    record["value_status"][name] if record else "MISSING_CLOCK"
                )
                row[name + "_source_line_nos"] = record["source_line_nos"] if record else []
            if prefix == "index":
                row["index_bar_open_ms"] = record["index_bar_open_ms"] if record else None
                row["index_completed_ms"] = record["index_completed_ms"] if record else None
            else:
                row["raw_T_ms"] = record["raw_T_ms"] if record else None
                row["research_visible_at_ms"] = record["research_visible_at_ms"] if record else None
                row["metrics_timezone_assumption"] = (
                    record["timezone_assumption"] if record else None
                )
                row["metrics_economic_time_status"] = "UNKNOWN"
        row["first_publication_time_status"] = "UNKNOWN"
        result.append(row)
    return result
