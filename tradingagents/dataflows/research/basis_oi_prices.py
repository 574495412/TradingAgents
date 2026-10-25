"""绑定原748合约分钟原文并按完成钟保留未知，不生成因子、标签或收益。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import resource
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from numbers import Integral
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from tradingagents.research.quant.artifacts import save, sha

MINUTE = 60_000
DAY = 86_400_000
START = 1_790_553_600_000
END = 1_792_022_400_000
EXPECTED_SYMBOLS = 748
ROOT = Path(__file__).resolve().parents[3]
SOURCE_PROTOCOL = ROOT / "results/research/settlement_unwind_20261016_v001/protocol.json"
SOURCE_PROTOCOL_SHA = "70a9bd0ccd476d0b72f5c57e7cf1ed9c576b93ab541a3e4130b7f6bb0718e5f8"
DYNAMIC_ROOT = Path("data/research/damorne_15d_alignment_20261014/dynamic_market_panel/raw")
UNSEEN_ROOT = Path("results/research/profit_factor_search_20261015/unseen_20261014_v002/raw")
RAW_SCHEMA = "damorne-dynamic-public-kline-day.v1"
SCHEMA = "ta-basis-oi-price-source.v1"
FIELD_POSITIONS = {
    "contract_1m_open": 1,
    "contract_1m_high": 2,
    "contract_1m_low": 3,
    "contract_1m_close": 4,
    "base_volume": 5,
    "quote_volume": 7,
    "trade_count": 8,
    "taker_buy_base_volume": 9,
    "taker_buy_quote_volume": 10,
}
FIELDS = tuple(FIELD_POSITIONS)
FLAGS = {
    "labels_loaded": False,
    "models_loaded": False,
    "returns_evaluated": False,
    "orders_enabled": False,
    "leader_labels_read": False,
    "basis_generated": False,
    "factors_generated": False,
    "production_eligible": False,
}
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


class RawFormatError(ValueError):
    """文件结构无法解释时显式保留PARSE_FAILED，不代表原文身份通过。"""


def _loads(payload):
    """拒绝非有限JSON数及重复对象键，避免后值覆盖源身份。"""

    def invalid(value):
        raise RawFormatError("JSON包含非有限数：" + value)

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise RawFormatError("JSON对象键重复：" + key)
            result[key] = value
        return result

    try:
        return json.loads(payload, parse_constant=invalid, object_pairs_hook=pairs)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RawFormatError("JSON结构无法解释") from error


def _day(stamp):
    return (EPOCH + timedelta(milliseconds=int(stamp))).date().isoformat()


def _day_start(day):
    if not isinstance(day, str) or not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", day):
        raise ValueError("原文件日必须是完整UTC日期")
    return (datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=UTC) - EPOCH).days * DAY


def _is_int(value):
    return isinstance(value, Integral) and not isinstance(value, bool)


def _value(raw, field):
    """原价格和量须是文本，只有原笔数使用整数，不把浮点价冒充原文。"""
    if field == "trade_count":
        if not _is_int(raw) or raw < 0:
            return None, "INVALID", ["INVALID_OR_NEGATIVE_INTEGER_COUNT"]
        return str(raw), "KNOWN_RAW_VALUE", []
    if raw is None or raw == "":
        return None, "EMPTY", ["EMPTY_OR_MISSING_FIELD"]
    if not isinstance(raw, str):
        return None, "INVALID", ["RAW_NUMERIC_TEXT_REQUIRED"]
    text = raw.strip()
    if not text:
        return None, "EMPTY", ["EMPTY_OR_MISSING_FIELD"]
    if not re.fullmatch(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", text):
        return None, "INVALID", ["INVALID_DECIMAL"]
    try:
        value = Decimal(text)
    except InvalidOperation:
        return None, "INVALID", ["INVALID_DECIMAL"]
    if not value.is_finite() or value < 0 or (field.startswith("contract_") and value == 0):
        return None, "INVALID", ["NONPOSITIVE_PRICE_OR_NEGATIVE_QUANTITY"]
    if value == 0:
        return "0", "KNOWN_RAW_VALUE", []
    sign, digits, exponent = value.as_tuple()
    while digits[-1] == 0:
        digits, exponent = digits[:-1], exponent + 1
    return str(Decimal((sign, digits, exponent))), "KNOWN_RAW_VALUE", []


def _consensus(group):
    """所有重复记录逐字段一致才有数值，不使用首末、均值或float。"""
    if len(group) == 1:
        return group[0]["values"], group[0]["value_status"], group[0]["value_reasons"]
    values, statuses, reasons = {}, {}, {}
    for field in FIELDS:
        known = [r["values"][field] for r in group if r["value_status"][field] == "KNOWN_RAW_VALUE"]
        if len(known) == len(group) and len({Decimal(v) for v in known}) == 1:
            values[field], statuses[field], reasons[field] = known[0], "KNOWN_RAW_VALUE", []
        elif len(group) == 1:
            values[field], statuses[field], reasons[field] = (
                None,
                group[0]["value_status"][field],
                group[0]["value_reasons"][field],
            )
        else:
            values[field], statuses[field] = None, "DUPLICATE_CONFLICT"
            reasons[field] = sorted(
                {
                    "DUPLICATE_FIELD_NOT_UNANIMOUS",
                    *(x for row in group for x in row["value_reasons"][field]),
                }
            )
    return values, statuses, reasons


def parse_raw(payload, symbol, day):
    """单日原JSON有界解析；保留缺口和异常，不读取原研究的标签或结果。"""
    first = _day_start(day)
    source = _loads(payload)
    if (
        not isinstance(source, dict)
        or source.get("schema") != RAW_SCHEMA
        or not isinstance(source.get("rows"), list)
    ):
        raise RawFormatError("原分钟文件schema或rows结构不匹配")
    if source.get("symbol") != symbol:
        raise ValueError("原分钟symbol身份漂移")
    rows = source["rows"]
    calculated = hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()
    if source.get("rows_sha256") != calculated:
        raise ValueError("原rows内容摘要不匹配")
    left, right = source.get("start_ms"), source.get("end_ms")
    if (
        not _is_int(left)
        or not _is_int(right)
        or left % MINUTE
        or right % MINUTE
        or not left < right
    ):
        raise ValueError("原分钟声明区间身份不合法")
    groups, anomalies = defaultdict(list), []
    previous, inversions, invalid_identity = None, 0, 0
    for number, raw in enumerate(rows, 1):
        if not isinstance(raw, list):
            anomalies.append({"source_row_no": number, "reasons": ["ROW_NOT_LIST"], "raw": raw})
            invalid_identity += 1
            continue
        codes = []
        if len(raw) != 12:
            codes.append("FIELD_COUNT_MISMATCH")
        stamp = raw[0] if raw else None
        valid_clock = _is_int(stamp) and stamp >= 0 and stamp % MINUTE == 0
        if not valid_clock:
            codes.append("INVALID_OR_OFFGRID_OPEN_TIME")
        else:
            if not first <= stamp < first + DAY or not left <= stamp < right:
                codes.append("OUTSIDE_FILE_DAY_OR_DECLARED_RANGE")
            if len(raw) <= 6 or not _is_int(raw[6]) or raw[6] != stamp + MINUTE - 1:
                codes.append("INVALID_CLOSE_TIME")
            inversions += previous is not None and stamp < previous
            previous = stamp
        values, statuses, reasons = {}, {}, {}
        for field, position in FIELD_POSITIONS.items():
            values[field], statuses[field], reasons[field] = _value(
                raw[position] if len(raw) > position else None, field
            )
        known = {field: Decimal(values[field]) for field in FIELDS[:4] if values[field] is not None}
        relations = (
            ("high", "open"),
            ("high", "low"),
            ("high", "close"),
            ("open", "low"),
            ("close", "low"),
        )
        if any(
            "contract_1m_" + a in known
            and "contract_1m_" + b in known
            and known["contract_1m_" + a] < known["contract_1m_" + b]
            for a, b in relations
        ):
            codes.append("INVALID_OHLC_RELATIONSHIP")
            for field in FIELDS[:4]:
                values[field], statuses[field] = None, "INVALID"
                reasons[field] = sorted({*reasons[field], "INVALID_OHLC_RELATIONSHIP"})
        invalid_codes = [
            c for c in codes if c not in {"FIELD_COUNT_MISMATCH", "INVALID_OHLC_RELATIONSHIP"}
        ]
        if codes or any(status != "KNOWN_RAW_VALUE" for status in statuses.values()):
            anomalies.append(
                {
                    "source_row_no": number,
                    "raw_open_ms": stamp,
                    "reasons": codes,
                    "value_status": statuses,
                    "value_reasons": reasons,
                    "raw": raw,
                }
            )
        if invalid_codes:
            invalid_identity += 1
            continue
        groups[int(stamp) + MINUTE].append(
            {
                "source_refs": [f"{day}:{number}"],
                "values": values,
                "value_status": statuses,
                "value_reasons": reasons,
            }
        )
    records = {}
    for clock, group in sorted(groups.items()):
        values, statuses, reasons = _consensus(group)
        if "DUPLICATE_CONFLICT" in statuses.values():
            source_rows = [int(ref.split(":")[-1]) for row in group for ref in row["source_refs"]]
            anomalies.append(
                {
                    "source_row_nos": source_rows,
                    "clock_ms": clock,
                    "reasons": ["DUPLICATE_CONFLICT"],
                    "value_reasons": reasons,
                    "raw": [rows[number - 1] for number in source_rows],
                }
            )
        records[clock] = {
            "symbol": symbol,
            "day": day,
            "clock_ms": clock,
            "open_ms": clock - MINUTE,
            "values": values,
            "value_status": statuses,
            "value_reasons": reasons,
            "source_refs": [ref for row in group for ref in row["source_refs"]],
            "duplicate_count": len(group),
        }
    valid_identity = sum(len(group) for group in groups.values())
    if valid_identity + invalid_identity != len(rows):
        raise AssertionError("原记录身份守恒失败")
    return {
        "symbol": symbol,
        "day": day,
        "source_status": "VERIFIED_RAW_BYTES",
        "records": records,
        "anomalies": anomalies,
        "profile": {
            "raw_rows": len(rows),
            "valid_identity_rows": valid_identity,
            "invalid_identity_rows": invalid_identity,
            "normalized_keys": len(records),
            "duplicate_extra_rows": valid_identity - len(records),
            "inverse_time_steps_in_input": inversions,
            "anomaly_entries": len(anomalies),
            "field_status_counts": {
                field: dict(Counter(row["value_status"][field] for row in records.values()))
                for field in FIELDS
            },
            "minimum_completed_ms": min(records) if records else None,
            "maximum_completed_ms": max(records) if records else None,
        },
    }


def _canonical(path):
    """只读当前TA规范路径，不接受软链接或外部历史路径回退。"""
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts or not path.is_relative_to(ROOT):
        raise ValueError("原分钟路径不在当前TA所有权树")
    if any(p.is_symlink() for p in (path, *path.parents) if p.is_relative_to(ROOT)):
        raise ValueError("原分钟路径包含软链接")
    if path.resolve() != path or not path.is_file():
        raise ValueError("原分钟路径不是既有规范文件")
    return path


class BoundPrices:
    """固定748×17原文件身份，首末全量核验，消费时再按字节核验。"""

    def __init__(self, protocol_path):
        self.protocol_path = _canonical(protocol_path)
        if self.protocol_path != SOURCE_PROTOCOL:
            raise ValueError("只接受计划固定位置的H01来源protocol")
        if sha(self.protocol_path) != SOURCE_PROTOCOL_SHA:
            raise ValueError("H01来源protocol不是冻结版本")
        protocol = _loads(self.protocol_path.read_bytes())
        if protocol.get("schema") != "ta-native-settlement-unwind-study.v1":
            raise ValueError("H01来源protocol schema不匹配")
        self.universe = protocol["universe"]
        self.symbols = sorted(self.universe)
        if len(self.symbols) != EXPECTED_SYMBOLS or any(
            not s or "/" in s or "\\" in s for s in self.symbols
        ):
            raise ValueError("原748身份数或符号不匹配")
        days = [_day(t) for t in range(START, END, DAY)]
        if not days or (END - START) % DAY:
            raise ValueError("固定分钟日历不合法")
        expected = {}
        for symbol in self.symbols:
            for day in days:
                base = UNSEEN_ROOT if day == "2026-10-14" else DYNAMIC_ROOT
                expected[str(ROOT / base / symbol / (day + ".json"))] = (symbol, day)
        identities = protocol["input_identities"]
        selected = {
            k: v
            for k, v in identities.items()
            if "/dynamic_market_panel/raw/" in k or "/unseen_20261014_v002/raw/" in k
        }
        if set(selected) != set(expected):
            raise ValueError("原市场依赖并非完整748×17逐日身份")
        self.inputs = {str(self.protocol_path): SOURCE_PROTOCOL_SHA, **selected}
        self.sources = {symbol: {} for symbol in self.symbols}
        for path, (symbol, day) in expected.items():
            _canonical(path)
            self.sources[symbol][day] = path
        self.verify_after()

    def consume(self, symbol, day):
        """消费绑定原字节，不沿reused_sources追读额外路径。"""
        path = self.sources[symbol][day]
        payload = _canonical(path).read_bytes()
        if hashlib.sha256(payload).hexdigest() != self.inputs[path]:
            raise ValueError("消费原分钟字节发生漂移")
        try:
            parsed = parse_raw(payload, symbol, day)
        except RawFormatError as error:
            parsed = {
                "symbol": symbol,
                "day": day,
                "source_status": "PARSE_FAILED",
                "records": {},
                "anomalies": [],
                "profile": {"parse_error": str(error)},
            }
        parsed["source_file"], parsed["source_sha256"] = path, self.inputs[path]
        for record in parsed["records"].values():
            record["source_refs"] = [path + "#" + ref for ref in record["source_refs"]]
        return parsed

    def verify_after(self):
        """包括无数据和最终边界原文，不能只核验已产生价格的成功行。"""
        for path, expected in self.inputs.items():
            if sha(_canonical(path)) != expected:
                raise ValueError("构建前后原分钟依赖发生漂移")


def bind_inputs(protocol_path=SOURCE_PROTOCOL):
    """生产入口无人口、时段或SHA覆盖参数。"""
    return BoundPrices(protocol_path)


def minute_rows(symbol, parsed_sources, metadata, clocks):
    """生成完整分钟人口，资格仅另列假设，不遮掉已观察价格或身份。"""
    sources, groups = {}, defaultdict(list)
    for source in parsed_sources:
        if source["symbol"] != symbol:
            raise ValueError("不能跨symbol附接价格")
        day, status = source["day"], source["source_status"]
        if day in sources and sources[day] != status:
            raise ValueError("同文件日来源状态冲突")
        if status not in {"VERIFIED_RAW_BYTES", "PARSE_FAILED"} or (
            status == "PARSE_FAILED" and source["records"]
        ):
            raise ValueError("价格来源状态与解析记录冲突")
        sources[day] = status
        for key, record in source["records"].items():
            if (
                record["symbol"] != symbol
                or record["day"] != day
                or record["clock_ms"] != key
                or _day(key - MINUTE) != day
            ):
                raise ValueError("规范原记录的经济键或文件日不匹配")
            groups[key].append(record)
    previous = None
    for value in clocks:
        if (
            not _is_int(value)
            or value < 0
            or value % MINUTE
            or (previous is not None and value <= previous)
        ):
            raise ValueError("完成钟须是严格递增整数分钟，不得舍入")
        clock = int(value)
        previous = clock
        day = _day(clock - MINUTE)
        group = groups.get(clock, [])
        if group:
            values, statuses, _ = _consensus(group)
        else:
            values = dict.fromkeys(FIELDS)
            statuses = dict.fromkeys(FIELDS, "MISSING_CLOCK")
        onboard, delivery = metadata.get("onboard_ms"), metadata.get("delivery_ms")
        dates_known = _is_int(onboard) and _is_int(delivery) and onboard < delivery
        row = {
            "symbol": symbol,
            "clock_ms": clock,
            "bar_open_ms": clock - MINUTE,
            "observed": bool(group),
            "price_source_day": day,
            "price_source_status": sources.get(day, "NO_BOUND_SOURCE"),
            "price_source_refs": [ref for record in group for ref in record["source_refs"]],
            "price_duplicate_count": sum(record["duplicate_count"] for record in group),
            "price_time_status": "RAW_OPEN_PLUS_1M_COMPLETION_ASSUMPTION",
            "price_first_publication_status": "UNKNOWN",
            "asset_class": metadata.get("asset_class", "UNKNOWN"),
            "captured_status": metadata.get("captured_status"),
            "onboard_ms": onboard if _is_int(onboard) else None,
            "delivery_ms": delivery if _is_int(delivery) else None,
            "historical_pti_proven": False,
            "late_static_qualified_at_open": onboard <= clock - MINUTE < delivery
            if dates_known
            else None,
            "late_static_qualification_status": "KNOWN_UNDER_LATE_STATIC_ASSUMPTION"
            if dates_known
            else "UNKNOWN",
        }
        for field in FIELDS:
            row[field], row[field + "_value_status"] = values[field], statuses[field]
        yield row


def _schema():
    columns = [
        ("symbol", pa.string()),
        ("clock_ms", pa.int64()),
        ("bar_open_ms", pa.int64()),
        ("observed", pa.bool_()),
    ]
    columns += [(name, pa.string()) for name in ("price_source_day", "price_source_status")]
    columns += [("price_source_refs", pa.list_(pa.string())), ("price_duplicate_count", pa.int64())]
    columns += [
        (name, pa.string())
        for name in (
            "price_time_status",
            "price_first_publication_status",
            "asset_class",
            "captured_status",
        )
    ]
    columns += [(name, pa.int64()) for name in ("onboard_ms", "delivery_ms")]
    columns += [
        ("historical_pti_proven", pa.bool_()),
        ("late_static_qualified_at_open", pa.bool_()),
        ("late_static_qualification_status", pa.string()),
    ]
    columns += [
        (name, pa.string()) for field in FIELDS for name in (field, field + "_value_status")
    ]
    return pa.schema(columns)


def _execution_sources():
    from tradingagents.research.quant import artifacts

    return [
        Path(__file__).resolve(),
        Path(__file__).with_name("basis_oi_prices_plan.md").resolve(),
        Path(artifacts.__file__).resolve(),
    ]


def _seal(output):
    save(
        output / "receipts.json",
        {
            str(p.relative_to(output)): sha(p)
            for p in sorted(output.rglob("*"))
            if p.is_file() and p.name != "receipts.json"
        },
    )


def run(protocol_path, output):
    """仅在新目录执行完整来源构建，失败保留部分原证据与失败收据。"""
    output = Path(output).resolve()
    protected = (
        (ROOT / DYNAMIC_ROOT).parent,
        (ROOT / UNSEEN_ROOT).parent,
        SOURCE_PROTOCOL.parent,
        ROOT / "tradingagents",
    )
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError("价格新输出不能覆盖、包含或嵌入原来源和包源码")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    writer = None
    try:
        execution = {str(p): sha(p) for p in _execution_sources()}
        bound = bind_inputs(protocol_path)
        snapshot = output / "source"
        snapshot.mkdir()
        for number, (path, expected_sha) in enumerate(execution.items()):
            target = snapshot / f"{number:02d}_{Path(path).name}"
            target.write_bytes(Path(path).read_bytes())
            if sha(target) != expected_sha:
                raise ValueError("执行依赖复制期间字节发生漂移")
        clocks = range(START, END, MINUTE)
        protocol = {
            "schema": SCHEMA,
            **FLAGS,
            "source_protocol": str(bound.protocol_path),
            "source_protocol_sha256": SOURCE_PROTOCOL_SHA,
            "source_sha256": execution,
            "symbols": bound.symbols,
            "universe": bound.universe,
            "start_ms": START,
            "end_ms": END,
            "minute_count_per_symbol": len(clocks),
            "expected_rows": len(clocks) * len(bound.symbols),
            "input_count": len(bound.inputs) - 1,
            "numeric_representation": "EXACT_DECIMAL_CANONICAL_TEXT",
            "runtime": {"python": sys.version, "pyarrow": pa.__version__},
            "raw_source_row_numbering": "ONE_BASED_INDEX_IN_ORIGINAL_ROWS_ARRAY",
            "price_time_assumption": "RAW_OPEN_PLUS_1M_COMPLETION_ASSUMPTION",
            "historical_pti_proven": False,
            "first_publication_time_proven": False,
            "cross_source_monthly_archives_merged": False,
        }
        save(output / "protocol.json", protocol)
        save(output / "input_identities.json", bound.inputs)
        schema = _schema()
        writer = pq.ParquetWriter(output / "minutes.parquet", schema, compression="zstd")
        field_counts = {field: Counter() for field in FIELDS}
        by_symbol, by_day, source_counts = {}, {}, Counter()
        total, observed, raw_count, invalid_identity, duplicates, outside = 0, 0, 0, 0, 0, 0
        with (
            (output / "source_profiles.jsonl").open("x", encoding="utf-8") as profiles,
            (output / "anomalies.jsonl").open("x", encoding="utf-8") as anomalies,
        ):
            for number, symbol in enumerate(bound.symbols, 1):
                parsed = []
                for day in sorted(bound.sources[symbol]):
                    source = bound.consume(symbol, day)
                    parsed.append(source)
                    source_counts[source["source_status"]] += 1
                    profile = source["profile"]
                    raw_count += profile.get("raw_rows", 0)
                    invalid_identity += profile.get("invalid_identity_rows", 0)
                    duplicates += profile.get("duplicate_extra_rows", 0)
                    outside += sum(not START <= key < END for key in source["records"])
                    context = {
                        k: source[k]
                        for k in ("symbol", "day", "source_file", "source_sha256", "source_status")
                    }
                    profiles.write(
                        json.dumps(
                            {**context, "profile": profile}, ensure_ascii=False, allow_nan=False
                        )
                        + "\n"
                    )
                    for item in source["anomalies"]:
                        anomalies.write(
                            json.dumps({**context, **item}, ensure_ascii=False, allow_nan=False)
                            + "\n"
                        )
                batch, count, known = [], 0, Counter()
                for row in minute_rows(symbol, parsed, bound.universe[symbol], clocks):
                    total += 1
                    count += 1
                    observed += row["observed"]
                    known["observed"] += row["observed"]
                    day_counts = by_day.setdefault(
                        _day(row["clock_ms"]),
                        {
                            "rows": 0,
                            "observed": 0,
                            "field_status_counts": {field: Counter() for field in FIELDS},
                        },
                    )
                    day_counts["rows"] += 1
                    day_counts["observed"] += row["observed"]
                    for field in FIELDS:
                        status = row[field + "_value_status"]
                        if (row[field] is not None) != (status == "KNOWN_RAW_VALUE"):
                            raise AssertionError("价格值与字段状态不守恒")
                        field_counts[field][status] += 1
                        day_counts["field_status_counts"][field][status] += 1
                        known[field] += status == "KNOWN_RAW_VALUE"
                    batch.append(row)
                    if len(batch) == 1440:
                        writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                        batch.clear()
                if batch:
                    writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                if count != len(clocks):
                    raise AssertionError("单符号完整分钟人口不守恒")
                by_symbol[symbol] = {"rows": count, **known}
                if number % 25 == 0 or number == len(bound.symbols):
                    print(
                        json.dumps(
                            {
                                "completed_symbols": number,
                                "symbols": len(bound.symbols),
                                "rows": total,
                                "elapsed_seconds": round(time.monotonic() - started, 2),
                            }
                        ),
                        flush=True,
                    )
        writer.close()
        writer = None
        expected = len(bound.symbols) * len(clocks)
        if total != expected or any(
            sum(counts.values()) != total for counts in field_counts.values()
        ):
            raise AssertionError("全分钟／逐字段人口不守恒")
        if pq.read_metadata(output / "minutes.parquet").num_rows != total:
            raise AssertionError("实际价格Parquet行数与构建记录不同")
        if any(row["rows"] != len(bound.symbols) * 1440 for row in by_day.values()):
            raise AssertionError("完整17日人口不守恒")
        if raw_count != observed + outside + invalid_identity + duplicates:
            raise AssertionError("原记录、正常分钟、重复、异常及区间外完成钟不守恒")
        if sum(source_counts.values()) != len(bound.inputs) - 1:
            raise AssertionError("完整原文件状态不守恒")
        bound.verify_after()
        for path, expected_sha in execution.items():
            if sha(path) != expected_sha:
                raise ValueError("实际执行源码或计划在构建期间漂移")
        coverage = {
            "rows": total,
            "observed": observed,
            "unobserved": total - observed,
            "raw_rows_in_parseable_files": raw_count,
            "invalid_identity_rows": invalid_identity,
            "duplicate_extra_rows": duplicates,
            "normalized_completed_keys_outside_panel": outside,
            "source_file_statuses": dict(source_counts),
            "field_status_counts": field_counts,
            "by_symbol": by_symbol,
            "by_clock_day": by_day,
            "raw_counts_include_parse_failed_files": False,
        }
        save(output / "coverage.json", coverage)
        result = {
            "schema": SCHEMA + ".result",
            "status": "COMPLETED_WITH_UNKNOWN_RETAINED",
            **FLAGS,
            "protocol_sha256": sha(output / "protocol.json"),
            "rows": total,
            "symbols": len(bound.symbols),
            "input_raw_files": len(bound.inputs) - 1,
            "source_file_statuses": dict(source_counts),
            "elapsed_seconds": time.monotonic() - started,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
        }
        save(output / "result.json", result)
        _seal(output)
        return result
    except Exception as error:
        if writer is not None:
            writer.close()
        save(
            output / "failure.json",
            {
                "schema": SCHEMA + ".failure",
                "status": "FAILED",
                **FLAGS,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        _seal(output)
        raise


def main():
    """命令行仅可指定冻结protocol路径和全新输出目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=SOURCE_PROTOCOL)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.protocol, args.output), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
