"""固定九〇九合约历史原文采集；只校验数据，不计算收益或生成订单。"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import re
import shutil
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack, suppress
from datetime import UTC, datetime
from decimal import Decimal
from email.utils import parsedate_to_datetime
from pathlib import Path

if __package__:
    from .historical_rest import capture_pages, compare_month
else:
    from historical_rest import capture_pages, compare_month

HERE = Path(__file__).resolve().parent
BASE = "https://data.binance.vision/"
FUNDING = "https://fapi.binance.com/fapi/v1/fundingRate"
MONTHS = ("2026-08", "2026-09")
START_MS, END_MS = 1785542400000, 1790812800000
MINUTE = 60_000
GIB = 1024**3
NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
COLUMNS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
)


def read(path):
    """非有限JSON不算有效采集状态。"""

    def invalid(value):
        raise ValueError(f"非法JSON常量：{value}")

    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=invalid)


def sha(path):
    """按原字节计算身份，不依赖业务字段。"""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_exclusive(path, payload):
    """完整写入暂存后原子排他发布；崩溃暂存保留，绝不覆盖既有原文。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.tmp"
    )
    with pending.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def save(path, value):
    """状态、失败和原文收据排他写入，重试另建编号。"""
    write_exclusive(
        path,
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2).encode(
            "utf-8"
        ),
    )


def verify_seal(root):
    """探测已封存目录必须文件集合及全部摘要一致。"""
    root = Path(root).resolve()
    receipt = read(root / "receipts.json")
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()} - {"receipts.json"}
    if actual != set(receipt):
        raise ValueError("来源封存集合变化")
    for name, checksum in receipt.items():
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root) or sha(path) != checksum:
            raise ValueError(f"来源封存字节变化：{name}")
    return len(receipt)


def month_bounds(month):
    """月份右开端点使用UTC日历，不猜分钟或结算数量。"""
    if month not in MONTHS:
        raise ValueError("月份不属于已冻结两个月")
    year, number = map(int, month.split("-"))
    start = datetime(year, number, 1, tzinfo=UTC)
    end = datetime(year + (number == 12), number % 12 + 1, 1, tzinfo=UTC)
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)


def archive_key(kind, symbol, month):
    """仅固定官方两类路径，不接受主机、路径或查询注入。"""
    if kind not in {"1m", "fundingRate"} or (
        not isinstance(symbol, str)
        or not symbol.endswith("USDT")
        or not all(c.isalnum() or c == "_" for c in symbol)
    ):
        raise ValueError("归档类型或符号不合法")
    month_bounds(month)
    prefix = "data/futures/um/monthly/"
    if kind == "1m":
        return prefix + f"klines/{symbol}/1m/{symbol}-1m-{month}.zip"
    return prefix + f"fundingRate/{symbol}/{symbol}-fundingRate-{month}.zip"


def archive_url(key):
    """Unicode合约名原样保留在任务，HTTP路径仅作UTF-8百分号编码。"""
    return BASE + urllib.parse.quote(key, safe="/")


def build_tasks(original, inventory, metadata):
    """保留并集全部symbol-month，不因未列出或状态不佳删除成员。"""
    original = set(original)
    symbols = sorted(original | set(inventory))
    selected = {}
    for row in metadata:
        if row["kind"] not in {"1m", "fundingRate"} or row["month"] not in MONTHS:
            continue
        identity = row["kind"], row["symbol"], row["month"]
        if (
            row["symbol"] not in symbols
            or row["key"] != archive_key(*identity)
            or identity in selected
        ):
            raise ValueError("元数据包含越界或重复归档键")
        if type(row["bytes"]) is not int or row["bytes"] <= 0:
            raise ValueError("已列归档大小不合法")
        selected[identity] = row
    tasks = []
    for symbol in symbols:
        for month in MONTHS:
            for kind in ("1m", "fundingRate"):
                row = selected.get((kind, symbol, month))
                listed = row is not None
                tasks.append(
                    {
                        "id": f"{kind}_{symbol}_{month}",
                        "symbol": symbol,
                        "month": month,
                        "kind": kind,
                        "original_748_member": symbol in original,
                        "key": archive_key(kind, symbol, month),
                        "listed": listed,
                        "checksum_key_listed": bool(row and row["checksum_key_listed"]),
                        "listed_bytes": row["bytes"] if listed else None,
                        "listing_last_modified": row["last_modified"] if listed else None,
                        "initial_status": "PENDING" if listed else "NOT_LISTED",
                        "historical_tradability": "UNKNOWN",
                    }
                )
    return symbols, tasks


def listing_keys(probe, kind, symbol):
    """再核对探测原始XML；只把完整已列出的键交给下载器。"""
    folder = probe / "listing" / kind / symbol
    pages = sorted(folder.glob("page_*.xml"))
    if not pages:
        raise ValueError("缺少固定符号的清单原文")
    found = {}
    expected_prefix = archive_key(kind, symbol, MONTHS[0]).rsplit("/", 1)[0] + "/"
    for index, path in enumerate(pages):
        if path.name != f"page_{index:03}.xml":
            raise ValueError("清单页编号不连续")
        receipt = read(path.with_suffix(path.suffix + ".receipt.json"))
        if receipt["status"] != 200 or sha(path) != receipt["sha256"]:
            raise ValueError("清单原文未验证")
        tree = ET.fromstring(path.read_bytes())
        if tree.findtext("s:Prefix", namespaces=NS) != expected_prefix:
            raise ValueError("清单前缀与固定任务不同")
        truncated = tree.findtext("s:IsTruncated", namespaces=NS)
        if truncated != ("false" if index == len(pages) - 1 else "true"):
            raise ValueError("清单分页不完整")
        for item in tree.findall("s:Contents", NS):
            key = item.findtext("s:Key", namespaces=NS)
            if not key.startswith(expected_prefix) or key in found:
                raise ValueError("清单键越界或重复")
            found[key] = int(item.findtext("s:Size", namespaces=NS))
    return found


def prepare(probe, output, *, context_root=None, eligibility=None):
    """离线冻结探测、九〇九任务、实际执行源码与仅供背景的整包快照。"""
    probe, output = Path(probe).resolve(), Path(output).resolve()
    if output.is_relative_to(probe) or probe.is_relative_to(output):
        raise ValueError("新采集不能覆盖探测来源")
    count = verify_seal(probe)
    protocol, result, inventory = (
        read(probe / n) for n in ("protocol.json", "result.json", "symbol_inventory.json")
    )
    if any(
        protocol[k] is not False
        for k in ("orders_enabled", "leader_labels_read", "price_returns_evaluated")
    ):
        raise ValueError("探测不是纯公开无收益研究")
    symbols, tasks = build_tasks(
        protocol["original_universe"],
        inventory["archive_usdt_symbols"],
        read(probe / "archive_metadata.json"),
    )
    if (
        len(protocol["original_universe"]) != 748
        or len(symbols) != 909
        or result["metadata_logical_jobs"] != 2727
        or result["metadata_failed_jobs"]
    ):
        raise ValueError("必须完整保留已探测的九〇九符号及二七二七清单")
    for symbol in symbols:
        for kind in ("1m", "fundingRate"):
            keys = listing_keys(probe, kind, symbol)
            for task in (r for r in tasks if r["symbol"] == symbol and r["kind"] == kind):
                if task["listed"] != (task["key"] in keys):
                    raise ValueError("派生元数据与清单原文不符")
                if task["listed"] and (
                    keys[task["key"]] != task["listed_bytes"]
                    or task["checksum_key_listed"] != (task["key"] + ".CHECKSUM" in keys)
                ):
                    raise ValueError("归档大小或校验键与清单不符")
    output.mkdir(parents=True, exist_ok=False)
    total_bytes = sum(t["listed_bytes"] or 0 for t in tasks)
    free = shutil.disk_usage(output).free
    if free < max(20 * GIB, total_bytes * 6):
        raise ValueError("磁盘不足以保留原文、规范表、暂存和审计余量")
    sources = {
        "probe": str(probe),
        "probe_receipts_sha256": sha(probe / "receipts.json"),
        "probe_verified_files": count,
    }
    for name in (
        "receipts.json",
        "protocol.json",
        "result.json",
        "symbol_inventory.json",
        "archive_metadata.json",
        "design_notes.md",
    ):
        target = output / "inputs" / ("probe_" + name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(probe / name, target)
    if eligibility is not None:
        eligibility = Path(eligibility).resolve()
        sources["eligibility_receipts_sha256"] = sha(eligibility / "receipts.json")
        sources["eligibility_verified_files"] = verify_seal(eligibility)
        for name in (
            "receipts.json",
            "result.json",
            "symbol_metadata.json",
            "bake_hifi_notice_crosscheck.json",
        ):
            shutil.copyfile(eligibility / name, output / "inputs" / ("eligibility_" + name))
    executed = {}
    for name in ("historical_capture.py", "historical_rest.py", "historical_capture_plan.md"):
        source, target = HERE / name, output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        executed[name] = sha(target)
        if sha(source) != executed[name]:
            raise ValueError("冻结实际执行源码时发生变化")
    context_root = (
        Path(context_root).resolve()
        if context_root
        else next(p for p in HERE.parents if (p / "tradingagents").is_dir())
    )
    context = {}
    paths = sorted(
        set((context_root / "tradingagents").rglob("*.py"))
        | set((context_root / "tradingagents/research/quant/plans").glob("*.md"))
    )
    for source in paths:
        relative = source.relative_to(context_root)
        target = output / "context_source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        before = sha(source)
        shutil.copyfile(source, target)
        if sha(target) != before:
            raise ValueError("整包背景归档复制不一致")
        context[str(relative)] = before
    plan = {
        "schema": "ta-fixed-historical-public-capture.v1",
        "sources": sources,
        "created_ms": time.time_ns() // 1_000_000,
        "symbols": symbols,
        "original_universe": protocol["original_universe"],
        "months": MONTHS,
        "start_ms": START_MS,
        "end_exclusive_ms": END_MS,
        "archive_tasks": tasks,
        "rest_symbols": symbols,
        "listed_zip_tasks": sum(t["listed"] for t in tasks),
        "listed_zip_bytes": total_bytes,
        "disk_free_bytes_at_prepare": free,
        "max_archive_workers": 4,
        "rest_workers": 1,
        "rest_minimum_request_interval_seconds": 1.5,
        "executed_source": executed,
        "context_source_only": context,
        "runtime": {"python": sys.version, "executable_sha256": sha(sys.executable)},
        "historical_membership_complete": False,
        "first_publication_proven": False,
        "orders_enabled": False,
        "leader_labels_read": False,
        "returns_evaluated": False,
    }
    save(output / "capture_plan.json", plan)
    save(
        output / "freeze_receipts.json",
        {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()},
    )
    return {
        "symbols": len(symbols),
        "archive_tasks": len(tasks),
        "listed_zip_tasks": plan["listed_zip_tasks"],
        "not_listed": sum(not t["listed"] for t in tasks),
        "rest_jobs": len(symbols),
        "listed_zip_bytes": total_bytes,
        "freeze_sha256": sha(output / "freeze_receipts.json"),
    }


def verify_runtime(output):
    """只核验实际collector/helper/计划与固定输入；无关当前包改动不影响长采集。"""
    output = Path(output).resolve()
    for name, expected in read(output / "freeze_receipts.json").items():
        path = output / name
        if path.is_symlink() or not path.resolve().is_relative_to(output) or sha(path) != expected:
            raise ValueError("固定输入或封存源码改变")
    plan = read(output / "capture_plan.json")
    actual = {
        "historical_capture.py": Path(__file__).resolve(),
        "historical_rest.py": Path(capture_pages.__code__.co_filename).resolve(),
    }
    for name, path in actual.items():
        if sha(path) != plan["executed_source"][name]:
            raise ValueError("实际执行collector或helper与冻结源码不符")
    if (
        sys.version != plan["runtime"]["python"]
        or sha(sys.executable) != plan["runtime"]["executable_sha256"]
    ):
        raise ValueError("实际Python运行时改变")
    return plan


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """不把公开请求转发给未登记主机。"""

    def redirect_request(self, request, response, code, message, headers, newurl):
        raise urllib.error.HTTPError(request.full_url, code, "不跟随公开源重定向", headers, None)


class Client:
    """只发固定无认证GET，原文和失败逐次保留；REST单线程至少间隔一点五秒。"""

    def __init__(
        self,
        output,
        plan,
        *,
        rest=False,
        transport=None,
        monotonic=time.monotonic,
        sleep=time.sleep,
    ):
        self.output, self.plan, self.rest = Path(output), plan, rest
        self.transport, self.monotonic, self.sleep = transport, monotonic, sleep
        self.lock, self.stopped, self.next_request = threading.Lock(), threading.Event(), 0.0
        self.allowed = {
            archive_url(t["key"] + suffix)
            for t in plan["archive_tasks"]
            if t["listed"] and t["checksum_key_listed"]
            for suffix in ("", ".CHECKSUM")
        }

    def validate_url(self, url):
        """查询严格只含symbol/start/end/limit，不读取密钥或接受任意主机。"""
        if not self.rest:
            if url not in self.allowed:
                raise ValueError("下载不是已冻结已列出的官方键")
            return
        parsed = urllib.parse.urlsplit(url)
        query = urllib.parse.parse_qs(parsed.query, strict_parsing=True)
        if (
            parsed.scheme + "://" + parsed.netloc + parsed.path != FUNDING
            or set(query) != {"symbol", "startTime", "endTime", "limit"}
            or any(len(v) != 1 for v in query.values())
        ):
            raise ValueError("REST地址或公开查询字段越界")
        if query["symbol"][0] not in self.plan["rest_symbols"] or query["limit"] != ["1000"]:
            raise ValueError("REST符号或分页上限越界")
        first, last = int(query["startTime"][0]), int(query["endTime"][0])
        if not self.plan["start_ms"] <= first <= last < self.plan["end_exclusive_ms"]:
            raise ValueError("REST查询超出固定右开日期")

    def fetch(self, url, folder, maximum):
        """每次尝试原文独立写入；限流停止当前进程并记录服务端退避。"""
        self.validate_url(url)
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        for retry in range(3):
            with self.lock:
                if self.stopped.is_set():
                    raise RuntimeError("CAPTURE_STOPPED_RATE_LIMIT_OR_DISK")
                backoffs = self.output / "backoffs" / ("rest" if self.rest else "archive")
                until = max((read(p)["not_before_ms"] for p in backoffs.glob("*.json")), default=0)
                if time.time_ns() // 1_000_000 < until:
                    self.stopped.set()
                    raise RuntimeError(f"RATE_LIMIT_BACKOFF_UNTIL:{until}")
                wait = max(0.0, self.next_request - self.monotonic())
                if wait:
                    self.sleep(wait)
                self.next_request = self.monotonic() + (1.5 if self.rest else 0.0)
            if shutil.disk_usage(self.output).free < 10 * GIB:
                self.stopped.set()
                raise RuntimeError("DISK_RESERVE_LT_10_GIB")
            number = (
                max(
                    (
                        int(match.group(1))
                        for p in folder.iterdir()
                        if (
                            match := re.fullmatch(
                                r"\.?([0-9]+)\.(?:body|json|tmp|(?:body|json)\..+\.tmp)", p.name
                            )
                        )
                    ),
                    default=-1,
                )
                + 1
            )
            stem = folder / f"{number:06}"
            started, status, payload, headers, error = (
                time.time_ns() // 1_000_000,
                None,
                None,
                {},
                None,
            )
            try:
                if self.transport:
                    status, headers, payload = self.transport(url, maximum)
                else:
                    request = urllib.request.Request(
                        url,
                        method="GET",
                        headers={"User-Agent": "TradingAgents-Public-Historical-Capture"},
                    )
                    with urllib.request.build_opener(NoRedirect()).open(
                        request, timeout=45
                    ) as response:
                        status, headers = response.status, dict(response.headers)
                        payload = response.read(maximum + 1)
                if payload is not None and len(payload) > maximum:
                    raise ValueError("公开响应超过固定大小上限")
            except urllib.error.HTTPError as exception:
                status, headers = exception.code, dict(exception.headers or {})
                payload, error = exception.read(min(maximum, 65536)), type(exception).__name__
            except (urllib.error.URLError, TimeoutError, ValueError) as exception:
                error = type(exception).__name__ + ":" + str(exception)[:300]
            body = stem.with_suffix(".body")
            if payload is not None:
                write_exclusive(body, payload)
            retry_after = str({k.lower(): v for k, v in headers.items()}.get("retry-after", ""))
            receipt = {
                "url": url,
                "method": "GET",
                "credentials_used": False,
                "requested_ms": started,
                "received_ms": time.time_ns() // 1_000_000,
                "status": status,
                "error": error,
                "bytes": len(payload) if payload is not None else None,
                "body": str(body.relative_to(self.output)) if payload is not None else None,
                "sha256": sha(body) if payload is not None else None,
                "retry_after": retry_after,
            }
            save(stem.with_suffix(".json"), receipt)
            if status in {418, 429}:
                seconds = 120 if status == 418 else 60
                if retry_after.isdigit():
                    seconds = max(seconds, int(retry_after))
                elif retry_after:
                    with suppress(ValueError, TypeError, OverflowError):
                        seconds = max(
                            seconds,
                            math.ceil(parsedate_to_datetime(retry_after).timestamp() - time.time()),
                        )
                save(
                    self.output
                    / "backoffs"
                    / ("rest" if self.rest else "archive")
                    / f"{time.time_ns()}.json",
                    {
                        "status": status,
                        "retry_after": retry_after,
                        "not_before_ms": receipt["received_ms"] + seconds * 1000,
                        "request_receipt_sha256": sha(stem.with_suffix(".json")),
                    },
                )
                self.stopped.set()
                raise RuntimeError(
                    f"HTTP_{status}_BACKOFF_REQUIRED:{retry_after or ('120' if status == 418 else '60')}"
                )
            if status == 200 and error is None and payload:
                return body, receipt
            if status == 404:
                return None, receipt
            if status is not None and status < 500:
                raise RuntimeError(f"PUBLIC_HTTP_{status}_NOT_DATA")
            if retry < 2:
                self.sleep(2**retry)
        raise RuntimeError("PUBLIC_TRANSPORT_FAILED_UNKNOWN")


def verified_checksum(archive, checksum, task):
    """官方摘要严格按UTF-8解码，原始ZIP摘要及中文文件名须逐字相符。"""
    tokens = Path(checksum).read_text(encoding="utf-8", errors="strict").split()
    if (
        len(tokens) != 2
        or not re.fullmatch(r"[0-9a-f]{64}", tokens[0])
        or tokens[1].lstrip("*") != Path(task["key"]).name
        or tokens[0] != sha(archive)
    ):
        raise ValueError("OFFICIAL_CHECKSUM_MISMATCH")


def profile_archive(path, task, normalized=None):
    """逐行校验原始字段及全部月度缺口，正量或归档存在不等于可交易。"""
    first, last = month_bounds(task["month"])
    seen, missing, positive, previous, count = set(), [], 0, None, 0
    funding = []
    output = None
    with zipfile.ZipFile(path) as archive, ExitStack() as stack:
        members = archive.infolist()
        if (
            len(members) != 1
            or members[0].filename != Path(task["key"]).name[:-4] + ".csv"
            or members[0].file_size > 128 * 1024**2
        ):
            raise ValueError("ZIP_MEMBER_OR_SIZE_INVALID")
        with archive.open(members[0]) as raw:
            rows = csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
            header = next(rows, None)
            if task["kind"] == "1m" and tuple(header or ()) != COLUMNS:
                raise ValueError("CANDLE_CSV_12_FIELDS_REQUIRED")
            if task["kind"] == "fundingRate" and header != [
                "calc_time",
                "funding_interval_hours",
                "last_funding_rate",
            ]:
                raise ValueError("FUNDING_ARCHIVE_FIELDS_UNKNOWN")
            if normalized is not None:
                Path(normalized).parent.mkdir(parents=True, exist_ok=True)
                output = stack.enter_context(
                    gzip.open(normalized, "xt", encoding="utf-8", newline="", compresslevel=3)
                )
            writer = csv.writer(output) if output else None
            if writer:
                writer.writerow(
                    [
                        *(COLUMNS if task["kind"] == "1m" else header),
                        *(
                            ["completed_boundary_ms", "symbol"]
                            if task["kind"] == "1m"
                            else ["symbol"]
                        ),
                    ]
                )
            for row in rows:
                if len(row) != len(header) or not row[0].isdigit():
                    raise ValueError("CSV_FIELD_OR_INTEGER_TIME_INVALID")
                stamp = int(row[0])
                if (
                    not first <= stamp < last
                    or stamp in seen
                    or (previous is not None and stamp <= previous)
                ):
                    raise ValueError("CSV_TIME_DUPLICATE_OR_OUTSIDE_MONTH")
                seen.add(stamp)
                values = [Decimal(v) for v in row]
                if not all(v.is_finite() for v in values):
                    raise ValueError("CSV_NONFINITE_FIELD")
                if task["kind"] == "1m":
                    if (
                        stamp % MINUTE
                        or int(row[6]) != stamp + MINUTE - 1
                        or not row[6].isdigit()
                        or not row[8].isdigit()
                    ):
                        raise ValueError("CANDLE_CLOCK_OR_COUNT_INVALID")
                    opening, high, low, close = values[1:5]
                    if (
                        min(opening, high, low, close) <= 0
                        or not low <= min(opening, close) <= max(opening, close) <= high
                    ):
                        raise ValueError("CANDLE_PRICE_INVALID")
                    if (
                        any(values[i] < 0 for i in (5, 7, 8, 9, 10))
                        or values[9] > values[5] + Decimal("1e-8") * max(Decimal(1), values[5])
                        or values[10] > values[7] + Decimal("1e-8") * max(Decimal(1), values[7])
                    ):
                        raise ValueError("CANDLE_VOLUME_INVALID")
                    start_gap = first if previous is None else previous + MINUTE
                    if stamp > start_gap:
                        missing.append([start_gap, stamp])
                    positive += values[5] > 0
                    if writer:
                        writer.writerow([*row, stamp + MINUTE, task["symbol"]])
                else:
                    if values[1] <= 0:
                        raise ValueError("FUNDING_INTERVAL_INVALID")
                    funding.append(dict(zip(header, row, strict=True)))
                    if writer:
                        writer.writerow([*row, task["symbol"]])
                previous, count = stamp, count + 1
    if task["kind"] == "1m":
        end_gap = previous + MINUTE if previous is not None else first
        if end_gap < last:
            missing.append([end_gap, last])
    profile = {
        "rows": count,
        "columns": header,
        "minimum_ms": min(seen) if seen else None,
        "maximum_ms": max(seen) if seen else None,
        "missing_minute_intervals_right_open": missing,
        "missing_minutes": sum((b - a) // MINUTE for a, b in missing)
        if task["kind"] == "1m"
        else None,
        "positive_volume_rows": positive if task["kind"] == "1m" else None,
        "all_12_candle_fields_retained": task["kind"] == "1m",
        "raw_time_ms_preserved": True,
        "funding_mark_and_type_available": False if task["kind"] == "fundingRate" else None,
        "historical_tradability_proven": False,
        "first_publication_proven": False,
    }
    return profile, funding


def latest_record(folder):
    """操作状态只取最后一次不可变记录，旧失败不删除。"""
    records = sorted(Path(folder).glob("attempt_*.json"))
    return read(records[-1]) if records else None


def verify_files(output, records):
    """恢复只复用已验证的原文字节，损坏不能静默复用。"""
    for name, expected in records.items():
        path = output / name
        if path.is_symlink() or not path.resolve().is_relative_to(output) or sha(path) != expected:
            raise ValueError("已捕获依赖字节变化")


def capture_archive(output, task, client):
    """单依赖保留缺失、每次失败与全部原文，不下载未列键。"""
    folder = output / "archive_jobs" / task["id"]
    folder.mkdir(parents=True, exist_ok=True)
    old = latest_record(folder)
    if old and old["status"] in {"VERIFIED", "NOT_LISTED"}:
        if (
            old.get("plan_sha256") != sha(output / "capture_plan.json")
            or old["task_id"] != task["id"]
        ):
            raise ValueError("归档恢复记录不属于固定任务")
        verify_files(output, old["files"])
        return old
    number = len(list(folder.glob("attempt_*.json")))
    record = {
        "task_id": task["id"],
        "symbol": task["symbol"],
        "month": task["month"],
        "plan_sha256": sha(output / "capture_plan.json"),
        "kind": task["kind"],
        "status": "UNKNOWN",
        "files": {},
        "profile": None,
    }
    try:
        if not task["listed"]:
            record["status"] = "NOT_LISTED"
        elif not task["checksum_key_listed"]:
            record["status"] = "CHECKSUM_NOT_LISTED_UNKNOWN"
        else:
            raw, receipt = client.fetch(
                archive_url(task["key"]), folder / "zip_requests", task["listed_bytes"]
            )
            if raw is None:
                record["status"] = "LISTED_PUBLIC_404_UNKNOWN"
            else:
                if raw.stat().st_size != task["listed_bytes"]:
                    raise ValueError("ZIP_SIZE_CHANGED_SINCE_FROZEN_LISTING")
                check, check_receipt = client.fetch(
                    archive_url(task["key"] + ".CHECKSUM"), folder / "checksum_requests", 4096
                )
                if check is None:
                    raise ValueError("LISTED_CHECKSUM_404_UNKNOWN")
                verified_checksum(raw, check, task)
                normalized = folder / f"normalized_{number:04}.csv.gz"
                profile, funding = profile_archive(raw, task, normalized)
                files = {
                    str(p.relative_to(output)): sha(p)
                    for p in (
                        raw,
                        raw.with_suffix(".json"),
                        check,
                        check.with_suffix(".json"),
                        normalized,
                    )
                }
                if task["kind"] == "fundingRate":
                    data = folder / f"funding_rows_{number:04}.json"
                    save(data, funding)
                    files[str(data.relative_to(output))] = sha(data)
                    record["funding_rows"] = str(data.relative_to(output))
                record.update(
                    status="VERIFIED",
                    profile=profile,
                    files=files,
                    zip_receipt=receipt,
                    checksum_receipt=check_receipt,
                )
    except Exception as exception:
        record.update(
            status="FAILED_UNKNOWN", error=type(exception).__name__ + ":" + str(exception)[:500]
        )
    save(folder / f"attempt_{number:04}.json", record)
    return record


def capture_rest(output, symbol, plan, client):
    """只负责原文请求及不可变检查点，分页及同钟去重由已绑定纯helper处理。"""
    folder = output / "rest_jobs" / symbol
    folder.mkdir(parents=True, exist_ok=True)
    old = latest_record(folder)
    if old and old["status"] in {"API_RANGE_VERIFIED", "API_EMPTY_RANGE"}:
        if old.get("plan_sha256") != sha(output / "capture_plan.json") or old["symbol"] != symbol:
            raise ValueError("REST恢复记录不属于固定任务")
        verify_files(output, old["files"])
        return old
    number = len(list(folder.glob("attempt_*.json")))
    files, checkpoint = {}, None
    checkpoints = sorted(
        p for p in folder.glob("checkpoint_*.json") if p.with_suffix(".sha256").is_file()
    )
    uncommitted = {
        str(p.relative_to(output)): sha(p)
        for p in folder.glob("checkpoint_*.json")
        if not p.with_suffix(".sha256").is_file()
    }
    if checkpoints:
        if checkpoints[-1].with_suffix(".sha256").read_text().strip() != sha(checkpoints[-1]):
            raise ValueError("REST恢复检查点原始字节改变")
        stored = read(checkpoints[-1])
        if stored["plan_sha256"] != sha(output / "capture_plan.json"):
            raise ValueError("REST恢复检查点不属于固定计划")
        verify_files(output, stored["files"])
        files, checkpoint = stored["files"], stored["checkpoint"]
    record = {
        "symbol": symbol,
        "status": "UNKNOWN",
        "files": {},
        "events": None,
        "plan_sha256": sha(output / "capture_plan.json"),
        "uncommitted_checkpoint_files": uncommitted,
    }
    try:

        def fetch(first, last):
            query = urllib.parse.urlencode(
                {"symbol": symbol, "startTime": first, "endTime": last, "limit": 1000}
            )
            raw, receipt = client.fetch(FUNDING + "?" + query, folder / "requests", 2 * 1024**2)
            if raw is None:
                raise ValueError("REST_404_IS_NOT_EMPTY_EVENTS")
            files[str(raw.relative_to(output))] = sha(raw)
            receipt_path = raw.with_suffix(".json")
            files[str(receipt_path.relative_to(output))] = sha(receipt_path)
            return read(raw)

        def checkpointed(value):
            index = len(list(folder.glob("checkpoint_*.json")))
            path = folder / f"checkpoint_{index:04}.json"
            save(
                path,
                {
                    "plan_sha256": sha(output / "capture_plan.json"),
                    "checkpoint": value,
                    "files": dict(files),
                },
            )
            write_exclusive(path.with_suffix(".sha256"), (sha(path) + "\n").encode("ascii"))

        result = capture_pages(
            symbol,
            plan["start_ms"],
            plan["end_exclusive_ms"],
            fetch,
            checkpoint=checkpoint,
            on_checkpoint=checkpointed,
        )
        data = folder / f"events_{number:04}.json"
        save(data, result["events"])
        files[str(data.relative_to(output))] = sha(data)
        record.update(
            status=result["status"],
            files=files,
            events=str(data.relative_to(output)),
            rows=len(result["events"]),
            pager={k: v for k, v in result.items() if k != "events"},
            server_history_completeness_independently_proven=False,
            historical_first_publication_proven=False,
        )
    except Exception as exception:
        record.update(
            status="FAILED_UNKNOWN",
            files=files,
            error=type(exception).__name__ + ":" + str(exception)[:500],
        )
    save(folder / f"attempt_{number:04}.json", record)
    return record


def compare_all(output, plan):
    """每个909乘两月都输出对账，任一来源未知不能补普通资金费零值。"""
    rows = []
    for symbol in plan["symbols"]:
        rest = latest_record(output / "rest_jobs" / symbol)
        for month in MONTHS:
            archive = latest_record(output / "archive_jobs" / f"fundingRate_{symbol}_{month}")
            if (
                archive
                and archive["status"] == "VERIFIED"
                and rest
                and rest["status"] in {"API_RANGE_VERIFIED", "API_EMPTY_RANGE"}
            ):
                verify_files(output, archive["files"])
                verify_files(output, rest["files"])
                comparison = compare_month(
                    read(output / archive["funding_rows"]),
                    read(output / rest["events"]),
                    *month_bounds(month),
                )
            else:
                comparison = {
                    "status": "UNPROVEN_MISSING_SOURCE",
                    "regular_correspondence_proven": False,
                }
            rows.append(
                {
                    "symbol": symbol,
                    "month": month,
                    "archive_status": archive["status"] if archive else "PENDING",
                    "rest_status": rest["status"] if rest else "PENDING",
                    **comparison,
                }
            )
    return rows


def capture(output, *, archives=False, rest=False, workers=4):
    """显式启动固定公开下载；单输出排他运行，任何收益研究均不在此入口。"""
    output = Path(output).resolve()
    if not (archives or rest) or not 1 <= workers <= 4:
        raise ValueError("须显式选择归档／REST且归档并发一至四")
    if (output / "receipts.json").exists():
        raise ValueError("采集已经最终封存，不能原位增加请求")
    lock = output / ".capture.lock"
    if lock.exists():
        previous = read(lock)
        try:
            os.kill(previous["pid"], 0)
        except ProcessLookupError:
            archive = output / "stale_locks" / f"{time.time_ns()}.json"
            archive.parent.mkdir(parents=True, exist_ok=True)
            lock.rename(archive)
        else:
            raise RuntimeError("同一输出仍有实际存活进程，不能重启采集")
    save(lock, {"pid": os.getpid(), "started_ms": time.time_ns() // 1_000_000})
    try:
        plan = verify_runtime(output)
        if archives:
            client = Client(output, plan)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(capture_archive, output, task, client)
                    for task in plan["archive_tasks"]
                ]
                for index, future in enumerate(as_completed(futures), 1):
                    future.result()
                    if index % 100 == 0:
                        print({"archive_jobs_finished": index, "total": len(futures)}, flush=True)
        if rest and (not archives or not client.stopped.is_set()):
            client = Client(output, plan, rest=True)
            for index, symbol in enumerate(plan["rest_symbols"], 1):
                capture_rest(output, symbol, plan, client)
                if client.stopped.is_set():
                    break
                if index % 25 == 0:
                    print(
                        {"rest_jobs_finished": index, "total": len(plan["rest_symbols"])},
                        flush=True,
                    )
        verify_runtime(output)
        runs = output / "runs"
        runs.mkdir(exist_ok=True)
        number = len(list(runs.glob("*.json")))
        result = status(output, plan)
        save(
            runs / f"{number:04}.json",
            {
                **result,
                "completed_ms": time.time_ns() // 1_000_000,
                "archives_selected": archives,
                "rest_selected": rest,
                "archive_workers": workers,
            },
        )
        return result
    finally:
        lock.unlink(missing_ok=True)


def status(output, plan=None):
    """状态完整列出各类依赖，不将未知或未列出视为已知零。"""
    output = Path(output).resolve()
    plan = plan or verify_runtime(output)
    archive_counts, rest_counts = Counter(), Counter()
    for task in plan["archive_tasks"]:
        item = latest_record(output / "archive_jobs" / task["id"])
        archive_counts[item["status"] if item else task["initial_status"]] += 1
    for symbol in plan["rest_symbols"]:
        item = latest_record(output / "rest_jobs" / symbol)
        rest_counts[item["status"] if item else "PENDING"] += 1
    return {
        "schema": "ta-historical-capture-status.v1",
        "symbols": len(plan["symbols"]),
        "archive_dependencies": dict(archive_counts),
        "rest_dependencies": dict(rest_counts),
        "orders_enabled": False,
        "returns_evaluated": False,
        "historical_membership_or_first_publication_proven": False,
    }


def seal(output):
    """只封存已完成的固定请求；缺档和空API仍是显式质量限制。"""
    output = Path(output).resolve()
    if (output / ".capture.lock").exists() or (output / "receipts.json").exists():
        raise ValueError("采集仍运行或已经封存")
    plan = verify_runtime(output)
    state = status(output, plan)
    if any(
        k in {"PENDING", "UNKNOWN", "FAILED_UNKNOWN"}
        for counts in (state["archive_dependencies"], state["rest_dependencies"])
        for k in counts
    ):
        raise ValueError("存在未完成或失败任务，应先恢复，不封存完成证书")
    plan_sha256 = sha(output / "capture_plan.json")
    for task in plan["archive_tasks"]:
        item = latest_record(output / "archive_jobs" / task["id"])
        if (
            item is None
            or any(item.get(k) != task[k] for k in ("symbol", "kind", "month"))
            or item.get("task_id") != task["id"]
            or item.get("plan_sha256") != plan_sha256
        ):
            raise ValueError("归档最终身份不属于固定任务")
        if (
            item["status"]
            not in {
                "VERIFIED",
                "NOT_LISTED",
                "CHECKSUM_NOT_LISTED_UNKNOWN",
                "LISTED_PUBLIC_404_UNKNOWN",
            }
            or (item["status"] == "NOT_LISTED") == task["listed"]
        ):
            raise ValueError("归档最终状态与冻结清单不符")
        verify_files(output, item["files"])
    for symbol in plan["rest_symbols"]:
        item = latest_record(output / "rest_jobs" / symbol)
        if (
            item is None
            or item.get("symbol") != symbol
            or item.get("plan_sha256") != plan_sha256
            or item["status"] not in {"API_RANGE_VERIFIED", "API_EMPTY_RANGE"}
        ):
            raise ValueError("REST最终身份或状态不属于固定任务")
        verify_files(output, item["files"])
    comparisons = compare_all(output, plan)
    save(output / "funding_month_comparisons.json", comparisons)
    save(
        output / "result.json",
        {
            **state,
            "capture_completed": True,
            "all_symbols_cash_backtest_ready": False,
            "regular_months_matched": sum(
                r.get("regular_correspondence_proven") is True for r in comparisons
            ),
            "complete_cash_all_symbols_proven": False,
        },
    )
    save(
        output / "receipts.json",
        {
            str(p.relative_to(output)): sha(p)
            for p in output.rglob("*")
            if p.is_file() and p.name != "receipts.json"
        },
    )
    return state


def main():
    """准备默认离线；只有明确capture子命令才产生公开GET。"""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--probe", required=True)
    prepare_parser.add_argument("--output", required=True)
    prepare_parser.add_argument("--eligibility")
    prepare_parser.add_argument("--context-root")
    worker = sub.add_parser("capture")
    worker.add_argument("--output", required=True)
    worker.add_argument("--archives", action="store_true")
    worker.add_argument("--rest", action="store_true")
    worker.add_argument("--workers", type=int, default=4, choices=range(1, 5))
    for name in ("status", "seal"):
        sub.add_parser(name).add_argument("--output", required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(
            args.probe, args.output, context_root=args.context_root, eligibility=args.eligibility
        )
    elif args.action == "capture":
        result = capture(args.output, archives=args.archives, rest=args.rest, workers=args.workers)
    elif args.action == "seal":
        result = seal(args.output)
    else:
        result = status(args.output)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
