"""固定BTC／ETH月度现货原文及微秒质量核验，不计算研究收益。"""

from __future__ import annotations

import argparse
import calendar
import csv
import hashlib
import io
import re
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from tradingagents.research.quant.artifacts import save, sha

ROOT = Path(__file__).resolve().parents[3]
SYMBOLS = ("BTCUSDT", "ETHUSDT")
MONTHS = ("2026-08", "2026-09")
BASE = "https://data.binance.vision/"
MINUTE_US = 60_000_000
HEADERS = {
    (
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
    ),
    (
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "ignore",
    ),
}


def key(symbol, month):
    """白名单任务不接受任意符号、月份或网络目标。"""
    if symbol not in SYMBOLS or month not in MONTHS:
        raise ValueError("任务不属于固定币月人口")
    return f"data/spot/monthly/klines/{symbol}/1m/{symbol}-1m-{month}.zip"


def parse_archive(payload, symbol, month):
    """逐行检查原微秒及字符串，完整月历缺口保留为计数，不生成标签。"""
    filename = Path(key(symbol, month)).name.removesuffix(".zip") + ".csv"
    year, month_number = map(int, month.split("-"))
    start = int(datetime(year, month_number, 1, tzinfo=UTC).timestamp()) * 1_000_000
    expected = calendar.monthrange(year, month_number)[1] * 1440
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if archive.namelist() != [filename] or archive.getinfo(filename).file_size > 100_000_000:
            raise ValueError("ZIP成员必须为预定单一月度CSV")
        raw = archive.read(filename)
    seen, zero_quote, rows, header = set(), 0, 0, False
    first, last = None, None
    for line, row in enumerate(csv.reader(io.StringIO(raw.decode("utf-8-sig"))), 1):
        if line == 1 and tuple(row) in HEADERS:
            header = True
            continue
        if len(row) != 12 or any(not re.fullmatch(r"[0-9]+", row[p]) for p in (0, 6, 8)):
            raise ValueError(f"第{line}行宽度或原整数列不合法")
        opening, closing, trades = (int(row[p]) for p in (0, 6, 8))
        if (
            opening % MINUTE_US
            or closing != opening + MINUTE_US - 1
            or not start <= opening < start + expected * MINUTE_US
        ):
            raise ValueError(f"第{line}行不是本月声明的精确微秒分钟")
        if opening in seen:
            raise ValueError(f"第{line}行重复原开盘钟")
        try:
            values = {p: Decimal(row[p]) for p in (1, 2, 3, 4, 5, 7, 9, 10, 11)}
        except InvalidOperation as error:
            raise ValueError(f"第{line}行存在非法原数值") from error
        if any(not v.is_finite() for v in values.values()):
            raise ValueError(f"第{line}行数值非有限")
        o, high, low, close = (values[p] for p in (1, 2, 3, 4))
        if (
            min(o, high, low, close) <= 0
            or low > min(o, close)
            or high < max(o, close)
            or low > high
        ):
            raise ValueError(f"第{line}行OHLC关系不合法")
        if (
            any(values[p] < 0 for p in (5, 7, 9, 10))
            or values[9] > values[5]
            or values[10] > values[7]
        ):
            raise ValueError(f"第{line}行成交量或taker范围不合法")
        seen.add(opening)
        rows += 1
        zero_quote += values[7] == 0
        first = opening if first is None else min(first, opening)
        last = opening if last is None else max(last, opening)
        assert trades >= 0
    if not rows:
        raise ValueError("月度归档没有可解释原行")
    return {
        "rows": rows,
        "unique_minutes": len(seen),
        "expected_minutes": expected,
        "unknown_minutes": expected - len(seen),
        "zero_quote_rows": zero_quote,
        "raw_open_min_us": first,
        "raw_open_max_us": last,
        "normalized_first_completed_ms": (first + MINUTE_US) // 1000,
        "normalized_last_completed_ms": (last + MINUTE_US) // 1000,
        "declared_time_unit": "microseconds",
        "header_present": header,
        "member_sha256": hashlib.sha256(raw).hexdigest(),
        "original_values_retained_in_zip": True,
    }


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """公共来源不接受跳往未登记网络目标。"""

    def redirect_request(self, request, response, code, message, headers, newurl):
        raise urllib.error.HTTPError(request.full_url, code, "不跟随重定向", headers, None)


def fetch(task_key, output, stem):
    """每次传输单独保存响应体和时间，即使最后失败也不覆盖。"""
    if not any(task_key in (key(s, m), key(s, m) + ".CHECKSUM") for s in SYMBOLS for m in MONTHS):
        raise ValueError("网络目标不属于本次任务")
    last = None
    for attempt in range(1, 4):
        started, status, error, body = time.time_ns() // 1_000_000, None, None, b""
        try:
            request = urllib.request.Request(
                BASE + task_key, headers={"User-Agent": "TradingAgents-source-research"}
            )
            with urllib.request.build_opener(NoRedirect()).open(request, timeout=45) as response:
                status = response.status
                body = response.read(25_000_001)
                if len(body) > 25_000_000:
                    raise ValueError("来源响应超过本次单文件上限")
        except urllib.error.HTTPError as exception:
            status, error = exception.code, type(exception).__name__
            body = exception.read(100_000)
        except (urllib.error.URLError, TimeoutError, ValueError) as exception:
            error = type(exception).__name__
        receipt = {
            "url": BASE + task_key,
            "attempt": attempt,
            "requested_ms": started,
            "received_ms": time.time_ns() // 1_000_000,
            "http_status": status,
            "error": error,
            "bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
        }
        leaf = output / f"{stem}_{attempt}"
        leaf.with_suffix(".body").write_bytes(body)
        save(leaf.with_suffix(".json"), receipt)
        last = receipt
        if status == 200 and error is None:
            return body, receipt
        if status == 404:
            break
        if attempt < 3:
            time.sleep(attempt)
    raise ValueError(
        f"公开来源未成功：{task_key}，最后HTTP={last['http_status']}，错误={last['error']}"
    )


def run(output):
    """排他封存四项来源和质量，不读取市场结果或模型。"""
    output = Path(output).resolve()
    if output.is_relative_to(ROOT / "tradingagents") or (ROOT / "tradingagents").is_relative_to(
        output
    ):
        raise ValueError("来源输出不能覆盖或包含包代码")
    paths = [Path(__file__), Path(__file__).with_name("spot_archives_plan.md")]
    frozen = {str(p): sha(p) for p in paths}
    output.mkdir(parents=True, exist_ok=False)
    save(
        output / "protocol.json",
        {
            "schema": "ta-fixed-major-spot-archives.v1",
            "symbols": SYMBOLS,
            "months": MONTHS,
            "task_count": 4,
            "population_reason": "BIS Crypto carry原研究BTC/ETH，取得新spot数据前登记；不是旧748赢家筛选",
            "declared_spot_timestamp_unit": "microseconds",
            "source_sha256": frozen,
            "basis_generated": False,
            "labels_generated": False,
            "returns_evaluated": False,
            "orders_enabled": False,
            "production_eligible": False,
        },
    )
    raw = output / "raw"
    raw.mkdir()
    for path in paths:
        shutil.copyfile(path, output / path.name)
    records = []
    for symbol in SYMBOLS:
        for month in MONTHS:
            record = {"symbol": symbol, "month": month, "status": "UNKNOWN"}
            try:
                payload, archive_receipt = fetch(key(symbol, month), raw, f"{symbol}_{month}_zip")
                checksum, checksum_receipt = fetch(
                    key(symbol, month) + ".CHECKSUM", raw, f"{symbol}_{month}_checksum"
                )
                match = re.fullmatch(rb"([a-fA-F0-9]{64})\s+\*?([^\r\n]+)\s*", checksum.strip())
                if (
                    match is None
                    or match[2].decode() != Path(key(symbol, month)).name
                    or match[1].decode().lower() != hashlib.sha256(payload).hexdigest()
                ):
                    raise ValueError("官方checksum文件名或原ZIP摘要不匹配")
                record.update(
                    status="VERIFIED_RAW_ARCHIVE",
                    profile=parse_archive(payload, symbol, month),
                    archive_receipt=archive_receipt,
                    checksum_receipt=checksum_receipt,
                )
            except Exception as error:
                record.update(
                    status="UNKNOWN_FAILED", error_type=type(error).__name__, error=str(error)
                )
            records.append(record)
            print({"symbol": symbol, "month": month, "status": record["status"]}, flush=True)
    if any(sha(Path(p)) != expected for p, expected in frozen.items()):
        save(output / "failure.json", {"status": "FAILED_SOURCE_CHANGED", "orders_enabled": False})
        status = "FAILED_SOURCE_CHANGED"
    else:
        status = (
            "COMPLETED_SOURCE_ONLY"
            if all(r["status"] == "VERIFIED_RAW_ARCHIVE" for r in records)
            else "COMPLETED_WITH_UNKNOWN"
        )
    result = {
        "status": status,
        "tasks": records,
        "verified_tasks": sum(r["status"] == "VERIFIED_RAW_ARCHIVE" for r in records),
        "rows": sum(r.get("profile", {}).get("rows", 0) for r in records),
        "basis_generated": False,
        "labels_generated": False,
        "returns_evaluated": False,
        "historical_first_publication_proven": False,
        "orders_enabled": False,
        "production_eligible": False,
    }
    save(output / "result.json", result)
    save(
        output / "receipts.json",
        {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()},
    )
    return result


def main(argv=None):
    """公开来源下载入口，不建立研究或交易账户。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    result = run(parser.parse_args(argv).output)
    print({k: result[k] for k in ("status", "verified_tasks", "rows")})


if __name__ == "__main__":
    main()
