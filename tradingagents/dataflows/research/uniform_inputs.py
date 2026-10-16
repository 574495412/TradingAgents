"""固定全部合约日期的公开指数和持仓归档，不根据结果删依赖或生成信号。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import math
import os
import shutil
import threading
import time
import uuid
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import HTTPRedirectHandler, build_opener

import pandas as pd

from tradingagents.research.quant.artifacts import read, save, sha
from tradingagents.research.quant.paths import source_path

BASE = "https://data.binance.vision/data/futures/um/daily"


class NoRedirects(HTTPRedirectHandler):
    """公开档案不允许跳转到未登记主机或路径。"""

    def redirect_request(self, request, response, code, message, headers, newurl):
        raise HTTPError(request.full_url, code, "公开归档不跟随重定向", headers, None)


def dependency(kind, symbol, day):
    """公开路径由固定类型、合约与日期构成，不能接受任意主机或认证参数。"""
    if kind not in {"metrics", "indexPriceKlines"} or not symbol or "/" in symbol or ".." in symbol:
        raise ValueError("公开归档类型或合约不合法")
    if pd.Timestamp(day).strftime("%Y-%m-%d") != day:
        raise ValueError("公开归档日期不合法")
    stem = f"{symbol}-metrics-{day}.zip" if kind == "metrics" else f"{symbol}-1m-{day}.zip"
    relative = (
        f"{kind}/{quote(symbol, safe='')}/"
        + ("1m/" if kind != "metrics" else "")
        + quote(stem, safe="")
    )
    return {
        "kind": kind,
        "symbol": symbol,
        "day": day,
        "url": BASE + "/" + relative,
        "file": f"{kind}/{symbol}/{stem}",
    }


def resolve_reuse_archive(root, record, kind):
    """旧相对路径锚定到显式缓存根目录，不能依赖冻结包的工作目录。"""
    root = Path(root).resolve()
    item = dependency(kind, record["symbol"], record["day"])
    canonical = root / item["file"]
    stored = Path(record["path"])
    suffix = (root.name, *Path(item["file"]).parts)
    if stored.is_absolute():
        if stored.resolve() != canonical.resolve():
            raise ValueError("旧缓存绝对路径与显式来源根目录不同")
    elif ".." in stored.parts or stored.parts[-len(suffix) :] != suffix:
        raise ValueError("旧缓存相对路径与固定归档身份不同")
    if not canonical.resolve().is_relative_to(root):
        raise ValueError("旧缓存归档越出显式来源根目录")
    return canonical


def profile(payload, item):
    """校验实际字段而不猜发布时刻，缺分钟和边界重复留给完整覆盖审计。"""
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if len(archive.namelist()) != 1 or archive.infolist()[0].file_size > 64 * 1024 * 1024:
            raise ValueError("公开归档不是单文件")
        content = archive.read(archive.namelist()[0]).decode("utf-8-sig")
    rows = list(csv.reader(io.StringIO(content)))
    if not rows:
        raise ValueError("公开归档为空")
    first = int(pd.Timestamp(item["day"], tz="UTC").value // 1_000_000)
    stamps = []
    if item["kind"] == "indexPriceKlines":
        values = rows[1:] if rows[0][0] in {"open_time", "Open time"} else rows
        for row in values:
            if (
                len(row) != 12
                or not row[0].isdigit()
                or int(row[0]) % 60_000
                or int(row[6]) != int(row[0]) + 59999
            ):
                raise ValueError("指数分钟时钟或字段异常")
            opening = int(row[0])
            if not first <= opening < first + 86_400_000 or any(
                not math.isfinite(float(row[k])) or float(row[k]) <= 0 for k in (1, 2, 3, 4)
            ):
                raise ValueError("指数分钟日期或价格异常")
            stamps.append(opening + 60_000)
        columns = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "count",
            "taker_volume",
            "taker_quote",
            "ignore",
        ]
    else:
        columns = rows[0]
        if len(columns) != len(set(columns)) or not {
            "create_time",
            "symbol",
            "sum_open_interest",
        } <= set(columns):
            raise ValueError("持仓归档列身份异常")
        time_index, symbol_index = columns.index("create_time"), columns.index("symbol")
        for row in rows[1:]:
            if len(row) != len(columns) or row[symbol_index] != item["symbol"]:
                raise ValueError("持仓归档合约或字段异常")
            stamp = int(pd.Timestamp(row[time_index], tz="UTC").value // 1_000_000)
            if not first <= stamp <= first + 86_400_000:
                raise ValueError("持仓归档日期越界")
            for index, name in enumerate(columns):
                if (
                    name not in {"create_time", "symbol"}
                    and row[index]
                    and not math.isfinite(float(row[index]))
                ):
                    raise ValueError("持仓归档存在非有限字段")
            if float(row[columns.index("sum_open_interest")]) < 0:
                raise ValueError("持仓量不能为负")
            stamps.append(stamp)
    return {
        "rows": len(stamps),
        "columns": columns,
        "minimum_timestamp_ms": min(stamps) if stamps else None,
        "maximum_timestamp_ms": max(stamps) if stamps else None,
        "duplicate_timestamps": len(stamps) - len(set(stamps)),
        "complete_1440_index_minutes": item["kind"] == "indexPriceKlines"
        and len(stamps) == 1440
        and len(set(stamps)) == 1440,
        "historical_first_publication_time_proven": False,
    }


class Fetcher:
    """所有线程共享限速与停止状态；只重试有限暂时失败，认证永不参与。"""

    def __init__(self, deadline):
        self.deadline, self.next_request = deadline, 0.0
        self.lock, self.stop = threading.Lock(), threading.Event()
        self.opener = build_opener(NoRedirects())

    def fetch(self, url, target):
        if not url.startswith(BASE + "/"):
            raise ValueError("公开采集地址不在固定档案主机与路径内")
        receipt = target.with_suffix(target.suffix + ".receipt.json")
        if receipt.exists():
            saved = read(receipt)
            if saved["url"] != url or saved["status"] not in {200, 404}:
                raise ValueError("公开下载缓存身份不同")
            if saved["status"] == 404:
                if target.exists():
                    raise ValueError("404缓存却存在文件")
                return None
            if sha(target) != saved["sha256"]:
                raise ValueError("公开下载缓存内容不同")
            return target.read_bytes()
        attempts = []
        for attempt in range(3):
            with self.lock:
                slot = max(time.monotonic(), self.next_request)
                if slot >= self.deadline or self.stop.wait(max(0, slot - time.monotonic())):
                    raise TimeoutError("归档采集已停止或到原期限")
                if time.monotonic() >= self.deadline:
                    raise TimeoutError("原采集期限已过，不能再发起请求")
                self.next_request = time.monotonic() + 0.5
            if (
                shutil.disk_usage(
                    target.parent
                    if target.parent.exists()
                    else next(p for p in target.parents if p.exists())
                ).free
                < 2 * 1024**3
            ):
                self.stop.set()
                raise ValueError("归档采集磁盘余量低于二GiB")
            started = time.time_ns() // 1_000_000
            payload, status, error = None, 0, None
            try:
                with self.opener.open(url, timeout=20) as response:
                    status = response.status
                    payload = response.read(32 * 1024 * 1024 + 1)
            except HTTPError as exc:
                status, error = exc.code, type(exc).__name__
            except (URLError, TimeoutError) as exc:
                error = type(exc).__name__
            attempts.append(
                {
                    "requested_ms": started,
                    "received_ms": time.time_ns() // 1_000_000,
                    "status": status,
                    "error": error,
                }
            )
            if payload is not None and (
                status != 200 or not payload or len(payload) > 32 * 1024 * 1024
            ):
                target.parent.mkdir(parents=True, exist_ok=True)
                failed = target.with_suffix(target.suffix + ".invalid-" + uuid.uuid4().hex)
                failed.write_bytes(payload)
                save(
                    failed.with_suffix(failed.suffix + ".json"),
                    {"url": url, "attempts": attempts, "sha256": sha(failed)},
                )
                raise ValueError("公开归档状态、大小或内容异常，原文保留")
            if status in {418, 429}:
                self.stop.set()
            if status in {200, 404}:
                target.parent.mkdir(parents=True, exist_ok=True)
                if status == 200:
                    if target.exists() and target.read_bytes() != payload:
                        raise ValueError("未封存公开文件内容冲突")
                    target.write_bytes(payload)
                save(
                    receipt,
                    {
                        "url": url,
                        "status": status,
                        "sha256": hashlib.sha256(payload).hexdigest() if status == 200 else None,
                        "attempts": attempts,
                        "credentials_used": False,
                    },
                )
                return payload if status == 200 else None
            if status not in {0, 500, 502, 503, 504} or attempt == 2:
                break
            if self.stop.wait(0.5 * (attempt + 1)):
                break
        target.parent.mkdir(parents=True, exist_ok=True)
        save(
            target.with_suffix(target.suffix + ".failed-" + uuid.uuid4().hex + ".json"),
            {"url": url, "attempts": attempts, "credentials_used": False},
        )
        raise ValueError("公开归档请求失败，不能冒充404")


def collect_records(output, dependencies, worker, stop):
    """先建逐项证据目录；主流程失败时立即停止新请求并取消未开始任务。"""
    if len({item["url"] for item in dependencies}) != len(dependencies):
        raise ValueError("完整采集依赖存在重复URL")
    records = []
    directory = Path(output) / "records"
    directory.mkdir(parents=True, exist_ok=False)
    with ThreadPoolExecutor(max_workers=3) as pool:
        tasks = [pool.submit(worker, item) for item in dependencies]
        try:
            for index, task in enumerate(as_completed(tasks), 1):
                record = task.result()
                save(
                    directory / (hashlib.sha256(record["url"].encode()).hexdigest() + ".json"),
                    record,
                )
                records.append(record)
                if index % 100 == 0:
                    print(
                        f"完整固定依赖：{index}/{len(dependencies)}，{dict(Counter(r['status'] for r in records))}。",
                        flush=True,
                    )
        except BaseException:
            stop.set()
            for task in tasks:
                task.cancel()
            raise
    return records


def run(panel, output, metrics_reuse, index_reuse):
    """先保存完整笛卡尔依赖，再采集；每项结果均保留，包括尚未处理和失败。"""
    panel, output = Path(panel).resolve(), Path(output).resolve()
    manifest = read(panel / "manifest.json")
    if (
        manifest["leader_labels_read"] is not False
        or manifest["orders_enabled"] is not False
        or len(manifest["contracts"]) != 748
    ):
        raise ValueError("只能沿用固定纯公开748合约全集")
    if shutil.disk_usage(output.parent).free < 10 * 1024**3:
        raise ValueError("完整归档采集启动前余量须十GiB")
    days = list(pd.date_range("2026-09-28", "2026-10-14", freq="D").strftime("%Y-%m-%d"))
    dependencies = [
        dependency(kind, symbol, day)
        for symbol in sorted(manifest["contracts"])
        for day in days
        for kind in ("metrics", "indexPriceKlines")
    ]
    reuse_manifests = {
        "metrics": Path(metrics_reuse).resolve() / "metrics_manifest.json",
        "indexPriceKlines": Path(index_reuse).resolve() / "manifest.json",
    }
    reuse_identities = {str(path): sha(path) for path in reuse_manifests.values()}
    plan = source_path("profit_factor_uniform_inputs_plan.md")
    identities = {
        str(p): sha(p)
        for p in (Path(__file__).resolve(), plan, Path(save.__code__.co_filename).resolve())
    }
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "schema": "cz-unconditional-public-factor-inputs.v1",
        "created_ms": time.time_ns() // 1_000_000,
        "manifest_sha256": sha(panel / "manifest.json"),
        "reuse_manifest_sha256": reuse_identities,
        "dependencies": dependencies,
        "source_sha256": identities,
        "workers": 3,
        "maximum_request_starts_per_second": 2,
        "maximum_run_seconds": 43200,
        "runtime": {"pandas": importlib.metadata.version("pandas")},
        "returns_evaluated": False,
        "first_publication_times_proven": False,
        "leader_labels_read": False,
        "orders_enabled": False,
        "production_eligible": False,
    }
    save(output / "protocol.json", protocol)
    (output / "frozen_inputs").mkdir()
    for kind, path in reuse_manifests.items():
        shutil.copyfile(path, output / "frozen_inputs" / (kind + "_manifest.json"))
    for filename in identities:
        target = output / "source" / Path(filename).name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(Path(filename).read_bytes())
    reuse = {}
    for kind, root, name in (
        ("metrics", metrics_reuse, "metrics_manifest.json"),
        ("indexPriceKlines", index_reuse, "manifest.json"),
    ):
        for record in read(Path(root) / name)["records"]:
            if record["status"] == "VERIFIED":
                reuse[kind, record["symbol"], record["day"]] = record | {
                    "resolved_archive_path": str(resolve_reuse_archive(root, record, kind))
                }
    fetcher = Fetcher(time.monotonic() + 43200)
    session = {"pid": os.getpid(), "started_ms": time.time_ns() // 1_000_000}
    save(output / "current_process.json", session)

    def one(item):
        target = output / item["file"]
        reused = reuse.get((item["kind"], item["symbol"], item["day"]))
        try:
            if fetcher.stop.is_set() or time.monotonic() >= fetcher.deadline:
                return item | {"status": "NOT_FETCHED_AFTER_STOP"}
            if reused:
                old = Path(reused["resolved_archive_path"])
                receipt, checksum = (
                    old.with_suffix(old.suffix + ".receipt.json"),
                    old.with_suffix(old.suffix + ".CHECKSUM"),
                )
                if (
                    not receipt.exists()
                    or not checksum.exists()
                    or read(receipt)["url"] != item["url"]
                    or read(receipt)["status"] != 200
                    or read(receipt)["sha256"] != reused["sha256"]
                    or sha(old) != reused["sha256"]
                    or checksum.read_text().split()[0] != reused["sha256"]
                ):
                    raise ValueError("旧归档未通过官方字节和来源校验")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(old, target)
                shutil.copyfile(checksum, target.with_suffix(target.suffix + ".CHECKSUM"))
                save(
                    target.with_suffix(target.suffix + ".receipt.json"),
                    read(receipt) | {"reused_verified_archive": True},
                )
            payload = fetcher.fetch(item["url"], target)
            if payload is None:
                return item | {"status": "PUBLIC_404"}
            check = target.with_suffix(target.suffix + ".CHECKSUM")
            checksum = (
                check.read_bytes()
                if check.exists()
                else fetcher.fetch(item["url"] + ".CHECKSUM", check)
            )
            if (
                checksum is None
                or checksum.decode().split()[0] != hashlib.sha256(payload).hexdigest()
            ):
                raise ValueError("公开归档与官方CHECKSUM不同")
            return item | {
                "status": "VERIFIED_BYTES_AND_PROFILE",
                "sha256": sha(target),
                "profile": profile(payload, item),
                "reused": reused is not None,
            }
        except Exception as exc:
            return item | {
                "status": "FAILED_OR_UNKNOWN",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    records = collect_records(output, dependencies, one, fetcher.stop)
    if any(sha(filename) != expected for filename, expected in identities.items()):
        raise ValueError("采集期间源码或依赖计划发生变化")
    if any(sha(filename) != expected for filename, expected in reuse_identities.items()):
        raise ValueError("采集期间公开复用清单发生变化")
    result = {
        "schema": "cz-unconditional-public-factor-inputs-result.v1",
        "planned_dependencies": len(dependencies),
        "recorded_dependencies": len(records),
        "statuses": dict(Counter(r["status"] for r in records)),
        "records": sorted(records, key=lambda r: (r["symbol"], r["day"], r["kind"])),
        "returns_evaluated": False,
        "stopped_by_source_limit": fetcher.stop.is_set(),
        "ended_ms": time.time_ns() // 1_000_000,
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
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics-reuse", type=Path, required=True)
    parser.add_argument("--index-reuse", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.panel, args.output, args.metrics_reuse, args.index_reuse)
    print({k: v for k, v in result.items() if k != "records"})


if __name__ == "__main__":
    main()
