"""限速采集实际到达的公开输入，封存接收时钟，不生成订单或收益结论。"""

from __future__ import annotations

import argparse
import base64
import fcntl
import gzip
import hashlib
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

from tradingagents.research.quant.strategy_audit import read, save, sha

BASE = "https://fapi.binance.com"
MINUTE = 60_000
ROUTES = {
    "clock": "/fapi/v1/time",
    "exchange": "/fapi/v1/exchangeInfo",
    "quotes": "/fapi/v1/ticker/bookTicker",
    "klines": "/fapi/v1/klines",
    "index": "/fapi/v1/indexPriceKlines",
    "global_ratio": "/futures/data/globalLongShortAccountRatio",
    "top_account_ratio": "/futures/data/topLongShortAccountRatio",
    "top_position_ratio": "/futures/data/topLongShortPositionRatio",
    "open_interest": "/futures/data/openInterestHist",
    "funding": "/fapi/v1/fundingRate",
}
METRICS = ("global_ratio", "top_account_ratio", "top_position_ratio", "open_interest")


def canonical(value):
    """封存时使用唯一编码，拒绝非有限值和隐式字符串转换。"""
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


class Pacer:
    """平滑限制研究请求及公开权重，避免集中突发占用本地出口。"""

    def __init__(self):
        self.lock = threading.Lock()
        self.next_request = self.next_weight = 0.0
        self.stop = threading.Event()

    def acquire(self, weight):
        with self.lock:
            now = time.monotonic()
            slot = max(now, self.next_request, self.next_weight if weight else now)
            self.next_request = slot + 1 / 40
            if weight:
                self.next_weight = slot + weight / 18
        return not self.stop.wait(max(0, slot - time.monotonic()))


def request(kind, symbol, params, pacer):
    """仅允许硬编码的公开 GET 路径，不读取或发送任何账户密钥。"""
    if kind not in ROUTES:
        raise ValueError("公开采集路径未登记")
    url = BASE + ROUTES[kind] + ("?" + urlencode(params) if params else "")
    weight = (
        0 if kind in METRICS or kind == "funding" else 2 if params.get("limit", 0) >= 100 else 1
    )
    if kind == "quotes":
        weight = 5
    if not pacer.acquire(weight):
        return {"kind": kind, "symbol": symbol, "status": "CANCELLED_RATE_LIMIT"}
    started = time.time_ns() // 1_000_000
    tick = time.monotonic()
    status, body, headers = 0, b"", {}
    error = None
    try:
        with urlopen(url, timeout=8) as response:
            status, body = response.status, response.read()
            headers = {k.lower(): v for k, v in response.headers.items()}
    except HTTPError as exc:
        status, body, headers = exc.code, exc.read(), {k.lower(): v for k, v in exc.headers.items()}
    except (URLError, TimeoutError, OSError) as exc:
        error = type(exc).__name__
    received = time.time_ns() // 1_000_000
    elapsed = (time.monotonic() - tick) * 1000
    if status in (418, 429):
        pacer.stop.set()
    record = {
        "kind": kind,
        "symbol": symbol,
        "url": url,
        "status": status,
        "requested_ms": started,
        "received_ms": received,
        "elapsed_monotonic_ms": elapsed,
        "local_clock_consistent": received >= started and abs(received - started - elapsed) <= 250,
        "body_base64": base64.b64encode(body).decode(),
        "body_sha256": hashlib.sha256(body).hexdigest(),
        "server_http_date": headers.get("date"),
        "used_weight_1m": headers.get("x-mbx-used-weight-1m"),
        "error": error,
        "credentials_used": False,
    }
    return record


def body(record):
    """读取原文前核验字节身份，损坏或非标准 JSON 不能进入决策。"""
    raw = base64.b64decode(record["body_base64"], validate=True)
    if hashlib.sha256(raw).hexdigest() != record["body_sha256"]:
        raise ValueError("公开接收原文校验失败")

    def invalid(value):
        raise ValueError(f"公开 JSON 存在非有限数值：{value}")

    return json.loads(raw, parse_constant=invalid)


def available_rows(record, *, decision_ms, cutoff_ms, clock_verified=False):
    """只返回决策前已收到且已完整结束的观测，档案晚到不能回填过去。"""
    if cutoff_ms > decision_ms:
        raise ValueError("输入截止时间不能晚于决策时间")
    if (
        not clock_verified
        or record.get("status") != 200
        or not record.get("local_clock_consistent")
        or record["received_ms"] > decision_ms
    ):
        return []
    values = body(record)
    if not isinstance(values, list):
        raise ValueError("公开分钟或持仓输入必须为列表")
    # 请求开始前必须已结束，另留一秒时钟误差缓冲；晚收到的未完成快照不能变成终值。
    mature = min(cutoff_ms, record["requested_ms"] - 1000)
    kind = record["kind"]
    if kind in ("klines", "index"):
        complete = []
        for row in values:
            if (
                len(row) != 12
                or type(row[0]) is not int
                or type(row[6]) is not int
                or row[0] % MINUTE
                or row[6] != row[0] + MINUTE - 1
            ):
                raise ValueError("实际接收分钟的时钟或字段格式不合法")
            if any(not math.isfinite(float(row[k])) or float(row[k]) <= 0 for k in (1, 2, 3, 4)):
                raise ValueError("实际接收分钟价格必须为有限正数")
            if row[6] + 1 <= mature:
                complete.append(row)
        return complete
    if kind in METRICS:
        limit = mature // (5 * MINUTE) * (5 * MINUTE) - 5 * MINUTE
        if any(
            row["symbol"] != record["symbol"] or type(row["timestamp"]) is not int for row in values
        ):
            raise ValueError("实际接收持仓的标的或时钟不一致")
        return [row for row in values if row["timestamp"] <= limit]
    raise ValueError("此输入不能冒充已完成分钟特征")


def clock_valid(record):
    """服务端时钟必须处于本地请求区间附近；失败时不声称因果可用。"""
    if record["status"] != 200 or not record["local_clock_consistent"]:
        return False
    server = body(record).get("serverTime")
    return (
        type(server) is int
        and record["requested_ms"] - 1000 <= server <= record["received_ms"] + 1000
    )


def dependencies(protocol, *, warmup, metrics, cutoff_ms=None):
    """合约全集来自事前固定清单，不能依据下载成败或后续收益缩减。"""
    jobs = [("quotes", None, {})]
    limit = 300 if warmup else 3
    ending = {"endTime": cutoff_ms - 1} if cutoff_ms is not None else {}
    for symbol in protocol["symbols"]:
        jobs.append(
            ("klines", symbol, {"symbol": symbol, "interval": "1m", "limit": limit} | ending)
        )
    for symbol in protocol["reference_symbols"]:
        jobs.append(("index", symbol, {"pair": symbol, "interval": "1m", "limit": limit} | ending))
        if metrics:
            jobs.extend(
                (kind, symbol, {"symbol": symbol, "period": "5m", "limit": 3}) for kind in METRICS
            )
    if metrics:
        jobs.extend((("exchange", None, {}), ("funding", None, {"limit": 1000})))
    return jobs


def initialize(panel, review, output):
    """此协议只冻结采集；完整策略前向验收尚未登记，也没有运行收益账户。"""
    manifest, proposal = read(panel / "manifest.json"), read(review)
    if (
        manifest.get("leader_labels_read") is not False
        or manifest.get("orders_enabled") is not False
    ):
        raise ValueError("采集集合必须来自纯公开行情")
    if (
        not proposal.get("model_inference_performed")
        or proposal.get("production_eligible") is not False
    ):
        raise ValueError("采集依据必须是无生产权限的真实模型研究")
    source = Path(__file__).resolve()
    protocol = {
        "schema": "cz-profit-forward-public-capture.v2",
        "created_ms": time.time_ns() // 1_000_000,
        "panel_manifest_sha256": sha(panel / "manifest.json"),
        "model_review_sha256": sha(review),
        "source_files": {
            str(source): sha(source),
            str(Path(read.__code__.co_filename).resolve()): sha(read.__code__.co_filename),
        },
        "symbols": sorted(manifest["symbols"]),
        "reference_symbols": sorted(
            s for s, v in manifest["contracts"].items() if v["asset_class"] == "TRADIFI_PERPETUAL"
        ),
        "routes": ROUTES,
        "metrics_interval_seconds": 300,
        "minimum_cycle_seconds": 60,
        "maximum_weight_per_second": 18,
        "maximum_requests_per_second": 40,
        "first_request_limit": 300,
        "later_request_limit": 3,
        "workers": 12,
        "rest_response_receipt_is_availability_time": True,
        "missing_or_late_inputs_not_backfilled": True,
        "completed_before_request_with_clock_buffer_ms": 1000,
        "shared_completed_minute_in_rest_queries": True,
        "capture_only": True,
        "forward_strategy_acceptance_registered": False,
        "profitability_evaluated": False,
        "funding_complete_coverage_verified": False,
        "orders_enabled": False,
        "leader_labels_read": False,
        "credentials_used": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    save(output / "protocol.json", protocol)
    archive = output / "source"
    archive.mkdir()
    for path in protocol["source_files"]:
        (archive / Path(path).name).write_bytes(Path(path).read_bytes())
    return protocol


def seal_batch(output, sequence, protocol, records, *, previous, started_ms, include_metrics):
    """每批原文与接收时钟压缩封存，并与上一完整批次相连。"""
    target = output / "batches" / f"{sequence:06d}"
    target.mkdir(parents=True, exist_ok=False)
    payload = target / "responses.json.gz"
    with payload.open("xb") as stream:
        stream.write(gzip.compress(canonical(records), mtime=0))
    receipt = {
        "schema": "cz-profit-forward-public-batch.v1",
        "sequence": sequence,
        "started_ms": started_ms,
        "finished_ms": time.time_ns() // 1_000_000,
        "protocol_sha256": digest(protocol),
        "previous_receipt_sha256": previous,
        "responses_sha256": sha(payload),
        "requests": len(records),
        "successful_requests": sum(r["status"] == 200 for r in records),
        "metrics_included": include_metrics,
        "clock_valid": all(clock_valid(r) for r in records if r["kind"] == "clock")
        and sum(r["kind"] == "clock" for r in records) == 2,
        "rate_limited": any(r["status"] in (418, 429, "CANCELLED_RATE_LIMIT") for r in records),
        "profitability_evaluated": False,
        "orders_enabled": False,
    }
    save(target / "receipt.json", receipt)
    return receipt


def verify(output):
    """恢复前验证已有完整批次；遇到中断批次保留现场，禁止静默跳过。"""
    protocol = read(output / "protocol.json")
    previous, sequences, last_metrics = None, [], 0
    for path in sorted((output / "batches").glob("*")):
        r = read(path / "receipt.json")
        if (
            r["sequence"] != len(sequences)
            or r["previous_receipt_sha256"] != previous
            or r["protocol_sha256"] != digest(protocol)
            or sha(path / "responses.json.gz") != r["responses_sha256"]
        ):
            raise ValueError("公开前向批次身份或顺序不一致")
        if r["metrics_included"]:
            last_metrics = r["finished_ms"]
        previous = digest(r)
        sequences.append(r)
    return protocol, previous, sequences, last_metrics


def capture(output, cycles):
    """同一采集目录只能由一个进程写入，失败后释放系统锁。"""
    with (output / "capture.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return capture_locked(output, cycles)


def capture_locked(output, cycles):
    """每批结束后再计持仓限速间隔，不按请求开始估算滚动窗口。"""
    protocol, previous, sequences, last_metrics = verify(output)
    for path, expected in protocol["source_files"].items():
        if sha(path) != expected:
            raise ValueError("公开前向采集源码已变化，不能继续旧协议")
    pacer = Pacer()
    for _ in range(cycles):
        begin = time.monotonic()
        started = time.time_ns() // 1_000_000
        metrics = started - last_metrics >= 300_000
        records = [request("clock", None, {}, pacer)]
        if not clock_valid(records[0]):
            save(output / f"clock_failure_{started}.json", records[0])
            raise ValueError("采集前服务端时钟核验失败")
        cutoff = (started - 1000) // MINUTE * MINUTE
        jobs = dependencies(protocol, warmup=not sequences, metrics=metrics, cutoff_ms=cutoff)
        with ThreadPoolExecutor(max_workers=protocol["workers"]) as pool:
            records.extend(pool.map(lambda values: request(*values, pacer), jobs))
        records.append(request("clock", None, {}, pacer))
        receipt = seal_batch(
            output,
            len(sequences),
            protocol,
            records,
            previous=previous,
            started_ms=started,
            include_metrics=metrics,
        )
        previous = digest(receipt)
        sequences.append(receipt)
        if metrics:
            last_metrics = receipt["finished_ms"]
        print(
            f"公开前向采集批次 {receipt['sequence']}：{receipt['successful_requests']}/{receipt['requests']}，时钟核验 {receipt['clock_valid']}。",
            flush=True,
        )
        if not receipt["clock_valid"] or receipt["rate_limited"]:
            raise ValueError("采集时钟或限速失败，已封存现场并停止")
        if len(sequences) and _ + 1 < cycles:
            time.sleep(max(0, 60 - (time.monotonic() - begin)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["initialize", "capture", "verify"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--cycles", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.cycles <= 1440:
        raise ValueError("每次必须显式限定一至1440个采集批次")
    if args.action == "initialize":
        if args.panel is None or args.review is None:
            parser.error("初始化需要公开面板和真实模型审查")
        initialize(args.panel.resolve(), args.review.resolve(), args.output.resolve())
    elif args.action == "capture":
        capture(args.output.resolve(), args.cycles)
    else:
        _, _, batches, _ = verify(args.output.resolve())
        print(f"已核验 {len(batches)} 个公开输入批次，不包含策略收益。")


if __name__ == "__main__":
    main()
