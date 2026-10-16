"""为盈利因子实验采集可恢复的公开持仓归档和真实资金费事件。"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

from tradingagents.research.quant.strategy_audit import read, save, sha

BASE = "https://data.binance.vision/data/futures/um/daily/metrics"
FUNDING = "https://fapi.binance.com/fapi/v1/fundingRate"


def fetch(url, path, *, missing_allowed=False):
    """只把官方四〇四当作缺失，网络失败保留错误并允许原位恢复。"""
    path = Path(path)
    receipt = path.with_suffix(path.suffix + ".receipt.json")
    if receipt.exists():
        record = read(receipt)
        if record["url"] != url or record["status"] not in (200, 404):
            raise ValueError("公开缓存身份变化")
        if record["status"] == 404:
            if not missing_allowed or path.exists():
                raise ValueError("公开缺失缓存不一致")
            return None
        if sha(path) != record["sha256"]:
            raise ValueError("公开缓存内容变化")
        return path.read_bytes()
    payload = None
    for attempt in range(4):
        try:
            with urlopen(url, timeout=30) as response:
                payload = response.read()
                if response.status != 200 or not payload:
                    raise ValueError("公开响应为空或状态异常")
            break
        except HTTPError as exc:
            if exc.code == 404 and missing_allowed:
                break
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 3:
                raise
            retry = exc.headers.get("Retry-After", "")
            time.sleep(min(30, max(2**attempt, int(retry) if retry.isdigit() else 0)))
        except (URLError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    path.parent.mkdir(parents=True, exist_ok=True)
    if payload is not None:
        if path.exists() and path.read_bytes() != payload:
            raise ValueError("未封存下载与已存在文件冲突")
        path.write_bytes(payload)
    save(
        receipt,
        {
            "url": url,
            "status": 200 if payload is not None else 404,
            "sha256": hashlib.sha256(payload).hexdigest() if payload is not None else None,
            "captured_at": datetime.now(UTC).isoformat(),
        },
    )
    return payload


def prepare(panel, output):
    """只依据可见价格条件决定数据依赖，不用后续收益选择要抓的品种。"""
    panel, output = Path(panel).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = read(panel / "manifest.json")
    if (
        manifest.get("leader_labels_read") is not False
        or manifest.get("orders_enabled") is not False
    ):
        raise ValueError("必须是纯公开行情面板")
    start, end = pd.Timestamp("2026-09-29", tz="UTC"), pd.Timestamp("2026-10-14", tz="UTC")
    needed, sources = set(), {}
    for at in pd.date_range(start, end - pd.Timedelta(days=1), freq="D"):
        name = f"candidate_{at.strftime('%Y%m%d')}.parquet"
        path = panel / name
        record = next(p for p in manifest["daily_files"] if p["path"] == name)
        if sha(path) != record["sha256"]:
            raise ValueError("公开面板与原始清单不一致")
        sources[str(path)] = record["sha256"]
        columns = [
            "asset_class",
            "eligible",
            "observed",
            "ret_20_pct",
            "ret_60_pct",
            "ret_20_rank_pct",
            "ret_60_rank_pct",
            "atr_rank_pct",
        ]
        frame = pd.read_parquet(
            path, columns=columns, filters=[("asset_class", "==", "TRADIFI_PERPETUAL")]
        )
        shock = (
            frame["eligible"]
            & frame["observed"]
            & frame["ret_20_pct"].ge(1.5)
            & frame["ret_60_pct"].ge(2)
            & frame[["ret_20_rank_pct", "ret_60_rank_pct"]].max(axis=1).ge(0.985)
            & frame["atr_rank_pct"].ge(0.93)
        )
        for symbol in frame.loc[shock].index.get_level_values("symbol").unique():
            # 确认窗口可跨日，五分钟滞后的持仓观测也可落在前一日。
            for delta in (-1, 0, 1):
                needed.add((str(symbol), (at + pd.Timedelta(days=delta)).date().isoformat()))
    save(
        output / "capture_plan.json",
        {
            "schema": "cz-profit-public-capture.v1",
            "start": start.isoformat(),
            "end": end.isoformat(),
            "panel": str(panel),
            "panel_source_hashes": sources,
            "manifest_sha256": sha(panel / "manifest.json"),
            "timestamp_semantics": "CANDLE_CLOSE; RAW_OPEN_PLUS_60000",
            "metrics": sorted(needed),
            "funding_symbols": sorted(manifest["symbols"]),
            "orders_enabled": False,
            "leader_labels_read": False,
        },
    )
    print(f"数据依赖已冻结：{len(needed)} 份持仓日档案。", flush=True)


def capture_metrics(output, reuse):
    """核验已有公开归档后复用，其余下载官方原文及校验和。"""
    plan, reuse = read(output / "capture_plan.json"), Path(reuse)

    def one(item):
        symbol, day = item
        if not symbol.isalnum():
            raise ValueError("标的格式不合法")
        stem = f"{symbol}-metrics-{day}.zip"
        target, old = output / "metrics" / symbol / stem, reuse / "metrics" / symbol / stem
        url = f"{BASE}/{symbol}/{stem}"
        old_receipt = old.with_suffix(".zip.receipt.json")
        old_check = old.with_suffix(".zip.CHECKSUM")
        if (
            not target.with_suffix(".zip.receipt.json").exists()
            and old.exists()
            and old_receipt.exists()
            and old_check.exists()
        ):
            receipt = read(old_receipt)
            if (
                receipt["url"] != url
                or receipt["status"] != 200
                or sha(old) != receipt["sha256"]
                or old_check.read_text().split()[0] != sha(old)
            ):
                raise ValueError("待复用公开归档无效")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(old, target)
            shutil.copyfile(old_check, target.with_suffix(".zip.CHECKSUM"))
            save(
                target.with_suffix(".zip.receipt.json"), receipt | {"reused_verified_archive": True}
            )
        data = fetch(url, target, missing_allowed=True)
        if data is None:
            return {"symbol": symbol, "day": day, "status": "PUBLIC_404", "path": str(target)}
        checksum = target.with_suffix(".zip.CHECKSUM")
        if not checksum.exists():
            fetch(url + ".CHECKSUM", checksum)
        if checksum.read_text().split()[0] != hashlib.sha256(data).hexdigest():
            raise ValueError("官方持仓校验和不匹配")
        return {
            "symbol": symbol,
            "day": day,
            "status": "VERIFIED",
            "path": str(target),
            "sha256": sha(target),
        }

    records = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(one, item) for item in plan["metrics"]]
        for i, future in enumerate(as_completed(futures), 1):
            records.append(future.result())
            if i % 100 == 0:
                print(f"持仓归档：{i}/{len(futures)}", flush=True)
    save(
        output / "metrics_manifest.json",
        {
            "records": sorted(records, key=lambda r: (r["symbol"], r["day"])),
            "missing_allowed_only_as_explicit_data_gaps": True,
        },
    )


def capture_funding(output):
    """对全市场按时间自适应细分，避免分页边界丢失同毫秒的其他合约。"""
    plan = read(output / "capture_plan.json")
    first = int(pd.Timestamp(plan["start"]).value // 1_000_000)
    last = int(pd.Timestamp(plan["end"]).value // 1_000_000) - 1
    events, leaves, request_count, next_request = {}, [], 0, 0.0

    def visit(left, right):
        nonlocal request_count, next_request
        path = output / "funding_raw" / f"{left}-{right}.json"
        url = FUNDING + "?" + urlencode({"startTime": left, "endTime": right, "limit": 1000})
        if not path.with_suffix(".json.receipt.json").exists():
            time.sleep(max(0.0, next_request - time.monotonic()))
            next_request = time.monotonic() + 1.05
        rows = json.loads(fetch(url, path))
        request_count += 1
        if not isinstance(rows, list) or len(rows) > 1000:
            raise ValueError("资金费公开响应不合法")
        if len(rows) == 1000:
            if left == right:
                raise ValueError("单毫秒资金费超过接口上限，需要按合约细分")
            middle = (left + right) // 2
            visit(left, middle)
            visit(middle + 1, right)
        else:
            leaves.append(
                {
                    "start_ms": left,
                    "end_ms": right,
                    "path": str(path),
                    "sha256": sha(path),
                    "rows": len(rows),
                }
            )
            for row in rows:
                stamp = int(row["fundingTime"])
                key = row["symbol"], stamp
                if not left <= stamp <= right or float(row["markPrice"]) <= 0:
                    raise ValueError("资金费时钟或标记价不合法")
                if key in events and events[key] != row:
                    raise ValueError("资金费公开记录冲突")
                events[key] = row
        if request_count % 40 == 0:
            print(f"资金费：{request_count} 次公开请求，{len(events)} 个事件。", flush=True)

    visit(first, last)
    ordered = sorted(leaves, key=lambda r: r["start_ms"])
    if (
        ordered[0]["start_ms"] != first
        or ordered[-1]["end_ms"] != last
        or any(a["end_ms"] + 1 != b["start_ms"] for a, b in zip(ordered, ordered[1:], strict=False))
    ):
        raise ValueError("资金费请求没有覆盖整个研究时间轴")
    save(
        output / "funding_events.json",
        sorted(events.values(), key=lambda r: (int(r["fundingTime"]), r["symbol"])),
    )
    save(
        output / "funding_manifest.json",
        {
            "leaves": ordered,
            "requests": request_count,
            "events": len(events),
            "api_time_range_coverage_verified": True,
            "server_historical_completeness_independently_proven": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "metrics", "funding"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel", type=Path)
    parser.add_argument("--reuse", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.panel, args.output)
    elif args.action == "metrics":
        capture_metrics(args.output, args.reuse)
    else:
        capture_funding(args.output)


if __name__ == "__main__":
    main()
