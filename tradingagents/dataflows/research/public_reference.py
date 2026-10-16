"""采集官方参考指数分钟档案，独立于研究结果和实盘账户。"""

from __future__ import annotations

import argparse
import csv
import io
import math
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import pandas as pd

from tradingagents.dataflows.research.public_capture import fetch
from tradingagents.research.quant.strategy_audit import read, save, sha


def parse_prices(data, *, day):
    """指数价格沿用同一合约计价，按完成时刻索引，不使用正在形成的分钟。"""
    first = int(pd.Timestamp(day, tz="UTC").value // 1_000_000)
    values = {}
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        if len(archive.namelist()) != 1:
            raise ValueError("参考指数档案结构异常")
        content = archive.read(archive.namelist()[0]).decode()
    for row in csv.reader(io.StringIO(content)):
        if row and row[0] in {"open_time", "Open time"}:
            continue
        if len(row) != 12 or not row[0].isdigit():
            raise ValueError("参考指数原文格式异常")
        opening, closing = int(row[0]), int(row[6])
        price = float(row[4])
        if (
            not first <= opening < first + 86_400_000
            or opening % 60_000
            or closing != opening + 59_999
            or opening + 60_000 in values
        ):
            raise ValueError("参考指数时钟不合法")
        if not math.isfinite(price) or price <= 0:
            raise ValueError("参考指数价格必须为有限正数")
        values[opening + 60_000] = price
    return values


def capture(public, output):
    """只按此前价格冲击依赖抓取指数，不依据后续盈亏决定获取哪些档案。"""
    public, output = Path(public).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    parent = read(public / "capture_plan.json")
    dependencies = [(symbol, day) for symbol, day in parent["metrics"] if day < "2026-10-14"]
    save(
        output / "protocol.json",
        {
            "schema": "cz-public-reference-data.v1",
            "dependencies": dependencies,
            "parent_capture_plan_sha256": sha(public / "capture_plan.json"),
            "source_sha256": sha(__file__),
            "kind": "indexPriceKlines",
            "holdout_dates_excluded": ["2026-10-14"],
            "returns_evaluated": False,
            "leader_labels_read": False,
            "orders_enabled": False,
        },
    )

    def one(item):
        symbol, day = item
        stem = f"{symbol}-1m-{day}.zip"
        path = output / "indexPriceKlines" / symbol / stem
        url = f"https://data.binance.vision/data/futures/um/daily/indexPriceKlines/{quote(symbol, safe='')}/1m/{quote(stem, safe='')}"
        payload = fetch(url, path, missing_allowed=True)
        if payload is None:
            return {"symbol": symbol, "day": day, "status": "PUBLIC_404", "path": str(path)}
        checksum = fetch(url + ".CHECKSUM", path.with_suffix(".zip.CHECKSUM"))
        if checksum.decode().split()[0] != sha(path):
            raise ValueError("参考指数官方校验和不一致")
        values = parse_prices(payload, day=day)
        return {
            "symbol": symbol,
            "day": day,
            "status": "VERIFIED",
            "path": str(path),
            "sha256": sha(path),
            "observed_minutes": len(values),
        }

    records = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(one, item) for item in dependencies]
        for i, task in enumerate(as_completed(futures), 1):
            records.append(task.result())
            if i % 100 == 0:
                print(f"官方参考指数：{i}/{len(futures)}。", flush=True)
    save(
        output / "manifest.json",
        {
            "records": sorted(records, key=lambda r: (r["symbol"], r["day"])),
            "returns_evaluated": False,
        },
    )


class ReferenceStore:
    """按当前信号收盘查精确指数记录，缺失值不前向补齐。"""

    def __init__(self, root):
        self.paths = {}
        for record in read(Path(root) / "manifest.json")["records"]:
            if record["status"] == "VERIFIED":
                if sha(record["path"]) != record["sha256"]:
                    raise ValueError("参考指数已封存输入发生变化")
                self.paths[record["symbol"], record["day"]] = Path(record["path"])
        self.cache = {}

    def at(self, symbol, clock):
        # 零点收盘属于前一日最后一分钟，而不是下一日尚未完成的第一分钟。
        key = symbol, pd.Timestamp(clock - 60_000, unit="ms", tz="UTC").date().isoformat()
        if key not in self.paths:
            return None
        if key not in self.cache:
            self.cache[key] = parse_prices(self.paths[key].read_bytes(), day=key[1])
        return self.cache[key].get(clock)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    capture(args.public, args.output)


if __name__ == "__main__":
    main()
