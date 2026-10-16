"""准备尚未参与本轮选参的后续整日行情，采集阶段不计算策略收益。"""

from __future__ import annotations

import argparse
import csv
import io
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import pandas as pd

from tradingagents.dataflows.research.public_capture import fetch
from tradingagents.research.quant.strategy_audit import read, save, sha


def capture(panel, output):
    """沿用此前固定的全部合约集合，不用后续收益或是否成功下载缩减品种。"""
    from tradingagents.research.quant.reference import raw_minutes as raw_reader

    panel, output = Path(panel).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = read(panel / "manifest.json")
    if manifest["leader_labels_read"] is not False or manifest["orders_enabled"] is not False:
        raise ValueError("只能沿用纯公开研究集合")
    start, end = pd.Timestamp("2026-10-14", tz="UTC"), pd.Timestamp("2026-10-15", tz="UTC")
    first, last = int(start.value // 1_000_000), int(end.value // 1_000_000)
    protocol = {
        "schema": "cz-profit-holdout-data.v1",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "universe": manifest["contracts"],
        "parent_manifest_sha256": sha(panel / "manifest.json"),
        "collector_source_sha256": sha(__file__),
        "source_reader_sha256": sha(raw_reader.__file__),
        "returns_evaluated": False,
        "leader_labels_read": False,
        "orders_enabled": False,
        "independent_forward_data": False,
    }
    if (output / "protocol.json").exists():
        if read(output / "protocol.json") != protocol:
            raise ValueError("后续数据采集协议发生变化")
    else:
        save(output / "protocol.json", protocol)

    def one(item):
        symbol, contract = item
        stem = f"{symbol}-1m-2026-10-14.zip"
        url = f"https://data.binance.vision/data/futures/um/daily/klines/{quote(symbol, safe='')}/1m/{quote(stem, safe='')}"
        target = output / "archives" / symbol / stem
        active = contract["onboard_ms"] < last and (
            not contract["delivery_ms"] or contract["delivery_ms"] > first
        )
        payload = fetch(url, target, missing_allowed=True) if active else None
        rows, status = [], "INACTIVE_BY_FIXED_CALENDAR" if not active else "PUBLIC_404"
        if payload is not None:
            checksum = fetch(url + ".CHECKSUM", target.with_suffix(".zip.CHECKSUM"))
            if checksum.decode().split()[0] != sha(target):
                raise ValueError("后续行情官方校验和不匹配")
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                if len(archive.namelist()) != 1:
                    raise ValueError("后续行情压缩包结构异常")
                text = archive.read(archive.namelist()[0]).decode()
            for row in csv.reader(io.StringIO(text)):
                if row and row[0] in {"open_time", "Open time"}:
                    continue
                if len(row) != 12 or not row[0].isdigit():
                    raise ValueError("后续原始 K 线格式异常")
                row[0], row[6], row[8] = int(row[0]), int(row[6]), int(row[8])
                rows.append(row)
            checked = {}
            raw_reader._merge_rows(checked, rows, start=first, end=last)
            rows, status = [checked[k] for k in sorted(checked)], "VERIFIED"
        path = output / "raw" / symbol / "2026-10-14.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "schema": raw_reader.RAW_SCHEMA,
            "symbol": symbol,
            "start_ms": first,
            "end_ms": last,
            "rows": rows,
            "rows_sha256": raw_reader._rows_sha(rows),
            "archive_status": status,
            "queried_intervals_ms": [[first, last]] if status == "VERIFIED" else [],
            "archive_url": url,
        }
        if path.exists():
            if read(path) != record:
                raise ValueError("后续原始数据与既有封存冲突")
        else:
            save(path, record)
        return {
            "symbol": symbol,
            "path": str(path),
            "sha256": sha(path),
            "status": status,
            "start_ms": first,
            "end_ms": last,
            "rows": len(rows),
        }

    records = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        tasks = [pool.submit(one, item) for item in manifest["contracts"].items()]
        for i, task in enumerate(as_completed(tasks), 1):
            records.append(task.result())
            if i % 100 == 0:
                print(f"后续整日原始行情：{i}/{len(tasks)}。", flush=True)
    save(
        output / "raw_manifest.json",
        {"records": sorted(records, key=lambda r: r["symbol"]), "returns_evaluated": False},
    )


def build(panel, output):
    """沿用相同原生价格特征公式，只输出行情和身份，不输出收益标签。"""
    from tradingagents.research.quant.reference.native_panel import build_chunk

    panel, output = Path(panel).resolve(), Path(output).resolve()
    parent, protocol = read(panel / "manifest.json"), read(output / "protocol.json")
    if sha(panel / "manifest.json") != protocol["parent_manifest_sha256"]:
        raise ValueError("后续行情的原始集合身份变化")
    records = {s: [] for s in parent["contracts"]}
    for symbol in records:
        matches = [
            Path(p) for p in parent["source_hashes"] if p.endswith(f"/raw/{symbol}/2026-10-13.json")
        ]
        if len(matches) != 1 or sha(matches[0]) != parent["source_hashes"][str(matches[0])]:
            raise ValueError("后续特征预热缺少原始封存行情")
        value = read(matches[0])
        records[symbol].append(
            {
                "path": str(matches[0]),
                "sha256": sha(matches[0]),
                "start_ms": value["start_ms"],
                "end_ms": value["end_ms"],
            }
        )
    for record in read(output / "raw_manifest.json")["records"]:
        if sha(record["path"]) != record["sha256"]:
            raise ValueError("后续行情原文变化")
        records[record["symbol"]].append(record)
    root = output / "panel"
    root.mkdir(exist_ok=False)
    start, end = pd.Timestamp(protocol["start"]), pd.Timestamp(protocol["end"])
    sources, parts = {}, []
    for day, last in ((start, end - pd.Timedelta(minutes=1)), (end, end)):
        frame, used = build_chunk(
            parent["contracts"],
            records,
            first_close=int(day.value // 1_000_000),
            last_close=int(last.value // 1_000_000),
        )
        path = root / f"candidate_{day.strftime('%Y%m%d')}.parquet"
        frame.to_parquet(path)
        sources.update(used)
        parts.append(
            {
                "path": path.name,
                "sha256": sha(path),
                "first_close_ms": int(day.value // 1_000_000),
                "last_close_ms": int(last.value // 1_000_000),
                "rows": len(frame),
            }
        )
        print(f"后续行情特征已生成：{path.name}，{len(frame)} 行。", flush=True)
    save(
        root / "manifest.json",
        {
            "schema": "dual-alpha-full-native-market.v1",
            "contracts": parent["contracts"],
            "symbols": parent["symbols"],
            "daily_files": parts,
            "source_hashes": sources,
            "market_panel_contract": parent["market_panel_contract"],
            "historical_eligibility_proven": False,
            "leader_labels_read": False,
            "orders_enabled": False,
            "returns_evaluated": False,
            "feature_builder_source_sha256": sha(Path(build_chunk.__code__.co_filename)),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["capture", "build"])
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (capture if args.action == "capture" else build)(args.panel, args.output)


if __name__ == "__main__":
    main()
