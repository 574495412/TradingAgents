"""连接真实现货、永续原分钟与实际结算事件，不生成对冲收益。"""

from __future__ import annotations

import argparse
import calendar
import csv
import io
import json
import re
import shutil
import time
import zipfile
from collections import Counter
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from tradingagents.research.quant.artifacts import read, save, sha, verify_receipts

from .historical_funding import BoundSource, canonical, record_identity, replay_rest
from .spot_archives import HEADERS, MONTHS, SYMBOLS

ROOT = Path(__file__).resolve().parents[3]
START, END, MINUTE = 1785542400000, 1790812800000, 60000
FIELDS = {
    "open": 1,
    "high": 2,
    "low": 3,
    "close": 4,
    "base_volume": 5,
    "quote_volume": 7,
    "trade_count": 8,
    "taker_buy_base": 9,
    "taker_buy_quote": 10,
    "ignore": 11,
}
ROOTS = {
    "spot": (
        "data/research/spot_majors_20261017_v001",
        "06bbc1fa4515338d98a71434a75f0facce071c7da56722abde7a4e653956c68e",
    ),
    "spot_audit": (
        "results/research/spot_majors_source_audit_20261017_v001",
        "75bf50f5fb51ed9b4c59a120126c13e21d50b2392ae48dd5ea03b38e7f23461a",
    ),
    "futures": (
        "data/research/historical_market_20261017_v002",
        "21624a01c5e1247dc4797fb020ccc4f94ba404fea654f89589784d5d16cfa880",
    ),
    "funding": (
        "results/research/historical_typed_funding_20261017_v001",
        "93172c8f3a419e3af0740d75ab8244d3c9674d10eca9686a1bbfc454ed70aec9",
    ),
}


def parse_market_zip(payload, symbol, month, market):
    """市场决定原单位，绝不按数值量级猜微秒或毫秒。"""
    if symbol not in SYMBOLS or month not in MONTHS or market not in {"spot", "perp"}:
        raise ValueError("原分钟不属于本次来源范围")
    scale = 1000 if market == "spot" else 1
    year, number = map(int, month.split("-"))
    first = int(datetime(year, number, 1, tzinfo=UTC).timestamp()) * 1000
    length = calendar.monthrange(year, number)[1] * 1440
    name = f"{symbol}-1m-{month}.csv"
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if archive.namelist() != [name] or archive.getinfo(name).file_size > 100_000_000:
            raise ValueError("归档成员不属于预定币月")
        raw = archive.read(name)
    records = {}
    for line, row in enumerate(csv.reader(io.StringIO(raw.decode("utf-8-sig"))), 1):
        if line == 1 and tuple(row) in HEADERS:
            continue
        if len(row) != 12 or any(not re.fullmatch(r"[0-9]+", row[p]) for p in (0, 6, 8)):
            raise ValueError("原分钟列宽或整数类型错误")
        opening, closing = int(row[0]), int(row[6])
        if opening % (MINUTE * scale) or closing != opening + MINUTE * scale - 1:
            raise ValueError("原分钟单位或开收盘边界错误")
        stamp = opening // scale
        if not first <= stamp < first + length * MINUTE or stamp in records:
            raise ValueError("原分钟越月或重复")
        try:
            values = {p: Decimal(row[p]) for p in (1, 2, 3, 4, 5, 7, 9, 10, 11)}
        except InvalidOperation as error:
            raise ValueError("原分钟数值非法") from error
        if any(not v.is_finite() for v in values.values()):
            raise ValueError("原分钟数值非有限")
        o, high, low, close = (values[p] for p in (1, 2, 3, 4))
        if not 0 < low <= min(o, close) <= max(o, close) <= high:
            raise ValueError("原分钟OHLC错误")
        if (
            any(values[p] < 0 for p in (5, 7, 9, 10))
            or values[9] > values[5]
            or values[10] > values[7]
        ):
            raise ValueError("原分钟成交量错误")
        records[stamp] = {
            "raw_open": opening,
            "raw_close": closing,
            "line": line,
            **{f: row[p] for f, p in FIELDS.items()},
        }
    if not records:
        raise ValueError("原归档没有可解释分钟")
    return records


def _futures_zip(bound, plan, symbol, month):
    """四归档重核原attempt、独立HTTP收据和官方SHA，不加载收益文件。"""
    task_id = f"1m_{symbol}_{month}"
    tasks = [t for t in plan["archive_tasks"] if t["id"] == task_id]
    if len(tasks) != 1:
        raise ValueError("期货来源任务不唯一")
    task = tasks[0]
    folder = "archive_jobs/" + task_id
    record = bound.latest(folder)
    record_identity(record, sha(bound.root / "capture_plan.json"), symbol, task)
    bound.record_files(record, folder)
    if record["status"] != "VERIFIED":
        raise ValueError("期货归档未验证")
    bodies = []
    for key, suffix in (("zip_receipt", ""), ("checksum_receipt", ".CHECKSUM")):
        receipt = record[key]
        relative = receipt["body"]
        if (
            relative not in record["files"]
            or receipt["sha256"] != record["files"][relative]
            or receipt["status"] != 200
            or receipt["method"] != "GET"
            or receipt["credentials_used"] is not False
            or receipt["url"] != "https://data.binance.vision/" + task["key"] + suffix
            or canonical(bound.load(str(Path(relative).with_suffix(".json")))) != canonical(receipt)
        ):
            raise ValueError("期货原始HTTP身份不一致")
        bodies.append(bound.path(relative))
    checksum = bodies[1].read_text().split()
    if (
        len(checksum) != 2
        or checksum[1].removeprefix("*") != Path(task["key"]).name
        or checksum[0] != sha(bodies[0])
    ):
        raise ValueError("期货官方SHA不一致")
    return bodies[0].read_bytes(), sha(bodies[0])


def funding_rows(typed, replayed, symbols=SYMBOLS):
    """逐事件绑定原REST规范记录；不取整，不计算库存现金。"""
    selected, seen = [], set()
    for row in typed:
        if row["symbol"] not in symbols:
            continue
        symbol, stamp = row["symbol"], row["time_ms"]
        quality = row["quality"]
        index = row["source"]["event_index"]
        if (
            type(stamp) is not int
            or not START <= stamp < END
            or row["type"] not in {"Regular", "Special"}
            or quality["trusted_typed_fields"] is not True
            or quality["identity_conflict"] is not False
            or quality["month_ready_under_source_assumptions"] is not True
            or not 0 <= index < len(replayed[symbol]["events"])
            or row["raw"] != replayed[symbol]["events"][index]
            or row["raw"]["fundingTime"] != stamp
            or row["raw"]["rateType"] != row["type"]
            or row["raw"]["symbol"] != symbol
        ):
            raise ValueError("资金费事件缺少精确原文或可信类型")
        rate, mark = Decimal(row["rate"]), Decimal(row["mark"])
        if (
            not rate.is_finite()
            or not mark.is_finite()
            or mark <= 0
            or rate != Decimal(row["raw"]["fundingRate"])
            or mark != Decimal(row["raw"]["markPrice"])
        ):
            raise ValueError("资金费原值或实际mark错误")
        identity = symbol, stamp, row["type"]
        if identity in seen:
            raise ValueError("资金费事件重复")
        seen.add(identity)
        selected.append(row)
    expected = {
        (symbol, e["fundingTime"], e["rateType"])
        for symbol in symbols
        for e in replayed[symbol]["events"]
    }
    if seen != expected:
        raise ValueError("typed视图没有完整保留两符号原事件")
    return sorted(selected, key=lambda r: (r["symbol"], r["time_ms"], r["type"]))


def _schema():
    fields = [
        pa.field("symbol", pa.string()),
        pa.field("raw_open_ms", pa.int64()),
        pa.field("completed_ms", pa.int64()),
    ]
    for market in ("spot", "perp"):
        fields.extend(
            [
                pa.field(market + "_raw_open", pa.int64()),
                pa.field(market + "_raw_close", pa.int64()),
                pa.field(market + "_source_line", pa.int64()),
                pa.field(market + "_archive_sha256", pa.string()),
                pa.field(market + "_status", pa.string()),
            ]
        )
        fields.extend(pa.field(market + "_" + name, pa.string()) for name in FIELDS)
    return pa.schema(fields)


def run(output):
    """冻结两腿来源，完整保留各原分钟及全部实际结算事件。"""
    roots = {name: ROOT / path for name, (path, _) in ROOTS.items()}
    output = Path(output).resolve()
    package = ROOT / "tradingagents"
    if any(
        output.is_relative_to(p) or p.is_relative_to(output) for p in (*roots.values(), package)
    ):
        raise ValueError("输出不能覆盖或包含来源及包")
    for name, root in roots.items():
        if sha(root / "receipts.json") != ROOTS[name][1]:
            raise ValueError("来源不等于冻结版本")
        if name != "futures":
            verify_receipts(root)
    capture = BoundSource(roots["futures"])
    plan = capture.load("capture_plan.json")
    typed_protocol = read(roots["funding"] / "protocol.json")
    if typed_protocol["source_receipts_sha256"] != capture.receipts_sha256 or typed_protocol[
        "source_root"
    ] != str(capture.root):
        raise ValueError("两腿与结算来源不是同一封存版本")
    coverage = [r for r in read(roots["funding"] / "coverage.json") if r["symbol"] in SYMBOLS]
    if (
        len(coverage) != 4
        or {(r["symbol"], r["month"]) for r in coverage}
        != {(s, m) for s in SYMBOLS for m in MONTHS}
        or any(r["month_ready_under_source_assumptions"] is not True for r in coverage)
    ):
        raise ValueError("两符号四个月份结算来源未就绪")
    files = [
        Path(__file__),
        Path(__file__).with_name("hedged_market_plan.md"),
        Path(__file__).with_name("historical_funding.py"),
        Path(__file__).with_name("historical_rest.py"),
        Path(__file__).with_name("spot_archives.py"),
        ROOT / "tradingagents/research/quant/artifacts.py",
    ]
    frozen = {str(p): sha(p) for p in files}
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    protocol = {
        "schema": "ta-hedged-major-source-panel.v1",
        "symbols": SYMBOLS,
        "start_raw_open_ms": START,
        "end_raw_open_exclusive_ms": END,
        "expected_rows": 175680,
        "source_receipts": {name: ROOTS[name][1] for name in roots},
        "source_roots": {k: str(v) for k, v in roots.items()},
        "source_sha256": frozen,
        "spot_time_unit": "microseconds",
        "perp_time_unit": "milliseconds",
        "funding_times_rounded": False,
        "historical_first_publication_proven": False,
        "historical_unit_and_tradability_proven": False,
        "basis_generated": False,
        "labels_generated": False,
        "returns_evaluated": False,
        "orders_enabled": False,
        "leader_labels_read": False,
        "production_eligible": False,
    }
    save(output / "protocol.json", protocol)
    try:
        source_dir = output / "source"
        source_dir.mkdir()
        for path in files:
            shutil.copyfile(path, source_dir / path.name)
            if sha(source_dir / path.name) != frozen[str(path)]:
                raise ValueError("执行依赖副本变化")
        profiles = []
        with pq.ParquetWriter(output / "minutes.parquet", _schema()) as writer:
            for symbol in SYMBOLS:
                for month in MONTHS:
                    task = next(
                        r
                        for r in read(roots["spot"] / "result.json")["tasks"]
                        if r["symbol"] == symbol and r["month"] == month
                    )
                    if task["status"] != "VERIFIED_RAW_ARCHIVE":
                        raise ValueError("现货归档未就绪")
                    attempt = task["archive_receipt"]["attempt"]
                    spot_zip = roots["spot"] / "raw" / f"{symbol}_{month}_zip_{attempt}.body"
                    spot_sha = sha(spot_zip)
                    payload, fut_sha = _futures_zip(capture, plan, symbol, month)
                    spot, perp = (
                        parse_market_zip(spot_zip.read_bytes(), symbol, month, "spot"),
                        parse_market_zip(payload, symbol, month, "perp"),
                    )
                    year, number = map(int, month.split("-"))
                    start = int(datetime(year, number, 1, tzinfo=UTC).timestamp()) * 1000
                    total = calendar.monthrange(year, number)[1] * 1440
                    rows = []
                    for stamp in range(start, start + total * MINUTE, MINUTE):
                        row = {
                            "symbol": symbol,
                            "raw_open_ms": stamp,
                            "completed_ms": stamp + MINUTE,
                        }
                        for market, records, checksum in (
                            ("spot", spot, spot_sha),
                            ("perp", perp, fut_sha),
                        ):
                            item = records.get(stamp)
                            row[market + "_status"] = (
                                "KNOWN_RAW_MINUTE" if item else "MISSING_MINUTE"
                            )
                            row[market + "_archive_sha256"] = checksum
                            for key in ("raw_open", "raw_close", "line"):
                                row[market + ("_source_line" if key == "line" else "_" + key)] = (
                                    item[key] if item else None
                                )
                            for field in FIELDS:
                                row[market + "_" + field] = item[field] if item else None
                        rows.append(row)
                    writer.write_table(pa.Table.from_pylist(rows, schema=_schema()))
                    profiles.append(
                        {
                            "symbol": symbol,
                            "month": month,
                            "rows": total,
                            "spot_known": len(spot),
                            "perp_known": len(perp),
                            "both_known": len(set(spot) & set(perp)),
                            "spot_sha256": spot_sha,
                            "perp_sha256": fut_sha,
                        }
                    )
                    print({"symbol": symbol, "month": month, "rows": total}, flush=True)
        replayed = {
            symbol: replay_rest(capture, plan, sha(capture.root / "capture_plan.json"), symbol)
            for symbol in SYMBOLS
        }
        if any(not r["source_consistent"] or r["conflicts"] for r in replayed.values()):
            raise ValueError("原REST资金费重放不一致")
        with (roots["funding"] / "typed_cash_events.jsonl").open() as stream:
            selected = funding_rows((json.loads(line) for line in stream), replayed)
        save(output / "funding_events.json", selected)
        save(output / "funding_coverage.json", coverage)
        save(
            output / "coverage.json",
            {
                "tasks": profiles,
                "rows": sum(r["rows"] for r in profiles),
                "both_known": sum(r["both_known"] for r in profiles),
            },
        )
        save(output / "consumed_futures_inputs.json", capture.used)
        capture.verify_used()
        for name, root in roots.items():
            if sha(root / "receipts.json") != ROOTS[name][1]:
                raise ValueError("源收据运行中发生变化")
            if name != "futures":
                verify_receipts(root)
        if any(sha(Path(p)) != checksum for p, checksum in frozen.items()):
            raise ValueError("执行源码或计划发生变化")
        result = {
            "status": "COMPLETED_SOURCE_PANEL_ONLY",
            "symbols": 2,
            "rows": sum(r["rows"] for r in profiles),
            "both_known": sum(r["both_known"] for r in profiles),
            "funding_events": len(selected),
            "funding_types": dict(Counter(r["type"] for r in selected)),
            "off_whole_minute_funding_events": sum(r["time_ms"] % MINUTE != 0 for r in selected),
            "funding_months_ready_under_source_assumptions": 4,
            "elapsed_seconds": time.monotonic() - started,
            "basis_generated": False,
            "labels_generated": False,
            "returns_evaluated": False,
            "historical_first_publication_proven": False,
            "orders_enabled": False,
            "production_eligible": False,
        }
        save(output / "result.json", result)
    except Exception as error:
        save(
            output / "failure.json",
            {"status": "FAILED", "type": type(error).__name__, "message": str(error)},
        )
        save(
            output / "receipts.json",
            {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()},
        )
        raise
    save(
        output / "receipts.json",
        {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()},
    )
    return result


def main(argv=None):
    """离线两腿来源入口，不调用网络、模型或订单。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    result = run(parser.parse_args(argv).output)
    print({k: result[k] for k in ("status", "rows", "funding_events")})


if __name__ == "__main__":
    main()
