"""从 CZ 已封存的 Binance 官方归档导出研究输入，不重新联网抓取。

历史公开时间加延迟属于回顾性可用时间假设，不能冒充当时真实采集记录。
同一快照只使用已完成的 K 线，以及已经发生且超过延迟的资金费事件。
"""

import csv
import hashlib
import io
import math
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .common import digest, file_hash, read

HOUR = 3600000


def utc_ms(value):
    return int(datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC).timestamp() * 1000)


def public_export(root, symbols, start, end, every_hours=24, evidence_path=None):
    root = Path(root).resolve()
    manifest = read(root / "manifest.json")
    if (
        manifest.get("leader_labels_read") is not False
        or manifest.get("orders_enabled") is not False
    ):
        raise ValueError("必须使用已确认的纯公开研究归档")
    if not set(symbols) <= set(manifest["symbols"]) or len(set(symbols)) != len(symbols):
        raise ValueError("请求标的不在公开归档中或存在重复")
    first, last = utc_ms(start), utc_ms(end)
    if first >= last or type(every_hours) is not int or every_hours < 1:
        raise ValueError("时间区间与采样步长不合法")
    raw, funding_all, source_records = {}, [], []
    for symbol in sorted(symbols):
        if not symbol.isalnum():
            raise ValueError("归档标的格式不合法")
        rows = {}
        for source in manifest["sources"]:
            if source["symbol"] != symbol or source.get("status") != 200:
                continue
            month = utc_ms(source["month"] + "-01")
            if month >= last or month + 32 * 24 * HOUR <= first - 24 * HOUR:
                continue
            path = root / "archives" / symbol / "1h" / Path(source["path"]).name
            if (
                not path.resolve().is_relative_to(root)
                or file_hash(path) != source["sha256"]
                or not source["checksum_verified"]
            ):
                raise ValueError("官方行情归档哈希不匹配")
            source_records.append({"url": source["url"], "sha256": source["sha256"]})
            with zipfile.ZipFile(path) as archive:
                if len(archive.namelist()) != 1:
                    raise ValueError("行情压缩包结构异常")
                content = archive.read(archive.namelist()[0]).decode()
            for row in csv.reader(io.StringIO(content)):
                if not row or not row[0].isdigit():
                    continue
                stamp = int(row[0])
                if not first - 24 * HOUR <= stamp < last:
                    continue
                if stamp in rows or stamp % HOUR or int(row[6]) != stamp + HOUR - 1:
                    raise ValueError("归档 K 线重复或时间边界错误")
                v = [float(row[k]) for k in (1, 2, 3, 4, 7, 10)]
                if (
                    not all(math.isfinite(x) for x in v)
                    or v[4] < 0
                    or not 0 <= v[5] <= v[4] * (1 + 1e-8)
                ):
                    raise ValueError("归档成交量不合法")
                rows[stamp] = {
                    "symbol": symbol,
                    "open_ms": stamp,
                    **dict(
                        zip(
                            ("open", "high", "low", "close", "quote_volume", "taker_buy_quote"),
                            v,
                            strict=True,
                        )
                    ),
                }
        if sorted(rows) != list(range(first - 24 * HOUR, last, HOUR)):
            raise ValueError("预热或回放行情不完整，禁止补价")
        raw[symbol] = rows
        funding_seen = set()
        for source in manifest["funding_sources"][symbol]:
            stamp = parse_qs(urlparse(source["url"]).query)["startTime"][0]
            if not stamp.isdigit():
                raise ValueError("资金费来源时钟不合法")
            path = root / "funding_raw" / symbol / f"{stamp}.json"
            if source["status"] != 200 or file_hash(path) != source["sha256"]:
                raise ValueError("资金费原文哈希不匹配")
            source_records.append({"url": source["url"], "sha256": source["sha256"]})
            for row in read(path):
                clock = int(row["fundingTime"])
                if first - 24 * HOUR <= clock < last:
                    if row["symbol"] != symbol or clock in funding_seen:
                        raise ValueError("资金费标的或事件唯一性错误")
                    funding_seen.add(clock)
                    funding_all.append(
                        {
                            "symbol": symbol,
                            "time_ms": clock,
                            "rate": float(row["fundingRate"]),
                            "mark_price": float(row["markPrice"]),
                            "source_url": source["url"],
                        }
                    )
    source_hash = digest(source_records)
    external = read(evidence_path) if evidence_path else []
    snapshots = []
    for clock in range(first, last - HOUR, every_hours * HOUR):
        as_of = clock + HOUR + 60000
        for symbol in sorted(symbols):
            window = [raw[symbol][t] for t in range(clock - 23 * HOUR, clock + HOUR, HOUR)]
            volume = sum(r["quote_volume"] for r in window)
            buy = sum(r["taker_buy_quote"] for r in window)
            features = {
                "momentum": math.tanh(
                    10 * math.log(window[-1]["close"] / raw[symbol][clock - 24 * HOUR]["close"])
                ),
                "flow": 2 * buy / volume - 1 if volume else 0.0,
                "crowding": 0.0,
                "activity": 1.0,
            }
            evidence = []
            known = [
                e
                for e in funding_all
                if e["symbol"] == symbol
                and e["time_ms"] + 60000 <= as_of
                and as_of < e["time_ms"] + 24 * HOUR
            ]
            if known:
                latest = max(known, key=lambda e: e["time_ms"])
                text = f"{symbol} 已结算资金费率 {latest['rate']}，结算时间戳 {latest['time_ms']}。正值由多头支付，负值由空头支付；不代表未来费率。"
                evidence.append(
                    {
                        "id": f"funding-{symbol}-{latest['time_ms']}",
                        "kind": "funding",
                        "source_url": latest["source_url"],
                        "published_at_ms": latest["time_ms"],
                        "available_at_ms": latest["time_ms"] + 60000,
                        "availability_basis": "assumed_publication",
                        "expires_at_ms": latest["time_ms"] + 24 * HOUR,
                        "text": text,
                        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "funding_rate": latest["rate"],
                    }
                )
            evidence.extend(
                {k: v for k, v in e.items() if k != "symbol"}
                for e in external
                if e["symbol"] == symbol and e["available_at_ms"] <= as_of < e["expires_at_ms"]
            )
            snapshots.append(
                {
                    "symbol": symbol,
                    "as_of_ms": as_of,
                    "features_available_at_ms": as_of,
                    "availability_basis": "assumed_publication",
                    "valid_until_ms": as_of + 6 * HOUR,
                    "features": features,
                    "evidence": evidence,
                    "data_gaps": [
                        "未接入真实历史持仓拥挤度，crowding 固定为中性零",
                        "历史公开可用时间采用延迟假设",
                    ],
                    "source_sha256": source_hash,
                }
            )
    return {
        "schema_version": "cz-ta-dataset.v1",
        "bar_ms": HOUR,
        "symbols": sorted(symbols),
        "bars": [raw[s][t] for t in range(first, last, HOUR) for s in sorted(symbols)],
        "funding": [e for e in funding_all if first <= e["time_ms"] < last],
        "snapshots": snapshots,
        "provenance": {
            "kind": "retrospective_public_archive",
            "manifest_sha256": file_hash(root / "manifest.json"),
            "source_records": source_records,
            "external_evidence_sha256": file_hash(evidence_path) if evidence_path else None,
            "availability": "historical_publication_plus_60_seconds_assumed",
            "funding_completeness_independently_verified": False,
            "news_archive_supplied": bool(evidence_path),
            "universe": "caller_fixed_for_integration",
            "independent_oos": False,
        },
    }


def demo_dataset():
    """确定性合成数据只用于安装自检，不输出市场盈利结论。"""
    first = utc_ms("2026-08-01")
    bars, snapshots = [], []
    text = "The protocol upgrade was approved."
    for i in range(30):
        price = 100 + i * 0.2
        bars.append(
            {
                "symbol": "DEMOUSDT",
                "open_ms": first + i * HOUR,
                "open": price,
                "close": price + 0.1,
                "high": price + 0.3,
                "low": price - 0.2,
            }
        )
        if i in (1, 9, 17):
            as_of = first + i * HOUR + 60000
            snapshots.append(
                {
                    "symbol": "DEMOUSDT",
                    "as_of_ms": as_of,
                    "features_available_at_ms": as_of,
                    "availability_basis": "observed",
                    "valid_until_ms": as_of + 6 * HOUR,
                    "source_sha256": digest("synthetic-fixture"),
                    "data_gaps": ["合成数据，仅用于联调"],
                    "features": {"momentum": 0.6, "flow": 0.4, "crowding": 0.0, "activity": 1.0},
                    "evidence": [
                        {
                            "id": f"demo-{i}",
                            "kind": "news",
                            "source_url": "https://example.org/demo",
                            "published_at_ms": as_of - 1000,
                            "available_at_ms": as_of,
                            "availability_basis": "observed",
                            "expires_at_ms": as_of + 6 * HOUR,
                            "text": text,
                            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                            "funding_rate": None,
                        }
                    ],
                }
            )
    return {
        "schema_version": "cz-ta-dataset.v1",
        "bar_ms": HOUR,
        "symbols": ["DEMOUSDT"],
        "bars": bars,
        "funding": [
            {"symbol": "DEMOUSDT", "time_ms": first + 4 * HOUR, "rate": 0.001, "mark_price": 101.0}
        ],
        "snapshots": snapshots,
        "provenance": {
            "kind": "synthetic_fixture",
            "independent_oos": False,
            "funding_completeness_independently_verified": False,
        },
    }
