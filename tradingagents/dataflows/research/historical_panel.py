"""由封存分钟及显式类型资金费生成完整小时双表，不执行模型或账户。"""

from __future__ import annotations

import argparse
import gzip
import importlib.metadata
import json
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from tradingagents.research.quant.artifacts import read, save, sha
from tradingagents.research.quant.factors.funding import (
    NAMES as FUNDING_NAMES,
    calculate as funding_factors,
)

MINUTE, HOUR = 60_000, 3_600_000
START_MS, END_MS = 1785542400000, 1790812800000
SYMBOL_COUNT, ORIGINAL_COUNT = 909, 748
MONTHS = ("2026-08", "2026-09")
HORIZONS = (20, 60, 240)
LEGACY_NAMES = (
    "return_3m",
    "return_20m",
    "return_60m",
    "flow_3m",
    "flow_12m",
    "volatility_60m",
    "log_quote_volume_60m",
)
FEATURES = LEGACY_NAMES + FUNDING_NAMES
IDENTITY = ("symbol", "asset_class", "clock_ms")
RAW_COLUMNS = (
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
ASSET_TYPES = {"PERPETUAL": "CRYPTO_PERPETUAL", "TRADIFI_PERPETUAL": "TRADIFI_PERPETUAL"}
HERE = Path(__file__).resolve().parent


def month_bounds(month):
    """右开月份边界只用UTC日历，不舍入资金费原毫秒。"""
    if month not in MONTHS:
        raise ValueError("月份不属于固定历史范围")
    start = pd.Timestamp(month + "-01", tz="UTC")
    return int(start.value // 1_000_000), int((start + pd.offsets.MonthBegin()).value // 1_000_000)


def _grid(frame):
    """所有窗口必须在真实日历网格上计算，不能压缩缺分钟。"""
    values = frame.index.to_numpy()
    if frame.index.has_duplicates or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("分钟索引必须唯一且为整数毫秒")
    if len(values) > 1 and not (np.diff(values) == MINUTE).all():
        raise ValueError("分钟网格压缩了缺失时间")


def price_features(frame):
    """原七项纯特征；不调用包含未来标签计算的旧面板函数。"""
    _grid(frame)
    close = np.log(frame["close"])
    result = pd.DataFrame(index=frame.index)
    for length in (3, 20, 60):
        result[f"return_{length}m"] = close - close.shift(length)
    for length in (3, 12):
        quote = frame["quote_volume"].rolling(length, min_periods=length).sum()
        buyer = frame["taker_buy_quote"].rolling(length, min_periods=length).sum()
        result[f"flow_{length}m"] = 2 * buyer / quote.where(quote > 0) - 1
    result["volatility_60m"] = close.diff().rolling(60, min_periods=60).std(ddof=1)
    result["log_quote_volume_60m"] = np.log1p(
        frame["quote_volume"].rolling(60, min_periods=60).sum()
    )
    complete = (
        np.isfinite(frame[["close", "quote_volume", "taker_buy_quote"]])
        .all(axis=1)
        .rolling(61, min_periods=61)
        .sum()
        .eq(61)
    )
    result.loc[~complete, list(LEGACY_NAMES)] = np.nan
    return result.replace([np.inf, -np.inf], np.nan)


def future_labels(frame, contract, horizons=HORIZONS):
    """独立计算旧开盘到开盘标签；端点与持有路径分别声明。"""
    _grid(frame)
    if any(type(h) is not int or h <= 0 for h in horizons) or len(set(horizons)) != len(horizons):
        raise ValueError("标签期限必须是互异正整数分钟")
    result = pd.DataFrame(index=frame.index)
    entry = frame["open"].shift(-2)
    observed = np.isfinite(frame[["open", "high", "low", "close"]]).all(axis=1)
    for horizon in horizons:
        exit_price = frame["open"].shift(-(horizon + 2))
        endpoints = np.isfinite(entry) & np.isfinite(exit_price) & (entry > 0) & (exit_price > 0)
        entry_eligible, entry_status = qualification(frame.index.to_numpy() + MINUTE, contract)
        exit_eligible, exit_status = qualification(
            frame.index.to_numpy() + (horizon + 1) * MINUTE, contract
        )
        endpoint_eligible = entry_eligible & exit_eligible
        label_status = np.full(len(frame), "KNOWN", dtype=object)
        label_status[~endpoints] = "UNKNOWN_OPEN_ENDPOINT"
        label_status[~endpoint_eligible] = "OUTSIDE_STATIC_INTERVAL"
        unknown = np.char.startswith(entry_status.astype(str), "UNKNOWN") | np.char.startswith(
            exit_status.astype(str), "UNKNOWN"
        )
        label_status[unknown] = "UNKNOWN_QUALIFICATION"
        result[f"target_{horizon}"] = (exit_price / entry - 1).where(endpoints & endpoint_eligible)
        result[f"label_end_{horizon}_ms"] = frame.index.to_numpy() + (horizon + 1) * MINUTE
        result[f"endpoint_observed_{horizon}"] = endpoints
        result[f"endpoint_eligible_{horizon}"] = endpoint_eligible
        result[f"label_status_{horizon}"] = label_status
        # 持有期间为入场分钟至退出前一分钟，退出open另由端点状态核验。
        result[f"holding_path_observed_{horizon}"] = endpoints & (
            observed.rolling(horizon, min_periods=horizon).sum().shift(-(horizon + 1)) == horizon
        )
    return result


def universe_from_metadata(symbols, original, metadata):
    """只按明确合约类型与晚期静态区间假设分类，未知身份不删除。"""
    if (
        len(symbols) != len(set(symbols))
        or len(original) != len(set(original))
        or not set(original) <= set(symbols)
    ):
        raise ValueError("固定全集重复或原748身份越界")
    if len(metadata) != len(symbols) or {r["symbol"] for r in metadata} != set(symbols):
        raise ValueError("资格调查与固定全集不双射")
    result = {}
    for row in metadata:
        symbol, snapshot = row["symbol"], row.get("snapshot")
        if row.get("original_748_member") is not (symbol in original):
            raise ValueError("原748与补充身份不一致")
        if snapshot is not None and snapshot.get("symbol") != symbol:
            raise ValueError("资格快照合约身份不同")
        found = row.get("snapshot_symbol_found") is True and isinstance(snapshot, dict)
        kind = ASSET_TYPES.get(snapshot.get("contractType"), "UNKNOWN") if found else "UNKNOWN"
        onboard, delivery = (
            snapshot.get(k) if found else None for k in ("onboardDate", "deliveryDate")
        )
        known_interval = all(type(v) is int and v >= 0 for v in (onboard, delivery))
        known_interval = known_interval and (delivery == 0 or onboard < delivery)
        result[symbol] = {
            "asset_class": kind,
            "onboard_ms": onboard if known_interval else None,
            "delivery_ms": delivery if known_interval else None,
            "qualification_interval_known": bool(known_interval),
            "snapshot_symbol_found": found,
            "original_748_member": symbol in original,
            "historical_point_in_time_metadata_proven": False,
        }
    return {symbol: result[symbol] for symbol in sorted(symbols)}


def qualification(clocks, contract):
    """资格与有限特征是两个独立状态，均不读取未来标签。"""
    values = np.asarray(clocks)
    status = np.full(len(values), "UNKNOWN_METADATA", dtype=object)
    eligible = np.zeros(len(values), dtype=bool)
    if contract["snapshot_symbol_found"]:
        status[:] = "UNKNOWN_TYPE"
        if contract["asset_class"] != "UNKNOWN":
            status[:] = "UNKNOWN_INTERVAL"
            if contract["qualification_interval_known"]:
                eligible = (values > contract["onboard_ms"]) & (
                    (contract["delivery_ms"] == 0) | (values < contract["delivery_ms"])
                )
                status[:] = "OUTSIDE_STATIC_INTERVAL"
                status[eligible] = "ELIGIBLE_UNDER_STATIC_ASSUMPTION"
    return eligible, status


def regular_factors(clocks, symbol, events, coverage):
    """仅在连续可信月份段调用原算子，未知月份与Special不能补零。"""
    rows = sorted(coverage, key=lambda row: row["start_ms"])
    if len(rows) != 2 or {row["month"] for row in rows} != set(MONTHS):
        raise ValueError("每个合约必须保留两个资金费月份")
    ready, by_month = {}, defaultdict(list)
    for row in rows:
        if row["symbol"] != symbol or (row["start_ms"], row["end_exclusive_ms"]) != month_bounds(
            row["month"]
        ):
            raise ValueError("资金费覆盖身份或月边界不同")
        if type(row["month_ready_under_source_assumptions"]) is not bool:
            raise ValueError("资金费覆盖状态必须明确为布尔值")
        ready[row["month"]] = row["month_ready_under_source_assumptions"]
    for event in events:
        timestamp = event.get("time_ms")
        if (
            event.get("symbol") != symbol
            or type(timestamp) is not int
            or not START_MS <= timestamp < END_MS
        ):
            raise ValueError("类型资金费事件身份或原毫秒越界")
        month = MONTHS[int(timestamp >= month_bounds(MONTHS[1])[0])]
        if event.get("type") not in {"Regular", "Special"}:
            raise ValueError("资金费事件类型不明确")
        quality = event.get("quality", {})
        if ready[month] and (
            quality.get("explicit_type_verified") is not True
            or quality.get("identity_conflict") is not False
            or quality.get("trusted_typed_fields") is not True
        ):
            raise ValueError("可信月份包含未知类型或冲突事件")
        if ready[month] and event["type"] == "Regular":
            decimal = Decimal(event["rate"])
            rate = float(decimal)
            if not decimal.is_finite() or not np.isfinite(rate) or (decimal != 0 and rate == 0):
                raise ValueError("普通资金费率不能转换为原算子有限数值")
            by_month[month].append({"symbol": symbol, "time_ms": timestamp, "rate": rate})
    for row in rows:
        month = row["month"]
        if ready[month] and len(by_month[month]) != row["observed_regular_events"]:
            raise ValueError("可信月份普通资金费事件数量不守恒")
        times = [event["time_ms"] for event in by_month[month]]
        if len(set(times)) != len(times):
            raise ValueError("可信月份普通资金费时间戳重复")
    segments = []
    for row in rows:
        if not ready[row["month"]]:
            continue
        if segments and segments[-1][1] == row["start_ms"]:
            segments[-1][1] = row["end_exclusive_ms"]
            segments[-1][2].extend(by_month[row["month"]])
        else:
            segments.append(
                [row["start_ms"], row["end_exclusive_ms"], list(by_month[row["month"]])]
            )
    result = pd.DataFrame(np.nan, index=pd.Index(clocks), columns=FUNDING_NAMES)
    for start, end, normalized in segments:
        current = funding_factors(clocks, normalized, start, end - 1)
        for name in FUNDING_NAMES:
            result[name] = result[name].combine_first(current[name])
    return result


class BoundInputs:
    """只强锁实际消费的封存文件；整包背景不参与执行身份。"""

    def __init__(self, root, identities):
        self.root = Path(root).resolve()
        self.expected = read(self.root / "receipts.json")
        self.identities = identities
        self._add(self.root / "receipts.json", sha(self.root / "receipts.json"))

    def _add(self, path, expected):
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(self.root)
            or sha(path) != expected
        ):
            raise ValueError("实际消费的封存来源身份变化")
        self.identities[str(path)] = expected
        return path

    def add(self, name):
        if name not in self.expected:
            raise ValueError("实际消费文件未列入来源收据")
        return self._add(self.root / name, self.expected[name])


def minute_sources(bound, plan):
    """从已封存的最后尝试选分钟源，未列出月份仍保留独立状态。"""
    result, seen = defaultdict(list), set()
    plan_sha = sha(bound.root / "capture_plan.json")
    attempts_by_task = defaultdict(list)
    for name in bound.expected:
        match = re.fullmatch(r"archive_jobs/(1m_[^/]+)/attempt_\d{4}\.json", name)
        if match:
            attempts_by_task[match[1]].append(name)
    for task in plan["archive_tasks"]:
        if task["kind"] != "1m":
            continue
        key = task["symbol"], task["month"]
        if key in seen or task["symbol"] not in plan["symbols"] or task["month"] not in MONTHS:
            raise ValueError("分钟归档任务重复或越界")
        seen.add(key)
        prefix = "archive_jobs/" + task["id"] + "/"
        attempts = sorted(attempts_by_task[task["id"]])
        if not attempts:
            raise ValueError("封存分钟归档缺少最终尝试")
        record = read(bound.add(attempts[-1]))
        if (
            record.get("plan_sha256") != plan_sha
            or any(record.get(k) != task[k] for k in ("symbol", "month", "kind"))
            or record.get("task_id") != task["id"]
        ):
            raise ValueError("分钟归档最终尝试与冻结任务不同")
        status, path = record["status"], None
        if status == "VERIFIED":
            names = [
                name
                for name in record["files"]
                if re.fullmatch(re.escape(prefix) + r"normalized_\d{4}\.csv\.gz", name)
            ]
            if len(names) != 1 or record["files"][names[0]] != bound.expected.get(names[0]):
                raise ValueError("已验证分钟归档没有唯一规范原文")
            if record["profile"]["all_12_candle_fields_retained"] is not True:
                raise ValueError("分钟原文未保留全部12字段")
            path = bound.add(names[0])
        elif status not in {
            "NOT_LISTED",
            "CHECKSUM_NOT_LISTED_UNKNOWN",
            "LISTED_PUBLIC_404_UNKNOWN",
        }:
            raise ValueError("分钟归档未完成，不能生成完整面板")
        result[task["symbol"]].append(
            {"month": task["month"], "status": status, "path": path, "profile": record["profile"]}
        )
    if seen != {(symbol, month) for symbol in plan["symbols"] for month in MONTHS}:
        raise ValueError("分钟归档没有覆盖全部固定合约月份身份")
    return result


def load_minutes(symbol, sources):
    """逐合约重读12字段，缺分钟只在完整日历中成为NaN。"""
    frames = []
    for source in sources:
        if source["path"] is None:
            continue
        with gzip.open(source["path"], "rt", encoding="utf-8") as stream:
            if stream.readline().strip().split(",") != [
                *RAW_COLUMNS,
                "completed_boundary_ms",
                "symbol",
            ]:
                raise ValueError("规范分钟原文字段不同")
        frame = pd.read_csv(source["path"], compression="gzip", dtype={"symbol": str})
        start, end = month_bounds(source["month"])
        clock = frame["open_time"].to_numpy()
        if any(
            not pd.api.types.is_integer_dtype(frame[name])
            for name in ("open_time", "close_time", "completed_boundary_ms", "count")
        ):
            raise ValueError("分钟时钟或交易数不是整数")
        numeric = frame[[name for name in RAW_COLUMNS if name != "ignore"]].to_numpy(dtype=float)
        if (
            not np.isfinite(numeric).all()
            or not frame["symbol"].eq(symbol).all()
            or len(frame) != source["profile"]["rows"]
        ):
            raise ValueError("规范分钟原文存在未知值或身份/数量差异")
        if (
            not ((clock >= start) & (clock < end) & (clock % MINUTE == 0)).all()
            or not frame["close_time"].eq(frame["open_time"] + MINUTE - 1).all()
            or not frame["completed_boundary_ms"].eq(frame["open_time"] + MINUTE).all()
        ):
            raise ValueError("规范分钟完成边界或月份非法")
        prices = frame[["open", "high", "low", "close"]]
        nonnegative = frame[
            ["volume", "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume"]
        ]
        if (
            not prices.gt(0).all().all()
            or not nonnegative.ge(0).all().all()
            or not frame["low"].le(frame[["open", "close"]].min(axis=1)).all()
            or not frame["high"].ge(frame[["open", "close"]].max(axis=1)).all()
        ):
            raise ValueError("分钟OHLC或非负数量非法")
        for buyer, total in (
            ("taker_buy_volume", "volume"),
            ("taker_buy_quote_volume", "quote_volume"),
        ):
            if not frame[buyer].le(frame[total] + 1e-8 * np.maximum(1, frame[total])).all():
                raise ValueError("分钟主动买量超过总量")
        frames.append(frame.set_index("completed_boundary_ms"))
    calendar = np.arange(START_MS + MINUTE, END_MS + 1, MINUTE, dtype=np.int64)
    if not frames:
        return pd.DataFrame(np.nan, index=calendar, columns=[*RAW_COLUMNS, "taker_buy_quote"])
    merged = pd.concat(frames)
    if merged.index.has_duplicates or not merged.index.is_monotonic_increasing:
        raise ValueError("分钟原文存在重复或倒序，不静默排序/去重")
    merged["taker_buy_quote"] = merged["taker_buy_quote_volume"]
    return merged.drop(columns="symbol").reindex(calendar)


def _stable(identities):
    """末验只锁实际依赖，后台其他模块开发不会使本次来源失效。"""
    if any(sha(path) != expected for path, expected in identities.items()):
        raise ValueError("运行中实际来源或执行源码改变")


def coverage_summary(features, labels, profiles):
    """覆盖沿完整日历守恒；特征计数不读取未来目标。"""
    current = features[list(IDENTITY)].copy()
    current["feature_eligible"] = features["feature_eligible"]
    current["finite_nine"] = np.isfinite(features[list(FEATURES)]).all(axis=1)
    current["scheduled_rows"] = 1
    clocks = pd.to_datetime(current["clock_ms"], unit="ms", utc=True)
    current["utc_date"], current["utc_hour"] = clocks.dt.strftime("%Y-%m-%d"), clocks.dt.hour
    counts = ["scheduled_rows", "feature_eligible", "finite_nine"]
    axes = {}
    for name, keys in (
        ("by_utc_date", ["utc_date"]),
        ("by_utc_hour", ["utc_hour"]),
        ("by_asset_clock", ["asset_class", "clock_ms"]),
    ):
        axes[name] = (
            current.groupby(keys, sort=True, observed=True)[counts]
            .sum()
            .reset_index()
            .to_dict("records")
        )
    return {
        "symbols": profiles,
        "row_count": len(features),
        "hour_count": int(features["clock_ms"].nunique()),
        "asset_class_members": features.groupby("asset_class", observed=True)["symbol"]
        .nunique()
        .to_dict(),
        "qualification_status_counts": features["qualification_status"].value_counts().to_dict(),
        "label_status_counts": {
            str(h): labels[f"label_status_{h}"].value_counts().to_dict() for h in HORIZONS
        },
        **axes,
    }


def run(source, funding_view, output):
    """先冻结实际来源与双表协议，再计算任何真实新特征或标签。"""
    source, funding_view, output = (Path(p).resolve() for p in (source, funding_view, output))
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in (source, funding_view)):
        raise ValueError("新输出不能覆盖或包含封存来源")
    inputs = {}
    raw, typed = BoundInputs(source, inputs), BoundInputs(funding_view, inputs)
    plan = read(raw.add("capture_plan.json"))
    if (
        plan.get("schema") != "ta-fixed-historical-public-capture.v1"
        or plan.get("start_ms") != START_MS
        or plan.get("end_exclusive_ms") != END_MS
        or plan.get("months") != list(MONTHS)
        or len(plan["symbols"]) != SYMBOL_COUNT
        or len(plan["original_universe"]) != ORIGINAL_COUNT
        or any(
            plan.get(k) is not False
            for k in ("orders_enabled", "leader_labels_read", "returns_evaluated")
        )
    ):
        raise ValueError("原始来源与事前909历史协议不同")
    if read(raw.add("result.json")).get("capture_completed") is not True:
        raise ValueError("来源采集尚未封存完成")
    metadata = read(raw.add("inputs/eligibility_symbol_metadata.json"))
    universe = universe_from_metadata(plan["symbols"], plan["original_universe"], metadata)
    minutes = minute_sources(raw, plan)
    comparisons = read(raw.add("funding_month_comparisons.json"))
    originals = {(r["symbol"], r["month"]): r for r in comparisons}
    funding_protocol = read(typed.add("protocol.json"))
    if funding_protocol.get(
        "schema"
    ) != "ta-historical-typed-funding-view.v1" or funding_protocol.get("input_identities", {}).get(
        str(source / "receipts.json")
    ) != sha(source / "receipts.json"):
        raise ValueError("类型资金费视图未绑定同一采集来源")
    if read(typed.add("result.json")).get("status") not in {"COMPLETED", "COMPLETED_WITH_UNKNOWN"}:
        raise ValueError("类型资金费视图未完整封存")
    coverage = read(typed.add("coverage.json"))
    if (
        len(coverage) != SYMBOL_COUNT * len(MONTHS)
        or len(comparisons) != SYMBOL_COUNT * len(MONTHS)
        or len(originals) != SYMBOL_COUNT * len(MONTHS)
        or {(r["symbol"], r["month"]) for r in coverage}
        != {(s, m) for s in universe for m in MONTHS}
    ):
        raise ValueError("资金费覆盖没有保留全部1818身份")
    for row in coverage:
        if row["original_comparison"] != originals[row["symbol"], row["month"]]:
            raise ValueError("原月份对账状态被改写")
    event_path = typed.add("typed_cash_events.jsonl")
    paths = [
        Path(__file__).resolve(),
        Path(funding_factors.__code__.co_filename).resolve(),
        Path(sha.__code__.co_filename).resolve(),
        HERE / "historical_panel_plan.md",
        HERE.parents[1] / "research/quant/plans/extended_funding_plan.md",
    ]
    sources = {str(p): sha(p) for p in paths}
    hours = np.arange(START_MS, END_MS, HOUR, dtype=np.int64)
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "schema": "ta-historical-funding-panel.v1",
        "created_ms": time.time_ns() // 1_000_000,
        "start_ms": START_MS,
        "end_ms": END_MS,
        "end_exclusive_ms": END_MS,
        "scheduled_hours": len(hours),
        "scheduled_rows": SYMBOL_COUNT * len(hours),
        "universe": universe,
        "source_root": str(source),
        "capture_root": str(source),
        "funding_view_root": str(funding_view),
        "capture_receipts_sha256": sha(source / "receipts.json"),
        "funding_view_receipts_sha256": sha(funding_view / "receipts.json"),
        "features": FEATURES,
        "identity": IDENTITY,
        "horizons_minutes": HORIZONS,
        "primary_horizon_minutes": 240,
        "input_identities": inputs,
        "source_sha256": sources,
        "feature_file": "features.parquet",
        "label_file": "labels.parquet",
        "feature_eligibility": "explicit_type_and_late_snapshot_static_interval_only",
        "feature_population_uses_future_labels": False,
        "minimum_members_per_asset_clock": 20,
        "funding_information_delay_ms": 300000,
        "feature_event_types": ["Regular"],
        "labels_are_endpoint_qualified_price_returns": True,
        "labels_not_masked_by_current_feature_eligibility": True,
        "study_plan_sha256": sha(paths[-1]),
        "label_end_rule": "clock_ms+(horizon+1)*60000; strict_less_than_refit_in_study",
        "holding_path_does_not_select_feature_members": True,
        "historical_point_in_time_metadata_proven": False,
        "historical_first_publication_time_proven": False,
        "independent_funding_completeness_proven": False,
        "economic_account_evaluated": False,
        "leader_labels_read": False,
        "orders_enabled": False,
        "production_eligible": False,
        "runtime": {
            "python": sys.version,
            "libraries": {n: importlib.metadata.version(n) for n in ("numpy", "pandas", "pyarrow")},
        },
    }
    save(output / "protocol.json", protocol)
    for path in paths:
        target = output / "source" / path.relative_to(HERE.parents[2])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    errors, profiles = [], []
    try:
        by_symbol, cov_by_symbol = defaultdict(list), defaultdict(list)
        capture_receipt_sha = sha(source / "receipts.json")
        with event_path.open(encoding="utf-8") as stream:
            for line in stream:
                event = json.loads(
                    line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value))
                )
                if (
                    event.get("symbol") not in universe
                    or event.get("source", {}).get("capture_receipts_sha256") != capture_receipt_sha
                ):
                    raise ValueError("类型事件不属于同一固定采集来源")
                # 原文由封存视图保留；计算只在内存保留普通算子所需字段。
                by_symbol[event["symbol"]].append(
                    {key: event[key] for key in ("symbol", "time_ms", "rate", "type", "quality")}
                )
        for row in coverage:
            cov_by_symbol[row["symbol"]].append(row)
        feature_parts, label_parts = [], []
        for number, (symbol, contract) in enumerate(universe.items(), 1):
            frame = load_minutes(symbol, sorted(minutes[symbol], key=lambda r: r["month"]))
            featured = price_features(frame).reindex(hours)
            featured[list(FUNDING_NAMES)] = regular_factors(
                hours, symbol, by_symbol[symbol], cov_by_symbol[symbol]
            )
            eligible, states = qualification(hours, contract)
            featured.loc[~eligible, list(FEATURES)] = np.nan
            featured["feature_eligible"], featured["qualification_status"] = eligible, states
            featured["original_748_member"] = contract["original_748_member"]
            # 决策首钟本身没有完成分钟，但其未来入场和退出open仍可观察。
            # 仅标签轴补起点虚拟NaN行，原分钟接口与特征预热保持不变。
            label_frame = frame.reindex(np.concatenate(([START_MS], frame.index.to_numpy())))
            labels = future_labels(label_frame, contract).reindex(hours)
            for table in (featured, labels):
                table["symbol"], table["asset_class"], table["clock_ms"] = (
                    symbol,
                    contract["asset_class"],
                    hours,
                )
            feature_parts.append(featured.reset_index(drop=True))
            label_parts.append(labels.reset_index(drop=True))
            profiles.append(
                {
                    "symbol": symbol,
                    "asset_class": contract["asset_class"],
                    "original_748_member": contract["original_748_member"],
                    "scheduled_hours": len(hours),
                    "minute_rows": int(frame["open"].notna().sum()),
                    "qualification_status_counts": dict(Counter(states)),
                    "finite_feature_counts": {
                        f: int(np.isfinite(featured[f]).sum()) for f in FEATURES
                    },
                    "finite_nine_hours": int(
                        np.isfinite(featured[list(FEATURES)]).all(axis=1).sum()
                    ),
                    "known_labels": {
                        str(h): int(np.isfinite(labels[f"target_{h}"]).sum()) for h in HORIZONS
                    },
                    "minute_source_status": [
                        {"month": r["month"], "status": r["status"]} for r in minutes[symbol]
                    ],
                    "funding_month_ready": {
                        r["month"]: r["month_ready_under_source_assumptions"]
                        for r in cov_by_symbol[symbol]
                    },
                }
            )
            if number % 100 == 0:
                print({"prepared_symbols": number, "scheduled_symbols": 909}, flush=True)
        features, labels = (
            pd.concat(feature_parts, ignore_index=True),
            pd.concat(label_parts, ignore_index=True),
        )
        if (
            len(features) != SYMBOL_COUNT * len(hours)
            or features.duplicated(list(IDENTITY)).any()
            or not features[list(IDENTITY)].equals(labels[list(IDENTITY)])
        ):
            raise ValueError("双表完整小时身份不守恒")
        features.to_parquet(output / "features.parquet", index=False)
        labels.to_parquet(output / "labels.parquet", index=False)
        save(
            output / "coverage.json",
            coverage_summary(features, labels, profiles),
        )
        _stable(inputs)
        _stable(sources)
        result = {
            "status": "COMPLETED_WITH_UNKNOWN",
            "rows": len(features),
            "symbols": len(universe),
            "source_files_verified": len(inputs),
            "protocol_sha256": sha(output / "protocol.json"),
            "features_sha256": sha(output / "features.parquet"),
            "labels_sha256": sha(output / "labels.parquet"),
            "models_called": False,
            "economic_account_evaluated": False,
            "orders_enabled": False,
        }
    except Exception as error:
        errors.append({"stage": "PANEL", "error": type(error).__name__ + ":" + str(error)[:600]})
        for name in ("features.parquet", "labels.parquet", "coverage.json"):
            path = output / name
            if path.exists():
                path.rename(output / ("partial_" + name))
        result = {
            "status": "FAILED_INTEGRITY",
            "models_called": False,
            "economic_account_evaluated": False,
            "orders_enabled": False,
        }
    save(output / "errors.json", errors)
    save(output / "result.json", result)
    save(
        output / "receipts.json",
        {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()},
    )
    return result


def main(argv=None):
    """显式离线双表入口，不导入回归或账户执行器。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--funding-view", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = run(args.source, args.funding_view, args.output)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["status"] != "FAILED_INTEGRITY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
