"""四小时资金费续检的48连续账户；只消费冻结预测和显式现金来源。"""

import argparse
import gc
import gzip
import hashlib
import importlib.metadata
import json
import math
import shutil
import sys
import tempfile
import time
from collections import Counter
from decimal import Decimal, InvalidOperation
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd

from tradingagents.dataflows.research import historical_panel

from .. import extended_funding_study as study
from ..artifacts import read, save, sha
from . import account as account_module, forecast, paired_forecast, search, settlement, signals

ASSETS, METHODS, IDENTITY = study.ASSETS, study.METHODS, study.IDENTITY
START, END, REFITS = study.START, study.END, study.REFITS
TRADING_MODELS = tuple(name for name in study.MODELS if name != "controls")
COSTS, MINUTE, HOUR = (3, 10, 11), 60_000, 3_600_000
EXPECTED_SYMBOLS, ORIGINAL_COUNT = 909, 748
PREDICTION_COLUMNS = (
    *IDENTITY,
    "coefficient_mode",
    "common_member_gate",
    *(f"prediction_{name}" for name in TRADING_MODELS),
)


def cases():
    """案例只由预登记笛卡尔积产生，不能根据统计或交易结果删选。"""
    result = {
        f"{method}:{asset}:240m:{model}:{cost}bps": (method, asset, model, cost)
        for method in METHODS
        for asset in ASSETS
        for model in TRADING_MODELS
        for cost in COSTS
    }
    if len(result) != 48:
        raise ValueError("预登记账户必须恰为48条")
    return result


def validate_predictions(predictions, universe):
    """完整909乘1464乘两方式；控制模型、目标和未来路径均不加载。"""
    if set(predictions) != set(PREDICTION_COLUMNS) or predictions.columns.has_duplicates:
        raise ValueError("账户预测列必须精确等于四交易模型白名单")
    hours = np.arange(START, END, HOUR, dtype=np.int64)
    if (
        len(universe) != EXPECTED_SYMBOLS
        or set(predictions.symbol) != set(universe)
        or len(predictions) != 2 * EXPECTED_SYMBOLS * len(hours)
        or predictions[[*IDENTITY, "coefficient_mode", "common_member_gate"]].isna().any().any()
        or predictions.duplicated([*IDENTITY, "coefficient_mode"]).any()
        or set(predictions.coefficient_mode) != set(METHODS)
    ):
        raise ValueError("账户预测没有完整保留原定人口和双方式身份")
    if not pd.api.types.is_bool_dtype(predictions.common_member_gate):
        raise ValueError("共同人口门必须为明确布尔值")
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
        for value in predictions.clock_ms
    ):
        raise ValueError("预测时钟必须为整数毫秒")
    for (_, symbol), part in predictions.groupby(["coefficient_mode", "symbol"], sort=False):
        asset = universe[symbol]["asset_class"]
        if (
            asset not in (*ASSETS, "UNKNOWN")
            or not part.asset_class.eq(asset).all()
            or not np.array_equal(np.sort(part.clock_ms.to_numpy()), hours)
        ):
            raise ValueError("预测资产或完整小时日历不同")
    if predictions.groupby(list(IDENTITY)).common_member_gate.nunique().gt(1).any():
        raise ValueError("固定与滚动方式不能改变共同人口")
    unknown = predictions.asset_class.eq("UNKNOWN")
    if predictions.loc[unknown, "common_member_gate"].any():
        raise ValueError("UNKNOWN资产不能进入两主类账户")
    counts = predictions.groupby(
        ["coefficient_mode", "asset_class", "clock_ms"]
    ).common_member_gate.sum()
    if ((counts > 0) & (counts < 20)).any():
        raise ValueError("共同人口截面不能保留少于20成员")
    training = predictions.clock_ms < REFITS[0]
    for name in TRADING_MODELS:
        column = predictions[f"prediction_{name}"]
        if not pd.api.types.is_numeric_dtype(column) or pd.api.types.is_bool_dtype(column):
            raise ValueError("交易预测必须为实数或明确NaN")
        values = column.to_numpy(dtype=float)
        if (
            np.isinf(values).any()
            or np.isfinite(values[~predictions.common_member_gate]).any()
            or np.isfinite(values[training]).any()
        ):
            raise ValueError("预测越过因果人口门或训练段不是未知")


def build_schedules(predictions, universe):
    """当前完整截面不足或任一预测未知时整模型时钟不下单，仍保留状态。"""
    validate_predictions(predictions, universe)
    inside = predictions[(predictions.clock_ms >= REFITS[0]) & (predictions.clock_ms < END)]
    schedules, records = {}, []
    for (method, asset, clock), frame in inside.groupby(
        ["coefficient_mode", "asset_class", "clock_ms"], sort=True
    ):
        if asset not in ASSETS:
            continue
        common = frame[frame.common_member_gate]
        for name in TRADING_MODELS:
            score = f"prediction_{name}"
            complete = len(common) >= 20 and np.isfinite(common[score]).all()
            selected = paired_forecast.select_pairs(common, score) if complete else []
            schedules[method, asset, name, int(clock)] = selected
            records.append(
                {
                    "method": method,
                    "asset_class": asset,
                    "model": name,
                    "clock_ms": int(clock),
                    "common_members": len(common),
                    "selected": selected,
                    "status": "INSUFFICIENT_COMMON_MEMBERS"
                    if len(common) < 20
                    else "INCOMPLETE_COMMON_PREDICTIONS"
                    if not complete
                    else "PAIRED_SIGNAL"
                    if selected
                    else "NO_PAIRED_THRESHOLD_SIGNAL",
                }
            )
    expected = len(METHODS) * len(ASSETS) * len(TRADING_MODELS) * ((END - REFITS[0]) // HOUR)
    if len(schedules) != expected:
        raise ValueError("调度遗漏失败或无交易小时")
    return schedules, {
        "input_rows": len(predictions),
        "account_rows": len(inside),
        "training_rows_excluded": int((predictions.clock_ms < REFITS[0]).sum()),
        "unknown_asset_rows_retained": int(predictions.asset_class.eq("UNKNOWN").sum()),
        "scheduled_model_clocks": len(schedules),
        "status_counts": dict(Counter(row["status"] for row in records)),
        "schedules": records,
    }


def _bind_sources(source, identities):
    """预测、小时来源、价格和类型现金必须沿同一封存来源链。"""
    predicted = historical_panel.BoundInputs(source, identities)
    parent = read(predicted.add("protocol.json"))
    if (source / "failure.json").exists():
        raise ValueError("失败研究不能成为账户预测来源")
    completed = read(predicted.add("result.json"))
    expected_rows = 2 * EXPECTED_SYMBOLS * ((END - START) // HOUR)
    if (
        completed.get("schema") != "ta-extended-funding-regression-result.v1"
        or completed.get("protocol_sha256") != identities[str(source / "protocol.json")]
        or completed.get("declared_tests") != 24
        or completed.get("prediction_rows") != expected_rows
        or any(
            completed.get(key) is not False
            for key in ("orders_enabled", "leader_labels_read", "production_eligible")
        )
    ):
        raise ValueError("账户只能消费成功封存的完整研究预测")
    if (
        parent.get("schema") != "ta-extended-funding-study.v1"
        or parent.get("start_ms") != START
        or parent.get("end_ms") != END
        or parent.get("horizon_minutes") != 240
        or parent.get("label_delay_ms") != 241 * MINUTE
        or tuple(parent["refit_times_ms"]) != REFITS
        or tuple(parent["features"]) != study.FEATURES
        or {key: tuple(value) for key, value in parent["models"].items()} != study.MODELS
        or {key: tuple(value) for key, value in parent["contrasts"].items()} != study.CONTRASTS
        or any(
            parent[k] is not False
            for k in ("orders_enabled", "leader_labels_read", "production_eligible")
        )
        or parent["source_sha256"].get(str(study.PLAN.resolve())) != sha(study.PLAN)
    ):
        raise ValueError("只能消费固定四小时资金费续检预测")
    prediction_path = predicted.add("predictions.parquet")
    panel_root = Path(parent["source"]).resolve()
    panel_path = panel_root / "protocol.json"
    if parent["input_identities"].get(str(panel_path)) != sha(panel_path):
        raise ValueError("小时来源元数据没有绑定研究预测")
    identities[str(panel_path)] = sha(panel_path)
    panel = read(panel_path)
    sealed_panel = historical_panel.BoundInputs(panel_root, identities)
    panel_result = read(sealed_panel.add("result.json"))
    if any(
        parent["input_identities"].get(str(panel_root / name)) != identities[str(panel_root / name)]
        for name in ("result.json", "receipts.json")
    ):
        raise ValueError("小时面板终态收据不同于研究冻结身份")
    if (
        panel_result.get("status") not in {"COMPLETED", "COMPLETED_WITH_UNKNOWN"}
        or panel_result.get("protocol_sha256") != identities[str(panel_path)]
    ):
        raise ValueError("账户不能消费失败或部分小时面板")
    if (
        panel.get("schema") != "ta-historical-funding-panel.v1"
        or panel.get("start_ms") != START
        or panel.get("end_ms") != END
        or panel.get("study_plan_sha256") != sha(study.PLAN)
        or panel["universe"] != parent["universe"]
        or any(
            panel[k] is not False
            for k in ("orders_enabled", "leader_labels_read", "production_eligible")
        )
    ):
        raise ValueError("小时面板不是预登记的完整历史来源")
    capture_root, funding_root = (
        Path(panel[k]).resolve() for k in ("capture_root", "funding_view_root")
    )
    raw = historical_panel.BoundInputs(capture_root, identities)
    typed = historical_panel.BoundInputs(funding_root, identities)
    if (
        sha(capture_root / "receipts.json") != panel["capture_receipts_sha256"]
        or sha(funding_root / "receipts.json") != panel["funding_view_receipts_sha256"]
    ):
        raise ValueError("市场或现金来源不同于小时面板冻结身份")
    plan = read(raw.add("capture_plan.json"))
    if (
        plan.get("schema") != "ta-fixed-historical-public-capture.v1"
        or plan.get("start_ms") != START
        or plan.get("end_exclusive_ms") != END
        or len(plan["symbols"]) != EXPECTED_SYMBOLS
        or len(plan["original_universe"]) != ORIGINAL_COUNT
        or read(raw.add("result.json")).get("capture_completed") is not True
    ):
        raise ValueError("原市场采集没有完整固定人口或尚未终态")
    metadata = read(raw.add("inputs/eligibility_symbol_metadata.json"))
    universe = historical_panel.universe_from_metadata(
        plan["symbols"], plan["original_universe"], metadata
    )
    if universe != parent["universe"]:
        raise ValueError("账户不能替换晚期静态资格假设或资产分类")
    minutes = historical_panel.minute_sources(raw, plan)
    funding_protocol = read(typed.add("protocol.json"))
    if (
        funding_protocol.get("schema") != "ta-historical-typed-funding-view.v1"
        or Path(funding_protocol["source_root"]).resolve() != capture_root
        or funding_protocol.get("input_identities", {}).get(str(capture_root / "receipts.json"))
        != sha(capture_root / "receipts.json")
        or read(typed.add("result.json")).get("status")
        not in {"COMPLETED", "COMPLETED_WITH_UNKNOWN"}
    ):
        raise ValueError("现金视图没有绑定同一完整采集")
    coverage_path, events_path = typed.add("coverage.json"), typed.add("typed_cash_events.jsonl")
    for path in (coverage_path, events_path):
        if panel["input_identities"].get(str(path)) != identities[str(path)]:
            raise ValueError("现金视图不同于特征来源的冻结视图")
    return parent, universe, minutes, prediction_path, coverage_path, events_path


def load_cash(events_path, coverage_path, universe):
    """同钟Regular/Special都保留；未知月份单独记质量，绝不补零。"""
    coverage = read(coverage_path)
    expected = {(symbol, month) for symbol in universe for month in historical_panel.MONTHS}
    indexed = {(row["symbol"], row["month"]): row for row in coverage}
    if len(coverage) != len(expected) or set(indexed) != expected:
        raise ValueError("现金覆盖须保留全部合约月份")
    ready = {}
    for key, row in indexed.items():
        if (row["start_ms"], row["end_exclusive_ms"]) != historical_panel.month_bounds(
            key[1]
        ) or type(row["month_ready_under_source_assumptions"]) is not bool:
            raise ValueError("现金月份身份或就绪状态不合法")
        ready[key] = row["month_ready_under_source_assumptions"]
    symbols = {name: name for name in universe}
    events, counts, untrusted, identities = [], Counter(), Counter(), set()
    with Path(events_path).open(encoding="utf-8") as stream:
        for serial, line in enumerate(stream):
            row = json.loads(line)
            symbol, timestamp, kind = row["symbol"], row["time_ms"], row["type"]
            if (
                symbol not in universe
                or type(timestamp) is not int
                or not START <= timestamp < END
                or kind not in {"Regular", "Special"}
            ):
                raise ValueError("显式现金事件身份、原毫秒或类型不合法")
            month = historical_panel.MONTHS[
                int(timestamp >= historical_panel.month_bounds(historical_panel.MONTHS[1])[0])
            ]
            key = symbol, month
            counts[symbol, month, kind] += 1
            quality = row.get("quality", {})
            trusted = (
                quality.get("explicit_type_verified") is True
                and quality.get("trusted_typed_fields") is True
                and quality.get("identity_conflict") is False
            )
            if (
                quality.get("month_ready_under_source_assumptions")
                is not indexed[key]["month_ready_under_source_assumptions"]
            ):
                raise ValueError("事件和覆盖表的月份状态不一致")
            if not trusted:
                ready[key] = False
                untrusted[key] += 1
                continue
            try:
                rate, mark = Decimal(row["rate"]), Decimal(row["mark"])
                numeric_rate, numeric_mark = float(rate), float(mark)
            except (InvalidOperation, ValueError, TypeError, OverflowError) as error:
                raise ValueError("可信现金事件不是有效Decimal") from error
            if (
                not rate.is_finite()
                or not mark.is_finite()
                or mark <= 0
                or not math.isfinite(numeric_rate)
                or not math.isfinite(numeric_mark)
                or numeric_mark <= 0
                or (rate != 0 and numeric_rate == 0)
            ):
                raise ValueError("现金费率或mark不能转换为有限计算值")
            identity = json.dumps(
                row["raw"],
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            identity_sha = hashlib.sha256(identity.encode()).hexdigest()
            if (symbol, identity_sha) in identities:
                raise ValueError("可信现金事件有无法解释的完全重复")
            identities.add((symbol, identity_sha))
            events.append(
                {
                    "symbol": symbols[symbol],
                    "time_ms": timestamp,
                    "rate": numeric_rate,
                    "mark_price": numeric_mark,
                    "rate_type": kind,
                    "rate_decimal": str(rate),
                    "mark_decimal": str(mark),
                    "source_event_sha256": identity_sha,
                    "source_event_index": row["source"]["event_index"],
                    "source_order": serial,
                }
            )
    for (symbol, month), row in indexed.items():
        if any(
            counts[symbol, month, kind] != row[f"observed_{kind.lower()}_events"]
            for kind in ("Regular", "Special")
        ):
            raise ValueError("现金事件数量不等于冻结月份观测数")
        if ready[symbol, month] and not sum(
            counts[symbol, month, kind] for kind in ("Regular", "Special")
        ):
            raise ValueError("空现金来源不能成为已知零月份")
    events.sort(key=lambda event: (event["time_ms"], event["symbol"], event["source_order"]))
    return (
        events,
        ready,
        {
            "observed_types": {
                kind: sum(value for (_, _, current), value in counts.items() if current == kind)
                for kind in ("Regular", "Special")
            },
            "trusted_cash_types": dict(Counter(event["rate_type"] for event in events)),
            "unknown_months": [
                {"symbol": symbol, "month": month, "untrusted_events": untrusted[symbol, month]}
                for (symbol, month), known in sorted(ready.items())
                if not known
            ],
            "month_ready_under_source_assumptions": sum(ready.values()),
            "independent_completeness_proven": False,
        },
    )


def memory_estimate(symbols):
    """映射最多四个字段；预测表在分钟运行之前释放。"""
    minutes = (END - REFITS[0]) // MINUTE + 1
    points = len(cases()) * minutes
    mapped = symbols * minutes * (2 * 8 + 2)
    return {
        "market_minutes": minutes,
        "curve_points": points,
        "market_mapped_bytes": mapped,
        "curve_estimate_bytes": points * 384,
        "other_working_reserve_bytes": 512 * 1024**2,
        "working_set_estimate_bytes": mapped + points * 384 + 512 * 1024**2,
        "market_fields": ["open", "close", "observed", "qualification"],
        "market_storage": "time_major_disk_memmap",
    }


def market_frames(universe, sources, scratch, report):
    """逐合约读取后释放原12列，同一分钟只构造一次市场供48账户广播。"""
    symbols = tuple(sorted(universe))
    clocks = np.arange(REFITS[0], END + MINUTE, MINUTE, dtype=np.int64)
    price_map = flags = None
    try:
        price_map = np.lib.format.open_memmap(
            Path(scratch) / "prices.npy",
            mode="w+",
            dtype=np.float64,
            shape=(len(clocks), len(symbols), 2),
        )
        flags = np.lib.format.open_memmap(
            Path(scratch) / "flags.npy",
            mode="w+",
            dtype=np.bool_,
            shape=(len(clocks), len(symbols), 2),
        )
        for number, symbol in enumerate(symbols):
            frame = historical_panel.load_minutes(symbol, sources[symbol])
            indexes = frame.index.get_indexer(clocks)
            if (indexes < 0).any():
                raise ValueError("分钟接口没有完整覆盖账户日历")
            values = frame.iloc[indexes][["open", "close"]].to_numpy(dtype=float)
            observed = np.isfinite(values).all(axis=1) & (values > 0).all(axis=1)
            eligible, _ = historical_panel.qualification(clocks - MINUTE, universe[symbol])
            price_map[:, number, :] = values
            flags[:, number, 0], flags[:, number, 1] = observed, eligible
            report[symbol] = {
                "minutes": len(clocks),
                "observed_minutes": int(observed.sum()),
                "qualified_minutes_under_static_assumption": int(eligible.sum()),
            }
            del frame, indexes, values, observed, eligible
            if number % 32 == 31:
                gc.collect()
        price_map.flush()
        flags.flush()
        for number, clock in enumerate(clocks):
            values, known = price_map[number], flags[number]
            yield (
                int(clock),
                {
                    symbol: (
                        float(values[i, 0]),
                        float(values[i, 1]),
                        bool(known[i, 1]),
                        bool(known[i, 0]),
                    )
                    for i, symbol in enumerate(symbols)
                },
            )
    finally:
        if price_map is not None:
            price_map._mmap.close()
        if flags is not None:
            flags._mmap.close()


def reject_unknown_cash(accounts, result, ready):
    """只作财务证据失效判定，不用未来覆盖质量提前避单或选赢家。"""
    failures = {}
    for key, account in accounts.items():
        prior = result["invalid_accounts"].get(key)
        stop = prior.get("at") if prior else None
        if stop is None:
            stop = account.curve[-1]["clock"] if account.curve else None
        intervals = list(account.trades)
        if stop is not None:
            intervals.extend(
                {
                    **position,
                    "symbol": symbol,
                    "exit_ms": stop,
                    "reason": "unresolved_position_at_failure",
                }
                for symbol, position in account.positions.items()
            )
        candidates = []
        for trade in intervals:
            entry, exit_clock = trade["entry_ms"], trade["exit_ms"]
            inclusive_exit = trade["reason"] not in {
                "time_close",
                "scheduled_end",
                "unresolved_position_at_failure",
            }
            for month in historical_panel.MONTHS:
                if ready[trade["symbol"], month]:
                    continue
                first, last = historical_panel.month_bounds(month)
                intersects = max(entry, first) < min(exit_clock, last) or (
                    inclusive_exit and entry < exit_clock == first
                )
                if intersects:
                    unknown_at = max(first, entry + 1)
                    candidates.append(
                        (
                            unknown_at,
                            trade["symbol"],
                            month,
                            trade["cycle_id"],
                            entry,
                            exit_clock,
                            trade["reason"],
                        )
                    )
        if candidates:
            first, symbol, month, cycle, entry, exit_clock, reason = min(candidates)
            cash_at = (first // MINUTE + 1) * MINUTE
            prior_at = prior.get("at") if prior else None
            cutoff = min(cash_at, prior_at) if prior_at is not None else cash_at
            failure = {
                "at": cutoff,
                "type": "UNKNOWN_FUNDING_COVERAGE",
                "reason": f"实际持有涉及未知现金月份：{symbol}:{month}",
                "numeric_conclusions_allowed": False,
                "first_unknown_cash_ms": first,
                "cash_coverage_postcheck": True,
                "cash_unknown_processing_frame_ms": cash_at,
                "prior_execution_failure": prior,
                "held_interval": {
                    "symbol": symbol,
                    "cycle_id": cycle,
                    "entry_ms": entry,
                    "exit_ms": exit_clock,
                    "exit_reason": reason,
                    "exit_same_clock_cash_included": reason
                    not in {"time_close", "scheduled_end", "unresolved_position_at_failure"},
                },
            }
            if prior is not None and prior_at is not None and prior_at <= cash_at:
                failure.update(type=prior["type"], reason=prior["reason"])
            failures[key] = failure
            result["invalid_accounts"][key] = failure
            result["metrics"].pop(key, None)
            result["financial_proofs"].pop(key, None)
    return failures


def _processed(fill, reasons):
    return fill["clock"] + (
        0
        if fill["action"] == "CLOSE"
        and reasons[fill["cycle_id"]] in {"time_close", "scheduled_end"}
        else MINUTE
    )


def save_ledgers(output, definitions, accounts, invalid):
    """未知现金后的反事实轨迹不作为有效账本保存，只保留失效前前缀。"""
    for key in definitions:
        path = output / (key.replace(":", "_") + ".json.gz")
        if path.exists():
            continue
        account = accounts.get(key)
        if account is not None and key not in invalid and account.curve:
            search.save_account(path, account)
            continue
        cutoff = invalid.get(key, {}).get("at") if account is not None else None
        reasons = {row["cycle_id"]: row["reason"] for row in account.trades} if account else {}
        trim = bool(account is not None and invalid.get(key, {}).get("cash_coverage_postcheck"))
        curve = (
            [row for row in account.curve if not trim or row["clock"] < cutoff] if account else []
        )
        fills = (
            [row for row in account.fills if not trim or _processed(row, reasons) < cutoff]
            if account
            else []
        )
        trades = (
            [
                row
                for row in account.trades
                if not trim
                or row["exit_ms"]
                + (0 if row["reason"] in {"time_close", "scheduled_end"} else MINUTE)
                < cutoff
            ]
            if account
            else []
        )
        funding = (
            [
                row
                for row in account.funding_ledger
                if not trim or (row["time_ms"] // MINUTE + 1) * MINUTE < cutoff
            ]
            if account
            else []
        )
        ledger = {
            "case": definitions[key],
            "execution_started": account is not None,
            "numeric_conclusions_allowed": False,
            "metrics": None,
            "failure": invalid.get(key),
            "cash_unknown_suffix_discarded": trim,
            "ending_state": None
            if account is None or trim
            else {
                "cash": account.cash,
                "positions": account.positions,
                "pending": account.pending,
                "closes": account.closes,
                "halted": account.halted,
                "peak": account.peak,
                "maximum_drawdown": account.max_dd,
            },
            "valid_prefix_proven_under_source_assumptions": trim,
            "last_valid_curve": curve[-1] if curve and trim else None,
            "last_recorded_curve": curve[-1] if curve else None,
            "trades": trades,
            "fills": fills,
            "funding": funding,
            "curve": curve,
            "declines": [
                row for row in account.declines if not trim or row["clock"] + MINUTE < cutoff
            ]
            if account
            else [],
        }
        with gzip.open(path, "xt", encoding="utf-8") as stream:
            json.dump(ledger, stream, ensure_ascii=False, allow_nan=False)


def execution_sources():
    """只冻结实际调用依赖，其他研究开发不会改变本轮身份。"""
    return tuple(
        dict.fromkeys(
            [
                Path(__file__).resolve(),
                Path(historical_panel.__file__).resolve(),
                Path(study.__file__).resolve(),
                Path(study.PLAN).resolve(),
                Path(sha.__code__.co_filename).resolve(),
                *(
                    Path(module.__file__).resolve()
                    for module in (
                        account_module,
                        forecast,
                        paired_forecast,
                        search,
                        settlement,
                        signals,
                    )
                ),
                Path(study.__file__).parent / "funding_study.py",
                Path(study.__file__).parent / "regression.py",
                Path(study.__file__).parent / "regression_run.py",
                Path(study.__file__).parent / "factors/funding.py",
            ]
        )
    )


def run(source, output):
    """仅显式运行；先冻结实际消费来源，最后再次核验并封存全部48案例。"""
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("账户输出不能覆盖或包含研究来源")
    output.mkdir(parents=True, exist_ok=False)
    definitions, accounts = cases(), {}
    invalid = {key: {} for key in definitions}
    identities, stage = {}, "bind_sources"
    try:
        parent, universe, minutes, prediction_path, coverage_path, events_path = _bind_sources(
            source, identities
        )
        if any(output.is_relative_to(Path(path).parent) for path in identities):
            raise ValueError("账户输出不能写入消费来源目录")
        funding, ready, cash_report = load_cash(events_path, coverage_path, universe)
        used = execution_sources()
        frozen = {str(path): sha(path) for path in used}
        estimate = memory_estimate(len(universe))
        save(
            output / "protocol.json",
            {
                "schema": "ta-extended-funding-account.v1",
                "created_ms": time.time_ns() // 1_000_000,
                "source": str(source),
                "start_ms": REFITS[0],
                "end_ms": END,
                "cases": definitions,
                "prediction_columns_loaded": PREDICTION_COLUMNS,
                "future_labels_loaded": False,
                "controls_traded": False,
                "holding_minutes": 240,
                "timed_exit_at_close": False,
                "take_profit_fraction": None,
                "predicted_return_threshold": forecast.PREDICTED_RETURN_THRESHOLD,
                "account_rules": {
                    key: signals.PROTOCOL[key]
                    for key in (
                        "initial_equity",
                        "position_fraction",
                        "max_positions",
                        "decision_latency_ms",
                        "max_signal_age_ms",
                        "fee_bps",
                        "maximum_account_drawdown",
                        "minimum_trade_notional",
                        "gate",
                    )
                },
                "source_sha256": frozen,
                "input_identities": identities,
                "memory_estimate": estimate,
                "cash_unknown_invalidates_account": True,
                "cash_unknown_does_not_select_entries": True,
                "qualification_clock": "minute_open_stamp=completed_boundary-60000",
                "historical_point_in_time_metadata_proven": False,
                "independent_funding_completeness_proven": False,
                "quantity_is_continuous": True,
                "orders_enabled": False,
                "leader_labels_read": False,
                "production_eligible": False,
                "independent_oos": False,
                "runtime": {
                    "python": sys.version,
                    "libraries": {
                        name: importlib.metadata.version(name)
                        for name in ("numpy", "pandas", "pyarrow")
                    },
                },
            },
        )
        for index, path in enumerate(used):
            target = output / "source" / f"{index:02}_{path.name}"
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(path, target)
        stage = "prediction_schedule"
        predictions = pd.read_parquet(prediction_path, columns=list(PREDICTION_COLUMNS))
        schedules, report = build_schedules(predictions, universe)
        del predictions
        gc.collect()
        save(output / "decisions.json", report)
        del report
        save(output / "cash_coverage.json", cash_report)
        stage = "minute_accounts"
        market_report = {}
        if shutil.disk_usage(output).free < estimate["market_mapped_bytes"] + 512 * 1024**2:
            raise ValueError("磁盘不足以保存有界分钟映射和账本")
        with tempfile.TemporaryDirectory(prefix=".minute-market-", dir=output) as scratch:
            frames = market_frames(universe, minutes, scratch, market_report)
            try:
                accounts, result = paired_forecast.drive_accounts(
                    definitions,
                    schedules,
                    frames,
                    funding,
                    (*REFITS, END),
                    holding_minutes=240,
                    timed_exit_at_close=False,
                )
            finally:
                frames.close()
        invalid = result["invalid_accounts"]
        reject_unknown_cash(accounts, result, ready)
        save(output / "market_coverage.json", market_report)
        if result["global_failure"]:
            save(output / "partial_result.json", result)
            raise ValueError("分钟输入失败，全部账户数字失效")
        stage = "verify_after"
        if any(sha(path) != expected for path, expected in (identities | frozen).items()):
            raise ValueError("实际消费来源或执行依赖在运行期间改变")
        result.update(
            schema="ta-extended-funding-account-result.v1",
            future_labels_loaded=False,
            stable_profitability_proven=False,
            orders_enabled=False,
            production_eligible=False,
            independent_oos=False,
            economic_conditions_by_account={
                key: search.gate(value) for key, value in result["metrics"].items()
            },
            case_execution_status={
                key: "INVALID_ACCOUNT" if key in invalid else "FINANCIALLY_VERIFIED"
                for key in definitions
            },
            memory_estimate=estimate,
        )
        save(output / "result.json", result)
    except Exception as error:
        invalid = {
            key: invalid.get(key)
            or {
                "type": type(error).__name__,
                "reason": str(error),
                "numeric_conclusions_allowed": False,
            }
            for key in definitions
        }
        save(
            output / "failure.json",
            {
                "stage": stage,
                "type": type(error).__name__,
                "reason": str(error),
                "numeric_conclusions_allowed": False,
            },
        )
        raise
    finally:
        save_ledgers(output, definitions, accounts, invalid)
        save(
            output / "receipts.json",
            {
                str(path.relative_to(output)): sha(path)
                for path in output.rglob("*")
                if path.is_file()
            },
        )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.source, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
