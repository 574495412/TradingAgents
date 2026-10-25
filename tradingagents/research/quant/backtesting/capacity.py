"""同一冻结预测的48容量账户，不训练模型、选赢家或操作生产。"""

import argparse
import gc
import gzip
import importlib.metadata
import json
import math
import shutil
import sys
import tempfile
import time
from decimal import Decimal, localcontext
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd

from tradingagents.dataflows.research import historical_panel

from .. import extended_funding_study as study
from ..artifacts import save, sha
from . import (
    capacity_account,
    capacity_sources,
    extended_funding,
    forecast,
    paired_forecast,
    search,
    signals,
)

PLAN = Path(__file__).resolve().parents[1] / "plans/capacity_execution_plan.md"
ROOT = Path(__file__).resolve().parents[4]
RULE_ROOT = ROOT / "data/research/damorne_15d_alignment_20261014/dynamic_market_panel"
ELIGIBILITY = ROOT / "results/research/historical_eligibility_20261017_v001"
MINUTE = 60_000


def registered_contract():
    """拒绝把源码或配置的门槛漂移偷偷纳入同一登记研究。"""
    expected = {
        "initial_equity": 1000.0,
        "position_fraction": 0.125,
        "max_positions": 4,
        "minimum_trade_notional": 10.0,
        "fee_bps": 5.0,
        "decision_latency_ms": MINUTE,
        "max_signal_age_ms": 5 * MINUTE,
        "maximum_account_drawdown": 0.08,
    }
    gate = signals.PROTOCOL["gate"]
    if (
        any(signals.PROTOCOL.get(key) != value for key, value in expected.items())
        or Decimal(str(capacity_sources.RHO)) != Decimal("0.10")
        or forecast.PREDICTED_RETURN_THRESHOLD != 0.0016
        or gate["minimum_trades_per_phase"] != 30
        or gate["minimum_wilson_lower"] != 0.4
        or gate["minimum_profit_factor"] != 1.1
    ):
        raise ValueError("实际容量或项目规则不同于登记合同")


def plain(value):
    """账本的Decimal保留十进制字符串；身份整数和有限汇总不改类型。"""
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("非有限Decimal不能封存")
        return str(value)
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("非有限float不能封存")
    return value


def save_json(path, value):
    """普通结果与账本使用同一明确的Decimal保存契约。"""
    save(path, plain(value))


def execution_sources():
    """只冻结本入口实际调用或读取常量的代码与计划。"""
    return tuple(
        dict.fromkeys(
            [
                Path(__file__).resolve(),
                PLAN,
                *(
                    Path(module.__file__).resolve()
                    for module in (
                        capacity_account,
                        capacity_sources,
                        extended_funding,
                        historical_panel,
                        study,
                        forecast,
                        paired_forecast,
                        search,
                        signals,
                    )
                ),
                study.PLAN.resolve(),
                Path(sha.__code__.co_filename).resolve(),
                Path(study.__file__).parent / "funding_study.py",
                Path(study.__file__).parent / "regression.py",
                Path(study.__file__).parent / "regression_run.py",
                Path(study.__file__).parent / "factors/funding.py",
            ]
        )
    )


def cash_failures(accounts, ready):
    """仅在实际库存区间后验现金质量，不把未知月份用于提前避单。"""
    for account in accounts.values():
        prior = account.invalid
        stop = prior.get("at") if prior else account.last_clock
        intervals = list(account.trades)
        if stop is not None:
            intervals.extend(
                {**position, "symbol": symbol, "exit_ms": stop, "reason": "unresolved_position"}
                for symbol, position in account.positions.items()
            )
        first_unknown = None
        for trade in intervals:
            entry, end = trade["entry_ms"], trade["exit_ms"]
            for month in historical_panel.MONTHS:
                if ready[trade["symbol"], month]:
                    continue
                begin, right = historical_panel.month_bounds(month)
                if max(entry, begin) < min(end, right) or entry < end == begin:
                    candidate = (max(entry + 1, begin), trade["symbol"], month)
                    first_unknown = min(first_unknown, candidate) if first_unknown else candidate
        if first_unknown is not None:
            first, symbol, month = first_unknown
            at = (first // MINUTE + 1) * MINUTE
            earlier = prior is not None and prior["at"] <= at
            account.invalid = {
                "at": min(prior["at"], at) if prior else at,
                "type": prior["type"] if earlier else "UNKNOWN_FUNDING_COVERAGE",
                "reason": prior["reason"]
                if earlier
                else f"实际持有涉及未知现金月：{symbol}:{month}",
                "first_unknown_cash_ms": first,
                "cash_unknown_processing_frame_ms": at,
                "cash_coverage_postcheck": True,
                "prior_execution_failure": prior,
                "numeric_conclusions_allowed": False,
            }


def ledger_data(account):
    """失败数值只保留最早失效帧之前，部分成交不依赖完整trade reason索引。"""
    cutoff = account.invalid["at"] if account.invalid else None

    def before(rows, clock):
        return [row for row in rows if cutoff is None or clock(row) < cutoff]

    fields = {
        "curve": before(account.curve, lambda row: row["clock"]),
        "fills": before(account.fills, lambda row: row["processing_clock_ms"]),
        "funding": before(account.funding_ledger, lambda row: row["processing_clock_ms"]),
        "trades": before(account.trades, lambda row: row["exit_ms"] + MINUTE),
        "declines": before(account.declines, lambda row: row["processing_clock_ms"]),
        "entry_plans": before(account.entry_plans, lambda row: row["processing_clock_ms"]),
    }
    result = {
        **fields,
        "numeric_conclusions_allowed": cutoff is None,
        "failure": account.invalid,
        "finished": account.finished,
        "metrics": account.metrics() if cutoff is None else None,
        "ending_state": None
        if cutoff is not None
        else {
            "cash": account.cash,
            "positions": account.positions,
            "closes": account.closes,
            "pending": account.pending,
            "halted": account.halted,
            "peak": account.peak,
        },
        "prefix_before_processing_clock_ms": cutoff,
        "counterfactual_suffix_saved_separately": cutoff is not None,
    }
    suffix = None
    if cutoff is not None:
        suffix = {
            "numeric_conclusions_allowed": False,
            "counterfactual_or_failed_frame": True,
            "failure": account.invalid,
            **{
                name: rows[len(fields[name]) :]
                for name, rows in (
                    ("curve", account.curve),
                    ("fills", account.fills),
                    ("funding", account.funding_ledger),
                    ("trades", account.trades),
                    ("declines", account.declines),
                    ("entry_plans", account.entry_plans),
                )
            },
            "untrusted_ending_state": {
                "cash": account.cash,
                "positions": account.positions,
                "pending": account.pending,
                "closes": account.closes,
                "halted": account.halted,
            },
        }
    return result, suffix


def save_ledgers(output, definitions, accounts, fallback):
    """完整保存48身份；失败原始后缀不能混作可晋级的完整账本。"""
    for key, definition in definitions.items():
        name = key.replace(":", "_") + ".json.gz"
        account = accounts.get(key)
        if account is None:
            data, suffix = (
                {
                    "case": definition,
                    "execution_started": False,
                    "numeric_conclusions_allowed": False,
                    "failure": fallback,
                    "metrics": None,
                },
                None,
            )
        else:
            data, suffix = ledger_data(account)
            data.update(
                case=definition,
                execution_started=True,
                quantity_and_money_encoding="Decimal字符串；汇总指标有限float",
            )
        with gzip.open(output / name, "xt", encoding="utf-8") as stream:
            json.dump(
                plain(data), stream, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            )
        if suffix is not None:
            path = output / "counterfactual" / name
            path.parent.mkdir(exist_ok=True)
            with gzip.open(path, "xt", encoding="utf-8") as stream:
                json.dump(
                    plain(suffix),
                    stream,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )


def drive(definitions, schedules, frames, funding, rules, boundaries):
    """每案例只创建一次；部分退出、库存及风险跨重训连续，个别失败不删案例。"""
    first, last = boundaries[0], boundaries[-1]
    accounts = {
        key: capacity_account.CapacityAccount(
            rules, funding, slippage_bps=cost, holding_minutes=240, end_ms=last, factor=model
        )
        for key, (_, _, model, cost) in definitions.items()
    }
    states = {key: {} for key in definitions}
    block_ends = {value - MINUTE for value in boundaries[1:-1]} | {last}
    previous, count, global_failure = None, 0, None
    try:
        for clock, market in frames:
            if (
                isinstance(clock, (bool, np.bool_))
                or not isinstance(clock, Integral)
                or clock != (first if previous is None else previous + MINUTE)
                or clock > last
            ):
                raise ValueError("容量账户时间轴缺口、重复、越界或起点不同")
            clock = int(clock)
            for key, account in accounts.items():
                method, asset, model, _ = definitions[key]
                account.step(clock, market, schedules.get((method, asset, model, clock), []))
                if clock in block_ends:
                    states[key][clock] = {
                        "halted": account.halted,
                        "open_positions": len(account.positions),
                        "pending_entry": account.pending is not None,
                        "pending_exits": len(account.closes),
                        "invalid": account.invalid,
                        "peak": account.peak,
                        "maximum_drawdown": account.max_dd,
                    }
            previous, count = clock, count + 1
            if count % 1440 == 0:
                print(
                    json.dumps(
                        {
                            "processed_minutes": count,
                            "invalid_accounts": sum(
                                account.invalid is not None for account in accounts.values()
                            ),
                        }
                    ),
                    flush=True,
                )
        if previous != last or count != (last - first) // MINUTE + 1:
            raise ValueError("容量账户没有覆盖预定全部分钟")
    except Exception as error:
        at = first if previous is None else min(previous + MINUTE, last)
        global_failure = {
            "at": at,
            "type": type(error).__name__,
            "reason": str(error),
            "numeric_conclusions_allowed": False,
        }
        for account in accounts.values():
            prior = account.invalid
            account.invalid = {
                **global_failure,
                "prior_execution_failure": prior,
                "at": min(at, prior["at"]) if prior else at,
            }
    return accounts, count, states, global_failure


def block_summaries(account, boundaries, states):
    """只给完整有效账户分段归因，不把部分退出当作完整腿。"""
    points = {row["clock"]: row for row in account.curve}
    previous, rows = account.initial, []
    for begin, end in zip(boundaries[:-1], boundaries[1:], strict=False):
        clock = end if end == boundaries[-1] else end - MINUTE
        current = points[clock]["equity"]

        def inside(value, begin=begin, end=end):
            return begin <= value < end or value == end == boundaries[-1]

        with localcontext() as context:
            context.prec = 60
            contribution = (current - previous) / account.initial
            segment_return = current / previous - 1
        rows.append(
            {
                "start_ms": begin,
                "end_ms": end,
                "opening_equity": previous,
                "closing_equity": current,
                "pnl_over_initial": contribution,
                "equity_return": segment_return,
                "complete_legs": sum(inside(row["exit_ms"] + MINUTE) for row in account.trades),
                "opening_fills": sum(
                    row["action"] == "OPEN" and inside(row["processing_clock_ms"])
                    for row in account.fills
                ),
                "reduction_fills": sum(
                    row["action"] != "OPEN" and inside(row["processing_clock_ms"])
                    for row in account.fills
                ),
                "ending_state": states[clock],
            }
        )
        previous = current
    return rows


def run(source, output):
    """冻结来源和实现后显式运行48账户；不写CZ、不发订单、不自动提交。"""
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("容量输出不能覆盖或包含研究来源")
    # 在任何实际依赖调用前冻结，不把预检期间改动接纳为新基线。
    used = execution_sources()
    frozen = {str(path): sha(path) for path in used}
    registered_contract()
    # 来源预检只读；在确认输出不嵌入消费目录前不创建任何失败或临时文件。
    identities = {}
    _, universe, minutes, prediction_path, coverage_path, events_path = (
        extended_funding._bind_sources(source, identities)
    )
    rules, rule_metadata = capacity_sources.load_rules(
        universe,
        RULE_ROOT / "exchange_info.json",
        RULE_ROOT / "catalog_binding.json",
        ELIGIBILITY / "result.json",
        ELIGIBILITY / "receipts.json",
        input_identities=identities,
    )
    if any(output.is_relative_to(Path(path).parent) for path in identities):
        raise ValueError("输出不能写入实际消费来源的内部")
    if any(sha(path) != expected for path, expected in frozen.items()):
        raise ValueError("只读预检期间执行依赖改变")
    output.mkdir(parents=True, exist_ok=False)
    definitions, accounts = extended_funding.cases(), {}
    stage, failure, result, global_failure = "load_cash", None, None, None
    try:
        funding, ready, cash_metadata = extended_funding.load_cash(
            events_path, coverage_path, universe
        )
        funding = [
            {**event, "rate": event["rate_decimal"], "mark_price": event["mark_decimal"]}
            for event in funding
        ]
        estimate = capacity_sources.memory_estimate(len(universe))
        estimate.update(
            curve_points=len(definitions) * ((study.END - study.REFITS[0]) // MINUTE + 1),
            assumed_bytes_per_curve_point=1024,
            other_working_reserve_bytes=1024**3,
        )
        estimate["working_set_estimate_bytes"] = (
            estimate["market_mapped_bytes"]
            + estimate["curve_points"] * estimate["assumed_bytes_per_curve_point"]
            + estimate["other_working_reserve_bytes"]
        )
        protocol = {
            "schema": "ta-capacity-account.v1",
            "created_ms": time.time_ns() // 1_000_000,
            "source": str(source),
            "cases": definitions,
            "start_ms": study.REFITS[0],
            "end_ms": study.END,
            "refit_times_ms": study.REFITS,
            "participation_rate": "0.10",
            "quote_window_minutes": 60,
            "formula": "rho*min(last_completed_quote,mean_60_contiguous_completed_quotes)",
            "source_sha256": frozen,
            "input_identities": identities,
            "prediction_columns_loaded": extended_funding.PREDICTION_COLUMNS,
            "historical_mark_prices_loaded": False,
            "historical_point_in_time_rules_proven": False,
            "min_notional_reference": "min(raw_open,simulated_fill)_proxy_not_historical_mark",
            "capacity_charge": "quantity*max(raw_open,simulated_fill)",
            "partial_exits": True,
            "entry_all_legs_preflight": True,
            "unresolved_end_inventory_invalidates_account": True,
            "future_labels_loaded": False,
            "orders_enabled": False,
            "leader_labels_read": False,
            "production_eligible": False,
            "independent_oos": False,
            "budget_arithmetic": "float64代理→Decimal(str(value))",
            "ledger_arithmetic": "Decimal60位；完整cycle才计胜负",
            "protocol_rules": signals.PROTOCOL,
            "memory_estimate": estimate,
            "python": sys.version,
            "libraries": {
                name: importlib.metadata.version(name) for name in ("numpy", "pandas", "pyarrow")
            },
        }
        save_json(output / "protocol.json", protocol)
        for index, path in enumerate(used):
            target = output / "source" / f"{index:02}_{path.name}"
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(path, target)
            if sha(target) != frozen[str(path)]:
                raise ValueError("执行快照副本与预检前摘要不同")
        save_json(output / "rules_coverage.json", rule_metadata)
        save_json(output / "rules.json", rules)
        save_json(output / "cash_coverage.json", cash_metadata)
        stage = "prediction_schedule"
        predictions = pd.read_parquet(
            prediction_path, columns=list(extended_funding.PREDICTION_COLUMNS)
        )
        schedules, report = extended_funding.build_schedules(predictions, universe)
        save_json(output / "decisions.json", report)
        del predictions, report
        gc.collect()
        stage, market_report = "capacity_accounts", {}
        if shutil.disk_usage(output).free < estimate["market_mapped_bytes"] + 1024**3:
            raise ValueError("磁盘不足以保存有界映射和账本")
        with tempfile.TemporaryDirectory(prefix=".capacity-market-", dir=output) as scratch:
            frames = capacity_sources.market_frames(
                universe, minutes, scratch, market_report, input_identities=identities
            )
            try:
                accounts, count, states, global_failure = drive(
                    definitions, schedules, frames, funding, rules, (*study.REFITS, study.END)
                )
            finally:
                frames.close()
        cash_failures(accounts, ready)
        save_json(output / "market_coverage.json", market_report)
        if global_failure:
            save_json(output / "global_failure.json", global_failure)
            raise ValueError("全局分钟来源失败，完整账户数值失效")
        stage = "verify_after"
        if any(sha(path) != expected for path, expected in (identities | frozen).items()):
            raise ValueError("实际消费来源或执行依赖在运行期间改变")
        save_json(output / "consumed_inputs.json", identities)
        stage = "metrics"
        metrics = {}
        for key, account in accounts.items():
            value = account.metrics()
            if value is not None:
                value["historical_blocks"] = block_summaries(
                    account, (*study.REFITS, study.END), states[key]
                )
                metrics[key] = value
        result = {
            "schema": "ta-capacity-account-result.v1",
            "frames": count,
            "cases": len(definitions),
            "metrics": metrics,
            "invalid_accounts": {
                key: account.invalid for key, account in accounts.items() if account.invalid
            },
            "case_execution_status": {
                key: "INVALID_ACCOUNT" if account.invalid else "COMPLETED_ACCOUNT"
                for key, account in accounts.items()
            },
            "economic_conditions_by_account": {
                key: search.gate(value) for key, value in metrics.items()
            },
            "global_failure": None,
            "protocol_sha256": sha(output / "protocol.json"),
            "future_labels_loaded": False,
            "orders_enabled": False,
            "leader_labels_read": False,
            "production_eligible": False,
            "stable_profitability_proven": False,
            "independent_oos": False,
        }
        save_json(output / "result.json", result)
    except Exception as error:
        failure = (
            {**global_failure, "stage": stage}
            if global_failure
            else {
                "stage": stage,
                "at": study.REFITS[0],
                "type": type(error).__name__,
                "reason": str(error),
                "numeric_conclusions_allowed": False,
            }
        )
        for account in accounts.values():
            if global_failure is not None:
                # drive已在真实中断帧保留最早失败，不抹掉可证明的部分成交前缀。
                continue
            prior = account.invalid
            at = (
                study.REFITS[0]
                if stage == "verify_after"
                else (
                    account.last_clock + MINUTE
                    if account.last_clock is not None
                    else study.REFITS[0]
                )
            )
            account.invalid = {
                **failure,
                "at": min(at, prior["at"]) if prior else at,
                "prior_execution_failure": prior,
            }
        save_json(output / "failure.json", failure)
        raise
    finally:
        save_ledgers(output, definitions, accounts, failure)
        save_json(
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
