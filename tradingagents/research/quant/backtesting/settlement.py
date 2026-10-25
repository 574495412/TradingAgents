"""按原生组合经理批准的结算后退出假说运行48条连续成本账户。"""

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from ..artifacts import read, save, sha, verify_receipts
from ..settlement_study import FEATURES, MODELS, predict
from .account import Account
from .forecast import PREDICTED_RETURN_THRESHOLD, cancel_incomplete_batch, shutil_copy
from .search import Audit, funding_source_path, gate, load_funding, save_account
from .signals import MINUTE, PROTOCOL, Signal

COSTS = (3, 10, 11)
METHODS = ("fixed", "rolling")


def decisions(frame, record, model):
    """只使用共同已知特征及事前训练参数，保留16bps多空配对门槛。"""
    if frame.symbol.duplicated().any():
        raise ValueError("同钟预测不能含重复合约")
    scores = predict(frame, record, model, FEATURES)
    selected = frame[["symbol"]].copy()
    selected["score"] = scores
    selected = selected[np.isfinite(selected.score)]
    positive = selected[selected.score >= PREDICTED_RETURN_THRESHOLD].sort_values(
        ["score", "symbol"], ascending=[False, True]
    )
    negative = selected[selected.score <= -PREDICTED_RETURN_THRESHOLD].sort_values(
        ["score", "symbol"]
    )
    size = min(2, len(positive), len(negative))
    return [
        (row.symbol, 1 if row.score > 0 else -1, abs(float(row.score)))
        for row in pd.concat([positive.head(size), negative.head(size)]).itertuples()
    ]


def blocks(account, boundaries, opportunities, states):
    """所有分块均来自同一权益曲线；收盘退出在本帧、开盘退出在下一帧落账。"""
    points = {row["clock"]: row["equity"] for row in account.curve}
    trades = {trade["cycle_id"]: trade for trade in account.trades}
    batches = {}
    for trade in account.trades:
        batches.setdefault(trade["episode"], []).append(trade)
    last, initial = boundaries[-1], PROTOCOL["initial_equity"]

    def fill_clock(fill):
        if fill["action"] == "CLOSE" and trades[fill["cycle_id"]]["reason"] in {
            "time_close",
            "scheduled_end",
        }:
            return fill["clock"]
        return fill["clock"] + MINUTE

    result = []
    for index, (begin, end) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
        closing_clock = last if end == last else end - MINUTE
        opening = initial if index == 0 else points[begin - MINUTE]

        def inside(clock, begin=begin, end=end):
            return begin <= clock and (clock < end or clock == end == last)

        fills = [fill for fill in account.fills if inside(fill_clock(fill))]
        row = {
            "start_ms": begin,
            "end_ms": end,
            "opening_equity": opening,
            "closing_equity": points[closing_clock],
            "equity_return": points[closing_clock] / opening - 1,
            "pnl_over_initial": (points[closing_clock] - opening) / initial,
            "fee_cash_outflow": sum(fill["fee"] for fill in fills),
            "funding_cash": sum(
                event["cash"]
                for event in account.funding_ledger
                if inside((event["time_ms"] // MINUTE + 1) * MINUTE)
            ),
            "opened_legs": sum(fill["action"] == "OPEN" for fill in fills),
            "closed_legs": sum(fill["action"] == "CLOSE" for fill in fills),
            "started_batches": sum(
                inside(min(trade["entry_ms"] for trade in group) + MINUTE)
                for group in batches.values()
            ),
            "completed_batches": sum(
                inside(
                    max(
                        trade["exit_ms"]
                        + (0 if trade["reason"] in {"time_close", "scheduled_end"} else MINUTE)
                        for trade in group
                    )
                )
                for group in batches.values()
            ),
            "opportunity_clocks": sum(inside(clock) for clock in opportunities),
            "decline_reasons": dict(
                Counter(row["reason"] for row in account.declines if inside(row["clock"] + MINUTE))
            ),
            "ending_state": states[closing_clock],
            "cashflow_recognition_clock": "account_curve_processing_minute",
        }
        result.append(row)
    return result


def run(source, market_panel, funding_root, output):
    """规则和全部案例先封存，逐分钟同时独立重建现金、库存、费用及资金费。"""
    source, market_panel, funding_root, output = (
        Path(path).resolve() for path in (source, market_panel, funding_root, output)
    )
    if any(
        output.is_relative_to(path) or path.is_relative_to(output)
        for path in (source, market_panel, funding_root)
    ):
        raise ValueError("新账户不能覆盖或包含研究来源")
    verify_receipts(source)
    parent = read(source / "protocol.json")
    coefficients = read(source / "coefficients.json")
    if (
        parent.get("account_family") != "settlement_unwind"
        or parent.get("hypothesis_id") != "phase3_settlement_unwind_01"
        or parent["orders_enabled"] is not False
        or parent["leader_labels_read"] is not False
        or tuple(parent["features"]) != tuple(FEATURES)
        or tuple(coefficients["features"]) != tuple(FEATURES)
        or set(coefficients["models"]) != set(MODELS)
    ):
        raise ValueError("只能运行已批准的完整结算退出研究家族")
    bounds = parent["phases"]["retrospective_validation"]
    start, end = pd.Timestamp(bounds["start"]), pd.Timestamp(bounds["end"])
    first, last = (int(value.value // 1_000_000) for value in (start, end))
    refits = sorted(int(value) for value in coefficients["refits"])
    if not refits or refits[0] != first or refits[-1] >= last:
        raise ValueError("训练快照没有覆盖原账户起点或越过终点")
    manifest, capture = (
        read(market_panel / "manifest.json"),
        read(funding_root / "capture_plan.json"),
    )
    if (
        Path(capture["panel"]).resolve() != market_panel
        or capture["manifest_sha256"] != sha(market_panel / "manifest.json")
        or manifest["contracts"] != parent["universe"]
        or manifest["market_panel_contract"]["timestamp_semantics"] != "CLOSE_TIME"
        or manifest["orders_enabled"] is not False
        or manifest["leader_labels_read"] is not False
        or pd.Timestamp(capture["start"]) > start
        or pd.Timestamp(capture["end"]) < end
    ):
        raise ValueError("分钟市场、人口或资金费覆盖与研究不同")
    identities = {
        str(source / "receipts.json"): sha(source / "receipts.json"),
        str(market_panel / "manifest.json"): sha(market_panel / "manifest.json"),
    }
    for name in ("capture_plan.json", "funding_manifest.json", "funding_events.json"):
        path = funding_root / name
        if parent["input_identities"].get(str(path)) != sha(path):
            raise ValueError("资金费现金与预测的冻结来源不同")
        identities[str(path)] = sha(path)
    funding_manifest = read(funding_root / "funding_manifest.json")
    identities.update(
        {
            str(funding_source_path(funding_root, leaf)): leaf["sha256"]
            for leaf in funding_manifest["leaves"]
        }
    )
    funding = load_funding(funding_root)
    # 回归标签只由研究模块读取；账户仅加载特征白名单。
    samples = pd.read_parquet(
        source / "hourly_samples.parquet", columns=["symbol", "asset_class", "clock_ms", *FEATURES]
    )
    samples = samples[(samples.clock_ms >= first) & (samples.clock_ms < last)]
    original_assets = {contract["asset_class"] for contract in parent["universe"].values()}
    if len(original_assets) != 2 or set(samples.asset_class.unique()) != original_assets:
        raise ValueError("连续账户必须完整保留原两个资产类别")
    cases = {
        f"{method}:{asset}:20m:{model}:{cost}bps": (method, asset, model, cost)
        for method in METHODS
        for asset in sorted(samples.asset_class.unique())
        for model in MODELS
        for cost in COSTS
    }
    package = Path(__file__).resolve().parents[3]
    paths = sorted(package.rglob("*.py")) + [
        Path(__file__).resolve().parents[1] / "plans/settlement_unwind_plan.md"
    ]
    frozen = {str(path): sha(path) for path in paths}
    output.mkdir(parents=True, exist_ok=False)
    save(
        output / "protocol.json",
        {
            "schema": "ta-settlement-unwind-account.v1",
            "created_ms": time.time_ns() // 1_000_000,
            "start_ms": first,
            "end_ms": last,
            "cases": cases,
            "source_sha256": frozen,
            "input_identities": identities,
            "predicted_return_threshold": PREDICTED_RETURN_THRESHOLD,
            "account_rules": {
                key: PROTOCOL[key]
                for key in (
                    "initial_equity",
                    "position_fraction",
                    "max_positions",
                    "decision_latency_ms",
                    "max_signal_age_ms",
                    "fee_bps",
                    "maximum_account_drawdown",
                    "minimum_trade_notional",
                )
            },
            "timed_exit_at_close": True,
            "holding_minutes": 20,
            "take_profit_fraction": None,
            "funding_entry_same_clock_goes_to_old_inventory": True,
            "funding_exit_same_clock_excluded_from_close_exit": True,
            "quantity_is_continuous": True,
            "all_or_none_available_open_batch": True,
            "real_time_forward_required": False,
            "independent_oos": False,
            "orders_enabled": False,
            "leader_labels_read": False,
            "production_eligible": False,
        },
    )
    for path in paths:
        shutil_copy(path, output / "source" / path.relative_to(package.parent))
    schedules = {}
    for (asset, clock), frame in samples.groupby(["asset_class", "clock_ms"], sort=True):
        for method in METHODS:
            refit = first if method == "fixed" else max(value for value in refits if value <= clock)
            record = coefficients["refits"][str(refit)]["groups"][f"{asset}:20m"]
            for model in MODELS:
                schedules[method, asset, model, int(clock)] = decisions(frame, record, model)
    save(
        output / "decisions.json",
        [
            {
                "method": method,
                "asset_class": asset,
                "model": model,
                "clock_ms": clock,
                "selected": selected,
            }
            for (method, asset, model, clock), selected in schedules.items()
        ],
    )
    accounts = {
        key: Account(model, 20, cost, funding, take_profit_fraction=None, timed_exit_at_close=True)
        for key, (_, _, model, cost) in cases.items()
    }
    audits = {key: Audit(account, funding) for key, account in accounts.items()}
    boundaries = [*refits, last]
    block_ends = {value - MINUTE for value in refits[1:]} | {last}
    states = {key: {} for key in cases}
    invalid, opportunities, previous, frames = {}, Counter(), None, 0
    for record in manifest["daily_files"]:
        if record["last_close_ms"] < first or record["first_close_ms"] > last:
            continue
        path = market_panel / record["path"]
        if sha(path) != record["sha256"]:
            raise ValueError("分钟原文身份与冻结不同")
        identities[str(path)] = record["sha256"]
        data = pd.read_parquet(
            path,
            columns=["open", "close", "eligible", "observed"],
            filters=[("bar_time", ">=", start), ("bar_time", "<=", end)],
        )
        for at, rows in data.groupby(level="bar_time", sort=True):
            clock = int(at.value // 1_000_000)
            if previous is not None and clock != previous + MINUTE:
                raise ValueError("完整账户时间轴有缺口或重复")
            current = rows.droplevel("bar_time")
            market = dict(
                zip(
                    current.index,
                    zip(
                        current.open, current.close, current.eligible, current.observed, strict=True
                    ),
                    strict=True,
                )
            )
            for key, account in accounts.items():
                if key in invalid:
                    continue
                method, asset, model, _ = cases[key]
                proposed = schedules.get((method, asset, model, clock), [])
                if proposed:
                    opportunities[key] += 1
                try:
                    cancel_incomplete_batch(account, market, clock - MINUTE)
                    expected = [
                        pending
                        for pending in account.pending.values()
                        if pending["available_ms"] <= clock - MINUTE
                    ]
                    prior_fills = len(account.fills)
                    account.open_phase(clock - MINUTE, market)
                    opened = [
                        fill for fill in account.fills[prior_fills:] if fill["action"] == "OPEN"
                    ]
                    if opened and (
                        len(opened) != len(expected)
                        or sum(fill["direction"] for fill in opened) != 0
                    ):
                        raise ValueError("同步多空批次出现部分成交")
                    pulses = (
                        []
                        if account.positions or account.pending
                        else [
                            Signal(symbol, direction, score, clock, f"settlement:{clock}")
                            for symbol, direction, score in proposed
                        ]
                    )
                    account.close_phase(clock, market, pulses, {}, final=clock == last)
                    audits[key].consume(clock, market, final=clock == last)
                    if clock in block_ends:
                        states[key][clock] = {
                            "halted": account.halted,
                            "open_positions": len(account.positions),
                            "pending_entries": len(account.pending),
                            "running_peak_equity": account.peak,
                            "running_maximum_drawdown": account.max_dd,
                        }
                except ValueError as error:
                    invalid[key] = {
                        "at": clock,
                        "reason": str(error),
                        "numeric_conclusions_allowed": False,
                    }
            previous, frames = clock, frames + 1
        print(
            {"market_day": path.name, "frames": frames, "invalid_accounts": len(invalid)},
            flush=True,
        )
    if previous != last or frames != (last - first) // MINUTE + 1:
        raise ValueError("账户没有覆盖完整预定区间")
    metrics, proofs = {}, {}
    for key, account in accounts.items():
        if key not in invalid:
            proofs[key] = audits[key].finish()
            summary = metrics[key] = account.metrics()
            batch_pnl = {}
            for trade in account.trades:
                batch_pnl[trade["episode"]] = (
                    batch_pnl.get(trade["episode"], 0.0) + trade["net_pnl"]
                )
            summary["complete_batches"] = len(batch_pnl)
            summary["positive_batches"] = sum(value > 0 for value in batch_pnl.values())
            method, asset, model, _ = cases[key]
            clocks = [
                clock
                for (m, a, model_name, clock), selected in schedules.items()
                if (m, a, model_name) == (method, asset, model) and selected
            ]
            summary["historical_blocks"] = blocks(account, boundaries, clocks, states[key])
        save_account(output / (key.replace(":", "_") + ".json.gz"), account)
    verify_receipts(source)
    if any(sha(path) != value for path, value in (frozen | identities).items()):
        raise ValueError("账户计算期间输入或源码变化")
    result = {
        "schema": "ta-settlement-unwind-account-result.v1",
        "frames": frames,
        "cases": len(cases),
        "metrics": metrics,
        "invalid_accounts": invalid,
        "financial_proofs": proofs,
        "opportunity_clocks": dict(opportunities),
        "economic_conditions_by_account": {key: gate(value) for key, value in metrics.items()},
        "direction_and_predictive_increment_require_separate_assessment": True,
        "real_time_forward_required": False,
        "independent_oos": False,
        "stable_profitability_proven": False,
        "orders_enabled": False,
        "production_eligible": False,
    }
    save(output / "input_identities.json", identities)
    save(output / "result.json", result)
    save(
        output / "receipts.json",
        {str(path.relative_to(output)): sha(path) for path in output.rglob("*") if path.is_file()},
    )
    return result


def main(argv=None):
    """账户独立于模型执行，仅消费已经冻结的回归结果。"""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "market-panel", "funding-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.source, args.market_panel, args.funding_root, args.output)
    print(
        {
            "cases": result["cases"],
            "valid_accounts": len(result["metrics"]),
            "production_eligible": False,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
