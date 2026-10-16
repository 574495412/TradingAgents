"""对固定训练预测执行成本门槛与连续账户检验，不按后段结果选择模型。"""

import argparse
import importlib.metadata
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from ..artifacts import read, save, sha, verify_receipts
from ..factors.funding import NAMES as FUNDING_NAMES
from ..factors.response import NAMES
from .account import Account
from .search import Audit, funding_source_path, load_funding, save_account
from .signals import MINUTE, PROTOCOL, Signal

TESTED_MODELS = ("legacy",) + tuple(f"add_{f}" for f in NAMES)
FUNDING_MODELS = ("legacy",) + tuple(f"add_{f}" for f in FUNDING_NAMES) + ("expanded",)
PREDICTED_RETURN_THRESHOLD = 0.0016


def decisions(frame, coefficients, features):
    """只看特征与固定训练斜率，未来标签列不参与读取、排名或筛选。"""
    known = frame[np.isfinite(frame[list(features)]).all(axis=1)].sort_values("symbol")
    if len(known) < 20 or coefficients is None:
        return []
    ranks = (known[list(features)].rank(method="average") - 1) / (len(known) - 1) - 0.5
    scores = sum(ranks[f].to_numpy() * b for f, b in coefficients.items())
    selected = known[["symbol"]].copy()
    selected["score"] = scores
    positive = selected[selected.score >= PREDICTED_RETURN_THRESHOLD].sort_values(
        ["score", "symbol"], ascending=[False, True]
    )
    negative = selected[selected.score <= -PREDICTED_RETURN_THRESHOLD].sort_values(
        ["score", "symbol"]
    )
    size = min(2, len(positive), len(negative))
    return [
        (r.symbol, 1 if r.score > 0 else -1, abs(float(r.score)))
        for r in pd.concat([positive.head(size), negative.head(size)]).itertuples()
    ]


def cancel_incomplete_batch(account, market, stamp):
    """一组同步入场有成员缺少有效开盘时全部取消，不让缺失腿变成单边选择。"""
    due = {s: p for s, p in account.pending.items() if p["available_ms"] <= stamp}
    if not due:
        return
    if any(
        s not in market
        or not market[s][2]
        or not market[s][3]
        or not math.isfinite(market[s][0])
        or market[s][0] <= 0
        for s in due
    ):
        for symbol in due:
            account.pending.pop(symbol)
            account.declines.append(
                {"clock": stamp, "symbol": symbol, "reason": "paired_batch_open_unavailable"}
            )


def run(source, market_panel, funding_root, output, *, extra_slippage_bps=None):
    """账户规则先封存，逐分钟共用本金与容量，并同步执行独立财务重建。"""
    source, market_panel, funding_root, output = (
        Path(p).resolve() for p in (source, market_panel, funding_root, output)
    )
    if extra_slippage_bps not in {None, 11}:
        raise ValueError("额外压力只允许预先声明的11bps，不允许降低原成本")
    costs = (3, 10) if extra_slippage_bps is None else (3, 10, 11)
    if any(
        output.is_relative_to(p) or p.is_relative_to(output)
        for p in (source, market_panel, funding_root)
    ):
        raise ValueError("账户输出不能覆盖或包含研究来源")
    verify_receipts(source)
    parent, result = read(source / "protocol.json"), read(source / "result.json")
    if any(parent[k] is not False for k in ("orders_enabled", "leader_labels_read")):
        raise ValueError("只允许公开量化研究结果")
    family = parent.get("account_family", "market_response")
    if family not in {"market_response", "settled_funding"}:
        raise ValueError("账户家族不属于预先声明的研究计划")
    tested_models = FUNDING_MODELS if family == "settled_funding" else TESTED_MODELS
    plan = "settled_funding_plan.md" if family == "settled_funding" else "forecast_account_plan.md"
    if family == "settled_funding":
        costs = (3, 10, 11)
        for name in ("capture_plan.json", "funding_manifest.json", "funding_events.json"):
            path = funding_root / name
            if parent.get("input_identities", {}).get(str(path)) != sha(path):
                raise ValueError("现金结算来源与资金费因子冻结来源不同")
    bounds = parent["phases"]["retrospective_validation"]
    start, end = pd.Timestamp(bounds["start"]), pd.Timestamp(bounds["end"])
    first, last = int(start.value // 1_000_000), int(end.value // 1_000_000)
    manifest, capture = (
        read(market_panel / "manifest.json"),
        read(funding_root / "capture_plan.json"),
    )
    if (
        Path(capture["panel"]).resolve() != market_panel
        or capture["manifest_sha256"] != sha(market_panel / "manifest.json")
        or manifest["market_panel_contract"]["timestamp_semantics"] != "CLOSE_TIME"
        or manifest["orders_enabled"] is not False
        or manifest["leader_labels_read"] is not False
        or manifest["contracts"] != parent["universe"]
        or pd.Timestamp(capture["start"]) > start
        or pd.Timestamp(capture["end"]) < end
    ):
        raise ValueError("市场面板、固定成员、收盘时钟或资金费来源不同")
    funding = load_funding(funding_root)
    funding_manifest = read(funding_root / "funding_manifest.json")
    # 只读取特征列，账户生成路径不加载本轮保存的未来标签。
    features = parent["features"]
    samples = pd.read_parquet(
        source / "hourly_samples.parquet", columns=["symbol", "asset_class", "clock_ms", *features]
    )
    samples = samples[(samples.clock_ms >= first) & (samples.clock_ms < last)]
    schedules = {}
    for (asset, clock), frame in samples.groupby(["asset_class", "clock_ms"], sort=True):
        for horizon in parent["horizons_minutes"]:
            group = result["groups"][f"{asset}:{horizon}m"]
            for model in tested_models:
                schedules[asset, horizon, model, int(clock)] = decisions(
                    frame, group["frozen_training_coefficients"][model], features
                )
    cases, identities = {}, {}
    for asset in sorted(samples.asset_class.unique()):
        for horizon in parent["horizons_minutes"]:
            for model in tested_models:
                for cost in costs:
                    key = f"{asset}:{horizon}m:{model}:{cost}bps"
                    cases[key] = (asset, horizon, model, cost)
    quant_root = Path(__file__).resolve().parents[1]
    paths = [
        Path(__file__).resolve(),
        Path(Account.__init__.__code__.co_filename).resolve(),
        Path(Audit.consume.__code__.co_filename).resolve(),
        Path(__file__).with_name("signals.py"),
        Path(sha.__code__.co_filename).resolve(),
        quant_root / "plans" / plan,
    ]
    # 保存完整包结构，让封存源码可在独立路径重新运行，不依赖当前工作区。
    package = quant_root.parents[1]
    paths = sorted(set(paths) | set(package.rglob("*.py")))
    if extra_slippage_bps is not None and family == "market_response":
        paths.append(quant_root / "plans/forecast_adverse_plan.md")
    frozen = {str(p): sha(p) for p in paths}
    identities.update(
        {
            str(source / "receipts.json"): sha(source / "receipts.json"),
            str(market_panel / "manifest.json"): sha(market_panel / "manifest.json"),
            str(funding_root / "funding_manifest.json"): sha(
                funding_root / "funding_manifest.json"
            ),
            str(funding_root / "funding_events.json"): sha(funding_root / "funding_events.json"),
            str(funding_root / "capture_plan.json"): sha(funding_root / "capture_plan.json"),
        }
    )
    identities.update(
        {
            str(funding_source_path(funding_root, leaf)): leaf["sha256"]
            for leaf in funding_manifest["leaves"]
        }
    )
    output.mkdir(parents=True, exist_ok=False)
    save(
        output / "protocol.json",
        {
            "schema": "ta-frozen-forecast-account.v1",
            "created_ms": time.time_ns() // 1_000_000,
            "start_ms": first,
            "end_ms": last,
            "cases": cases,
            "account_rules": {
                k: PROTOCOL[k]
                for k in (
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
            "case_slippage_bps": costs,
            "account_family": family,
            "case_models": tested_models,
            "posthoc_adverse_cost_check": extra_slippage_bps is not None
            and family == "market_response",
            "source_layout": "complete_python_package",
            "predicted_return_threshold": PREDICTED_RETURN_THRESHOLD,
            "maximum_pairs": 2,
            "entry_only_when_flat": True,
            "all_or_none_available_open_batch": True,
            "take_profit_fraction": None,
            "reentry": False,
            "source_sha256": frozen,
            "input_identities": identities,
            "parent_statistics_previously_viewed": True,
            "account_rules_frozen_before_account_outcomes": True,
            "independent_oos": False,
            "quantity_is_continuous": True,
            "historical_quote_arrival_times_observed": False,
            "runtime": {
                "python": sys.version,
                "libraries": {
                    name: importlib.metadata.version(name)
                    for name in ("numpy", "pandas", "pyarrow")
                },
            },
            "orders_enabled": False,
            "leader_labels_read": False,
            "production_eligible": False,
        },
    )
    save(
        output / "decisions.json",
        [
            {"asset_class": a, "horizon_minutes": h, "model": m, "clock_ms": t, "selected": value}
            for (a, h, m, t), value in schedules.items()
        ],
    )
    for p in paths:
        target = output / "source" / p.relative_to(package.parent)
        shutil_copy(p, target)
    accounts = {
        key: Account(model, horizon, cost, funding, take_profit_fraction=None)
        for key, (_, horizon, model, cost) in cases.items()
    }
    audits = {key: Audit(account, funding) for key, account in accounts.items()}
    invalid, opportunity_counts, previous, frames = {}, Counter(), None, 0
    for record in manifest["daily_files"]:
        if record["last_close_ms"] < first or record["first_close_ms"] > last:
            continue
        path = market_panel / record["path"]
        if sha(path) != record["sha256"]:
            raise ValueError("分钟市场来源与封存身份不同")
        identities[str(path)] = record["sha256"]
        data = pd.read_parquet(
            path,
            columns=["open", "close", "eligible", "observed"],
            filters=[("bar_time", ">=", start), ("bar_time", "<=", end)],
        )
        for at, rows in data.groupby(level="bar_time", sort=True):
            clock = int(at.value // 1_000_000)
            if previous is not None and clock != previous + MINUTE:
                raise ValueError("连续账户时间轴存在缺口或重复")
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
                asset, horizon, model, _ = cases[key]
                proposed = schedules.get((asset, horizon, model, clock), [])
                if proposed:
                    opportunity_counts[key] += 1
                pulses = [
                    Signal(s, d, score, clock, f"forecast:{clock}") for s, d, score in proposed
                ]
                try:
                    cancel_incomplete_batch(account, market, clock - MINUTE)
                    expected = [
                        p for p in account.pending.values() if p["available_ms"] <= clock - MINUTE
                    ]
                    prior_fills = len(account.fills)
                    account.open_phase(clock - MINUTE, market)
                    opened = [f for f in account.fills[prior_fills:] if f["action"] == "OPEN"]
                    if opened and (
                        len(opened) != len(expected) or sum(f["direction"] for f in opened) != 0
                    ):
                        raise ValueError("同步多空批次只成交了部分成员")
                    if account.positions or account.pending:
                        pulses = []
                    account.close_phase(clock, market, pulses, {}, final=clock == last)
                    audits[key].consume(clock, market, final=clock == last)
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
        raise ValueError("账户未覆盖原定完整区间")
    summaries, financial_proofs = {}, {}
    for key, account in accounts.items():
        if key not in invalid:
            financial_proofs[key] = audits[key].finish()
            summaries[key] = account.metrics()
            batches = {}
            for trade in account.trades:
                batches[trade["episode"]] = batches.get(trade["episode"], 0.0) + trade["net_pnl"]
            summaries[key]["complete_batches"] = len(batches)
            summaries[key]["positive_batches"] = sum(value > 0 for value in batches.values())
            summaries[key]["batch_win_rate"] = (
                sum(value > 0 for value in batches.values()) / len(batches) if batches else None
            )
        save_account(output / f"{key.replace(':', '_')}.json.gz", account)
    verify_receipts(source)
    if any(sha(p) != v for p, v in (frozen | identities).items()):
        raise ValueError("连续账户计算期间源码或输入改变")
    save(output / "input_identities.json", identities)
    outcome = {
        "schema": "ta-frozen-forecast-account-result.v1",
        "frames": frames,
        "cases": len(cases),
        "metrics": summaries,
        "invalid_accounts": invalid,
        "opportunity_clocks": dict(opportunity_counts),
        "financial_proofs": financial_proofs,
        "independent_oos": False,
        "orders_enabled": False,
        "stable_profitability_proven": False,
        "production_eligible": False,
    }
    save(output / "result.json", outcome)
    save(
        output / "receipts.json",
        {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()},
    )
    return outcome


def shutil_copy(source, target):
    """以排他创建保存源码，不能覆盖其他研究已有文件。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(source.read_bytes())


def main(argv=None):
    """同时支持统一量化子命令和独立模块调用。"""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "market-panel", "funding-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--extra-slippage-bps", type=int, choices=[11])
    args = parser.parse_args(argv)
    result = run(
        args.source,
        args.market_panel,
        args.funding_root,
        args.output,
        extra_slippage_bps=args.extra_slippage_bps,
    )
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
