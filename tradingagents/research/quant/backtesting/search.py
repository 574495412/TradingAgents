"""固定因子候选的连续账户搜索，包含真实资金费及前后阶段成本检验。"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

import pandas as pd

from tradingagents.research.quant.backtesting.account import Account
from tradingagents.research.quant.backtesting.signals import MINUTE, NAMES, PROTOCOL, Signals
from tradingagents.research.quant.paths import source_path
from tradingagents.research.quant.strategy_audit import PublicPositioning, read, save, sha

COLUMNS = [
    "open",
    "close",
    "eligible",
    "observed",
    "asset_class",
    "ret_3_pct",
    "ret_20_pct",
    "ret_60_pct",
    "ret_20_rank_pct",
    "ret_60_rank_pct",
    "ret_20_residual_pct",
    "taker_imbalance_3",
    "taker_imbalance_12",
    "atr_rank_pct",
    "atr_pct",
    "volume_ratio_rank_pct",
    "liquidity_rank_pct",
    "return_volatility_60_pct",
]


def case_name(factor, hold, cost, reentry):
    return f"{factor}_{hold}m_{'reentry' if reentry else 'seed'}_{cost}bps"


def load_funding(public):
    """使用已覆盖整段请求区间的原始资金费，不假定固定结算频率。"""
    manifest = read(public / "funding_manifest.json")
    if manifest["api_time_range_coverage_verified"] is not True:
        raise ValueError("资金费采集区间未覆盖")
    original, funding = {}, []
    for part in manifest["leaves"]:
        path = Path(part["path"])
        if sha(path) != part["sha256"]:
            raise ValueError("原始资金费响应被修改")
        for row in read(path):
            key = row["symbol"], int(row["fundingTime"])
            if key in original:
                raise ValueError("资金费事件重复")
            original[key] = row
            rate, mark = float(row["fundingRate"]), float(row["markPrice"])
            if not math.isfinite(rate) or not math.isfinite(mark) or abs(rate) > 1 or mark <= 0:
                raise ValueError("资金费数值不合法")
            funding.append(
                {
                    "symbol": key[0],
                    "time_ms": key[1],
                    "rate": rate,
                    "mark_price": mark,
                    "source_sha256": part["sha256"],
                }
            )
    public_rows = {
        (r["symbol"], int(r["fundingTime"])): r for r in read(public / "funding_events.json")
    }
    if public_rows != original:
        raise ValueError("便利资金费表不能重建原始响应")
    return sorted(funding, key=lambda e: (e["time_ms"], e["symbol"]))


def gate(metrics):
    rules = PROTOCOL["gate"]
    return (
        metrics["net_return"] > 0
        and metrics["maximum_close_drawdown"] < PROTOCOL["maximum_account_drawdown"]
        and metrics["closed_trades"] >= rules["minimum_trades_per_phase"]
        and metrics["wilson_lower"] >= rules["minimum_wilson_lower"]
        and (
            metrics["profit_factor"] is None
            or metrics["profit_factor"] >= rules["minimum_profit_factor"]
        )
    )


def select_candidate(training, definitions, *, selectable_names=None):
    """只有事前指定的主候选有晋级资格，控制组盈利不能替代主假说。"""
    candidates = []
    for factor, hold, _, reentry in definitions:
        if selectable_names is not None and factor not in selectable_names:
            continue
        base, stress = (case_name(factor, hold, cost, reentry) for cost in PROTOCOL["slippage_bps"])
        if (
            base in training
            and stress in training
            and gate(training[base])
            and gate(training[stress])
        ):
            candidates.append(
                (
                    training[base]["net_return"] - 0.75 * training[base]["maximum_close_drawdown"],
                    base,
                    stress,
                )
            )
    return max(candidates) if candidates else None


def save_account(path, account):
    """压缩保存完整现金、成交、资金费和含浮亏权益，而不是只留标题指标。"""
    data = {
        "factor": account.factor,
        "hold_ms": account.hold,
        "slippage_fraction": account.slip,
        "reentry": account.reentry,
        "trades": account.trades,
        "fills": account.fills,
        "funding": account.funding_ledger,
        "curve": account.curve,
        "declines": account.declines,
        "metrics": account.metrics(),
    }
    with gzip.open(path, "xt", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


class Audit:
    """从公开成交价和原始资金费重新记账，不调用账户的盈亏或结算方法。"""

    def __init__(self, account, funding):
        self.account, self.source = account, funding
        self.cash = self.peak = PROTOCOL["initial_equity"]
        self.inventory, self.trade_map, self.receipt_used = {}, {}, set()
        self.fill_cursor = self.funding_cursor = self.curve_cursor = 0
        self.settlements, self.closed, self.dd = [], [], 0.0

    def consume(self, clock, market, *, final=False):
        fills, source = self.account.fills, self.source
        while True:
            fill = fills[self.fill_cursor] if self.fill_cursor < len(fills) else None
            fund = source[self.funding_cursor] if self.funding_cursor < len(source) else None
            fill_ready = fill and (fill["clock"] < clock or (final and fill["clock"] == clock))
            fund_ready = fund and fund["time_ms"] < clock
            if not fill_ready and not fund_ready:
                break
            if fund_ready and (not fill_ready or fund["time_ms"] <= fill["clock"]):
                p = self.inventory.get(fund["symbol"])
                if p:
                    cash = -p["direction"] * p["quantity"] * fund["mark_price"] * fund["rate"]
                    self.cash += cash
                    p["funding"] += cash
                    self.settlements.append(
                        {
                            **fund,
                            "cycle_id": p["cycle_id"],
                            "direction": p["direction"],
                            "quantity": p["quantity"],
                            "cash": cash,
                        }
                    )
                self.funding_cursor += 1
                continue
            symbol, direction, quantity = fill["symbol"], fill["direction"], fill["quantity"]
            if fill["clock"] not in (clock - MINUTE, clock):
                raise ValueError("成交不能匹配真实分钟开盘或预定期末")
            raw = market[symbol][1 if fill["clock"] == clock else 0]
            price = raw * (
                1 + direction * self.account.slip * (1 if fill["action"] == "OPEN" else -1)
            )
            fee = quantity * price * PROTOCOL["fee_bps"] / 10000
            if not math.isclose(price, fill["price"], abs_tol=1e-9) or not math.isclose(
                fee, fill["fee"], abs_tol=1e-9
            ):
                raise ValueError("模拟成交价或手续费不符合原始行情和固定成本")
            if fill["action"] == "OPEN":
                if (
                    symbol in self.inventory
                    or fill["clock"] < fill["signal_ms"] + PROTOCOL["decision_latency_ms"]
                ):
                    raise ValueError("重复库存或信号完成之前成交")
                parent = fill["receipt_from"]
                if parent:
                    prior = self.trade_map.get(parent)
                    if (
                        parent in self.receipt_used
                        or not prior
                        or prior["net_pnl"] <= 0
                        or prior["exit_ms"] >= fill["signal_ms"]
                        or prior["symbol"] != symbol
                        or prior["direction"] != direction
                    ):
                        raise ValueError("续开权限不来自未消费的真实净盈利平仓")
                    self.receipt_used.add(parent)
                eq = self.cash + sum(
                    p["direction"] * p["quantity"] * (market[s][0] - p["price"])
                    for s, p in self.inventory.items()
                )
                if price * quantity > eq * PROTOCOL["position_fraction"] + 1e-7:
                    raise ValueError("开仓超过权益仓位预算")
                self.cash -= fee
                self.inventory[symbol] = {
                    "cycle_id": fill["cycle_id"],
                    "quantity": quantity,
                    "direction": direction,
                    "price": price,
                    "entry_fee": fee,
                    "funding": 0.0,
                }
                gross = sum(p["quantity"] * market[s][0] for s, p in self.inventory.items())
                after = self.cash + sum(
                    p["direction"] * p["quantity"] * (market[s][0] - p["price"])
                    for s, p in self.inventory.items()
                )
                if gross > after + 1e-7 or len(self.inventory) > PROTOCOL["max_positions"]:
                    raise ValueError("实际成交超过共享容量或一倍净值")
            else:
                p = self.inventory.pop(symbol)
                if quantity != p["quantity"] or p["cycle_id"] != fill["cycle_id"]:
                    raise ValueError("平仓不属于真实库存")
                gross = direction * quantity * (price - p["price"])
                net = gross - p["entry_fee"] - fee + p["funding"]
                self.cash += gross - fee
                trade = self.account.trades[len(self.closed)]
                if trade["cycle_id"] != p["cycle_id"]:
                    raise ValueError("逐笔结果与真实平仓顺序不同")
                if not math.isclose(net, trade["net_pnl"], abs_tol=1e-7):
                    raise ValueError("逐笔净收益漏计成本或资金费")
                self.trade_map[p["cycle_id"]] = trade
                self.closed.append(net)
            self.fill_cursor += 1
        curve = self.account.curve[self.curve_cursor]
        eq = self.cash + sum(
            p["direction"] * p["quantity"] * (market[s][1] - p["price"])
            for s, p in self.inventory.items()
        )
        if (
            curve["clock"] != clock
            or not math.isclose(eq, curve["equity"], abs_tol=1e-7)
            or not math.isclose(self.cash, curve["cash"], abs_tol=1e-7)
        ):
            raise ValueError("独立现金及持仓不能重建每个分钟收盘权益")
        self.peak = max(self.peak, eq)
        self.dd = max(self.dd, 1 - eq / self.peak)
        self.curve_cursor += 1

    def finish(self):
        if (
            self.inventory
            or self.fill_cursor != len(self.account.fills)
            or self.settlements != self.account.funding_ledger
            or len(self.closed) != len(self.account.trades)
        ):
            raise ValueError("完整账户或资金费账本不一致")
        metrics = self.account.metrics()
        if not math.isclose(
            self.cash / PROTOCOL["initial_equity"] - 1, metrics["net_return"], abs_tol=1e-9
        ) or not math.isclose(self.dd, metrics["maximum_close_drawdown"], abs_tol=1e-9):
            raise ValueError("收益或最大回撤未通过独立复核")
        wins = sum(v > 0 for v in self.closed)
        expected_win = wins / len(self.closed) if self.closed else None
        profit, loss = sum(max(v, 0) for v in self.closed), -sum(min(v, 0) for v in self.closed)
        expected_pf = profit / loss if loss else None
        if (
            metrics["closed_trades"] != len(self.closed)
            or metrics["win_rate"] != expected_win
            or (metrics["profit_factor"] is None) != (expected_pf is None)
            or (
                expected_pf is not None
                and not math.isclose(metrics["profit_factor"], expected_pf, abs_tol=1e-9)
            )
        ):
            raise ValueError("完整净胜率或利润因子未通过独立重建")
        return {
            "passed": True,
            "cash": self.cash,
            "fills": self.fill_cursor,
            "settlements": len(self.settlements),
            "curve_points": self.curve_cursor,
            "profit_receipts_used": len(self.receipt_used),
        }


def run(
    args,
    *,
    names=NAMES,
    signals_factory=Signals,
    study_protocol=None,
    extra_sources=(),
    account_factory=Account,
    reentry_names=None,
    hold_minutes=None,
    selectable_names=None,
):
    study = PROTOCOL | (study_protocol or {})
    panel, public, output = args.panel.resolve(), args.public.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest, capture = read(panel / "manifest.json"), read(public / "capture_plan.json")
    if (
        capture["panel"] != str(panel)
        or capture["manifest_sha256"] != sha(panel / "manifest.json")
        or capture["start"] != PROTOCOL["start"]
        or capture["end"] != PROTOCOL["end"]
    ):
        raise ValueError("公开采集计划与本轮市场或时间范围不一致")
    if manifest["market_panel_contract"]["timestamp_semantics"] != "CLOSE_TIME":
        raise ValueError("必须显式使用收盘时间面板")
    start, cut, end = (pd.Timestamp(PROTOCOL[k]) for k in ("start", "training_end", "end"))
    start_ms, cut_ms, end_ms = (int(t.value // 1_000_000) for t in (start, cut, end))
    parts = []
    for record in manifest["daily_files"]:
        if record["last_close_ms"] < start_ms - MINUTE or record["first_close_ms"] > end_ms:
            continue
        path = panel / record["path"]
        if sha(path) != record["sha256"]:
            raise ValueError("市场特征与原始清单不一致")
        parts.append(path)
    metrics_manifest = read(public / "metrics_manifest.json")
    verified = {}
    for record in metrics_manifest["records"]:
        if record["status"] == "VERIFIED":
            if sha(record["path"]) != record["sha256"]:
                raise ValueError("持仓公开输入发生变化")
            verified[record["path"]] = record["sha256"]
    lookup = PublicPositioning(verified)
    funding = load_funding(public)
    definitions = [
        (factor, hold, cost, reentry)
        for factor in names
        for hold in (PROTOCOL["hold_minutes"] if hold_minutes is None else hold_minutes)
        for cost in PROTOCOL["slippage_bps"]
        for reentry in (
            [False, True]
            if factor in (PROTOCOL["reentry_variants"] if reentry_names is None else reentry_names)
            else [False]
        )
    ]
    accounts = {
        case_name(*d): account_factory(d[0], d[1], d[2], funding, reentry=d[3]) for d in definitions
    }
    source_paths = [
        Path(__file__),
        source_path("profit_factor_account.py"),
        source_path("profit_factor_signals.py"),
        source_path("strategy_factor_audit.py"),
    ]
    source_paths.extend(Path(p) for p in extra_sources)
    save(
        output / "protocol.json",
        study
        | {
            "cases": list(accounts),
            "selectable_names": list(selectable_names)
            if selectable_names is not None
            else list(names),
            "source_files": {str(p): sha(p) for p in source_paths},
            "public_panel": {str(p): sha(p) for p in parts},
            "capture_plan_sha256": sha(public / "capture_plan.json"),
            "metrics_manifest_sha256": sha(public / "metrics_manifest.json"),
            "funding_events_sha256": sha(public / "funding_events.json"),
            "funding_manifest_sha256": sha(public / "funding_manifest.json"),
            "official_metrics_missing_files": sum(
                r["status"] != "VERIFIED" for r in metrics_manifest["records"]
            ),
            "historic_timestamp_text_corrected": True,
        },
    )
    signaler, invalid, training, selection, previous = (
        signals_factory(lookup.at),
        {},
        {},
        None,
        None,
    )
    for path in parts:
        frame = pd.read_parquet(
            path,
            columns=COLUMNS,
            filters=[("bar_time", ">=", start - pd.Timedelta(minutes=1)), ("bar_time", "<=", end)],
        )
        for at, rows in frame.groupby(level="bar_time", sort=True):
            clock = int(at.value // 1_000_000)
            if previous is not None and clock != previous + MINUTE:
                raise ValueError("研究时间轴不能跳过分钟")
            current = rows.droplevel("bar_time")
            market = dict(
                zip(
                    current.index,
                    zip(
                        current["open"],
                        current["close"],
                        current["eligible"],
                        current["observed"],
                        strict=True,
                    ),
                    strict=True,
                )
            )
            pulses, qualified = signaler.observe(clock, current)
            if clock >= start_ms:
                for name, account in accounts.items():
                    if name in invalid:
                        continue
                    try:
                        account.open_phase(clock - MINUTE, market)
                        account.close_phase(
                            clock,
                            market,
                            pulses[account.factor],
                            qualified[account.factor],
                            final=clock == end_ms,
                        )
                    except ValueError as exc:
                        invalid[name] = {
                            "at": clock,
                            "reason": str(exc),
                            "numeric_conclusions_allowed": False,
                        }
                if clock == cut_ms:
                    training = {
                        name: a.metrics() for name, a in accounts.items() if name not in invalid
                    }
                    save(output / "training.json", training)
                    selection = select_candidate(
                        training, definitions, selectable_names=selectable_names
                    )
                    save(
                        output / "selection.json",
                        {
                            "selected_base": selection[1] if selection else None,
                            "selected_stress": selection[2] if selection else None,
                            "training_sha256": sha(output / "training.json"),
                            "frozen_before_validation_outcomes": True,
                        },
                    )
                if at.hour == 0 and at.minute == 0:
                    print(
                        f"连续账户：{at.isoformat()}，有效 {len(accounts) - len(invalid)}/{len(accounts)}。",
                        flush=True,
                    )
            previous = clock
    if previous != end_ms:
        raise ValueError("没有到达固定研究终点")
    summaries = {
        name: {"full": a.metrics(), "validation": a.metrics(after=cut_ms)}
        for name, a in accounts.items()
        if name not in invalid
    }
    for name, account in accounts.items():
        if name not in invalid:
            save_account(output / f"{name}.json.gz", account)
    print("开始独立重建成交、资金费与每分钟权益。", flush=True)
    auditors = {name: Audit(a, funding) for name, a in accounts.items() if name not in invalid}
    for path in parts:
        frame = pd.read_parquet(
            path,
            columns=["open", "close", "eligible", "observed"],
            filters=[("bar_time", ">=", start), ("bar_time", "<=", end)],
        )
        for at, rows in frame.groupby(level="bar_time", sort=True):
            clock, current = int(at.value // 1_000_000), rows.droplevel("bar_time")
            market = dict(
                zip(
                    current.index,
                    zip(
                        current["open"],
                        current["close"],
                        current["eligible"],
                        current["observed"],
                        strict=True,
                    ),
                    strict=True,
                )
            )
            for auditor in auditors.values():
                auditor.consume(clock, market, final=clock == end_ms)
    audit = {name: a.finish() for name, a in auditors.items()}
    passed = bool(
        selection
        and selection[1] in summaries
        and selection[2] in summaries
        and gate(summaries[selection[1]]["validation"])
        and gate(summaries[selection[2]]["validation"])
    )
    freeze = read(output / "protocol.json")
    if any(
        sha(path) != checksum
        for path, checksum in (freeze["source_files"] | freeze["public_panel"]).items()
    ):
        raise ValueError("研究期间源码或行情发生变化")
    save(
        output / "result.json",
        {
            "schema": study["schema"],
            "accounts": summaries,
            "training": training,
            "selection": selection,
            "selected_retrospective_gate_passed": passed,
            "invalid_accounts": invalid,
            "audit": audit,
            "missing_positioning_observations": signaler.missing_positioning,
            "missing_open_interest_observations": getattr(signaler, "missing_interest", None),
            "study_diagnostics": getattr(signaler, "diagnostics", {}),
            "funding_included": True,
            "independent_oos": False,
            "production_eligible": False,
            "leader_labels_read": False,
            "orders_enabled": False,
        },
    )
    save(output / "receipts.json", {p.name: sha(p) for p in output.iterdir() if p.is_file()})
    print(f"研究完成，训练冻结候选后段通过：{passed}。", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
