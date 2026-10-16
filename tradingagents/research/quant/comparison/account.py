"""四个实验组共用的连续账户模型，完全独立于生产状态。

在决策完成后的下一可用开盘成交；资金费按事件真实时刻和标记价结算。
使用固定持仓期限和收盘风险控制，不假定盘中止损已成交。研究数量连续，
尚未验证历史交易所手数限制，因此结果不得作为生产订单参数。
"""

import math
from copy import deepcopy
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Protocol:
    initial_equity: float = 1000.0
    fee_bps: float = 5.0
    slippage_bps: float = 3.0
    position_fraction: float = 0.25
    max_positions: int = 4
    max_hold_bars: int = 4
    drawdown_stop: float = 0.10
    signal_threshold: float = 0.10
    event_weight: float = 0.50
    decision_latency_ms: int = 300000

    def __post_init__(self):
        for value in asdict(self).values():
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
            ):
                raise ValueError("实验协议必须使用有限数值")
        if not (
            self.initial_equity > 0
            and 0 <= self.fee_bps < 1000
            and 0 <= self.slippage_bps < 1000
            and 0 < self.position_fraction <= 1
            and 1 <= self.max_positions <= 100
            and self.max_hold_bars >= 1
            and 0 < self.drawdown_stop < 1
            and 0 < self.signal_threshold <= 1
            and 0 <= self.event_weight <= 1
            and self.decision_latency_ms > 0
        ):
            raise ValueError("实验协议超出研究边界")
        if any(
            type(v) is not int
            for v in (self.max_positions, self.max_hold_bars, self.decision_latency_ms)
        ):
            raise ValueError("数量与时钟配置必须为整数")


def validate_market(dataset):
    if dataset.get("schema_version") != "cz-ta-dataset.v1":
        raise ValueError("不支持的数据协议")
    symbols = dataset["symbols"]
    step = dataset["bar_ms"]
    if not symbols or len(set(symbols)) != len(symbols) or type(step) is not int or step <= 0:
        raise ValueError("标的集合或时间间隔不合法")
    bars = {}
    for bar in dataset["bars"]:
        clock, symbol = bar["open_ms"], bar["symbol"]
        if type(clock) is not int or clock % step or symbol not in symbols:
            raise ValueError("K 线时钟或标的不合法")
        values = [bar[k] for k in ("open", "high", "low", "close")]
        if any(
            isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0
            for v in values
        ):
            raise ValueError("价格必须为有限正数")
        opening, high, low, closing = values
        if high < max(opening, low, closing) or low > min(opening, high, closing):
            raise ValueError("OHLC 区间不合法")
        if symbol in bars.setdefault(clock, {}):
            raise ValueError("重复 K 线")
        bars[clock][symbol] = bar
    clocks = sorted(bars)
    if len(clocks) < 2 or any(b - a != step for a, b in zip(clocks, clocks[1:], strict=False)):
        raise ValueError("账户回放要求连续时间轴")
    if any(set(items) != set(symbols) for items in bars.values()):
        raise ValueError("任何持仓估值时点都不能缺失行情")
    seen = set()
    funding = sorted(dataset["funding"], key=lambda e: (e["time_ms"], e["symbol"]))
    for event in funding:
        key = event["time_ms"], event["symbol"]
        if (
            type(key[0]) is not int
            or key[1] not in symbols
            or key in seen
            or not clocks[0] <= key[0] < clocks[-1] + step
        ):
            raise ValueError("资金费事件重复或超出回放范围")
        if not all(
            isinstance(event[k], (int, float))
            and not isinstance(event[k], bool)
            and math.isfinite(event[k])
            for k in ("rate", "mark_price")
        ):
            raise ValueError("资金费率与标记价必须为有限数值")
        if abs(event["rate"]) > 1 or event["mark_price"] <= 0:
            raise ValueError("资金费率或标记价不合法")
        seen.add(key)
    return bars, clocks, funding


def replay(dataset, decisions, protocol):
    """一次回放只拥有一个账户；所有币共享本金、容量与总名义敞口。"""
    bars, clocks, funding = validate_market(dataset)
    step = dataset["bar_ms"]
    pending = sorted(deepcopy(decisions), key=lambda d: (d["available_ms"], d["symbol"]))
    keys = set()
    for decision in pending:
        key = decision["as_of_ms"], decision["symbol"]
        if (
            key in keys
            or decision["symbol"] not in dataset["symbols"]
            or not decision["as_of_ms"] < decision["available_ms"] < decision["expires_ms"]
            or type(decision["direction"]) is not int
            or decision["direction"] not in (-1, 0, 1)
            or not math.isfinite(decision["rank"])
        ):
            raise ValueError("决策时间、方向、排序或唯一性校验失败")
        keys.add(key)
    cash = protocol.initial_equity
    positions, active = {}, {}
    trades, fills, curve, funding_ledger = [], [], [], []
    peak, max_dd, max_gross = cash, 0.0, 0.0
    fee_rate, slip = protocol.fee_bps / 10000, protocol.slippage_bps / 10000
    fi, di, halted = 0, 0, False

    def equity(prices):
        return cash + sum(
            p["direction"] * p["quantity"] * (prices[s] - p["entry_price"])
            for s, p in positions.items()
        )

    def close(symbol, raw_price, clock, reason):
        nonlocal cash
        p = positions.pop(symbol)
        price = raw_price * (1 - p["direction"] * slip)
        fee = p["quantity"] * price * fee_rate
        gross = p["direction"] * p["quantity"] * (price - p["entry_price"])
        cash += gross - fee
        trades.append(
            {
                **p,
                "symbol": symbol,
                "exit_ms": clock,
                "exit_price": price,
                "exit_fee": fee,
                "gross_pnl": gross,
                "net_pnl": gross - p["entry_fee"] - fee + p["funding"],
                "reason": reason,
            }
        )
        fills.append(
            {
                "symbol": symbol,
                "time_ms": clock,
                "kind": "exit",
                "price": price,
                "quantity": p["quantity"],
                "direction": -p["direction"],
                "fee": fee,
            }
        )

    def settle(until, inclusive):
        nonlocal fi, cash
        while fi < len(funding) and (
            funding[fi]["time_ms"] <= until if inclusive else funding[fi]["time_ms"] < until
        ):
            e = funding[fi]
            if e["symbol"] in positions:
                p = positions[e["symbol"]]
                amount = -p["direction"] * p["quantity"] * e["mark_price"] * e["rate"]
                cash += amount
                p["funding"] += amount
                funding_ledger.append(
                    {**e, "quantity": p["quantity"], "direction": p["direction"], "amount": amount}
                )
            fi += 1

    for index, clock in enumerate(clocks):
        opening = {s: bar["open"] for s, bar in bars[clock].items()}
        closing = {s: bar["close"] for s, bar in bars[clock].items()}
        # 边界资金费先结算旧持仓，再处理同一开盘的平仓与新开仓。
        settle(clock, True)
        while di < len(pending) and pending[di]["available_ms"] <= clock:
            d = pending[di]
            active[d["symbol"]] = d
            di += 1
        current_equity = equity(opening)
        halted = (
            halted or current_equity <= 0 or 1 - current_equity / peak >= protocol.drawdown_stop
        )
        for symbol, p in list(positions.items()):
            d = active.get(symbol)
            changed = (
                d is not None and d["as_of_ms"] > p["as_of_ms"] and d["direction"] != p["direction"]
            )
            if halted or index - p["entry_index"] >= protocol.max_hold_bars or changed:
                close(
                    symbol,
                    opening[symbol],
                    clock,
                    "risk" if halted else "signal" if changed else "time",
                )
        # 一份决策最多开仓一次；旧信号不会在平仓后无限续开。
        candidates = sorted(active.values(), key=lambda d: (-d["rank"], d["symbol"]))
        for d in candidates:
            if (
                halted
                or index == len(clocks) - 1
                or d.get("consumed")
                or d["direction"] == 0
                or d["expires_ms"] <= clock
                or d["symbol"] in positions
            ):
                continue
            if len(positions) >= protocol.max_positions:
                continue
            eq = equity(opening)
            gross = sum(p["quantity"] * opening[s] for s, p in positions.items())
            room = max(0.0, eq - gross)
            symbol, direction = d["symbol"], d["direction"]
            fill_ratio = 1 + direction * slip
            # 按开盘标记价计敞口，同时扣除成交浮亏与手续费；空头滑点会增大数量。
            equity_use_per_notional = (1 + slip) / fill_ratio + fee_rate
            notional = min(eq * protocol.position_fraction, room / equity_use_per_notional)
            if notional < 10:
                continue
            price = opening[symbol] * fill_ratio
            quantity = notional / price
            fee = quantity * price * fee_rate
            cash -= fee
            positions[symbol] = {
                "entry_ms": clock,
                "entry_index": index,
                "entry_price": price,
                "quantity": quantity,
                "direction": direction,
                "entry_fee": fee,
                "funding": 0.0,
                "as_of_ms": d["as_of_ms"],
            }
            d["consumed"] = True
            fills.append(
                {
                    "symbol": symbol,
                    "time_ms": clock,
                    "kind": "entry",
                    "price": price,
                    "quantity": quantity,
                    "direction": direction,
                    "fee": fee,
                }
            )
        settle(clock + step, False)
        if index == len(clocks) - 1:
            for symbol in list(positions):
                close(symbol, closing[symbol], clock + step, "scheduled_end")
        eq = equity(closing)
        peak = max(peak, eq)
        max_dd = max(max_dd, 1 - eq / peak)
        halted = halted or max_dd >= protocol.drawdown_stop
        gross = sum(p["quantity"] * closing[s] for s, p in positions.items())
        max_gross = max(max_gross, gross / eq if eq > 0 else 0)
        # 独立核对已实现、未实现、费用和资金费是否能重建当前权益。
        reconstructed = (
            protocol.initial_equity
            + sum(t["net_pnl"] for t in trades)
            + sum(
                p["direction"] * p["quantity"] * (closing[s] - p["entry_price"])
                - p["entry_fee"]
                + p["funding"]
                for s, p in positions.items()
            )
        )
        if not math.isclose(eq, reconstructed, abs_tol=1e-7):
            raise AssertionError("账户经济恒等式不成立")
        curve.append(
            {
                "time_ms": clock + step,
                "equity": eq,
                "cash": cash,
                "unrealized": eq - cash,
                "open_positions": len(positions),
                "gross_notional": gross,
            }
        )
    wins = sum(t["net_pnl"] > 0 for t in trades)
    loss = -sum(min(t["net_pnl"], 0) for t in trades)
    profit = sum(max(t["net_pnl"], 0) for t in trades)
    return {
        "metrics": {
            "net_return": cash / protocol.initial_equity - 1,
            "ending_equity": cash,
            "closed_trades": len(trades),
            "win_rate": wins / len(trades) if trades else None,
            "maximum_close_drawdown": max_dd,
            "profit_factor": profit / loss if loss else None,
            "fees": sum(f["fee"] for f in fills),
            "funding": sum(e["amount"] for e in funding_ledger),
            "max_close_gross_fraction": max_gross,
            "remaining_positions": len(positions),
        },
        "trades": trades,
        "fills": fills,
        "equity_curve": curve,
        "funding_ledger": funding_ledger,
    }
