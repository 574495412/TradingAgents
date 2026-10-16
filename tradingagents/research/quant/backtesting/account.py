"""因子搜索的单一连续账户，真实资金费反馈权益、仓位和盈利续开权限。"""

from __future__ import annotations

import math
from dataclasses import asdict

from tradingagents.research.quant.backtesting.signals import MINUTE, PROTOCOL


class Account:
    """所有品种共享本金与容量；只有真实模拟平仓净盈利才产生单次续开资格。"""

    def __init__(self, factor, hold_minutes, slippage_bps, funding, *, reentry=False):
        self.factor, self.hold, self.slip = factor, hold_minutes * MINUTE, slippage_bps / 10000
        self.reentry, self.funding, self.funding_cursor = reentry, funding, 0
        self.cash = self.peak = PROTOCOL["initial_equity"]
        self.fee = PROTOCOL["fee_bps"] / 10000
        self.positions, self.pending, self.receipts, self.closes = {}, {}, {}, {}
        self.trades, self.fills, self.funding_ledger, self.curve, self.declines = [], [], [], [], []
        self.max_dd, self.closed_net, self.halted = 0.0, 0.0, False

    def equity(self, market, column):
        total = self.cash
        for symbol, p in self.positions.items():
            if symbol not in market or not market[symbol][2] or not market[symbol][3]:
                raise ValueError(f"已有库存缺少真实可交易估值：{symbol}")
            price = market[symbol][column]
            if not math.isfinite(price) or price <= 0:
                raise ValueError("已有库存价格不合法")
            total += p["direction"] * p["quantity"] * (price - p["entry_price"])
        return total

    def settle(self, until, inclusive):
        while self.funding_cursor < len(self.funding):
            event = self.funding[self.funding_cursor]
            stamp = event["time_ms"]
            if stamp > until or (stamp == until and not inclusive):
                break
            p = self.positions.get(event["symbol"])
            if p:
                cash = -p["direction"] * p["quantity"] * event["mark_price"] * event["rate"]
                self.cash += cash
                p["funding"] += cash
                self.funding_ledger.append(
                    {
                        **event,
                        "cycle_id": p["cycle_id"],
                        "direction": p["direction"],
                        "quantity": p["quantity"],
                        "cash": cash,
                    }
                )
            self.funding_cursor += 1

    def close(self, symbol, raw_price, stamp, reason):
        p = self.positions.pop(symbol)
        self.closes.pop(symbol, None)
        price = raw_price * (1 - p["direction"] * self.slip)
        fee = p["quantity"] * price * self.fee
        gross = p["direction"] * p["quantity"] * (price - p["entry_price"])
        net = gross - p["entry_fee"] - fee + p["funding"]
        self.cash += gross - fee
        self.closed_net += net
        trade = {
            **p,
            "symbol": symbol,
            "exit_ms": stamp,
            "exit_price": price,
            "exit_fee": fee,
            "gross_pnl": gross,
            "net_pnl": net,
            "reason": reason,
        }
        self.trades.append(trade)
        self.fills.append(
            {
                "cycle_id": p["cycle_id"],
                "symbol": symbol,
                "clock": stamp,
                "action": "CLOSE",
                "direction": p["direction"],
                "quantity": p["quantity"],
                "price": price,
                "fee": fee,
            }
        )
        if self.reentry and net > 0 and reason not in {"risk", "scheduled_end"}:
            self.receipts[symbol] = {
                "source_cycle_id": p["cycle_id"],
                "episode": p["episode"],
                "direction": p["direction"],
                "available_ms": stamp,
                "expires_ms": stamp + PROTOCOL["profit_receipt_valid_ms"],
                "net_pnl": net,
            }
        else:
            self.receipts.pop(symbol, None)

    def open_phase(self, stamp, market):
        # 同钟资金费归旧库存；结算后才按真实开盘处理成交。
        self.settle(stamp, True)
        opening_equity = self.equity(market, 0)
        self.halted = (
            self.halted
            or opening_equity <= 0
            or 1 - opening_equity / self.peak >= PROTOCOL["maximum_account_drawdown"]
        )
        if self.halted:
            for symbol in self.positions:
                self.closes.setdefault(
                    symbol,
                    {"available_ms": stamp + PROTOCOL["decision_latency_ms"], "reason": "risk"},
                )
        for symbol, p in list(self.positions.items()):
            requested = self.closes.get(symbol)
            due = requested and requested["available_ms"] <= stamp
            if due or stamp - p["entry_ms"] >= self.hold:
                self.close(symbol, market[symbol][0], stamp, requested["reason"] if due else "time")
        due = sorted(
            ((s, p) for s, p in self.pending.items() if p["available_ms"] <= stamp),
            key=lambda pair: (-pair[1]["signal"]["priority"], pair[0]),
        )
        for symbol, pending in due:
            del self.pending[symbol]
            signal = pending["signal"]
            if (
                self.halted
                or symbol in self.positions
                or stamp > pending["expires_ms"]
                or len(self.positions) >= PROTOCOL["max_positions"]
            ):
                self.declines.append(
                    {"clock": stamp, "symbol": symbol, "reason": "risk_capacity_or_expiry"}
                )
                continue
            quote = market.get(symbol)
            if (
                quote is None
                or not quote[2]
                or not quote[3]
                or not math.isfinite(quote[0])
                or quote[0] <= 0
            ):
                self.declines.append(
                    {"clock": stamp, "symbol": symbol, "reason": "no_observed_open"}
                )
                continue
            receipt_id = pending.get("receipt_from")
            if receipt_id and (
                symbol not in self.receipts
                or self.receipts[symbol]["source_cycle_id"] != receipt_id
            ):
                raise ValueError("盈利续开缺少未消费的真实净盈利结算")
            eq = self.equity(market, 0)
            gross = sum(p["quantity"] * market[s][0] for s, p in self.positions.items())
            ratio = 1 + signal["direction"] * self.slip
            use = (1 + self.slip) / ratio + self.fee
            notional = min(eq * PROTOCOL["position_fraction"], max(0.0, eq - gross) / use)
            if notional < PROTOCOL["minimum_trade_notional"]:
                continue
            price, quantity = quote[0] * ratio, notional / (quote[0] * ratio)
            fee = quantity * price * self.fee
            self.cash -= fee
            cycle = f"{self.factor}:{symbol}:{stamp}:{len(self.fills)}"
            self.positions[symbol] = {
                "cycle_id": cycle,
                "episode": pending["episode"],
                "entry_ms": stamp,
                "signal_ms": signal["as_of_ms"],
                "direction": signal["direction"],
                "quantity": quantity,
                "entry_price": price,
                "entry_fee": fee,
                "funding": 0.0,
                "receipt_from": receipt_id,
            }
            self.fills.append(
                {
                    "cycle_id": cycle,
                    "symbol": symbol,
                    "clock": stamp,
                    "action": "OPEN",
                    "direction": signal["direction"],
                    "quantity": quantity,
                    "price": price,
                    "fee": fee,
                    "signal_ms": signal["as_of_ms"],
                    "receipt_from": receipt_id,
                }
            )
            if receipt_id:
                self.receipts.pop(symbol)

    def close_phase(self, clock, market, pulses, qualified, *, final=False):
        self.settle(clock, False)
        if final:
            for symbol in list(self.positions):
                self.close(symbol, market[symbol][1], clock, "scheduled_end")
            self.pending.clear()
            self.receipts.clear()
        eq = self.equity(market, 1)
        self.peak = max(self.peak, eq)
        self.max_dd = max(self.max_dd, 1 - eq / self.peak)
        self.halted = self.halted or self.max_dd >= PROTOCOL["maximum_account_drawdown"] or eq <= 0
        for symbol, p in self.positions.items():
            mark = market[symbol][1]
            liquidation = mark * (1 - p["direction"] * self.slip)
            net = (
                p["direction"] * p["quantity"] * (liquidation - p["entry_price"])
                - p["entry_fee"]
                - p["quantity"] * liquidation * self.fee
                + p["funding"]
            )
            target = (
                p["direction"] * (mark / p["entry_price"] - 1)
                >= PROTOCOL["take_profit_price_fraction"]
            )
            if symbol not in self.closes and (self.halted or (target and net > 0)):
                self.closes[symbol] = {
                    "available_ms": clock + PROTOCOL["decision_latency_ms"],
                    "reason": "risk" if self.halted else "profit",
                }
        if not final and not self.halted:
            if self.reentry:
                for symbol, receipt in list(self.receipts.items()):
                    if clock >= receipt["expires_ms"]:
                        pending = self.pending.get(symbol)
                        if pending and pending.get("receipt_from") == receipt["source_cycle_id"]:
                            self.pending.pop(symbol)
                            self.declines.append(
                                {
                                    "clock": clock,
                                    "symbol": symbol,
                                    "reason": "profit_receipt_expired_before_fill",
                                }
                            )
                        self.receipts.pop(symbol)
                        continue
                    signal = qualified.get(symbol)
                    same_activity = signal and (
                        not self.factor.startswith("shock") or signal.episode == receipt["episode"]
                    )
                    if (
                        same_activity
                        and signal.direction == receipt["direction"]
                        and clock > receipt["available_ms"]
                    ):
                        self.schedule(signal, clock, receipt)
            for signal in pulses:
                self.schedule(signal, clock)
        reconstructed = (
            PROTOCOL["initial_equity"]
            + self.closed_net
            + sum(
                p["direction"] * p["quantity"] * (market[s][1] - p["entry_price"])
                - p["entry_fee"]
                + p["funding"]
                for s, p in self.positions.items()
            )
        )
        if not math.isclose(eq, reconstructed, abs_tol=1e-7):
            raise AssertionError("完整权益与已实现、未实现、费用、资金费不守恒")
        self.curve.append(
            {
                "clock": clock,
                "equity": eq,
                "cash": self.cash,
                "positions": len(self.positions),
                "gross_notional": sum(
                    p["quantity"] * market[s][1] for s, p in self.positions.items()
                ),
            }
        )

    def schedule(self, signal, clock, receipt=None):
        symbol = signal.symbol
        if symbol in self.positions or symbol in self.pending:
            return
        if receipt and clock + PROTOCOL["decision_latency_ms"] >= receipt["expires_ms"]:
            return
        self.pending[symbol] = {
            "signal": asdict(signal),
            "available_ms": clock + PROTOCOL["decision_latency_ms"],
            "expires_ms": clock + PROTOCOL["max_signal_age_ms"],
            "episode": receipt["episode"] if receipt else signal.episode,
            "receipt_from": receipt["source_cycle_id"] if receipt else None,
        }

    def metrics(self, *, after=None):
        curve = self.curve if after is None else [v for v in self.curve if v["clock"] >= after]
        trades = self.trades if after is None else [v for v in self.trades if v["exit_ms"] > after]
        initial = PROTOCOL["initial_equity"] if after is None else curve[0]["equity"]
        peak, dd = initial, 0.0
        for point in curve:
            peak = max(peak, point["equity"])
            dd = max(dd, 1 - point["equity"] / peak)
        n, wins = len(trades), sum(v["net_pnl"] > 0 for v in trades)
        profit, loss = (
            sum(max(v["net_pnl"], 0) for v in trades),
            -sum(min(v["net_pnl"], 0) for v in trades),
        )
        p, z = wins / n if n else 0, 1.96
        wilson = (
            (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)))
            / (1 + z * z / n)
            if n
            else 0
        )
        return {
            "net_return": curve[-1]["equity"] / initial - 1,
            "maximum_close_drawdown": dd,
            "closed_trades": n,
            "win_rate": wins / n if n else None,
            "wilson_lower": wilson,
            "profit_factor": profit / loss if loss else None,
            "ending_equity": curve[-1]["equity"],
            "funding_cash": sum(
                v["cash"] for v in self.funding_ledger if after is None or v["time_ms"] > after
            ),
            "open_positions": len(self.positions),
            "halted": self.halted,
            "funding_included": True,
        }
