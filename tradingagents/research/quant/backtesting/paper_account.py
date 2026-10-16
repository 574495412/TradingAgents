"""独立双向基差的连续公开报价模拟账户；没有真实订单或生产接口。"""

from __future__ import annotations

import copy
import math
from decimal import ROUND_FLOOR, Decimal, localcontext

from tradingagents.dataflows.research.live_capture import MINUTE, digest
from tradingagents.research.quant.backtesting.bidirectional import HYPOTHESIS, MAXIMUM_DATA_AGE_MS
from tradingagents.research.quant.backtesting.forward_inputs import normalized_row
from tradingagents.research.quant.backtesting.paper_inputs import (
    contract_rules,
    covers,
    funding_window,
    number,
    quote,
)

EXECUTION = "received_quote_paper_v1"
INITIAL = Decimal("1000")
FEE = Decimal("0.0005")
FRACTION = Decimal("0.125")
MAX_DRAWDOWN = Decimal("0.08")


def boundary(stamp):
    """至少等待六十秒后，取不早于该时刻的UTC分钟边界。"""
    return (stamp + MINUTE + MINUTE - 1) // MINUTE * MINUTE


def plain(value):
    """检查点使用确定性十进制字符串，不把财务值转换为二进制浮点数。"""
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def joint_step(lots):
    """同时满足两类数量步长；不能用较大步长替代非整数倍约束。"""
    digits = max(max(0, -step.as_tuple().exponent) for _, _, step in lots)
    scale = 10**digits
    return Decimal(math.lcm(*(int(step * scale) for _, _, step in lots))) / scale


class PaperAccount:
    """单一账户共享现金和四仓容量，按实际收到的报价记录模拟成交。"""

    def __init__(self, symbols, slippage_bps, *, period_end_ms=None):
        if slippage_bps not in (3, 10) or len(set(symbols)) != len(symbols) or not symbols:
            raise ValueError("账户身份、固定全集或成本场景不合法")
        self.protocol = {
            "hypothesis": HYPOTHESIS,
            "execution_version": EXECUTION,
            "symbols": sorted(symbols),
            "initial_equity": str(INITIAL),
            "position_fraction": str(FRACTION),
            "fee_bps_each_side": 5,
            "slippage_bps_each_side": slippage_bps,
            "maximum_positions": 4,
            "maximum_gross_equity_ratio": 1,
            "hold_ms": 20 * MINUTE,
            "take_profit_fraction": "0.01",
            "maximum_drawdown": str(MAX_DRAWDOWN),
            "period_end_ms": period_end_ms,
            "orders_enabled": False,
            "production_eligible": False,
        }
        self.slip = Decimal(slippage_bps) / 10000
        self.cash = self.peak = INITIAL
        self.max_dd, self.halted = Decimal(0), False
        self.clock = self.closing = self.last_financial_decision_ms = None
        self.positions, self.pending, self.activities = {}, {}, {}
        self.fills, self.trades, self.payments, self.curve, self.declines, self.quality_errors = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        self.funding_events, self.funding_windows, self.missing = (
            {},
            {s: [] for s in symbols},
            dict.fromkeys(symbols, 0),
        )
        self.event_count, self.event_head = 0, None

    def checkpoint(self):
        """只作为重放校验目标；不能直接信任检查点中的现金和库存。"""
        return plain({k: v for k, v in self.__dict__.items() if k != "slip"})

    @classmethod
    def restore(cls, frames, checkpoint):
        """重放相同封存输入，再逐字段核对状态，拒绝截短或篡改的恢复。"""
        p = checkpoint["protocol"]
        value = cls(p["symbols"], p["slippage_bps_each_side"], period_end_ms=p["period_end_ms"])
        for frame in frames:
            value.apply(frame)
        if value.checkpoint() != checkpoint:
            raise ValueError("账户检查点不能由完整输入前缀重建")
        return value

    def apply(self, frame):
        """单帧事务失败不留下半笔库存；输入由外层持久日志负责先封存。"""
        candidate = copy.deepcopy(self)
        with localcontext() as context:
            context.prec = 40
            candidate._advance(frame)
        candidate.event_head = digest(
            {
                "sequence": self.event_count,
                "previous_sha256": self.event_head,
                "frame_sha256": digest(frame),
            }
        )
        candidate.event_count += 1
        self.__dict__ = candidate.__dict__
        return self.checkpoint()

    def _decline(self, symbol, reason, now, *, episode=None):
        self.declines.append(
            {"symbol": symbol, "reason": reason, "actual_ms": now, "episode": episode}
        )

    def _funding(self, records, now):
        for raw in records:
            window = funding_window(raw, now)
            symbol = window["symbol"]
            if symbol not in self.funding_windows:
                raise ValueError("资金费不属于冻结全集")
            returned = {f"{symbol}|{e['time_ms']}|{e['rate_type']}" for e in window["events"]}
            if any(
                e["symbol"] == symbol
                and window["start_ms"] <= e["time_ms"] <= window["end_ms"]
                and key not in returned
                for key, e in self.funding_events.items()
            ):
                raise ValueError("后续资金费窗口撤回了已确认事件")
            for event in window["events"]:
                key = f"{symbol}|{event['time_ms']}|{event['rate_type']}"
                economics = {
                    k: event[k] for k in ("symbol", "time_ms", "rate_type", "rate", "mark_price")
                }
                prior = self.funding_events.get(key)
                if prior is not None:
                    if prior != economics:
                        raise ValueError("已收到的资金费事件发生经济修订")
                    continue
                self.funding_events[key] = economics
                owners = [
                    p for p in [*self.positions.values(), *self.trades] if p["symbol"] == symbol
                ]
                owners = [
                    p for p in owners if p["entry_ms"] < event["time_ms"] <= p.get("exit_ms", now)
                ]
                if len(owners) > 1:
                    raise ValueError("资金费不能归属重叠库存")
                if not owners:
                    continue
                owner = owners[0]
                cash = (
                    -owner["direction"]
                    * owner["quantity"]
                    * number(event["mark_price"])
                    * number(event["rate"])
                )
                self.cash += cash
                owner["funding"] += cash
                if "exit_ms" in owner:
                    owner["net_pnl"] += cash
                self.payments.append({**event, "cycle_id": owner["cycle_id"], "cash": cash})
                if (
                    self.last_financial_decision_ms is not None
                    and event["time_ms"] <= self.last_financial_decision_ms
                ):
                    self.quality_errors.append(
                        {"reason": "FUNDING_ARRIVED_AFTER_A_FINANCIAL_DECISION", "event": key}
                    )
                    self.halted = True
            self.funding_windows[symbol].append({k: v for k, v in window.items() if k != "events"})

    def _marks(self, frame, now, closing):
        marks = {}
        for symbol, value in frame["marks"].items():
            if symbol not in self.funding_windows:
                raise ValueError("估值不属于冻结全集")
            normalized_row(value["row"])
            if (
                value["row"][6] + 1 != closing
                or not closing - 1000 <= value["received_ms"] <= now
                or not value["source_sha256"]
            ):
                raise ValueError("估值缺少同钟完整分钟或实际接收证据")
            marks[symbol] = number(value["row"][4], positive=True)
        return marks

    def equity(self, marks):
        return self.cash + sum(
            (
                p["direction"] * p["quantity"] * (marks[s] - p["entry_price"])
                for s, p in self.positions.items()
            ),
            Decimal(0),
        )

    def _valid_evidence(self, value, symbol, now, closing):
        if not value.get("ready"):
            return False
        if (
            value.get("hypothesis") != HYPOTHESIS
            or value.get("symbol") != symbol
            or type(value.get("actual_decision_ms")) is not int
            or not value["latest_input_received_ms"] <= value["actual_decision_ms"] <= now
            or type(value.get("data_clock_ms")) is not int
            or value["data_clock_ms"] % MINUTE
            or not 0 <= now - value["data_clock_ms"] <= MAXIMUM_DATA_AGE_MS
            or value["data_clock_ms"] > closing
            or value["direction"] not in (-1, 0, 1)
            or value.get("orders_enabled") is not False
            or value.get("production_eligible") is not False
        ):
            raise ValueError("方向证据时序、版本或来源不合法")
        return True

    def _risk(self, marks, now):
        if self.halted:
            for p in self.positions.values():
                p.setdefault("risk_due_ms", boundary(now))
        if set(self.positions) - set(marks):
            self.quality_errors.append(
                {"reason": "OPEN_POSITION_MARK_UNAVAILABLE", "actual_ms": now}
            )
            return None
        eq = self.equity(marks)
        self.peak = max(self.peak, eq)
        self.max_dd = max(self.max_dd, 1 - eq / self.peak)
        if eq <= 0 or self.max_dd >= MAX_DRAWDOWN:
            self.halted = True
        if self.halted:
            for p in self.positions.values():
                p.setdefault("risk_due_ms", boundary(now))
        return eq

    def _exit(self, symbol, frame, marks, now):
        p = self.positions[symbol]
        due = [(p["entry_ms"] + 20 * MINUTE, 1, "time")]
        due.extend(
            (p[k], priority, reason)
            for k, priority, reason in (("risk_due_ms", 0, "risk"), ("profit_due_ms", 2, "profit"))
            if k in p
        )
        ending = self.protocol["period_end_ms"]
        if ending is not None:
            due.append((ending, 0, "scheduled_end"))
        stamp, _, reason = min(due)
        if stamp > now:
            return
        try:
            q = quote(frame["quotes"], symbol, now, not_before_ms=stamp)
        except (KeyError, ValueError):
            self._decline(symbol, "EXIT_QUOTE_UNAVAILABLE", now, episode=p["episode"])
            return
        side = "bid" if p["direction"] == 1 else "ask"
        if q[f"{side}_qty"] < p["quantity"]:
            self._decline(symbol, "EXIT_DISPLAYED_CAPACITY_INSUFFICIENT", now, episode=p["episode"])
            return
        price = q[side] * (1 - p["direction"] * self.slip)
        fee = p["quantity"] * price * FEE
        gross = p["direction"] * p["quantity"] * (price - p["entry_price"])
        self.cash += gross - fee
        trade = p | {
            "exit_ms": now,
            "exit_price": price,
            "exit_fee": fee,
            "gross_pnl": gross,
            "net_pnl": gross - p["entry_fee"] - fee + p["funding"],
            "reason": reason,
            "quote_source_sha256": q["source_sha256"],
        }
        self.trades.append(trade)
        self.fills.append({"action": "CLOSE", **trade})
        del self.positions[symbol]
        self.activities[symbol]["flat_since_ms"] = now
        if now >= stamp + MINUTE:
            self.quality_errors.append(
                {"reason": "EXIT_DELAY_EXCEEDED_ONE_MINUTE", "cycle_id": p["cycle_id"]}
            )

    def _entry(self, symbol, pending, value, frame, marks, now, closing):
        if now >= pending["due_ms"] + MINUTE:
            return "RECHECK_DEADLINE_EXPIRED"
        if (
            not self._valid_evidence(value, symbol, now, closing)
            or value["actual_decision_ms"] < pending["due_ms"]
            or value["direction"] != pending["direction"]
        ):
            return "RECHECK_FAILED"
        if self.halted or symbol in self.positions or len(self.positions) >= 4:
            return "HALT_OR_SHARED_CAPACITY"
        if symbol not in marks:
            return "ENTRY_MARK_UNAVAILABLE"
        if set(self.positions) - set(marks):
            return "OPEN_POSITION_MARK_UNAVAILABLE"
        if any(
            not covers(self.funding_windows[s], p["entry_ms"] + 1, closing)
            for s, p in self.positions.items()
        ):
            return "OPEN_POSITION_FUNDING_COVERAGE_INCOMPLETE"
        if any(
            not covers(self.funding_windows[t["symbol"]], t["entry_ms"] + 1, t["exit_ms"])
            for t in self.trades
        ):
            return "CLOSED_POSITION_FUNDING_COVERAGE_INCOMPLETE"
        try:
            rules = contract_rules(frame["contracts"], symbol, now)
            q = quote(frame["quotes"], symbol, now, not_before_ms=value["actual_decision_ms"])
        except (KeyError, ValueError):
            return "ENTRY_RULES_OR_QUOTE_UNAVAILABLE"
        direction, mark = pending["direction"], marks[symbol]
        side = "ask" if direction == 1 else "bid"
        price = q[side] * (1 + direction * self.slip)
        eq = self.equity(marks)
        gross = sum((p["quantity"] * marks[s] for s, p in self.positions.items()), Decimal(0))
        drag = direction * (price - mark) + price * FEE
        denominator = mark + drag
        if eq <= gross or self.cash <= 0 or denominator <= 0:
            return "SHARED_EQUITY_BUDGET_EXHAUSTED"
        budget_qty = min(
            eq * FRACTION / price, (eq - gross) / denominator, self.cash / (price * FEE)
        )
        step = joint_step(rules["lots"])
        quantity = (min(budget_qty, *(hi for _, hi, _ in rules["lots"])) / step).to_integral_value(
            rounding=ROUND_FLOOR
        ) * step
        if (
            quantity < max(lo for lo, _, _ in rules["lots"])
            or quantity * min(mark, price) < rules["minimum_notional"]
        ):
            return "MINIMUM_LOT_OR_NOTIONAL_EXCEEDS_BUDGET"
        if quantity > q[f"{side}_qty"]:
            return "ENTRY_DISPLAYED_CAPACITY_INSUFFICIENT"
        fee = quantity * price * FEE
        if 1 - (eq - quantity * drag) / self.peak >= MAX_DRAWDOWN:
            return "IMMEDIATE_ACCOUNT_RISK"
        self.cash -= fee
        cycle = f"{EXECUTION}:{symbol}:{now}:{self.event_count}"
        p = {
            "symbol": symbol,
            "cycle_id": cycle,
            "episode": pending["episode"],
            "direction": direction,
            "entry_ms": now,
            "quantity": quantity,
            "entry_price": price,
            "entry_fee": fee,
            "funding": Decimal(0),
            "signal_ms": pending["signal_ms"],
            "recheck_ms": value["actual_decision_ms"],
            "scheduled_boundary_ms": pending["due_ms"],
            "quote_source_sha256": q["source_sha256"],
            "rules_source_sha256": rules["source_sha256"],
        }
        self.positions[symbol] = p
        self.fills.append({"action": "OPEN", **p})
        return None

    def _activities(self, evidence, now, closing, *, gap):
        for symbol, value in sorted(evidence.items()):
            activity = self.activities.setdefault(
                symbol, {"armed": False, "flat_since_ms": now, "data_clock_ms": None}
            )
            if gap:
                activity.update(armed=False, flat_since_ms=now, data_clock_ms=None)
            if not self._valid_evidence(value, symbol, now, closing):
                self.missing[symbol] += 1
                activity.update(armed=False, flat_since_ms=now, data_clock_ms=None)
                continue
            data_clock = value["data_clock_ms"]
            previous = activity["data_clock_ms"]
            if previous is not None and data_clock > previous + MINUTE:
                activity.update(armed=False, flat_since_ms=now)
            if previous is not None and data_clock <= previous:
                continue
            activity["data_clock_ms"] = data_clock
            if symbol in self.positions or symbol in self.pending:
                continue
            if (
                value["two_completed_minutes_within_band"]
                and data_clock - MINUTE >= activity["flat_since_ms"]
            ):
                activity["armed"] = True
            if not activity["armed"] or not value["direction"]:
                continue
            episode = f"{HYPOTHESIS}:{symbol}:{data_clock}"
            activity["armed"] = False
            activity["flat_since_ms"] = now
            if self.halted:
                self._decline(symbol, "HALTED_ACTIVITY_CONSUMED", now, episode=episode)
                continue
            self.pending[symbol] = {
                "episode": episode,
                "direction": value["direction"],
                "signal_ms": now,
                "due_ms": boundary(now),
            }

    def _advance(self, frame):
        now, closing = frame["actual_ms"], frame["canonical_close_ms"]
        phase = frame.get("phase", "decision")
        if (
            type(now) is not int
            or type(closing) is not int
            or closing % MINUTE
            or not closing + 1000 <= now < closing + 2 * MINUTE
            or (self.clock is not None and now <= self.clock)
            or (self.closing is not None and closing < self.closing)
            or frame.get("clock_verified") is not True
            or set(frame["evidence"]) != set(self.protocol["symbols"])
            or phase not in {"decision", "exit_only"}
            or (phase == "decision" and now >= closing + MINUTE)
        ):
            raise ValueError("账户帧不满足真实时钟、完整冻结集合或单调性")
        gap = bool(frame.get("source_restart")) or (
            self.closing is not None and closing > self.closing + MINUTE
        )
        if gap:
            self.quality_errors.append(
                {"reason": "OBSERVATION_GAP", "previous_ms": self.closing, "next_ms": closing}
            )
            for symbol, pending in self.pending.items():
                self._decline(symbol, "GAP_ACTIVITY_CONSUMED", now, episode=pending["episode"])
            self.pending.clear()
            for activity in self.activities.values():
                activity.update(armed=False, flat_since_ms=now, data_clock_ms=None)
        self._funding(frame.get("funding", []), now)
        for failure in frame.get("operational_errors", []):
            self.quality_errors.append({"reason": failure["reason"], "actual_ms": now})
        marks = self._marks(frame, now, closing)
        if self.protocol["period_end_ms"] is not None and now >= self.protocol["period_end_ms"]:
            self.halted = True
            for symbol, pending in self.pending.items():
                self._decline(
                    symbol, "PERIOD_END_ACTIVITY_CONSUMED", now, episode=pending["episode"]
                )
            self.pending.clear()
        self._risk(marks, now)
        for symbol, p in self.positions.items():
            if symbol not in marks:
                continue
            if p["direction"] * (marks[symbol] / p["entry_price"] - 1) >= Decimal("0.01"):
                p.setdefault("profit_due_ms", boundary(now))
        for symbol in sorted(self.positions):
            self._exit(symbol, frame, marks, now)
        for symbol, pending in sorted(self.pending.items()) if phase == "decision" else []:
            if now < pending["due_ms"]:
                continue
            del self.pending[symbol]
            reason = self._entry(
                symbol, pending, frame["evidence"][symbol], frame, marks, now, closing
            )
            if reason:
                self._decline(symbol, reason, now, episode=pending["episode"])
        if phase == "decision":
            self._activities(frame["evidence"], now, closing, gap=gap)
        eq = self._risk(marks, now)
        if eq is not None:
            reconstructed = (
                INITIAL
                + sum((t["net_pnl"] for t in self.trades), Decimal(0))
                + sum(
                    (
                        p["direction"] * p["quantity"] * (marks[s] - p["entry_price"])
                        - p["entry_fee"]
                        + p["funding"]
                        for s, p in self.positions.items()
                    ),
                    Decimal(0),
                )
            )
            if abs(eq - reconstructed) > Decimal("1e-30"):
                raise ValueError("完整账户权益不守恒")
        self.curve.append(
            {
                "actual_ms": now,
                "canonical_close_ms": closing,
                "equity": eq,
                "cash": self.cash,
                "positions": len(self.positions),
            }
        )
        self.clock, self.closing, self.last_financial_decision_ms = now, closing, now

    def funding_verified(self):
        """每个暴露区间要求两次明确区间查询，其中一次至少晚于平仓61秒。"""
        if self.positions:
            return False
        for trade in self.trades:
            windows = self.funding_windows[trade["symbol"]]
            first = [w for w in windows if w["requested_ms"] <= trade["exit_ms"] + MINUTE]
            later = [w for w in windows if w["requested_ms"] >= trade["exit_ms"] + MINUTE + 1000]
            if not covers(first, trade["entry_ms"] + 1, trade["exit_ms"]) or not covers(
                later, trade["entry_ms"] + 1, trade["exit_ms"]
            ):
                return False
        return True

    def metrics(self):
        """未确认资金费或退出时收益仅为暂记，不能据此宣布稳定或上线。"""
        with localcontext() as context:
            context.prec = 40
            nets = [t["net_pnl"] for t in self.trades]
            loss = -sum((min(v, Decimal(0)) for v in nets), Decimal(0))
            eq = self.curve[-1]["equity"] if self.curve else INITIAL
            return plain(
                {
                    "schema": "cz-received-quote-paper-account.v1",
                    "execution_version": EXECUTION,
                    "equity": eq,
                    "net_return": eq / INITIAL - 1 if eq is not None else None,
                    "maximum_observed_drawdown": self.max_dd,
                    "drawdown_fully_observed": all(v["equity"] is not None for v in self.curve)
                    and not any(v["reason"] == "OBSERVATION_GAP" for v in self.quality_errors),
                    "closed_trades": len(nets),
                    "initial_activities_traded": len(
                        {f["episode"] for f in self.fills if f["action"] == "OPEN"}
                    ),
                    "open_positions": len(self.positions),
                    "pending_entries": len(self.pending),
                    "net_win_rate": Decimal(sum(v > 0 for v in nets)) / len(nets) if nets else None,
                    "profit_factor": sum((max(v, Decimal(0)) for v in nets), Decimal(0)) / loss
                    if loss
                    else None,
                    "funding_cash": sum((p["cash"] for p in self.payments), Decimal(0)),
                    "funding_coverage_rechecked": self.funding_verified(),
                    "quality_errors": self.quality_errors,
                    "missing_evidence_observations": self.missing,
                    "cash_and_pnl_are_provisional": bool(self.positions)
                    or not self.funding_verified()
                    or eq is None,
                    "stable_profitability_proven": False,
                    "real_fills_proven": False,
                    "orders_enabled": False,
                    "production_eligible": False,
                }
            )
