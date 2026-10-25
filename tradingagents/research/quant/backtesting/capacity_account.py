"""受事前成交额和离散数量共同约束的连续账户，不强造残仓退出。"""

from __future__ import annotations

import math
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal, InvalidOperation, localcontext
from numbers import Integral

from .forecast import PREDICTED_RETURN_THRESHOLD
from .signals import MINUTE, PROTOCOL

D = Decimal
ZERO = D(0)
PRECISION = 60


def decimal(value):
    """数值按十进制文本进入账本，拒绝布尔、未知和非有限值。"""
    if isinstance(value, bool) or value is None:
        raise ValueError("金额或数量不能是布尔或未知")
    try:
        result = value if isinstance(value, D) else D(str(value))
    except (InvalidOperation, ValueError, TypeError) as error:
        raise ValueError("金额或数量不是十进制数") from error
    if not result.is_finite():
        raise ValueError("金额或数量必须有限")
    return result


def integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(name + "必须为非负整数")
    return int(value)


class QuantityRule:
    """两个数量网格用整数CRT精确求交，不把minQty偏移默认为零。"""

    def __init__(self, rule):
        self.source = rule
        self.known = rule.get("rule_known") is True
        self.reason = rule.get("unknown_reason")
        if not self.known:
            return
        try:
            lots = rule["lots"]
            if len(lots) != 2 or {row["filter_type"] for row in lots} != {
                "LOT_SIZE",
                "MARKET_LOT_SIZE",
            }:
                raise ValueError("必须同时有两种数量规则")
            values = [
                {name: decimal(row[name]) for name in ("min_qty", "max_qty", "step_size", "offset")}
                for row in lots
            ]
            if any(
                row["min_qty"] < 0
                or row["max_qty"] <= 0
                or row["min_qty"] > row["max_qty"]
                or row["step_size"] <= 0
                or row["offset"] != row["min_qty"]
                for row in values
            ):
                raise ValueError("数量边界、正步长或minQty偏移不合法")
            self.minimum = max(row["min_qty"] for row in values)
            self.maximum = min(row["max_qty"] for row in values)
            self.min_notional = decimal(rule["min_notional"])
            if self.min_notional < 0 or self.maximum < self.minimum:
                raise ValueError("两种数量上下限没有交集")
            places = max(
                0, *(-value.as_tuple().exponent for row in values for value in row.values())
            )
            self.scale = 10**places
            with localcontext() as context:
                context.prec = max(PRECISION, places + 40)
                offsets = [int(row["offset"] * self.scale) for row in values]
                steps = [int(row["step_size"] * self.scale) for row in values]
                a, b = offsets
                m, n = steps
                common = math.gcd(m, n)
                if (b - a) % common:
                    raise ValueError("两个数量偏移网格没有交集")
                reduced_n = n // common
                multiplier = (
                    ((b - a) // common * pow(m // common, -1, reduced_n)) % reduced_n
                    if reduced_n > 1
                    else 0
                )
                self.step_units = m // common * n
                self.offset_units = (a + m * multiplier) % self.step_units
                self.step = D(self.step_units) / self.scale
                self.offset = D(self.offset_units) / self.scale
        except (KeyError, ValueError, TypeError, InvalidOperation) as error:
            self.known, self.reason = False, "INVALID_RULE: " + str(error)

    def floor(self, cap):
        """只能向下取可行数量，不抬高到最低量或最小额。"""
        if not self.known:
            return ZERO
        with localcontext() as context:
            context.prec = PRECISION
            cap = min(decimal(cap), self.maximum)
            units = int((cap * self.scale).to_integral_value(rounding=ROUND_FLOOR))
            units = (
                self.offset_units
                + ((units - self.offset_units) // self.step_units) * self.step_units
            )
            quantity = D(units) / self.scale
            return quantity if quantity > 0 and quantity >= self.minimum else ZERO

    def evidence(self):
        return {
            "rule_known": self.known,
            "unknown_reason": self.reason,
            "common_step": self.step if self.known else None,
            "common_offset": self.offset if self.known else None,
            "minimum": self.minimum if self.known else None,
            "maximum": self.maximum if self.known else None,
            "lots": self.source.get("lots", []),
            "source_snapshot_sha256": self.source.get("source_snapshot_sha256"),
            "historical_pti_proven": False,
        }

    def reserve_minimum(self, lower):
        """仅向上求合法余仓下限；此值不用于抬高实际成交数量。"""
        if not self.known:
            return None
        with localcontext() as context:
            context.prec = PRECISION
            lower = max(decimal(lower), self.minimum)
            units = int((lower * self.scale).to_integral_value(rounding=ROUND_CEILING))
            multiples = -((self.offset_units - units) // self.step_units)
            quantity = D(self.offset_units + multiples * self.step_units) / self.scale
            return quantity if 0 < quantity <= self.maximum else None


class CapacityAccount:
    """一批同预权益开仓，按预算逐步退出；只有原腿全清才计完成交易。"""

    def __init__(
        self,
        rules,
        funding,
        *,
        slippage_bps,
        holding_minutes=240,
        end_ms,
        factor="capacity",
    ):
        self.slip = decimal(slippage_bps) / 10000
        if not 0 <= self.slip < 1:
            raise ValueError("滑点必须处于[0,10000)bps")
        holding_minutes = integer(holding_minutes, "持有分钟")
        if not holding_minutes:
            raise ValueError("持有分钟必须为正")
        self.end_ms = integer(end_ms, "期末")
        if self.end_ms % MINUTE or not isinstance(factor, str) or not factor:
            raise ValueError("期末必须整分钟且模型身份不能为空")
        self.factor, self.hold = factor, holding_minutes * MINUTE
        self.rules = {symbol: QuantityRule(rule) for symbol, rule in rules.items()}
        self.funding, self.funding_cursor = funding, 0
        previous = -1
        for event in funding:
            stamp = integer(event["time_ms"], "资金费原毫秒")
            if stamp < previous or event.get("rate_type") not in {"Regular", "Special"}:
                raise ValueError("资金费须按原毫秒排序且有显式Regular/Special类型")
            previous = stamp
        self.initial = decimal(PROTOCOL["initial_equity"])
        self.fee = decimal(PROTOCOL["fee_bps"]) / 10000
        self.position_fraction = decimal(PROTOCOL["position_fraction"])
        self.minimum_notional = decimal(PROTOCOL["minimum_trade_notional"])
        self.cash = self.peak = self.initial
        self.closed_net = self.max_dd = self.maximum_open_drawdown = ZERO
        self.positions, self.closes = {}, {}
        self.pending = None
        self.fills, self.trades, self.reduction_fills = [], [], []
        self.curve, self.funding_ledger, self.declines, self.entry_plans = [], [], [], []
        self.halted = self.finished = False
        self.invalid, self.last_clock = None, None
        self._clock, self._used_budget = None, {}

    def _fail(self, kind, reason):
        self.invalid = {
            "at": self._clock,
            "type": kind,
            "reason": reason,
            "numeric_conclusions_allowed": False,
            "remaining_symbols": sorted(self.positions),
        }

    def _quote(self, symbol, market, column):
        quote = market.get(symbol)
        if (
            quote is None
            or type(quote.get("qualified")) is not bool
            or type(quote.get("observed")) is not bool
        ):
            raise ValueError("缺少明确分钟身份：" + symbol)
        if not quote["qualified"] or not quote["observed"]:
            raise ValueError("已有/待成交合约无真实可交易分钟：" + symbol)
        price = decimal(quote[column])
        if price <= 0:
            raise ValueError("真实价格必须为正：" + symbol)
        return price

    def _budget(self, symbol, market, phase):
        quote = market.get(symbol)
        if quote is None:
            return None
        maximum = self._clock - MINUTE if phase == "open" else self._clock
        asof = integer(quote[phase + "_budget_asof_ms"], "预算可见时钟")
        if asof > maximum:
            raise ValueError("预算引用了阶段之后的数据：" + symbol)
        value = quote[phase + "_budget"]
        if value is None:
            return None
        value = decimal(value)
        if value < 0:
            raise ValueError("预算必须非负或明确UNKNOWN")
        if phase == "open":
            value -= self._used_budget.get(symbol, ZERO)
            if value < 0:
                raise AssertionError("同分钟预算已透支")
        return value

    def equity(self, market, column):
        return self.cash + sum(
            (
                D(p["direction"])
                * p["quantity"]
                * (self._quote(symbol, market, column) - p["entry_price"])
                for symbol, p in self.positions.items()
            ),
            ZERO,
        )

    def _settle(self, until, inclusive):
        while self.funding_cursor < len(self.funding):
            event = self.funding[self.funding_cursor]
            stamp = event["time_ms"]
            if stamp > until or (stamp == until and not inclusive):
                break
            position = self.positions.get(event["symbol"])
            if position is not None:
                rate, mark = decimal(event["rate"]), decimal(event["mark_price"])
                if mark <= 0:
                    raise ValueError("资金费mark必须为正")
                cash = -D(position["direction"]) * position["quantity"] * mark * rate
                self.cash += cash
                position["funding"] += cash
                self.funding_ledger.append(
                    {
                        **event,
                        "rate": rate,
                        "mark_price": mark,
                        "cycle_id": position["cycle_id"],
                        "direction": position["direction"],
                        "quantity": position["quantity"],
                        "cash": cash,
                        "processing_clock_ms": self._clock,
                        "phase": "before_open" if inclusive else "before_close",
                    }
                )
            self.funding_cursor += 1

    def _halt(self, equity, phase_clock):
        drawdown = 1 - equity / self.peak
        if self.halted or equity <= 0 or drawdown >= decimal(PROTOCOL["maximum_account_drawdown"]):
            self.halted = True
            if self.pending is not None:
                self._decline("risk_halt_pending_cancelled", self.pending["selected"])
                self.pending = None
            for symbol in self.positions:
                self.closes.setdefault(
                    symbol, {"available_ms": phase_clock + MINUTE, "reason": "risk"}
                )

    def _decline(self, reason, selected, **evidence):
        self.declines.append(
            {
                "clock": self._clock,
                "processing_clock_ms": self._clock,
                "reason": reason,
                "symbols": [row[0] for row in selected],
                **evidence,
            }
        )

    def _reduce(self, symbol, market, stamp, request):
        position, rule = self.positions[symbol], self.rules[symbol]
        raw = self._quote(symbol, market, "open")
        fill = raw * (1 - D(position["direction"]) * self.slip)
        budget = self._budget(symbol, market, "open")
        if budget is None:
            self._decline("unknown_exit_budget", [(symbol, 0, 0)], raw_stamp_ms=stamp)
            return
        before = position["quantity"]
        capacity_floor = rule.floor(min(before, budget / max(raw, fill)))
        if capacity_floor * max(raw, fill) > budget:
            capacity_floor = rule.floor(capacity_floor - rule.step)
        quantity, reserved = capacity_floor, None
        minimum_proxy = max(rule.min_notional, self.minimum_notional)
        if 0 < quantity < before:
            total_units = before * rule.scale
            compatible_split = (
                total_units == total_units.to_integral_value()
                and (int(total_units) - 2 * rule.offset_units) % rule.step_units == 0
            )
            if not compatible_split:
                quantity = ZERO
            else:
                reserved = rule.reserve_minimum(minimum_proxy / min(raw, fill))
                if reserved is not None and reserved * min(raw, fill) < minimum_proxy:
                    reserved = rule.reserve_minimum(reserved + rule.step)
                quantity = (
                    rule.floor(min(capacity_floor, before - reserved))
                    if reserved is not None
                    else ZERO
                )
                remaining = before - quantity
                if rule.floor(remaining) != remaining or remaining * min(raw, fill) < minimum_proxy:
                    quantity = ZERO
        proxy = quantity * min(raw, fill)
        if quantity <= 0 or proxy < minimum_proxy:
            self._decline(
                "exit_budget_or_minimum_wait",
                [(symbol, 0, 0)],
                raw_stamp_ms=stamp,
                budget=budget,
                remaining_quantity=position["quantity"],
                capacity_quantity_floor=capacity_floor,
                minimum_reserved_quantity=reserved,
            )
            return
        charge, fee = quantity * max(raw, fill), quantity * fill * self.fee
        if charge > budget:
            raise AssertionError("部分退出不能突破当分钟剩余预算")
        gross = D(position["direction"]) * quantity * (fill - position["entry_price"])
        self._used_budget[symbol] = self._used_budget.get(symbol, ZERO) + charge
        self.cash += gross - fee
        position["quantity"] -= quantity
        position["gross_pnl_reduced"] += gross
        position["exit_fees"] += fee
        position["exit_notional"] += quantity * fill
        position["reduced_quantity"] += quantity
        position["first_exit_ms"] = (
            stamp if position["first_exit_ms"] is None else position["first_exit_ms"]
        )
        finished = position["quantity"] == 0
        row = {
            "cycle_id": position["cycle_id"],
            "symbol": symbol,
            "clock": stamp,
            "processing_clock_ms": self._clock,
            "phase": "open",
            "action": "CLOSE" if finished else "REDUCE",
            "reason": request["reason"],
            "exit_request": dict(request),
            "direction": position["direction"],
            "quantity": quantity,
            "quantity_before": before,
            "qty_remaining": position["quantity"],
            "quantity_after": position["quantity"],
            "capacity_quantity_floor": capacity_floor,
            "minimum_reserved_quantity": reserved,
            "dust_avoidance_quantity_reduction": capacity_floor - quantity,
            "raw_price": raw,
            "price": fill,
            "fee": fee,
            "gross_pnl": gross,
            "capacity_charge": charge,
            "open_budget_remaining_before": budget,
            "open_budget_asof_ms": market[symbol]["open_budget_asof_ms"],
            "minimum_notional_proxy": proxy,
            "historical_mark_used": False,
            "rule": rule.evidence(),
        }
        position["reduction_fill_indices"].append(len(self.fills))
        self.fills.append(row)
        self.reduction_fills.append(row)
        if finished:
            net = (
                position["gross_pnl_reduced"]
                - position["entry_fee"]
                - position["exit_fees"]
                + position["funding"]
            )
            self.trades.append(
                {
                    **position,
                    "symbol": symbol,
                    "quantity": position["original_quantity"],
                    "exit_ms": stamp,
                    "exit_price": position["exit_notional"] / position["original_quantity"],
                    "exit_fee": position["exit_fees"],
                    "gross_pnl": position["gross_pnl_reduced"],
                    "net_pnl": net,
                    "reason": request["reason"],
                }
            )
            self.closed_net += net
            del self.positions[symbol]
            self.closes.pop(symbol, None)

    def _open_batch(self, market, stamp):
        pending = self.pending
        if pending is None or stamp < pending["available_ms"]:
            return
        self.pending = None
        selected = pending["selected"]
        plan = {
            "event": "ENTRY_ATTEMPT",
            "batch_id": pending["batch_id"],
            "signal_ms": pending["signal_ms"],
            "clock": stamp,
            "processing_clock_ms": self._clock,
            "phase": "open",
            "legs": [],
            "decision_budgets": pending["decision_budgets"],
            "status": "ZERO_BATCH",
            "historical_mark_used": False,
        }
        self.entry_plans.append(plan)
        if self.halted or self.positions or stamp > pending["expires_ms"]:
            plan["reason"] = "risk_capacity_or_expiry"
            self._decline(plan["reason"], selected, batch_id=pending["batch_id"])
            return
        equity = self.equity(market, "open")
        target, scale = equity * self.position_fraction, D(1)
        plan["pre_batch_equity"], plan["common_target_notional"] = equity, target
        legs = [
            {
                "symbol": symbol,
                "direction": direction,
                "priority": priority,
                "frozen_decision_budget": pending["decision_budgets"][symbol],
                "frozen_budget_asof_ms": pending["decision_budget_asofs"][symbol],
            }
            for symbol, direction, priority in selected
        ]
        plan["legs"] = legs
        for leg in legs:
            symbol, direction = leg["symbol"], leg["direction"]
            rule = self.rules.get(symbol)
            frozen = pending["decision_budgets"][symbol]
            current = self._budget(symbol, market, "open")
            try:
                raw = self._quote(symbol, market, "open")
            except ValueError:
                raw = None
            leg.update(
                raw_price=raw,
                current_open_budget=current,
                open_budget_asof_ms=market.get(symbol, {}).get("open_budget_asof_ms"),
                rule=rule.evidence() if rule is not None else {"rule_known": False},
            )
            if rule is None or not rule.known or frozen is None or current is None or raw is None:
                plan["reason"] = "unknown_rule_budget_or_open"
                plan["offending_symbol"] = symbol
                self._decline(plan["reason"], selected, batch_id=pending["batch_id"])
                return
            fill = raw * (1 + D(direction) * self.slip)
            desired = target / fill
            cap = min(frozen, current)
            scale = min(scale, cap / (desired * max(raw, fill)), rule.maximum / desired)
            leg.update(
                price=fill,
                desired_quantity=desired,
                effective_budget=cap,
            )
        required_capital = sum(
            (
                row["desired_quantity"]
                * (max(row["raw_price"], row["price"]) + row["price"] * self.fee)
                for row in legs
            ),
            ZERO,
        )
        scale = min(scale, equity / required_capital)
        plan["unscaled_shared_capital_proxy"] = required_capital
        plan["common_scale"] = scale
        for leg in legs:
            rule = self.rules[leg["symbol"]]
            quantity = rule.floor(leg["desired_quantity"] * scale)
            leg.update(
                quantity=quantity,
                unrounded_scaled_quantity=leg["desired_quantity"] * scale,
                minimum_notional_proxy=quantity * min(leg["raw_price"], leg["price"]),
                capacity_charge=quantity * max(leg["raw_price"], leg["price"]),
            )
            if quantity <= 0 or leg["minimum_notional_proxy"] < max(
                rule.min_notional, self.minimum_notional
            ):
                plan["legs"], plan["reason"] = legs, "discrete_minimum_zero_batch"
                self._decline(plan["reason"], selected, batch_id=pending["batch_id"])
                return
            if leg["capacity_charge"] > leg["effective_budget"]:
                raise AssertionError("离散数量不能突破共同缩放后的预算")
        plan["legs"], plan["status"] = legs, "FILLED"
        plan["signed_fill_notional_residual"] = sum(
            (D(row["direction"]) * row["quantity"] * row["price"] for row in legs), ZERO
        )
        plan["signed_raw_notional_residual"] = sum(
            (D(row["direction"]) * row["quantity"] * row["raw_price"] for row in legs), ZERO
        )
        for leg in legs:
            symbol, quantity, price = leg["symbol"], leg["quantity"], leg["price"]
            fee = quantity * price * self.fee
            cycle = f"{pending['batch_id']}:{symbol}"
            self.cash -= fee
            self._used_budget[symbol] = self._used_budget.get(symbol, ZERO) + leg["capacity_charge"]
            self.positions[symbol] = {
                "cycle_id": cycle,
                "episode": pending["batch_id"],
                "entry_ms": stamp,
                "signal_ms": pending["signal_ms"],
                "direction": leg["direction"],
                "quantity": quantity,
                "original_quantity": quantity,
                "entry_price": price,
                "entry_fee": fee,
                "funding": ZERO,
                "gross_pnl_reduced": ZERO,
                "exit_fees": ZERO,
                "exit_notional": ZERO,
                "reduced_quantity": ZERO,
                "first_exit_ms": None,
                "reduction_fill_indices": [],
            }
            self.fills.append(
                {
                    **leg,
                    "cycle_id": cycle,
                    "clock": stamp,
                    "processing_clock_ms": self._clock,
                    "phase": "open",
                    "action": "OPEN",
                    "signal_ms": pending["signal_ms"],
                    "fee": fee,
                    "quantity_before": ZERO,
                    "qty_remaining": quantity,
                    "quantity_after": quantity,
                    "pre_batch_equity": equity,
                    "batch_id": pending["batch_id"],
                    "historical_mark_used": False,
                }
            )

    def _schedule(self, clock, market, proposed):
        if not proposed or self.halted or self.positions or self.pending is not None:
            return
        if (
            len(proposed) not in {2, 4}
            or len(proposed) > PROTOCOL["max_positions"]
            or len({row[0] for row in proposed}) != len(proposed)
        ):
            raise ValueError("入场必须是两腿或四腿完整固定批次")
        if sum(row[1] for row in proposed) != 0:
            raise ValueError("入场须为双向等腿")
        selected = []
        budgets, asofs = {}, {}
        for symbol, direction, priority in proposed:
            if (
                not isinstance(symbol, str)
                or not symbol
                or type(direction) is not int
                or direction not in {-1, 1}
                or decimal(priority) < decimal(PREDICTED_RETURN_THRESHOLD)
            ):
                raise ValueError("候选方向或16bps优先级不合法")
            budgets[symbol] = self._budget(symbol, market, "decision")
            asofs[symbol] = market.get(symbol, {}).get("decision_budget_asof_ms")
            selected.append((symbol, direction, priority))
        batch = f"{self.factor}:batch:{clock}:{len(self.entry_plans)}"
        self.pending = {
            "batch_id": batch,
            "signal_ms": clock,
            "available_ms": clock + MINUTE,
            "expires_ms": clock + int(PROTOCOL["max_signal_age_ms"]),
            "selected": selected,
            "decision_budgets": budgets,
            "decision_budget_asofs": asofs,
        }
        self.entry_plans.append(
            {
                "event": "SIGNAL_FROZEN",
                "batch_id": batch,
                "clock": clock,
                "processing_clock_ms": clock,
                "phase": "close",
                "selected": selected,
                "decision_budgets": budgets,
                "decision_budget_asofs": asofs,
            }
        )

    def _record(self, clock, market, equity, opening_equity):
        reconstructed = (
            self.initial
            + self.closed_net
            + sum(
                (
                    p["gross_pnl_reduced"]
                    - p["entry_fee"]
                    - p["exit_fees"]
                    + p["funding"]
                    + D(p["direction"])
                    * p["quantity"]
                    * (self._quote(symbol, market, "close") - p["entry_price"])
                    for symbol, p in self.positions.items()
                ),
                ZERO,
            )
        )
        if abs(reconstructed - equity) > D("1e-40") * max(abs(equity), D(1)):
            raise AssertionError("完整腿、残仓、费用和资金费不守恒")
        self.curve.append(
            {
                "clock": clock,
                "equity": equity,
                "cash": self.cash,
                "opening_equity": opening_equity,
                "positions": len(self.positions),
                "gross_notional": sum(
                    (
                        p["quantity"] * self._quote(symbol, market, "close")
                        for symbol, p in self.positions.items()
                    ),
                    ZERO,
                ),
            }
        )

    def step(self, clock, market, proposed):
        """先结算旧库存同钟现金，再按开盘削仓/入场，最后结算收盘前现金。"""
        if self.invalid is not None or self.finished:
            return False
        self._clock = clock
        try:
            clock = integer(clock, "完成边界")
            if (
                clock < MINUTE
                or clock % MINUTE
                or clock > self.end_ms
                or (self.last_clock is not None and clock != self.last_clock + MINUTE)
            ):
                raise ValueError("账户须按完整连续分钟运行至固定期末")
            self._clock, self._used_budget = clock, {}
            stamp = clock - MINUTE
            with localcontext() as context:
                context.prec = PRECISION
                self._settle(stamp, True)
                opening = self.equity(market, "open")
                self.maximum_open_drawdown = max(
                    self.maximum_open_drawdown, 1 - opening / self.peak
                )
                self._halt(opening, stamp)
                for symbol, position in list(self.positions.items()):
                    if stamp - position["entry_ms"] >= self.hold:
                        due = position["entry_ms"] + self.hold
                        prior = self.closes.get(symbol)
                        # 同钟新风控不得推迟已成熟的到期退出，原请求仍留证据。
                        if prior is None or due < prior["available_ms"]:
                            self.closes[symbol] = {"available_ms": due, "reason": "time"}
                            if prior is not None:
                                self.closes[symbol]["superseded_exit_request"] = prior
                    requested = self.closes.get(symbol)
                    if requested is not None and requested["available_ms"] <= stamp:
                        self._reduce(symbol, market, stamp, requested)
                self._open_batch(market, stamp)
                self._settle(clock, False)
                equity = self.equity(market, "close")
                self.peak = max(self.peak, equity)
                self.max_dd = max(self.max_dd, 1 - equity / self.peak)
                self._halt(equity, clock)
                if clock != self.end_ms:
                    self._schedule(clock, market, proposed)
                self._record(clock, market, equity, opening)
            self.last_clock = clock
            if clock == self.end_ms:
                self.finished = True
                if self.pending is not None:
                    self._decline("end_pending_cancelled", self.pending["selected"])
                    self.pending = None
                if self.positions:
                    self._fail("RESIDUAL_INVENTORY_AT_END", "期末仍有真实残仓，不能强造预算外平仓")
            return self.invalid is None
        except (ValueError, KeyError, TypeError, ArithmeticError, AssertionError) as error:
            self._fail(type(error).__name__, str(error))
            return False

    def metrics(self):
        """失败或残仓不产生有效数字结论；部分fill不增加完整腿样本。"""
        if self.invalid is not None or not self.finished or self.positions or not self.curve:
            return None
        with localcontext() as context:
            context.prec = PRECISION
            count = len(self.trades)
            wins = sum(row["net_pnl"] > 0 for row in self.trades)
            profit = sum((max(row["net_pnl"], ZERO) for row in self.trades), ZERO)
            loss = -sum((min(row["net_pnl"], ZERO) for row in self.trades), ZERO)
            p, z = (wins / count if count else 0), 1.96
            wilson = (
                (
                    (
                        p
                        + z * z / (2 * count)
                        - z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count))
                    )
                    / (1 + z * z / count)
                )
                if count
                else 0
            )
            batches = {}
            for row in self.trades:
                batches[row["episode"]] = batches.get(row["episode"], ZERO) + row["net_pnl"]
            return {
                "net_return": float(self.curve[-1]["equity"] / self.initial - 1),
                "ending_equity": float(self.curve[-1]["equity"]),
                "maximum_close_drawdown": float(self.max_dd),
                "maximum_open_drawdown": float(self.maximum_open_drawdown),
                "closed_trades": count,
                "win_rate": wins / count if count else None,
                "wilson_lower": wilson,
                "profit_factor": float(profit / loss) if loss else None,
                "complete_batches": len(batches),
                "positive_batches": sum(v > 0 for v in batches.values()),
                "funding_cash": float(sum((r["cash"] for r in self.funding_ledger), ZERO)),
                "open_positions": 0,
                "halted": self.halted,
                "funding_included": True,
                "numeric_conclusions_allowed": True,
                "minimum_notional_uses_raw_price_proxy": True,
                "historical_mark_available": False,
                "market_average_fill_requires_tick_grid": False,
                "historical_rule_pti_proven": False,
            }
