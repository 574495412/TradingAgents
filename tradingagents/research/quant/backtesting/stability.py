"""保留旧三十日前向方案的可选复核，不作为当前历史量化验收的必要条件。

用户于2026-10-16取消强制真实前向要求。当前历史滚动路线见
historical_rolling_plan.md 及 forecast 的 historical_assessment；旧实验保持可重算。
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal, localcontext

import numpy as np

from tradingagents.dataflows.research.live_capture import MINUTE, digest
from tradingagents.research.quant.backtesting.bidirectional import HYPOTHESIS
from tradingagents.research.quant.backtesting.paper_inputs import covers, number

DAY = 24 * 60 * MINUTE
DAYS = 30
BLOCK_DAYS = 5
REPLICATES = 10_000
SEED = 20261016
INITIAL = Decimal("1000")


def block_interval(daily_pnl_over_initial):
    """固定五日循环块、六块一组和一万次重采样，不提供事后选参接口。"""
    values = np.asarray(daily_pnl_over_initial, dtype=np.float64)
    if values.shape != (DAYS,) or not np.isfinite(values).all():
        raise ValueError("统计输入必须为完整三十日有限净损益序列")
    rng = np.random.Generator(np.random.PCG64(SEED))
    starts = rng.integers(0, DAYS, size=(REPLICATES, DAYS // BLOCK_DAYS))
    indices = (starts[:, :, None] + np.arange(BLOCK_DAYS)) % DAYS
    totals = values[indices].sum(axis=(1, 2))
    lower, upper = np.quantile(totals, [0.025, 0.975], method="linear")
    return {
        "lower": float(lower),
        "upper": float(upper),
        "block_days": BLOCK_DAYS,
        "replicates": REPLICATES,
        "seed": SEED,
        "generator": "PCG64",
        "quantile_method": "linear",
        "numpy_version": np.__version__,
    }


def funding_complete(state):
    """全部库存已结算且每笔寿命有首轮与稍后两次完整区间证明。"""
    if state["positions"]:
        return False
    for trade in state["trades"]:
        windows = state["funding_windows"][trade["symbol"]]
        early = [w for w in windows if w["requested_ms"] <= trade["exit_ms"] + MINUTE]
        late = [w for w in windows if w["requested_ms"] >= trade["exit_ms"] + MINUTE + 1000]
        if not covers(early, trade["entry_ms"] + 1, trade["exit_ms"]) or not covers(
            late, trade["entry_ms"] + 1, trade["exit_ms"]
        ):
            return False
    return True


def daily_equities(curve, start, end, final_cash):
    """日界使用界限前最后已观察权益；最后一天包含全部期末退出尾部费用。"""
    if not curve or curve[0]["actual_ms"] < start or curve[0]["actual_ms"] > start + 2 * MINUTE:
        raise ValueError("阶段起点没有及时的账户观察")
    if not end <= curve[-1]["actual_ms"] <= end + 3 * MINUTE:
        raise ValueError("阶段尚未结束或期末退出尾部超出上限")
    previous, previous_close = None, None
    for row in curve:
        now, closing = row["actual_ms"], row["canonical_close_ms"]
        if (
            type(now) is not int
            or type(closing) is not int
            or closing % MINUTE
            or not closing + 1000 <= now < closing + 2 * MINUTE
            or (
                previous is not None
                and (
                    not previous < now <= previous + 2 * MINUTE
                    or closing - previous_close not in (0, MINUTE)
                )
            )
            or row["equity"] is None
        ):
            raise ValueError("账户曲线存在时钟、缺价或观测断层")
        number(row["equity"])
        previous, previous_close = now, closing
    equities, clocks, cursor = [INITIAL], [], 0
    for day in range(1, DAYS):
        boundary = start + day * DAY
        while cursor + 1 < len(curve) and curve[cursor + 1]["actual_ms"] < boundary:
            cursor += 1
        row = curve[cursor]
        if not 0 < boundary - row["actual_ms"] <= 2 * MINUTE:
            raise ValueError("日界前缺少及时且实际已经知道的权益")
        equities.append(number(row["equity"]))
        clocks.append(row["actual_ms"])
    equities.append(final_cash)
    clocks.append(curve[-1]["actual_ms"])
    return equities, clocks


def assess_phase(state, *, start_ms, end_ms, ledger_identity):
    """只计算已核验账本的经济必要条件，独立数据与运行证明仍须另行完成。"""
    if (
        type(start_ms) is not int
        or type(end_ms) is not int
        or start_ms % DAY
        or end_ms - start_ms != DAYS * DAY
        or not isinstance(ledger_identity, str)
        or not ledger_identity
    ):
        raise ValueError("阶段必须有身份和固定UTC三十日区间")
    with localcontext() as context:
        context.prec = 40
        return _assess(state, start_ms, end_ms, ledger_identity)


def _assess(state, start, end, identity):
    protocol = state["protocol"]
    reasons = []
    expected = {
        "hypothesis": HYPOTHESIS,
        "execution_version": "received_quote_paper_v1",
        "slippage_bps_each_side": 10,
        "fee_bps_each_side": 5,
        "initial_equity": "1000",
        "position_fraction": "0.125",
        "maximum_positions": 4,
        "maximum_gross_equity_ratio": 1,
        "hold_ms": 20 * MINUTE,
        "take_profit_fraction": "0.01",
        "maximum_drawdown": "0.08",
        "orders_enabled": False,
        "production_eligible": False,
    }
    if any(protocol.get(k) != value for k, value in expected.items()):
        reasons.append("PRESSURE_ACCOUNT_CONTRACT_MISMATCH")
    if protocol["period_end_ms"] != end:
        reasons.append("REGISTERED_PERIOD_MISMATCH")
    if state["positions"] or state["pending"]:
        reasons.append("UNFINISHED_INVENTORY_OR_ENTRY")
    if state["quality_errors"]:
        reasons.append("ACCOUNT_QUALITY_FAILURE")
    costs_complete = funding_complete(state)
    if not costs_complete:
        reasons.append("FUNDING_COVERAGE_NOT_RECHECKED")
    settled = not state["positions"] and not state["pending"] and costs_complete
    trades, cash = state["trades"], number(state["cash"])
    episodes = {t["episode"] for t in trades}
    if len(episodes) < 30:
        reasons.append("FEWER_THAN_THIRTY_COMPLETE_ACTIVITIES")
    if len(episodes) != len(trades):
        reasons.append("ACTIVITY_REUSED")
    if any(
        not start <= t["entry_ms"] < end or not t["entry_ms"] < t["exit_ms"] <= end + 3 * MINUTE
        for t in trades
    ):
        reasons.append("TRADE_OUTSIDE_REGISTERED_PHASE")
    nets = [number(t["net_pnl"]) for t in trades]
    if not state["positions"] and abs(cash - INITIAL - sum(nets, Decimal(0))) > Decimal("1e-28"):
        reasons.append("CLOSED_CASH_NOT_CONSERVED")
    drawdown = number(state["max_dd"])
    if not 0 <= drawdown < Decimal("0.08"):
        reasons.append("DRAWDOWN_GATE_FAILED")
    if state["event_count"] != len(state["curve"]):
        reasons.append("ACCOUNT_PREFIX_INCOMPLETE")
    try:
        if not settled:
            raise ValueError("库存或费用尚未完整结算")
        if (
            state["curve"]
            and state["curve"][-1]["equity"] is not None
            and (
                number(state["curve"][-1]["equity"]) != cash
                or number(state["curve"][-1]["cash"]) != cash
            )
        ):
            raise ValueError("期末空仓权益与现金不一致")
        equities, observed = daily_equities(state["curve"], start, end, cash)
        daily = [(equities[i + 1] - equities[i]) / INITIAL for i in range(DAYS)]
        if abs(sum(daily, Decimal(0)) - (cash / INITIAL - 1)) > Decimal("1e-28"):
            raise ValueError("每日净损益未包含完整期末成本")
        interval = block_interval([float(v) for v in daily])
    except ValueError as exc:
        daily, observed, interval = [], [], None
        reasons.append("DAILY_SERIES_UNAVAILABLE: " + str(exc))
    net_return = cash / INITIAL - 1 if settled else None
    if net_return is None or net_return <= 0:
        reasons.append("PRESSURE_RETURN_NOT_POSITIVE")
    if interval is None or interval["lower"] <= 0:
        reasons.append("BLOCK_INTERVAL_LOWER_NOT_POSITIVE")
    profit = sum((max(v, Decimal(0)) for v in nets), Decimal(0))
    loss = -sum((min(v, Decimal(0)) for v in nets), Decimal(0))
    diagnostic = {"entry_utc_week": {}, "direction": {}, "symbol": {}}
    for trade, net in zip(trades, nets, strict=True):
        calendar = datetime.fromtimestamp(trade["entry_ms"] / 1000, UTC).isocalendar()
        for name, key in (
            ("entry_utc_week", f"{calendar.year}-W{calendar.week:02d}"),
            ("direction", str(trade["direction"])),
            ("symbol", trade["symbol"]),
        ):
            row = diagnostic[name].setdefault(key, {"closed_trades": 0, "net_pnl_usdt": Decimal(0)})
            row["closed_trades"] += 1
            row["net_pnl_usdt"] += net
    return {
        "schema": "cz-frozen-phase-economic-conditions.v1",
        "start_ms": start,
        "end_ms": end,
        "ledger_identity": identity,
        "account_protocol_sha256": digest(
            {k: v for k, v in protocol.items() if k != "period_end_ms"}
        ),
        "net_return": str(net_return) if net_return is not None else None,
        "closed_trades": len(trades),
        "complete_initial_activities": len(episodes),
        "net_win_rate": sum(v > 0 for v in nets) / len(nets) if nets and settled else None,
        "profit_factor": str(profit / loss) if loss and settled else None,
        "costs_and_inventory_fully_settled": settled,
        "maximum_observed_drawdown": str(drawdown),
        "funding_cash_usdt": str(sum((number(v["cash"]) for v in state["payments"]), Decimal(0))),
        "daily_pnl_over_initial": [str(v) for v in daily],
        "daily_observation_clocks": observed,
        "block_interval": interval,
        "diagnostic_groups": {
            name: {
                k: {**row, "net_pnl_usdt": str(row["net_pnl_usdt"])} for k, row in groups.items()
            }
            for name, groups in diagnostic.items()
        },
        "economic_conditions_met": not reasons,
        "reasons": reasons,
        "source_and_runtime_proofs_assessed_here": False,
        "stable_profitability_proven": False,
        "production_eligible": False,
        "orders_enabled": False,
    }


def compare_phases(first, second):
    """两段分别通过、连续且独立，禁止拼接赢家或用同一账本复验自己。"""
    reasons = []
    if not first["economic_conditions_met"] or not second["economic_conditions_met"]:
        reasons.append("BOTH_PHASES_MUST_PASS_SEPARATELY")
    if (
        first["end_ms"] != second["start_ms"]
        or first["ledger_identity"] == second["ledger_identity"]
    ):
        reasons.append("PHASES_NOT_CONTIGUOUS_AND_INDEPENDENT")
    if first["account_protocol_sha256"] != second["account_protocol_sha256"]:
        reasons.append("ACCOUNT_RULES_CHANGED_BETWEEN_PHASES")
    if first["block_interval"] is not None and second["block_interval"] is not None:
        keys = ("block_days", "replicates", "seed", "generator", "quantile_method", "numpy_version")
        if any(first["block_interval"][k] != second["block_interval"][k] for k in keys):
            reasons.append("BOOTSTRAP_IMPLEMENTATION_CHANGED")
    return {
        "economic_replication_conditions_met": not reasons,
        "reasons": reasons,
        "stable_profitability_proven": False,
        "source_and_runtime_proofs_still_required": True,
        "production_eligible": False,
        "orders_enabled": False,
    }
