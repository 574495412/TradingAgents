"""显式多空预测调度的连续分钟驱动器，不发现模型、来源或交易品种。"""

import math
from collections import Counter
from numbers import Integral

import numpy as np
import pandas as pd

from .account import Account
from .forecast import PREDICTED_RETURN_THRESHOLD, cancel_incomplete_batch
from .search import Audit
from .settlement import blocks
from .signals import MINUTE, Signal


def select_pairs(frame, score_column):
    """同原规则：双向各最多两腿、16bps、按分数及合约名稳定排序。"""
    if frame.symbol.duplicated().any():
        raise ValueError("配对预测截面存在重复合约")
    selected = frame[["symbol", score_column]].rename(columns={score_column: "score"})
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


def _clock(value):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or value < 0
        or value % MINUTE
    ):
        raise ValueError("分钟驱动时钟须为非负、对齐的整数毫秒")
    return int(value)


def _validate(cases, schedules, boundaries):
    boundaries = tuple(_clock(value) for value in boundaries)
    if (
        not cases
        or len(boundaries) < 2
        or any(b <= a for a, b in zip(boundaries, boundaries[1:], strict=False))
    ):
        raise ValueError("账户和严格递增的连续分块边界不能为空")
    groups = set()
    for value in cases.values():
        if len(value) != 4 or any(not isinstance(v, str) or not v for v in value[:3]):
            raise ValueError("案例须明确方式、资产、模型及滑点")
        cost = value[3]
        if (
            isinstance(cost, (bool, np.bool_))
            or not isinstance(cost, (int, float))
            or not math.isfinite(cost)
            or cost < 0
        ):
            raise ValueError("滑点须为非负有限数")
        groups.add(tuple(value[:3]))
    for key, selected in schedules.items():
        if len(key) != 4 or tuple(key[:3]) not in groups:
            raise ValueError("调度不属于明确案例")
        clock = _clock(key[3])
        if not boundaries[0] <= clock < boundaries[-1]:
            raise ValueError("决策调度越出账户区间")
        if len(selected) not in {0, 2, 4} or len({row[0] for row in selected}) != len(selected):
            raise ValueError("配对批次须为不同合约的两腿或四腿")
        for symbol, direction, priority in selected:
            if (
                not isinstance(symbol, str)
                or not symbol
                or type(direction) is not int
                or direction not in {-1, 1}
                or not math.isfinite(priority)
                or priority < PREDICTED_RETURN_THRESHOLD
            ):
                raise ValueError("配对方向、合约或优先级不合法")
        if sum(row[1] for row in selected) != 0 or list(selected) != sorted(
            selected, key=lambda row: (-row[1], -row[2], row[0])
        ):
            raise ValueError("配对批次必须双向等腿且顺序固定")
    return boundaries


def drive_accounts(
    cases,
    schedules,
    market_frames,
    funding,
    boundaries,
    *,
    holding_minutes=20,
    timed_exit_at_close=True,
):
    """每案例只创建一次账户；失败留下部分账本，其它案例继续。

    market_frames 显式提供 (完整分钟完成边界, 原始市场字典)，不允许缺帧。
    settlement.blocks 只复用其通用账本分块汇总，不调用旧 H01 run/决策/系数。
    """
    if (
        isinstance(holding_minutes, (bool, np.bool_))
        or not isinstance(holding_minutes, Integral)
        or not 0 < holding_minutes <= np.iinfo(np.int64).max // MINUTE
        or type(timed_exit_at_close) is not bool
    ):
        raise ValueError("持有分钟须为正整数，收盘到期选项须为明确布尔值")
    boundaries = _validate(cases, schedules, boundaries)
    first, last = boundaries[0], boundaries[-1]
    accounts = {
        key: Account(
            model,
            int(holding_minutes),
            cost,
            funding,
            take_profit_fraction=None,
            timed_exit_at_close=timed_exit_at_close,
        )
        for key, (_, _, model, cost) in cases.items()
    }
    audits = {key: Audit(account, funding) for key, account in accounts.items()}
    states, invalid, opportunities = {key: {} for key in cases}, {}, Counter()
    block_ends = {value - MINUTE for value in boundaries[1:-1]} | {last}
    previous, frames, global_failure = None, 0, None
    try:
        for clock, market in market_frames:
            clock = _clock(clock)
            if clock != (first if previous is None else previous + MINUTE) or clock > last:
                raise ValueError("完整账户时间轴起点、缺口、重复或终点不符")
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
                        row
                        for row in account.pending.values()
                        if row["available_ms"] <= clock - MINUTE
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
                            Signal(symbol, direction, score, clock, f"paired_forecast:{clock}")
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
                except Exception as error:
                    invalid[key] = {
                        "at": clock,
                        "type": type(error).__name__,
                        "reason": str(error),
                        "numeric_conclusions_allowed": False,
                    }
            previous, frames = clock, frames + 1
        if previous != last or frames != (last - first) // MINUTE + 1:
            raise ValueError("账户没有覆盖完整预定分钟区间")
    except Exception as error:
        global_failure = {
            "at": previous,
            "type": type(error).__name__,
            "reason": str(error),
            "numeric_conclusions_allowed": False,
        }
        for key in cases:
            invalid.setdefault(
                key, {**global_failure, "reason": "MARKET_INPUT_FAILURE: " + str(error)}
            )
    metrics, proofs = {}, {}
    for key, account in accounts.items():
        if key in invalid:
            continue
        try:
            proofs[key] = audits[key].finish()
            summary = account.metrics()
            batches = {}
            for trade in account.trades:
                batches[trade["episode"]] = batches.get(trade["episode"], 0.0) + trade["net_pnl"]
            summary["complete_batches"] = len(batches)
            summary["positive_batches"] = sum(value > 0 for value in batches.values())
            method, asset, model, _ = cases[key]
            clocks = [
                clock
                for (m, a, n, clock), selected in schedules.items()
                if (m, a, n) == (method, asset, model) and selected
            ]
            summary["historical_blocks"] = blocks(account, boundaries, clocks, states[key])
            metrics[key] = summary
        except Exception as error:
            proofs.pop(key, None)
            invalid[key] = {
                "at": last,
                "type": type(error).__name__,
                "reason": str(error),
                "numeric_conclusions_allowed": False,
            }
    return accounts, {
        "frames": frames,
        "cases": len(cases),
        "metrics": metrics,
        "financial_proofs": proofs,
        "invalid_accounts": invalid,
        "opportunity_clocks": {key: opportunities[key] for key in cases},
        "global_failure": global_failure,
    }
