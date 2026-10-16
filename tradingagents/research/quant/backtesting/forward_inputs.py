"""从真实已收到的公开分钟构造共同截面，并在收到依赖后检验固定基差条件。"""

from __future__ import annotations

import math
from decimal import Decimal
from statistics import median

import numpy as np
import pandas as pd

from tradingagents.dataflows.research.live_capture import MINUTE, available_rows, body
from tradingagents.research.quant.backtesting.basis import BASIS_LIMIT
from tradingagents.research.quant.backtesting.search import COLUMNS
from tradingagents.research.quant.reference import market_features as market

HISTORY_BARS = 300
REQUIRED_COMPLETE_BARS = 62
RATIOS = ("global_ratio", "top_account_ratio", "top_position_ratio")


def normalized_row(row):
    """按官方十二字段验证终值；数值表示差异不算冲突，价格差异必须拒绝。"""
    if (
        len(row) != 12
        or type(row[0]) is not int
        or type(row[6]) is not int
        or row[0] % MINUTE
        or row[6] != row[0] + MINUTE - 1
        or type(row[8]) is not int
        or row[8] < 0
    ):
        raise ValueError("公开分钟格式或时钟不合法")
    values = {k: Decimal(str(row[k])) for k in (1, 2, 3, 4, 5, 7, 9, 10)}
    if any(not v.is_finite() or not math.isfinite(float(v)) for v in values.values()):
        raise ValueError("公开分钟必须包含有限数值")
    if min(values[k] for k in (1, 2, 3, 4)) <= 0 or min(values[k] for k in (5, 7, 9, 10)) < 0:
        raise ValueError("公开分钟价格或成交量不合法")
    if (
        values[2] < max(values[1], values[4])
        or values[3] > min(values[1], values[4])
        or values[9] > values[5]
        or values[10] > values[7]
    ):
        raise ValueError("公开分钟范围或主动成交量不守恒")
    return tuple(row[k] if k in (0, 6, 8) else str(values[k].normalize()) for k in range(11))


class Bars:
    """保留首次真实接收时刻，重复原文不能回写已经封存的决策。"""

    def __init__(self, symbols):
        self.values = {s: {} for s in symbols}
        self.latest_close = 0

    def add(self, symbol, row, *, received_ms, source_sha256, completed_verified=False):
        if (
            not completed_verified
            or symbol not in self.values
            or type(received_ms) is not int
            or received_ms < row[6] + 1 - 1000
        ):
            raise ValueError("分钟不属于固定集合或缺少接收时钟")
        normalized = normalized_row(row)
        closing = row[6] + 1
        existing = self.values[symbol].get(closing)
        if existing and existing["normalized"] != normalized:
            raise ValueError(f"同一分钟出现不同终值：{symbol}:{closing}")
        if existing and existing["received_ms"] <= received_ms:
            return
        self.values[symbol][closing] = {
            "row": row,
            "normalized": normalized,
            "received_ms": received_ms,
            "source_sha256": source_sha256,
        }
        self.latest_close = max(self.latest_close, closing)

    def prune(self, closing):
        earliest = closing - (HISTORY_BARS - 1) * MINUTE
        for symbol in self.values:
            self.values[symbol] = {t: v for t, v in self.values[symbol].items() if t >= earliest}

    def at(self, symbol, closing, decision_ms):
        value = self.values[symbol].get(closing)
        return value if value and value["received_ms"] <= decision_ms else None


def venue_contracts(record, fixed, *, decision_ms, maximum_age_ms=5 * MINUTE):
    """只使用已到达且未过时的交易所身份；集合保持固定，未知成员不能静默删除。"""
    if (
        record.get("status") != 200
        or not record.get("local_clock_consistent")
        or not 0 <= decision_ms - record["received_ms"] <= maximum_age_ms
    ):
        return None
    rows = body(record).get("symbols")
    if not isinstance(rows, list):
        raise ValueError("公开合约身份缺少列表")
    selected = {}
    for row in rows:
        symbol = row["symbol"]
        if symbol not in fixed:
            continue
        if symbol in selected:
            raise ValueError("公开合约身份重复")
        selected[symbol] = {
            "asset_class": "TRADIFI_PERPETUAL"
            if row["contractType"] == "TRADIFI_PERPETUAL"
            else "CRYPTO_PERPETUAL",
            "onboard_ms": int(row["onboardDate"]),
            "delivery_ms": int(row["deliveryDate"]),
            "trading": row["status"] == "TRADING"
            and row["quoteAsset"] == "USDT"
            and row["contractType"] in ("PERPETUAL", "TRADIFI_PERPETUAL"),
        }
        if selected[symbol]["asset_class"] != fixed[symbol]["asset_class"]:
            raise ValueError("固定合约的资产类别身份发生变化")
    return selected if set(selected) == set(fixed) else None


def snapshot(bars, contracts, *, closing, decision_ms):
    """先核验共同分钟覆盖，再复用原生滚动公式和同钟排名，不压缩缺口。"""
    if closing % MINUTE or decision_ms < closing + 1000 or set(contracts) != set(bars.values):
        raise ValueError("共同截面时钟或固定集合不合法")
    clocks = list(range(closing - (HISTORY_BARS - 1) * MINUTE, closing + MINUTE, MINUTE))
    index = pd.to_datetime(clocks, unit="ms", utc=True).rename("bar_time")
    outputs = [c for c in COLUMNS if c in market.PANEL_COLUMNS]
    calculations = market._calculation_columns(outputs)
    data, missing = [], []
    for symbol, contract in contracts.items():
        rows, valid = [], []
        for clock in clocks:
            eligible = (
                contract.get("trading", True)
                and clock - MINUTE >= contract["onboard_ms"]
                and (not contract["delivery_ms"] or clock - MINUTE < contract["delivery_ms"])
            )
            value = bars.at(symbol, clock, decision_ms)
            observed = eligible and value is not None
            if (
                eligible
                and not observed
                and clock >= closing - (REQUIRED_COMPLETE_BARS - 1) * MINUTE
            ):
                missing.append((symbol, clock))
            row = value["row"] if observed else None
            rows.append([float(row[k]) for k in (1, 2, 3, 4, 5, 7, 10)] if row else [np.nan] * 7)
            valid.append((eligible, observed))
        frame = pd.DataFrame(rows, index=index, columns=market.RAW_COLUMNS)
        featured = market._build_symbol_features(frame, required_columns=frozenset(calculations))
        value = featured.iloc[-1].to_dict() | {
            "bar_time": index[-1],
            "symbol": symbol,
            "asset_class": contract["asset_class"],
            "eligible": valid[-1][0],
            "observed": valid[-1][1],
        }
        data.append(value)
    if missing:
        return None, {
            "reason": "INCOMPLETE_CAUSAL_CALENDAR",
            "missing_count": len(missing),
            "examples": missing[:10],
        }
    result = pd.DataFrame(data).set_index(["bar_time", "symbol"]).sort_index()
    for target, source in {
        "ret_20_rank_pct": "ret_20_pct",
        "ret_60_rank_pct": "ret_60_pct",
        "atr_rank_pct": "atr_pct",
        "volume_ratio_rank_pct": "quote_volume_ratio_30",
        "liquidity_rank_pct": "quote_volume_sum_12",
    }.items():
        result[target] = market.cross_sectional_rank.calculate(
            result[source].where(result["eligible"])
        )
    result["ret_20_residual_pct"] = market.residual_momentum.calculate(
        result["ret_20_pct"].where(result["eligible"])
    )
    return result.loc[:, COLUMNS].droplevel("bar_time"), {"reason": "READY", "members": len(result)}


def price_ready(frame):
    """该门槛与已冻结急涨子机制相同，只决定公开依赖，不读取后续结果。"""
    mask = (
        frame["eligible"]
        & frame["observed"]
        & frame["asset_class"].eq("TRADIFI_PERPETUAL")
        & frame["ret_20_pct"].ge(1.5)
        & frame["ret_60_pct"].ge(2)
        & frame[["ret_20_rank_pct", "ret_60_rank_pct"]].max(axis=1).ge(0.985)
        & frame["atr_rank_pct"].ge(0.93)
    )
    return sorted(frame.index[mask])


def dependency_jobs(symbols, closing):
    """两个滞后偏离各需此前六十分钟，指数一次取得六十二根完整终值。"""
    jobs = []
    for symbol in sorted(symbols):
        jobs.extend(
            (kind, symbol, {"symbol": symbol, "period": "5m", "limit": 3}) for kind in RATIOS
        )
        jobs.append(
            (
                "index",
                symbol,
                {
                    "pair": symbol,
                    "interval": "1m",
                    "startTime": closing - 63 * MINUTE,
                    "endTime": closing - MINUTE - 1,
                    "limit": 62,
                },
            )
        )
    return jobs


def evaluate_dependencies(bars, records, symbol, *, closing, decision_ms, clock_verified):
    """数据晚到、缺失或口径不全时不授权；数值正负不能覆盖时序失败。"""
    matching = [r for r in records if r["symbol"] == symbol]
    selected = {r["kind"]: r for r in matching}
    if set(selected) != set(RATIOS) | {"index"} or len(matching) != len(selected):
        return {"ready": False, "reason": "DEPENDENCY_SET_INCOMPLETE"}
    if not clock_verified or any(not r.get("local_clock_consistent") for r in matching):
        return {"ready": False, "reason": "CLOCK_UNVERIFIED"}
    if any(selected[k]["received_ms"] > decision_ms for k in RATIOS):
        return {"ready": False, "reason": "DEPENDENCY_RECEIVED_AFTER_DECISION"}
    if any(selected[k]["status"] != 200 for k in RATIOS):
        return {"ready": False, "reason": "PUBLIC_DEPENDENCY_FAILED"}
    wanted = closing // (5 * MINUTE) * (5 * MINUTE) - 5 * MINUTE
    ratios = []
    for kind in RATIOS:
        rows = available_rows(
            selected[kind],
            decision_ms=decision_ms,
            cutoff_ms=closing,
            clock_verified=clock_verified,
        )
        values = [r for r in rows if r["timestamp"] == wanted]
        if len(values) != 1:
            return {"ready": False, "reason": "EXACT_POSITIONING_OBSERVATION_MISSING"}
        ratio = float(values[0]["longShortRatio"])
        if not math.isfinite(ratio) or ratio <= 0:
            raise ValueError("公开多空比必须为有限正数")
        ratios.append(ratio)
    positioning = {
        "positioning_ready": True,
        "positioning": ratios,
        "positioning_pass": sum(v < 1 for v in ratios) >= 2,
    }
    if selected["index"]["received_ms"] > decision_ms:
        return positioning | {"ready": False, "reason": "DEPENDENCY_RECEIVED_AFTER_DECISION"}
    if selected["index"]["status"] != 200:
        return positioning | {"ready": False, "reason": "OFFICIAL_INDEX_REQUEST_FAILED"}
    rows = available_rows(
        selected["index"],
        decision_ms=decision_ms,
        cutoff_ms=closing - MINUTE,
        clock_verified=clock_verified,
    )
    indices = {r[6] + 1: float(r[4]) for r in rows}
    clocks = list(range(closing - 62 * MINUTE, closing, MINUTE))
    if (
        len(indices) != len(rows)
        or set(indices) != set(clocks)
        or any(bars.at(symbol, t, decision_ms) is None for t in clocks)
    ):
        return positioning | {"ready": False, "reason": "EXACT_INDEX_OR_PERP_HISTORY_MISSING"}
    basis = [float(bars.at(symbol, t, decision_ms)["row"][4]) / indices[t] - 1 for t in clocks]
    older, latest = basis[60] - median(basis[:60]), basis[61] - median(basis[1:61])
    active = len({indices[t] for t in clocks[-5:]}) > 1
    return positioning | {
        "ready": True,
        "reason": "READY",
        "older_basis_delta": older,
        "latest_basis_delta": latest,
        "index_active": active,
        "basis_short_pass": active
        and older > 0
        and latest > 0
        and min(older, latest) >= BASIS_LIMIT,
        "latest_received_ms": max(r["received_ms"] for r in selected.values()),
        "actual_decision_ms": decision_ms,
        "canonical_close_ms": closing,
    }
