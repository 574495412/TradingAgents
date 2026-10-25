"""H04既有费率特征与等基础数量对冲标签；未来现金不进入特征。"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

MINUTE = 60000
HOUR = 60 * MINUTE
DAY = 24 * HOUR
DELAY = 5 * MINUTE
CAPITAL = 1000.0
SPOT_FEE, PERP_FEE = 0.001, 0.0005
SLIPPAGES = (3, 10, 11)


def feature(events, clock, start, end):
    """使用左开右闭、延迟五分钟的完整历史事件窗。"""
    left, right = clock - DAY - DELAY, clock - DELAY
    if left < start or right >= end:
        return math.nan, "FEATURE_WINDOW_OUTSIDE_SOURCE"
    rates = [
        float(e["rate"]) for e in events if e["type"] == "Regular" and left < e["time_ms"] <= right
    ]
    return -math.fsum(rates), "KNOWN_UNDER_SOURCE_ASSUMPTIONS"


def cashflows(q, spot_entry, spot_exit, perp_entry, perp_exit, events, a, b, slip):
    """先结算后成交：原毫秒(a,b]事件才由本组库存领取。"""
    held = [e for e in events if a < e["time_ms"] <= b]
    s = slip / 10000
    ask_s, bid_s = spot_entry * (1 + s), spot_exit * (1 - s)
    bid_p, ask_p = perp_entry * (1 - s), perp_exit * (1 + s)
    gross = q / (1 - SPOT_FEE)
    price = q * ((spot_exit - spot_entry) - (perp_exit - perp_entry))
    funding = q * math.fsum(float(e["rate"]) * float(e["mark"]) for e in held)
    slippage = q * s * (spot_entry + spot_exit + perp_entry + perp_exit)
    spot_fee = q * SPOT_FEE * (ask_s / (1 - SPOT_FEE) + bid_s)
    perp_fee = q * PERP_FEE * (bid_p + ask_p)
    return {
        "price_pnl": price,
        "funding_cash": funding,
        "slippage_cost": slippage,
        "spot_fee": spot_fee,
        "perp_fee": perp_fee,
        "borrow_cost": 0.0,
        "spot_gross_quantity": gross,
        "entry_cash_required": gross * ask_s + q * perp_entry + q * PERP_FEE * bid_p,
        "funding_count": len(held),
        "special_count": sum(e["type"] == "Special" for e in held),
        "last_funding_ms": max((e["time_ms"] for e in held), default=None),
        "net_pnl": price + funding - slippage - spot_fee - perp_fee,
    }


def samples(minutes, events, *, start, cutoff, end):
    """保留每小时与每档成本，特征暖机和标签边界分别记录。"""
    rows = []
    for symbol, group in minutes.groupby("symbol", sort=True):
        group = group.set_index("raw_open_ms")
        if not group.index.is_unique:
            raise ValueError("分钟身份重复")
        selected = sorted((e for e in events if e["symbol"] == symbol), key=lambda e: e["time_ms"])

        def price(clock, field, group=group):
            if clock not in group.index:
                return math.nan
            row = group.loc[clock]
            market = field.split("_", 1)[0]
            if row[f"{market}_status"] != "KNOWN_RAW_MINUTE":
                return math.nan
            value = float(row[field])
            return value if math.isfinite(value) and value > 0 else math.nan

        for t in range(start, end, HOUR):
            a, b = t + MINUTE, t + MINUTE + DAY
            x, feature_status = feature(selected, t, start, end)
            st, pt = price(t - MINUTE, "spot_close"), price(t - MINUTE, "perp_close")
            q = CAPITAL / (2 * (st + pt)) if np.isfinite([st, pt]).all() else math.nan
            values = [
                price(a, "spot_open"),
                price(b, "spot_open"),
                price(a, "perp_open"),
                price(b, "perp_open"),
            ]
            for slip in SLIPPAGES:
                record = {
                    "symbol": symbol,
                    "clock_ms": t,
                    "day": pd.Timestamp(t, unit="ms", tz="UTC").date().isoformat(),
                    "slippage_bps": slip,
                    "x": x,
                    "feature_status": feature_status,
                    "entry_ms": a,
                    "exit_ms": b,
                    "q": q,
                    "spot_decision_close": st,
                    "perp_decision_close": pt,
                    **dict(
                        zip(
                            (
                                "spot_entry_open",
                                "spot_exit_open",
                                "perp_entry_open",
                                "perp_exit_open",
                            ),
                            values,
                            strict=True,
                        )
                    ),
                    "label_information_end_ms": b + MINUTE,
                    "y": math.nan,
                    "label_status": "LABEL_OUTSIDE_SOURCE"
                    if b >= end
                    else "PRICE_OR_QUANTITY_UNKNOWN",
                }
                if b < end and np.isfinite([q, *values]).all():
                    cash = cashflows(q, *values, selected, a, b, slip)
                    record.update(cash)
                    last = cash["last_funding_ms"]
                    record["label_information_end_ms"] = max(
                        b + MINUTE, last + DELAY if last is not None else 0
                    )
                    feasible = cash["entry_cash_required"] <= CAPITAL
                    record["label_status"] = (
                        "KNOWN_RESEARCH_LABEL" if feasible else "CAPITAL_INFEASIBLE"
                    )
                    if feasible:
                        record["y"] = cash["net_pnl"] / CAPITAL
                info = record["label_information_end_ms"]
                record["phase"] = (
                    "training"
                    if t < cutoff and info < cutoff
                    else "training_embargo"
                    if t < cutoff
                    else "validation"
                    if info < end
                    else "validation_immature"
                )
                rows.append(record)
    return pd.DataFrame(rows)
