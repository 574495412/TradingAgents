"""容量试验的固定晚期规则与因果成交额来源，不读取标签或执行账户。"""

from __future__ import annotations

import gc
import json
from collections import Counter
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
import pandas as pd

from tradingagents.dataflows.research import historical_panel

from ..artifacts import sha

MINUTE = 60_000
DAY = 86_400_000
RHO = Decimal("0.10")
ELIGIBILITY_RECEIPTS_SHA256 = "2f15b89d59eafd3954a22ace04fa35f4632a918c194e5328f1570aefbf864598"
RULE_FILTERS = ("LOT_SIZE", "MARKET_LOT_SIZE", "MIN_NOTIONAL", "PRICE_FILTER")


def _read(path):
    """严格读取本地JSON，不接受非有限常量。"""

    def reject(value):
        raise ValueError("规则JSON存在非有限常量：" + value)

    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


def _positive(value):
    """源字段必须为字符串，规范串不使用科学计数或二进制浮点舍入。"""
    if not isinstance(value, str):
        raise ValueError("交易所过滤器值必须保留原字符串")
    try:
        number = Decimal(value)
    except InvalidOperation as error:
        raise ValueError("交易所过滤器不是Decimal") from error
    if not number.is_finite() or number <= 0:
        raise ValueError("交易所过滤器必须为有限正Decimal")
    text = format(number, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def parse_rules(symbols, snapshot, *, snapshot_sha256, captured_at_utc):
    """全人口左连接；两lot原点各为minQty，交集由精确数量内核处理。"""
    symbols = tuple(symbols)
    if len(set(symbols)) != len(symbols) or any(not isinstance(s, str) or not s for s in symbols):
        raise ValueError("规则人口须为明确唯一合约，不允许删未知成员")
    rows = snapshot.get("symbols")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("交易所原文缺少完整symbols列表")
    indexed = {}
    for row in rows:
        symbol = row.get("symbol")
        if not isinstance(symbol, str) or symbol in indexed:
            raise ValueError("交易所原文合约身份缺失或重复")
        indexed[symbol] = row
    output = {}
    for symbol in symbols:
        row = indexed.get(symbol)
        rule = {
            "symbol": symbol,
            "rule_known": False,
            "unknown_reason": "MISSING_EXCHANGE_SYMBOL",
            "lots": [],
            "min_notional": None,
            "price_filter": None,
            "raw_filters": deepcopy(row.get("filters")) if row else None,
            "contract_type": row.get("contractType") if row else None,
            "captured_status": row.get("status") if row else None,
            "onboard_ms": row.get("onboardDate") if row else None,
            "delivery_ms": row.get("deliveryDate") if row else None,
            "source_snapshot_sha256": snapshot_sha256,
            "source_captured_at_utc": captured_at_utc,
            "historical_pti_proven": False,
            "min_notional_uses_raw_price_proxy": True,
            "market_average_fill_requires_tick_grid": False,
        }
        if row is not None:
            try:
                if row.get("quoteAsset") != "USDT":
                    raise ValueError("原文quoteAsset不是USDT，不猜测金额单位")
                filters = row.get("filters")
                if not isinstance(filters, list):
                    raise ValueError("原文没有filters列表")
                selected = {}
                for kind in RULE_FILTERS:
                    found = [
                        value
                        for value in filters
                        if isinstance(value, dict) and value.get("filterType") == kind
                    ]
                    if len(found) != 1:
                        raise ValueError("过滤器缺失或重复：" + kind)
                    selected[kind] = found[0]
                lots = []
                for kind in RULE_FILTERS[:2]:
                    value = selected[kind]
                    minimum, maximum, step = (
                        _positive(value.get(key)) for key in ("minQty", "maxQty", "stepSize")
                    )
                    if Decimal(maximum) < Decimal(minimum):
                        raise ValueError("数量最大值小于最小值")
                    lots.append(
                        {
                            "filter_type": kind,
                            "min_qty": minimum,
                            "max_qty": maximum,
                            "step_size": step,
                            "offset": minimum,
                        }
                    )
                price = selected["PRICE_FILTER"]
                price_filter = {
                    name: _positive(price.get(key))
                    for name, key in (
                        ("min_price", "minPrice"),
                        ("max_price", "maxPrice"),
                        ("tick_size", "tickSize"),
                    )
                }
                if Decimal(price_filter["max_price"]) < Decimal(price_filter["min_price"]):
                    raise ValueError("价格最大值小于最小值")
                rule.update(
                    rule_known=True,
                    unknown_reason=None,
                    lots=lots,
                    min_notional=_positive(selected["MIN_NOTIONAL"].get("notional")),
                    price_filter=price_filter,
                )
            except (ValueError, KeyError, TypeError) as error:
                # 缺规则不是已知零容量，也不能使其余909身份消失。
                rule["unknown_reason"] = "INVALID_EXCHANGE_FILTERS: " + str(error)
        output[symbol] = rule
    return output


def load_rules(
    symbols,
    exchange_info_path,
    catalog_binding_path,
    eligibility_result_path,
    eligibility_receipts_path,
    *,
    input_identities,
):
    """只用Oct14既有证据链，绑定四个实际输入并首末核验原字节。"""
    symbols = tuple(symbols)
    if len(symbols) != 909:
        raise ValueError("真实规则视图必须完整保留909人口")
    paths = [
        Path(value)
        for value in (
            exchange_info_path,
            catalog_binding_path,
            eligibility_result_path,
            eligibility_receipts_path,
        )
    ]
    if any(path.is_symlink() for path in paths):
        raise ValueError("规则来源不能为符号链接")
    paths = [path.resolve() for path in paths]
    identities = {str(path): sha(path) for path in paths}
    if identities[str(paths[3])] != ELIGIBILITY_RECEIPTS_SHA256:
        raise ValueError("资格收据不是已封存Oct14来源链")
    for name, value in identities.items():
        if name in input_identities and input_identities[name] != value:
            raise ValueError("实际规则来源与已绑定摘要不同")
    snapshot, binding, eligibility, receipts = map(_read, paths)
    if (
        receipts.get("result.json") != identities[str(paths[2])]
        or eligibility.get("snapshot_sha256") != identities[str(paths[0])]
        or binding.get("sha256") != identities[str(paths[0])]
        or eligibility.get("source_paths_sha256", {}).get(str(paths[0]))
        != identities[str(paths[0])]
        or eligibility.get("source_paths_sha256", {}).get(str(paths[1]))
        != identities[str(paths[1])]
        or eligibility.get("snapshot_captured_at_utc") != binding.get("captured_at_utc")
        or eligibility.get("historical_point_in_time_membership_proven") is not False
    ):
        raise ValueError("规则原文、采集时钟、资格结果与封存收据链不一致")
    rules = parse_rules(
        symbols,
        snapshot,
        snapshot_sha256=identities[str(paths[0])],
        captured_at_utc=binding["captured_at_utc"],
    )
    if any(sha(path) != expected for path, expected in identities.items()):
        raise ValueError("读取期间实际规则来源改变")
    input_identities.update(identities)
    metadata = {
        "schema": "ta-static-capacity-rules.v1",
        "symbols": len(symbols),
        "snapshot_symbols": len(snapshot["symbols"]),
        "rule_status_counts": dict(
            Counter(
                "KNOWN_STATIC_RULE" if value["rule_known"] else "UNKNOWN_RULE"
                for value in rules.values()
            )
        ),
        "input_identities": identities,
        "captured_at_utc": binding["captured_at_utc"],
        "snapshot_server_time_ms": snapshot.get("serverTime"),
        "late_status_filters_historical_qualification": False,
        "historical_point_in_time_filters_proven": False,
        "historical_minute_mark_prices_available": False,
        "min_notional_price_basis": "conservative_min_raw_open_and_fill_proxy_not_historical_mark",
        "market_average_fill_rounded_to_tick": False,
    }
    return rules, metadata


def quote_budget_view(frame):
    """完成边界t只看[t−59分钟,t]，60分钟缺任一值即未知，真实零仍为零。"""
    values = frame.index.to_numpy()
    if (
        frame.index.has_duplicates
        or not np.issubdtype(values.dtype, np.integer)
        or (values < 0).any()
        or (values % MINUTE != 0).any()
        or (len(values) > 1 and not np.equal(np.diff(values), MINUTE).all())
    ):
        raise ValueError("容量分钟须为严格连续整数完成边界，不能压缩缺口")
    quote = pd.to_numeric(frame["quote_volume"], errors="raise").astype(float)
    if (quote < 0).any():
        raise ValueError("原始成交金额不能为负")
    finite = np.isfinite(quote)
    clean = quote.where(finite)
    known = finite.rolling(60, min_periods=60).sum().eq(60)
    mean = clean.rolling(60, min_periods=60).mean()
    budget = float(RHO) * np.minimum(clean, mean)
    budget = budget.where(known & np.isfinite(budget))
    return pd.DataFrame(
        {
            "last_completed_quote": clean,
            "mean_quote_60m": mean,
            "known60": known,
            "budget": budget,
            "window_start_ms": values - 59 * MINUTE,
            "window_end_ms": values,
        },
        index=frame.index,
    )


def _verify_minute_inputs(records, input_identities):
    """每个实际规范CSV都须已由来源收据绑定，不能把无摘要路径加入执行。"""
    for record in records:
        if record["path"] is None:
            continue
        path = Path(record["path"])
        if path.is_symlink() or input_identities.get(str(path.resolve())) != sha(path):
            raise ValueError("容量实际分钟来源未绑定或摘要变化")


def memory_estimate(symbol_count, *, start_ms=None, end_ms=None):
    """只估六值时间优先映射磁盘；账户曲线内存由调用方单独预算。"""
    start = historical_panel.START_MS + 15 * DAY if start_ms is None else start_ms
    end = historical_panel.END_MS if end_ms is None else end_ms
    if (
        type(symbol_count) is not int
        or symbol_count <= 0
        or type(start) is not int
        or type(end) is not int
        or start % MINUTE
        or end % MINUTE
        or end <= start
    ):
        raise ValueError("映射估计须为正人口及完整分钟区间")
    minutes = (end - start) // MINUTE + 1
    return {
        "market_minutes": minutes,
        "market_mapped_bytes": symbol_count * minutes * 34,
        "bytes_per_symbol_minute": 34,
        "market_storage": "time_major_disk_memmap",
        "budget_rho": str(RHO),
    }


def market_frames(
    universe, sources, scratch, report, *, input_identities, start_ms=None, end_ms=None
):
    """逐合约读取12字段并映射六个值；每帧只给账户预算，不给当前bar金额。"""
    start = historical_panel.START_MS + 15 * DAY if start_ms is None else start_ms
    end = historical_panel.END_MS if end_ms is None else end_ms
    if (
        type(start) is not int
        or type(end) is not int
        or start % MINUTE
        or end % MINUTE
        or not historical_panel.START_MS < start < end <= historical_panel.END_MS
        or set(sources) != set(universe)
        or not universe
    ):
        raise ValueError("容量市场来源人口或固定完整分钟范围不同")
    clocks = np.arange(start, end + MINUTE, MINUTE, dtype=np.int64)
    full_calendar = np.arange(
        historical_panel.START_MS + MINUTE, historical_panel.END_MS + MINUTE, MINUTE, dtype=np.int64
    )
    symbols = tuple(sorted(universe))
    maps = []
    try:
        if any(
            (Path(scratch) / name).exists()
            for name in ("capacity_prices.npy", "capacity_flags.npy", "capacity_budgets.npy")
        ):
            raise ValueError("容量临时映射已存在，不覆盖原现场")
        prices = np.lib.format.open_memmap(
            Path(scratch) / "capacity_prices.npy",
            mode="w+",
            dtype=np.float64,
            shape=(len(clocks), len(symbols), 2),
        )
        maps.append(prices)
        flags = np.lib.format.open_memmap(
            Path(scratch) / "capacity_flags.npy",
            mode="w+",
            dtype=np.bool_,
            shape=(len(clocks), len(symbols), 2),
        )
        maps.append(flags)
        budgets = np.lib.format.open_memmap(
            Path(scratch) / "capacity_budgets.npy",
            mode="w+",
            dtype=np.float64,
            shape=(len(clocks), len(symbols), 2),
        )
        maps.append(budgets)
        for number, symbol in enumerate(symbols):
            _verify_minute_inputs(sources[symbol], input_identities)
            frame = historical_panel.load_minutes(symbol, sources[symbol])
            if not np.array_equal(frame.index.to_numpy(), full_calendar):
                raise ValueError("容量输入不是完整87840分钟原日历")
            budget_view = quote_budget_view(frame)
            selected = frame.loc[clocks, ["open", "close"]].to_numpy(dtype=float)
            # open阶段只核已存在的开盘，不用本bar未来close来提前避单。
            observed = np.isfinite(selected[:, 0]) & (selected[:, 0] > 0)
            qualified, _ = historical_panel.qualification(clocks - MINUTE, universe[symbol])
            open_budget = budget_view.budget.shift(1).reindex(clocks).to_numpy(dtype=float)
            decision_budget = budget_view.budget.reindex(clocks).to_numpy(dtype=float)
            prices[:, number, :], flags[:, number, 0], flags[:, number, 1] = (
                selected,
                qualified,
                observed,
            )
            budgets[:, number, 0], budgets[:, number, 1] = open_budget, decision_budget
            _verify_minute_inputs(sources[symbol], input_identities)
            report[symbol] = {
                "input_minutes": len(frame),
                "minutes": len(clocks),
                "observed_minutes": int(observed.sum()),
                "qualified_minutes_under_static_assumption": int(qualified.sum()),
                "known_open_budget_minutes": int(np.isfinite(open_budget).sum()),
                "zero_open_budget_minutes": int(np.equal(open_budget, 0).sum()),
                "known_decision_budget_minutes": int(np.isfinite(decision_budget).sum()),
                "zero_decision_budget_minutes": int(np.equal(decision_budget, 0).sum()),
                "rho": str(RHO),
                "open_budget_source_lag_ms": MINUTE,
            }
            del frame, budget_view, selected, observed, qualified, open_budget, decision_budget
            if number % 32 == 31:
                gc.collect()
        for mapped in maps:
            mapped.flush()
        for number, clock in enumerate(clocks):
            yield (
                int(clock),
                {
                    symbol: {
                        "open": float(prices[number, column, 0]),
                        "close": float(prices[number, column, 1]),
                        "qualified": bool(flags[number, column, 0]),
                        "observed": bool(flags[number, column, 1]),
                        "open_budget": float(budgets[number, column, 0])
                        if np.isfinite(budgets[number, column, 0])
                        else None,
                        "decision_budget": float(budgets[number, column, 1])
                        if np.isfinite(budgets[number, column, 1])
                        else None,
                        "open_budget_asof_ms": int(clock) - MINUTE,
                        "decision_budget_asof_ms": int(clock),
                    }
                    for column, symbol in enumerate(symbols)
                },
            )
    finally:
        for mapped in maps:
            mapped._mmap.close()
