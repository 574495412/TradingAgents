"""Damorne discovery, frozen public-factor candidates and independent perpetual accounts.

Public order records are offline evidence. Runtime receives market bars, frozen rules
and its own inventory only. No exchange authentication or order path exists here.
"""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .evidence import save, sha

MINUTE, DAY = 60_000, 86_400_000
FEATURES = [
    "return5",
    "return15",
    "return60",
    "range_location",
    "atr",
    "volume_ratio",
    "flow",
    "quote60",
]


def millis(value):
    return int(datetime.fromisoformat(value).timestamp() * 1000)


def feature_frame(trade, mark):
    f = trade[["open", "high", "low", "close", "quote_volume", "taker_buy_quote_volume"]].copy()
    f["mark"] = mark.open
    f["mark_high"], f["mark_low"] = mark.high, mark.low
    if (
        not np.isfinite(f.to_numpy()).all()
        or (f[["open", "high", "low", "close", "mark"]] <= 0).any().any()
    ):
        raise ValueError("Invalid price observations")
    for minutes in [5, 15, 60]:
        f[f"return{minutes}"] = f.close.pct_change(minutes)
    high, low = f.high.rolling(60).max(), f.low.rolling(60).min()
    f["range_location"] = (f.close - low) / (high - low).replace(0, np.nan)
    tr = pd.concat(
        [f.high - f.low, (f.high - f.close.shift()).abs(), (f.low - f.close.shift()).abs()],
        axis=1,
    ).max(axis=1)
    f["atr"] = tr.rolling(60).mean() / f.close
    baseline = f.quote_volume.rolling(1440, min_periods=60).median()
    f["volume_ratio"] = f.quote_volume / baseline.replace(0, np.nan)
    f["flow"] = 2 * f.taker_buy_quote_volume / f.quote_volume.replace(0, np.nan) - 1
    f["quote60"] = f.quote_volume.rolling(60).sum()
    return f


def load_market(root):
    protocol = json.loads((root / "protocol.json").read_text())
    sources = json.loads((root / "market_sources.json").read_text())
    for r in json.loads((root / "rest_backfill_hashes.json").read_text()):
        sources.append(r)
    lower, upper = millis(protocol["start"]), millis(protocol["end"])
    market, quality = {}, []
    for symbol in protocol["symbols"]:
        parts = {}
        for kind in ["klines", "markPriceKlines"]:
            frames = []
            for source in sources:
                if (source["symbol"], source["kind"]) != (symbol, kind):
                    continue
                path = Path(source["path"])
                if sha(path) != source["sha256"]:
                    raise ValueError("Market archive source changed")
                if source.get("format") == "rest_json":
                    frame = pd.DataFrame(
                        json.loads(path.read_text()),
                        columns=[
                            "open_time",
                            "open",
                            "high",
                            "low",
                            "close",
                            "volume",
                            "close_time",
                            "quote_volume",
                            "count",
                            "taker_buy_volume",
                            "taker_buy_quote_volume",
                            "ignore",
                        ],
                    )
                    frame = frame.apply(pd.to_numeric)
                else:
                    with (
                        zipfile.ZipFile(path) as archive,
                        archive.open(archive.namelist()[0]) as stream,
                    ):
                        frame = pd.read_csv(stream)
                if "open_time" not in frame:
                    raise ValueError("Official archive schema changed")
                frame = frame[(frame.open_time >= lower) & (frame.open_time < upper)]
                frames.append(frame)
            if not frames:
                parts[kind] = pd.DataFrame()
                continue
            frame = pd.concat(frames).sort_values("open_time")
            if frame.open_time.duplicated().any():
                raise ValueError("Duplicate market bars; reconcile sources first")
            parts[kind] = frame.set_index("open_time")
        trade, mark = parts["klines"], parts["markPriceKlines"]
        expected = (upper - lower) // MINUTE
        ok = len(trade) == len(mark) == expected and trade.index.equals(mark.index)
        if ok:
            ok = np.array_equal(trade.index.to_numpy(), np.arange(lower, upper, MINUTE))
        quality.append(
            {
                "symbol": symbol,
                "trade_rows": len(trade),
                "mark_rows": len(mark),
                "expected": expected,
                "complete": bool(ok),
            }
        )
        if not ok:
            continue
        f = feature_frame(trade, mark)
        f["funding"] = 0.0
        funding_source = next(
            x
            for x in json.loads((root / "funding_sources.json").read_text())
            if x["symbol"] == symbol
        )
        funding_path = Path(funding_source["path"])
        if sha(funding_path) != funding_source["sha256"]:
            raise ValueError("Funding source changed")
        rows = json.loads(funding_path.read_text())
        for row in rows:
            clock = int(row["fundingTime"])
            # Preserve scheduled events with millisecond publication jitter in their minute.
            clock = clock // MINUTE * MINUTE
            if clock in f.index:
                f.loc[clock, "funding"] += float(row["fundingRate"])
        times = sorted(int(r["fundingTime"]) for r in rows)
        # A complete observed schedule is required, not merely a successful empty response.
        funding_ok = (
            bool(times)
            and times[0] - lower <= 8 * 3_600_000 + MINUTE
            and upper - times[-1] <= 8 * 3_600_000 + MINUTE
        )
        funding_ok &= all(
            b - a <= 8 * 3_600_000 + MINUTE for a, b in zip(times, times[1:], strict=False)
        )
        quality[-1]["funding_schedule_covered"] = bool(funding_ok)
        if not funding_ok:
            continue
        market[symbol] = f
    return protocol, market, quality


def reconstruct_orders(document):
    """Reconstruct only observed inventory; missing opening inventory stays unknown."""
    inventory, closed, events, problems = {}, [], [], []
    gap_ranges = [(r["startTime"], r["endTime"]) for r in document["unavailable_windows"]]
    last_clock = document["start_ms"]
    for row in document["rows"]:
        when = int(row["orderUpdateTime"])
        if any(last_clock <= b and a <= when for a, b in gap_ranges):
            inventory.clear()
        last_clock = when
        side = row["positionSide"]
        if side not in {"LONG", "SHORT"}:
            raise ValueError("Explicit hedge position side required")
        key = (row["symbol"], side)
        quantity, price = float(row["executedQty"]), float(row["avgPrice"])
        if quantity <= 0 or price <= 0:
            raise ValueError("Non-positive executed order")
        opening = row["side"] == ("BUY" if side == "LONG" else "SELL")
        p = inventory.get(key)
        event = {**row, "clock": when, "known_inventory_before": p is not None}
        if opening:
            event["event"] = "ADD" if p else "OPEN_OR_UNKNOWN_REENTRY"
            if not p:
                p = {
                    "symbol": key[0],
                    "side": side,
                    "opened_ms": when,
                    "quantity": 0.0,
                    "opened_quantity": 0.0,
                    "entry_value": 0.0,
                    "exit_value": 0.0,
                    "closed_quantity": 0.0,
                    "reported_gross": 0.0,
                    "adds": 0,
                    "reductions": 0,
                }
                inventory[key] = p
            else:
                p["adds"] += 1
            p["quantity"] += quantity
            p["opened_quantity"] += quantity
            p["entry_value"] += quantity * price
        elif not p or quantity > p["quantity"] + max(1e-8, quantity * 1e-8):
            event["event"] = "CLOSE_UNKNOWN_INVENTORY"
            problems.append(event)
            inventory.pop(key, None)
        else:
            p["quantity"] -= quantity
            p["closed_quantity"] += quantity
            p["exit_value"] += quantity * price
            p["reported_gross"] += float(row["totalPnl"])
            if p["quantity"] <= max(1e-8, p["opened_quantity"] * 1e-8):
                event["event"] = "CLOSE"
                p["closed_ms"] = when
                p["entry"] = p["entry_value"] / p["opened_quantity"]
                p["exit"] = p["exit_value"] / p["closed_quantity"]
                p["gross_price_return"] = (1 if side == "LONG" else -1) * (
                    p["exit"] / p["entry"] - 1
                )
                p["notional"] = p["entry_value"]
                p["duration_minutes"] = (when - p["opened_ms"]) / MINUTE
                closed.append(p)
                del inventory[key]
            else:
                event["event"] = "REDUCE"
                p["reductions"] += 1
        events.append(event)
    return events, closed, problems


def _quantiles(values):
    return (
        {str(q): float(np.quantile(values, q)) for q in [0.1, 0.25, 0.5, 0.75, 0.9]}
        if values
        else None
    )


def describe(root, protocol, market, output):
    orders = json.loads((root / "orders_cursor_v002/orders.json").read_text())
    events, positions, problems = reconstruct_orders(orders)
    train_end = millis(protocol["train_end"])
    train = [p for p in positions if p["closed_ms"] < train_end]
    notionals = sorted(p["notional"] for p in train)
    gaps = [(b / a, a, b) for a, b in zip(notionals, notionals[1:], strict=False) if a > 0]
    _, left, right = max(gaps) if gaps else (0, 500, 500)
    size_threshold = math.sqrt(left * right)
    small = [p for p in train if p["notional"] <= size_threshold]
    large = [p for p in train if p["notional"] > size_threshold]
    waits = [max(0, int(r["orderUpdateTime"]) - int(r["orderTime"])) / 1000 for r in events]
    findings = {
        "public_order_summaries": len(events),
        "event_counts": dict(Counter(e["event"] for e in events)),
        "order_types": dict(Counter(e["type"] for e in events)),
        "unavailable_windows": orders["unavailable_windows"],
        "complete_observed_inventory_chains": len(positions),
        "unknown_inventory_closes": len(problems),
        "order_submission_to_last_update_seconds": _quantiles(waits),
        "training_only_size_split_notional": size_threshold,
        "train_small": {
            "count": len(small),
            "gross_price_return": _quantiles([p["gross_price_return"] for p in small]),
            "duration_minutes": _quantiles([p["duration_minutes"] for p in small]),
        },
        "train_large": {
            "count": len(large),
            "gross_price_return": _quantiles([p["gross_price_return"] for p in large]),
            "duration_minutes": _quantiles([p["duration_minutes"] for p in large]),
        },
        "private_tp_sl_recovered": False,
        "current_position_hidden": True,
        "reported_order_gross_pnl_is_not_full_cost_account_profit": True,
        "scope": "Missing intervals invalidate adjacent inventory; opening may be reentry into unknown inventory; descriptive chains require later reconciliation to closed summaries.",
    }
    save(output / "order_findings.json", findings)
    save(output / "order_events.json", events)
    save(output / "observed_inventory_chains.json", positions)
    # Actual public filled orders are cases. Background is an explicit sampled control,
    # not evidence that the leader observed HOLD or placed no unfilled orders.
    samples, targets, clocks = [], [], []
    completed_training_openings = {(p["symbol"], p["side"], p["opened_ms"]) for p in train}
    for e in events:
        symbol = e["symbol"]
        when = int(e["orderTime"]) // MINUTE * MINUTE
        if (
            e["side"] != ("BUY" if e["positionSide"] == "LONG" else "SELL")
            or symbol not in market
            or when >= train_end
            or int(e["orderUpdateTime"]) >= train_end
            or (symbol, e["positionSide"], int(e["orderUpdateTime"]))
            not in completed_training_openings
        ):
            continue
        source = market[symbol]
        previous = when - MINUTE
        if previous not in source.index:
            continue
        vector = source.loc[previous, FEATURES].to_numpy(dtype=float)
        if np.isfinite(vector).all():
            samples.append(vector)
            targets.append(1 if e["positionSide"] == "LONG" else 2)
            clocks.append(when)
    for _symbol, source in sorted(market.items()):
        controls = source[(source.index < train_end - MINUTE)].iloc[60::30]
        for when, row in controls.iterrows():
            vector = row[FEATURES].to_numpy(dtype=float)
            if np.isfinite(vector).all():
                samples.append(vector)
                targets.append(0)
                clocks.append(int(when) + MINUTE)
    return findings, np.asarray(samples), np.asarray(targets), clocks


def tree_export(model, node=0):
    t = model.tree_
    if t.children_left[node] == t.children_right[node]:
        index = int(np.argmax(t.value[node][0]))
        return {"action": int(model.classes_[index]), "samples": int(t.n_node_samples[node])}
    return {
        "feature": FEATURES[t.feature[node]],
        "threshold": float(t.threshold[node]),
        "left": tree_export(model, t.children_left[node]),
        "right": tree_export(model, t.children_right[node]),
    }


def predict_tree(node, features):
    while "action" not in node:
        node = (
            node["left"]
            if np.float32(features[node["feature"]]) <= node["threshold"]
            else node["right"]
        )
    return node["action"]


def make_candidates(findings, x, y):
    from sklearn.tree import DecisionTreeClassifier

    quantile = findings["train_small"]["gross_price_return"]
    observed_target = min(0.02, max(0.003, quantile["0.5"] if quantile else 0.003))
    result = []
    for admission in ["range_reversal", "trend_pullback", "tree3", "tree5"]:
        tree = None
        if admission.startswith("tree"):
            if len(set(y)) < 3:
                continue
            model = DecisionTreeClassifier(
                max_depth=int(admission[-1]),
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=0,
            ).fit(x, y)
            tree = tree_export(model)
            if any(
                predict_tree(tree, dict(zip(FEATURES, v, strict=True))) != int(pred)
                for v, pred in zip(x, model.predict(x), strict=True)
            ):
                raise ValueError("Exported tree disagrees with fit")
        for continuation in [False, True]:
            result.append(
                {
                    "id": f"{admission}-rearm{int(continuation)}",
                    "admission": admission,
                    "tree": tree,
                    "continuation": continuation,
                    "small_target": observed_target,
                    "small_stop": 0.01,
                    "large_target": 0.02,
                    "large_stop": 0.01,
                    "small_max_minutes": 180,
                    "large_max_minutes": 2160,
                    "add_at_fraction_of_target": 0.5,
                    "trim_fraction": 0.5,
                    "max_additions": 1,
                    "max_positions": 6,
                    "max_margin_fraction": 0.30,
                    "leverage": 2,
                    "small_margin": 50,
                    "large_margin": 500,
                    "capital": 10000,
                    "fee_bps": 5,
                    "slippage_bps": 3,
                    "features": FEATURES,
                    "background_is_sampled_control_not_observed_hold": True,
                    "production_eligible": False,
                }
            )
    return result


def signal(config, f):
    if not all(math.isfinite(f[k]) for k in FEATURES) or f["quote60"] < 1_000_000:
        return 0
    if config["admission"].startswith("tree"):
        value = predict_tree(config["tree"], f)
        return {0: 0, 1: 1, 2: -1}[value]
    if config["admission"] == "range_reversal":
        if f["range_location"] <= 0.2 and f["return5"] > 0 and f["flow"] > 0:
            return 1
        if f["range_location"] >= 0.8 and f["return5"] < 0 and f["flow"] < 0:
            return -1
    else:
        for sign in [1, -1]:
            if (
                sign * f["return60"] >= 0.01
                and sign * f["return15"] > 0
                and sign * f["return5"] < 0
                and sign * f["flow"] > 0.1
            ):
                return sign
    return 0


class ContractBook:
    """One shared wallet, independent sides; only whole flat-to-flat chains count as trades."""

    def __init__(self, config, metadata):
        self.config, self.metadata = config, metadata
        self.wallet = config["capital"]
        self.positions, self.closed, self.last_close = {}, [], {}
        self.pending, self.decisions, self.curve = [], [], []
        self.fees, self.funding, self.counter = 0.0, 0.0, 0
        self.rejections = Counter()

    def equity(self, marks):
        return self.wallet + math.fsum(
            p["sign"] * p["qty"] * (marks[s] - p["entry"])
            - p["qty"] * marks[s] * self.config["fee_bps"] / 10000
            for (s, _), p in self.positions.items()
        )

    def reserve(self):
        return math.fsum(p["margin"] for p in self.positions.values())

    def transact(self, action, symbol, sign, price, when, marks, sleeve="small"):
        key = (symbol, sign)
        p = self.positions.get(key)
        c = self.config
        fee_rate, slip = c["fee_bps"] / 10000, c["slippage_bps"] / 10000
        if action in {"OPEN", "ADD"}:
            if (action == "OPEN" and p is not None) or (action == "ADD" and p is None):
                self.rejections["inventory_mismatch"] += 1
                return
            if not p and len(self.positions) >= c["max_positions"]:
                self.rejections["position_capacity"] += 1
                return
            sleeve = p["sleeve"] if p else sleeve
            margin = c[f"{sleeve}_margin"]
            fill = price * (1 + sign * slip)
            metadata = self.metadata[symbol]
            step = metadata["step"]
            quantity = math.floor(margin * c["leverage"] / fill / step + 1e-9) * step
            if (
                quantity < metadata["min_qty"]
                or quantity * fill < metadata["min_notional"]
                or quantity > metadata["max_qty"]
            ):
                self.rejections["lot_or_notional"] += 1
                return
            margin = quantity * fill / c["leverage"]
            fee = quantity * fill * fee_rate
            equity_after = (
                self.equity(marks)
                - fee
                + sign * quantity * (marks[symbol] - fill)
                - quantity * marks[symbol] * fee_rate
            )
            if self.reserve() + margin > min(equity_after, c["capital"] * c["max_margin_fraction"]):
                self.rejections["shared_margin_cap"] += 1
                return
            if p and p["adds"] >= c["max_additions"]:
                self.rejections["addition_cap"] += 1
                return
            self.wallet -= fee
            self.fees += fee
            if p:
                p["entry"] = (p["qty"] * p["entry"] + quantity * fill) / (p["qty"] + quantity)
                p["qty"] += quantity
                p["margin"] += margin
                p["invested_margin"] += margin
                p["fees"] += fee
                p["adds"] += 1
            else:
                self.counter += 1
                self.positions[key] = {
                    "id": self.counter,
                    "symbol": symbol,
                    "sign": sign,
                    "entry": fill,
                    "qty": quantity,
                    "margin": margin,
                    "invested_margin": margin,
                    "fees": fee,
                    "funding": 0.0,
                    "gross_realized": 0.0,
                    "opened_ms": when,
                    "sleeve": sleeve,
                    "adds": 0,
                    "trimmed": False,
                }
        else:
            if p is None:
                self.rejections["missing_owned_position"] += 1
                return
            quantity = p["qty"]
            if action == "TRIM":
                step = self.metadata[symbol]["step"]
                quantity = math.floor(quantity * c["trim_fraction"] / step + 1e-9) * step
                if (
                    quantity < self.metadata[symbol]["min_qty"]
                    or quantity * price < self.metadata[symbol]["min_notional"]
                ):
                    self.rejections["trim_below_lot"] += 1
                    p["trimmed"] = True
                    return
            fill = price * (1 - sign * slip)
            gross = sign * quantity * (fill - p["entry"])
            fee = quantity * fill * fee_rate
            self.wallet += gross - fee
            self.fees += fee
            p["fees"] += fee
            p["gross_realized"] += gross
            if action == "TRIM":
                p["margin"] *= 1 - quantity / p["qty"]
                p["qty"] -= quantity
                p["trimmed"] = True
            else:
                net = p["gross_realized"] + p["funding"] - p["fees"]
                self.closed.append({**p, "closed_ms": when, "exit": fill, "net_pnl": net})
                self.last_close[key] = {"when": when, "net_pnl": net, "price": fill}
                del self.positions[key]

    def snapshot(self):
        return {
            "wallet": self.wallet,
            "positions": list(self.positions.values()),
            "closed": self.closed,
            "last_close": [
                {"symbol": s, "sign": sign, **p} for (s, sign), p in self.last_close.items()
            ],
            "pending": self.pending,
            "decisions": self.decisions,
            "curve": self.curve,
            "fees": self.fees,
            "funding": self.funding,
            "counter": self.counter,
            "rejections": dict(self.rejections),
        }

    def restore(self, state):
        self.wallet, self.closed = state["wallet"], state["closed"]
        self.positions = {(p["symbol"], p["sign"]): p for p in state["positions"]}
        self.last_close = {
            (p["symbol"], p["sign"]): {k: v for k, v in p.items() if k not in {"symbol", "sign"}}
            for p in state["last_close"]
        }
        for key in ["pending", "decisions", "curve", "fees", "funding", "counter"]:
            setattr(self, key, state[key])
        self.rejections = Counter(state["rejections"])


def metadata_for(catalog):
    result = {}
    for row in catalog["symbols"]:
        filters = {f["filterType"]: f for f in row["filters"]}
        lot = filters.get("MARKET_LOT_SIZE", filters["LOT_SIZE"])
        if float(lot["stepSize"]) == 0:
            lot = filters["LOT_SIZE"]
        result[row["symbol"]] = {
            "step": float(lot["stepSize"]),
            "min_qty": float(lot["minQty"]),
            "max_qty": float(lot["maxQty"]),
            "min_notional": float(filters.get("MIN_NOTIONAL", {}).get("notional", 5)),
        }
    return result


def account(config, market, metadata, start_ms, end_ms, state=None, break_ms=None):
    # No leader files, observed order prices/times, labels or teacher inventory accepted here.
    book = ContractBook(config, metadata)
    if state:
        book.restore(state)
    symbols = sorted(market)
    index = next(iter(market.values())).index
    indices = np.where((index >= start_ms) & (index < end_ms))[0]
    arrays = {
        s: market[s][["open", "mark", "funding", *FEATURES]].to_numpy(dtype=float) for s in symbols
    }
    for i in indices:
        when = int(index[i])
        if break_ms is not None and when >= break_ms:
            return None, book.snapshot()
        rows = {s: arrays[s][i] for s in symbols}
        marks = {s: r[1] for s, r in rows.items()}
        for (symbol, sign), p in sorted(book.positions.items()):
            value = -sign * p["qty"] * marks[symbol] * rows[symbol][2]
            book.wallet += value
            book.funding += value
            p["funding"] += value
        for order in book.pending:
            if order["due_ms"] != when:
                raise ValueError("Noncontiguous owned account; no stale fill")
            book.transact(
                order["action"],
                order["symbol"],
                order["sign"],
                rows[order["symbol"]][0],
                when,
                marks,
                order.get("sleeve", "small"),
            )
        book.pending = []
        equity = book.equity(marks)
        if equity <= 0:
            raise ValueError("Research account insolvent; candidate failed")
        book.curve.append(
            {
                "timestamp": when,
                "equity": equity,
                "wallet": book.wallet,
                "margin": book.reserve(),
                "positions": len(book.positions),
            }
        )
        for symbol in symbols:
            if i == 0:
                continue
            f = dict(zip(FEATURES, arrays[symbol][i - 1][3:], strict=True))
            direction = signal(config, f)
            for sign in [1, -1]:
                key, action, sleeve = (symbol, sign), None, "small"
                p = book.positions.get(key)
                if p:
                    gross_return = sign * (marks[symbol] / p["entry"] - 1)
                    age = (when - p["opened_ms"]) / MINUTE
                    target, stop = config[f"{p['sleeve']}_target"], config[f"{p['sleeve']}_stop"]
                    if (
                        gross_return <= -stop
                        or age >= config[f"{p['sleeve']}_max_minutes"]
                        or gross_return >= 2 * target
                    ):
                        action = "CLOSE"
                    elif gross_return >= target and not p["trimmed"]:
                        action = "TRIM"
                    elif (
                        p["adds"] < config["max_additions"]
                        and not p["trimmed"]
                        and age >= 15
                        and gross_return >= target * config["add_at_fraction_of_target"]
                        and sign * f["return15"] > 0
                    ):
                        action = "ADD"
                elif direction == sign:
                    previous = book.last_close.get(key)
                    delay = when - previous["when"] if previous else DAY
                    cooldown = (
                        2 * MINUTE
                        if config["continuation"] and previous and previous["net_pnl"] > 0
                        else 60 * MINUTE
                    )
                    if delay >= cooldown:
                        action = "OPEN"
                        # Proposed capital rule, independently evaluated; never infer equity from leader size.
                        if (
                            f["quote60"] >= 5_000_000
                            and f["volume_ratio"] >= 2
                            and sign * f["return60"] >= 0.01
                            and sign * f["return15"] >= 0.002
                            and sign * f["flow"] > 0.1
                        ):
                            sleeve = "large"
                if action:
                    order = {
                        "decision_ms": when,
                        "due_ms": when + MINUTE,
                        "symbol": symbol,
                        "sign": sign,
                        "action": action,
                        "sleeve": sleeve,
                    }
                    book.pending.append(order)
                    book.decisions.append(order)
    return score(book, marks), book.snapshot()


def score(book, marks):
    c = book.config
    terminal = book.equity(marks)
    open_net = {}
    for (symbol, _), p in book.positions.items():
        open_net[p["id"]] = (
            p["gross_realized"]
            + p["sign"] * p["qty"] * (marks[symbol] - p["entry"])
            + p["funding"]
            - p["fees"]
            - p["qty"] * marks[symbol] * c["fee_bps"] / 10000
        )
    expected = (
        c["capital"] + math.fsum(p["net_pnl"] for p in book.closed) + math.fsum(open_net.values())
    )
    if not math.isclose(terminal, expected, abs_tol=1e-6):
        raise ValueError("Whole-position cash reconciliation failed")
    values = np.array([r["equity"] for r in book.curve])
    drawdown = float(
        np.max((np.maximum.accumulate(values) - values) / np.maximum.accumulate(values))
    )
    midnight = [r for r in book.curve if r["timestamp"] % DAY == 0]
    returns = np.array(
        [b["equity"] / a["equity"] - 1 for a, b in zip(midnight, midnight[1:], strict=False)]
    )
    if any(
        b["timestamp"] - a["timestamp"] != DAY for a, b in zip(midnight, midnight[1:], strict=False)
    ):
        raise ValueError("Daily observation gap")
    std = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0
    downside = float(np.sqrt(np.mean(np.minimum(returns, 0) ** 2))) if len(returns) else 0
    pnl = [p["net_pnl"] for p in book.closed]
    loss = -math.fsum(min(0, v) for v in pnl)
    sleeves = {}
    for name in ["small", "large"]:
        complete = [p for p in book.closed if p["sleeve"] == name]
        unrealized = math.fsum(
            open_net[p["id"]] for p in book.positions.values() if p["sleeve"] == name
        )
        sleeves[name] = {
            "complete_positions": len(complete),
            "net_pnl": math.fsum(p["net_pnl"] for p in complete) + unrealized,
            "whole_net_win_rate": sum(p["net_pnl"] > 0 for p in complete) / len(complete)
            if complete
            else None,
            "terminal_open_net_pnl": unrealized,
        }
    return {
        "candidate": c["id"],
        "net_return": terminal / c["capital"] - 1,
        "ending_equity": terminal,
        "complete_positions": len(pnl),
        "whole_net_win_rate": sum(v > 0 for v in pnl) / len(pnl) if pnl else None,
        "profit_factor": math.fsum(max(0, v) for v in pnl) / loss if loss else None,
        "minute_observed_mark_open_drawdown": drawdown,
        "daily_sharpe": float(np.mean(returns)) / std * math.sqrt(365) if std > 1e-12 else None,
        "daily_sortino": float(np.mean(returns)) / downside * math.sqrt(365)
        if downside > 1e-12
        else None,
        "complete_calendar_days_for_risk_metrics": len(returns),
        "fees": book.fees,
        "funding": book.funding,
        "terminal_open_positions": len(book.positions),
        "terminal_open_net_pnl": math.fsum(open_net.values()),
        "sleeves": sleeves,
        "whole_position_cash_reconciled": True,
        "partial_trims_count_as_wins": False,
        "rejections": dict(book.rejections),
        "pending_terminal_orders": len(book.pending),
        "queue_impact_liquidation_model_verified": False,
        "current_catalog_retrospective_universe_and_lot_rules": True,
        "actual_live_forward_evidence": False,
        "production_eligible": False,
    }


def study(root, output):
    output.mkdir(exist_ok=False)
    protocol, market, quality = load_market(root)
    save(output / "market_quality.json", quality)
    catalog = json.loads((root / "discovery/catalog.json").read_text())
    crypto = {r["symbol"] for r in catalog["symbols"] if "Crypto" in r.get("underlyingSubType", [])}
    market = {s: f for s, f in market.items() if s in crypto}
    findings, x, y, clocks = describe(root, protocol, market, output)
    if len(market) < 5:
        raise ValueError("Insufficient complete crypto market sources")
    candidates = make_candidates(findings, x, y)
    source = Path(__file__)
    save(
        output / "candidate_freeze.json",
        {
            "registered_utc": datetime.now(UTC).isoformat(),
            "protocol_sha256": sha(root / "protocol.json"),
            "study_source_sha256": sha(source),
            "capture_source_sha256": sha(root / "capture_source.py"),
            "market_symbols": sorted(market),
            "training_cases": dict(Counter(map(int, y))),
            "candidates": candidates,
            "holdout_read": False,
            "all_experiments_on_old_inspected_history_exploratory": True,
        },
    )
    (output / "study_source.py").write_bytes(source.read_bytes())
    metadata = metadata_for(catalog)
    validation_start, test_start, end = (
        millis(protocol["train_end"]),
        millis(protocol["validation_end"]),
        millis(protocol["end"]),
    )
    validation = []
    for candidate in candidates:
        summary, _ = account(candidate, market, metadata, validation_start, test_start)
        summary["selection_gate"] = bool(
            summary["net_return"] > 0
            and summary["minute_observed_mark_open_drawdown"] <= 0.15
            and summary["complete_positions"] >= 30
            and all(s["net_pnl"] > 0 for s in summary["sleeves"].values())
        )
        validation.append(summary)
        print(
            {
                "validation": candidate["id"],
                "net_return": summary["net_return"],
                "gate": summary["selection_gate"],
            },
            flush=True,
        )
    save(output / "validation_scores.json", validation)
    eligible = [r for r in validation if r["selection_gate"]]
    ranked = sorted(
        eligible or validation,
        key=lambda r: (-r["net_return"], r["minute_observed_mark_open_drawdown"], r["candidate"]),
    )
    chosen = next(c for c in candidates if c["id"] == ranked[0]["candidate"])
    save(
        output / "selection.json",
        {
            "candidate": chosen,
            "eligible_on_validation": bool(eligible),
            "scope": "selected for held-out research"
            if eligible
            else "diagnostic best validation candidate only; no strategy promotion",
            "selected_utc": datetime.now(UTC).isoformat(),
            "holdout_outcomes_read": False,
        },
    )
    holdout, state = account(chosen, market, metadata, test_start, end)
    save(output / "holdout_score.json", holdout)
    save(output / "holdout_state.json", state)
    # Independently restart from an exact owned-state checkpoint in the same account.
    middle = test_start + (end - test_start) // (2 * MINUTE) * MINUTE
    _, prefix = account(chosen, market, metadata, test_start, end, break_ms=middle)
    resumed_score, resumed = account(
        chosen, market, metadata, middle, end, state=json.loads(json.dumps(prefix))
    )
    if state != resumed or holdout != resumed_score:
        raise ValueError("Restart changes terminal account or economics")
    save(
        output / "restart_verification.json",
        {"exact_terminal_state_equal": True, "score_equal": True, "checkpoint_ms": middle},
    )
    stresses = []
    for extra in [2, 5]:
        stressed = {**chosen, "slippage_bps": chosen["slippage_bps"] + extra}
        summary, _ = account(stressed, market, metadata, test_start, end)
        stresses.append({"additional_slippage_bps_per_side": extra, **summary})
    save(output / "cost_stress.json", stresses)
    if (
        sha(source)
        != json.loads((output / "candidate_freeze.json").read_text())["study_source_sha256"]
    ):
        raise ValueError("Source changed during evaluation")
    verdict = (
        "RESEARCH_CANDIDATE_REQUIRES_FORWARD_VALIDATION"
        if eligible
        and holdout["net_return"] > 0
        and all(s["net_pnl"] > 0 for s in holdout["sleeves"].values())
        else "REJECT_LIVE_REMAIN_RESEARCH"
    )
    save(
        output / "run_status.json",
        {
            "status": "completed",
            "verdict": verdict,
            "production_eligible": False,
            "orders_enabled": False,
        },
    )
    print({"holdout": holdout, "verdict": verdict}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    study(args.root.absolute(), args.output.absolute())
