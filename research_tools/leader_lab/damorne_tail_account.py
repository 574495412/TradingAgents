"""Owned perpetual limit-order account: profit receipts, bounded scaling and loss circuits."""

from __future__ import annotations

import json
import math
from collections import Counter
from datetime import UTC, datetime

import numpy as np

from .damorne_research import DAY, MINUTE, ContractBook, millis, score
from .evidence import save, sha


def price_barriers(entry, sign, target_fraction, stop_fraction, tick):
    target = entry * (1 + sign * target_fraction) / tick
    stop = entry * (1 - sign * stop_fraction) / tick
    if sign == 1:
        return math.ceil(target - 1e-9) * tick, math.floor(stop + 1e-9) * tick
    return math.floor(target + 1e-9) * tick, math.ceil(stop - 1e-9) * tick


def candidates(economics):
    rows = []
    for strict in [False, True]:
        for additions in [False, True]:
            for reductions in [False, True]:
                rows.append(
                    {
                        **economics,
                        "id": f"payoff-{'strict' if strict else 'balanced'}-add{int(additions)}-trim{int(reductions)}",
                        "seed_probability": 0.65 if strict else 0.60,
                        "rearm_probability": 0.60 if strict else 0.55,
                        "large_probability": 0.53 if strict else 0.48,
                        "small_min_expected_net": 0.0008 if strict else 0.0005,
                        "large_min_expected_net": 0.002 if strict else 0.0015,
                        "allow_additions": additions,
                        "allow_reductions": reductions,
                        "fee_bps": economics["trading_fee_bps"] + economics["execution_cost_bps"],
                        "slippage_bps": 0.0,
                        "engine": "owned_limit_payoff_v1",
                        "production_eligible": False,
                    }
                )
    return rows


def protective_stop(p, fee_rate):
    # Net loss at stop includes all previously paid costs and the future exit reserve.
    rhs = (
        p["sign"] * p["qty"] * p["entry"]
        - p["gross_realized"]
        - p["funding"]
        + p["fees"]
        - p["risk_budget"]
    )
    return rhs / (p["qty"] * (p["sign"] - fee_rate))


def _open(book, order, when, marks):
    key = (order["symbol"], order["sign"])
    if key in book.positions:
        return False
    book.transact(
        "OPEN",
        order["symbol"],
        order["sign"],
        order["price"],
        when + MINUTE - 1,
        marks,
        order["sleeve"],
    )
    p = book.positions.get(key)
    if p is None:
        return False
    c = book.config
    p["entry_bar_ms"] = when
    p["expiry_ms"] = when + c[f"{p['sleeve']}_max_minutes"] * MINUTE
    p["target"], p["stop"] = price_barriers(
        p["entry"],
        p["sign"],
        c[f"{p['sleeve']}_target"],
        c[f"{p['sleeve']}_stop"],
        book.metadata[p["symbol"]]["price_tick"],
    )
    rate = c["fee_bps"] / 10000
    p["risk_budget"] = (
        p["qty"] * abs(p["entry"] - p["stop"]) + p["fees"] + p["qty"] * p["stop"] * rate
    )
    p["initial_margin"] = p["margin"]
    p["defensive_reduced"] = False
    return True


def _add(book, operation, price, when, marks):
    key = (operation["symbol"], operation["sign"])
    p = book.positions.get(key)
    if p is None or p["id"] != operation["position_id"] or p["adds"] >= 1:
        return False
    c = book.config
    step = book.metadata[p["symbol"]]["step"]
    qty = math.floor(c[f"{p['sleeve']}_margin"] * c["leverage"] / price / step + 1e-9) * step
    if qty <= 0:
        return False
    projected = {
        **p,
        "qty": p["qty"] + qty,
        "entry": (p["qty"] * p["entry"] + qty * price) / (p["qty"] + qty),
        "fees": p["fees"] + qty * price * c["fee_bps"] / 10000,
    }
    new_stop = protective_stop(projected, c["fee_bps"] / 10000)
    new_stop = max(p["stop"], new_stop) if p["sign"] == 1 else min(p["stop"], new_stop)
    tick = book.metadata[p["symbol"]]["price_tick"]
    new_stop = (
        math.ceil(new_stop / tick - 1e-9) if p["sign"] == 1 else math.floor(new_stop / tick + 1e-9)
    ) * tick
    if p["sign"] * (price - new_stop) <= 2 * book.metadata[p["symbol"]]["price_tick"]:
        book.rejections["add_stop_has_no_room"] += 1
        return False
    previous_adds = p["adds"]
    book.transact("ADD", p["symbol"], p["sign"], price, when, marks)
    if p["adds"] == previous_adds:
        return False
    p["stop"] = new_stop
    p["target"] = price_barriers(
        p["entry"], p["sign"], c[f"{p['sleeve']}_target"], c[f"{p['sleeve']}_stop"], tick
    )[0]
    return True


def run_account(
    config, market, signals, metadata, start_ms, end_ms, endpoint_marks, state=None, break_ms=None
):
    book = ContractBook(config, metadata)
    quotes, operations, campaigns, events = [], [], {}, []
    day, day_start, peak, halted_until = None, config["capital"], config["capital"], 0
    circuits, rejections = Counter(), Counter()
    if state:
        book.restore(state["book"])
        quotes, operations, events = state["quotes"], state["operations"], state["events"]
        campaigns = {(p["symbol"], p["sign"]): p["started_ms"] for p in state["campaigns"]}
        day, day_start, peak, halted_until = (
            state["day"],
            state["day_start"],
            state["peak"],
            state["halted_until"],
        )
        circuits, rejections = Counter(state["circuits"]), Counter(state["rejections"])
    symbols = sorted(market)
    index = next(iter(market.values())).index
    for symbol in symbols:
        if not market[symbol].index.equals(index) or signals[symbol].shape != (len(index), 2, 4):
            raise ValueError("Market and prediction clocks differ")
    columns = ["open", "high", "low", "close", "mark", "funding", "quote60", "atr", "volume_ratio"]
    data = {s: market[s][columns].to_numpy(dtype=float) for s in symbols}

    def snapshot(next_ms):
        return {
            "book": book.snapshot(),
            "quotes": quotes,
            "operations": operations,
            "events": events,
            "campaigns": [
                {"symbol": s, "sign": sign, "started_ms": value}
                for (s, sign), value in sorted(campaigns.items())
            ],
            "day": day,
            "day_start": day_start,
            "peak": peak,
            "halted_until": halted_until,
            "circuits": dict(circuits),
            "rejections": dict(rejections),
            "next_ms": next_ms,
        }

    selected_indices = np.flatnonzero((index >= start_ms) & (index < end_ms))
    for i in selected_indices:
        when = int(index[i])
        if break_ms is not None and when >= break_ms:
            return None, snapshot(when)
        rows = {s: data[s][i] for s in symbols}
        marks = {s: r[4] for s, r in rows.items()}
        for (symbol, sign), p in sorted(book.positions.items()):
            cash = -sign * p["qty"] * marks[symbol] * rows[symbol][5]
            book.wallet += cash
            book.funding += cash
            p["funding"] += cash
        # A settled debit consumes the original dollar risk budget. Tighten after
        # observing settlement, never using a future rate or loosening protection.
        for (symbol, sign), p in list(book.positions.items()):
            if not config.get("funding_adjusted_stop", False) or rows[symbol][5] == 0:
                continue
            required = protective_stop(p, config["fee_bps"] / 10000)
            tick = metadata[symbol]["price_tick"]
            required = (
                math.ceil(required / tick - 1e-9)
                if sign == 1
                else math.floor(required / tick + 1e-9)
            ) * tick
            tightened = max(p["stop"], required) if sign == 1 else min(p["stop"], required)
            if sign * (tightened - p["stop"]) > tick * 1e-8:
                p["stop"] = tightened
                events.append(
                    {
                        "action": "FUNDING_STOP_TIGHTEN",
                        "symbol": symbol,
                        "sign": sign,
                        "executed_ms": when,
                        "new_stop": tightened,
                        "initial_risk_budget": p["risk_budget"],
                    }
                )
                if sign * (rows[symbol][0] - tightened) <= 0:
                    book.transact("CLOSE", symbol, sign, rows[symbol][0], when, marks)
                    events.append(
                        {
                            "action": "FUNDING_BUDGET_CLOSE",
                            "symbol": symbol,
                            "sign": sign,
                            "executed_ms": when,
                        }
                    )
        for operation in operations:
            if operation["due_ms"] != when:
                raise ValueError("Missing owned-operation execution clock")
            key = (operation["symbol"], operation["sign"])
            p = book.positions.get(key)
            if not p or p["id"] != operation["position_id"]:
                rejections["stale_owned_operation"] += 1
                continue
            if operation["action"] == "ADD":
                if _add(book, operation, rows[key[0]][0], when, marks):
                    events.append(
                        {
                            **operation,
                            "executed_ms": when,
                            "new_stop": p["stop"],
                            "initial_risk_budget": p["risk_budget"],
                        }
                    )
            else:
                book.transact(operation["action"], key[0], key[1], rows[key[0]][0], when, marks)
                if operation["action"] == "TRIM":
                    p["defensive_reduced"] = True
                events.append({**operation, "executed_ms": when})
        operations = []
        # Expiry is known from entry; it never uses the just-opening candle's future range.
        for (symbol, sign), p in list(book.positions.items()):
            if when >= p["expiry_ms"]:
                book.transact("CLOSE", symbol, sign, rows[symbol][0], when, marks)
                events.append(
                    {"action": "EXPIRE", "symbol": symbol, "sign": sign, "executed_ms": when}
                )
        equity = book.equity(marks)
        if equity <= 0:
            raise ValueError("Account insolvent")
        current_day = when // DAY
        if day != current_day:
            day, day_start = current_day, equity
        peak = max(peak, equity)
        drawdown = (peak - equity) / peak
        if when >= halted_until and (
            equity / day_start - 1 <= -config["daily_loss_cap"]
            or drawdown >= config["account_drawdown_cap"]
        ):
            kind = (
                "account_drawdown" if drawdown >= config["account_drawdown_cap"] else "daily_loss"
            )
            circuits[kind] += 1
            halted_until = 2**53 if kind == "account_drawdown" else (current_day + 1) * DAY
            quotes = []
            for (symbol, sign), p in sorted(book.positions.items()):
                operations.append(
                    {
                        "action": "CLOSE",
                        "symbol": symbol,
                        "sign": sign,
                        "position_id": p["id"],
                        "due_ms": when + MINUTE,
                    }
                )
        book.curve.append(
            {
                "timestamp": when,
                "equity": equity,
                "wallet": book.wallet,
                "margin": book.reserve(),
                "positions": len(book.positions),
            }
        )
        # All decisions below precede this bar's unknown intrabar fills and barriers.
        halted = when < halted_until
        pending_keys = {(q["symbol"], q["sign"]) for q in quotes}
        if not halted:
            for (symbol, sign), p in sorted(book.positions.items()):
                side = 0 if sign == 1 else 1
                prediction = signals[symbol][i, side]
                probability, expected = prediction[:2] if p["sleeve"] == "small" else prediction[2:]
                if not np.isfinite([probability, expected]).all():
                    continue
                move = sign * (marks[symbol] / p["entry"] - 1)
                age = (when - p["opened_ms"]) / MINUTE
                operation = None
                if (
                    config["allow_reductions"]
                    and not p["defensive_reduced"]
                    and not p["trimmed"]
                    and age >= 10
                    and move <= -0.003
                    and expected < 0
                    and probability < 0.45
                ):
                    operation = "TRIM"
                elif (
                    config["allow_additions"]
                    and p["adds"] < 1
                    and not p["trimmed"]
                    and age >= 5
                    and move >= 0.5 * config[f"{p['sleeve']}_target"]
                    and expected >= 0.0015
                    and probability >= 0.65
                ):
                    operation = "ADD"
                if operation:
                    operations.append(
                        {
                            "action": operation,
                            "symbol": symbol,
                            "sign": sign,
                            "position_id": p["id"],
                            "decision_ms": when,
                            "due_ms": when + MINUTE,
                        }
                    )
            proposals = []
            for symbol in symbols:
                if (
                    i == 0
                    or data[symbol][i - 1, 6] < 1_000_000
                    or not math.isfinite(data[symbol][i - 1, 7])
                ):
                    continue
                for side, sign in enumerate([1, -1]):
                    key = (symbol, sign)
                    if key in book.positions or key in pending_keys:
                        continue
                    previous = book.last_close.get(key)
                    if (
                        previous
                        and previous["net_pnl"] <= 0
                        and when - previous["when"] < config["loss_cooldown_minutes"] * MINUTE
                    ):
                        rejections["own_loss_cooldown"] += 1
                        continue
                    rearm = bool(
                        previous
                        and previous["net_pnl"] > 0
                        and when - previous["when"] < config["episode_hours"] * 3600000
                        and when - campaigns.get(key, -(2**53)) < config["episode_hours"] * 3600000
                    )
                    if (
                        previous
                        and previous["net_pnl"] > 0
                        and when - previous["when"] < config["profit_cooldown_minutes"] * MINUTE
                    ):
                        continue
                    ps, ms, pl, ml = signals[symbol][i, side]
                    if not np.isfinite([ps, ms, pl, ml]).all():
                        continue
                    # Volatility must leave room inside the registered protection distance.
                    small_ok = (
                        ps >= config["rearm_probability" if rearm else "seed_probability"]
                        and ms >= config["small_min_expected_net"]
                        and data[symbol][i - 1, 7] <= config["small_stop"] / 2
                    )
                    large_ok = (
                        pl >= config["large_probability"]
                        and ml >= config["large_min_expected_net"]
                        and data[symbol][i - 1, 6] >= 5_000_000
                        and data[symbol][i - 1, 7] <= config["large_stop"] / 2
                    )
                    if not small_ok and not large_ok:
                        continue
                    sleeve = "large" if large_ok else "small"
                    edge = ml if large_ok else ms
                    tick = metadata[symbol]["price_tick"]
                    raw = data[symbol][i - 1, 3]
                    price = (
                        math.floor(raw / tick + 1e-9) if sign == 1 else math.ceil(raw / tick - 1e-9)
                    ) * tick
                    proposals.append(
                        {
                            "symbol": symbol,
                            "sign": sign,
                            "price": price,
                            "sleeve": sleeve,
                            "priority": float(edge),
                            "role": "own_profit_rearm" if rearm else "fresh_seed",
                            "decision_ms": when,
                            "active_ms": when + MINUTE,
                            "expires_ms": when + (config["quote_ttl_minutes"] + 1) * MINUTE,
                        }
                    )
            for q in sorted(proposals, key=lambda r: (-r["priority"], r["symbol"], r["sign"])):
                if len(book.positions) + len(quotes) >= config["max_positions"]:
                    rejections["positions_and_quotes_capacity"] += 1
                    continue
                large_count = sum(p["sleeve"] == "large" for p in book.positions.values()) + sum(
                    x["sleeve"] == "large" for x in quotes
                )
                if q["sleeve"] == "large" and large_count >= config["max_large_positions"]:
                    rejections["large_slots"] += 1
                    continue
                virtual_margin = (
                    book.reserve()
                    + math.fsum(config[f"{x['sleeve']}_margin"] for x in quotes)
                    + config[f"{q['sleeve']}_margin"]
                )
                if virtual_margin > min(equity, config["capital"] * config["max_margin_fraction"]):
                    rejections["pending_margin_cap"] += 1
                    continue
                quotes.append(q)
                book.decisions.append({**q, "action": "PLACE_LIMIT"})
        remaining = []
        # First reserve/fill all eligible quotes; unknown same-bar profits cannot fund others.
        for q in sorted(quotes, key=lambda r: (-r["priority"], r["symbol"], r["sign"])):
            if when < q["active_ms"]:
                remaining.append(q)
                continue
            if when >= q["expires_ms"]:
                rejections["limit_expired"] += 1
                continue
            tick, row = metadata[q["symbol"]]["price_tick"], rows[q["symbol"]]
            crossed = row[2] < q["price"] - tick if q["sign"] == 1 else row[1] > q["price"] + tick
            if crossed:
                if _open(book, q, when, marks):
                    if q["role"] == "fresh_seed":
                        campaigns[(q["symbol"], q["sign"])] = when + MINUTE - 1
                    events.append(
                        {
                            **q,
                            "action": "LIMIT_FILL_PROXY",
                            "observed_bar_end_ms": when + MINUTE - 1,
                        }
                    )
            else:
                remaining.append(q)
        quotes = remaining
        for (symbol, sign), p in sorted(book.positions.items()):
            row, tick = rows[symbol], metadata[symbol]["price_tick"]
            stopped = row[2] <= p["stop"] if sign == 1 else row[1] >= p["stop"]
            active = when > p["entry_bar_ms"]
            hit_target = row[1] > p["target"] + tick if sign == 1 else row[2] < p["target"] - tick
            if stopped:
                price = min(row[0], p["stop"]) if sign == 1 else max(row[0], p["stop"])
                book.transact("CLOSE", symbol, sign, price, when + MINUTE - 1, marks)
                events.append(
                    {
                        "action": "STOP",
                        "symbol": symbol,
                        "sign": sign,
                        "observed_bar_end_ms": when + MINUTE - 1,
                    }
                )
                continue
            if not active:
                continue
            half_target = price_barriers(
                p["entry"],
                sign,
                0.5 * config[f"{p['sleeve']}_target"],
                config[f"{p['sleeve']}_stop"],
                tick,
            )[0]
            hit_half = row[1] > half_target + tick if sign == 1 else row[2] < half_target - tick
            if config["allow_reductions"] and not p["trimmed"] and hit_half:
                book.transact("TRIM", symbol, sign, half_target, when + MINUTE - 1, marks)
                events.append(
                    {
                        "action": "PROFIT_TRIM",
                        "symbol": symbol,
                        "sign": sign,
                        "observed_bar_end_ms": when + MINUTE - 1,
                    }
                )
            if hit_target:
                book.transact("CLOSE", symbol, sign, p["target"], when + MINUTE - 1, marks)
                events.append(
                    {
                        "action": "TARGET",
                        "symbol": symbol,
                        "sign": sign,
                        "observed_bar_end_ms": when + MINUTE - 1,
                    }
                )
    if set(endpoint_marks) != set(symbols):
        raise ValueError("Exact endpoint marks required; no stale terminal valuation")
    terminal = book.equity(endpoint_marks)
    book.curve.append(
        {
            "timestamp": end_ms,
            "equity": terminal,
            "wallet": book.wallet,
            "margin": book.reserve(),
            "positions": len(book.positions),
        }
    )
    result = score(book, endpoint_marks)
    result.update(
        trading_fee_component=book.fees * config["trading_fee_bps"] / config["fee_bps"],
        modeled_execution_cost_component=book.fees
        * config["execution_cost_bps"]
        / config["fee_bps"],
        fee_field_includes_explicit_execution_cost_reserve=True,
        limit_fill_never_above_buy_or_below_sell_cap=True,
        queue_fill_and_liquidation_model_verified=False,
        loss_circuits=dict(circuits),
        policy_rejections=dict(rejections),
        open_limit_quotes=len(quotes),
        event_counts=dict(Counter(e["action"] for e in events)),
        decisions_use_own_state_and_previous_complete_bars=True,
    )
    for sleeve, details in result["sleeves"].items():
        rows = [p for p in book.closed if p["sleeve"] == sleeve]
        wins = [p["net_pnl"] for p in rows if p["net_pnl"] > 0]
        losses = [-p["net_pnl"] for p in rows if p["net_pnl"] < 0]
        details.update(
            mean_win=math.fsum(wins) / len(wins) if wins else None,
            mean_loss=math.fsum(losses) / len(losses) if losses else None,
            worst_loss=max(losses) if losses else None,
            gap_or_funding_risk_budget_breaches=int(
                sum(p["net_pnl"] < -p["risk_budget"] - 1e-6 for p in rows)
            ),
        )
    return result, snapshot(end_ms)


def evaluate_candidates(market, signals, metadata, output, protocol, transfer_market=None):
    rows = candidates(protocol["economics"])
    save(
        output / "candidate_freeze.json",
        {
            "registered_utc": datetime.now(UTC).isoformat(),
            "candidates": rows,
            "models_sha256": sha(output / "models.pkl"),
            "leader_actions_in_runtime": False,
        },
    )
    validation_start, evaluation_start, end = (
        millis("2026-10-10T00:00:00+00:00"),
        millis("2026-10-14T00:00:00+00:00"),
        millis("2026-10-24T00:00:00+00:00"),
    )
    validation_marks = {s: float(f.loc[evaluation_start, "mark"]) for s, f in market.items()}
    all_endpoint = json.loads((output.parent / "combined_evaluation_marks.json").read_text())
    endpoint = {
        "timestamp": all_endpoint["timestamp"],
        "marks": {s: all_endpoint["marks"][s] for s in market},
    }
    if endpoint["timestamp"] != end:
        raise ValueError("Endpoint source clock differs")
    scores = []
    for c in rows:
        result, _ = run_account(
            c, market, signals, metadata, validation_start, evaluation_start, validation_marks
        )
        small, large = result["sleeves"]["small"], result["sleeves"]["large"]
        result["selection_gate"] = bool(
            small["net_pnl"] > 0
            and large["net_pnl"] > 0
            and small["complete_positions"] >= 20
            and large["complete_positions"] >= 5
            and result["minute_observed_mark_open_drawdown"] <= 0.08
        )
        scores.append(result)
        print(
            {
                "validation": c["id"],
                "net_return": result["net_return"],
                "small": small["net_pnl"],
                "large": large["net_pnl"],
                "gate": result["selection_gate"],
            },
            flush=True,
        )
    save(output / "validation_scores.json", scores)
    qualified = [r for r in scores if r["selection_gate"]]
    best = sorted(qualified or scores, key=lambda r: (-r["net_return"], r["candidate"]))[0]
    policy = next(c for c in rows if c["id"] == best["candidate"])
    save(
        output / "selection.json",
        {
            "candidate": policy,
            "validation_gate": bool(qualified),
            "evaluation_outcomes_read": False,
            "scope": "historical research candidate"
            if qualified
            else "diagnostic best candidate only",
        },
    )
    result, checkpoint = run_account(
        policy, market, signals, metadata, evaluation_start, end, endpoint["marks"]
    )
    save(output / "evaluation_score.json", result)
    save(output / "evaluation_state.json", checkpoint)
    middle = evaluation_start + (end - evaluation_start) // (2 * MINUTE) * MINUTE
    _, prefix = run_account(
        policy, market, signals, metadata, evaluation_start, end, endpoint["marks"], break_ms=middle
    )
    resumed, state = run_account(
        policy,
        market,
        signals,
        metadata,
        middle,
        end,
        endpoint["marks"],
        state=json.loads(json.dumps(prefix)),
    )
    if result != resumed or checkpoint != state:
        raise ValueError("Owned-limit checkpoint changes account or score")
    save(
        output / "restart_verification.json",
        {"exact_state_equal": True, "exact_score_equal": True, "checkpoint_ms": middle},
    )
    stresses = []
    for extra in [2, 5]:
        c = {
            **policy,
            "execution_cost_bps": policy["execution_cost_bps"] + extra,
            "fee_bps": policy["fee_bps"] + extra,
        }
        stressed, _ = run_account(
            c, market, signals, metadata, evaluation_start, end, endpoint["marks"]
        )
        stresses.append({"extra_cost_bps_each_side": extra, **stressed})
    save(output / "cost_stress.json", stresses)
    transfer_score = None
    if transfer_market:
        transfer_marks = {s: all_endpoint["marks"][s] for s in transfer_market}
        transfer_score, transfer_state = run_account(
            policy, transfer_market, signals, metadata, evaluation_start, end, transfer_marks
        )
        save(output / "transfer_score.json", transfer_score)
        save(output / "transfer_state.json", transfer_state)
    small, large = result["sleeves"]["small"], result["sleeves"]["large"]
    passed = bool(
        qualified
        and small["net_pnl"] > 0
        and large["net_pnl"] > 0
        and small["complete_positions"] >= 30
        and large["complete_positions"] >= 15
        and result["minute_observed_mark_open_drawdown"] <= 0.08
        and all(x["net_pnl"] > 0 for x in stresses[0]["sleeves"].values())
    )
    save(
        output / "run_status.json",
        {
            "status": "completed",
            "paper_research_gate": passed,
            "production_eligible": False,
            "orders_enabled": False,
            "genuinely_unseen_forward_evidence": False,
        },
    )
    print({"evaluation": result, "paper_research_gate": passed}, flush=True)
