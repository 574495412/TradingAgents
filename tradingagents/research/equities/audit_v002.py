"""Independent Decimal ledger reconstruction. Does not import either engine.

Audit confirms arithmetic against captured public inputs, not market authenticity.
"""

from collections import defaultdict
from decimal import ROUND_CEILING, ROUND_HALF_UP, Decimal


def D(value):
    return Decimal(str(value))


def expected_fee(market, date, side, q, price, spec):
    value = D(q) * D(price)
    cent = Decimal(".01")

    def rounded(x, upward=False):
        return x.quantize(cent, rounding=ROUND_CEILING if upward else ROUND_HALF_UP)

    result = {
        "commission": rounded(
            max(D(spec["minimum_fee"]), value * D(spec["commission_bps"]) / 10000)
        )
    }
    if market == "HK":
        for name, rate in [("sfc", ".000027"), ("afrc", ".0000015"), ("exchange", ".0000565")]:
            result[name] = rounded(value * D(rate))
        result["stamp"] = D(0)
        result["settlement"] = rounded(
            value * D(".000042")
            if date >= "2025-06-30"
            else min(D(100), max(D(2), value * D(".00002")))
        )
    else:
        if date < "2023-02-27":
            sec = "22.9"
        elif date < "2024-05-22":
            sec = "8"
        elif date < "2025-05-14":
            sec = "27.8"
        elif date < "2026-04-04":
            sec = "0"
        else:
            sec = "20.6"
        taf, cap = (
            (".000145", "7.27")
            if date[:4] == "2023"
            else (".000195", "9.79")
            if date[:4] == "2026"
            else (".000166", "8.30")
        )
        result["sec31"] = rounded(value * D(sec) / 1000000, True) if side == "SELL" else D(0)
        result["taf"] = rounded(min(D(cap), D(q) * D(taf)), True) if side == "SELL" else D(0)
    return result


def audit(state, frames, calendar):
    binding = state["binding"]
    spec, market = binding["spec"], binding["market"]
    cash, receivable = D(spec["initial_cash"]), D(0)
    quantities, cycle = defaultdict(int), defaultdict(lambda: D(0))
    remaining_cost = defaultdict(lambda: D(0))
    pending, closed = [], []
    fills = defaultdict(list)
    for f in state["fills"]:
        fills[f["day"]].append(f)
    maximum_error = D(0)
    fees = defaultdict(lambda: D(0))
    lots = {s["symbol"]: s["lot_size"] for s in spec["symbols"]}
    dividends = []
    peak = D(spec["initial_cash"])
    max_low = D(0)
    for row in state["equity_curve"][1:]:
        day = row["day"]
        import pandas as pd

        date = pd.Timestamp(day)
        idx = calendar.index.get_loc(date)
        cash += sum((amount for release, amount in pending if release <= idx), D(0))
        pending = [(release, amount) for release, amount in pending if release > idx]
        for s, q in quantities.items():
            amount = D(q) * D(frames[s].at[date, "dividend"]) * (1 - D(spec["dividend_haircut"]))
            if amount:
                receivable += amount
                cycle[s] += amount
                dividends.append((s, day, q, amount))
        for fill in fills[day]:
            s, side, q = fill["symbol"], fill["side"], fill["quantity"]
            assert isinstance(q, int) and q > 0 and q % lots[s] == 0
            assert idx > 0 and fill["decision_day"] == str(calendar.index[idx - 1].date())
            assert fill["clock"] == int(calendar.at[date, "open"].timestamp() * 1000)
            raw = D(frames[s].at[date, "open"])
            expected = raw * (1 + D(binding["slippage_bps"]) / 10000 * (1 if side == "BUY" else -1))
            assert abs(D(fill["price"]) - expected) < D(".000000001")
            parts = expected_fee(market, day, side, q, fill["price"], spec)
            assert {k: D(v) for k, v in fill["fee_parts"].items()} == parts
            fee = sum(parts.values(), D(0))
            assert abs(D(fill["fee"]) - fee) < D(".000000001")
            for k, v in parts.items():
                fees[k] += v
            value = D(q) * D(fill["price"])
            if side == "BUY":
                cash -= value + fee
                quantities[s] += q
                cycle[s] -= value + fee
                remaining_cost[s] += value + fee
                assert cash >= D("-.0000001")
            else:
                assert side == "SELL" and quantities[s] >= q
                remaining_cost[s] *= 1 - D(q) / D(quantities[s])
                quantities[s] -= q
                pending.append((idx + 3, value - fee))
                cycle[s] += value - fee
                if not quantities[s]:
                    closed.append((s, fill["clock"], cycle[s]))
                    cycle[s] = D(0)
        unsettled = sum((v for _, v in pending), D(0))
        balance = cash + unsettled + receivable
        eq = balance + sum(
            (D(q) * D(frames[s].at[date, "close"]) for s, q in quantities.items()), D(0)
        )
        low = balance + sum(
            (D(q) * D(frames[s].at[date, "low"]) for s, q in quantities.items()), D(0)
        )
        reserve = D(0)
        for s, q in quantities.items():
            if q:
                mark = float(frames[s].at[date, "close"])
                slip = binding["slippage_bps"] / 10000
                reserve += D(q) * D(mark) * D(slip) + sum(
                    expected_fee(market, day, "SELL", q, mark * (1 - slip), spec).values(), D(0)
                )
        max_low = max(max_low, 1 - (low - reserve) / peak, 1 - eq / peak)
        peak = max(peak, eq)
        for observed, reconstructed in [
            (row["equity"], eq),
            (row["cash"], cash),
            (row["unsettled"], unsettled),
            (row["receivables"], receivable),
            (row["low_equity"], low),
            (row["exit_reserve"], reserve),
        ]:
            error = abs(D(observed) - reconstructed)
            maximum_error = max(maximum_error, error)
            assert error < D(".00001"), (day, error)
        assert abs(D(row["drawdown"]) - (1 - eq / peak)) < D(".00000001")
    assert len(closed) == len(state["trades"])
    for (s, clock, pnl), trade in zip(closed, state["trades"], strict=True):
        assert s == trade["symbol"] and clock == trade["exit_ms"]
        assert abs(pnl - D(trade["net_pnl"])) < D(".00001")
    assert {s: q for s, q in quantities.items() if q} == {
        s: p["quantity"] for s, p in state["positions"].items()
    }
    for s, p in state["positions"].items():
        assert abs(cycle[s] - D(p["cycle_net"])) < D(".00001")
        assert abs(remaining_cost[s] - D(p["remaining_cost"])) < D(".00001")
    assert abs(max_low - D(state["max_low_dd"])) < D(".00000001")
    assert len(dividends) == len(state["dividends"])
    for (s, day, q, amount), event in zip(dividends, state["dividends"], strict=True):
        assert (s, day, q) == (event["symbol"], event["day"], event["quantity"])
        assert abs(amount - D(event["amount"])) < D(".00001")
    for k, v in fees.items():
        assert abs(v - D(state["fees"][k])) < D(".00001")
    return {
        "passed": True,
        "method": "independent Decimal, no engine imports",
        "curve_points": len(state["equity_curve"]) - 1,
        "fills": len(state["fills"]),
        "closed_cycles": len(closed),
        "max_absolute_balance_error": str(maximum_error),
        "source_certified": False,
    }
