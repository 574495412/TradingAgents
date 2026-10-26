"""Audit recorded signals/requests using scalar rules, never replay accounts.

The expected side uses original decimal strings, calendar months and fractions.
The producer is invoked only to expose frozen target books for comparison. No
application initializers, dotenv, broker APIs or new strategy variants are used.
"""

import argparse
import csv
import hashlib
import json
import sys
import types
from collections import defaultdict
from datetime import date, datetime
from decimal import ROUND_FLOOR, Decimal
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PLAN = ROOT / "docs/research/equities_pit_stocks_20261026/signal-order-contract-plan.json"
for name in (
    "tradingagents",
    "tradingagents.research",
    "tradingagents.research.equities",
    "factors",
    "factors.equity",
    "factors.common",
):
    package = types.ModuleType(name)
    package.__path__ = [str(ROOT.joinpath(*name.split(".")))]
    sys.modules[name] = package

import pandas as pd  # noqa: E402

from factors.equity.disclosed_stocks import targets  # noqa: E402
from tradingagents.research.equities.audit_v002 import expected_fee  # noqa: E402

assert "dotenv" not in sys.modules


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def D(value):
    return Decimal(str(value))


def month_number(day):
    return day.year * 12 + day.month - 1


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    plan = read(PLAN)
    parent = ROOT / plan["source_run"]
    parent_reg = read(parent / "registration.json")
    assert all(sha(ROOT / p) == h for p, h in parent_reg["inputs_sha256"].items())
    pit = ROOT / "data/research/equity_pit_20261026_v001"
    strategy, identities, annual = [
        read(pit / n) for n in ("protocol.json", "identities.json", "annual.json")
    ]
    symbols = [r["symbol"] for r in identities]
    with (pit / "prices/signal_sessions.csv").open(newline="") as handle:
        sessions = {date.fromisoformat(r["date"]): r for r in csv.DictReader(handle)}
    days = sorted(sessions)
    calendar = pd.read_csv(
        pit / "prices/signal_sessions.csv", index_col="date", parse_dates=["date"]
    )
    for key in ("open", "close"):
        calendar[key] = pd.to_datetime(calendar[key], utc=True)
    source, frames = {}, {}
    for symbol in symbols:
        path = pit / "prices" / (symbol + "_signal.csv")
        with path.open(newline="") as handle:
            source[symbol] = {
                date.fromisoformat(r["date"]): {k: D(v) for k, v in r.items() if k != "date"}
                for r in csv.DictReader(handle)
            }
        frames[symbol] = pd.read_csv(path, index_col="date", parse_dates=["date"])
    observed_books, _, _ = targets(frames, calendar, identities, annual, strategy)
    month_ends = {}
    for day in days:
        month_ends[month_number(day)] = day
    first, last = (
        date.fromisoformat(plan["first_evaluation"]),
        date.fromisoformat(plan["last_evaluation"]),
    )
    previous_decision = max(d for d in month_ends.values() if d < first)
    decisions = sorted([previous_decision, *[d for d in month_ends.values() if first <= d <= last]])
    expected_books, decision_rows, failures = {}, [], []
    policies = strategy["policies"]
    eligible_annual_inputs = 0
    for decision in decisions:
        period = month_number(decision)
        left, right = month_ends[period - 12], month_ends[period - 1]
        scores, quality, details = {}, {}, []
        clock = datetime.fromisoformat(sessions[decision]["close"])
        for identity in identities:
            symbol = identity["symbol"]
            available = source[symbol]
            if left in available and right in available:
                interval = [d for d in days if left < d <= right]
                if all(d in available for d in interval):
                    if all(available[d]["dividend"] == 0 for d in interval):
                        value = (
                            Fraction(available[right]["close"]) / Fraction(available[left]["close"])
                            - 1
                        )
                    else:
                        value, prior_close = Fraction(1), available[left]["close"]
                        for d in interval:
                            value *= Fraction(
                                available[d]["close"] + available[d]["dividend"]
                            ) / Fraction(prior_close)
                            prior_close = available[d]["close"]
                        value -= 1
                    scores[symbol] = value
            eligible = [
                r
                for r in annual
                if r["cik"] == identity["cik"]
                and datetime.fromisoformat(r["available_at"]) <= clock
                and 0 <= (decision - date.fromisoformat(r["period_end"])).days <= 550
            ]
            record = (
                max(eligible, key=lambda r: (r["period_end"], r["available_at"], r["accession"]))
                if eligible
                else None
            )
            if record and record["status"] == "available":
                ratio = Fraction(D(record["gross_profit"])) / Fraction(D(record["assets"]))
                quality[symbol] = ratio
                eligible_annual_inputs += 1
                if (
                    abs(float(ratio) - record["gross_profitability"])
                    > plan["score_absolute_tolerance"]
                ):
                    failures.append(
                        {"kind": "annual_ratio", "day": str(decision), "symbol": symbol}
                    )
            details.append(
                {
                    "symbol": symbol,
                    "momentum_score": float(scores[symbol]) if symbol in scores else None,
                    "annual_accession": record["accession"] if record else None,
                    "annual_available_at": record["available_at"] if record else None,
                    "annual_status": record["status"] if record else None,
                    "quality_score": float(quality[symbol]) if symbol in quality else None,
                }
            )
        momentum_winners = [
            s for s in sorted(scores, key=lambda s: (-scores[s], s)) if scores[s] > 0
        ][:3]
        quality_winners = [
            s for s in sorted(quality, key=lambda s: (-quality[s], s)) if quality[s] > 0
        ][:3]
        books = {p: {s: Fraction(0) for s in symbols} for p in policies}
        books["disclosed_equal"] = dict.fromkeys(symbols, Fraction(4, 5 * len(symbols)))
        for s in momentum_winners:
            books["french_mom_12_2"][s] = Fraction(4, 15)
        for s in quality_winners:
            books["gross_profitability"][s] = Fraction(4, 15)
        books["stock_blend_equal"] = {
            s: (books["french_mom_12_2"][s] + books["gross_profitability"][s]) / 2 for s in symbols
        }
        expected_books[decision] = books
        for policy in policies:
            for symbol in symbols:
                observed = observed_books[policy].at[pd.Timestamp(decision), symbol]
                if abs(float(books[policy][symbol]) - observed) > plan["weight_absolute_tolerance"]:
                    failures.append(
                        {
                            "kind": "monthly_weight",
                            "day": str(decision),
                            "policy": policy,
                            "symbol": symbol,
                            "expected": str(books[policy][symbol]),
                            "observed": float(observed),
                        }
                    )
        decision_rows.append(
            {
                "decision": str(decision),
                "decision_close": clock.isoformat(),
                "momentum_denominator_close": str(left),
                "momentum_numerator_close": str(right),
                "holding_month": f"{(period + 1) // 12:04d}-{(period + 1) % 12 + 1:02d}",
                "momentum_winners": momentum_winners,
                "quality_winners": quality_winners,
                "inputs": details,
            }
        )
    day_indices = {d: i for i, d in enumerate(days)}
    evaluations = [d for d in days if first <= d <= last]
    daily_weight_checks = 0
    for day in [previous_decision, *evaluations]:
        decision = max(d for d in decisions if d <= day)
        for policy in policies:
            for s in symbols:
                daily_weight_checks += 1
                if (
                    abs(
                        float(expected_books[decision][policy][s])
                        - observed_books[policy].at[pd.Timestamp(day), s]
                    )
                    > plan["weight_absolute_tolerance"]
                ):
                    failures.append(
                        {"kind": "daily_weight", "day": str(day), "policy": policy, "symbol": s}
                    )
    accounts, integer_boundaries = [], []
    for row in read(parent / "summary.json")["records"]:
        path = parent / row["name"] / "state.json"
        state = read(path)
        cash, inventory, pending = D(10000), dict.fromkeys(symbols, 0), []
        fills, requests = defaultdict(dict), defaultdict(dict)
        for fill in state["fills"]:
            assert fill["symbol"] not in fills[fill["day"]]
            fills[fill["day"]][fill["symbol"]] = fill
        for request in state["requests"]:
            assert request["symbol"] not in requests[request["day"]]
            requests[request["day"]][request["symbol"]] = request
        actual_requests = filled_count = off_cycle_fills = 0
        for previous_nav, observation in zip(
            state["equity_curve"][:-1], state["equity_curve"][1:], strict=True
        ):
            day = date.fromisoformat(observation["day"])
            idx, current_fills = day_indices[day], fills[str(day)]
            prior = days[idx - 1]
            cash += sum((v for release, v in pending if release <= idx), D(0))
            pending = [(release, value) for release, value in pending if release > idx]
            decision = max(d for d in decisions if d <= prior)
            wanted = {}
            for s in symbols:
                weight = expected_books[decision][row["policy"]][s]
                ratio = (
                    D(weight.numerator)
                    / D(weight.denominator)
                    * D(previous_nav["equity"])
                    / source[s][prior]["close"]
                )
                target = int(ratio.to_integral_value(rounding=ROUND_FLOOR))
                if weight > 0 and abs(ratio - ratio.to_integral_value()) < D("1e-10"):
                    integer_boundaries.append(
                        {
                            "account": row["name"],
                            "day": str(day),
                            "symbol": s,
                            "target_ratio": str(ratio),
                        }
                    )
                difference = target - inventory[s]
                if target and abs(difference) * source[s][prior]["close"] < D(".05") * D(
                    previous_nav["equity"]
                ):
                    difference = 0
                history = days[idx - 20 : idx]
                assert len(history) == 20 and all(d in source[s] for d in history)
                cap = int(
                    (
                        sum((source[s][d]["volume"] for d in history), D(0)) / 20 / 1000
                    ).to_integral_value(rounding=ROUND_FLOOR)
                )
                capped = min(abs(difference), cap) * (1 if difference > 0 else -1)
                if difference:
                    wanted[s] = (difference, capped)
            for s in set(wanted) | set(requests[str(day)]):
                actual_requests += 1
                request = requests[str(day)].get(s)
                expected = wanted.get(s)
                if (
                    request is None
                    or expected is None
                    or (request["requested"], request["volume_capped"]) != expected
                ):
                    failures.append(
                        {
                            "kind": "integer_request",
                            "account": row["name"],
                            "day": str(day),
                            "symbol": s,
                            "expected": expected,
                            "actual": request,
                        }
                    )
            # Audit the recorded fills in the frozen sell-then-buy identity order.
            for side in ("SELL", "BUY"):
                for s in symbols:
                    difference, capped = wanted.get(s, (0, 0))
                    if (side == "SELL" and capped >= 0) or (side == "BUY" and capped <= 0):
                        continue
                    price = D(
                        float(source[s][day]["open"])
                        * (1 + row["slippage_bps"] / 10000 * (1 if side == "BUY" else -1))
                    )
                    qty = (
                        min(-capped, inventory[s])
                        if side == "SELL"
                        else min(
                            capped, int((cash / price).to_integral_value(rounding=ROUND_FLOOR))
                        )
                    )
                    if side == "BUY":
                        while qty > 0 and qty * price + sum(
                            expected_fee(
                                "US", str(day), side, qty, float(price), state["binding"]["spec"]
                            ).values(),
                            D(0),
                        ) > cash + D("1e-9"):
                            qty -= 1
                    actual = current_fills.get(s)
                    if (
                        qty > 0
                        and (actual is None or actual["quantity"] != qty or actual["side"] != side)
                    ) or (qty == 0 and actual is not None):
                        failures.append(
                            {
                                "kind": "affordable_fill_quantity",
                                "account": row["name"],
                                "day": str(day),
                                "symbol": s,
                                "expected_quantity": qty,
                                "actual": actual,
                            }
                        )
                    if actual:
                        fee = sum(
                            expected_fee(
                                "US",
                                str(day),
                                side,
                                actual["quantity"],
                                actual["price"],
                                state["binding"]["spec"],
                            ).values(),
                            D(0),
                        )
                        inventory[s] += actual["quantity"] * (1 if side == "BUY" else -1)
                        if side == "BUY":
                            cash -= D(actual["quantity"]) * D(actual["price"]) + fee
                        else:
                            pending.append(
                                (idx + 3, D(actual["quantity"]) * D(actual["price"]) - fee)
                            )
                        filled_count += 1
                        off_cycle_fills += prior not in decisions
            if abs(cash - D(observation["cash"])) > D(".00001"):
                failures.append(
                    {"kind": "cash_after_observed_fills", "account": row["name"], "day": str(day)}
                )
        accounts.append(
            {
                "account": row["name"],
                "request_checks": actual_requests,
                "recorded_fills_checked": filled_count,
                "recorded_fill_count": len(state["fills"]),
                "fills_outside_first_session_after_month_end": off_cycle_fills,
            }
        )
        if filled_count != len(state["fills"]):
            failures.append({"kind": "unprocessed_fill", "account": row["name"]})
    inputs = {str(PLAN.relative_to(ROOT)): sha(PLAN), **parent_reg["inputs_sha256"]}
    for path in [
        pit / "prices/signal_sessions.csv",
        parent / "registration.json",
        parent / "completion.json",
        parent / "summary.json",
        *parent.glob("*/state.json"),
    ]:
        inputs[str(path.relative_to(ROOT))] = sha(path)
    for path in (ROOT / "data/research/equity_signal_contract_20261027_v001").glob("*"):
        inputs[str(path.relative_to(ROOT))] = sha(path)
    write(output / "monthly-inputs.json", decision_rows)
    write(output / "account-checks.json", accounts)
    write(output / "mismatches.json", failures)
    write(output / "integer-boundaries.json", integer_boundaries)
    result = {
        "schema": "equity-signal-order-contract-audit.v1",
        "monthly_decisions": len(decisions),
        "eligible_annual_inputs": eligible_annual_inputs,
        "daily_weight_checks": daily_weight_checks,
        "integer_request_checks": sum(r["request_checks"] for r in accounts),
        "recorded_fill_checks": sum(r["recorded_fills_checked"] for r in accounts),
        "mismatches": len(failures),
        "internal_contract_checks_passed": not failures,
        "full_paper_replication": False,
        "producer_score_function_reused_on_expected_side": False,
        "new_financial_accounts": 0,
        "new_hypotheses": 0,
        "external_attestation": False,
        "profit_goal_met": False,
        "scope": "Frozen ten-stock mathematical adaptation and recorded target/execution logic only. Not market authenticity, original market-wide factor performance, causal alpha or independent financial acceptance.",
    }
    write(output / "summary.json", result)
    write(
        output / "registration.json",
        {
            "inputs_sha256": inputs,
            "analysis_sha256": sha(Path(__file__)),
            "expected_methods": "Scalar calendar months, original decimal strings, Fraction ranks, Decimal target/cash/volume; no financial replay engine called",
            "registration_scope": "Current audit input consistency only; these hashes do not establish original data publication time or retroactively seal earlier financial evaluations.",
            "external_attestation": False,
        },
    )
    (output / "analysis_source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
