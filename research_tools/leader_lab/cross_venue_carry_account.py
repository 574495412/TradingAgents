"""Two-native-wallet carry diagnostic, with actual rates and explicit HL mark proxy."""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .damorne_payoff import metadata_with_ticks
from .damorne_research import millis
from .evidence import save, sha

HOUR = 3600000


def load_data(root, parent):
    source = root / "carry_expansion_v003/cross_venue_sources"
    receipt = json.loads((source / "capture_reconciliation_v002.json").read_text())
    fx = {int(r[0]): float(r[1]) for r in json.loads((source / "USDCUSDT_hourly.json").read_text())}
    funding = json.loads((parent / "combined_data/funding_sources.json").read_text())
    cache = parent / "combined_mechanisms/market_cache"
    manifest = json.loads((cache / "manifest.json").read_text())
    hl_meta = json.loads((root / "hyperliquid_meta.json").read_text())
    hl_steps = {r["name"]: 10 ** -int(r["szDecimals"]) for r in hl_meta["universe"]}
    bn_meta = metadata_with_ticks(
        json.loads((parent / "combined_data/discovery/catalog.json").read_text())
    )
    data = {}
    for row in receipt["records"]:
        coin = row["coin"]
        b = next(x for x in funding if x["symbol"] == coin + "USDT")
        bpath = Path(b["path"])
        m = next(x for x in manifest if x["symbol"] == coin + "USDT")
        mpath = cache / m["file"]
        if sha(bpath) != b["sha256"] or sha(mpath) != m["sha256"]:
            raise ValueError("Binance carry input changed")
        for kind in ["funding", "candles"]:
            if sha(source / f"{coin}_{kind}.json") != row[kind + "_sha256"]:
                raise ValueError("Hyperliquid carry input changed")
        with np.load(mpath, allow_pickle=False) as values:
            columns = {c: i for i, c in enumerate(m["columns"])}
            prices = {
                int(t): {"open": float(v[columns["open"]]), "mark": float(v[columns["mark"]])}
                for t, v in zip(values["index"], values["values"], strict=True)
                if int(t) % HOUR == 0
            }
        data[coin] = {
            "bn_prices": prices,
            "hl_prices": {
                int(r["t"]): float(r["o"])
                for r in json.loads((source / f"{coin}_candles.json").read_text())
            },
            "bn_funding": json.loads(bpath.read_text()),
            "hl_funding": json.loads((source / f"{coin}_funding.json").read_text()),
            "bn_meta": bn_meta[coin + "USDT"],
            "hl_step": hl_steps[coin],
        }
    terminal = json.loads((parent / "combined_evaluation_marks.json").read_text())
    for coin, d in data.items():
        # The last minute is not the exact endpoint. Only the captured true mark
        # is inserted at the terminal clock; it is never a past trading input.
        d["terminal_bn_mark"] = terminal["marks"][coin + "USDT"]
    return data, fx


def equity(state, data, fx, when, rate, terminal=False):
    value = state["wallet_bn"] + state["wallet_hl"] * fx[when]
    for coin, p in sorted(state["positions"].items()):
        b = data[coin]["terminal_bn_mark"] if terminal else data[coin]["bn_prices"][when]["mark"]
        h = data[coin]["hl_prices"][when]
        value += p["qty"] * p["sign_bn"] * (b - p["entry_bn"])
        value += p["qty"] * -p["sign_bn"] * (h - p["entry_hl"]) * fx[when]
        value -= p["qty"] * (b + h * fx[when]) * rate
    return value


def run(policy, data, fx, contexts, start, end, state=None, break_ms=None):
    capital, rate = policy["capital"], policy["cost_bps_per_fill"] / 10000
    if state is None:
        state = {
            "wallet_bn": capital / 2,
            "wallet_hl": capital / 2 / fx[start],
            "initial_bn": capital / 2,
            "initial_hl": capital / 2 / fx[start],
            "positions": {},
            "closed": [],
            "pending": [],
            "curve": [],
            "counter": 0,
            "peak": capital,
            "halted": False,
            "next_clock": start,
            "rejections": {},
            "events": [],
        }
    else:
        state = copy.deepcopy(state)
        if state["next_clock"] != start:
            raise ValueError("Carry checkpoint clock mismatch")
    rejects = Counter(state["rejections"])

    def prices(coin, when):
        return data[coin]["bn_prices"][when]["open"], data[coin]["hl_prices"][when]

    def close(coin, when, reason):
        p = state["positions"].pop(coin)
        b, h = prices(coin, when)
        gb, gh = (
            p["sign_bn"] * p["qty"] * (b - p["entry_bn"]),
            -p["sign_bn"] * p["qty"] * (h - p["entry_hl"]),
        )
        fb, fh = p["qty"] * b * rate, p["qty"] * h * rate
        state["wallet_bn"] += gb - fb
        state["wallet_hl"] += gh - fh
        p["fees_bn"] += fb
        p["fees_hl"] += fh
        nb, nh = gb + p["funding_bn"] - p["fees_bn"], gh + p["funding_hl"] - p["fees_hl"]
        state["closed"].append(
            {
                **p,
                "closed_ms": when + 1000,
                "exit_reason": reason,
                "net_bn_usdt": nb,
                "net_hl_usdc": nh,
                "exit_fx": fx[when],
                "net_usdt_at_close": nb + nh * fx[when],
            }
        )

    for when in range(start, end, HOUR):
        if break_ms is not None and when >= break_ms:
            state["next_clock"], state["rejections"] = when, dict(rejects)
            return None, state
        # Hourly executions are modelled at open+1s. Current settlements at
        # 0..139ms precede fills; a newly opened hedge cannot claim that coupon.
        for coin, p in sorted(state["positions"].items()):
            d = data[coin]
            for r in d["bn_funding"]:
                t = int(r["fundingTime"])
                if max(p["opened_ms"], when - HOUR + 1000) < t <= when + 1000:
                    payment = (
                        -p["sign_bn"] * p["qty"] * float(r["markPrice"]) * float(r["fundingRate"])
                    )
                    p["funding_bn"] += payment
                    state["wallet_bn"] += payment
            for r in d["hl_funding"]:
                t = int(r["time"])
                if max(p["opened_ms"], when - HOUR + 1000) < t <= when + 1000:
                    # HL historical mark is unavailable: actual rate times the
                    # contemporaneous hourly trade-open proxy is explicit.
                    price = d["hl_prices"][t // HOUR * HOUR]
                    payment = p["sign_bn"] * p["qty"] * price * float(r["fundingRate"])
                    p["funding_hl"] += payment
                    state["wallet_hl"] += payment
        due = state["pending"]
        state["pending"] = []
        for order in due:
            if order["due_ms"] != when:
                raise ValueError("Carry intent execution clock gap")
            coin = order["coin"]
            if order["action"] == "CLOSE":
                if coin in state["positions"]:
                    close(coin, when, order["reason"])
                continue
            if state["halted"] or coin in state["positions"] or len(state["positions"]) >= 3:
                rejects["entry_capacity_or_halt"] += 1
                continue
            b, h = prices(coin, when)
            step = max(data[coin]["bn_meta"]["step"], data[coin]["hl_step"])
            q = math.floor(capital * 0.1 / ((b + h * fx[when]) / 2) / step + 1e-9) * step
            if (
                q <= 0
                or q < data[coin]["bn_meta"]["min_qty"]
                or q * b < data[coin]["bn_meta"]["min_notional"]
                or q * h < 10
                or abs(q / data[coin]["hl_step"] - round(q / data[coin]["hl_step"])) > 1e-6
            ):
                rejects["lot_or_min_notional"] += 1
                continue
            reserved_bn = sum(p["qty"] * p["entry_bn"] / 2 for p in state["positions"].values())
            reserved_hl = sum(p["qty"] * p["entry_hl"] / 2 for p in state["positions"].values())
            fb, fh = q * b * rate, q * h * rate
            if (
                reserved_bn + q * b / 2 + fb > state["wallet_bn"]
                or reserved_hl + q * h / 2 + fh > state["wallet_hl"]
            ):
                rejects["per_venue_margin"] += 1
                continue
            state["counter"] += 1
            state["wallet_bn"] -= fb
            state["wallet_hl"] -= fh
            state["positions"][coin] = {
                "id": state["counter"],
                "coin": coin,
                "sign_bn": -1 if order["short_binance_long_hl"] else 1,
                "qty": q,
                "entry_bn": b,
                "entry_hl": h,
                "opened_ms": when + 1000,
                "expiry_ms": when + policy["horizon_hours"] * HOUR,
                "fees_bn": fb,
                "fees_hl": fh,
                "funding_bn": 0.0,
                "funding_hl": 0.0,
            }
        for coin, p in sorted(state["positions"].items()):
            if when >= p["expiry_ms"]:
                close(coin, when, "registered_expiry")
        e = equity(state, data, fx, when, rate)
        state["peak"] = max(state["peak"], e)
        if e <= 0:
            raise ValueError("Carry account insolvent")
        if (state["peak"] - e) / state["peak"] >= 0.08:
            state["halted"] = True
        closing = set()
        for coin, p in sorted(state["positions"].items()):
            b, h = data[coin]["bn_prices"][when]["mark"], data[coin]["hl_prices"][when]
            pair = (
                p["sign_bn"] * p["qty"] * (b - p["entry_bn"])
                + p["funding_bn"]
                - p["fees_bn"]
                + (-p["sign_bn"] * p["qty"] * (h - p["entry_hl"]) + p["funding_hl"] - p["fees_hl"])
                * fx[when]
                - p["qty"] * (b + h * fx[when]) * rate
            )
            if pair <= -capital * 0.002 or state["halted"]:
                closing.add(coin)
                state["pending"].append(
                    {
                        "action": "CLOSE",
                        "coin": coin,
                        "due_ms": when + HOUR,
                        "reason": "observed_pair_loss_or_account_circuit",
                    }
                )
        if not state["halted"]:
            available = [
                x
                for x in contexts.get(when, [])
                if x["cost_admission"]
                and x["horizon_hours"] == policy["horizon_hours"]
                and x["coin"] not in state["positions"]
            ]
            for x in sorted(
                available, key=lambda x: (-x["expected_after_four_fill_cost_bps"], x["coin"])
            )[: max(0, 3 - len(state["positions"]))]:
                state["pending"].append(
                    {
                        "action": "OPEN",
                        "coin": x["coin"],
                        "due_ms": when + HOUR,
                        "short_binance_long_hl": x["short_binance_long_hl"],
                    }
                )
        state["curve"].append({"timestamp": when, "equity": e, "pairs": len(state["positions"])})
        state["next_clock"] = when + HOUR
    e = equity(state, data, fx, end, rate, terminal=True)
    state["curve"].append({"timestamp": end, "equity": e, "pairs": len(state["positions"])})
    state["rejections"] = dict(rejects)
    # Native currency receipts reconcile before any aggregate currency conversion.
    bn_expected = (
        state["initial_bn"]
        + math.fsum(p["net_bn_usdt"] for p in state["closed"])
        + math.fsum(p["funding_bn"] - p["fees_bn"] for p in state["positions"].values())
    )
    hl_expected = (
        state["initial_hl"]
        + math.fsum(p["net_hl_usdc"] for p in state["closed"])
        + math.fsum(p["funding_hl"] - p["fees_hl"] for p in state["positions"].values())
    )
    if not (
        math.isclose(bn_expected, state["wallet_bn"], abs_tol=1e-7)
        and math.isclose(hl_expected, state["wallet_hl"], abs_tol=1e-7)
    ):
        raise ValueError("Per-native-currency carry ledger does not reconcile")
    curve = np.array([x["equity"] for x in state["curve"]])
    pnl = [x["net_usdt_at_close"] for x in state["closed"]]
    loss = -sum(min(x, 0) for x in pnl)
    initial_fx = state["initial_hl"] * (fx[end] - fx[state["curve"][0]["timestamp"]])
    receipt_fx = math.fsum(p["net_hl_usdc"] * (fx[end] - p["exit_fx"]) for p in state["closed"])
    open_net = 0.0
    for coin, p in state["positions"].items():
        b, h = data[coin]["terminal_bn_mark"], data[coin]["hl_prices"][end]
        open_net += (
            p["sign_bn"] * p["qty"] * (b - p["entry_bn"])
            + p["funding_bn"]
            - p["fees_bn"]
            + (-p["sign_bn"] * p["qty"] * (h - p["entry_hl"]) + p["funding_hl"] - p["fees_hl"])
            * fx[end]
            - p["qty"] * (b + h * fx[end]) * rate
        )
    trading_net = math.fsum(pnl) + open_net
    if not math.isclose(capital + trading_net + initial_fx + receipt_fx, e, abs_tol=1e-7):
        raise ValueError("Trading PNL versus cash FX revaluation does not reconcile")
    score = {
        "policy": policy,
        "net_return": e / capital - 1,
        "ending_equity_usdt": e,
        "complete_hedged_pairs": len(pnl),
        "whole_pair_net_win_rate": sum(x > 0 for x in pnl) / len(pnl) if pnl else None,
        "profit_factor": sum(max(x, 0) for x in pnl) / loss if loss else None,
        "hourly_observed_drawdown": float(
            ((np.maximum.accumulate(curve) - curve) / np.maximum.accumulate(curve)).max()
        ),
        "closed_pair_net_pnl_at_close_usdt": math.fsum(pnl),
        "terminal_open_pairs": len(state["positions"]),
        "terminal_pending_intents": len(state["pending"]),
        "wallet_bn_usdt": state["wallet_bn"],
        "wallet_hl_usdc": state["wallet_hl"],
        "per_native_currency_cash_reconciled": True,
        "terminal_open_pair_net_pnl_usdt": open_net,
        "trading_net_pnl_excluding_cash_fx_usdt": trading_net,
        "initial_prefunded_usdc_cash_fx_revaluation_usdt": initial_fx,
        "closed_hl_receipts_subsequent_fx_revaluation_usdt": receipt_fx,
        "trading_and_cash_fx_reconciled": True,
        "entry_fees_and_four_fill_cost_reserve_included": True,
        "historical_hl_mark_verified": False,
        "hl_funding_notional_uses_trade_open_proxy": True,
        "paired_execution_atomicity_margin_and_liquidation_verified": False,
        "production_eligible": False,
        "orders_enabled": False,
    }
    return score, state


def study(root, parent, output):
    output.mkdir(exist_ok=False)
    policies = [{"horizon_hours": h, "capital": 10000.0, "cost_bps_per_fill": 8} for h in [72, 168]]
    files = [Path(__file__), Path(__file__).with_name("cross_venue_carry_screen.py")]
    protocol = {
        "registered_utc": datetime.now(UTC).isoformat(),
        "candidates": policies,
        "source_hashes": {str(p): sha(p) for p in files},
        "context_sha256": sha(root / "carry_expanded_cost_v002/known_observations.json"),
        "selection": "rank validation net return, no outcome-based symbol selection; no production promotion",
        "scope": "retrospective separate paired carry diagnostic; not combined with intraday small-sleeve account",
        "additional_continuous_diagnostics": "both fixed horizons from first known-rate context through endpoint, each one owned account; never merge validation and evaluation cold starts",
        "execution": "next hourly open+1s proxy,8bps per leg fill covers5bps fee and3bps execution reserve; settlements precede fill",
        "limitations": [
            "HL historical mark absent; funding notional and unrealized PNL use trade-open proxy",
            "prefunded USDT/USDC wallets, observed FX revaluation included; transfer and conversion costs not executed",
            "hourly risk observations and no verified atomic fills, historical leverage/margin or liquidation",
        ],
        "orders_enabled": False,
        "production_eligible": False,
    }
    save(output / "protocol.json", protocol)
    snapshots = output / "executed_sources"
    snapshots.mkdir()
    for p in files:
        (snapshots / p.name).write_bytes(p.read_bytes())
    data, fx = load_data(root, parent)
    contexts = {}
    for x in json.loads((root / "carry_expanded_cost_v002/known_observations.json").read_text()):
        contexts.setdefault(x["clock"], []).append(x)
    start, middle, end = (
        millis(x)
        for x in [
            "2026-10-07T00:00:00+00:00",
            "2026-10-14T00:00:00+00:00",
            "2026-10-24T00:00:00+00:00",
        ]
    )
    validations = [run(p, data, fx, contexts, start, middle)[0] for p in policies]
    save(output / "validation_scores.json", validations)
    chosen = sorted(validations, key=lambda x: (-x["net_return"], x["policy"]["horizon_hours"]))[0][
        "policy"
    ]
    save(
        output / "selection.json",
        {
            "candidate": chosen,
            "evaluation_outcomes_read": False,
            "scope": "diagnostic only, required HL mark and execution evidence missing",
        },
    )
    result, state = run(chosen, data, fx, contexts, middle, end)
    save(output / "evaluation_score.json", result)
    save(output / "evaluation_state.json", state)
    checkpoint = middle + (end - middle) // (2 * HOUR) * HOUR
    _, prefix = run(chosen, data, fx, contexts, middle, end, break_ms=checkpoint)
    resumed, restored = run(
        chosen, data, fx, contexts, checkpoint, end, state=json.loads(json.dumps(prefix))
    )
    if result != resumed or state != restored:
        raise ValueError("Paired wallet checkpoint changes account")
    save(
        output / "restart_verification.json", {"exact_state_equal": True, "exact_score_equal": True}
    )
    stresses = [
        run({**chosen, "cost_bps_per_fill": 8 + extra}, data, fx, contexts, middle, end)[0]
        for extra in [2, 5]
    ]
    save(output / "cost_stress.json", stresses)
    # Capital comparisons remain independent accounts; no merging lifecycle or PNL.
    scaled = [
        run({**chosen, "capital": c}, data, fx, contexts, middle, end)[0]
        for c in [1000.0, 100000.0]
    ]
    save(output / "capital_comparison.json", scaled)
    for policy in policies:
        continuous_start = min(contexts)
        score, ledger = run(policy, data, fx, contexts, continuous_start, end)
        name = f"continuous_{policy['horizon_hours']}h"
        save(output / f"{name}_score.json", score)
        save(output / f"{name}_state.json", ledger)
    if any(sha(Path(p)) != digest for p, digest in protocol["source_hashes"].items()):
        raise ValueError("Carry source changed while running")
    save(
        output / "run_status.json",
        {
            "status": "completed",
            "production_eligible": False,
            "orders_enabled": False,
            "independent_forward_evidence": False,
            "joint_small_large_account_tested": False,
            "scope": "paired carry hypothesis, required mark/execution and sample sufficiency unresolved",
        },
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    study(args.root.absolute(), args.parent.absolute(), args.output.absolute())
