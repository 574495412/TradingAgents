"""Internal independent cash/quantity audit. Does not import the paper engine."""

import argparse
from decimal import ROUND_FLOOR, Decimal
from pathlib import Path

from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest

D = Decimal
UNIT = D("0.00000001")
# New derived-price arithmetic only: bounded inputs <=1e8 and <100 Decimal28
# operations imply aggregate roundoff <1e-18USDT. Ledger identities stay exact.
DERIVED_BOUND = D("0.000000000000000001")


def audit(trial, output):
    output.mkdir(parents=True, exist_ok=False)
    verify_manifest(trial)
    cfg = read(trial / "plan.json")["account"]
    account, obs = read(trial / "account.json"), read(trial / "observations.json")
    quote = D(cfg["capital"]) * (1 - D(cfg["future_wallet_fraction"]))
    future = D(cfg["capital"]) * D(cfg["future_wallet_fraction"])
    positions, event_checks = {}, []
    for event in account["events"]:
        asset = event["asset"]
        if event["type"] == "stay_cash":
            assert not positions.get(asset)
            continue
        p = positions.setdefault(
            asset, {"spot_atoms": 0, "short_atoms": 0, "future_entry_quote": D(0)}
        )
        f, market = event["fill"], event["market"]
        q, cash = D(f["quantity"]), D(f["quote"])
        phase = "arrival" if event["type"] == "paper_entry" else "unwind"
        assert event["time_ns"] == obs[asset][phase + "_ns"] > obs[asset]["decision_ns"]
        entry = event["type"] == "paper_entry"
        side = ("buy" if entry else "sell") if market == "spot" else ("sell" if entry else "buy")
        if q:
            levels = obs[asset][phase][market]["asks" if side == "buy" else "bids"]
            levels = sorted([(D(x), D(y)) for x, y in levels], reverse=side == "sell")
            best, left, raw = levels[0][0], q, D(0)
            for px, available in levels:
                if side == "buy" and px > best * (1 + D(cfg["maximum_book_walk_fraction"])):
                    break
                if side == "sell" and px < best * (1 - D(cfg["maximum_book_walk_fraction"])):
                    break
                take = min(left, available * D(cfg["depth_fraction"]))
                raw += take * px
                left -= take
                if not left:
                    break
            assert left == 0 and raw == D(f["raw_quote"])
            stressed = raw * (
                1 + D(cfg["adverse_price_fraction"])
                if side == "buy"
                else 1 - D(cfg["adverse_price_fraction"])
            )
            assert stressed == cash
            assert obs[asset]["book_valid"][phase][market]
        if market == "spot" and entry:
            fee = q * D(cfg["spot_base_fee_fraction"])
            assert fee == D(event["base_fee"])
            net_atoms = (q - fee) / UNIT
            assert net_atoms == net_atoms.to_integral_value()
            p["spot_atoms"] += int(net_atoms)
            quote -= cash
        elif market == "future" and entry:
            fee = cash * D(cfg["future_fee_fraction"])
            assert fee == D(event["quote_fee"])
            future -= fee
            p["short_atoms"] += int(q / UNIT)
            p["future_entry_quote"] += cash
        elif market == "spot":
            fee = cash * D(cfg["spot_exit_fee_fraction"])
            assert fee == D(event["quote_fee"])
            quote += cash - fee
            p["spot_atoms"] -= int(q / UNIT)
        else:
            fee = cash * D(cfg["future_fee_fraction"])
            assert fee == D(event["quote_fee"])
            old = D(p["short_atoms"]) * UNIT
            release = p["future_entry_quote"] * q / old if old else D(0)
            future += release - cash - fee
            p["future_entry_quote"] -= release
            p["short_atoms"] -= int(q / UNIT)
        assert min(p["spot_atoms"], p["short_atoms"]) >= 0
        event_checks.append(
            {"asset": asset, "event_type": event["type"], "market": market, "exact": True}
        )
    assert quote == D(account["quote_cash"]) and future == D(account["future_cash"])
    assert set(positions) == set(account["positions"])
    for asset, p in positions.items():
        actual = account["positions"][asset]
        assert p["spot_atoms"] == actual["spot_atoms"] and p["short_atoms"] == actual["short_atoms"]
        assert p["future_entry_quote"] == D(actual["future_entry_quote"])
    screens = []
    for decision in account["decisions"]:
        target, screen = decision["target"], decision["screen"]
        gross, short = D(target["gross"]), D(target["short"])
        if "net_terminal_scenario" not in screen:
            assert not screen["eligible"]
            continue
        spot, ft = D(screen["spot_entry_price"]), D(screen["future_entry_price"])
        terminal = spot * D(cfg["terminal_price_cost_stress_multiple"])
        expected = short * ft - gross * spot
        expected -= short * ft * D(cfg["future_fee_fraction"])
        expected -= (
            short
            * terminal
            * (
                1
                - (1 - D(cfg["adverse_price_fraction"])) * (1 - D(cfg["spot_exit_fee_fraction"]))
                + D(cfg["settlement_fee_fraction"])
            )
        )
        difference = abs(expected - D(screen["net_terminal_scenario"]))
        assert difference <= DERIVED_BOUND
        assert screen["eligible"] == (
            D(screen["net_terminal_scenario"]) >= D(cfg["minimum_useful_pnl_per_asset_usdt"])
        )
        assert gross * spot <= D(target["cash_cap"])
        asset = decision["asset"]
        ss = D(
            next(
                r["stepSize"]
                for r in obs[asset]["symbols"]["spot"]["filters"]
                if r["filterType"] == "LOT_SIZE"
            )
        )
        net = gross * (1 - D(cfg["spot_base_fee_fraction"]))
        assert net >= short and net - short < ss * (1 - D(cfg["spot_base_fee_fraction"]))
        screens.append(
            {
                "asset": asset,
                "derived_identity_abs_error": str(difference),
                "eligible": screen["eligible"],
            }
        )
    spot_value, floating, reserve = D(0), D(0), D(0)
    for asset, p in positions.items():
        assert all(obs[asset]["book_valid"]["mark"].values())
        spot_q, short_q = D(p["spot_atoms"]) * UNIT, D(p["short_atoms"]) * UNIT
        s_bid, f_ask = (
            D(obs[asset]["mark"]["spot"]["bids"][0][0]),
            D(obs[asset]["mark"]["future"]["asks"][0][0]),
        )
        spot_value += spot_q * s_bid
        floating += p["future_entry_quote"] - short_q * f_ask
        sym = obs[asset]["symbols"]["spot"]
        grid = D(next(r["stepSize"] for r in sym["filters"] if r["filterType"] == "LOT_SIZE"))
        sellable = (spot_q / grid).to_integral_value(rounding=ROUND_FLOOR) * grid
        notional = next(r for r in sym["filters"] if r["filterType"] == "NOTIONAL")
        if sellable * D(obs[asset]["spot_notional_reference"]) < D(notional["minNotional"]):
            sellable = D(0)
        dust = spot_q - sellable
        assert int(dust / UNIT) == account["positions"][asset]["owned_unliquidatable_atoms"]
        assert dust * s_bid == D(account["positions"][asset]["dust_reserve"])
        reserve += dust * s_bid
        reserve += (
            sellable
            * s_bid
            * (1 - (1 - D(cfg["adverse_price_fraction"])) * (1 - D(cfg["spot_exit_fee_fraction"])))
        )
        reserve += (
            short_q
            * f_ask
            * ((1 + D(cfg["adverse_price_fraction"])) * (1 + D(cfg["future_fee_fraction"])) - 1)
        )
    observed = account["observations"][-1]
    assert all(
        abs(D(observed[k]) - value) <= DERIVED_BOUND
        for k, value in [
            ("spot_value", spot_value),
            ("future_unrealized", floating),
            ("noncash_exit_reserve", reserve),
            ("equity_after_exit_reserve", quote + future + spot_value + floating - reserve),
        ]
    )
    causal = read(trial / "causal_books.json")
    assert all(p["known_at_ns"] <= p["clock_ns"] for p in causal)
    assert account["complete_lifecycles"] == 0 and account["win_rate"] is None
    result = {
        "internal_cash_and_integer_audit_passed": True,
        "independent_of_paper_engine_import": True,
        "external_third_party_review": False,
        "exact_event_checks": event_checks,
        "economic_identities": screens,
        "causal_book_states": len(causal),
        "actual_order_eligibility_or_full_strategy_proven": False,
        "profit_goal_met": False,
    }
    save(output / "verification.json", result)
    save(
        output / "completion.json",
        {
            "manifest": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
            "trial_completion_sha256": sha(trial / "completion.json"),
            "audit_source_sha256": sha(Path(__file__)),
        },
    )
    print(result, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trial", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    audit(a.trial, a.output)
