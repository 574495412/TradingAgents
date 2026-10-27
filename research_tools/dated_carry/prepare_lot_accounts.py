"""Freeze past-bar-only integer targets and explicit retrospective grid scenarios."""

import argparse
from pathlib import Path

import pandas as pd

from factors.common.lot_inventory import bounded_cover
from research_tools.dated_carry.run_accounts import load
from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).with_name("lot_account_plan.json")
    plan = read(source)
    base = Path(plan["base_prepared_inputs"])
    verify_manifest(base)
    original, markets = load(base)
    for f, digest in original["source_hashes"].items():
        assert sha(Path(f)) == digest
    plan["contracts"] = {
        s: {**c, "quantity_rules": plan["quantity_rules"][s]}
        for s, c in original["contracts"].items()
    }
    previous = pd.Timestamp(plan["window"]["start"]) - pd.Timedelta(minutes=5)
    a = plan["account"]
    sf = a["spot_fee_bps"] / 10000
    buffer = 1 + a["entry_slippage_buffer_bps"] / 10000
    targets, witnesses = {}, []
    for s in plan["contracts"]:
        spot, future = markets[0][s].loc[previous], markets[1][s].loc[previous]
        q = min(
            a["capital"]
            / len(plan["contracts"])
            * a["asset_spot_fraction"]
            * (1 - sf)
            / (spot.close * buffer),
            max(0, spot.quote_volume) * a["participation"] * (1 - sf) / (spot.close * buffer),
            max(0, future.quote_volume) * a["participation"] / (future.close * buffer),
        )
        rule = plan["quantity_rules"][s]
        target = bounded_cover(
            gross_spot_cap=str(q / (1 - sf)),
            short_base_cap=str(q),
            spot_step=rule["spot_step"],
            future_base_step=rule["future_base_step"],
            base_fee_rate=str(sf),
        )
        targets[s] = {k: target[k] for k in ["gross_spot", "short_base"]}
        witnesses.append(
            {
                "symbol": s,
                "past_bar_open": previous.isoformat(),
                "known_initial_capital": a["capital"],
                "past_spot_close": float(spot.close),
                "past_future_close": float(future.close),
                "past_spot_quote_volume": float(spot.quote_volume),
                "past_future_quote_volume": float(future.quote_volume),
                "unchanged_net_quantity_cap": float(q),
                "target": target,
                "old_financial_fills_or_future_prices_used_for_sizing": False,
                "retrospective_grid_assumption_not_historical_rule_availability": True,
            }
        )
    for profile in plan["profiles"]:
        signals = read(base / (profile + "_signals.json"))
        for row in signals:
            for s, signal in row["assets"].items():
                if signal["entry"]:
                    signal["lot_target"] = targets[s]
        save(output / (profile + "_signals.json"), signals)
    sources = [
        source,
        Path(__file__),
        Path("factors/common/lot_inventory.py"),
        Path(plan["engine"]),
        Path("research_tools/dated_carry/account_audit_lots_v2.py"),
        Path("research_tools/dated_carry/run_accounts.py"),
        Path("tests/research_mainline/test_dated_account_lots.py"),
        Path("tests/research_mainline/test_dated_lot_ledger.py"),
        base / "completion.json",
    ]
    plan["source_hashes"] = {str(p): sha(p) for p in sources}
    save(output / "plan.json", plan)
    save(output / "target_witnesses.json", witnesses)
    save(
        output / "registration.json",
        {
            "before_new_financial_accounts": True,
            "plan_sha256": sha(output / "plan.json"),
            "conditional_hypothetical_rules": True,
            "causal_historical_venue_execution_certified": False,
            "new_economic_variants": 0,
            "actual_accounts_planned": 4,
            "independent_oos": False,
        },
    )
    save(
        output / "completion.json",
        {
            "passed": True,
            "manifest": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
            "profit_goal_met": False,
        },
    )
    print(
        {"targets": targets, "accounts_planned": 4, "historical_rules_certified": False}, flush=True
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    prepare(p.parse_args().output)
