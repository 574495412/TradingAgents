"""Integer-inventory economic/lifecycle counterexamples for conditional v2."""

import copy
import json

import pandas as pd
import pytest
from test_dated_account_contract import engine, inputs


def data():
    d = list(inputs())
    d[5]["unliquidatable_dust_reserve_fraction"] = 1.0
    d[6]["BTCUSDT"]["quantity_rules"] = {
        "scope": "hypothetical_grid_stress",
        "historical_rules_certified": False,
        "inventory_unit": "0.00000001",
        "spot_step": "0.00001",
        "future_base_step": "0.001",
    }
    d[4][0]["assets"]["BTCUSDT"]["lot_target"] = {"gross_spot": "0.01302", "short_base": "0.013"}
    return d


def replay(d, end="2025-01-01T01:00Z", **kwargs):
    return engine("spot_dated_account_v2.py").replay(
        *d[:6],
        "2025-01-01T00:05Z",
        end,
        contracts=d[6],
        **kwargs,
    )


def test_planned_residual_is_owned_not_a_failed_or_exactly_matched_hedge():
    r = replay(data(), end="2025-01-01T00:20Z", final_end="2025-01-01T01:00Z")
    p = r["positions"]["BTCUSDT"]
    assert p["spot_atoms"] == 1300698 and p["perp_atoms"] == 1300000
    assert p["planned_residual_atoms"] == 698
    assert not p["hedge_failed"] and r["capacity"]["hedge_failures"] == 0
    assert r["curve"].iloc[-1].unliquidatable_spot_value > 0


def test_settlement_and_exit_preserve_dust_and_full_reserve_without_closing_campaign():
    r = replay(data())
    p = r["positions"]["BTCUSDT"]
    assert p["perp_atoms"] == 0 and p["spot_atoms"] == 698 and p["spot_q"] == 0.00000698
    assert not r["trades"] and p["exit_request"] and p["dust_first_at"]
    last = r["curve"].iloc[-1]
    assert last.exit_reserve == last.spot_assets == last.unliquidatable_spot_value
    assert last.equity == r["wallets"]["quote"] + r["wallets"]["futures"]
    assert len([e for e in r["events"] if e["type"] == "settlement"]) == 1
    assert all(
        e["quantity_atoms"] % 1000 == 0
        for e in r["events"]
        if e["type"] == "exit" and e["market"] == "spot"
    )


def test_actual_one_leg_fill_failure_is_not_hidden_as_expected_residual():
    d = data()
    d[1]["BTCUSDT"].loc[pd.Timestamp("2025-01-01T00:05Z"), "quote_volume"] = 0
    r = replay(d)
    assert r["capacity"]["hedge_failures"] == 1 and r["positions"]["BTCUSDT"]["hedge_failed"]
    assert not any(e["type"] == "entry" and e["market"] == "perp" for e in r["events"])


def test_json_restart_after_spot_sale_cannot_drop_dust_or_repeat_settlement():
    d = data()
    full = replay(d)
    prefix = replay(d, end="2025-01-01T00:40Z", final_end="2025-01-01T01:00Z")
    state = json.loads(json.dumps(prefix["resume_state"], allow_nan=False))
    child = engine("spot_dated_account_v2.py").replay(
        *d[:6],
        "2025-01-01T00:40Z",
        "2025-01-01T01:00Z",
        contracts=d[6],
        initial_state=state,
        final_end="2025-01-01T01:00Z",
    )
    pd.testing.assert_frame_equal(
        pd.concat([prefix["curve"], child["curve"].iloc[1:]], ignore_index=True),
        full["curve"],
        check_exact=True,
    )
    assert prefix["events"] + child["events"] == full["events"]
    assert child["resume_state"] == full["resume_state"]


def test_checkpoint_integer_quantity_disagreement_is_rejected():
    d = data()
    prefix = replay(d, end="2025-01-01T00:40Z", final_end="2025-01-01T01:00Z")
    prefix["resume_state"]["positions"]["BTCUSDT"]["spot_atoms"] = 0
    with pytest.raises(ValueError, match="integer inventory"):
        engine("spot_dated_account_v2.py").replay(
            *d[:6],
            "2025-01-01T00:40Z",
            "2025-01-01T01:00Z",
            contracts=d[6],
            initial_state=prefix["resume_state"],
            final_end="2025-01-01T01:00Z",
        )


def test_fee_unit_rounding_cannot_be_invented():
    d = data()
    d[6]["BTCUSDT"]["quantity_rules"]["inventory_unit"] = "0.0000001"
    with pytest.raises(ValueError, match="exactly"):
        replay(d)


def test_cannot_claim_real_historical_rules_or_remove_dust_reserve():
    d = data()
    d[6]["BTCUSDT"]["quantity_rules"]["historical_rules_certified"] = True
    with pytest.raises(ValueError, match="hypothetical"):
        replay(d)
    d = data()
    d[5]["unliquidatable_dust_reserve_fraction"] = 0
    with pytest.raises(ValueError, match="full noncash"):
        replay(d)


def test_future_settlement_price_does_not_change_preexpiry_integer_path():
    d = data()
    a = replay(d, end="2025-01-01T00:30Z", final_end="2025-01-01T01:00Z")
    d[6]["BTCUSDT"]["settlement_price"] = 999
    b = replay(d, end="2025-01-01T00:30Z", final_end="2025-01-01T01:00Z")
    pd.testing.assert_frame_equal(a["curve"], b["curve"], check_exact=True)
    assert a["events"] == b["events"]


def test_zero_base_fee_fixture_closes_only_when_inventory_really_zero():
    d = data()
    d[5]["spot_fee_bps"] = 0
    d[4][0]["assets"]["BTCUSDT"]["lot_target"]["gross_spot"] = "0.013"
    r = replay(d)
    assert not r["positions"] and len(r["trades"]) == 1 and r["trades"][0]["matched_hedge"]


def test_market_source_frames_are_not_mutated_by_execution():
    d = data()
    before = copy.deepcopy(d[:3])
    replay(d)
    for expected, actual in zip(before, d[:3], strict=True):
        pd.testing.assert_frame_equal(expected["BTCUSDT"], actual["BTCUSDT"], check_exact=True)
