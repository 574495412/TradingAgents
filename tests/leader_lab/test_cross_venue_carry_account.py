import json

import pytest

from research_tools.leader_lab.cross_venue_carry_account import HOUR, run


def fixture():
    prices = {t: {"open": 100.0, "mark": 100.0} for t in range(0, 5 * HOUR, HOUR)}
    data = {
        "X": {
            "bn_prices": prices,
            "hl_prices": dict.fromkeys(prices, 100.0),
            "terminal_bn_mark": 100.0,
            "bn_funding": [
                {"fundingTime": t + 17, "fundingRate": "0.01", "markPrice": "100"}
                for t in range(0, 4 * HOUR, HOUR)
            ],
            "hl_funding": [
                {"time": t + 139, "fundingRate": "0.001"} for t in range(0, 4 * HOUR, HOUR)
            ],
            "bn_meta": {"step": 0.01, "min_qty": 0.01, "min_notional": 5},
            "hl_step": 0.01,
        }
    }
    fx = dict.fromkeys(prices, 1.0)
    contexts = {
        0: [
            {
                "coin": "X",
                "horizon_hours": 1,
                "cost_admission": True,
                "expected_after_four_fill_cost_bps": 100,
                "short_binance_long_hl": True,
            }
        ]
    }
    policy = {"capital": 1000.0, "cost_bps_per_fill": 8, "horizon_hours": 1}
    return policy, data, fx, contexts


def test_cannot_collect_settlement_before_entry_and_all_four_costs_are_paid():
    p, d, fx, contexts = fixture()
    score, ledger = run(p, d, fx, contexts, 0, 4 * HOUR)
    assert score["complete_hedged_pairs"] == 1
    trade = ledger["closed"][0]
    assert trade["funding_bn"] == pytest.approx(1.0)
    assert trade["funding_hl"] == pytest.approx(-0.1)
    assert trade["fees_bn"] == pytest.approx(0.16)
    assert trade["fees_hl"] == pytest.approx(0.16)
    assert score["trading_net_pnl_excluding_cash_fx_usdt"] == pytest.approx(0.58)
    assert score["per_native_currency_cash_reconciled"]


def test_nonempty_dual_wallet_restart_is_exact():
    p, d, fx, contexts = fixture()
    full, ledger = run(p, d, fx, contexts, 0, 4 * HOUR)
    _, prefix = run(p, d, fx, contexts, 0, 4 * HOUR, break_ms=2 * HOUR)
    assert prefix["positions"]
    resumed, state = run(
        p, d, fx, contexts, 2 * HOUR, 4 * HOUR, state=json.loads(json.dumps(prefix))
    )
    assert resumed == full and state == ledger


def test_prefunded_currency_revaluation_is_separate_from_strategy_profit():
    p, d, fx, _ = fixture()
    fx[4 * HOUR] = 1.01
    score, _ = run(p, d, fx, {}, 0, 4 * HOUR)
    assert score["net_return"] > 0
    assert score["initial_prefunded_usdc_cash_fx_revaluation_usdt"] == pytest.approx(5)
    assert score["trading_net_pnl_excluding_cash_fx_usdt"] == 0
    assert score["complete_hedged_pairs"] == 0
    assert score["whole_pair_net_win_rate"] is None
