"""Known cash identities must permit profits and apply each cost once."""

import pytest

from research_tools.evaluation_review.known_cash import (
    halted_recovery_case,
    load_engine,
    run_case,
)


@pytest.fixture(scope="module")
def engine():
    return load_engine()


@pytest.mark.parametrize("slippage_bps", [0, 3, 10])
@pytest.mark.parametrize(
    ("side", "exit_reference", "funding_rate"),
    [
        (1, 105, "0"),
        (-1, 95, "0"),
        (1, 95, "0"),
        (-1, 105, "0"),
        (1, 100, "0"),
        (1, 100, ".002"),
        (-1, 100, ".002"),
    ],
)
def test_exact_closed_cash_fee_funding_and_profit_controls(
    engine, side, exit_reference, funding_rate, slippage_bps
):
    account, expected = run_case(engine, side, exit_reference, slippage_bps, funding_rate)
    entry, exit_event = [e for e in account["events"] if e["type"] in ["entry", "exit"]]
    assert entry["quantity"] == exit_event["quantity"] == 40
    assert entry["price"] == pytest.approx(float(expected["entry_price"]), abs=1e-10)
    assert exit_event["price"] == pytest.approx(float(expected["exit_price"]), abs=1e-10)
    assert entry["fee"] + exit_event["fee"] == pytest.approx(
        float(expected["commission"]), abs=1e-10
    )
    assert sum(e["cash_delta"] for e in account["events"] if e["type"] == "funding") == (
        pytest.approx(float(expected["funding_cash"]), abs=1e-10)
    )
    assert account["cash"] == pytest.approx(float(expected["closed_cash"]), abs=1e-9)
    assert account["curve"].iloc[-1].equity == pytest.approx(account["cash"], abs=1e-9)
    assert not account["positions"] and len(account["trades"]) == 1
    assert account["trades"][0]["net_pnl"] == pytest.approx(account["cash"] - 10000, abs=1e-9)
    if side * (exit_reference - 100) > 0:
        assert account["cash"] > 10000
    elif side * (exit_reference - 100) < 0:
        assert account["cash"] < 10000


def test_permanent_halt_is_policy_and_prevents_reentry_after_recovery(engine):
    account = halted_recovery_case(engine)
    assert account["resume_state"]["risk_halted"] is True
    assert len([e for e in account["events"] if e["type"] == "entry"]) == 1
    assert [e["reason"] for e in account["events"] if e["type"] == "exit"] == ["risk_halt"]
    assert account["cash"] < 10000 and not account["positions"]
