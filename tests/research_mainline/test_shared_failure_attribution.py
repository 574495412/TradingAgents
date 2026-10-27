import pandas as pd
import pytest

from research_tools.shared_failure.fixed_path import matched_btc_path, price_increments


def events():
    return [
        {
            "type": "entry",
            "time": "2025-01-01T00:05Z",
            "submitted_at": "2025-01-01T00:00Z",
            "symbol": "ALT",
            "position_id": 1,
            "side": 1,
            "quantity": 2,
            "fee": 1,
            "slippage_cost": 1,
        },
        {
            "type": "exit",
            "time": "2025-01-01T00:15Z",
            "symbol": "ALT",
            "position_id": 1,
            "side": 1,
            "quantity": 2,
            "fee": 1,
            "slippage_cost": 1,
        },
    ]


def test_benchmark_has_no_return_before_fill_but_includes_exit_bar():
    index = pd.date_range("2025-01-01", periods=4, freq="5min", tz="UTC")
    btc = pd.Series([10.0, 20.0, 30.0, 40.0], index=index)
    inc, closed, _ = matched_btc_path(
        events(), index, btc, lambda s, t: 100.0, lambda s, t: {"beta": 2.0}
    )
    assert list(inc.unit) == [0, 0, 100, 100]
    assert list(inc.beta) == [0, 0, 200, 200]
    assert closed[0]["price_pnl"] == 0 and closed[0]["unit_market_pnl"] == 200


def test_unknown_beta_stays_unknown_until_owned_inventory_is_closed():
    index = pd.date_range("2025-01-01", periods=5, freq="5min", tz="UTC")
    inc, _, missing = matched_btc_path(
        events(),
        index,
        pd.Series([10.0, 20.0, 30.0, 40.0, 50.0], index=index),
        lambda s, t: 100.0,
        lambda s, t: {"beta": None, "reason": "insufficient_history"},
    )
    assert inc.beta.iloc[2:4].isna().all() and inc.beta.iloc[4] == 0
    assert len(missing) == 1 and inc.unit.notna().all()


def test_partial_exit_and_addition_keep_reference_notional_and_benchmark_ownership():
    index = pd.date_range("2025-01-01", periods=6, freq="5min", tz="UTC")
    es = events()
    es[1]["time"] = index[3].isoformat()
    es[1]["quantity"] = 1
    es.insert(
        1,
        {
            **es[0],
            "time": index[2].isoformat(),
            "submitted_at": index[1].isoformat(),
            "quantity": 2,
        },
    )
    es.append({**es[1], "type": "exit", "time": index[5].isoformat(), "quantity": 3})
    inc, closed, _ = matched_btc_path(
        es,
        index,
        pd.Series([10.0, 20.0, 40.0, 60.0, 80.0, 100.0], index=index),
        lambda s, t: 100.0,
        lambda s, t: {"beta": 1.0},
    )
    assert len(closed) == 1
    assert inc.unit.sum() == pytest.approx(closed[0]["unit_market_pnl"])
    assert inc.beta.sum() == pytest.approx(inc.unit.sum())


def test_cash_price_identity_restores_reserve_changes_without_double_cost():
    index = pd.date_range("2025-01-01", periods=4, freq="5min", tz="UTC")
    curve = pd.DataFrame(
        {
            "date": index,
            "equity": [100.0, 97.0, 100.0, 104.0],
            "gross_exposure": [0.0, 200.0, 200.0, 0.0],
        }
    )
    result = price_increments(curve, events(), 0.005)
    assert result.price.sum() == 8  # Net4 plus two fees and two modeled slippages.
    assert list(result.exit_reserve_change) == [0, 1, 0, -1]


def test_non_owned_or_unclosed_exposure_cannot_be_accepted_as_zero():
    index = pd.date_range("2025-01-01", periods=4, freq="5min", tz="UTC")
    with pytest.raises(ValueError):
        matched_btc_path(
            events()[:1],
            index,
            pd.Series(10.0, index=index),
            lambda s, t: 100.0,
            lambda s, t: {"beta": 1.0},
        )
    with pytest.raises(ValueError):
        matched_btc_path(
            events()[1:],
            index,
            pd.Series(10.0, index=index),
            lambda s, t: 100.0,
            lambda s, t: {"beta": 1.0},
        )
