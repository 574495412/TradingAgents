"""检验完整活动贡献与扣成本标签，防止平仓胜率掩盖剩余亏损。"""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from tradingagents.research.quant import strategy_audit as audit
from tradingagents.research.quant.strategy_extension import fresh_mask


def fixture_account():
    seed = {
        "cycle_id": "seed",
        "episode_id": "activity",
        "symbol": "PUBLIC",
        "direction": "LONG",
        "authority_kind": "AUTONOMOUS_SEED",
        "open_fill": {"quantity": 1.0, "price": 100.0, "fee_usdt": 0.05},
        "close_fill": {"quantity": 1.0, "price": 102.0, "fee_usdt": 0.051},
    }
    child = {
        **seed,
        "cycle_id": "child",
        "authority_kind": "OWN_PROFIT_CONTINUATION",
        "open_fill": {"quantity": 1.0, "price": 102.0, "fee_usdt": 0.051},
        "close_fill": None,
    }
    ending = 1000 + 2 - 0.152 - 12
    checkpoint = {
        "lifecycle": {
            "state": {
                "cycles": {"seed": seed, "child": child},
                "episodes": {"activity": {"symbol": "PUBLIC"}},
            }
        },
        "trade_ledger": [{"cycle_id": "seed", "net_usdt": 1.899}],
        "marks": {"PUBLIC": 90.0},
        "equity_ledger": [{"equity_usdt": ending}],
    }
    metrics = {"net_return_ex_funding": ending / 1000 - 1}
    config = {"fee_bps_per_side": 5, "slippage_bps_per_side": 3, "initial_equity_usdt": 1000}
    return checkpoint, metrics, config


def test_profitable_parent_and_losing_child_are_one_losing_activity():
    checkpoint, metrics, config = fixture_account()
    before = deepcopy(checkpoint)
    result = audit.account_attribution(checkpoint, metrics, config)
    assert checkpoint == before
    assert result["initial_episodes"] == 1
    assert result["open_cycles"] == 1
    assert result["positive_marked_episodes"] == 0
    assert result["marked_net_usdt"] == pytest.approx(-10.152)
    assert result["episodes"][0]["closed_cycles"] == 1
    assert result["pending_exit_cost_usdt"] > 0
    assert result["initial_plus_all_episode_contributions"] == pytest.approx(
        result["ending_equity_usdt"]
    )


@pytest.mark.parametrize("field", ["equity", "trade", "headline"])
def test_altered_economics_rejected(field):
    checkpoint, metrics, config = fixture_account()
    if field == "equity":
        checkpoint["equity_ledger"][-1]["equity_usdt"] += 1
    elif field == "trade":
        checkpoint["trade_ledger"][0]["net_usdt"] += 1
    else:
        metrics["net_return_ex_funding"] += 0.1
    with pytest.raises(ValueError):
        audit.account_attribution(checkpoint, metrics, config)


@pytest.mark.parametrize("direction", [-1, 1])
def test_flat_price_has_two_sided_costs(direction):
    base = audit.net_directional_return(100.0, 100.0, direction)
    stress = audit.net_directional_return(100.0, 100.0, direction, slippage_bps=10)
    assert stress < base < 0
    assert -0.00161 < base < -0.00159
    assert audit.net_directional_return(100.0, 100.0, direction, fee_bps=0, slippage_bps=0) == 0


def test_factor_direction_symmetry_and_coherence():
    frame = pd.DataFrame(
        {
            "atr_pct": [1.0],
            "ret_20_rank_pct": [0.9],
            "ret_60_rank_pct": [0.8],
            "ret_20_residual_pct": [2.0],
            "ret_60_residual_pct": [3.0],
            "taker_imbalance_3": [0.2],
            "taker_imbalance_12": [0.4],
            "volume_ratio_rank_pct": [0.8],
            "atr_rank_pct": [0.7],
        }
    )
    inverse = frame.copy()
    for key in ("ret_20_rank_pct", "ret_60_rank_pct"):
        inverse[key] = 1 - inverse[key]
    for key in (
        "ret_20_residual_pct",
        "ret_60_residual_pct",
        "taker_imbalance_3",
        "taker_imbalance_12",
    ):
        inverse[key] *= -1
    ratios = np.array([[0.5, 0.7, 0.8]])
    a, b = audit.factor_components(frame, ratios), audit.factor_components(inverse, 1 / ratios)
    for key in ("momentum", "flow", "crowding"):
        assert a[key] == pytest.approx(-b[key])
    assert a["coherence"][0] == pytest.approx(1)
    assert a["coherence"] == pytest.approx(b["coherence"])


def test_directional_labels_use_both_actual_execution_prices():
    long = audit.net_directional_return(100.0, 101.0, 1, fee_bps=0, slippage_bps=0)
    short = audit.net_directional_return(100.0, 101.0, -1, fee_bps=0, slippage_bps=0)
    assert long == pytest.approx(0.01)
    assert short == pytest.approx(-0.01)


def test_fresh_edges_do_not_multiply_persistent_signals_or_bridge_gaps():
    times = pd.to_datetime(
        [
            "2026-10-06 00:00Z",
            "2026-10-06 00:01Z",
            "2026-10-06 00:02Z",
            "2026-10-06 00:03Z",
            "2026-10-06 00:05Z",
            "2026-10-06 00:06Z",
        ]
    )
    index = pd.MultiIndex.from_arrays(
        [times, ["PUBLIC"] * len(times)], names=["bar_time", "symbol"]
    )
    assert fresh_mask(np.array([0, -1, -1, 0, -1, -1]), index).tolist() == [
        False,
        True,
        False,
        False,
        False,
        False,
    ]
