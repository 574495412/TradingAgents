"""纯统计人工样例验证拒绝门槛；这些输入没有市场收益或来源证明含义。"""

import copy
from decimal import Decimal

import pytest

from tradingagents.research.quant.backtesting.paper_account import PaperAccount
from tradingagents.research.quant.backtesting.stability import (
    DAY,
    DAYS,
    MINUTE,
    assess_phase,
    block_interval,
    compare_phases,
)


def mathematical_state(start=20000 * DAY):
    """人工每日一单位净收入，只用于验证统计与时间口径，不能提交生产。"""
    end = start + DAYS * DAY
    account = PaperAccount(["STATISTICAL_FIXTURE"], 10, period_end_ms=end)
    state = account.checkpoint()
    state["curve"] = [
        {
            "actual_ms": start + i * MINUTE + 15000,
            "canonical_close_ms": start + i * MINUTE,
            "equity": str(Decimal("1000") + Decimal(i) / 1440),
            "cash": "1000",
            "positions": 0,
        }
        for i in range(DAYS * 1440 + 1)
    ]
    state["cash"] = "1030"
    state["curve"][-1]["cash"] = "1030"
    state["event_count"] = len(state["curve"])
    state["trades"], windows = [], []
    for day in range(DAYS):
        entry = start + day * DAY + 10 * MINUTE
        exit_ms = entry + 20 * MINUTE
        state["trades"].append(
            {
                "episode": f"人工活动-{day}",
                "entry_ms": entry,
                "exit_ms": exit_ms,
                "direction": 1 if day % 2 else -1,
                "symbol": "STATISTICAL_FIXTURE",
                "net_pnl": "1",
            }
        )
        windows.extend(
            {"start_ms": entry + 1, "end_ms": exit_ms, "requested_ms": exit_ms + delay}
            for delay in (2000, MINUTE + 1000)
        )
    state["funding_windows"]["STATISTICAL_FIXTURE"] = windows
    return state, start, end


@pytest.mark.parametrize("daily,total", [(0.001, 0.03), (0.0, 0.0), (-0.001, -0.03)])
def test_constant_circular_blocks_preserve_total_and_fixed_sampling_identity(daily, total):
    result = block_interval([daily] * DAYS)
    assert result["lower"] == pytest.approx(total)
    assert result["upper"] == pytest.approx(total)
    assert result == block_interval([daily] * DAYS)
    assert (
        result["block_days"] == 5 and result["replicates"] == 10000 and result["seed"] == 20261016
    )


@pytest.mark.parametrize("values", [[0.001] * 29, [float("nan")] * 30])
def test_short_or_nonfinite_daily_inputs_cannot_form_a_confidence_interval(values):
    with pytest.raises(ValueError, match="完整三十日"):
        block_interval(values)


def test_mathematical_economic_pass_cannot_grant_source_runtime_or_production_qualification():
    state, start, end = mathematical_state()
    result = assess_phase(state, start_ms=start, end_ms=end, ledger_identity="人工统计账本")
    assert result["economic_conditions_met"]
    assert len(result["daily_pnl_over_initial"]) == 30
    assert sum(Decimal(v) for v in result["daily_pnl_over_initial"]) == Decimal("0.03")
    assert result["block_interval"]["lower"] > 0
    assert result["source_and_runtime_proofs_assessed_here"] is False
    assert result["stable_profitability_proven"] is False and result["production_eligible"] is False


@pytest.mark.parametrize("failure", ["only_endpoints", "unknown_equity", "mismatched_final_cash"])
def test_terminal_gain_without_continuous_observations_is_insufficient(failure):
    state, start, end = mathematical_state()
    if failure == "only_endpoints":
        state["curve"] = [state["curve"][0], state["curve"][-1]]
        state["event_count"] = 2
    elif failure == "unknown_equity":
        state["curve"][100]["equity"] = None
    else:
        state["curve"][-1]["equity"] = "1000"
    result = assess_phase(state, start_ms=start, end_ms=end, ledger_identity="人工统计账本")
    assert not result["economic_conditions_met"] and result["block_interval"] is None


@pytest.mark.parametrize("failure", ["missing_funding", "reused_episode", "quality_failure"])
def test_complete_costs_unique_activities_and_quality_are_separate_requirements(failure):
    state, start, end = mathematical_state()
    if failure == "missing_funding":
        state["funding_windows"]["STATISTICAL_FIXTURE"] = []
    elif failure == "reused_episode":
        state["trades"][-1]["episode"] = state["trades"][0]["episode"]
    else:
        state["quality_errors"] = [{"reason": "OBSERVATION_GAP"}]
    result = assess_phase(state, start_ms=start, end_ms=end, ledger_identity="人工统计账本")
    assert not result["economic_conditions_met"]
    if failure == "missing_funding":
        assert result["net_return"] is None and result["block_interval"] is None


@pytest.mark.parametrize(
    "failure", ["same_ledger", "noncontiguous", "changed_rules", "bootstrap_version"]
)
def test_replication_cannot_reuse_a_ledger_skip_a_period_or_change_the_account(failure):
    state, start, end = mathematical_state()
    first = assess_phase(state, start_ms=start, end_ms=end, ledger_identity="人工第一阶段")
    second_state, second_start, second_end = mathematical_state(end)
    second = assess_phase(
        second_state, start_ms=second_start, end_ms=second_end, ledger_identity="人工第二阶段"
    )
    assert compare_phases(first, second)["economic_replication_conditions_met"]
    second = copy.deepcopy(second)
    if failure == "same_ledger":
        second["ledger_identity"] = first["ledger_identity"]
    elif failure == "noncontiguous":
        second["start_ms"] += DAY
    elif failure == "changed_rules":
        second["account_protocol_sha256"] = "另一套账户规则"
    else:
        second["block_interval"]["numpy_version"] = "另一统计实现"
    result = compare_phases(first, second)
    assert not result["economic_replication_conditions_met"] and not result["production_eligible"]
