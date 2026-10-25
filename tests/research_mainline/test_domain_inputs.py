"""Domain masks preserve observed timing, cash requests and risk state intentions."""

import pandas as pd
import pytest

from research_tools.regime_program.prepare_domains import gate_frame


def test_state_boundary_and_unknown_are_causal_without_rescaling_or_risk_changes():
    index = pd.date_range("2025-01-01", periods=48, freq="h", tz="UTC")
    base = pd.DataFrame({"target_A": 0.2, "target_B": -0.1, "risk_scale": 0.5}, index=index)
    states = pd.DataFrame({"state": ["trend_up", "range"]}, index=index[[0, 24]])
    out, mask = gate_frame(base, states, ["trend_up"])
    pd.testing.assert_frame_equal(out.loc[index[:24]], base.loc[index[:24]])
    assert out.loc[index[24:], ["target_A", "target_B"]].eq(0).all().all()
    assert mask.sum() == 24 and out.risk_scale.eq(0.5).all()
    changed = states.copy()
    changed.iloc[1, 0] = "trend_up"
    future, _ = gate_frame(base, changed, ["trend_up"])
    pd.testing.assert_frame_equal(future.iloc[:24], out.iloc[:24])
    early = base.copy()
    early.index = early.index - pd.Timedelta(days=2)
    unknown, _ = gate_frame(early, states, ["trend_up"])
    assert unknown.filter(like="target_").eq(0).all().all()


def test_liquidity_stress_cannot_be_enabled_to_manufacture_trades():
    index = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
    base = pd.DataFrame({"target_A": 1.0, "risk_scale": 1.0}, index=index)
    states = pd.DataFrame({"state": ["liquidity_stress"]}, index=index[:1])
    with pytest.raises(ValueError, match="never be enabled"):
        gate_frame(base, states, ["liquidity_stress"])
