"""Source forecast amplitude, absolute risk targets and causal input scaling."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from factors.portfolio.source_sizing import sized_intent

P = json.loads(
    (
        Path(__file__).resolve().parents[2] / "research_tools/regime_program/sizing_plan.json"
    ).read_text()
)["parameters"]


def test_weak_forecasts_are_not_scaled_back_up_to_full_risk():
    vol = pd.Series({"A": 0.02, "B": 0.02})
    weak = sized_intent(pd.Series({"A": 0.4, "B": -0.4}), vol, 1.0, P)
    strong = sized_intent(pd.Series({"A": 0.8, "B": -0.8}), vol, 1.0, P)
    np.testing.assert_allclose(strong["risk"], weak["risk"] * 2)
    assert weak["risk"].abs().sum() < 0.8
    assert strong["risk"].abs().sum() < 1
    assert weak["conviction"].abs().sum() == 1


def test_risk_budget_and_source_buffer_follow_independent_cash_risk_equations():
    raw = pd.Series({"A": 1.0, "B": -2.0})
    vol = pd.Series({"A": 0.02, "B": 0.04})
    result = sized_intent(raw, vol, 1, P)
    average_a = 0.16 / (365**0.5) / 0.02 / 2
    average_b = 0.16 / (365**0.5) / 0.04 / 2
    assert abs(result["risk"]["A"] - average_a / 0.8) < 1e-12
    assert abs(result["risk"]["B"] + average_b * 2 / 0.8) < 1e-12
    assert abs(result["buffer"]["A"] - average_a * 0.1 / 0.8) < 1e-12


def test_forecast_caps_and_gross_caps_only_reduce_sizing():
    raw = pd.Series({"A": 50.0, "B": -10.0})
    vol = pd.Series({"A": 0.001, "B": 0.001})
    result = sized_intent(raw, vol, 1, P)
    assert abs(result["risk"].abs().sum() - 1) < 1e-12
    assert abs(result["risk"]["A"] + result["risk"]["B"]) < 1e-12
    missing = sized_intent(raw * float("nan"), vol, 1, P)
    assert missing["risk"].eq(0).all()
