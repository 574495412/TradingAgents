"""Causal formation and statistical-family regression checks; no alpha assertions."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from factors.portfolio.cointegrated_pairs import bh_adjust, form_pairs

stats = pytest.importorskip(
    "statsmodels.stats.multitest", reason="Optional isolated pair research runtime"
)
pytestmark = pytest.mark.skipif(
    int(pd.__version__.split(".")[0]) >= 3,
    reason="Pinned statsmodels0.14.5 uses isolated pandas2 research runtime",
)


def parameters():
    return json.loads(Path("research_tools/regime_program/pair_plan.json").read_text())["formation"]


def synthetic():
    rng = np.random.default_rng(7102)
    index = pd.date_range("2023-01-01", periods=450, tz="UTC", freq="D")
    common = 4 + np.cumsum(rng.normal(0, 0.04, len(index)))
    error = np.zeros(len(index))
    for i in range(1, len(index)):
        error[i] = 0.5 * error[i - 1] + rng.normal(0, 0.025)
    close = pd.DataFrame(
        {"A": np.exp(common), "B": np.exp(0.3 + 1.2 * common + error)}, index=index
    )
    return close, close * 0 + 100_000_000, index[365]


def test_bh_matches_reference_and_retains_unknown_family_slots():
    p = [0.003, 0.04, 0.9, 0.001, 0.07]
    np.testing.assert_allclose(bh_adjust(p), stats.multipletests(p, method="fdr_bh")[1])
    result = bh_adjust([0.01, np.nan, 0.03])
    np.testing.assert_allclose(result[[0, 2]], [0.03, 0.045])
    assert np.isnan(result[1])
    with pytest.raises(ValueError, match="P-values"):
        bh_adjust([-0.01, 0.04])


def test_known_cointegration_and_disjoint_lifecycle_ownership():
    close, quote, clock = synthetic()
    result = form_pairs(close, quote, clock, parameters(), start=close.index[0])
    assert result["directional_tests"] == 2
    assert result["selected"][0]["beta"] == pytest.approx(1.2, abs=0.03)
    assert result["selected"][0]["half_life_days"] == pytest.approx(1, abs=0.4)
    close["C"] = close.B * 1.3
    quote["C"] = quote.B
    result = form_pairs(close, quote, clock, parameters(), start=close.index[0])
    assert len(result["selected"]) == 1
    symbols = [r[s] for r in result["selected"] for s in ("x", "y")]
    assert len(set(symbols)) == len(symbols)


def test_current_and_future_prices_turnover_cannot_change_formation():
    close, quote, clock = synthetic()
    expected = form_pairs(close, quote, clock, parameters(), start=close.index[0])
    close.loc[clock:] *= np.arange(len(close.loc[clock:]))[:, None] + 100
    quote.loc[clock:] = 0
    assert form_pairs(close, quote, clock, parameters(), start=close.index[0]) == expected


def test_unknown_dates_rejected_and_cost_gate_does_not_become_profit():
    close, quote, clock = synthetic()
    with pytest.raises(ValueError, match="timezone"):
        form_pairs(close, quote, clock.tz_localize(None), parameters())
    p = parameters()
    p["minimum_two_sigma_edge_unit_gross"] = 1
    result = form_pairs(close, quote, clock, p, start=close.index[0])
    assert result["selected"] == []
    assert result["tests"][0]["rejection"] == "cost_hurdle"
    assert result["returns_used_for_selection"] is False
