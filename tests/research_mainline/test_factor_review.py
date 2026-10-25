"""Causal training and economic selection checks for the bounded review."""

import json

import numpy as np
import pandas as pd

from research_tools.factor_lab_review.rolling import fit_predict
from research_tools.factor_lab_review.run import selection


def sample():
    t = pd.date_range("2024-01-01", "2025-01-10", freq="h", tz="UTC")
    index = pd.MultiIndex.from_product([t, ["A", "B", "C", "D"]], names=["time", "symbol"])
    rng = np.random.default_rng(42)
    x = pd.DataFrame(rng.normal(size=(len(index), 2)) + [1, -2], columns=["a", "b"], index=index)
    y = pd.Series(x.a * 0.01 - x.b * 0.02 + 0.001, index=index)
    return x, y


def test_future_labels_and_maturation_gap_never_change_model_or_predictions():
    x, y = sample()
    score, m = fit_predict(x, y, "2025-01-01", "2025-01-05", list(x))
    other = y.copy()
    other.loc[other.index.get_level_values("time") >= pd.Timestamp("2024-12-31", tz="UTC")] = 1000
    score2, m2 = fit_predict(x, other, "2025-01-01", "2025-01-05", list(x))
    pd.testing.assert_series_equal(score, score2)
    assert m == m2
    assert pd.Timestamp(m["training_last_label_clock"]) < pd.Timestamp(m["evaluation_start"])
    # Features after the evaluated prefix cannot change fitted coefficients or earlier forecasts.
    altered = x.copy()
    altered.loc[altered.index.get_level_values("time") >= pd.Timestamp("2025-01-03", tz="UTC")] = (
        999
    )
    score3, m3 = fit_predict(altered, y, "2025-01-01", "2025-01-05", list(x))
    pd.testing.assert_series_equal(score.iloc[:100], score3.iloc[:100])
    assert m == m3


def test_centered_ridge_matches_augmented_unpenalized_intercept_solution():
    x, y = sample()
    _, m = fit_predict(x, y, "2025-01-01", "2025-01-05", list(x))
    clocks = x.index.get_level_values("time")
    start = pd.Timestamp("2025-01-01", tz="UTC")
    train = (clocks >= start - pd.Timedelta(days=180)) & (clocks + pd.Timedelta(hours=24) < start)
    z = ((x.loc[train] - pd.Series(m["median"])) / pd.Series(m["scale"])).clip(-5, 5)
    design = np.column_stack([np.ones(len(z)), z])
    reference = np.linalg.solve(design.T @ design + np.diag([0, 100, 100]), design.T @ y.loc[train])
    np.testing.assert_allclose([m["intercept"], *m["coefficients"]], reference, atol=1e-12)


def test_selection_rejects_cost_sensitive_candidate_and_ignores_later_results(tmp_path):
    rows = []
    for cost in (3, 10):
        for i in range(4):
            identity = f"{cost}_{i}"
            folder = tmp_path / "accounts" / identity
            folder.mkdir(parents=True)
            (folder / "trades.json").write_text(
                json.dumps([{"net_pnl": 10}] * 8 + [{"net_pnl": -1}] * 2)
            )
            rows.append(
                {
                    "id": identity,
                    "candidate": "test",
                    "window": str(i),
                    "slippage_bps": cost,
                    "metrics": {
                        "net_return": 0.01 if cost == 3 or i < 2 else -0.001,
                        "sharpe": 1,
                        "max_drawdown": 0.02,
                        "beta": 0,
                        "turnover": 1,
                        "open_pnl_with_exit_reserve": 0,
                        "fees": 1,
                        "slippage_cost": 1,
                    },
                }
            )
    # A wildly profitable later window must not rescue a failed validation candidate.
    rows.append(
        {"candidate": "test", "window": "2026", "slippage_bps": 10, "metrics": {"net_return": 100}}
    )
    result = selection(
        rows,
        tmp_path,
        {
            "candidates": ["test"],
            "costs_slippage_bps": [3, 10],
            "validation_windows": list(map(str, range(4))),
        },
    )
    assert result["selected_candidate"] == "cash"
    assert "10:insufficient_positive_windows" in result["assessments"][0]["reasons"]
