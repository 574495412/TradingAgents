"""Check forward filtering against path enumeration and strict training maturity."""

import itertools
import math

import numpy as np
import pandas as pd

from research_tools.regime_program.models import filter_step, predictions


def test_gaussian_filter_matches_independent_enumeration_without_future_observations():
    observations = np.array([[0.1], [1.5], [-0.5]])
    start = np.array([0.6, 0.4])
    transition = np.array([[0.9, 0.1], [0.2, 0.8]])
    means, variances = np.array([[0.0], [2.0]]), np.array([[1.0], [0.5]])
    previous = None
    for end in range(1, 4):
        total = np.zeros(2)
        for path in itertools.product(range(2), repeat=end):
            likelihood = start[path[0]]
            for t, state in enumerate(path):
                if t:
                    likelihood *= transition[path[t - 1], state]
                v, mu = variances[state, 0], means[state, 0]
                likelihood *= math.exp(-0.5 * (observations[t, 0] - mu) ** 2 / v) / math.sqrt(
                    2 * math.pi * v
                )
            total[path[-1]] += likelihood
        reference = total / total.sum()
        previous = filter_step(observations[end - 1], start, transition, means, variances, previous)
        np.testing.assert_allclose(previous, reference, atol=1e-12)


def test_fixed_models_do_not_use_future_labels_scaling_or_observations():
    rng = np.random.default_rng(11)
    index = pd.date_range("2024-01-01", "2025-03-31", freq="D", tz="UTC")
    x = pd.DataFrame(
        {
            "trend_z": rng.normal(0, 1, len(index)),
            "efficiency": rng.uniform(0, 1, len(index)),
            "volatility_ratio": rng.uniform(0.5, 2, len(index)),
            "breadth": rng.uniform(0, 1, len(index)),
            "eligible_count": rng.integers(10, 30, len(index)),
        },
        index=index,
    )
    benchmark = pd.DataFrame(
        {"open": np.exp(rng.normal(0, 0.02, len(index)).cumsum())}, index=index
    )
    result, fits = predictions(x, benchmark, end="2025-03-01")
    changed = x.copy()
    changed.loc["2025-02-01":, "trend_z"] *= 100
    prices = benchmark.copy()
    prices.loc["2025-02-01":, "open"] *= 3
    alternative, _ = predictions(changed, prices, end="2025-02-01")
    pd.testing.assert_frame_equal(result.loc[:"2025-01-31"], alternative)
    assert fits and all(
        pd.Timestamp(f["maximum_label_available"]) < pd.Timestamp(f["fit_at"]) for f in fits
    )
    assert all(f["status"] == "fitted" for f in fits)
    assert result.linear_probability.notna().all()
