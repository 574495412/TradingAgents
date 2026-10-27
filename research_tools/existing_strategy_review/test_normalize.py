"""Financial boundary cases that can otherwise invert a strategy comparison."""

import math

import pandas as pd
import pytest

from research_tools.existing_strategy_review.normalize import curve_stats


def test_same_clock_pre_and_post_settlement_drawdown_is_preserved():
    frame = pd.DataFrame(
        {"date": ["2025-01-01", "2025-01-02", "2025-01-02"], "equity": [100.0, 200.0, 100.0]}
    )
    result = curve_stats(frame, 100.0)
    assert result["sampled_full_drawdown"] == 0.5
    assert result["last_equity"] == 100.0


def test_month_return_uses_known_initial_cash_not_first_post_cost_observation():
    frame = pd.DataFrame(
        {"date": ["2025-01-01 01:00:00", "2025-02-01 00:00:00"], "equity": [90.0, 110.0]}
    )
    result = curve_stats(frame, 100.0, "2025-01-01")
    assert result["known_initial_cash_seed_added"]
    assert result["month_returns"]["2025-01"] == pytest.approx(0.1)
    assert result["sampled_full_drawdown"] == pytest.approx(0.1)


def test_crypto_and_equity_sharpe_use_the_declared_calendar():
    frame = pd.DataFrame(
        {"date": ["2025-01-01", "2025-01-02", "2025-01-03"], "equity": [100.0, 110.0, 110.0]}
    )
    assert curve_stats(frame, 100.0, annualization=365)["recomputed_daily_sharpe"] == pytest.approx(
        math.sqrt(365 / 2)
    )
    assert curve_stats(frame, 100.0, annualization=252)["recomputed_daily_sharpe"] == pytest.approx(
        math.sqrt(252 / 2)
    )


def test_equity_weekend_does_not_erase_monday_return():
    frame = pd.DataFrame(
        {"date": ["2025-01-03", "2025-01-06", "2025-01-07"], "equity": [100.0, 110.0, 110.0]}
    )
    result = curve_stats(frame, 100.0, annualization=252)
    assert result["daily_return_count"] == 2
    assert result["recomputed_daily_sharpe"] == pytest.approx(math.sqrt(252 / 2))
