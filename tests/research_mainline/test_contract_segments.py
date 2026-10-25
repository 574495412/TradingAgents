"""New launch timing, data-age reset and old-wallet separation invariants."""

import copy
from pathlib import Path

import pandas as pd
import pytest

from research_tools.contract_lifecycles.episode_cash import require_registered_episode_cash
from research_tools.contract_lifecycles.segments import (
    age_and_quote,
    context_at,
    require_episode_terminal_cash,
)
from research_tools.contract_lifecycles.timeline import UnverifiedContractHistory
from research_tools.factor_lab.io import read


def registry():
    return read(Path("docs/research/regime_program/contract_segments_20261026/registry.json"))


def market():
    return pd.DataFrame(
        {"quote_volume": 1e8, "price_liquidity_feature_usable": True},
        index=pd.date_range("2024-09-10", "2026-08-31", freq="D", tz="UTC"),
    )


def test_launch_is_not_backdated_to_its_notice_publication():
    r = registry()
    assert context_at(r, "AERGOUSDT", "2025-04-16T08:00Z")["blocked"]
    after = context_at(r, "AERGOUSDT", "2025-04-16T11:00Z")
    assert not after["blocked"] and after["feature_start"] == "2025-04-17T00:00:00+00:00"
    assert not after["portfolio_risk_reset"]


def test_partial_launch_day_and_prior_contract_do_not_supply_age_or_lookback():
    r, f = registry(), market()
    for clock, days in [
        ("2025-04-21T00:00Z", 4),
        ("2025-04-28T00:00Z", 11),
        ("2025-05-05T00:00Z", 18),
        ("2025-05-12T00:00Z", 25),
    ]:
        context = context_at(r, "AERGOUSDT", clock)
        result = age_and_quote(f, context, clock)
        assert result["valid_full_observations_in_episode"] == days
        assert result["quote28"] is None and not result["eligible"]


def test_180_full_days_are_required_inside_the_new_episode():
    r, f = registry(), market()
    for clock, count in [("2025-10-13T00:00Z", 179), ("2025-10-14T00:00Z", 180)]:
        a = age_and_quote(f, context_at(r, "AERGOUSDT", clock), clock)
        assert a["valid_full_observations_in_episode"] == count
        assert a["eligible"] == (count == 180)


def test_new_launch_never_clears_unverified_old_position_settlement():
    r = registry()
    old = next(
        c
        for c in r["cards"]
        if c["symbol"] == "AERGOUSDT" and c["settlement_at"].startswith("2025-03-27")
    )
    with pytest.raises(UnverifiedContractHistory, match="cannot transfer"):
        require_episode_terminal_cash(r, "AERGOUSDT", old["episode"], "2025-04-21T00:00Z", 1)
    require_episode_terminal_cash(r, "AERGOUSDT", old["episode"], "2025-04-21T00:00Z", 0)


def test_future_terminal_notice_has_no_effect_on_earlier_context():
    r = registry()
    pruned = copy.deepcopy(r)
    pruned["cards"] = [
        c
        for c in r["cards"]
        if pd.Timestamp(c["available_at_assumed"]) <= pd.Timestamp("2025-05-01T00:00Z")
    ]
    assert context_at(r, "AERGOUSDT", "2025-05-01T00:00Z") == context_at(
        pruned, "AERGOUSDT", "2025-05-01T00:00Z"
    )


def test_generic_futures_dates_are_distinct_from_spot_headlines():
    r = registry()
    alpaca = next(c for c in r["cards"] if c["symbol"] == "ALPACAUSDT")
    assert alpaca["scope_indirect"] and alpaca["settlement_at"].startswith("2025-04-30")
    assert context_at(r, "ALPACAUSDT", "2025-05-01T00:00Z")["blocked"]
    btt = next(c for c in r["cards"] if c["symbol"] == "BTTUSDT")
    assert btt["entry_stop_at"] is None


def test_reported_trade_archives_do_not_invent_a_BLZ_reopening():
    r = registry()
    assert r["source_conflicts"][0]["reported_trades_after_notice_end"]
    assert context_at(r, "BLZUSDT", "2023-09-04T00:00Z")["blocked"]


def test_unregistered_or_foreign_inventory_episode_cannot_skip_terminal_cash():
    r = registry()
    with pytest.raises(UnverifiedContractHistory, match="not registered"):
        require_registered_episode_cash(r, "AERGOUSDT", "invented", "2025-04-21T00:00Z", 1)
    foreign = next(c["episode"] for c in r["cards"] if c["symbol"] == "FTTUSDT")
    with pytest.raises(UnverifiedContractHistory, match="not registered"):
        require_registered_episode_cash(r, "AERGOUSDT", foreign, "2025-04-21T00:00Z", 1)
