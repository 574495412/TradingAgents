"""Meaningful boundary and revision cases from the captured official schedules."""

import copy
from pathlib import Path

import pytest

from research_tools.contract_lifecycles.timeline import (
    UnverifiedContractHistory,
    daily_feature_allowed,
    require_certified_history,
    require_terminal_cash,
    restriction_at,
    schedules_at,
    validate,
)
from research_tools.factor_lab.io import read


def cards():
    return read(Path("research_tools/contract_lifecycles/event_cards.json"))["cards"]


def test_future_notice_does_not_backdate_restriction():
    c = cards()
    assert not schedules_at(c, "FTTUSDT", "2022-11-12T00:00Z")
    assert restriction_at(c, "FTTUSDT", "2022-11-14T04:00Z")["blocked"]


def test_postponements_replace_deadlines_only_after_assumed_availability():
    c = cards()
    before = schedules_at(c, "OMGUSDT", "2024-12-13T00:00Z")
    after = schedules_at(c, "OMGUSDT", "2024-12-19T00:00Z")
    final = schedules_at(c, "OMGUSDT", "2024-12-21T00:00Z")
    assert len(before) == len(after) == len(final) == 1
    assert before[0]["settlement_at"].startswith("2024-12-16")
    assert after[0]["settlement_at"].startswith("2024-12-30")
    assert final[0]["settlement_at"].startswith("2025-01-31")
    assert final[0]["entry_cutoff_inherited_not_reaffirmed"]
    require_terminal_cash(c, "OMGUSDT", "2024-12-17T00:00Z", 1)
    with pytest.raises(UnverifiedContractHistory):
        require_terminal_cash(c, "OMGUSDT", "2025-01-31T09:00Z", 1)


def test_later_terminal_notice_never_invents_relisting_or_continuity():
    c = cards()
    assert restriction_at(c, "BLZUSDT", "2023-06-01T00:00Z")["blocked"]
    assert restriction_at(c, "AERGOUSDT", "2026-06-01T00:00Z")["blocked"]


def test_missing_entry_cutoff_is_not_automatically_half_hour_before_end():
    c = cards()
    s = schedules_at(c, "FTTUSDT", "2022-11-14T03:59Z")
    assert s[0]["effective_conservative_entry_stop"] is None
    assert not restriction_at(c, "FTTUSDT", "2022-11-14T03:59Z")["blocked"]


def test_partial_terminal_day_cannot_enter_a_completed_daily_feature():
    c = cards()
    assert daily_feature_allowed(c, "FTTUSDT", "2022-11-13T00:00Z", "2022-11-15T00:00Z")
    assert not daily_feature_allowed(c, "FTTUSDT", "2022-11-14T00:00Z", "2022-11-15T00:00Z")
    with pytest.raises(ValueError, match="unfinished"):
        daily_feature_allowed(c, "FTTUSDT", "2022-11-15T00:00Z", "2022-11-15T10:00Z")


def test_flat_terminal_does_not_need_fictitious_cash_but_nonflat_does():
    c = cards()
    require_terminal_cash(c, "FTTUSDT", "2022-11-14T04:00Z", 0)
    with pytest.raises(UnverifiedContractHistory):
        require_terminal_cash(c, "FTTUSDT", "2022-11-14T04:00Z", 3)
    with pytest.raises(UnverifiedContractHistory, match="provisional"):
        require_certified_history(c)


def test_product_and_revision_identity_are_not_interchangeable():
    c = copy.deepcopy(cards())
    a = next(x for x in c if x["kind"] == "schedule_amendment")
    a["episode"] = "unrelated"
    with pytest.raises(ValueError, match="another contract episode"):
        validate(c)
