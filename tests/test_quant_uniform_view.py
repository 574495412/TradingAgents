"""来源冲突、原状态保留和未完成采集的合并选择反证。"""

import pytest

from tradingagents.dataflows.research.uniform_view import CACHE_PATH_ERROR, VERIFIED, choose_source


def test_matching_success_uses_primary_while_unequal_bytes_stay_conflicted():
    primary = {"status": VERIFIED, "sha256": "abc"}
    assert choose_source(primary, primary) == (VERIFIED, "primary")
    assert choose_source(primary, primary | {"sha256": "def"}) == (
        "CONFLICTING_SUCCESSFUL_BYTES",
        None,
    )


def test_official_404_cannot_be_erased_by_successful_cache():
    absent = {"status": "PUBLIC_404"}
    assert choose_source(absent, {"status": VERIFIED}) == ("PUBLIC_404_CONFLICTS_WITH_CACHE", None)
    assert choose_source(absent, None) == ("PUBLIC_404", "primary")


@pytest.mark.parametrize(
    "primary",
    [
        None,
        {"status": "NOT_FETCHED_AFTER_STOP"},
        {"status": "FAILED_OR_UNKNOWN", "error": CACHE_PATH_ERROR},
    ],
)
def test_only_declared_pending_or_path_failure_can_use_registered_cache(primary):
    assert choose_source(primary, {"status": VERIFIED}) == (VERIFIED, "registered_cache")


def test_other_failures_are_not_hidden_even_when_old_cache_succeeded():
    unknown = {"status": "FAILED_OR_UNKNOWN", "error": "未知数据字段冲突"}
    assert choose_source(unknown, {"status": VERIFIED}) == ("FAILED_OR_UNKNOWN", None)
    assert choose_source(None, {"status": "FAILED_OR_UNKNOWN"}) == (
        "PRIMARY_NOT_YET_RECORDED",
        None,
    )
