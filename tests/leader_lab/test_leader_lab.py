from dataclasses import asdict

import pytest

from research_tools.leader_lab.demo import make_panel
from research_tools.leader_lab.evidence import (
    Position,
    identity,
    load,
    normalize,
    profile,
    public_capture,
    resolve_identity,
)
from research_tools.leader_lab.learning import (
    HedgeBook,
    learn,
    predict,
    read_panel,
    replay,
    split_panel,
)


def position(index="a", side="LONG", opened=1704067200000, closed=1704067800000, pnl=1):
    return Position(index, "ETHUSDT", side, opened, closed, 100, 101, 1, 1, 2, pnl)


@pytest.mark.parametrize(
    "name",
    [
        "damorne",
        "DAmorné",
        "5219911688676500225",
        "https://www.binance.com/en/copy-trading/lead-details/5219911688676500225?timeRange=30D",
    ],
)
def test_one_leader_identity_across_name_url_and_id(name):
    assert identity(name) == "5219911688676500225"


@pytest.mark.parametrize(
    "name",
    [
        "https://attacker.test/lead-details/5219911688676500225",
        "https://user:pass@www.binance.com/en/copy-trading/lead-details/5219911688676500225",
        "javascript:alert(1)",
    ],
)
def test_other_destinations_and_url_credentials_are_not_accepted(name):
    with pytest.raises(ValueError):
        identity(name)


def test_duplicate_source_rows_and_wrong_portfolio_binding_reject(tmp_path):
    import json

    raw = asdict(position())
    with pytest.raises(ValueError, match="Repeated"):
        normalize([raw, raw])
    p = tmp_path / "cache.json"
    p.write_text(json.dumps({"portfolio_id": "111111", "closed_positions": [{"raw": raw}]}))
    with pytest.raises(ValueError, match="portfolio"):
        load(p, "222222")


def test_behavior_profile_preserves_hedging_and_loss_reentry_counterexample():
    a = position(closed=1704069000000, pnl=-2)
    b = position("b", "SHORT", 1704067260000, 1704067920000)
    c = position("c", opened=1704069060000, closed=1704069660000)
    result = profile([a, b, c])
    assert result["openings_with_prior_opposite_side_active"] == 1
    assert result["counterexamples_to_universal_profit_only_reentry"] == 1
    assert result["quant_strategy_identified"] is None
    assert result["reported_pnl_is_not_recomputed_net_account_return"]


def test_hedge_book_keeps_both_sides_and_reconciles_fee_and_funding_cash():
    book = HedgeBook(1000, 20, 2, 5, 0)
    book.execute("ETH", "OPEN_LONG", 100, 0, {"ETH": 100})
    book.execute("ETH", "OPEN_SHORT", 100, 1, {"ETH": 100})
    assert len(book.positions) == 2 and book.reserved() == 40
    book.settle_funding("ETH", 0.001, 100)
    assert book.funding == pytest.approx(0)
    book.execute("ETH", "CLOSE_LONG", 110, 2, {"ETH": 110})
    assert ("ETH", "SHORT") in book.positions
    book.execute("ETH", "CLOSE_SHORT", 90, 3, {"ETH": 90})
    assert book.wallet == pytest.approx(1007.92)
    assert sum(p["net_pnl"] for p in book.closed) == pytest.approx(7.92)
    assert book.fees == pytest.approx(0.08)


def test_margin_includes_opening_fees_and_slippage_losses():
    book = HedgeBook(20, 20, 2, 5, 3)
    book.execute("ETH", "OPEN_LONG", 100, 0, {"ETH": 100})
    assert not book.positions and book.rejections["insufficient_shared_margin"] == 1


def test_autonomous_replay_cannot_read_teacher_actions_or_inventory():
    import pandas as pd

    policy = {
        "features": ["x_long_open"],
        "tree": {
            "feature": "x_long_open",
            "threshold": 0.5,
            "left": {"action": "OPEN_LONG"},
            "right": {"action": "CLOSE_LONG"},
        },
    }
    frame = pd.DataFrame(
        [
            {
                "timestamp": i * 60000,
                "symbol": "ETH",
                "open": 100 + i,
                "mark": 100 + i,
                "funding_rate": 0,
                "x_long_open": 1,
                "action": "HOLD",
                "campaign_id": "teacher",
            }
            for i in range(20)
        ]
    )
    economics = {"capital": 1000, "stake": 20, "leverage": 2, "fee_bps": 5, "slippage_bps": 3}
    a = replay(policy, frame, economics)
    frame["x_long_open"], frame["action"], frame["campaign_id"] = 0, "OPEN_SHORT", "poisoned"
    b = replay(policy, frame, economics)
    assert a == b and a[0]["complete_positions"] > 0
    assert a[0]["positions_cash_reconciled"]


def test_panel_rejects_future_features_and_unobserved_holds_are_not_training_labels(tmp_path):
    import pandas as pd

    path = tmp_path / "panel.csv"
    make_panel(path, 600)
    frame, _, clocks = read_panel(path)
    train, _, _, _, _ = split_panel(frame, clocks)
    assert len(train) > 20
    frame.loc[frame.action.eq("HOLD"), "observed"] = False
    with pytest.raises(ValueError, match="observed HOLD"):
        split_panel(frame, clocks)
    raw = pd.read_csv(path)
    raw.loc[0, "available_at"] = "2030-01-01T00:00:00+00:00"
    raw.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Future feature"):
        read_panel(path)


def test_cross_boundary_episode_is_purged(tmp_path):
    path = tmp_path / "panel.csv"
    make_panel(path, 600)
    frame, _, clocks = read_panel(path)
    frame.loc[frame.index.isin([100, 400, 550]), "campaign_id"] = "cross-boundary"
    train, validation, test, _, _ = split_panel(frame, clocks)
    assert "cross-boundary" not in set(train.campaign_id)
    assert "cross-boundary" not in set(validation.campaign_id)
    assert "cross-boundary" in set(test.campaign_id)


def test_end_to_end_known_quant_rule_recovery_is_explicitly_synthetic(tmp_path):
    import json

    path, output = tmp_path / "synthetic.csv", tmp_path / "run"
    make_panel(path, 1200)
    output.mkdir()
    economics = {"capital": 1000, "stake": 20, "leverage": 2, "fee_bps": 5, "slippage_bps": 3}
    result = learn(path, output, economics, "999999")
    assert result["action_review"]["holdout_teacher_forced"]["macro_f1"] > 0.95
    assert result["account"]["complete_positions"] > 0
    assert result["account"]["data_kind"] == "SYNTHETIC_KNOWN_POLICY"
    assert "SYNTHETIC_KNOWN_POLICY" in (output / "report.md").read_text()
    assert result["account"]["production_eligible"] is False
    policy = json.loads((output / "candidate.json").read_text())
    assert predict(policy, {"x_signal": 0, "x_long_open": 0, "x_short_open": 0}) == "OPEN_LONG"
    other = tmp_path / "wrong"
    other.mkdir()
    with pytest.raises(ValueError, match="identity"):
        learn(path, other, economics, "111111")


def test_position_notional_overflow_is_rejected():
    row = asdict(position())
    row["entry_price"] = row["max_quantity"] = 1e308
    with pytest.raises(ValueError, match="Nonfinite"):
        normalize([row])


def fake_public(monkeypatch, bodies):
    import json
    import urllib.request

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, size):
            return json.dumps(bodies.pop(0)).encode()[:size]

    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: Response())


def test_public_pages_merge_only_complete_unique_closed_history(monkeypatch, tmp_path):
    raw = {**asdict(position()), "positionId": "a", "status": "All Closed"}
    other = {**raw, "positionId": "b"}
    fake_public(
        monkeypatch,
        [
            {"code": "000000", "data": {"list": [raw], "total": 2}},
            {"code": "000000", "data": {"list": [other], "total": 2}},
        ],
    )
    path = public_capture("123456", tmp_path)
    positions, info = load(path, "123456")
    assert len(positions) == 2 and info["coverage"] == "complete_closed_as_declared_only"


def test_public_history_that_changes_midpagination_is_not_accepted(monkeypatch, tmp_path):
    raw = {**asdict(position()), "positionId": "a", "status": "All Closed"}
    fake_public(
        monkeypatch,
        [
            {"code": "000000", "data": {"list": [raw], "total": 2}},
            {"code": "000000", "data": {"list": [], "total": 3}},
        ],
    )
    with pytest.raises(ValueError, match="changed"):
        public_capture("123456", tmp_path)
    assert not (tmp_path / "public_snapshot.json").exists()


def test_public_nickname_resolution_rejects_ambiguous_identical_names(monkeypatch):
    fake_public(
        monkeypatch,
        [
            {
                "code": "000000",
                "data": {"list": [{"nickname": "Some Quant", "leadPortfolioId": "123456"}]},
            }
        ],
    )
    assert resolve_identity("Some Quant")[0] == "123456"
    fake_public(
        monkeypatch,
        [
            {
                "code": "000000",
                "data": {
                    "list": [
                        {"nickname": "Some Quant", "leadPortfolioId": "123456"},
                        {"nickname": "Some Quant", "leadPortfolioId": "654321"},
                    ]
                },
            }
        ],
    )
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_identity("Some Quant")


def test_equal_clock_opposite_positions_still_establish_hedged_inventory():
    result = profile([position(), position("b", "SHORT")])
    assert result["openings_with_prior_opposite_side_active"] == 0
    assert result["opposite_side_overlapping_position_pairs"] == 1


def test_reentry_after_long_flat_gap_is_not_same_episode_counterexample():
    result = profile([position(pnl=-1), position("b", opened=1704240000000, closed=1704240060000)])
    assert result["counterexamples_to_universal_profit_only_reentry"] == 1
    assert result["same_assumed_episode_transitions"] == 0


def test_standalone_namespace_does_not_import_application_dotenv_bootstrap():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import research_tools.leader_lab; assert 'tradingagents' not in sys.modules; assert 'dotenv' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
