"""验证资金费反馈、盈利续开权限、成交时序与独立现金审计。"""

import io
import zipfile
from copy import deepcopy

import pandas as pd
import pytest

from tradingagents.dataflows.research.public_reference import ReferenceStore, parse_prices
from tradingagents.research.quant.backtesting.account import Account
from tradingagents.research.quant.backtesting.basis import BasisSignals
from tradingagents.research.quant.backtesting.basis_attribution import Quotes, components
from tradingagents.research.quant.backtesting.exit import ExitAccount
from tradingagents.research.quant.backtesting.open_interest import InterestSignals
from tradingagents.research.quant.backtesting.search import Audit, select_candidate
from tradingagents.research.quant.backtesting.signals import MINUTE, Signal, Signals
from tradingagents.research.quant.strategy_audit import save, sha


def market(price=100.0):
    return {"PUBLIC": (price, price, True, True)}


def step(account, auditor, clock, price, pulses=(), qualified=None, final=False):
    quotes = market(price)
    account.open_phase(clock - MINUTE, quotes)
    account.close_phase(clock, quotes, pulses, qualified or {}, final=final)
    auditor.consume(clock, quotes, final=final)


@pytest.mark.parametrize("direction", [-1, 1])
def test_funding_uses_actual_held_inventory_and_future_open(direction):
    funding = [{"symbol": "PUBLIC", "time_ms": 3 * MINUTE + 1, "mark_price": 100.0, "rate": 0.01}]
    account = Account("test", 20, 3, funding)
    audit = Audit(account, funding)
    signal = Signal("PUBLIC", direction, 1.0, MINUTE, "activity")
    for clock in range(MINUTE, 7 * MINUTE, MINUTE):
        step(
            account,
            audit,
            clock,
            100,
            [signal] if clock == MINUTE else [],
            final=clock == 6 * MINUTE,
        )
    assert account.fills[0]["clock"] == 2 * MINUTE
    assert account.funding_ledger[0]["cash"] * direction < 0
    assert account.trades[0]["funding"] == account.funding_ledger[0]["cash"]
    assert audit.finish()["passed"]


def test_funding_cost_cancels_apparent_profitable_receipt():
    funding = [{"symbol": "PUBLIC", "time_ms": 3 * MINUTE + 1, "mark_price": 101.0, "rate": 0.03}]
    account = Account("damorne_countermove", 2, 3, funding, reentry=True)
    audit = Audit(account, funding)
    signal = Signal("PUBLIC", 1, 1.0, MINUTE, "activity")
    for clock in range(MINUTE, 7 * MINUTE, MINUTE):
        current = Signal("PUBLIC", 1, 1.0, clock, "activity")
        step(
            account,
            audit,
            clock,
            100 if clock < 4 * MINUTE else 101,
            [signal] if clock == MINUTE else [],
            {"PUBLIC": current},
            final=clock == 6 * MINUTE,
        )
    assert account.trades[0]["gross_pnl"] > 0
    assert account.trades[0]["net_pnl"] < 0
    assert len(account.trades) == 1 and not account.receipts
    audit.finish()


def test_real_positive_close_authorizes_only_owned_single_use_reentry():
    account = Account("damorne_countermove", 2, 3, [], reentry=True)
    audit = Audit(account, [])
    signal = Signal("PUBLIC", 1, 1.0, MINUTE, "activity")
    for clock in range(MINUTE, 9 * MINUTE, MINUTE):
        current = Signal("PUBLIC", 1, 1.0, clock, "activity")
        step(
            account,
            audit,
            clock,
            100 if clock < 4 * MINUTE else 102,
            [signal] if clock == MINUTE else [],
            {"PUBLIC": current},
            final=clock == 8 * MINUTE,
        )
    assert len(account.trades) == 2
    assert account.trades[1]["receipt_from"] == account.trades[0]["cycle_id"]
    assert audit.finish()["profit_receipts_used"] == 1


def test_source_funding_omission_rejected_by_independent_audit():
    funding = [{"symbol": "PUBLIC", "time_ms": 3 * MINUTE + 1, "mark_price": 100.0, "rate": 0.01}]
    account = Account("test", 20, 3, funding)
    signal = Signal("PUBLIC", 1, 1.0, MINUTE, "activity")
    quotes_by_clock = {}
    for clock in range(MINUTE, 7 * MINUTE, MINUTE):
        quotes = market()
        account.open_phase(clock - MINUTE, quotes)
        account.close_phase(
            clock, quotes, [signal] if clock == MINUTE else [], {}, final=clock == 6 * MINUTE
        )
        quotes_by_clock[clock] = quotes
    tampered = deepcopy(account)
    tampered.funding_ledger.clear()
    audit = Audit(tampered, funding)
    with pytest.raises(ValueError):
        for clock, quotes in quotes_by_clock.items():
            audit.consume(clock, quotes, final=clock == 6 * MINUTE)
        audit.finish()


@pytest.fixture
def shock_frame():
    row = {
        "eligible": True,
        "observed": True,
        "asset_class": "TRADIFI_PERPETUAL",
        "ret_20_pct": 2.0,
        "ret_60_pct": 3.0,
        "ret_20_rank_pct": 0.99,
        "ret_60_rank_pct": 0.99,
        "atr_rank_pct": 0.95,
        "atr_pct": 1.0,
        "ret_3_pct": -0.1,
        "taker_imbalance_3": -0.2,
        "volume_ratio_rank_pct": 0.5,
        "liquidity_rank_pct": 0.5,
        "ret_20_residual_pct": 2.0,
        "return_volatility_60_pct": 0.3,
        "taker_imbalance_12": -0.2,
    }
    return pd.DataFrame([row], index=pd.Index(["PUBLIC"], name="symbol"))


def test_shock_anchor_does_not_expire_into_invalid_lookup_while_price_stays_ready(shock_frame):
    signals = Signals(lambda symbol, clock: (0.5, 0.5, 0.5))
    pulses = []
    for clock in range(MINUTE, 63 * MINUTE, MINUTE):
        fresh, _ = signals.observe(clock, shock_frame)
        pulses.extend(fresh["shock_confirmed_fade_positioning"])
    assert len(pulses) == 1


def test_unknown_positioning_does_not_create_crowding_signal(shock_frame):
    signals = Signals(lambda symbol, clock: None)
    pulses, _ = signals.observe(MINUTE, shock_frame)
    assert pulses["shock_fade"]
    assert not pulses["shock_confirmed_fade_positioning"]
    assert not pulses["shock_positioning_long"]


def test_profit_receipt_cannot_authorize_fill_at_its_expiry_boundary():
    account = Account("damorne_countermove", 60, 3, [], reentry=True)
    receipt = {
        "source_cycle_id": "positive",
        "episode": "activity",
        "direction": -1,
        "available_ms": MINUTE,
        "expires_ms": 4 * MINUTE,
        "net_pnl": 1.0,
    }
    signal = Signal("PUBLIC", -1, 1.0, 3 * MINUTE, "activity")
    account.receipts["PUBLIC"] = receipt
    account.schedule(signal, 3 * MINUTE, receipt)
    assert not account.pending


def test_expiring_reserved_receipt_cancels_its_pending_order():
    account = Account("damorne_countermove", 60, 3, [], reentry=True)
    receipt = {
        "source_cycle_id": "positive",
        "episode": "activity",
        "direction": -1,
        "available_ms": MINUTE,
        "expires_ms": 4 * MINUTE,
        "net_pnl": 1.0,
    }
    signal = Signal("PUBLIC", -1, 1.0, 2 * MINUTE, "activity")
    account.receipts["PUBLIC"] = receipt
    account.schedule(signal, 2 * MINUTE, receipt)
    account.close_phase(4 * MINUTE, market(), [], {})
    assert not account.pending and not account.receipts
    assert account.declines[-1]["reason"] == "profit_receipt_expired_before_fill"


def test_open_interest_change_uses_only_declared_lagged_observations(shock_frame):
    calls = []

    class Observations:
        def at(self, symbol, clock):
            calls.append(clock)
            return (90.0, 1.2, 1.3, 0.9) if clock == 5 * MINUTE else (100.0, 1.2, 1.3, 0.9)

    signals = InterestSignals(lambda symbol, clock: (1.2, 1.3, 0.9), Observations())
    pulses, _ = signals.observe(10 * MINUTE, shock_frame)
    assert calls == [5 * MINUTE, -10 * MINUTE]
    assert len(pulses["shock_fade_oi_decrease_long_crowded"]) == 1
    assert not pulses["shock_fade_oi_increase"]


def test_entry_atr_does_not_read_the_unfinished_fill_candle():
    signal = Signal("PUBLIC", 1, 1.0, MINUTE, "activity", atr_fraction=0.005)
    outcomes = []
    for future_close in (80.0, 120.0):
        account = ExitAccount("damorne_public__atr_stop", 60, 3, [])
        account.schedule(signal, MINUTE)
        account.open_phase(2 * MINUTE, {"PUBLIC": (100.0, future_close, True, True)})
        outcomes.append(
            (
                account.positions["PUBLIC"]["entry_atr_fraction"],
                account.positions["PUBLIC"]["quantity"],
            )
        )
    assert outcomes[0] == outcomes[1] and outcomes[0][0] == 0.005


def test_atr_stop_waits_for_later_open_and_is_independently_audited():
    account = ExitAccount("damorne_public__atr_stop", 60, 3, [])
    audit = Audit(account, [])
    signal = Signal("PUBLIC", 1, 1.0, MINUTE, "activity", atr_fraction=0.005)
    for clock in range(MINUTE, 8 * MINUTE, MINUTE):
        step(
            account,
            audit,
            clock,
            100 if clock < 4 * MINUTE else 98.9,
            [signal] if clock == MINUTE else [],
            final=clock == 7 * MINUTE,
        )
    assert account.trades[0]["reason"] == "two_entry_atr_stop"
    assert account.trades[0]["exit_ms"] == 5 * MINUTE
    assert account.trades[0]["net_pnl"] < 0
    audit.finish()


def test_reversal_exit_requires_observed_direction_then_actual_opposite_flow():
    account = ExitAccount("damorne_public__state_reversal", 60, 3, [])
    audit = Audit(account, [])
    signal = Signal("PUBLIC", 1, 1.0, MINUTE, "activity")
    for clock in range(MINUTE, 8 * MINUTE, MINUTE):
        account.current_state = {"PUBLIC": (0.1, 0.2) if clock < 4 * MINUTE else (-0.1, -0.2)}
        step(
            account,
            audit,
            clock,
            100,
            [signal] if clock == MINUTE else [],
            final=clock == 7 * MINUTE,
        )
    assert account.trades[0]["reason"] == "confirmed_flow_reversal_failed"
    assert account.trades[0]["exit_ms"] == 5 * MINUTE
    audit.finish()


def reference_zip(value="100", *, clock=1791935940000):
    """构造官方字段格式的单根指数分钟，不包含模型或交易数据。"""
    row = [clock, "100", "101", "99", value, "0", clock + 59999, "0", "0", "0", "0", "0"]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("PUBLIC.csv", ",".join(str(v) for v in row) + "\n")
    return buffer.getvalue()


def test_reference_uses_close_clock_and_midnight_previous_day(tmp_path):
    clock = int(pd.Timestamp("2026-10-14", tz="UTC").value // 1_000_000)
    path = tmp_path / "reference.zip"
    path.write_bytes(reference_zip(clock=clock - MINUTE))
    values = parse_prices(path.read_bytes(), day="2026-10-13")
    assert values == {clock: 100.0}
    save(
        tmp_path / "manifest.json",
        {
            "records": [
                {
                    "symbol": "PUBLIC",
                    "day": "2026-10-13",
                    "status": "VERIFIED",
                    "path": str(path),
                    "sha256": sha(path),
                }
            ]
        },
    )
    store = ReferenceStore(tmp_path)
    assert store.at("PUBLIC", clock) == 100
    assert store.at("PUBLIC", clock + MINUTE) is None


@pytest.mark.parametrize("value", ["nan", "inf", "0", "-1"])
def test_invalid_reference_price_is_rejected(value):
    clock = int(pd.Timestamp("2026-10-13", tz="UTC").value // 1_000_000)
    with pytest.raises(ValueError):
        parse_prices(reference_zip(value, clock=clock), day="2026-10-13")


class ChangingReference:
    """指数每分钟有轻微可见变化，便于检验更新与时序边界。"""

    paths = {("PUBLIC", "2026-09-28"): None}

    def at(self, symbol, clock):
        return 100 + clock / MINUTE * 0.00001


def test_basis_requires_two_older_completed_confirmations(shock_frame):
    reference = ChangingReference()
    warmup = {t * MINUTE: {"PUBLIC": reference.at("PUBLIC", t * MINUTE)} for t in range(1, 71)}
    signals = BasisSignals(
        lambda symbol, clock: (0.5, 0.5, 0.5), reference, probability=1, warmup=warmup
    )
    counts = []
    for t in (71, 72, 73):
        frame = shock_frame.assign(close=reference.at("PUBLIC", t * MINUTE) * 1.006)
        pulses, _ = signals.observe(t * MINUTE, frame)
        counts.append(len(pulses["basis_candidate"]))
    assert counts == [0, 0, 1]


def test_basis_median_excludes_the_value_being_tested():
    reference = ChangingReference()
    signals = BasisSignals(lambda symbol, clock: (0.5, 0.5, 0.5), reference, probability=0)
    signals.history["PUBLIC"].extend(v * 0.0001 for v in range(60))
    clock = 71 * MINUTE
    signals.update_reference("PUBLIC", clock, reference.at("PUBLIC", clock) * 1.01)
    assert signals.last_delta["PUBLIC"] == pytest.approx(0.01 - 0.00295)


def test_missing_reference_never_becomes_zero_deviation(shock_frame):
    class MissingReference(ChangingReference):
        def at(self, symbol, clock):
            return None

    signals = BasisSignals(lambda symbol, clock: (0.5, 0.5, 0.5), MissingReference(), probability=1)
    pulses, _ = signals.observe(MINUTE, shock_frame.assign(close=100.0))
    assert pulses["basis_parent"]
    assert not pulses["basis_candidate"] and not pulses["basis_sparse_control"]


def test_profitable_control_cannot_be_promoted_over_the_frozen_primary():
    definitions = [("primary", 20, 3.0, False), ("control", 20, 3.0, False)]
    good = {
        "net_return": 0.02,
        "maximum_close_drawdown": 0.01,
        "closed_trades": 50,
        "wilson_lower": 0.5,
        "profit_factor": 2.0,
    }
    training = {
        f"{name}_20m_seed_{cost}bps": good | {"net_return": -0.01 if name == "primary" else 0.02}
        for name in ("primary", "control")
        for cost in (3.0, 10.0)
    }
    assert select_candidate(training, definitions, selectable_names=("primary",)) is None
    assert select_candidate(training, definitions)[1].startswith("control_")


@pytest.mark.parametrize("direction", [-1, 1])
def test_basis_explanation_separates_index_catchup_from_contract_profit(direction):
    values = components(110, 110, 100, 110, direction)
    assert values["contract_signed_log_bps"] == 0
    assert values["index_signed_log_bps"] == pytest.approx(-values["basis_signed_log_bps"])
    repaired = components(110, 100, 100, 100, direction)
    assert repaired["index_signed_log_bps"] == 0
    assert repaired["contract_signed_log_bps"] == pytest.approx(repaired["basis_signed_log_bps"])


def test_attribution_uses_actual_index_open_and_explicit_terminal_close(tmp_path):
    clock = int(pd.Timestamp("2026-10-14", tz="UTC").value // 1_000_000)
    path = tmp_path / "index.zip"
    path.write_bytes(reference_zip("105", clock=clock - MINUTE))
    save(
        tmp_path / "manifest.json",
        {
            "records": [
                {
                    "symbol": "PUBLIC",
                    "day": "2026-10-13",
                    "status": "VERIFIED",
                    "path": str(path),
                    "sha256": sha(path),
                }
            ]
        },
    )
    quotes = Quotes(tmp_path)
    assert quotes.at("PUBLIC", clock - MINUTE) == 100
    assert quotes.at("PUBLIC", clock, closing=True) == 105
    assert quotes.at("PUBLIC", clock) is None
