"""人工时间驱动整个实时账户循环，核对重连与真实缺口的不同权限。"""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_research_continuous_feed import candle
from tests.test_research_paper_account import MINUTE, frame, public
from tradingagents.research.quant.backtesting import (
    paper_live as module,
    paper_live_audit as auditor,
)
from tradingagents.research.quant.backtesting.continuous_feed import VERSION
from tradingagents.research.quant.backtesting.paper_audit import audit
from tradingagents.research.quant.backtesting.paper_store import StudyStore


@pytest.mark.parametrize("missing_minute", [False, True])
def test_real_account_loop_preserves_pending_across_wire_change_but_consumes_it_after_a_gap(
    tmp_path, monkeypatch, missing_minute
):
    now, original_sleep = [100 * MINUTE + 2000], asyncio.sleep
    fixed = {
        "A": {
            "asset_class": "TRADIFI_PERPETUAL",
            "trading": True,
            "onboard_ms": 0,
            "delivery_ms": 0,
        }
    }
    store = StudyStore(
        tmp_path / "artificial_study",
        {
            "symbols": ["A"],
            "contracts": fixed,
            "simulation_end_ms": 109 * MINUTE,
            "runtime_version": VERSION,
            "fixtures_only": True,
        },
    )
    captured = []

    class Client:
        def batch(self, jobs):
            return []

    async def sleep(seconds):
        now[0] += int(seconds * 1000)
        await original_sleep(0)

    async def startup(client, journal, contracts):
        return frame(100)["contracts"], fixed

    async def warm(client, journal, bars, symbols, closing, members):
        return {"binding_ms": closing, "stream_overlap_members": 1, "fixtures_only": True}

    async def feed(symbols, pending, journal, stop, status, **kwargs):
        status.update(
            subscription_confirmed=True,
            connection_generation="fixture:1",
            connection_started_ms=now[0],
        )
        emitted = None
        while not stop.is_set():
            if now[0] >= 106 * MINUTE:
                status.update(connection_generation="fixture:2", connection_started_ms=106 * MINUTE)
            closing = (now[0] - 1001) // MINUTE * MINUTE
            if closing > 100 * MINUTE and emitted != closing:
                emitted = closing
                if not missing_minute or closing != 106 * MINUTE:
                    record = candle("A", closing, status["connection_generation"], now[0])
                    journal.add(record)
                    pending.append(record)
            await original_sleep(0)

    async def capture(
        client, study, bars, metadata, closing, phase, generation, index_records=None
    ):
        minute = closing // MINUTE
        actual = max(now[0], closing + 6000)
        starts = [
            p["entry_ms"] + 1
            for account in study.accounts.values()
            for p in [*account.positions.values(), *account.trades]
        ]
        funding = []
        if starts and min(starts) <= actual - 3000:
            funding = [
                public(
                    "funding",
                    [],
                    actual - 2000,
                    actual - 1500,
                    symbol="A",
                    params={
                        "symbol": "A",
                        "startTime": min(starts),
                        "endTime": actual - 3000,
                        "limit": 1000,
                    },
                )
            ]
        value = frame(minute, direction=1 if minute >= 105 else 0, now=actual, funding=funding)
        value.update(phase=phase, source_generation=generation)
        now[0] = actual
        return value, value["contracts"]

    original_commit = store.commit

    def commit(value):
        captured.append(value)
        original_commit(value)

    monkeypatch.setattr(module, "milliseconds", lambda: now[0])
    monkeypatch.setattr(module.asyncio, "sleep", sleep)
    monkeypatch.setattr(module, "startup_identity", startup)
    monkeypatch.setattr(module, "warm_history", warm)
    monkeypatch.setattr(module, "continuous_stream", feed)
    monkeypatch.setattr(module, "capture_frame", capture)
    monkeypatch.setattr(store, "commit", commit)
    try:
        result = asyncio.run(module.live(store, Client()))
        assert result["failure"] is None
        for account in store.accounts.values():
            audit(iter(captured), account.checkpoint())
            if missing_minute:
                assert not account.fills
                assert any(e["reason"] == "OBSERVATION_GAP" for e in account.quality_errors)
                assert any(e["reason"] == "GAP_ACTIVITY_CONSUMED" for e in account.declines)
            else:
                assert len(account.trades) == 1 and not account.pending
                assert not account.quality_errors
                assert account.fills[0]["entry_ms"] >= 107 * MINUTE
                assert not any(f["source_restart"] for f in captured)
                assert {f["source_generation"] for f in captured} == {"fixture:1", "fixture:2"}
    finally:
        store.close()


@pytest.mark.parametrize(
    "profile,duration,storage",
    [
        ("finite", 3600, "segmented-gzip-v1"),
        ("diagnostic-soak", 30 * 86400, "segmented-gzip-v1"),
        ("diagnostic-soak", 3600, "flat-jsonl"),
    ],
)
def test_diagnostic_profile_cannot_be_used_as_an_unregistered_thirty_day_entry(
    tmp_path, profile, duration, storage
):
    with pytest.raises(ValueError):
        module.run(
            tmp_path,
            tmp_path / "proposal.json",
            tmp_path / "output",
            duration,
            False,
            storage,
            profile,
        )
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("condition", ["still_possible", "missed", "early_proven_late_pending"])
def test_irrecoverable_early_funding_does_not_create_an_endless_retry_loop(condition):
    trade = {"symbol": "A", "entry_ms": 100 * MINUTE, "exit_ms": 120 * MINUTE}
    windows = []
    if condition == "early_proven_late_pending":
        windows.append(
            {
                "start_ms": trade["entry_ms"] + 1,
                "end_ms": trade["exit_ms"],
                "requested_ms": trade["exit_ms"] + 2000,
            }
        )
    account = SimpleNamespace(trades=[trade], funding_windows={"A": windows})
    now = trade["exit_ms"] + (MINUTE - 1 if condition == "still_possible" else MINUTE + 1)
    if condition == "missed":
        with pytest.raises(ValueError, match="首轮资金费确认期限已错过"):
            module.require_recoverable_funding({"10": account}, now)
    else:
        module.require_recoverable_funding({"10": account}, now)


def test_auditor_checks_loaded_model_in_addition_to_archived_paths():
    functions = [
        auditor.evidence,
        auditor.Bars.add,
        auditor.clock_valid,
        auditor.closed_candle,
        auditor.PaperAccount.apply,
        auditor.funding_window,
        auditor.entries,
        auditor.read,
        auditor.save,
        auditor.sha,
        auditor.verify_handoff,
        auditor.segment_entries,
    ]
    protocol = {
        "runtime_version": VERSION,
        "storage": "segmented-gzip-v1",
        "source_sha256": {
            str(Path(f.__code__.co_filename).resolve()): auditor.sha(Path(f.__code__.co_filename))
            for f in functions
        },
    }
    auditor.require_loaded_sources(protocol)
    name = str(Path(auditor.PaperAccount.apply.__code__.co_filename).resolve())
    protocol["source_sha256"][name] = "另一目录中保存的不同模型身份"
    with pytest.raises(ValueError, match="实际载入的核验公式或账户模型"):
        auditor.require_loaded_sources(protocol)
