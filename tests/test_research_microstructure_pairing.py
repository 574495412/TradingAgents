"""配对保留未知时钟、未来消息、最新空盘口及所有被拒活动。"""

import copy

import pytest

from tests.test_research_microstructure_audit import fixture
from tests.test_research_paper_account import frame, public
from tradingagents.research.quant.backtesting import microstructure_pairing as module
from tradingagents.research.quant.backtesting.bidirectional import index_jobs
from tradingagents.research.quant.backtesting.microstructure import source_available
from tradingagents.research.quant.backtesting.paper_account import PaperAccount
from tradingagents.research.quant.strategy_audit import save


def test_latest_empty_quote_cannot_be_replaced_with_an_older_positive_quote(tmp_path):
    root = tmp_path / "观察"
    fixture(root, "latest_empty")
    messages = module.ReceivedMessages([root], tmp_path / "索引.db", "关联研究")
    try:
        value, epoch = messages.latest("quote", "A000USDT", 100_900, 102_000)
        assert value["update_id"] == 2 and value["ready"] is False
        assert (
            source_available(value, epoch, 100_900, 102_000, quote=True)
            == "CLOCK_OR_DISPLAYED_QUOTE_UNKNOWN"
        )
        assert messages.latest("quote", "A000USDT", 100_750, 102_000) == (None, None)
        assert messages.latest("quote", "A000USDT", 100_900, 100_850) == (None, None)
    finally:
        messages.close()


def test_old_message_cannot_cross_clock_unknown_or_future_prefix_watermark():
    value = {
        "generation": "连接",
        "received_ms": 100_900,
        "transaction_ms": 100_790,
        "event_ms": 100_800,
        "clock_verified": True,
        "ready": True,
    }
    physical = {
        "generation": "连接",
        "confirmed_ms": 100_300,
        "end_ms": None,
        "observed_through_ms": 104_000,
    }
    windows = [
        {"known_from_ms": 100_000, "known_through_ms": 100_950},
        {"known_from_ms": 101_100, "known_through_ms": 104_000},
    ]
    clipped = module.clock_epoch(value, physical, windows, 120_000)
    assert clipped["end_ms"] == 100_950
    assert source_available(value, clipped, 100_900, 102_000, quote=True) is not None
    physical["observed_through_ms"] = 100_920
    assert module.clock_epoch(value, physical, windows, 120_000)["end_ms"] == 100_920


def test_a_duplicate_physical_prefix_cannot_count_as_new_observations(tmp_path):
    first, second = tmp_path / "来源一", tmp_path / "来源二"
    fixture(first)
    fixture(second)
    with pytest.raises(ValueError, match="重复前缀"):
        module.ReceivedMessages([first, second], tmp_path / "索引.db", "关联研究")


def test_overlapping_independent_channels_cannot_be_selected_after_seeing_prices(tmp_path):
    first, second = tmp_path / "来源一", tmp_path / "来源二"
    fixture(first)
    fixture(second, generation_prefix="另一物理来源")
    with pytest.raises(ValueError, match="物理区间重叠"):
        module.ReceivedMessages([first, second], tmp_path / "索引.db", "关联研究")


@pytest.mark.parametrize("future", [False, True])
def test_price_point_uses_original_sixty_bar_baseline_and_exact_last_millisecond(future):
    closing = 120 * module.MINUTE
    _, _, params = index_jobs(["A"], closing)[0]
    rows, bars = [], {}
    for stamp in range(closing - 63 * module.MINUTE, closing + module.MINUTE, module.MINUTE):
        value = frame(stamp // module.MINUTE)
        rows.append(value["marks"]["A"]["row"])
        bars[stamp] = value["marks"]["A"]
    bars[closing] = copy.deepcopy(bars[closing])
    bars[closing]["row"][4] = "101"
    raw = public("index", rows, closing + 1100, closing + (6000 if future else 2000), "A", params)

    class Bars:
        def at(self, symbol, stamp, decision):
            return bars.get(stamp)

    class Cursor:
        indices = [(closing, raw)]
        symbols = ["A"]
        bars = Bars()

    point = frame(120)
    point["evidence"]["A"]["actual_decision_ms"] = closing + 4000
    if future:
        with pytest.raises(ValueError):
            module.price_point(Cursor(), point, "A", "活动", "账户")
    else:
        result = module.price_point(Cursor(), point, "A", "活动", "账户")
        assert result["arguments"]["data_ms"] == closing - 1
        assert result["arguments"]["price"] == "101" and result["arguments"]["median_basis"] == "0"


@pytest.mark.parametrize("startup_active", [False, True])
def test_activity_mapping_keeps_capacity_rejects_and_separate_cost_ledgers(
    monkeypatch, tmp_path, startup_active
):
    source = tmp_path / "原账户"
    source.mkdir()
    symbols = ["A", "B", "C", "D", "E"]
    protocol = {"symbols": symbols, "simulation_end_ms": 200 * module.MINUTE}
    models = {
        name: PaperAccount(symbols, int(name), period_end_ms=protocol["simulation_end_ms"])
        for name in ("3", "10")
    }
    frames = [
        frame(t, symbols, direction=1 if startup_active or t >= 103 else 0) for t in range(100, 106)
    ]
    for f in frames:
        for model in models.values():
            model.apply(f)
    save(source / "protocol.json", protocol)
    save(
        source / "checkpoint.json",
        {"accounts": {name: model.checkpoint() for name, model in models.items()}},
    )

    class Cursor:
        def __init__(self, *args):
            pass

        def advance(self, *args, **kwargs):
            pass

        def finish(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(module, "require_loaded_sources", lambda p: None)
    monkeypatch.setattr(module, "SourceCursor", Cursor)
    monkeypatch.setattr(
        module,
        "segment_entries",
        lambda p: (
            iter([(i, {"frame": f, "source_prefix": {}}, None) for i, f in enumerate(frames)])
            if p.name == "account_frames"
            else iter([])
        ),
    )
    monkeypatch.setattr(
        module,
        "price_point",
        lambda *a: {"endpoint_available": False, "reason": "人工未观察到同期价格"},
    )
    result, metrics = module.activities(source, tmp_path)
    assert all(len(ledger) == (0 if startup_active else 5) for ledger in result.values())
    if not startup_active:
        for name, ledger in result.items():
            assert sum(any(f["action"] == "OPEN" for f in a["fills"]) for a in ledger.values()) == 4
            assert sum(bool(a["declines"]) for a in ledger.values()) == 1
            assert all(
                a["ledger_identity"].endswith("slippage-bps-" + name) for a in ledger.values()
            )
