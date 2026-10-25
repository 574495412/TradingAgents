"""容量来源的纯合成试验；不读取真实历史或运行收益账户。"""

import csv
import gzip
from copy import deepcopy
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from tradingagents.research.quant.artifacts import save, sha
from tradingagents.research.quant.backtesting import capacity_sources as sources


def snapshot_row(symbol="A", **changes):
    row = {
        "symbol": symbol,
        "quoteAsset": "USDT",
        "contractType": "PERPETUAL",
        "status": "SETTLING",
        "onboardDate": 0,
        "deliveryDate": 0,
        "filters": [
            {"filterType": "LOT_SIZE", "minQty": "0.0030", "maxQty": "1000", "stepSize": "0.0020"},
            {
                "filterType": "MARKET_LOT_SIZE",
                "minQty": "0.0050",
                "maxQty": "100",
                "stepSize": "0.0040",
            },
            {"filterType": "MIN_NOTIONAL", "notional": "5.000"},
            {
                "filterType": "PRICE_FILTER",
                "minPrice": "0.0000100",
                "maxPrice": "100000",
                "tickSize": "0.0000100",
            },
        ],
    }
    row.update(changes)
    return row


def parse(rows, symbols=("A", "UNKNOWN")):
    return sources.parse_rules(
        symbols,
        {"symbols": rows},
        snapshot_sha256="f" * 64,
        captured_at_utc="2026-10-14T02:05:38.339564Z",
    )


def test_rules_keep_full_population_decimal_origins_and_raw_filters():
    raw = snapshot_row()
    rules = parse([raw])
    assert set(rules) == {"A", "UNKNOWN"}
    row = rules["A"]
    assert row["rule_known"] and row["captured_status"] == "SETTLING"
    assert row["lots"] == [
        {
            "filter_type": "LOT_SIZE",
            "min_qty": "0.003",
            "max_qty": "1000",
            "step_size": "0.002",
            "offset": "0.003",
        },
        {
            "filter_type": "MARKET_LOT_SIZE",
            "min_qty": "0.005",
            "max_qty": "100",
            "step_size": "0.004",
            "offset": "0.005",
        },
    ]
    assert row["min_notional"] == "5" and row["price_filter"]["tick_size"] == "0.00001"
    assert row["raw_filters"] == raw["filters"]
    row["raw_filters"][0]["minQty"] = "99"
    assert raw["filters"][0]["minQty"] == "0.0030"
    assert not row["historical_pti_proven"] and row["min_notional_uses_raw_price_proxy"]
    assert not row["market_average_fill_requires_tick_grid"]
    assert rules["UNKNOWN"]["rule_known"] is False
    assert rules["UNKNOWN"]["lots"] == [] and rules["UNKNOWN"]["min_notional"] is None


@pytest.mark.parametrize("value", ["0", "-1", "NaN", "Infinity", "bad", 0.01, None])
def test_invalid_decimal_keeps_unknown_instead_of_defaulting(value):
    raw = snapshot_row()
    raw["filters"][0]["stepSize"] = value
    rule = parse([raw])["A"]
    assert not rule["rule_known"] and rule["lots"] == [] and rule["min_notional"] is None


def test_absent_duplicate_and_reversed_filters_are_unknown():
    for field in ("absent", "duplicate", "reversed", "wrong_quote"):
        raw = snapshot_row()
        if field == "absent":
            raw["filters"].pop(1)
        elif field == "duplicate":
            raw["filters"].append(deepcopy(raw["filters"][0]))
        elif field == "reversed":
            raw["filters"][0]["maxQty"] = "0.001"
        else:
            raw["quoteAsset"] = "USDC"
        assert not parse([raw])["A"]["rule_known"]


def test_incompatible_lot_origins_not_silently_reset_to_zero():
    row = snapshot_row()
    row["filters"][0].update(minQty="1", stepSize="2")
    row["filters"][1].update(minQty="2", stepSize="4")
    result = parse([row])["A"]
    assert result["rule_known"]
    assert [lot["offset"] for lot in result["lots"]] == ["1", "2"]
    assert [lot["step_size"] for lot in result["lots"]] == ["2", "4"]


def test_unicode_and_duplicate_identities():
    assert parse([snapshot_row("牛来USDT")], ("牛来USDT",))["牛来USDT"]["rule_known"]
    with pytest.raises(ValueError, match="重复"):
        parse([snapshot_row(), snapshot_row()])
    with pytest.raises(ValueError, match="唯一"):
        parse([snapshot_row()], ("A", "A"))


def binding_fixture(tmp_path, monkeypatch):
    raw, binding, result, receipt = [
        tmp_path / name
        for name in ("exchange_info.json", "catalog_binding.json", "result.json", "receipts.json")
    ]
    save(raw, {"serverTime": 100, "symbols": [snapshot_row("A0")]})
    save(binding, {"sha256": sha(raw), "captured_at_utc": "2026-10-14T02:05:38.339564Z"})
    save(
        result,
        {
            "snapshot_sha256": sha(raw),
            "snapshot_captured_at_utc": "2026-10-14T02:05:38.339564Z",
            "historical_point_in_time_membership_proven": False,
            "source_paths_sha256": {str(raw): sha(raw), str(binding): sha(binding)},
        },
    )
    save(receipt, {"result.json": sha(result)})
    monkeypatch.setattr(sources, "ELIGIBILITY_RECEIPTS_SHA256", sha(receipt))
    return [raw, binding, result, receipt]


def test_load_rules_binds_four_inputs_and_preserves_909(tmp_path, monkeypatch):
    paths = binding_fixture(tmp_path, monkeypatch)
    identities = {}
    rules, metadata = sources.load_rules(
        [f"A{i}" for i in range(909)], *paths, input_identities=identities
    )
    assert len(rules) == 909 and len(identities) == 4
    assert metadata["rule_status_counts"] == {"KNOWN_STATIC_RULE": 1, "UNKNOWN_RULE": 908}
    assert metadata["input_identities"] == identities
    assert metadata["historical_minute_mark_prices_available"] is False


@pytest.mark.parametrize(
    "problem",
    ["wrong_receipt", "changed_raw", "prebound", "symlink", "partial_population", "concurrent"],
)
def test_rule_binding_rejects_wrong_sources_and_concurrent_change(tmp_path, monkeypatch, problem):
    paths = binding_fixture(tmp_path, monkeypatch)
    symbols = [f"A{i}" for i in range(909)]
    identities = {}
    if problem == "wrong_receipt":
        monkeypatch.setattr(sources, "ELIGIBILITY_RECEIPTS_SHA256", "0" * 64)
    elif problem == "changed_raw":
        paths[0].write_text("{}")
    elif problem == "prebound":
        identities[str(paths[0])] = "0" * 64
    elif problem == "symlink":
        linked = tmp_path / "linked.json"
        linked.symlink_to(paths[0])
        paths[0] = linked
    elif problem == "partial_population":
        symbols.pop()
    else:
        original = sources._read

        def changed(path):
            value = original(path)
            if path == paths[-1]:
                paths[0].write_text("{}")
            return value

        monkeypatch.setattr(sources, "_read", changed)
    with pytest.raises(ValueError):
        sources.load_rules(symbols, *paths, input_identities=identities)


def minutes(n=180):
    return pd.DataFrame(
        {"quote_volume": np.arange(n, dtype=float) + 100},
        index=np.arange(1, n + 1, dtype=np.int64) * sources.MINUTE,
    )


def test_budget_exact_sixty_completed_minutes_and_future_perturbation():
    frame = minutes()
    actual = sources.quote_budget_view(frame)
    assert actual.iloc[:59].budget.isna().all()
    for index in (59, 80, 179):
        expected = 0.1 * min(
            frame.quote_volume.iloc[index], frame.quote_volume.iloc[index - 59 : index + 1].mean()
        )
        assert actual.iloc[index].budget == pytest.approx(expected)
        assert actual.iloc[index].window_start_ms == frame.index[index - 59]
        assert actual.iloc[index].window_end_ms == frame.index[index]
    changed = frame.copy()
    changed.loc[frame.index[80] :, "quote_volume"] *= 1000
    pd.testing.assert_frame_equal(actual.iloc[:80], sources.quote_budget_view(changed).iloc[:80])


def test_budget_zero_is_known_gap_and_nonfinite_are_unknown_for_all_sixty():
    frame = minutes()
    frame.loc[:, "quote_volume"] = 0
    zero = sources.quote_budget_view(frame)
    assert zero.iloc[59:].known60.all() and zero.iloc[59:].budget.eq(0).all()
    for missing in (np.nan, np.inf):
        broken = frame.copy()
        broken.iloc[80, 0] = missing
        values = sources.quote_budget_view(broken)
        assert values.iloc[80:140].budget.isna().all()
        assert values.iloc[140].known60 and values.iloc[140].budget == 0
    frame.iloc[80, 0] = -1
    with pytest.raises(ValueError, match="负"):
        sources.quote_budget_view(frame)


@pytest.mark.parametrize("problem", ["gap", "duplicate", "float", "off_grid"])
def test_calendar_cannot_compress_missing_minutes(problem):
    frame = minutes()
    if problem == "gap":
        frame = frame.drop(index=frame.index[70])
    elif problem == "duplicate":
        frame = pd.concat([frame.iloc[:1], frame])
    elif problem == "float":
        frame.index = frame.index.astype(float)
    else:
        frame.index = frame.index + 1
    with pytest.raises(ValueError, match="完成边界"):
        sources.quote_budget_view(frame)


def fake_market(tmp_path, monkeypatch, *, mutate=False, truncated=False):
    monkeypatch.setattr(sources.historical_panel, "START_MS", 0)
    monkeypatch.setattr(sources.historical_panel, "END_MS", 180 * sources.MINUTE)
    universe, records, identities = {}, {}, {}
    frames = {}
    for symbol, asset, quote in (
        ("A", "CRYPTO_PERPETUAL", 100),
        ("B", "TRADIFI_PERPETUAL", 1000),
        ("UNKNOWN", "UNKNOWN", 0),
    ):
        universe[symbol] = {
            "snapshot_symbol_found": True,
            "qualification_interval_known": True,
            "asset_class": asset,
            "onboard_ms": 0,
            "delivery_ms": 0,
        }
        path = tmp_path / (symbol + ".csv.gz")
        path.write_bytes(b"synthetic source identity")
        identities[str(path)] = sha(path)
        records[symbol] = [{"path": path}]
        frame = minutes()
        frame["quote_volume"], frame["open"], frame["close"] = quote, 100.0, 101.0
        frames[symbol] = frame
    calls = []

    def loader(symbol, selected):
        assert selected == records[symbol]
        calls.append(symbol)
        if mutate:
            selected[0]["path"].write_bytes(b"changed during read")
        return frames[symbol].iloc[:-1] if truncated else frames[symbol]

    monkeypatch.setattr(sources.historical_panel, "load_minutes", loader)
    return universe, records, identities, frames, calls


def test_market_two_budgets_have_exact_asof_do_not_mix_assets_or_future_close(
    tmp_path, monkeypatch
):
    universe, records, identities, frames, calls = fake_market(tmp_path, monkeypatch)
    clock = 81 * sources.MINUTE
    frames["A"].loc[clock, "quote_volume"] = 10_000
    frames["A"].loc[clock, "close"] = np.nan
    report = {}
    emitted = list(
        sources.market_frames(
            universe,
            records,
            tmp_path,
            report,
            input_identities=identities,
            start_ms=clock,
            end_ms=clock + sources.MINUTE,
        )
    )
    _, market = emitted[0]
    assert calls == ["A", "B", "UNKNOWN"]
    assert set(market) == set(universe)
    assert market["A"]["open_budget"] == 10
    assert market["A"]["decision_budget"] > 10
    assert market["B"]["open_budget"] == market["B"]["decision_budget"] == 100
    assert market["UNKNOWN"]["open_budget"] == 0 and not market["UNKNOWN"]["qualified"]
    assert market["A"]["observed"] and np.isnan(market["A"]["close"])
    assert market["A"]["open_budget_asof_ms"] == clock - sources.MINUTE
    assert market["A"]["decision_budget_asof_ms"] == clock
    assert set(market["A"]) == {
        "open",
        "close",
        "qualified",
        "observed",
        "open_budget",
        "decision_budget",
        "open_budget_asof_ms",
        "decision_budget_asof_ms",
    }
    assert report["A"]["input_minutes"] == 180 and report["A"]["minutes"] == 2


@pytest.mark.parametrize(
    "problem", ["unbound", "changed", "short", "wrong_population", "existing_map"]
)
def test_market_requires_bound_full_source_and_preserves_failed_scratch(
    tmp_path, monkeypatch, problem
):
    universe, records, identities, _, _ = fake_market(
        tmp_path, monkeypatch, mutate=problem == "changed", truncated=problem == "short"
    )
    if problem == "unbound":
        identities.clear()
    elif problem == "wrong_population":
        records.pop("B")
    elif problem == "existing_map":
        (tmp_path / "capacity_budgets.npy").write_bytes(b"retain prior attempt")
    with pytest.raises(ValueError):
        list(
            sources.market_frames(
                universe,
                records,
                tmp_path,
                {},
                input_identities=identities,
                start_ms=81 * sources.MINUTE,
                end_ms=82 * sources.MINUTE,
            )
        )
    if problem == "existing_map":
        assert (tmp_path / "capacity_budgets.npy").read_bytes() == b"retain prior attempt"


def test_primary_rho_and_full_memory_shape():
    assert Decimal("0.10") == sources.RHO
    estimate = sources.memory_estimate(909)
    assert estimate["market_minutes"] == 66_241
    assert estimate["market_mapped_bytes"] == 909 * 66_241 * 34
    assert estimate["budget_rho"] == "0.10"


def test_real_twelve_field_loader_bridge_keeps_all_87840_input_minutes(tmp_path):
    # 仅写120行合成原文字段，实际load_minutes负责验证并保留整段日历中的未知。
    first_output = sources.historical_panel.START_MS + 15 * sources.DAY
    start_open = first_output - 120 * sources.MINUTE
    path = tmp_path / "synthetic.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([*sources.historical_panel.RAW_COLUMNS, "completed_boundary_ms", "symbol"])
        for index in range(120):
            opening = start_open + index * sources.MINUTE
            writer.writerow(
                [
                    opening,
                    "100",
                    "102",
                    "99",
                    "101",
                    "2",
                    opening + sources.MINUTE - 1,
                    str(100 + index),
                    2,
                    "1",
                    "50",
                    "0",
                    opening + sources.MINUTE,
                    "A",
                ]
            )
    universe = {
        "A": {
            "snapshot_symbol_found": True,
            "qualification_interval_known": True,
            "asset_class": "CRYPTO_PERPETUAL",
            "onboard_ms": 0,
            "delivery_ms": 0,
        }
    }
    records = {
        "A": [
            {"path": path, "month": "2026-08", "profile": {"rows": 120}},
            {"path": None, "month": "2026-09", "profile": {}},
        ]
    }
    report = {}
    emitted = list(
        sources.market_frames(
            universe,
            records,
            tmp_path,
            report,
            input_identities={str(path): sha(path)},
            start_ms=first_output,
            end_ms=first_output + sources.MINUTE,
        )
    )
    first, after_gap = emitted[0][1]["A"], emitted[1][1]["A"]
    assert first["open_budget"] == pytest.approx(0.1 * np.mean(np.arange(159, 219)))
    assert first["decision_budget"] == pytest.approx(0.1 * np.mean(np.arange(160, 220)))
    assert (
        after_gap["decision_budget"] is None
        and after_gap["open_budget"] == first["decision_budget"]
    )
    assert not after_gap["observed"]
    assert report["A"]["input_minutes"] == 87_840
