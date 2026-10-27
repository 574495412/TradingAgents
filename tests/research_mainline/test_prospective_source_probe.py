"""A current-looking payload cannot substitute for a valid pre-quote barrier."""

import copy

import pytest

from research_tools.dated_carry.collect_prospective_probe_v3 import (
    clock_valid,
    configured_http_proxy,
    select_contracts,
)


def fixture():
    spot = {
        "symbols": [
            {
                "symbol": "BTCUSDT",
                "baseAsset": "BTC",
                "quoteAsset": "USDT",
                "status": "TRADING",
                "isSpotTradingAllowed": True,
                "filters": [{"filterType": "LOT_SIZE"}],
            }
        ]
    }
    common = {
        "baseAsset": "BTC",
        "quoteAsset": "USDT",
        "marginAsset": "USDT",
        "pair": "BTCUSDT",
        "status": "TRADING",
        "filters": [{"filterType": "LOT_SIZE"}],
    }
    future = {
        "symbols": [
            {
                **common,
                "symbol": "BTCUSDT_261225",
                "contractType": "CURRENT_QUARTER",
                "deliveryDate": 1798185600000,
            },
            {
                **common,
                "symbol": "BTCUSDT_270326",
                "contractType": "NEXT_QUARTER",
                "deliveryDate": 1806048000000,
            },
        ]
    }
    return spot, future


def test_stale_time_or_failed_http_must_not_open_quote_gate():
    limits = {"maximum_server_vs_client_interval_error_ms": 5000, "maximum_request_rtt_ms": 5000}
    receipt = {
        "status": 200,
        "rtt_ms": 100,
        "sent_wall_ns": 1793091600000000000,
        "received_wall_ns": 1793091600100000000,
    }
    assert clock_valid(receipt, {"serverTime": 1793091600050}, limits)
    assert not clock_valid(receipt, {"serverTime": 1793072475512}, limits)
    receipt["status"] = 500
    assert not clock_valid(receipt, {"serverTime": 1793091600050}, limits)


def test_nearest_contract_selection_does_not_rank_on_price_or_return():
    spot, future = fixture()
    future["symbols"][0]["futureReturn"] = -100
    future["symbols"][1]["futureReturn"] = 100
    chosen = select_contracts(spot, future, ["BTC"], 1793091600000)
    assert chosen["BTC"]["future"]["symbol"] == "BTCUSDT_261225"


def test_margin_currency_mismatch_and_wrong_expiry_alias_rejected():
    spot, future = fixture()
    future["symbols"] = future["symbols"][:1]
    wrong = copy.deepcopy(future)
    wrong["symbols"][0]["marginAsset"] = "BTC"
    with pytest.raises(ValueError, match="No exact live"):
        select_contracts(spot, wrong, ["BTC"], 1793091600000)
    future["symbols"][0]["symbol"] = "BTCUSDT_270326"
    with pytest.raises(ValueError, match="expiry date"):
        select_contracts(spot, future, ["BTC"], 1793091600000)


def test_spot_identity_cannot_be_inferred_from_symbol_string_alone():
    spot, future = fixture()
    spot["symbols"][0]["baseAsset"] = "SOMETHING_ELSE"
    with pytest.raises(ValueError, match="Spot identity"):
        select_contracts(spot, future, ["BTC"], 1793091600000)


def test_only_existing_http_proxy_is_used_without_socks_dependency_or_bypass():
    assert (
        configured_http_proxy(
            {"https": "http://127.0.0.1:7897", "socks": "socks5://127.0.0.1:7897"}
        )
        == "http://127.0.0.1:7897"
    )
    with pytest.raises(ValueError, match="do not bypass"):
        configured_http_proxy({"socks": "socks5://127.0.0.1:7897"})
