"""Bounded prospective public source probe; no strategy, fills or account access."""

import argparse
import gzip
import io
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import getproxies

import requests
from websockets.sync.client import connect

from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest


def configured_http_proxy(proxies):
    """Reuse an existing HTTP CONNECT route; never rewrite machine proxy settings."""
    route = proxies.get("https") or proxies.get("http")
    if route:
        if urlsplit(route).scheme not in ["http", "https"]:
            raise ValueError("No existing supported HTTP CONNECT proxy route")
        return route
    if proxies.get("socks") or proxies.get("all"):
        raise ValueError("Only unsupported proxy transport configured; do not bypass it")
    return True


def clock_valid(receipt, data, limits):
    if (
        receipt.get("status") != 200
        or not isinstance(data, dict)
        or not isinstance(data.get("serverTime"), int)
    ):
        return False
    server = data["serverTime"]
    allowance = limits["maximum_server_vs_client_interval_error_ms"]
    return (
        receipt["rtt_ms"] <= limits["maximum_request_rtt_ms"]
        and receipt["sent_wall_ns"] / 1e6 - allowance
        <= server
        <= receipt["received_wall_ns"] / 1e6 + allowance
    )


def select_contracts(spot, future, assets, now_ms):
    if not isinstance(spot, dict) or not isinstance(future, dict):
        raise ValueError("Both raw metadata bodies required")
    spots = {s["symbol"]: s for s in spot.get("symbols", [])}
    selected = {}
    for asset in assets:
        symbol = asset + "USDT"
        cash = spots.get(symbol)
        if (
            not cash
            or cash.get("baseAsset") != asset
            or cash.get("quoteAsset") != "USDT"
            or cash.get("status") != "TRADING"
            or cash.get("isSpotTradingAllowed") is not True
        ):
            raise ValueError("Spot identity/status not available")
        candidates = [
            s
            for s in future.get("symbols", [])
            if s.get("baseAsset") == asset
            and s.get("quoteAsset") == s.get("marginAsset") == "USDT"
            and s.get("pair") == symbol
            and s.get("status") == "TRADING"
            and s.get("contractType") in ["CURRENT_QUARTER", "NEXT_QUARTER"]
            and re.fullmatch(symbol + r"_\d{6}", s.get("symbol", ""))
            and s.get("deliveryDate", 0) > now_ms + 600000
        ]
        if not candidates:
            raise ValueError("No exact live dated candidate")
        dated = min(candidates, key=lambda s: (s["deliveryDate"], s["symbol"]))
        if (
            datetime.fromtimestamp(dated["deliveryDate"] / 1000, UTC).strftime("%y%m%d")
            != dated["symbol"].rsplit("_", 1)[1]
        ):
            raise ValueError("Native expiry date and symbol disagree")
        for market in [cash, dated]:
            filters = market.get("filters", [])
            if not filters or len({f["filterType"] for f in filters}) != len(filters):
                raise ValueError("Missing/duplicate reported filters")
        selected[asset] = {"spot": cash, "future": dated}
    return selected


def collect(output):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "plan.json").exists() or (output / "completion.json").exists():
        raise ValueError("Do not overwrite/restart an existing capture")
    src = Path(__file__).with_name("prospective_source_plan_v003.json")
    plan = read(src)
    verify_manifest(Path(plan["parent"]))
    verify_manifest(Path(plan["technical_parent"]))
    save(
        output / "plan.json",
        {
            **plan,
            "source_hashes": {
                str(p): sha(p)
                for p in [
                    src,
                    Path(__file__),
                    Path(plan["parent"]) / "completion.json",
                    Path(plan["technical_parent"]) / "completion.json",
                ]
            },
        },
    )
    save(
        output / "registration.json",
        {
            "registered_at_utc": datetime.now(UTC).isoformat(),
            "registered_wall_ns": time.time_ns(),
            "plan_sha256": sha(output / "plan.json"),
            "before_native_quotes_and_any_economic_decision": True,
            "MCP_metadata_was_exploratory_before_registration": True,
            "new_financial_accounts": 0,
        },
    )
    receipts, lock = [], threading.Lock()

    def get(market, kind, stem, params=None):
        with lock:
            if len(receipts) >= plan["budget"]["maximum_native_REST_requests"]:
                raise ValueError("Native HTTP budget exhausted")
            receipt = {"stem": stem, "market": market, "kind": kind, "status": None}
            receipts.append(receipt)
        url = plan[market + "_rest"] + plan[market + "_endpoints"][kind]
        receipt.update(url=url, sent_wall_ns=time.time_ns(), sent_monotonic_ns=time.monotonic_ns())
        result = None
        try:
            with requests.get(
                url,
                params=params,
                headers={"Accept-Encoding": "gzip", "Cache-Control": "no-cache"},
                timeout=(10, 15),
                stream=True,
                allow_redirects=False,
            ) as response:
                receipt.update(
                    url=response.url,
                    status=response.status_code,
                    headers={
                        h: response.headers[h]
                        for h in [
                            "Date",
                            "Age",
                            "Cache-Control",
                            "ETag",
                            "Last-Modified",
                            "Content-Type",
                            "Content-Encoding",
                        ]
                        if h in response.headers
                    },
                )
                chunks, size = [], 0
                while piece := response.raw.read(65536, decode_content=False):
                    size += len(piece)
                    if size > plan["budget"]["maximum_HTTP_response_bytes"]:
                        raise ValueError("HTTP byte budget exceeded")
                    chunks.append(piece)
                body = b"".join(chunks)
            (output / (stem + ".raw")).write_bytes(body)
            receipt.update(raw_bytes=len(body), raw_sha256=sha(output / (stem + ".raw")))
            if (
                receipt["headers"].get("Content-Encoding", "").lower() == "gzip"
                or body[:2] == b"\x1f\x8b"
            ):
                with gzip.GzipFile(fileobj=io.BytesIO(body)) as zipped:
                    body = zipped.read(plan["budget"]["maximum_decoded_HTTP_bytes"] + 1)
            if len(body) > plan["budget"]["maximum_decoded_HTTP_bytes"]:
                raise ValueError("Decoded HTTP byte budget exceeded")
            (output / (stem + ".decoded")).write_bytes(body)
            receipt.update(
                decoded_bytes=len(body), decoded_sha256=sha(output / (stem + ".decoded"))
            )
            if receipt["status"] == 200:
                result = json.loads(body)
        except Exception as e:
            receipt.update(error_type=type(e).__name__, error=str(e)[:400])
        receipt.update(received_wall_ns=time.time_ns(), received_monotonic_ns=time.monotonic_ns())
        receipt["rtt_ms"] = (receipt["received_monotonic_ns"] - receipt["sent_monotonic_ns"]) / 1e6
        save(output / (stem + ".receipt.json"), receipt)
        print(
            {
                "stem": stem,
                "status": receipt["status"],
                "bytes": receipt.get("decoded_bytes"),
                "rtt_ms": receipt["rtt_ms"],
                "error_type": receipt.get("error_type"),
            },
            flush=True,
        )
        return receipt, result

    pre, post, stream_results, selection, failures = {}, {}, {}, None, []
    for market in ["spot", "future"]:
        receipt, body = get(market, "time", market + "_time_pre")
        pre[market + "_clock_valid"] = clock_valid(receipt, body, plan["clock"])
        pre[market + "_time"] = body
        params = (
            {"symbols": json.dumps([a + "USDT" for a in plan["assets"]], separators=(",", ":"))}
            if market == "spot"
            else None
        )
        _, pre[market + "_metadata"] = get(market, "metadata", market + "_metadata_pre", params)
    try:
        if not all(pre[m + "_clock_valid"] for m in ["spot", "future"]):
            raise ValueError(
                "Independent native clocks missing, stale or uncertain; do not use metadata.serverTime"
            )
        selection = select_contracts(
            pre["spot_metadata"],
            pre["future_metadata"],
            plan["assets"],
            max(pre[m + "_time"]["serverTime"] for m in ["spot", "future"]),
        )
        save(
            output / "selection.json",
            {
                "selected_at_wall_ns": time.time_ns(),
                "before_native_quote_access": True,
                "selection_not_ranked_on_basis_or_returns": True,
                "symbols": selection,
            },
        )
    except Exception as e:
        failures.append({"stage": "pre_quote_barrier", "type": type(e).__name__, "error": str(e)})

    if selection:
        connected = {m: threading.Event() for m in ["spot", "future"]}

        def stream(market):
            symbols = [selection[a][market]["symbol"].lower() for a in plan["assets"]]
            uri = plan[market + "_stream"].format(
                streams="/".join(s + "@" + c for s in symbols for c in plan["channels"])
            )
            record = {
                "market": market,
                "url": uri,
                "selection_sha256": sha(output / "selection.json"),
                "messages": 0,
                "bytes": 0,
                "attempted_wall_ns": time.time_ns(),
                "connected": False,
            }
            try:
                proxy = configured_http_proxy(getproxies())
                if isinstance(proxy, str):
                    route = urlsplit(proxy)
                    record["existing_proxy_route"] = {
                        "scheme": route.scheme,
                        "host": route.hostname,
                        "port": route.port,
                        "system_configuration_changed": False,
                    }
                with connect(
                    uri, proxy=proxy, open_timeout=10, close_timeout=1, max_size=1048576
                ) as ws:
                    record.update(connected=True, connected_wall_ns=time.time_ns())
                    connected[market].set()
                    print({"stream": market, "connected": True}, flush=True)
                    deadline = time.monotonic() + plan["budget"]["stream_seconds"]
                    with (output / (market + "_native.ndjson")).open("w") as file:
                        while time.monotonic() < deadline:
                            try:
                                native = ws.recv(timeout=min(1, deadline - time.monotonic()))
                            except TimeoutError:
                                continue
                            if not isinstance(native, str):
                                raise ValueError("Unexpected binary application message")
                            line = (
                                json.dumps(
                                    {
                                        "received_wall_ns": time.time_ns(),
                                        "received_monotonic_ns": time.monotonic_ns(),
                                        "native_raw": native,
                                    }
                                )
                                + "\n"
                            )
                            record["bytes"] += len(line.encode())
                            record["messages"] += 1
                            if (
                                record["bytes"] > plan["budget"]["maximum_stream_bytes_each"]
                                or record["messages"] > plan["budget"]["maximum_messages_each"]
                            ):
                                raise ValueError("Native stream budget reached")
                            file.write(line)
            except Exception as e:
                record.update(error_type=type(e).__name__, error=str(e)[:400])
            record["finished_wall_ns"] = time.time_ns()
            save(output / (market + "_stream.receipt.json"), record)
            return record

        with ThreadPoolExecutor(max_workers=2) as executor:
            jobs = {m: executor.submit(stream, m) for m in ["spot", "future"]}
            ready = [connected[m].wait(11) for m in ["spot", "future"]]
            if all(ready):
                with ThreadPoolExecutor(max_workers=2) as snapshots:
                    calls = [
                        snapshots.submit(
                            get,
                            market,
                            "depth",
                            market + "_" + asset + "_snapshot",
                            {"symbol": selection[asset][market]["symbol"], "limit": 1000},
                        )
                        for market in ["spot", "future"]
                        for asset in plan["assets"]
                    ]
                    for call in calls:
                        call.result()
            for market, job in jobs.items():
                stream_results[market] = job.result()
        for market in ["spot", "future"]:
            params = (
                {"symbols": json.dumps([a + "USDT" for a in plan["assets"]], separators=(",", ":"))}
                if market == "spot"
                else None
            )
            _, post[market + "_metadata"] = get(
                market, "metadata", market + "_metadata_post", params
            )
            receipt, body = get(market, "time", market + "_time_post")
            post[market + "_clock_valid"] = clock_valid(receipt, body, plan["clock"])
            post[market + "_time"] = body
    save(
        output / "probe_summary.json",
        {
            "pre_clock_valid": {m: pre[m + "_clock_valid"] for m in ["spot", "future"]},
            "selection_frozen": bool(selection),
            "native_HTTP_requests": len(receipts),
            "stream_results": stream_results,
            "failures": failures,
            "post_clock_valid": {m: post.get(m + "_clock_valid") for m in ["spot", "future"]},
            "source_observability_audited": False,
            "forward_strategy_validation_completed": False,
            "new_financial_accounts": 0,
            "orders_enabled": False,
            "profit_goal_met": False,
        },
    )
    save(
        output / "completion.json",
        {
            "capture_finished": True,
            "manifest": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
            "profit_goal_met": False,
        },
    )
    print(read(output / "probe_summary.json"), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    collect(p.parse_args().output)
