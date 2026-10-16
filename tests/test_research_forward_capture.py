"""验证真实接收时序、原文身份及固定采集集合，不调用外网。"""

import base64
import gzip
import hashlib
import json

import pytest

from tradingagents.dataflows.research.live_capture import (
    MINUTE,
    available_rows,
    body,
    canonical,
    clock_valid,
    dependencies,
    digest,
    seal_batch,
    verify,
)
from tradingagents.research.quant.strategy_audit import save


def record(values, *, kind="klines", requested=3 * MINUTE, received=3 * MINUTE + 100):
    raw = canonical(values)
    return {
        "kind": kind,
        "symbol": "PUBLIC",
        "status": 200,
        "requested_ms": requested,
        "received_ms": received,
        "local_clock_consistent": True,
        "body_base64": base64.b64encode(raw).decode(),
        "body_sha256": hashlib.sha256(raw).hexdigest(),
    }


def candle(opening):
    return [
        opening,
        "100",
        "101",
        "99",
        "100",
        "1",
        opening + MINUTE - 1,
        "100",
        1,
        "1",
        "100",
        "0",
    ]


def test_late_arriving_inputs_cannot_backfill_an_earlier_decision():
    r = record([candle(MINUTE)])
    assert (
        available_rows(
            r, decision_ms=r["received_ms"] - 1, cutoff_ms=2 * MINUTE, clock_verified=True
        )
        == []
    )
    assert available_rows(
        r, decision_ms=r["received_ms"], cutoff_ms=2 * MINUTE, clock_verified=True
    ) == [candle(MINUTE)]
    assert available_rows(r, decision_ms=r["received_ms"], cutoff_ms=2 * MINUTE) == []


def test_unfinished_response_stays_unfinished_even_if_it_arrives_after_close():
    r = record([candle(2 * MINUTE)], requested=3 * MINUTE - 500, received=3 * MINUTE + 200)
    assert (
        available_rows(r, decision_ms=4 * MINUTE, cutoff_ms=3 * MINUTE, clock_verified=True) == []
    )


def test_forward_input_cutoff_cannot_be_in_the_future():
    with pytest.raises(ValueError, match="不能晚于"):
        available_rows(record([candle(MINUTE)]), decision_ms=3 * MINUTE, cutoff_ms=4 * MINUTE)


def test_metric_observation_requires_completed_five_minute_lag_and_matching_symbol():
    values = [
        {"timestamp": t * MINUTE, "symbol": "PUBLIC", "longShortRatio": "0.5"} for t in (5, 10)
    ]
    r = record(values, kind="global_ratio", requested=11 * MINUTE, received=11 * MINUTE + 100)
    assert (
        available_rows(r, decision_ms=12 * MINUTE, cutoff_ms=11 * MINUTE, clock_verified=True)
        == values[:1]
    )
    r["symbol"] = "OTHER"
    with pytest.raises(ValueError, match="标的"):
        available_rows(r, decision_ms=12 * MINUTE, cutoff_ms=11 * MINUTE, clock_verified=True)


def test_body_corruption_and_nonfinite_json_are_rejected():
    r = record([])
    r["body_base64"] = base64.b64encode(b"[1]").decode()
    with pytest.raises(ValueError, match="校验"):
        body(r)
    raw = b"[NaN]"
    r.update(
        body_base64=base64.b64encode(raw).decode(), body_sha256=hashlib.sha256(raw).hexdigest()
    )
    with pytest.raises(ValueError, match="非有限"):
        body(r)


def test_server_clock_outside_request_interval_is_not_accepted():
    r = record({"serverTime": 3 * MINUTE}, kind="clock")
    assert clock_valid(r)
    assert not clock_valid(record({"serverTime": 0}, kind="clock"))
    r["local_clock_consistent"] = False
    assert not clock_valid(r)


def test_dependency_set_preserves_all_frozen_symbols():
    p = {"symbols": ["PUBLIC", "币安人生USDT"], "reference_symbols": ["PUBLIC"]}
    jobs = dependencies(p, warmup=False, metrics=True)
    assert [symbol for kind, symbol, _ in jobs if kind == "klines"] == p["symbols"]
    assert len([kind for kind, _, _ in jobs if "ratio" in kind]) == 3
    assert all("apiKey" not in params and "signature" not in params for _, _, params in jobs)
    assert dependencies(p, warmup=True, metrics=False)[1][2]["limit"] == 300
    timed = dependencies(p, warmup=False, metrics=False, cutoff_ms=3 * MINUTE)
    assert all(
        params["endTime"] == 3 * MINUTE - 1
        for kind, _, params in timed
        if kind in ("klines", "index")
    )


def test_sealed_batch_chain_rejects_corruption_and_incomplete_batches(tmp_path):
    p = {"schema": "test", "orders_enabled": False}
    save(tmp_path / "protocol.json", p)
    r = record({"serverTime": 3 * MINUTE}, kind="clock")
    receipt = seal_batch(
        tmp_path, 0, p, [r, r], previous=None, started_ms=3 * MINUTE, include_metrics=True
    )
    _, previous, batches, last_metrics = verify(tmp_path)
    assert previous == digest(receipt) and len(batches) == 1
    assert last_metrics == receipt["finished_ms"]
    broken = tmp_path / "batches/000001"
    broken.mkdir()
    with pytest.raises(FileNotFoundError):
        verify(tmp_path)
    broken.rmdir()
    path = tmp_path / "batches/000000/responses.json.gz"
    path.write_bytes(gzip.compress(json.dumps([]).encode()))
    with pytest.raises(ValueError, match="身份"):
        verify(tmp_path)
