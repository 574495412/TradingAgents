"""Known-rate cross-perpetual carry cost admission; observations are not trades."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .evidence import save, sha

HOUR = 3600000
DAY = HOUR * 24


def known_window(rows, clock, field):
    end = clock - 300000
    return [r for r in rows if end - DAY < int(r[field]) <= end]


def known_binance_funding(rows, clock):
    end = clock - 300000
    history = [
        int(r["fundingTime"]) // HOUR * HOUR
        for r in rows
        if end - 3 * DAY < int(r["fundingTime"]) <= end
    ]
    if len(history) < 2:
        return None
    cadence = min(b - a for a, b in zip(history, history[1:], strict=False))
    if cadence not in [HOUR, 2 * HOUR, 4 * HOUR, 8 * HOUR]:
        return None
    observed = known_window(rows, clock, "fundingTime")
    hours = [int(r["fundingTime"]) // HOUR * HOUR for r in observed]
    if len(observed) != DAY // cadence or any(
        b - a != cadence for a, b in zip(hours, hours[1:], strict=False)
    ):
        return None
    return observed


def screen(root, parent, output):
    output.mkdir(exist_ok=False)
    source = root / "cross_venue_sources"
    receipts = json.loads((source / "capture_reconciliation_v002.json").read_text())
    protocol = {
        "registered_utc": datetime.now(UTC).isoformat(),
        "source_sha256": sha(Path(__file__)),
        "symbols": sorted(r["coin"] for r in receipts["records"]),
        "horizons_hours": [72, 168],
        "fee_bps_per_leg_per_fill": 5,
        "execution_cost_bps_per_leg_per_fill": 3,
        "four_fill_total_bps_of_one_leg_notional": 32,
        "minimum_net_cost_buffer_bps": 16,
        "availability_lag_ms": 300000,
        "known_rate_window_hours": 24,
        "binance_cadence": "infer1/2/4/8h from prior72h observations; require complete24h count and contiguous cadence",
        "max_absolute_cross_venue_price_basis": 0.01,
        "max_absolute_usdcusdt_deviation": 0.01,
        "source_receipt_sha256": sha(source / "capture_reconciliation_v002.json"),
        "fx_sha256": sha(source / "USDCUSDT_hourly.json"),
        "scope": "preliminary cost admission, not a funded account backtest or independent profit evidence",
        "prediction_assumption": "last observed24h funding-rate difference persists through72/168h; no future rates used",
        "production_eligible": False,
        "orders_enabled": False,
    }
    save(output / "protocol.json", protocol)
    (output / "executed_source.py").write_bytes(Path(__file__).read_bytes())
    funding_source = json.loads((parent / "combined_data/funding_sources.json").read_text())
    fx_raw = json.loads((source / "USDCUSDT_hourly.json").read_text())
    fx = {int(r[0]): float(r[4]) for r in fx_raw}
    records, quality = [], []
    for coin in protocol["symbols"]:
        meta = next(r for r in receipts["records"] if r["coin"] == coin)
        hl_path = source / f"{coin}_funding.json"
        candle_path = source / f"{coin}_candles.json"
        if sha(hl_path) != meta["funding_sha256"] or sha(candle_path) != meta["candles_sha256"]:
            raise ValueError("Cross-venue source digest mismatch")
        hl = json.loads(hl_path.read_text())
        candles = {int(x["t"]): x for x in json.loads(candle_path.read_text())}
        item = next(r for r in funding_source if r["symbol"] == coin + "USDT")
        bn_path = Path(item["path"])
        if sha(bn_path) != item["sha256"]:
            raise ValueError("Binance funding source digest mismatch")
        bn = json.loads(bn_path.read_text())
        # Price basis uses prior fully completed hourly closes. Binance's prior
        # funding mark is deliberately not substituted for contemporaneous price.
        cache_path = parent / "combined_mechanisms/market_cache" / f"{coin}USDT.npz"
        manifest = json.loads(
            (parent / "combined_mechanisms/market_cache/manifest.json").read_text()
        )
        cache_meta = next(r for r in manifest if r["symbol"] == coin + "USDT")
        if sha(cache_path) != cache_meta["sha256"]:
            raise ValueError("Binance market cache digest mismatch")
        with np.load(cache_path, allow_pickle=False) as data:
            minute_closes = dict(
                zip(
                    data["index"].astype(int),
                    data["values"][:, cache_meta["columns"].index("close")],
                    strict=True,
                )
            )
        missing = 0
        for clock in range(receipts["start_ms"] + DAY + HOUR, receipts["end_ms_exclusive"], HOUR):
            a, b = known_binance_funding(bn, clock), known_window(hl, clock, "time")
            if a is None or len(b) != 24:
                missing += 1
                continue
            previous = clock - HOUR
            fx_known = fx[previous]
            basis = minute_closes[clock - 60000] / (float(candles[previous]["c"]) * fx_known) - 1
            rate_diff = sum(float(x["fundingRate"]) for x in a) - sum(
                float(x["fundingRate"]) for x in b
            )
            for horizon in protocol["horizons_hours"]:
                gross_bps = abs(rate_diff) * horizon / 24 * 10000
                records.append(
                    {
                        "coin": coin,
                        "clock": clock,
                        "horizon_hours": horizon,
                        "latest_bn_observation_ms": max(int(x["fundingTime"]) for x in a),
                        "latest_hl_observation_ms": max(int(x["time"]) for x in b),
                        "known_24h_rate_difference": rate_diff,
                        "short_binance_long_hl": rate_diff > 0,
                        "expected_gross_bps": gross_bps,
                        "expected_after_four_fill_cost_bps": gross_bps - 32,
                        "known_prior_hour_price_basis": basis,
                        "fx_previous_close": fx_known,
                        "cost_admission": bool(
                            gross_bps - 32 >= 16
                            and abs(basis) <= 0.01
                            and abs(fx_known - 1) <= 0.01
                        ),
                    }
                )
        quality.append(
            {
                "coin": coin,
                "missing_or_irregular_rate_windows_excluded": missing,
                "original_settlement_timestamps_preserved": True,
            }
        )
    save(output / "known_observations.json", records)
    save(output / "quality.json", quality)
    summaries = []
    for coin in protocol["symbols"]:
        for horizon in protocol["horizons_hours"]:
            values = [x for x in records if x["coin"] == coin and x["horizon_hours"] == horizon]
            summaries.append(
                {
                    "coin": coin,
                    "horizon_hours": horizon,
                    "hourly_contexts": len(values),
                    "cost_admitted_contexts": sum(x["cost_admission"] for x in values),
                    "median_expected_gross_bps": float(
                        np.median([x["expected_gross_bps"] for x in values])
                    )
                    if values
                    else None,
                    "max_expected_gross_bps": max(x["expected_gross_bps"] for x in values)
                    if values
                    else None,
                }
            )
    save(output / "summary.json", summaries)
    if sha(Path(__file__)) != protocol["source_sha256"]:
        raise ValueError("Carry screen implementation changed")
    save(
        output / "run_status.json",
        {
            "status": "completed",
            "observations_are_not_account_positions": True,
            "production_eligible": False,
            "orders_enabled": False,
        },
    )
    print(json.dumps(summaries), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    screen(args.root.absolute(), args.parent.absolute(), args.output.absolute())
