"""Collect original BTC/USD candles and audit fixed stock-account exposures.

Run directly. Does not import TradingAgents, load dotenv, or place orders.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[3]
PLAN = Path(__file__).with_name("crypto_exposure_plan.json")
DATA = ROOT / "data/research/equity_crypto_exposure_20261027_v001"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def capture(url, path):
    import requests

    receipt_path = path.with_suffix(path.suffix + ".receipt.json")
    if path.exists() and receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["url"] != url or receipt["sha256"] != digest(path):
            raise ValueError("Previously captured source changed")
        return receipt
    if path.exists() or receipt_path.exists():
        raise ValueError("Partial capture needs inspection")
    events = []
    for attempt in range(3):
        time.sleep(0.4)
        try:
            response = requests.get(
                url, timeout=(10, 25), headers={"User-Agent": "EquityResearch/1.0"}
            )
            events.append({"attempt": attempt + 1, "status": response.status_code})
            if response.status_code == 429:
                delay = float(response.headers.get("Retry-After", "5"))
                if delay > 60:
                    raise ValueError(
                        "Provider asks for a longer wait; stop rather than retry early"
                    )
                time.sleep(max(0, delay))
            response.raise_for_status()
            path.write_bytes(response.content)
            receipt = {
                "url": url,
                "sha256": digest(path),
                "bytes": len(response.content),
                "captured_at_utc": datetime.now(UTC).isoformat(),
                "events": events,
            }
            write(receipt_path, receipt)
            return receipt
        except requests.RequestException as exc:
            events.append({"error": type(exc).__name__})
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if attempt == 2 or status in {400, 401, 403, 404}:
                write(path.with_suffix(".failure.json"), {"url": url, "events": events})
                raise
            time.sleep(1)
    raise AssertionError("Unreachable capture state")


def valid_candle(row):
    if not isinstance(row, list) or len(row) != 6:
        raise ValueError("Coinbase candle does not have six fields")
    if any(
        isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in row
    ):
        raise ValueError("Invalid numeric candle")
    stamp, low, high, opened, close, volume = row
    if (
        stamp != int(stamp)
        or stamp % 3600
        or not 0 < low <= min(opened, close) <= max(opened, close) <= high
        or volume < 0
    ):
        raise ValueError("Invalid candle time/OHLCV")
    return int(stamp)


def closing_bucket(hourly, cutoff):
    """Require the exact completed hourly bucket; no stale/future substitute."""
    if cutoff.tzinfo is None or cutoff.minute or cutoff.second or cutoff.microsecond:
        raise ValueError("Ambiguous or non-hour close cannot use this bucket method")
    start = cutoff - timedelta(hours=1)
    if int(start.timestamp()) not in hourly:
        raise ValueError("Required closing bucket missing; no fill or sample removal")
    entry = hourly[int(start.timestamp())]
    if valid_candle(entry["candle"]) != int(start.timestamp()):
        raise ValueError("Hourly key and original candle clock differ")
    return entry


def collect():
    DATA.mkdir(parents=True, exist_ok=True)
    raw = DATA / "coinbase"
    raw.mkdir(exist_ok=True)
    marker = DATA / "collection-plan.json"
    declaration = {
        "plan_sha256": digest(PLAN),
        "source_sha256": digest(Path(__file__)),
        "role": "Source collection before risk regression; no new economic account",
    }
    if marker.exists():
        if json.loads(marker.read_text())["plan_sha256"] != declaration["plan_sha256"]:
            raise ValueError("Collection plan was changed")
    else:
        write(marker, declaration)
    start, end = datetime(2022, 12, 30, tzinfo=UTC), datetime(2024, 6, 27, tzinfo=UTC)
    intervals = []
    current = start
    while current < end:
        stop = min(current + timedelta(hours=299), end)
        intervals.append((current, stop))
        current = stop

    def fetch(bounds):
        first, last = bounds
        params = {"granularity": 3600, "start": first.isoformat(), "end": last.isoformat()}
        url = "https://api.exchange.coinbase.com/products/BTC-USD/candles?" + urlencode(params)
        path = raw / (first.strftime("%Y%m%dT%H%M%S") + ".json")
        receipt = capture(url, path)
        rows = json.loads(path.read_text())
        if not isinstance(rows, list):
            raise ValueError("Source returned an error object instead of candles")
        stamps = [valid_candle(row) for row in rows]
        if len(stamps) != len(set(stamps)):
            raise ValueError("Duplicate candle inside one response")
        return path, receipt, rows

    rows, lineage, receipts = {}, {}, []
    with ThreadPoolExecutor(max_workers=2) as executor:
        for path, receipt, batch in executor.map(fetch, intervals):
            relative = str(path.relative_to(ROOT))
            receipts.append({"file": relative, **receipt})
            for i, candle in enumerate(batch):
                stamp = valid_candle(candle)
                if stamp in rows and rows[stamp] != candle:
                    raise ValueError("Conflicting boundary candles; do not select one")
                rows[stamp] = candle
                lineage.setdefault(stamp, []).append(
                    {"file": relative, "row": i, "sha256": receipt["sha256"]}
                )
            print(
                json.dumps({"completed_batches": len(receipts), "total_batches": len(intervals)}),
                flush=True,
            )
    expected = set(range(int(start.timestamp()), int(end.timestamp()), 3600))
    missing = sorted(expected - rows.keys())
    write(
        DATA / "coinbase-source-manifest.json",
        {
            "schema": "equity-crypto-source-manifest.v1",
            "plan_sha256": digest(PLAN),
            "receipts": receipts,
            "requested_hour_count": len(expected),
            "observed_requested_hour_count": len(expected & rows.keys()),
            "missing_hours_utc": [datetime.fromtimestamp(x, UTC).isoformat() for x in missing],
            "returned_outside_global_window": len(rows.keys() - expected),
            "complete": not missing,
        },
    )
    write(
        DATA / "hourly-candles.json",
        [{"candle": rows[x], "lineage": lineage[x]} for x in sorted(expected & rows.keys())],
    )
    if missing:
        raise ValueError(
            f"Missing {len(missing)} requested hourly buckets; no imputation or regression"
        )


def load_regression():
    path = Path(__file__).with_name("mechanism_attribution.py")
    spec = importlib.util.spec_from_file_location("fixed_mechanism", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.regression, path


def analyze(output):
    import numpy as np
    import pandas as pd

    plan = json.loads(PLAN.read_text())
    manifest_path = DATA / "coinbase-source-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["plan_sha256"] != digest(PLAN):
        raise ValueError("Source collection plan mismatch")
    amendment_path = Path(__file__).with_name("crypto_exposure_source_amendment.json")
    amendment = json.loads(amendment_path.read_text())
    if amendment["original_plan_sha256"] != digest(PLAN):
        raise ValueError("Source amendment belongs to a different plan")
    if manifest["missing_hours_utc"] != amendment["unfilled_hours_utc"]:
        raise ValueError("Unreviewed source gap; no endpoint analysis")
    incident_path = ROOT / amendment["incident_record"]
    if digest(incident_path) != amendment["incident_sha256"]:
        raise ValueError("Original maintenance incident record changed")
    raw_batches = {}
    for receipt in manifest["receipts"]:
        if digest(ROOT / receipt["file"]) != receipt["sha256"]:
            raise ValueError("Captured source changed")
        raw_batches[receipt["file"]] = json.loads((ROOT / receipt["file"]).read_text())
    output.mkdir(parents=True, exist_ok=False)
    prior = ROOT / plan["source_attribution_run"]
    paired_path = prior / "paired-daily-returns.csv"
    daily = pd.read_csv(paired_path, index_col="date")
    calendar_path = ROOT / plan["session_calendar"]
    calendar = pd.read_csv(calendar_path, index_col="date")
    days = [plan["first_anchor_day"], *daily.index]
    calendar = calendar.loc[days]
    hourly_path = DATA / "hourly-candles.json"
    entries = json.loads(hourly_path.read_text())
    for entry in entries:
        if not entry["lineage"]:
            raise ValueError("Normalized candle lacks original lineage")
        for reference in entry["lineage"]:
            if (
                digest(ROOT / reference["file"]) != reference["sha256"]
                or raw_batches[reference["file"]][reference["row"]] != entry["candle"]
            ):
                raise ValueError("Normalized candle differs from original response")
    hourly = {valid_candle(entry["candle"]): entry for entry in entries}
    if len(hourly) != len(entries):
        raise ValueError("Normalized hourly index has duplicates")
    marks = []
    for day, session in calendar.iterrows():
        end = pd.Timestamp(session["close"])
        start = end - pd.Timedelta(hours=1)
        entry = closing_bucket(hourly, end)
        marks.append(
            {
                "date": day,
                "stock_close_utc": end.isoformat(),
                "bucket_start_utc": start.isoformat(),
                "bucket_end_exclusive_utc": end.isoformat(),
                "stock_close_new_york": end.tz_convert("America/New_York").isoformat(),
                "BTCUSD_close": entry["candle"][4],
                "raw_lineage": entry["lineage"],
            }
        )
    btc_marks = pd.DataFrame(marks).set_index("date")
    btc_returns = btc_marks["BTCUSD_close"].pct_change(fill_method=None).loc[daily.index]
    factors = daily[["SPY_gross_total_return", "QQQ_minus_SPY_gross_return"]].copy()
    factors["BTCUSD_close_to_close_return"] = btc_returns
    coin_path = ROOT / "data/research/equity_pit_20261026_v001/prices/COIN.US.csv"
    coin = (
        pd.read_csv(coin_path, index_col="date")
        .loc[days, "close"]
        .pct_change(fill_method=None)
        .loc[daily.index]
    )
    focal, control = plan["focal_account"], plan["fixed_equity_control"]
    outcomes = {
        "focal_net_account": daily[focal],
        "focal_minus_fixed_equity_control": daily[focal] - daily[control],
        "COIN_raw_price": coin,
    }
    if (
        len(daily) != plan["expected_input_periods"]
        or factors.isna().any().any()
        or any(y.isna().any() for y in outcomes.values())
    ):
        raise ValueError("Unaligned or missing returns; no filled or dropped periods")
    regression, helper_path = load_regression()
    model = {}
    z = np.column_stack([np.ones(len(factors)), factors.iloc[:, :2].to_numpy()])
    btc = btc_returns.to_numpy()
    residual_btc = btc - z @ np.linalg.lstsq(z, btc, rcond=None)[0]
    for name, y in outcomes.items():
        residual_y = y.to_numpy() - z @ np.linalg.lstsq(z, y.to_numpy(), rcond=None)[0]
        model[name] = {
            "two_factor": regression(y, factors.iloc[:, :2]),
            "three_factor": regression(y, factors),
            "raw_BTC_correlation": float(y.corr(btc_returns)),
            "conditional_BTC_correlation": float(np.corrcoef(residual_y, residual_btc)[0, 1]),
        }
    probes_path = DATA / "binance-fixed-probes.json"
    probes = json.loads(probes_path.read_text())
    comparisons = []
    for probe in probes["records"]:
        result = probe["result"]
        raw = result["structuredContent"]["result"]
        if result.get("isError") or len(raw) != 1 or probe["args"]["symbol"] != "BTCUSDT":
            raise ValueError("Invalid fixed Binance probe")
        candle = raw[0]
        day = probe["day"]
        mark = btc_marks.loc[day]
        if (
            candle[0] != int(pd.Timestamp(mark["bucket_start_utc"]).timestamp() * 1000)
            or candle[6] != int(pd.Timestamp(mark["stock_close_utc"]).timestamp() * 1000) - 1
        ):
            raise ValueError("Cross-market probe uses a different clock")
        value = float(candle[4])
        comparisons.append(
            {
                "date": day,
                "coinbase_BTCUSD": mark["BTCUSD_close"],
                "binance_BTCUSDT": value,
                "relative_quoted_price_difference": value / mark["BTCUSD_close"] - 1,
                "USD_USDT_conversion_verified": False,
            }
        )
    write(output / "clock-marks.json", marks)
    write(output / "cross-market-probes.json", comparisons)
    write(
        output / "diagnostics.json",
        {
            "schema": "equity-crypto-exposure-diagnostics.v1",
            "window": [daily.index[0], daily.index[-1]],
            "models": model,
            "observations": len(daily),
            "clock_marks": len(marks),
            "full_hourly_coverage_complete": manifest["complete"],
            "unfilled_hourly_buckets": manifest["missing_hours_utc"],
            "source_amendment": amendment,
            "shortened_sessions": [
                m for m in marks if pd.Timestamp(m["stock_close_new_york"]).hour != 16
            ],
            "BTC_stock_close_return_total": float(np.prod(1 + btc_returns) - 1),
            "new_economic_accounts": 0,
            "causal_alpha_identified": False,
            "selection_corrected": False,
            "fit_or_execute_allocations": False,
            "external_independent_attestation": False,
            "profit_goal_met": False,
            "limitations": [
                plan["regressions"]["baseline"],
                plan["primary_crypto_source"]["limits"],
                "Binance probes use USDT and another exchange; price differences are not failures or a verified FX conversion.",
                "No existing crypto strategy PnL is aligned or admitted by this spot risk-factor analysis; no common-capital combined account was run.",
                "Missing EXAS history and merger cash evidence remain unresolved. No original minimum alpha effect or power plan was supplied by this audit.",
            ],
        },
    )
    pd.DataFrame({**outcomes, **{key: factors[key] for key in factors}}).to_csv(
        output / "aligned-returns.csv", index_label="date"
    )
    inputs = [
        PLAN,
        amendment_path,
        incident_path,
        paired_path,
        calendar_path,
        coin_path,
        hourly_path,
        manifest_path,
        probes_path,
        prior / "diagnostics.json",
    ]
    write(
        output / "registration.json",
        {
            "schema": "equity-crypto-exposure-registration.v1",
            "source_sha256": digest(Path(__file__)),
            "helper_source_sha256": digest(helper_path),
            "inputs_sha256": {str(p.relative_to(ROOT)): digest(p) for p in inputs},
            "versions": {p: version(p) for p in ["numpy", "pandas", "statsmodels", "scipy"]},
            "new_economic_accounts": 0,
        },
    )
    (output / "analysis_source.py").write_bytes(Path(__file__).read_bytes())
    (output / "plan.json").write_bytes(PLAN.read_bytes())
    (output / "source-amendment.json").write_bytes(amendment_path.read_bytes())
    print(
        json.dumps(
            {
                "observations": len(daily),
                "clock_marks": len(marks),
                "models": {
                    k: {
                        "raw_BTC_correlation": v["raw_BTC_correlation"],
                        "conditional_BTC_correlation": v["conditional_BTC_correlation"],
                        "BTC": v["three_factor"]["coefficients"]["BTCUSD_close_to_close_return"],
                        "intercept": v["three_factor"]["coefficients"]["intercept_daily_return"],
                    }
                    for k, v in model.items()
                },
            },
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["collect", "analyze"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.stage == "collect":
        collect()
    elif args.output is None:
        parser.error("analyze requires --output")
    else:
        analyze(args.output.absolute())
