"""Checksummed positioning factors with conservative, explicit observation availability."""

from __future__ import annotations

import argparse
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .damorne_data import get
from .damorne_mechanisms import market_cache
from .damorne_payoff import (
    FEATURES,
    feature_tables,
    fit_models,
    metadata_with_ticks,
    predict_models,
)
from .damorne_research import MINUTE
from .damorne_tail_account import evaluate_candidates
from .evidence import save, sha

EXTRA_FEATURES = [
    "oi1h",
    "oi4h",
    "oi24h",
    "oi_value4h",
    "oi_turnover",
    "crowding",
    "crowding4h",
    "oi_price_pressure",
    "weighted_close24h_deviation",
    "trend24h",
    "volatility_ratio",
    "range24h",
    "metric_age_minutes",
]


def capture(root):
    out = root / "derivative_sources"
    out.mkdir(exist_ok=False)
    symbols = json.loads((root / "liquid_universe_registration.json").read_text())["symbols"]
    reuse = Path(__file__).resolve().parents[2] / "data/research/dual_alpha_native_fusion_20261015"

    def one(symbol, day):
        name = f"{symbol}-metrics-{day}.zip"
        for version in ["001", "002", "003", "004"]:
            old = reuse / f"native_positioning_{version}" / "raw" / symbol / name
            checksum = old.with_name(name + ".CHECKSUM")
            if old.exists() and checksum.exists() and sha(old) == checksum.read_text().split()[0]:
                return {
                    "symbol": symbol,
                    "day": day,
                    "path": str(old),
                    "sha256": sha(old),
                    "reused": True,
                }
        url = f"https://data.binance.vision/data/futures/um/daily/metrics/{symbol}/{name}"
        check = get(url + ".CHECKSUM", limit=1024)
        body = get(url, limit=4_000_000)
        path = out / "raw" / symbol / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        path.with_name(name + ".CHECKSUM").write_bytes(check)
        if sha(path) != check.decode().split()[0]:
            raise ValueError("Positioning archive checksum mismatch")
        return {
            "symbol": symbol,
            "day": day,
            "path": str(path),
            "sha256": sha(path),
            "url": url,
            "reused": False,
        }

    days = [
        (datetime(2026, 9, 11, tzinfo=UTC) + timedelta(days=i)).date().isoformat()
        for i in range(42)
    ]
    success, failures = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        tasks = {pool.submit(one, symbol, day): (symbol, day) for symbol in symbols for day in days}
        for task in as_completed(tasks):
            try:
                success.append(task.result())
            except Exception as e:
                failures.append({"job": tasks[task], "error": type(e).__name__ + ":" + str(e)})
            if (len(success) + len(failures)) % 120 == 0:
                print({"positioning_archives": len(success), "gaps": len(failures)}, flush=True)
    save(out / "archives.json", success)
    save(out / "gaps.json", failures)
    save(
        out / "capture_source.json",
        {
            "source_sha256": sha(Path(__file__)),
            "row_availability_delay_minutes": 5,
            "metric_max_stale_minutes": 4,
            "missing_is_not_zero": True,
        },
    )
    (out / "executed_capture_source.py").write_bytes(Path(__file__).read_bytes())
    print({"complete_archives": len(success), "gaps": len(failures)}, flush=True)


def positioning_frames(root):
    out = root / "derivative_sources"
    hourly = out / "hourly_connector_sources.json"
    if hourly.exists():
        document = json.loads(hourly.read_text())
        result = {}
        grid = np.arange(document["start_ms"], document["end_ms_exclusive"], 3600000)
        for symbol, value in document["symbols"].items():
            ratios = {int(x["timestamp"]): float(x["longShortRatio"]) for x in value["ratio"]}
            rows = [
                {
                    "clock": int(x["timestamp"]),
                    "sum_open_interest": float(x["sumOpenInterest"]),
                    "sum_open_interest_value": float(x["sumOpenInterestValue"]),
                    "count_long_short_ratio": ratios.get(int(x["timestamp"]), np.nan),
                }
                for x in value["oi"]
            ]
            f = pd.DataFrame(rows).set_index("clock")
            if f.index.duplicated().any():
                raise ValueError("Duplicate hourly positioning observation")
            f = f.reindex(grid)
            f[f <= 0] = np.nan
            result[symbol] = f
        return result
    sources = json.loads((out / "archives.json").read_text())
    frames = {}
    for row in sources:
        path = Path(row["path"])
        if sha(path) != row["sha256"]:
            raise ValueError("Positioning source changed")
        with zipfile.ZipFile(path) as z, z.open(z.namelist()[0]) as stream:
            f = pd.read_csv(stream)
        required = [
            "create_time",
            "symbol",
            "sum_open_interest",
            "sum_open_interest_value",
            "count_long_short_ratio",
        ]
        f = f[required].copy()
        if not f.symbol.eq(row["symbol"]).all():
            raise ValueError("Positioning symbol mismatch")
        f["clock"] = pd.to_datetime(f.create_time, utc=True).dt.as_unit("ms").astype("int64")
        frames.setdefault(row["symbol"], []).append(f)
    live = json.loads((out / "connector_backfills.json").read_text())
    for symbol, parts in live.items():
        oi, ratio = parts["oi"], parts["ratio"]
        by_time = {int(x["timestamp"]): x for x in ratio}
        rows = [
            {
                "clock": int(x["timestamp"]),
                "symbol": symbol,
                "sum_open_interest": float(x["sumOpenInterest"]),
                "sum_open_interest_value": float(x["sumOpenInterestValue"]),
                "count_long_short_ratio": float(by_time[int(x["timestamp"])]["longShortRatio"]),
            }
            for x in oi
            if int(x["timestamp"]) in by_time
        ]
        frames.setdefault(symbol, []).append(pd.DataFrame(rows))
    result = {}
    grid = np.arange(1789084800000, 1792800000000, 5 * MINUTE)
    for symbol, parts in frames.items():
        f = pd.concat(parts).sort_values("clock")
        if f.clock.duplicated().any():
            raise ValueError("Duplicate positioning snapshot clock")
        f = (
            f.set_index("clock")[
                ["sum_open_interest", "sum_open_interest_value", "count_long_short_ratio"]
            ]
            .apply(pd.to_numeric)
            .reindex(grid)
        )
        f[f <= 0] = np.nan
        result[symbol] = f
    return result


def extra_vectors(frame, positions, sign):
    interval = int(np.median(np.diff(positions.index)))
    hourly_steps = 3600000 // interval
    d = pd.DataFrame(index=positions.index)
    d["oi1h"] = positions.sum_open_interest.pct_change(hourly_steps, fill_method=None)
    d["oi4h"] = positions.sum_open_interest.pct_change(4 * hourly_steps, fill_method=None)
    d["oi24h"] = positions.sum_open_interest.pct_change(24 * hourly_steps, fill_method=None)
    d["oi_value4h"] = positions.sum_open_interest_value.pct_change(
        4 * hourly_steps, fill_method=None
    )
    d["crowding"] = sign * np.log(positions.count_long_short_ratio)
    d["crowding4h"] = d.crowding.diff(4 * hourly_steps)
    d["oi_value"] = positions.sum_open_interest_value
    d["available"] = positions.index + 5 * MINUTE
    merged = pd.merge_asof(
        pd.DataFrame({"clock": frame.index}),
        d.reset_index(drop=True),
        left_on="clock",
        right_on="available",
        direction="backward",
        tolerance=interval - MINUTE,
    )
    merged.index = frame.index
    merged["oi_turnover"] = np.log1p(merged.oi_value / frame.quote60.shift(1))
    merged["oi_price_pressure"] = merged.oi1h * sign * frame.close.pct_change(60).shift(1)
    weighted = (frame.close * frame.quote_volume).rolling(1440).sum() / frame.quote_volume.rolling(
        1440
    ).sum()
    merged["weighted_close24h_deviation"] = sign * (frame.close / weighted - 1).shift(1)
    merged["trend24h"] = sign * frame.close.pct_change(1440).shift(1)
    returns = frame.close.pct_change()
    merged["volatility_ratio"] = (returns.rolling(60).std() / returns.rolling(1440).std()).shift(1)
    location = (frame.close - frame.low.rolling(1440).min()) / (
        frame.high.rolling(1440).max() - frame.low.rolling(1440).min()
    ).replace(0, np.nan)
    merged["range24h"] = (location if sign == 1 else 1 - location).shift(1)
    merged["metric_age_minutes"] = (merged.clock - merged.available) / MINUTE
    return merged[EXTRA_FEATURES].to_numpy(dtype=float)


def study(root, output):
    output.mkdir(exist_ok=False)
    source = root / "combined_data"
    protocol_data = json.loads((source / "protocol.json").read_text())
    market = {
        s: f
        for s, f in market_cache(source, root / "combined_mechanisms").items()
        if s in protocol_data["training_symbols"]
    }
    metadata = metadata_with_ticks(json.loads((source / "discovery/catalog.json").read_text()))
    files = [
        Path(__file__),
        Path(__file__).with_name("damorne_tail_account.py"),
        Path(__file__).with_name("damorne_payoff.py"),
    ]
    frozen = {str(p): sha(p) for p in files}
    protocol = json.loads((root / "payoff_v002/protocol.json").read_text())
    protocol.update(
        features=FEATURES + EXTRA_FEATURES,
        source_hashes=frozen,
        registered_utc=datetime.now(UTC).isoformat(),
        positioning_availability_delay_minutes=5,
        positioning_interval_minutes=60,
        positioning_max_known_age_minutes=59,
        positioning_source_sha256=sha(root / "derivative_sources/hourly_connector_sources.json"),
        parent_label_sha256=sha(root / "payoff_v002/payoff_labels.npz"),
        parent_model_stayed_flat=True,
        derivation="same conservative payoff labels and same8candidate risk policies; add positioning and24hstate, no gate relaxation",
    )
    save(output / "protocol.json", protocol)
    snapshot = output / "executed_sources"
    snapshot.mkdir()
    for p in files:
        (snapshot / p.name).write_bytes(p.read_bytes())
    positions = positioning_frames(root)
    base = feature_tables(market)
    augmented = {
        s: np.stack(
            [
                np.concatenate([base[s][:, side], extra_vectors(f, positions[s], sign)], axis=1)
                for side, sign in enumerate([1, -1])
            ],
            axis=1,
        )
        for s, f in market.items()
    }
    x, old_x, times = [], [], []
    for symbol, f in sorted(market.items()):
        for i in range(240, len(f) - 247, 15):
            for side in [0, 1]:
                if np.isfinite(base[symbol][i, side]).all():
                    old_x.append(base[symbol][i, side])
                    x.append(augmented[symbol][i, side])
                    times.append(int(f.index[i]))
    parent = np.load(root / "payoff_v002/payoff_labels.npz", allow_pickle=False)
    if not np.array_equal(np.asarray(old_x), parent["features"]) or not np.array_equal(
        np.asarray(times), parent["timestamps"]
    ):
        raise ValueError("Augmented factor samples do not match original payoff labels")
    x, times = np.asarray(x), np.asarray(times)
    valid = np.isfinite(x).all(axis=1)
    save(
        output / "availability_quality.json",
        {
            "parent_rows": len(x),
            "available_rows": int(valid.sum()),
            "missing_rows_excluded": int((~valid).sum()),
            "zero_imputation": False,
        },
    )
    models = fit_models(
        x[valid], times[valid], {s: parent[s][valid] for s in ["small", "large"]}, output
    )
    signals = {}
    for symbol, values in sorted(augmented.items()):
        signals[symbol] = predict_models(models, values.reshape(-1, values.shape[-1])).reshape(
            len(values), 2, 4
        )
    evaluate_candidates(market, signals, metadata, output, protocol)
    if any(sha(Path(p)) != digest for p, digest in frozen.items()):
        raise ValueError("Factor sources changed during run")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["capture", "study"])
    p.add_argument("root", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if a.command == "capture":
        capture(a.root.absolute())
    else:
        study(a.root.absolute(), a.output.absolute())
