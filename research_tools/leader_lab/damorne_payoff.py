"""Market-trained net-payoff filters, independent of leader actions at runtime."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .damorne_mechanisms import market_cache
from .damorne_research import MINUTE, millis
from .damorne_tail_account import price_barriers
from .evidence import save, sha

FEATURES = [
    "ret1",
    "ret5",
    "ret15",
    "ret60",
    "ret240",
    "location60",
    "location240",
    "atr60",
    "vol60",
    "volume_log",
    "flow1",
    "flow5",
    "flow15",
    "quote_log",
    "basis",
    "funding8h",
    "btc_residual5",
    "btc_residual60",
    "breadth",
]
ECONOMICS = {
    "capital": 10000.0,
    "small_margin": 50.0,
    "large_margin": 500.0,
    "leverage": 2.0,
    "trading_fee_bps": 5.0,
    "execution_cost_bps": 3.0,
    "small_target": 0.009,
    "small_stop": 0.007,
    "large_target": 0.02,
    "large_stop": 0.01,
    "small_max_minutes": 120,
    "large_max_minutes": 240,
    "quote_ttl_minutes": 5,
    "max_positions": 6,
    "max_large_positions": 2,
    "max_margin_fraction": 0.30,
    "max_additions": 1,
    "trim_fraction": 0.5,
    "loss_cooldown_minutes": 1440,
    "profit_cooldown_minutes": 2,
    "episode_hours": 6,
    "daily_loss_cap": 0.015,
    "account_drawdown_cap": 0.08,
}


def feature_tables(market, reference_market=None):
    reference_market = reference_market or market
    btc = reference_market["BTCUSDT"]
    breadth = pd.concat(
        [(f.close.pct_change(60) > 0).astype(float) for f in reference_market.values()], axis=1
    ).mean(axis=1)
    result = {}
    for symbol, f in market.items():
        base = pd.DataFrame(index=f.index)
        for n in [1, 5, 15, 60, 240]:
            base[f"ret{n}"] = f.close.pct_change(n)
        for n in [60, 240]:
            hi, lo = f.high.rolling(n).max(), f.low.rolling(n).min()
            base[f"location{n}"] = (f.close - lo) / (hi - lo).replace(0, np.nan)
        base["atr60"], base["vol60"] = f.atr, base.ret1.rolling(60).std()
        base["volume_log"] = np.log1p(f.volume_ratio.clip(lower=0))
        base["flow1"] = f.flow
        for n in [5, 15]:
            base[f"flow{n}"] = (
                2
                * f.taker_buy_quote_volume.rolling(n).sum()
                / f.quote_volume.rolling(n).sum().replace(0, np.nan)
                - 1
            )
        base["quote_log"] = np.log1p(f.quote60.clip(lower=0))
        # Both prices refer to the same earlier open, never mismatched open/close basis.
        base["basis"] = f.mark / f.open - 1
        base["funding8h"] = f.funding.rolling(480, min_periods=1).sum()
        base["btc_residual5"] = base.ret5 - btc.close.pct_change(5)
        base["btc_residual60"] = base.ret60 - btc.close.pct_change(60)
        base["breadth"] = breadth
        sides = []
        for sign in [1, -1]:
            x = base.copy()
            for name in [
                "ret1",
                "ret5",
                "ret15",
                "ret60",
                "ret240",
                "flow1",
                "flow5",
                "flow15",
                "basis",
                "funding8h",
                "btc_residual5",
                "btc_residual60",
            ]:
                x[name] *= sign
            if sign < 0:
                x["location60"], x["location240"], x["breadth"] = (
                    1 - x.location60,
                    1 - x.location240,
                    1 - x.breadth,
                )
            # At minute t only a completed t-1 bar is available.
            sides.append(x[FEATURES].shift(1).to_numpy(dtype=np.float64))
        result[symbol] = np.stack(sides, axis=1)
    return result


def rounded_quote(price, sign, tick):
    return (math.floor(price / tick + 1e-9) if sign == 1 else math.ceil(price / tick - 1e-9)) * tick


def payoff_label(bars, decision_index, sign, target, stop, horizon, tick, cost_bps=8):
    """One-unit limit-at-prior-close outcome, with no-fill=zero and fixed maximum horizon."""
    quote = rounded_quote(float(bars[decision_index - 1, 3]), sign, tick)
    cost = cost_bps / 10000
    opened = None
    for j in range(decision_index + 1, decision_index + 6):
        if bars[j, 2] < quote - tick if sign == 1 else bars[j, 1] > quote + tick:
            opened = j
            break
    if opened is None:
        return 0.0, False
    take, protection = price_barriers(quote, sign, target, stop, tick)
    carry = 0.0
    for j in range(opened, opened + horizon + 1):
        if j > opened:
            carry -= sign * bars[j, 4] * bars[j, 5] / quote
        stopped = bars[j, 2] <= protection if sign == 1 else bars[j, 1] >= protection
        won = bars[j, 1] > take + tick if sign == 1 else bars[j, 2] < take - tick
        if stopped:
            exit_price = min(bars[j, 0], protection) if sign == 1 else max(bars[j, 0], protection)
        elif j > opened and won:
            exit_price = take
        elif j == opened + horizon:
            exit_price = bars[j, 0]
        else:
            continue
        value = sign * (exit_price / quote - 1) + carry - cost * (1 + exit_price / quote)
        return float(value), True
    raise AssertionError("Finite payoff horizon required")


def prepare_labels(market, feature_map, metadata, output):
    data, targets, clocks = [], {"small": [], "large": []}, []
    identifiers, fills = [], {"small": 0, "large": 0}
    for symbol, f in sorted(market.items()):
        bars = f[["open", "high", "low", "close", "mark", "funding"]].to_numpy(dtype=float)
        for i in range(240, len(f) - 247, 15):
            for side, sign in enumerate([1, -1]):
                vector = feature_map[symbol][i, side]
                if not np.isfinite(vector).all():
                    continue
                data.append(vector)
                clocks.append(int(f.index[i]))
                identifiers.append(f"{symbol}:{sign}")
                for sleeve in ["small", "large"]:
                    value, filled = payoff_label(
                        bars,
                        i,
                        sign,
                        ECONOMICS[f"{sleeve}_target"],
                        ECONOMICS[f"{sleeve}_stop"],
                        ECONOMICS[f"{sleeve}_max_minutes"],
                        metadata[symbol]["price_tick"],
                    )
                    targets[sleeve].append(value)
                    fills[sleeve] += int(filled)
        print({"labels_symbol": symbol, "rows": len(clocks)}, flush=True)
    np.savez_compressed(
        output / "payoff_labels.npz",
        features=np.asarray(data),
        timestamps=np.asarray(clocks),
        small=np.asarray(targets["small"]),
        large=np.asarray(targets["large"]),
    )
    save(
        output / "label_contract.json",
        {
            "features": FEATURES,
            "rows": len(clocks),
            "filled": fills,
            "sampling_minutes": 15,
            "max_label_horizon_minutes": 247,
            "no_fill_outcome": 0,
            "leader_actions_or_teacher_inventory_used": False,
            "scope": "Hypothetical future market payoff labels only, not historical leader decisions; official1m bars cannot establish queue priority",
        },
    )
    return np.asarray(data), np.asarray(clocks), {k: np.asarray(v) for k, v in targets.items()}


def fit_models(x, times, targets, output):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, mean_absolute_error

    train_end, cal_end = millis("2026-10-07T00:00:00+00:00"), millis("2026-10-10T00:00:00+00:00")
    train = times + 247 * MINUTE < train_end
    calibration = (times >= train_end) & (times + 247 * MINUTE < cal_end)
    models, diagnostics = {}, {}
    for sleeve in ["small", "large"]:
        y = targets[sleeve]
        clf = HistGradientBoostingClassifier(
            max_iter=80,
            max_leaf_nodes=7,
            min_samples_leaf=80,
            learning_rate=0.05,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=0,
        ).fit(x[train], y[train] > 0)
        reg = HistGradientBoostingRegressor(
            max_iter=80,
            max_leaf_nodes=7,
            min_samples_leaf=80,
            learning_rate=0.05,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=0,
        ).fit(x[train], y[train])
        raw = np.clip(clf.predict_proba(x[calibration])[:, 1], 1e-6, 1 - 1e-6)
        calibration_x = np.log(raw / (1 - raw)).reshape(-1, 1)
        calibrator = LogisticRegression(C=1.0, random_state=0).fit(
            calibration_x, y[calibration] > 0
        )
        calibrated = calibrator.predict_proba(calibration_x)[:, 1]
        bias = float(np.mean(reg.predict(x[calibration]) - y[calibration]))
        models[sleeve] = {
            "classifier": clf,
            "regressor": reg,
            "calibrator": calibrator,
            "mean_bias": bias,
        }
        diagnostics[sleeve] = {
            "training_rows": int(train.sum()),
            "calibration_rows": int(calibration.sum()),
            "training_positive_fraction": float(np.mean(y[train] > 0)),
            "calibration_positive_fraction": float(np.mean(y[calibration] > 0)),
            "calibration_in_sample_brier": float(brier_score_loss(y[calibration] > 0, calibrated)),
            "calibration_in_sample_net_return_mae": float(
                mean_absolute_error(y[calibration], reg.predict(x[calibration]) - bias)
            ),
            "scope": "calibrator fitted on this calibration subset; these are not independent calibration scores",
        }
    with (output / "models.pkl").open("xb") as stream:
        pickle.dump(models, stream, protocol=5)
    save(output / "model_diagnostics.json", diagnostics)
    print({"fit_completed": diagnostics}, flush=True)
    return models


def predict_models(models, vectors):
    result = np.full((len(vectors), 4), np.nan, dtype=float)
    valid = np.isfinite(vectors).all(axis=1)
    for offset, sleeve in [(0, "small"), (2, "large")]:
        m = models[sleeve]
        raw = np.clip(m["classifier"].predict_proba(vectors[valid])[:, 1], 1e-6, 1 - 1e-6)
        result[valid, offset] = m["calibrator"].predict_proba(
            np.log(raw / (1 - raw)).reshape(-1, 1)
        )[:, 1]
        result[valid, offset + 1] = m["regressor"].predict(vectors[valid]) - m["mean_bias"]
    return result


def build_signal_cache(models, features, market, output):
    signals, manifest = {}, []
    folder = output / "signals"
    folder.mkdir()
    for symbol, values in sorted(features.items()):
        shape = values.shape
        result = predict_models(models, values.reshape(-1, len(FEATURES))).reshape(shape[0], 2, 4)
        target = folder / f"{symbol}.npz"
        np.savez_compressed(target, values=result)
        signals[symbol] = result
        manifest.append({"symbol": symbol, "path": str(target), "sha256": sha(target)})
        print({"signals_symbol": symbol}, flush=True)
    save(output / "signal_sources.json", manifest)
    return signals


def metadata_with_ticks(catalog):
    from .damorne_research import metadata_for

    result = metadata_for(catalog)
    for row in catalog["symbols"]:
        result[row["symbol"]]["price_tick"] = float(
            next(f for f in row["filters"] if f["filterType"] == "PRICE_FILTER")["tickSize"]
        )
    return result


def run(source, mechanism_root, output):
    from .damorne_tail_account import evaluate_candidates

    output.mkdir(exist_ok=False)
    catalog = json.loads((source / "discovery/catalog.json").read_text())
    crypto = {r["symbol"] for r in catalog["symbols"] if "Crypto" in r.get("underlyingSubType", [])}
    market = {s: f for s, f in market_cache(source, mechanism_root).items() if s in crypto}
    source_protocol = json.loads((source / "protocol.json").read_text())
    primary = {s: market[s] for s in source_protocol["training_symbols"]}
    transfer_symbols = [
        "1000XECUSDT",
        "AGTUSDT",
        "BTCUSDT",
        "BTWUSDT",
        "ETHUSDT",
        "NEARUSDT",
        "PENGUUSDT",
        "QUSDT",
        "RAYSOLUSDT",
        "RLCUSDT",
        "STRKUSDT",
        "TIAUSDT",
        "UAIUSDT",
        "UNIUSDT",
        "ZECUSDT",
        "ZROUSDT",
    ]
    transfer = {s: market[s] for s in transfer_symbols}
    metadata = metadata_with_ticks(catalog)
    source_files = [
        Path(__file__),
        Path(__file__).with_name("damorne_tail_account.py"),
        Path(__file__).with_name("damorne_research.py"),
        Path(__file__).with_name("damorne_mechanisms.py"),
    ]
    frozen = {str(p): sha(p) for p in source_files}
    protocol = {
        "registered_utc": datetime.now(UTC).isoformat(),
        "features": FEATURES,
        "economics": ECONOMICS,
        "source_hashes": frozen,
        "market_cache_sha256": sha(mechanism_root / "market_cache/manifest.json"),
        "primary_market_symbols": sorted(primary),
        "transfer_market_symbols": sorted(transfer),
        "source_protocol_sha256": sha(source / "protocol.json"),
        "catalog_sha256": sha(source / "discovery/catalog.json"),
        "train_end": "2026-10-07T00:00:00+00:00",
        "calibration_end": "2026-10-10T00:00:00+00:00",
        "validation_end": "2026-10-14T00:00:00+00:00",
        "evaluation_end": "2026-10-24T00:00:00+00:00",
        "historical_prices_previously_inspected": True,
        "results_scope": "retrospective historical development, not fresh forward evidence",
        "candidate_count": 8,
        "validation_selection": "each sleeve net positive; >=20small and>=5large whole positions; observedDD<=8%; rank net return",
        "evaluation_gate": "each sleeve net positive; >=30small and>=15large; observedDD<=8%; +2bps execution cost stress positive each sleeve",
        "production_eligible": False,
        "orders_enabled": False,
    }
    save(output / "protocol.json", protocol)
    snapshot = output / "executed_sources"
    snapshot.mkdir()
    for p in source_files:
        (snapshot / p.name).write_bytes(p.read_bytes())
    features = feature_tables(market, primary)
    x, times, targets = prepare_labels(primary, features, metadata, output)
    models = fit_models(x, times, targets, output)
    signals = build_signal_cache(models, features, market, output)
    evaluate_candidates(primary, signals, metadata, output, protocol, transfer_market=transfer)
    if any(sha(Path(p)) != v for p, v in frozen.items()):
        raise ValueError("Implementation changed during evaluation")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("--mechanisms", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.source.absolute(), a.mechanisms.absolute(), a.output.absolute())
