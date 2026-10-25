"""Independently derive daily source values, ranks and beta hedge via NumPy."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from research_tools.factor_lab.io import read, save, sha


def audit(root, output):
    if output.exists():
        raise FileExistsError("Preserve previous independent input audit")
    for name, digest in read(root / "completion.json")["manifest"].items():
        if Path(name).name != name or sha(root / name) != digest:
            raise ValueError("Frozen target changed")
    plan = read(root / "plan.json")
    data = Path(plan["data"])
    p = plan["parameters"]
    for path, digest in plan["source_hashes"].items():
        if sha(Path(path)) != digest:
            raise ValueError("Preparation source changed")
    symbols = sorted(read(data / "prepare_plan.json")["symbols"])
    closes = {}
    quotes = {}
    for s in symbols:
        raw = pd.read_parquet(data / (s + "_5m.parquet"))
        index = pd.date_range(raw.index[0], raw.index[-1], freq="5min")
        if not raw.index.equals(index) or len(raw) % 288:
            raise ValueError("Source day incomplete")
        closes[s] = raw.close.to_numpy().reshape(-1, 288)[:, -1]
        quotes[s] = raw.quote_volume.to_numpy().reshape(-1, 288).sum(axis=1)
    daily_index = pd.date_range(index[0].floor("D"), periods=len(closes[symbols[0]]), freq="D")
    feature_table = pd.read_parquet(root / "features_daily.parquet")
    anchors = read(root / "anchors.json")
    returns = {s: closes[s][1:] / closes[s][:-1] - 1 for s in symbols}
    checked = 0
    max_error = 0.0
    for window in plan["windows"]:
        first = pd.Timestamp(window["start"], tz="UTC")
        end = pd.Timestamp(window["end"], tz="UTC")
        hour_index = pd.date_range(first, end, freq="h", inclusive="left")
        clocks = sorted(
            set([first] + list(pd.date_range(first, end, freq="W-MON", inclusive="left")))
        )
        by_candidate = {c: [] for c in plan["candidates"]}
        for t in clocks:
            k = daily_index.get_loc(t)
            rows = {}
            for s in symbols:
                r = returns[s][k - 61 : k - 1]
                m = returns["BTCUSDT"][k - 61 : k - 1]
                beta = float((r - r.mean()) @ (m - m.mean()) / ((m - m.mean()) @ (m - m.mean())))
                rows[s] = {
                    "max28": float(np.max(returns[s][k - 29 : k - 1])),
                    "momentum28": float(closes[s][k - 1] / closes[s][k - 29] - 1),
                    "beta60": beta,
                    "quote28": float(np.mean(quotes[s][k - 28 : k])),
                    "observations": k,
                }
                stored = feature_table.loc[(s, t)]
                for key in ["max28", "momentum28", "beta60"]:
                    error = abs(rows[s][key] - stored[key])
                    max_error = max(max_error, error)
                    if error > 1e-10:
                        raise ValueError("Independent daily feature differs " + s + key)
            eligible = [
                s
                for s in symbols
                if s != "BTCUSDT"
                and rows[s]["observations"] >= p["minimum_listing_observations"]
                and rows[s]["quote28"] >= p["minimum_daily_quote_volume"]
            ]
            eligible = sorted(eligible, key=lambda s: (-rows[s]["quote28"], s))[
                : p["maximum_eligible"]
            ]
            for candidate in plan["candidates"]:
                weights = dict.fromkeys(symbols, 0.0)
                if len(eligible) >= p["minimum_eligible"]:
                    key = "max28" if candidate.startswith("max28") else "momentum28"
                    ordered = sorted(eligible, key=lambda s: (-rows[s][key], s))
                    for s in ordered[:4]:
                        weights[s] = 0.125
                    for s in ordered[-4:]:
                        weights[s] = -0.125
                    if "btc_beta" in candidate:
                        weights["BTCUSDT"] = -sum(
                            weights[s] * rows[s]["beta60"] for s in symbols if s != "BTCUSDT"
                        )
                    gross = sum(abs(v) for v in weights.values())
                    if gross > 1:
                        weights = {s: v / gross for s, v in weights.items()}
                recorded = next(
                    a
                    for a in anchors
                    if a["window"] == window["id"]
                    and a["candidate"] == candidate
                    and a["clock"] == t.isoformat()
                )
                if recorded["eligible"] != len(eligible):
                    raise ValueError("Independent eligible set differs")
                by_candidate[candidate].append(pd.Series(weights, name=t))
                checked += 1
        for candidate, values in by_candidate.items():
            independent = pd.DataFrame(values).reindex(hour_index, method="ffill")
            stored = pd.read_parquet(root / f"{window['id']}_{candidate}_targets.parquet")
            error = float(
                np.max(
                    abs(
                        independent.to_numpy() - stored[["target_" + s for s in symbols]].to_numpy()
                    )
                )
            )
            max_error = max(max_error, error)
            if error > 1e-10 or not (stored.rebalance.to_numpy() == hour_index.isin(clocks)).all():
                raise ValueError("Independent hourly held book or rebalance clocks differ")
    result = {
        "schema": "max-effect-independent-input-audit.v1",
        "passed": True,
        "anchor_cases": checked,
        "symbols": len(symbols),
        "maximum_feature_or_weight_error": max_error,
        "input_completion_sha256": sha(root / "completion.json"),
        "audit_source_sha256": sha(Path(__file__)),
        "account_returns_computed": False,
    }
    save(output, result)
    print(result, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    audit(a.inputs, a.output)
