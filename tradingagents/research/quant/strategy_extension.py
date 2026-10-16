"""固定价格机制的十五天扩样，缺少持仓档案时不冒充完整拥挤度策略。"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from tradingagents.research.quant.paths import source_path
from tradingagents.research.quant.strategy_audit import (
    FEATURES,
    net_directional_return,
    read,
    save,
    sha,
)


def directions(frame):
    """保持原生价格阈值；同时保留顺势和反向控制，不据收益更改门槛。"""
    m = (
        2 * frame["ret_20_rank_pct"]
        - 1
        + 2 * frame["ret_60_rank_pct"]
        - 1
        + np.tanh(frame["ret_20_residual_pct"] / (frame["atr_pct"] * math.sqrt(20)))
        + np.tanh(frame["ret_60_residual_pct"] / (frame["atr_pct"] * math.sqrt(60)))
    ) / 4
    f = (frame["taker_imbalance_3"] + frame["taker_imbalance_12"]) / 2
    common = frame["liquidity_rank_pct"].ge(0.5) & (
        frame["volume_ratio_rank_pct"].ge(0.9)
        | frame["ret_20_rank_pct"].ge(0.985)
        | frame["ret_20_rank_pct"].le(0.015)
        | frame["ret_60_rank_pct"].ge(0.985)
        | frame["ret_60_rank_pct"].le(0.015)
    )
    da = frame["volume_ratio_rank_pct"].ge(0.9) & frame["liquidity_rank_pct"].ge(0.5)
    da_long = (
        da
        & frame["ret_20_rank_pct"].ge(0.9)
        & frame["ret_20_residual_pct"].gt(0)
        & frame["taker_imbalance_3"].gt(0)
    )
    da_short = (
        da
        & frame["ret_20_rank_pct"].le(0.1)
        & frame["ret_20_residual_pct"].lt(0)
        & frame["taker_imbalance_3"].lt(0)
    )
    shock = (
        frame["asset_class"].eq("TRADIFI_PERPETUAL")
        & frame["ret_20_pct"].ge(1.5)
        & frame["ret_60_pct"].ge(2)
        & frame[["ret_20_rank_pct", "ret_60_rank_pct"]].max(axis=1).ge(0.985)
        & frame["atr_rank_pct"].ge(0.93)
    )
    return {
        "momentum": np.where(common & m.abs().ge(0.25), np.sign(m), 0),
        "flow": np.where(common & f.abs().ge(0.25), np.sign(f), 0),
        "damorne_public_direction": np.where(da_long, 1, np.where(da_short, -1, 0)),
        "omofun_price_shock_long": np.where(shock, 1, 0),
        "omofun_price_shock_fade": np.where(shock, -1, 0),
    }


def fresh_mask(direction, index):
    """只把真实相邻分钟的条件新出现作为新事件，跨行情缺口不生成触发。"""
    series = pd.Series(direction, index=index)
    previous = series.groupby(level="symbol").shift()
    clocks = pd.Series(index.get_level_values("bar_time"), index=index)
    prior_clock = clocks.groupby(level="symbol").shift()
    return series.ne(0) & series.ne(previous) & clocks.sub(prior_clock).eq(pd.Timedelta(minutes=1))


def run(args):
    panel, output = args.panel.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    start, end = pd.Timestamp("2026-09-29", tz="UTC"), pd.Timestamp("2026-10-14", tz="UTC")
    paths = [
        panel / f"candidate_{at.strftime('%Y%m%d')}.parquet"
        for at in pd.date_range(start - pd.Timedelta(days=1), end, freq="D")
    ]
    manifest = read(panel / "manifest.json")
    if (
        manifest.get("leader_labels_read") is not False
        or manifest.get("orders_enabled") is not False
    ):
        raise ValueError("扩样只能使用纯公开行情")
    freeze = {
        "schema": "cz-price-factor-extension.v1",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "source_files": {str(p): sha(p) for p in paths},
        "manifest_sha256": sha(panel / "manifest.json"),
        "source_code_sha256": sha(__file__),
        "shared_audit_source_sha256": sha(source_path("strategy_factor_audit.py")),
        "horizons_minutes": [20, 60],
        "entry_after_feature_bar_open_minutes": 2,
        "samples": ["five_minute_condition", "fresh_minute_edge"],
        "fee_bps_per_side": 5,
        "base_slippage_bps_per_side": 3,
        "stress_slippage_bps_per_side": 10,
        "crowding_input_used": False,
        "not_the_complete_omofun_trigger": True,
        "funding_included": False,
        "independent_oos": False,
        "orders_enabled": False,
        "boundaries_selected_before_extension_outcomes": True,
    }
    save(output / "protocol.json", freeze)
    statistics, events = [], []
    for day in pd.date_range(start, end - pd.Timedelta(days=1), freq="D"):
        path = panel / f"candidate_{day.strftime('%Y%m%d')}.parquet"
        frame = pd.read_parquet(
            path, columns=["open", "eligible", "observed", "asset_class", *FEATURES]
        ).sort_index()
        previous_path = (
            panel / f"candidate_{(day - pd.Timedelta(days=1)).strftime('%Y%m%d')}.parquet"
        )
        previous = pd.read_parquet(
            previous_path,
            columns=list(frame.columns),
            filters=[("bar_time", ">=", day - pd.Timedelta(minutes=1))],
        )
        next_path = panel / f"candidate_{(day + pd.Timedelta(days=1)).strftime('%Y%m%d')}.parquet"
        following = pd.read_parquet(
            next_path,
            columns=["open", "eligible", "observed"],
            filters=[("bar_time", "<=", day + pd.Timedelta(days=1, minutes=62))],
        )
        features = pd.concat([previous, frame]).sort_index()
        if not features.index.is_unique:
            raise ValueError("扩样特征时钟重复")
        valid_features = (
            features["eligible"]
            & features["observed"]
            & np.isfinite(features.loc[:, FEATURES]).all(axis=1)
            & features["atr_pct"].gt(0)
        )
        signals = directions(features)
        for name in signals:
            signals[name] = np.where(valid_features, signals[name], 0)
        openings = pd.concat([frame[["open", "eligible", "observed"]], following]).sort_index()
        for name, direction in signals.items():
            clock = features.index.get_level_values("bar_time")
            current = (clock >= day) & (clock < day + pd.Timedelta(days=1))
            samples = {
                "five_minute_condition": current & (clock.minute % 5 == 0) & (direction != 0),
                "fresh_minute_edge": current & fresh_mask(direction, features.index).to_numpy(),
            }
            for sample, mask in samples.items():
                chosen = features.loc[mask].copy()
                if chosen.empty:
                    continue
                chosen["direction"] = np.asarray(direction)[mask]
                times, symbols = (
                    chosen.index.get_level_values("bar_time"),
                    chosen.index.get_level_values("symbol"),
                )
                entry_times = times + pd.Timedelta(minutes=2)
                for h in (20, 60):
                    exit_times = entry_times + pd.Timedelta(minutes=h)
                    values, keep = [], np.asarray(exit_times <= end).copy()
                    for offset in range(h + 1):
                        rows = openings.reindex(
                            pd.MultiIndex.from_arrays(
                                [entry_times + pd.Timedelta(minutes=offset), symbols],
                                names=openings.index.names,
                            )
                        )
                        keep &= (
                            rows["eligible"].eq(True).to_numpy()
                            & rows["observed"].eq(True).to_numpy()
                            & rows["open"].gt(0).to_numpy()
                            & np.isfinite(rows["open"].to_numpy())
                        )
                        if offset in (0, h):
                            values.append(rows["open"].to_numpy())
                    valid = chosen.iloc[np.flatnonzero(keep)].copy()
                    if valid.empty:
                        continue
                    d = valid["direction"].to_numpy()
                    valid["base"] = net_directional_return(values[0][keep], values[1][keep], d)
                    valid["stress"] = net_directional_return(
                        values[0][keep], values[1][keep], d, slippage_bps=10
                    )
                    valid["clock"] = valid.index.get_level_values("bar_time")
                    for lane, selected in valid.groupby("asset_class"):
                        statistics.append(
                            {
                                "factor": name,
                                "sample": sample,
                                "day": day.date().isoformat(),
                                "horizon_minutes": h,
                                "asset_class": lane,
                                "events": len(selected),
                                "symbols": selected.index.get_level_values("symbol").nunique(),
                                "clocks": selected["clock"].nunique(),
                                "mean_net_bps_ex_funding": float(selected["base"].mean() * 10000),
                                "mean_stress_net_bps_ex_funding": float(
                                    selected["stress"].mean() * 10000
                                ),
                                "positive_fraction": float(selected["base"].gt(0).mean()),
                            }
                        )
                    if (
                        name == "omofun_price_shock_fade"
                        and sample == "fresh_minute_edge"
                        and h == 60
                    ):
                        events.extend(
                            {
                                "symbol": symbol,
                                "feature_bar_time": at.isoformat(),
                                "base_net_bps_ex_funding": float(row["base"] * 10000),
                                "stress_net_bps_ex_funding": float(row["stress"] * 10000),
                            }
                            for (at, symbol), row in valid.iterrows()
                        )
        print(f"价格机制扩样完成：{day.date()}", flush=True)
    if any(sha(path) != checksum for path, checksum in freeze["source_files"].items()):
        raise ValueError("扩样期间公开输入发生变化")
    save(output / "statistics.json", statistics)
    save(output / "shock_fade_events.json", events)
    aggregate = []
    for name in signals:
        for sample in ("five_minute_condition", "fresh_minute_edge"):
            for lane in ("CRYPTO_PERPETUAL", "TRADIFI_PERPETUAL"):
                for phase, allowed_days in {
                    "before_two_day_window": lambda d: d < "2026-10-06",
                    "two_day_window": lambda d: "2026-10-06" <= d < "2026-10-08",
                    "after_two_day_window": lambda d: d >= "2026-10-08",
                    "full_fifteen_days": lambda d: True,
                }.items():
                    rows = [
                        v
                        for v in statistics
                        if v["factor"] == name
                        and v["sample"] == sample
                        and v["asset_class"] == lane
                        and v["horizon_minutes"] == 60
                        and allowed_days(v["day"])
                    ]
                    n = sum(v["events"] for v in rows)
                    if n:
                        aggregate.append(
                            {
                                "factor": name,
                                "sample": sample,
                                "asset_class": lane,
                                "phase": phase,
                                "horizon_minutes": 60,
                                "days": len(rows),
                                "events": n,
                                "positive_stress_days": sum(
                                    v["mean_stress_net_bps_ex_funding"] > 0 for v in rows
                                ),
                                "mean_net_bps_ex_funding": sum(
                                    v["events"] * v["mean_net_bps_ex_funding"] for v in rows
                                )
                                / n,
                                "mean_stress_net_bps_ex_funding": sum(
                                    v["events"] * v["mean_stress_net_bps_ex_funding"] for v in rows
                                )
                                / n,
                            }
                        )
    save(output / "aggregate.json", aggregate)
    save(output / "receipts.json", {p.name: sha(p) for p in output.iterdir() if p.is_file()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
