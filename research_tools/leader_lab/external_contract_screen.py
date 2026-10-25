"""Frozen, bounded perpetual adaptations of inspectable public strategy rules.

Prepare indicators in the existing Freqtrade environment; run shared-wallet
accounts in the isolated research environment. No downloaded code is executed.
"""

from __future__ import annotations

import argparse
import inspect
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .damorne_mechanisms import market_cache
from .damorne_payoff import ECONOMICS, metadata_with_ticks
from .damorne_research import MINUTE, millis
from .damorne_tail_account import candidates, run_account
from .evidence import save, sha

SMALL = ["hb-bb3-range", "hb-macdbb3", "pb-ema-recovery", "hb-st3"]
LARGE = ["hb-st15", "ft-adxsma1h", "ft-triple-st1h"]


def policies():
    base = candidates(ECONOMICS)[0]
    return [
        {
            **base,
            "id": f"{small}__{large}",
            "small_family": small,
            "large_family": large,
            "large_target": 0.025,
            "large_stop": 0.012,
            "large_max_minutes": 720,
            "funding_adjusted_stop": True,
            "signal_scope": "BOOLEAN_PUBLIC_RULE_ADAPTATION_NOT_PROBABILITY",
            "allow_additions": False,
            "allow_reductions": False,
            "max_additions": 0,
            "orders_enabled": False,
        }
        for small in SMALL
        for large in LARGE
    ]


def completed_bars(frame, minutes):
    f = frame.copy()
    f.index = pd.to_datetime(f.index, unit="ms", utc=True)
    grouped = f.resample(f"{minutes}min", origin="epoch", label="left", closed="left")
    bars = grouped.agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    bars = bars[grouped.close.count() == minutes]
    return bars


def at_decision_clock(values, index, minutes):
    # Aggregate labelled by its opening time becomes observable only at its end.
    result = values.copy()
    result.index = result.index + pd.Timedelta(minutes=minutes)
    return result.reindex(pd.to_datetime(index, unit="ms", utc=True), method="ffill")


def rule_features(frame):
    import pandas_ta as ta
    import talib
    from technical.indicators import supertrend

    tables = {n: completed_bars(frame, n) for n in [3, 15, 60]}
    result = pd.DataFrame(index=frame.index)
    for n in [3, 15]:
        f = tables[n]
        st = ta.supertrend(f.high, f.low, f.close, length=20, multiplier=4.0)
        values = pd.DataFrame(index=f.index)
        values["direction"] = st["SUPERTd_20_4.0"]
        values["distance"] = (f.close - st["SUPERT_20_4.0"]).abs() / f.close
        gate = pd.DataFrame(
            {
                "long": (values.direction == 1) & (values.distance < 0.01),
                "short": (values.direction == -1) & (values.distance < 0.01),
            }
        )
        mapped = at_decision_clock(gate, frame.index, n).fillna(False)
        result[f"hb-st{n}"] = (
            mapped.long.astype(int).to_numpy() - mapped.short.astype(int).to_numpy()
        )
    f = tables[3]
    bb = ta.bbands(f.close, length=100, lower_std=2.0, upper_std=2.0)
    bbp = bb[next(c for c in bb if c.startswith("BBP_"))]
    macd = ta.macd(f.close, fast=21, slow=42, signal=9)
    macdline, histogram = macd["MACD_21_42_9"], macd["MACDh_21_42_9"]
    signals = pd.DataFrame(
        {
            "bb": np.select([bbp < 0, bbp > 1], [1, -1], default=0),
            "macdbb": np.select(
                [
                    (bbp < 0) & (histogram > 0) & (macdline < 0),
                    (bbp > 1) & (histogram < 0) & (macdline > 0),
                ],
                [1, -1],
                default=0,
            ),
        },
        index=f.index,
    )
    mapped = at_decision_clock(signals, frame.index, 3).fillna(0)
    h = tables[60]
    adx = pd.Series(talib.ADX(h.high, h.low, h.close, timeperiod=14), index=h.index)
    hourly_adx = at_decision_clock(adx, frame.index, 60).to_numpy()
    result["hb-bb3-range"] = np.where(hourly_adx <= 25, mapped.bb.to_numpy(), 0)
    result["hb-macdbb3"] = mapped.macdbb.to_numpy()
    # EMA envelope context is borrowed; reversal confirmation and hard stops are
    # our bounded adaptation, without Passivbot's adverse-price martingale.
    close = frame.close
    ema = pd.concat(
        [close.ewm(span=n, adjust=False, min_periods=n).mean() for n in [60, 120, 240]], axis=1
    )
    long = (close < ema.min(axis=1) * 0.997) & (close > close.shift(1))
    short = (close > ema.max(axis=1) * 1.003) & (close < close.shift(1))
    recovery = pd.Series(np.select([long, short], [1, -1], default=0), index=frame.index)
    result["pb-ema-recovery"] = np.where(hourly_adx <= 25, recovery.shift(1).fillna(0), 0)
    fast, slow = talib.SMA(h.close, timeperiod=12), talib.SMA(h.close, timeperiod=48)
    crossing = pd.Series(
        np.select(
            [
                (fast > slow) & (fast.shift(1) <= slow.shift(1)) & (adx > 30),
                (fast < slow) & (fast.shift(1) >= slow.shift(1)) & (adx > 30),
            ],
            [1, -1],
            default=0,
        ),
        index=h.index,
    )
    result["ft-adxsma1h"] = at_decision_clock(crossing, frame.index, 60).fillna(0).to_numpy()
    buys = [supertrend(h, period=p, multiplier=m)[1] for m, p in [(4, 8), (7, 9), (1, 8)]]
    sells = [supertrend(h, period=p, multiplier=m)[1] for m, p in [(1, 16), (3, 18), (6, 18)]]
    buy = np.logical_and.reduce([x == "up" for x in buys])
    sell = np.logical_and.reduce([x == "down" for x in sells])
    # Conflicting long/short inputs stay flat rather than relying on source row order.
    triple = pd.Series(np.select([buy & ~sell, sell & ~buy], [1, -1], default=0), index=h.index)
    result["ft-triple-st1h"] = at_decision_clock(triple, frame.index, 60).fillna(0).to_numpy()
    return result.astype(np.int8)


def prepare(parent, root, output):
    import pandas_ta as ta
    import talib
    from technical.indicators import supertrend

    output.mkdir(exist_ok=False)
    names = [
        "external_contract_screen.py",
        "damorne_tail_account.py",
        "damorne_research.py",
        "damorne_mechanisms.py",
        "damorne_payoff.py",
    ]
    files = [Path(__file__).with_name(n) for n in names]
    frozen = {str(p): sha(p) for p in files}
    protocol = {
        "registered_utc": datetime.now(UTC).isoformat(),
        "candidates": policies(),
        "source_hashes": frozen,
        "parent": str(parent),
        "public_source_manifest_sha256": sha(root / "source_manifest.json"),
        "universe_registration_sha256": sha(parent / "liquid_universe_registration.json"),
        "indicator_versions": {"pandas_ta": ta.__version__, "talib": talib.__version__},
        "indicator_sources": {
            str(Path(inspect.getfile(f))): sha(Path(inspect.getfile(f)))
            for f in [ta.bbands, ta.macd, ta.supertrend, supertrend]
        },
        "validation_start": "2026-10-07T00:00:00+00:00",
        "evaluation_start": "2026-10-14T00:00:00+00:00",
        "end": "2026-10-24T00:00:00+00:00",
        "validation_gate": "each sleeve net and closed PNL positive, >=20small and>=5large; PF>=1.1; observedDD<=8%",
        "evaluation_gate": "each sleeve net and closed PNL positive, >=30small and>=15large; PF>=1.1; observedDD<=8%;+2bps each side profitable each sleeve",
        "selection_rule": "rank validation net return only; fixed validation best diagnostic if none qualify",
        "scope": "retrospective bounded adaptations; not native public-bot performance or genuinely unseen forward",
        "deviations": [
            "common protective exits and shared wallet replace original controllers' executors",
            "BB3 adds closed-hour ADX<=25; EMA60/120/240 and0.3%offset with recovery are explicit assumptions",
            "no adverse-price DCA; no additions or partial reductions in this entry-factor screen",
            "Supertrend3m/15m20x4; ADX/SMA1h14/12/48; triple-ST1h uses source buy/sell dictionaries",
            "local pandas_ta0.3.16 and technical indicator kernels; exact upstream runtime parity unverified",
        ],
        "production_eligible": False,
        "orders_enabled": False,
    }
    save(output / "protocol.json", protocol)
    saved = output / "executed_sources"
    saved.mkdir()
    for p in files:
        (saved / p.name).write_bytes(p.read_bytes())
    data = parent / "combined_data"
    symbols = json.loads((data / "protocol.json").read_text())["training_symbols"]
    market = {
        s: f for s, f in market_cache(data, parent / "combined_mechanisms").items() if s in symbols
    }
    cache = output / "factors"
    cache.mkdir()
    records = []
    for symbol, f in sorted(market.items()):
        values = rule_features(f)
        path = cache / f"{symbol}.npz"
        np.savez_compressed(path, timestamps=f.index.to_numpy(), values=values.to_numpy())
        records.append(
            {
                "symbol": symbol,
                "columns": values.columns.tolist(),
                "sha256": sha(path),
                "nonzero_observations": {c: int((values[c] != 0).sum()) for c in values},
            }
        )
    save(output / "factor_manifest.json", records)
    if any(sha(Path(p)) != digest for p, digest in frozen.items()):
        raise ValueError("Strategy source changed during preparation")
    print({"prepared_symbols": len(records), "registered_candidates": len(policies())}, flush=True)


def load_factors(output, market):
    result = {}
    for row in json.loads((output / "factor_manifest.json").read_text()):
        path = output / "factors" / f"{row['symbol']}.npz"
        if sha(path) != row["sha256"]:
            raise ValueError("Factor cache digest changed")
        with np.load(path, allow_pickle=False) as data:
            if not np.array_equal(data["timestamps"], market[row["symbol"]].index):
                raise ValueError("Factor/market clocks differ")
            result[row["symbol"]] = {c: data["values"][:, i] for i, c in enumerate(row["columns"])}
    return result


def signals_for(policy, factors):
    result = {}
    for symbol, values in factors.items():
        small, large = values[policy["small_family"]], values[policy["large_family"]]
        signals = np.zeros((len(small), 2, 4))
        for side, sign in enumerate([1, -1]):
            signals[:, side, 0] = (small == sign).astype(float)
            signals[:, side, 1] = np.where(small == sign, 0.002, 0)
            signals[:, side, 2] = (large == sign).astype(float)
            signals[:, side, 3] = np.where(large == sign, 0.004, 0)
        result[symbol] = signals
    return result


def economic_gate(score, small_count, large_count):
    return bool(
        all(
            p["net_pnl"] > 0 and p["net_pnl"] - p["terminal_open_net_pnl"] > 0
            for p in score["sleeves"].values()
        )
        and score["sleeves"]["small"]["complete_positions"] >= small_count
        and score["sleeves"]["large"]["complete_positions"] >= large_count
        and score["profit_factor"] is not None
        and score["profit_factor"] >= 1.1
        and score["minute_observed_mark_open_drawdown"] <= 0.08
    )


def study(output):
    protocol = json.loads((output / "protocol.json").read_text())
    if (output / "validation_scores.json").exists():
        raise ValueError("Refuse to overwrite account experiment")
    if any(sha(Path(p)) != digest for p, digest in protocol["source_hashes"].items()):
        raise ValueError("Registered implementation changed")
    parent = Path(protocol["parent"])
    data = parent / "combined_data"
    symbols = json.loads((data / "protocol.json").read_text())["training_symbols"]
    market = {
        s: f for s, f in market_cache(data, parent / "combined_mechanisms").items() if s in symbols
    }
    meta = metadata_with_ticks(json.loads((data / "discovery/catalog.json").read_text()))
    factors = load_factors(output, market)
    start, middle, end = (
        millis(protocol[k]) for k in ["validation_start", "evaluation_start", "end"]
    )
    mid_marks = {s: float(f.loc[middle, "mark"]) for s, f in market.items()}
    terminal = json.loads((parent / "combined_evaluation_marks.json").read_text())
    if terminal["timestamp"] != end:
        raise ValueError("Endpoint mark clock differs")
    endpoint = {s: terminal["marks"][s] for s in market}
    rows = []
    for policy in protocol["candidates"]:
        score, _ = run_account(
            policy, market, signals_for(policy, factors), meta, start, middle, mid_marks
        )
        score["selection_gate"] = economic_gate(score, 20, 5)
        rows.append(score)
        print(
            {
                "validation": policy["id"],
                "net_return": score["net_return"],
                "small": score["sleeves"]["small"]["net_pnl"],
                "large": score["sleeves"]["large"]["net_pnl"],
                "gate": score["selection_gate"],
            },
            flush=True,
        )
    save(output / "validation_scores.json", rows)
    qualified = [x for x in rows if x["selection_gate"]]
    selected = sorted(qualified or rows, key=lambda x: (-x["net_return"], x["candidate"]))[0]
    policy = next(x for x in protocol["candidates"] if x["id"] == selected["candidate"])
    save(
        output / "selection.json",
        {
            "candidate": policy,
            "validation_gate": bool(qualified),
            "evaluation_outcomes_read": False,
            "scope": "retrospective candidate"
            if qualified
            else "validation-ranked diagnostic only",
        },
    )
    signals = signals_for(policy, factors)
    result, state = run_account(policy, market, signals, meta, middle, end, endpoint)
    save(output / "evaluation_score.json", result)
    save(output / "evaluation_state.json", state)
    checkpoint = middle + (end - middle) // 120000 * MINUTE
    _, prefix = run_account(
        policy, market, signals, meta, middle, end, endpoint, break_ms=checkpoint
    )
    resumed, restored = run_account(
        policy,
        market,
        signals,
        meta,
        checkpoint,
        end,
        endpoint,
        state=json.loads(json.dumps(prefix)),
    )
    if resumed != result or restored != state:
        raise ValueError("Public strategy account restart differs")
    save(
        output / "restart_verification.json", {"exact_state_equal": True, "exact_score_equal": True}
    )
    stress = []
    for extra in [2, 5]:
        c = {
            **policy,
            "execution_cost_bps": policy["execution_cost_bps"] + extra,
            "fee_bps": policy["fee_bps"] + extra,
        }
        score, _ = run_account(c, market, signals, meta, middle, end, endpoint)
        stress.append({"extra_cost_bps_each_side": extra, **score})
    save(output / "cost_stress.json", stress)
    eligible = bool(
        qualified
        and economic_gate(result, 30, 15)
        and all(
            p["net_pnl"] > 0 and p["net_pnl"] - p["terminal_open_net_pnl"] > 0
            for p in stress[0]["sleeves"].values()
        )
    )
    save(
        output / "run_status.json",
        {
            "status": "completed",
            "historical_paper_screen_passed": eligible,
            "production_eligible": False,
            "orders_enabled": False,
            "independent_forward_evidence": False,
        },
    )
    print(
        {"evaluation_net_return": result["net_return"], "historical_paper_screen_passed": eligible},
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "study"])
    parser.add_argument("output", type=Path)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.parent.absolute(), args.root.absolute(), args.output.absolute())
    else:
        study(args.output.absolute())
