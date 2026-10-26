"""Repair prior-volume input context; retain all frozen economic rules and data.

Direct invocation installs namespace-only packages so application initializers
and dotenv are never loaded. This is retrospective implementation correction,
not independent financial evidence, a capacity test or a new strategy search.
"""

import argparse
import hashlib
import json
import math
import subprocess
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PLAN = ROOT / "docs/research/equities_pit_stocks_20261026/volume-context-correction-plan.json"
for name in (
    "tradingagents",
    "tradingagents.research",
    "tradingagents.research.equities",
    "factors",
    "factors.common",
    "factors.equity",
):
    package = types.ModuleType(name)
    package.__path__ = [str(ROOT.joinpath(*name.split(".")))]
    sys.modules[name] = package

import pandas as pd  # noqa: E402

from tradingagents.research.equities.audit_v002 import audit  # noqa: E402
from tradingagents.research.equities.continuous import replay, statistics  # noqa: E402
from tradingagents.research.equities.data import validate  # noqa: E402
from tradingagents.research.equities.pit_stocks import load  # noqa: E402

assert "dotenv" not in sys.modules


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def context_inputs(frames, calendar, start, last):
    first = calendar.index[calendar.index >= pd.Timestamp(start)][0]
    offset = calendar.index.get_loc(first)
    if offset < 20:
        raise ValueError("20 genuine pre-evaluation observations required")
    context = calendar.loc[calendar.index[offset - 20] : last]
    selected = {}
    for symbol, frame in frames.items():
        if not context.index.isin(frame.index).all():
            raise ValueError("Missing pre-evaluation source observation")
        selected[symbol] = frame.loc[context.index].copy()
        validate(selected[symbol], context)
    return selected, context


def verify_parent(plan):
    parent = ROOT / plan["parent_run"]
    reg = read(parent / "registration.json")
    pit = ROOT / plan["pit_dataset"]
    inputs = {}
    for name, digest in reg["source_hashes"].items():
        assert sha(ROOT / name) == digest, name
        inputs[name] = digest
    for name, digest in reg["input_hashes"].items():
        assert sha(pit / name) == digest, name
        inputs[str((pit / name).relative_to(ROOT))] = digest
    manifest = pit / "prices/manifest.json"
    assert sha(manifest) == reg["price_manifest_sha256"]
    inputs[str(manifest.relative_to(ROOT))] = sha(manifest)
    for row in read(manifest)["symbols"].values():
        for file_key, hash_key in (("file", "sha256"), ("signal_file", "signal_sha256")):
            path = pit / "prices" / row[file_key]
            assert sha(path) == row[hash_key]
            inputs[str(path.relative_to(ROOT))] = sha(path)
    for path in sorted(parent.glob("*/state.json")):
        inputs[str(path.relative_to(ROOT))] = sha(path)
    return inputs


def prepare(plan):
    original = load(ROOT / plan["pit_dataset"])
    strategy, ids, annual, full_frames, signal_calendar, books, regimes = original[:7]
    old_frames, old_calendar, spec = original[8:]
    frames, calendar = context_inputs(
        full_frames, signal_calendar, strategy["window"][0], old_calendar.index[-1]
    )
    end = str((old_calendar.index[-1] + pd.Timedelta(days=1)).date())
    return (
        strategy,
        ids,
        annual,
        books,
        regimes,
        old_frames,
        old_calendar,
        frames,
        calendar,
        spec,
        end,
    )


def recover(output):
    registration = read(output / "registration.json")
    assert sha(Path(__file__)) == registration["analysis_sha256"]
    assert all(sha(ROOT / name) == digest for name, digest in registration["inputs_sha256"].items())
    plan = read(output / "plan.json")
    strategy, _, _, books, regimes, _, _, frames, calendar, spec, end = prepare(plan)
    receipts = []
    for record in read(output / "summary.json")["records"]:
        folder = output / record["name"]
        state = replay(
            frames,
            calendar,
            books[record["policy"]].loc[calendar.index],
            regimes.loc[calendar.index],
            spec,
            market="US",
            policy=record["policy"],
            slippage_bps=record["slippage_bps"],
            identity=registration["identity"],
            start="2024-01-01",
            end=end,
            state=read(folder / "checkpoint.json"),
        )
        expected = read(folder / "state.json")
        assert state == expected, record["name"]
        write(folder / "resumed-state.json", state)
        receipts.append(
            {
                "name": record["name"],
                "exact_state_matches": True,
                "state_sha256": sha(folder / "state.json"),
                "resumed_sha256": sha(folder / "resumed-state.json"),
                "checkpoint_had_positions": bool(read(folder / "checkpoint.json")["positions"]),
            }
        )
    write(
        output / "fresh-process-restart.json",
        {"passed": True, "accounts": receipts, "external_attestation": False},
    )
    print("Fresh interpreter JSON recovery passed for eight corrected accounts", flush=True)


def run(output):
    plan = read(PLAN)
    inputs = verify_parent(plan)
    inputs[str(PLAN.relative_to(ROOT))] = sha(PLAN)
    strategy, _, _, books, regimes, old_frames, old_calendar, frames, calendar, spec, end = prepare(
        plan
    )
    output.mkdir(parents=True, exist_ok=False)
    write(output / "plan.json", plan)
    (output / "pre-correction-objectives.json").write_bytes(
        (ROOT / "tradingagents/research/equities/objectives.json").read_bytes()
    )
    first = next(day for day in calendar.index if day >= pd.Timestamp(strategy["window"][0]))
    prior = calendar.index[calendar.index.get_loc(first) - 1]
    old_mean = old_frames[next(iter(old_frames))].volume.rolling(20).mean()
    unavailable = [
        str(day.date())
        for day in old_calendar.index
        if first <= day
        and pd.isna(old_mean.at[old_calendar.index[old_calendar.index.get_loc(day) - 1]])
    ]
    warmup = {
        s: {
            "known_observations_before_first_fill": int(len(f.loc[:prior].tail(20))),
            "known_mean_volume_on_prior_day": float(f.loc[:prior].volume.tail(20).mean()),
            "old_prior_mean_was_missing": bool(
                pd.isna(old_frames[s].volume.rolling(20).mean().at[prior])
            ),
        }
        for s, f in frames.items()
    }
    assert len(unavailable) == 19 and all(
        r["known_observations_before_first_fill"] == 20 and r["old_prior_mean_was_missing"]
        for r in warmup.values()
    )
    write(
        output / "warmup-proof.json",
        {
            "first_evaluation_day": str(first.date()),
            "first_prior_day": str(prior.date()),
            "context_start": str(calendar.index[0].date()),
            "old_context_start": str(old_calendar.index[0].date()),
            "old_zero_cap_days": unavailable,
            "symbols": warmup,
            "pre_window_trades": 0,
        },
    )
    registration = {
        "inputs_sha256": inputs,
        "analysis_sha256": sha(Path(__file__)),
        "plan_sha256": sha(PLAN),
        "pre_correction_goal_sha256": sha(ROOT / "tradingagents/research/equities/objectives.json"),
        "declared_actual_replays": 8,
        "new_strategy_hypotheses": 0,
        "independent_OOS": False,
        "external_attestation": False,
    }
    registration["identity"] = hashlib.sha256(
        json.dumps(registration, sort_keys=True).encode()
    ).hexdigest()
    write(output / "registration.json", registration)
    (output / "analysis_source.py").write_bytes(Path(__file__).read_bytes())
    records, audits, comparisons = [], [], []
    for policy in strategy["policies"]:
        for slip in strategy["slippage_bps"]:
            name = f"US_{policy}_10000_{slip}bps"
            folder = output / name
            folder.mkdir()
            args = (
                frames,
                calendar,
                books[policy].loc[calendar.index],
                regimes.loc[calendar.index],
                spec,
            )
            kwargs = {
                "market": "US",
                "policy": policy,
                "slippage_bps": slip,
                "identity": registration["identity"],
            }
            state = replay(*args, **kwargs, start=strategy["window"][0], end=end)
            assert all(f["day"] >= str(first.date()) for f in state["fills"])
            for request in state["requests"]:
                day = pd.Timestamp(request["day"])
                previous = calendar.index[calendar.index.get_loc(day) - 1]
                cap = math.floor(
                    float(frames[request["symbol"]].volume.rolling(20).mean().at[previous]) * 0.001
                )
                assert abs(request["volume_capped"]) <= cap
            write(folder / "state.json", state)
            write(
                folder / "checkpoint.json",
                replay(*args, **kwargs, start=strategy["window"][0], end="2024-01-01"),
            )
            proof = audit(state, frames, calendar)
            audits.append({"name": name, **proof})
            write(folder / "internal-ledger-audit.json", proof)
            metrics = statistics(state)
            record = {
                "name": name,
                "market": "US",
                "policy": policy,
                "capital": 10000,
                "slippage_bps": slip,
                "metrics": metrics,
            }
            records.append(record)
            result = {
                "schema": "ta-equity-account.v2",
                "asset_class": "equity",
                "market": "US",
                "currency": "USD",
                "candidate": policy,
                "phase": "prior_volume_context_implementation_correction",
                "status": "同协议成交量上下文纠正；来源及独立验收仍未通过",
                "identity": registration["identity"],
                "initial_cash": 10000,
                "slippage_bps_per_side": slip,
                "broker_verified": False,
                "production_eligible": False,
                "independent_forward_oos": False,
                "external_independent_verified": False,
                "economic_account_evaluated": True,
                "ledger_verified": True,
                "restart_verified": False,
                "window": {"start_at": strategy["window"][0], "end_at": end},
                "account": {
                    "metrics": metrics,
                    "equity_curve": state["equity_curve"],
                    "trades": state["trades"],
                },
                "limitations": [
                    "Raw source conflicts and EXAS long-window/cash gaps retained",
                    "Daily target maintenance is inherited; monthly signal is not monthly-only execution",
                    "Internal consistency and hypothetical fills do not certify profit or actual execution",
                ],
            }
            write(folder / "result.json", result)
            previous = read(ROOT / plan["parent_run"] / name / "state.json")
            old_metrics = statistics(previous)
            comparisons.append(
                {
                    "name": name,
                    "policy": policy,
                    "slippage_bps": slip,
                    "old_metrics": old_metrics,
                    "corrected_metrics": metrics,
                    "economic_curve_exactly_unchanged": previous["equity_curve"]
                    == state["equity_curve"],
                    "fills_exactly_unchanged": previous["fills"] == state["fills"],
                    "old_first_fill": previous["fills"][0]["day"] if previous["fills"] else None,
                    "corrected_first_fill": state["fills"][0]["day"] if state["fills"] else None,
                }
            )
            write(
                output / "status.json",
                {
                    "status": "running",
                    "completed_correction_accounts": len(records),
                    "actual_financial_replays": len(records),
                },
            )
            print(
                json.dumps(
                    {
                        "completed": len(records),
                        "name": name,
                        "cagr": metrics["annualized_return"],
                        "close_drawdown": metrics["close_max_drawdown"],
                    }
                ),
                flush=True,
            )
    summary = {
        "schema": "equity-pit-context-correction.v1",
        "records": records,
        "parent_run": plan["parent_run"],
        "original_requested_window": strategy["window"],
        "bounded_window": [strategy["window"][0], end],
        "actual_financial_replays": 8,
        "new_strategy_hypotheses": 0,
        "profit_goal_status": "not_met",
        "actual_qualified_total_account": None,
        "full_window_available": False,
        "production_eligible": False,
        "independent_forward_oos": False,
    }
    write(output / "summary.json", summary)
    write(output / "comparisons.json", comparisons)
    write(
        output / "internal-ledger-audits.json",
        {
            "passed": all(r["passed"] for r in audits),
            "accounts": audits,
            "external_attestation": False,
        },
    )
    child = subprocess.run(
        [sys.executable, str(Path(__file__).absolute()), "--recover", str(output.absolute())],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    print(child.stdout.strip(), flush=True)
    for record in records:
        path = output / record["name"] / "result.json"
        result = read(path)
        result["restart_verified"] = True
        write(path, result)
    write(
        output / "status.json",
        {
            "status": "complete",
            "completed_correction_accounts": 8,
            "actual_financial_replays": 8,
            "new_strategy_hypotheses": 0,
        },
    )
    write(
        output / "completion.json",
        {
            "manifest": {
                str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()
            },
            "completed_financial_replays": 8,
            "profit_goal_met": False,
            "external_attestation": False,
        },
    )


def self_test():
    days = pd.bdate_range("2023-01-02", periods=22)
    calendar = pd.DataFrame(
        index=days,
        data={
            "open": [pd.Timestamp(d, tz="UTC") + pd.Timedelta(hours=14, minutes=30) for d in days],
            "close": [pd.Timestamp(d, tz="UTC") + pd.Timedelta(hours=21) for d in days],
        },
    )
    frame = pd.DataFrame(
        index=days,
        data={
            "open": 10.0,
            "high": 10.0,
            "low": 10.0,
            "close": 10.0,
            "volume": 100000.0,
            "dividend": 0.0,
            "split": 0.0,
        },
    )
    first = days[20]
    frame.loc[first, "volume"] = 1.0
    frames, context = context_inputs({"TEST.US": frame}, calendar, str(first.date()), days[-1])
    spec = {
        "initial_cash": 10000,
        "minimum_fee": 1,
        "commission_bps": 1,
        "dividend_haircut": 0.3,
        "symbols": [{"symbol": "TEST.US", "lot_size": 1, "instrument": "equity"}],
    }
    state = replay(
        frames,
        context,
        pd.DataFrame(0.8, index=context.index, columns=["TEST.US"]),
        pd.Series("test", index=context.index),
        spec,
        market="US",
        start=str(first.date()),
        end=str((first + pd.Timedelta(days=1)).date()),
        policy="test",
        slippage_bps=5,
        identity="synthetic-regression-test",
    )
    assert state["fills"][0]["day"] == str(first.date()) and state["fills"][0]["quantity"] == 100
    assert len(state["equity_curve"]) == 2 and len(state["fills"]) == 1
    try:
        context_inputs({"TEST.US": frame.iloc[1:]}, calendar, str(first.date()), days[-1])
    except ValueError:
        pass
    else:
        raise AssertionError("Missing genuine context accepted")
    print(
        "Synthetic regression checks: first-day prior-only cap, no pre-window trading, missing context rejected; no research account created"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--recover", type=Path)
    parser.add_argument("--self-test", action="store_true")
    options = parser.parse_args()
    if options.self_test:
        self_test()
    elif options.recover:
        recover(options.recover)
    elif options.output:
        run(options.output)
    else:
        parser.error("Choose --output NEW_DIRECTORY, --recover or --self-test")
