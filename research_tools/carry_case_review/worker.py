"""One real interpreter segment of a preregistered, selected short carry case."""

import argparse
import os
from pathlib import Path

import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.factor_lab.study import executor


def inputs(source, symbol):
    markets = [
        {symbol: pd.read_parquet(source / "symbols" / f"{symbol}_{k}_5m.parquet")}
        for k in ["spot", "perpetual", "mark"]
    ]
    funding = {symbol: pd.read_parquet(source / "funding" / (symbol + "_exact.parquet"))}
    return (*markets, funding)


def metrics(result, capital):
    nav = result["curve"].equity
    return {
        "net_return": float(nav.iloc[-1] / capital - 1),
        "net_pnl": float(nav.iloc[-1] - capital),
        "closed_net_pnl": sum(t["net_pnl"] for t in result["trades"]),
        "max_drawdown": float((1 - nav / nav.cummax()).max()),
        "fees": sum(e.get("fee", 0.0) for e in result["events"]),
        "funding_net": sum(e["cash_delta"] for e in result["events"] if e["type"] == "funding"),
        "closed_trades": len(result["trades"]),
        "cagr": None,
        "sharpe": None,
        "win_rate": None,
        "scope": "Selected short case; no annualized or population inference",
    }


def run(source, symbol, slippage, mode, output, checkpoint=None):
    plan = read(source / "plan.json")
    case = next(c for c in plan["cases"] if c["symbol"] == symbol)
    config = {**plan["account"], "slippage_bps": slippage}
    begin, end = case["decision"], case["exit_deadline"]
    initial = None
    if mode == "first":
        end = (pd.Timestamp(begin) + pd.Timedelta(hours=4, minutes=5)).isoformat()
    elif mode == "restore":
        if checkpoint is None:
            raise ValueError("Restore requires actual disk checkpoint")
        initial = read(checkpoint)
        begin = initial["next_clock"]
    signals = [
        {"time": case["decision"], "assets": {symbol: {"entry": mode != "cash", "exit": False}}}
    ]
    output.mkdir(parents=True, exist_ok=False)
    save(
        output / "invocation.json",
        {
            "pid": os.getpid(),
            "mode": mode,
            "start": begin,
            "end": end,
            "terminal": case["exit_deadline"],
            "source_completion_sha256": sha(source / "completion.json"),
            "checkpoint_sha256": sha(checkpoint) if checkpoint else None,
        },
    )
    engine = executor(plan["engine"])
    result = engine.replay(
        *inputs(source, symbol),
        signals,
        config,
        begin,
        end,
        initial_state=initial,
        final_end=case["exit_deadline"],
    )
    result["curve"].to_parquet(output / "equity_5m.parquet")
    for key in ["events", "trades", "requests", "positions", "wallets", "flags"]:
        save(output / (key + ".json"), result[key])
    save(output / "signals.json", signals)
    save(output / "state.json", result["resume_state"])
    report = {
        "config": config,
        "window": {"start": begin, "end": end},
        "case": case,
        "mode": mode,
        "metrics": metrics(result, config["capital"]),
        "flags": result["flags"],
        "risk_halted": result["risk_halted"],
        "capacity": result["capacity"],
        "source_hashes": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
        "financial_strategy_accounts": 0,
        "profit_goal_met": False,
        "portfolio_actual": None,
    }
    save(output / "result.json", report)
    print(
        {
            "pid": os.getpid(),
            "symbol": symbol,
            "slippage": slippage,
            "mode": mode,
            "nav_points": len(result["curve"]),
            "open_pairs": len(result["positions"]),
        },
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--slippage", required=True, type=int)
    parser.add_argument("--mode", required=True, choices=["full", "first", "restore", "cash"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    run(args.source, args.symbol, args.slippage, args.mode, args.output, args.checkpoint)
