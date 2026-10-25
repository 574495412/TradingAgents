"""Run: python -m tradingagents.research.equities --help."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .data import LongbridgeQuotes, capture_yahoo, digest, load_dataset, write_json
from .engine import account, signals

HERE = Path(__file__).resolve().parent


def freeze(plan_path, legacy_path, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    frozen = output / "source"
    frozen.mkdir()
    for file in HERE.glob("*.py"):
        shutil.copyfile(file, frozen / file.name)
    shutil.copyfile(plan_path, output / "protocol.json")
    shutil.copyfile(legacy_path, frozen / "legacy_ema_daily.py")
    inventory = {file.name: digest(file) for file in sorted(frozen.glob("*.py"))}
    inventory["protocol.json"] = digest(output / "protocol.json")
    write_json(output / "source-identities.json", inventory)
    packages = {}
    for name in ("numpy", "pandas", "yfinance", "exchange-calendars", "longport"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    write_json(output / "environment.json", {"python": sys.version, "packages": packages})
    return output


def legacy_function(output):
    spec = importlib.util.spec_from_file_location(
        "frozen_equity_factor", output / "source/legacy_ema_daily.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.calculate_factor


def choose(training):
    qualified = {}
    for candidate, cases in training.items():
        if all(
            m["net_return"] > 0
            and m["closed_trades"] >= 20
            and m["profit_factor"] is not None
            and m["profit_factor"] >= 1.1
            and m["max_drawdown"] <= 0.20
            for m in cases
        ):
            qualified[candidate] = min(m["net_return"] for m in cases)
    return max(qualified, key=lambda c: (qualified[c], c)) if qualified else "cash"


def study(plan, dataset, output):
    frames, calendars, manifest = load_dataset(dataset, plan)
    write_json(
        output / "dataset-identity.json",
        {
            "path": str(Path(dataset).resolve()),
            "manifest_sha256": digest(Path(dataset) / "manifest.json"),
            "manifest": manifest,
        },
    )
    legacy = legacy_function(output)
    decisions = {
        c: {s: signals(f, c, legacy) for s, f in frames.items()}
        for c in [*plan["candidates"], "cash", "buy_hold"]
    }
    selection, records = {}, []

    def run(market, phase, candidate, cost):
        spec = plan["markets"][market]
        symbols = [s["symbol"] for s in spec["symbols"]]
        payload = account(
            {s: frames[s] for s in symbols},
            calendars[market],
            {s: decisions[candidate][s] for s in symbols},
            spec,
            start=plan[phase][0],
            end=plan[phase][1],
            cost_bps=cost,
        )
        name = f"{market}_{phase}_{candidate}_{cost}bps"
        target = output / name
        target.mkdir()
        result = {
            "schema": "ta-equity-account.v1",
            "status": "历史研究，未获实盘资格",
            "asset_class": "equity",
            "market": market,
            "currency": spec["currency"],
            "phase": phase,
            "candidate": candidate,
            "cost_bps_per_side": cost,
            "provider": manifest["provider"],
            "broker_verified": manifest["broker_verified"],
            "dividend_cash_verified": False,
            "production_eligible": False,
            "independent_forward_oos": False,
            "economic_account_evaluated": True,
            "selected_by_training": selection.get(market, {}).get("candidate") == candidate,
            "window": {"start_at": plan[phase][0], "end_at": plan[phase][1]},
            "account": payload,
            "limitations": plan["assumptions"],
        }
        write_json(target / "result.json", result)
        records.append(
            {
                "name": name,
                "market": market,
                "phase": phase,
                "candidate": candidate,
                "cost_bps": cost,
                "metrics": payload["metrics"],
            }
        )
        return payload["metrics"]

    # Persist the entire training evidence and selection before evaluating any
    # later-period account. All later candidates are diagnostic comparisons.
    for market, spec in plan["markets"].items():
        training = {
            c: [run(market, "train", c, cost) for cost in spec["cost_bps"]]
            for c in plan["candidates"]
        }
        selection[market] = {"candidate": choose(training), "training": training}
    write_json(output / "selection.json", selection)
    for market, spec in plan["markets"].items():
        candidates = list(
            dict.fromkeys([selection[market]["candidate"], *plan["candidates"], "buy_hold"])
        )
        for phase in ("validation", "holdout", "continuous"):
            for candidate in candidates:
                for cost in spec["cost_bps"]:
                    run(market, phase, candidate, cost)
    summary = {
        "schema": "ta-equity-study.v1",
        "status": "研究回测完成，生产资格关闭",
        "asset_class": "equity",
        "production_eligible": False,
        "economic_account_evaluated": True,
        "independent_forward_oos": False,
        "provider": manifest["provider"],
        "broker_verified": manifest["broker_verified"],
        "selection": {market: item["candidate"] for market, item in selection.items()},
        "account_count": len(records),
        "records": records,
        "limitations": plan["assumptions"],
    }
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="港美股研究：长桥只读行情、公开历史及独立现金回测")
    sub = parser.add_subparsers(dest="command", required=True)
    capture = sub.add_parser(
        "capture", help="capture public Yahoo research history, explicitly not broker data"
    )
    capture.add_argument("--plan", type=Path, default=HERE / "plan.json")
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument(
        "--vendor-cache",
        type=Path,
        help="reuse raw CSVs from an incomplete capture with the identical plan",
    )
    capture.add_argument(
        "--daily-overrides", type=Path, help="explicit hashed hourly sources for missing daily bars"
    )
    run = sub.add_parser("run", help="freeze sources and run time-separated stock accounts")
    run.add_argument("--plan", type=Path, default=HERE / "plan.json")
    run.add_argument("--dataset", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--legacy-factor", type=Path, required=True)
    broker = sub.add_parser(
        "broker-capture", help="Longbridge quote-only capture using caller environment"
    )
    broker.add_argument("--start", required=True)
    broker.add_argument("--end", required=True, help="exclusive local date")
    broker.add_argument("--symbols", nargs="+", required=True)
    broker.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "broker-capture":
        from longport.openapi import Config

        values = [
            os.getenv("LONGPORT_" + key) or os.getenv("LONGBRIDGE_" + key)
            for key in ("APP_KEY", "APP_SECRET", "ACCESS_TOKEN")
        ]
        if not all(values):
            parser.error(
                "export LONGPORT_APP_KEY / APP_SECRET / ACCESS_TOKEN (LONGBRIDGE aliases accepted)"
            )
        config = Config.from_apikey(*values, enable_print_quote_packages=False)
        args.output.mkdir(parents=True, exist_ok=False)
        quotes = LongbridgeQuotes(config)
        for symbol in args.symbols:
            if not symbol.endswith((".US", ".HK")) or not symbol.replace(".", "").isalnum():
                raise ValueError("unsupported broker symbol")
            quotes.history(symbol, args.start, args.end).to_csv(args.output / f"{symbol}.csv")
        write_json(
            args.output / "manifest.json",
            {
                "provider": "longbridge-quote-sdk",
                "start": args.start,
                "end": args.end,
                "corporate_actions_complete": False,
                "production_eligible": False,
                "status": "quotes captured; corporate actions and calendar audit required before cash accounting",
                "files": {f.name: digest(f) for f in args.output.glob("*.csv")},
            },
        )
        print("broker quotes captured; corporate-action ledger remains required")
        return
    plan = json.loads(args.plan.read_text())
    if args.command == "capture":
        overrides = json.loads(args.daily_overrides.read_text()) if args.daily_overrides else None
        manifest = capture_yahoo(plan, args.output, args.vendor_cache, overrides)
        print(
            json.dumps(
                {"provider": manifest["provider"], "symbols": manifest["symbols"]},
                ensure_ascii=False,
            )
        )
    else:
        output = freeze(args.plan, args.legacy_factor, args.output)
        summary = study(plan, args.dataset, output)
        print(
            json.dumps(
                {k: summary[k] for k in ("account_count", "selection", "production_eligible")},
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
