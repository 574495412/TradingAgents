"""Exercise the independent sparse ledger audit on the real-source regression."""

import argparse
from pathlib import Path

import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.factor_lab.metrics import portfolio_metrics
from research_tools.max_effect.run_accounts import verify_manifest
from research_tools.regime_program.storage import write_account
from research_tools.sparse_market.account_audit import audit_account
from research_tools.sparse_market.engineering_worker import inputs
from research_tools.sparse_market.identity import directives


def verify(regression, output):
    output.mkdir(parents=True, exist_ok=False)
    verify_manifest(regression)
    plan = read(regression / "plan.json")
    bars, rates, scores = inputs(plan, "extended")
    folder = regression / "extended"
    state = read(folder / "resume_state.json")
    a = {
        key: read(folder / (key + ".json"))
        for key in ["events", "trades", "requests", "positions", "resume_state"]
    }
    a["curve"] = pd.read_parquet(folder / "curve.parquet")
    a.update(
        pending_exits=state["pending_exits"],
        capacity={},
        risk_halted=state["risk_halted"],
        engine="repeated_historical_engineering_regression_only",
    )
    days = pd.date_range(plan["start"], plan["end"], freq="D", tz="UTC")
    benchmark = bars["BTCUSDT"].close.reindex(days - pd.Timedelta(minutes=5))
    benchmark.index = days
    a["metrics"] = portfolio_metrics(
        a["curve"], a["trades"], a["events"], plan["config"]["capital"], benchmark
    )
    write_account(
        output / "accounts" / "regression",
        a,
        plan["config"],
        {"start": plan["start"], "end": plan["end"]},
    )
    result = audit_account(
        output,
        {"id": "regression"},
        bars,
        rates,
        scores,
        directives(read(Path(plan["registry"])), sorted(bars)),
    )
    receipt = {
        "schema": "sparse-auditor-real-regression.v1",
        "passed": True,
        "source_hashes": {
            str(p): sha(p)
            for p in [
                Path(__file__),
                Path(__file__).with_name("account_audit.py"),
                regression / "completion.json",
            ]
        },
        "independent_ledger_check": result,
        "window_days": 1,
        "engineering_only": True,
        "new_alpha_accounts": 0,
        "independent_oos": False,
        "profit_goal_met": False,
    }
    save(output / "verification.json", receipt)
    save(
        output / "completion.json",
        {
            "manifest": {
                str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()
            },
            "profit_goal_met": False,
        },
    )
    print(receipt, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--regression", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    verify(a.regression, a.output)
