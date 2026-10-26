"""Read-only economic attribution of the frozen stock diagnostics.

No engine import, new trades, candidate removal or admission from post-hoc
regression. Statistical controls are descriptions, not causal identification.
"""

import argparse
import hashlib
import json
import math
import shutil
from collections import defaultdict
from decimal import Decimal
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def dec(x):
    return Decimal(str(x))


def reconcile_account(state, frames):
    inventory = dict.fromkeys(frames, 0)
    totals = {s: defaultdict(Decimal) for s in frames}
    cashflows = dict.fromkeys(frames, Decimal(0))
    fills, dividends = defaultdict(list), defaultdict(list)
    for row in state["fills"]:
        fills[row["day"]].append(row)
    for row in state["dividends"]:
        dividends[row["day"]].append(row)
    prior_equity = dec(state["binding"]["spec"]["initial_cash"])
    tolerance = Decimal(".0000001")
    checks, daily, exposures, concentration = [], [], [], []
    for row in state["equity_curve"][1:]:
        day = pd.Timestamp(row["day"])
        previous_inventory = inventory.copy()
        costs, fee = dict.fromkeys(frames, Decimal(0)), dict.fromkeys(frames, Decimal(0))
        for fill in fills[row["day"]]:
            symbol, q = fill["symbol"], fill["quantity"]
            if isinstance(q, bool) or not isinstance(q, int) or q <= 0:
                raise ValueError("Attribution requires positive integer shares")
            opened, price = dec(frames[symbol].at[day, "open"]), dec(fill["price"])
            if abs(opened - dec(fill["raw_open"])) > tolerance:
                raise ValueError("Frozen fill differs from source opening price")
            direction = 1 if fill["side"] == "BUY" else -1 if fill["side"] == "SELL" else 0
            if not direction or inventory[symbol] + direction * q < 0:
                raise ValueError("Unexpected side or short inventory")
            cost = direction * (price - opened) * q
            if cost < 0 or abs(cost - dec(fill["slippage"])) > tolerance:
                raise ValueError("Execution differs from registered adverse slippage")
            costs[symbol] += cost
            fee[symbol] += dec(fill["fee"])
            inventory[symbol] += direction * q
            cashflows[symbol] += -direction * price * q - dec(fill["fee"])
        income = dict.fromkeys(frames, Decimal(0))
        for d in dividends[row["day"]]:
            if d["quantity"] != previous_inventory[d["symbol"]]:
                raise ValueError("Ex-date entitlement uses wrong inventory")
            income[d["symbol"]] += dec(d["amount"])
        total = Decimal(0)
        day_values = {}
        for s, f in frames.items():
            i = f.index.get_loc(day)
            if not i:
                raise ValueError("Missing prior session anchor")
            opened, close = dec(f.at[day, "open"]), dec(f.at[day, "close"])
            prior_close = dec(f.iloc[i - 1]["close"])
            expected_income = (
                previous_inventory[s]
                * dec(f.at[day, "dividend"])
                * (1 - dec(state["binding"]["spec"]["dividend_haircut"]))
            )
            if abs(expected_income - income[s]) > tolerance:
                raise ValueError("Dividend cash entitlement differs from frozen source")
            components = {
                "overnight_price_pnl": previous_inventory[s] * (opened - prior_close),
                "intraday_price_pnl": inventory[s] * (close - opened),
                "dividend_receivable": income[s],
                "fees": -fee[s],
                "slippage": -costs[s],
            }
            pnl = sum(components.values(), Decimal(0))
            for key, value in components.items():
                totals[s][key] += value
            totals[s]["net_marked_pnl"] += pnl
            cashflows[s] += income[s]
            total += pnl
            day_values[s] = inventory[s] * close
            daily.append(
                {
                    "day": row["day"],
                    "symbol": s,
                    **{k: float(v) for k, v in components.items()},
                    "net_marked_pnl": float(pnl),
                    "prior_quantity": previous_inventory[s],
                    "closing_quantity": inventory[s],
                    "closing_value": float(day_values[s]),
                }
            )
        equity = dec(row["equity"])
        error = abs(total - (equity - prior_equity))
        if error > tolerance:
            raise ValueError("Daily per-stock PnL does not reconcile to frozen equity")
        checks.append(float(error))
        exposures.append(float(sum(day_values.values(), Decimal(0)) / equity))
        concentration.append(float(max(day_values.values(), default=Decimal(0)) / equity))
        prior_equity = equity
    stocks = []
    last = pd.Timestamp(state["last_day"])
    for s in frames:
        expected = state["positions"].get(s, {}).get("quantity", 0)
        marked = inventory[s] * dec(frames[s].at[last, "close"])
        if (
            inventory[s] != expected
            or abs(cashflows[s] + marked - totals[s]["net_marked_pnl"]) > tolerance
        ):
            raise ValueError("Terminal cash-flow/inventory contribution mismatch")
        stocks.append(
            {
                "symbol": s,
                **{k: float(v) for k, v in totals[s].items()},
                "net_trade_cash_and_income": float(cashflows[s]),
                "ending_quantity": inventory[s],
                "ending_mark_value": float(marked),
                "closed_cycles": sum(t["symbol"] == s for t in state["trades"]),
            }
        )
    pnl = float(sum(r["net_marked_pnl"] for r in stocks))
    positive = sum(max(0, r["net_marked_pnl"]) for r in stocks)
    best = max(stocks, key=lambda r: r["net_marked_pnl"])
    by_day = pd.DataFrame(daily).groupby("day").net_marked_pnl.sum()
    return {
        "schema": "equity-stock-pnl-attribution.v1",
        "stocks": stocks,
        "net_marked_pnl": pnl,
        "exit_reserve": state["equity_curve"][-1]["exit_reserve"],
        "pnl_after_exit_reserve": pnl - state["equity_curve"][-1]["exit_reserve"],
        "max_daily_reconciliation_error": max(checks, default=0),
        "mean_close_exposure": float(np.mean(exposures)),
        "max_close_exposure": max(exposures),
        "close_sessions_above_80pct": sum(x > 0.800000001 for x in exposures),
        "max_single_stock_close_weight": max(concentration),
        "largest_contributor": best["symbol"],
        "largest_contributor_pnl": best["net_marked_pnl"],
        "largest_share_of_positive_contributions": best["net_marked_pnl"] / positive
        if positive
        else None,
        "pnl_minus_largest_contributor": pnl - best["net_marked_pnl"],
        "pnl_minus_top5_days": pnl - float(by_day.nlargest(5).sum()),
        "contribution_removal_is_tradable_counterfactual": False,
        "external_independent_attestation": False,
    }, daily


def regression(y, factors):
    import statsmodels.api as sm

    x = np.column_stack([np.ones(len(factors)), factors.to_numpy()])
    fit = sm.OLS(np.asarray(y), x).fit()
    robust = fit.get_robustcov_results(
        cov_type="HAC", maxlags=5, kernel="bartlett", use_correction=True, use_t=True
    )
    names = ["intercept_daily_return", *factors.columns]
    ci = robust.conf_int(alpha=0.05)
    return {
        "observations": int(fit.nobs),
        "rsquared": float(fit.rsquared),
        "design_condition_number": float(fit.condition_number),
        "hac_lags": 5,
        "coefficients": {
            name: {
                "value": float(robust.params[i]),
                "standard_error_hac": float(robust.bse[i]),
                "confidence95": ci[i].tolist(),
                "descriptive_pvalue": float(robust.pvalues[i]),
            }
            for i, name in enumerate(names)
        },
        "selection_corrected": False,
        "causal_alpha_identified": False,
    }


def paired_bootstrap(difference):
    n, length, draws = len(difference), 20, 2000
    rng = np.random.default_rng(20261026)
    samples = []
    values = np.asarray(difference)
    for _ in range(draws):
        starts = rng.integers(0, n, math.ceil(n / length))
        indices = np.concatenate([(s + np.arange(length)) % n for s in starts])[:n]
        samples.append(float(values[indices].mean()))
    return {
        "paired_mean_daily_return_difference": float(values.mean()),
        "circular_block_sessions": length,
        "draws": draws,
        "seed": 20261026,
        "descriptive_percentile95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "selection_corrected": False,
        "forward_forecast": False,
    }


def run(output):
    plan = json.loads((HERE / "depth_review_plan.json").read_text())
    run_dir, pit = ROOT / plan["frozen_run"], ROOT / plan["pit_dataset"]
    reg = json.loads((run_dir / "registration.json").read_text())
    manifest = json.loads((pit / "prices/manifest.json").read_text())
    if digest(pit / "prices/manifest.json") != reg["price_manifest_sha256"]:
        raise ValueError("Frozen quote manifest changed")
    frames = {}
    for symbol, record in manifest["symbols"].items():
        file = pit / "prices" / record["file"]
        if digest(file) != record["sha256"]:
            raise ValueError("Frozen stock prices changed")
        frames[symbol] = pd.read_csv(file, index_col="date", parse_dates=["date"])
    accounts = [plan["focal_account"], *plan["controls"]]
    states, inputs = (
        {},
        {
            str((run_dir / "registration.json").relative_to(ROOT)): digest(
                run_dir / "registration.json"
            ),
            str((pit / "prices/manifest.json").relative_to(ROOT)): digest(
                pit / "prices/manifest.json"
            ),
        },
    )
    for record in manifest["symbols"].values():
        source = pit / "prices" / record["file"]
        inputs[str(source.relative_to(ROOT))] = digest(source)
    for name in accounts:
        file = run_dir / name / "state.json"
        proof = json.loads((run_dir / name / "independent-audit.json").read_text())
        if digest(file) != proof["state_sha256"]:
            raise ValueError("Frozen account state changed")
        states[name] = json.loads(file.read_text())
        inputs[str(file.relative_to(ROOT))] = digest(file)
    benchmark_dir = ROOT / "data/research/equities_hkus_20261024_v006"
    benchmark_manifest = json.loads((benchmark_dir / "manifest.json").read_text())
    benchmarks = {}
    for symbol in ["SPY.US", "QQQ.US"]:
        r = benchmark_manifest["symbols"][symbol]
        file = benchmark_dir / r["file"]
        if digest(file) != r["sha256"]:
            raise ValueError("Frozen risk-control price changed")
        frame = pd.read_csv(file, index_col="date", parse_dates=["date"])
        benchmarks[symbol] = (frame.close + frame.dividend) / frame.close.shift(1) - 1
        inputs[str(file.relative_to(ROOT))] = digest(file)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, output / "attribution_source.py")
    shutil.copyfile(HERE / "depth_review_plan.json", output / "plan.json")
    registration = {
        "registered_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "role": plan["role"],
        "new_economic_accounts": 0,
        "source_sha256": digest(__file__),
        "plan_sha256": digest(output / "plan.json"),
        "frozen_run_identity": reg["identity"],
        "inputs_sha256": inputs,
        "versions": {p: version(p) for p in ["numpy", "pandas", "statsmodels", "scipy"]},
    }
    registration["identity"] = hashlib.sha256(
        json.dumps(registration, sort_keys=True).encode()
    ).hexdigest()
    write_json(output / "registration.json", registration)
    summaries, rows, returns = {}, [], {}
    for name, state in states.items():
        summary, detail = reconcile_account(state, frames)
        summaries[name] = summary
        rows.extend({"account": name, **r} for r in detail)
        returns[name] = pd.Series(
            {pd.Timestamp(r["day"]): r["daily_return"] for r in state["equity_curve"][1:]}
        )
    daily = pd.DataFrame(returns)
    if daily.isna().any().any():
        raise ValueError("Paired account windows do not align")
    b = pd.DataFrame(benchmarks).loc[daily.index]
    if b.isna().any().any():
        raise ValueError("Risk controls incomplete; no filled returns")
    factors = pd.DataFrame(
        {"SPY_gross_total_return": b["SPY.US"], "QQQ_minus_SPY_gross_return": b["QQQ.US"]}
    )
    focal, baseline = plan["focal_account"], plan["controls"][0]
    diff = daily[focal] - daily[baseline]
    analysis = {
        "schema": "equity-mechanism-diagnostics.v1",
        "focal_account": focal,
        "window": [str(daily.index[0].date()), str(daily.index[-1].date())],
        "risk_controls": "gross ETF returns, not implementable net benchmark accounts",
        "net_account_regression": regression(daily[focal], factors),
        "paired_net_difference_regression": regression(diff, factors),
        "paired_difference_bootstrap": paired_bootstrap(diff),
        "risk_control_correlation": float(b.iloc[:, 0].corr(b.iloc[:, 1])),
        "focal_fixed_cohort_net_correlation": float(daily[focal].corr(daily[baseline])),
        "new_economic_accounts": 0,
        "candidate_selection_changed": False,
        "external_independent_attestation": False,
        "profit_goal_met": False,
        "limitations": [
            plan["exposure_diagnostics"]["limits"],
            "Only the existing truncated common period; 64 EXAS sessions and terminal cash evidence remain missing.",
            "Per-stock and best-day subtraction is an arithmetic fragility diagnostic, not a new executable portfolio.",
            "An 80% target can drift with prices and integer-share/no-trade rules. Observed closing exposure does not certify intraday risk.",
        ],
    }
    write_json(output / "attribution.json", summaries)
    write_json(output / "diagnostics.json", analysis)
    pd.DataFrame(rows).to_csv(output / "daily-stock-pnl.csv", index=False)
    daily.join(factors).to_csv(output / "paired-daily-returns.csv", index_label="date")
    print(
        json.dumps(
            {
                "accounts_reconciled": len(summaries),
                "new_accounts": 0,
                "focal": summaries[focal],
                "risk_diagnostics": analysis,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args().output)
