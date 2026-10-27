"""One bounded diagnostic of sealed paths; does not execute a trading strategy."""

import argparse
import math
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

from research_tools.cross_sectional_momentum.offline_coverage import source_versions
from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest
from research_tools.shared_failure.fixed_path import matched_btc_path, price_increments
from research_tools.sparse_market.cohort import feature_views
from research_tools.sparse_market.identity import crypto_context


def select_accounts(plan):
    cases = []
    for name in plan["runs"]:
        root = Path(name)
        verify_manifest(root)
        summary = read(root / "summary.json")
        for row in summary.get("account_rows", summary.get("rows", [])):
            if (
                row.get("phase") == "continuous2025_2026"
                and row.get("variant") == "daily_archive"
                and row.get("candidate") != "cash"
            ):
                cases.append(
                    {
                        "root": str(root),
                        "id": row["id"],
                        "candidate": row["candidate"],
                        "cost": row["slippage_bps"],
                    }
                )
    if len(cases) != 12 or len({(r["candidate"], r["cost"]) for r in cases}) != 12:
        raise ValueError("Frozen all-policy/all-cost diagnostic roster differs")
    return cases


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    path = Path(__file__).with_name("plan.json")
    plan = read(path)
    cases = select_accounts(plan)
    assert len(cases) <= plan["budget"]["maximum_existing_accounts"]
    cohort, daily, bundle = (Path(plan[k]) for k in ["cohort", "daily", "bundle"])
    for folder in [cohort, daily, bundle]:
        verify_manifest(folder)
    bindings = [
        path,
        Path(__file__),
        Path(__file__).with_name("fixed_path.py"),
        Path("research_tools/cross_sectional_momentum/offline_coverage.py"),
        Path("research_tools/sparse_market/cohort.py"),
        Path("research_tools/sparse_market/identity.py"),
        Path("research_tools/contract_lifecycles/segments.py"),
        Path("research_tools/contract_lifecycles/timeline.py"),
        Path("factors/common/max_effect.py"),
        cohort / "completion.json",
        daily / "completion.json",
        bundle / "completion.json",
    ]
    for case in cases:
        folder = Path(case["root"]) / "accounts" / case["id"]
        account = read(folder / "result.json")
        if account["open_positions"]:
            raise ValueError("Expected observed flat terminal path; do not manufacture closure")
        for name, h in account["source_hashes"].items():
            if sha(folder / name) != h:
                raise ValueError("Sealed account changed")
        bindings.extend([folder / "result.json", *[folder / n for n in account["source_hashes"]]])
    save(output / "plan.json", plan)
    save(
        output / "registration.json",
        {
            "wall_ns": time.time_ns(),
            "roster": cases,
            "source_hashes": {str(p): sha(p) for p in bindings},
            "after_observing_sealed_loss_results": True,
            "before_new_attribution_results": True,
            "new_financial_accounts": 0,
            "new_economic_variants": 0,
        },
    )
    (output / "paths").mkdir()
    registry = read(cohort / "registry.json")
    symbols = read(cohort / "symbols.json")
    prices = {
        s: pd.read_parquet(
            bundle / "symbols" / (s + "_5m.parquet"), columns=["close", "price_usable"]
        )
        for s in symbols
    }
    rawpaths = sorted((daily / "symbols").glob("*_1d.parquet"))
    versions = source_versions(plan, rawpaths)
    betas = {}
    for variant, raw in versions.items():
        base, episodes = feature_views(raw, registry)
        betas[variant] = (base, episodes)
    rows = []
    daily_rows = []
    campaigns = []
    price_series = {}
    active_series = {}
    maxerror = 0.0
    for case in cases:
        folder = Path(case["root"]) / "accounts" / case["id"]
        account = read(folder / "result.json")
        events = read(folder / "events.json")
        curve = pd.read_parquet(folder / "equity_5m.parquet")
        index = pd.DatetimeIndex(curve.date)
        refs = index - pd.Timedelta(minutes=5)
        btc_source = prices["BTCUSDT"].reindex(refs)
        if not btc_source.price_usable.eq(True).all():
            raise ValueError("Missing actual BTC benchmark price")
        btc = pd.Series(btc_source.close.to_numpy(), index=index)

        def quote(symbol, clock):
            row = prices[symbol].loc[clock - pd.Timedelta(minutes=5)]
            if not row.price_usable:
                raise ValueError("Reference fill is not raw usable price")
            return float(row.close)

        for event in events:
            if event["type"] != "funding":
                ref = quote(event["symbol"], pd.Timestamp(event["time"]))
                error = abs(event["quantity"] * abs(event["price"] - ref) - event["slippage_cost"])
                if error > 1e-7:
                    raise ValueError("Raw fill reference differs from recorded slippage")
        comp = price_increments(
            curve, events, (account["config"]["fee_bps"] + case["cost"]) / 10000
        )
        expected = (
            account["metrics"]["net_pnl"]
            - account["metrics"]["funding_net"]
            + account["metrics"]["fees"]
            + account["metrics"]["slippage_cost"]
        )
        error = abs(comp.price.sum() - expected)
        maxerror = max(maxerror, error)
        if error > 1e-6:
            raise ValueError("NAV reference-price identity differs")
        costmap = defaultdict(lambda: {"fees": 0.0, "slippage": 0.0, "funding": 0.0})
        for event in events:
            d = costmap[(event["symbol"], event["position_id"])]
            if event["type"] == "funding":
                d["funding"] += event["cash_delta"]
            else:
                d["fees"] += event["fee"]
                d["slippage"] += event["slippage_cost"]
        row = {
            **case,
            "net_pnl": account["metrics"]["net_pnl"],
            "price_component": float(comp.price.sum()),
            "fees": account["metrics"]["fees"],
            "slippage": account["metrics"]["slippage_cost"],
            "funding": account["metrics"]["funding_net"],
            "risk_halted": account["risk_halted"],
            "closed_cycles": account["metrics"]["closed_trades"],
            "beta_sources": {},
        }
        for variant, (base, episodes) in betas.items():

            def beta_at(symbol, submitted, base=base, episodes=episodes):
                context = crypto_context(registry, symbol, submitted)
                view = (
                    episodes[(symbol, context["feature_start"])]
                    if context["feature_start"]
                    else base[symbol]
                )
                day = submitted.floor("D")
                value = float(view.at[day, "beta60"]) if day in view.index else float("nan")
                return {
                    "beta": value if math.isfinite(value) else None,
                    "source_day": day.isoformat(),
                    "reason": "known_prior60daily_OLS"
                    if math.isfinite(value)
                    else "incomplete_or_invalid_prior60daily_OLS",
                }

            increments, closed, missing = matched_btc_path(events, index, btc, quote, beta_at)
            rawprice = math.fsum(c["price_pnl"] for c in closed)
            error = abs(rawprice - comp.price.sum())
            maxerror = max(maxerror, error)
            if error > 1e-6:
                raise ValueError("Independent actual reference-fill basis does not reconcile")
            unit = float(increments.unit.sum())
            error = abs(unit - math.fsum(c["unit_market_pnl"] for c in closed))
            maxerror = max(maxerror, error)
            if error > 1e-6:
                raise ValueError("Benchmark timing/basis sums differ")
            known = increments.beta.notna().all()
            beta = float(increments.beta.sum()) if known else None
            if known:
                error = abs(beta - math.fsum(c["beta_market_pnl"] for c in closed))
                maxerror = max(maxerror, error)
                if error > 1e-6:
                    raise ValueError("Prior-beta matched shares do not reconcile")
            result = {
                "unit_btc_market_price": unit,
                "unit_btc_relative_price": rawprice - unit,
                "lag_beta_btc_market_price": beta,
                "lag_beta_btc_relative_price": rawprice - beta if beta is not None else None,
                "unknown_beta_entry_count": len(missing),
                "unknown_beta_intervals": int(increments.beta.isna().sum()),
                "missing_entry_details": missing,
            }
            row["beta_sources"][variant] = result
            for c in closed:
                costs = costmap[(c["symbol"], c["position_id"])]
                net = c["price_pnl"] + costs["funding"] - costs["fees"] - costs["slippage"]
                campaigns.append(
                    {
                        "account_id": case["id"],
                        "beta_source": variant,
                        **c,
                        **costs,
                        "net_pnl": net,
                        "unit_btc_relative_price": c["price_pnl"] - c["unit_market_pnl"],
                        "lag_beta_btc_relative_price": c["price_pnl"] - c["beta_market_pnl"]
                        if c["beta_market_pnl"] is not None
                        else None,
                    }
                )
            comp[variant + "_unit_btc_price"] = increments.unit
            comp[variant + "_lag_beta_btc_price"] = increments.beta
            comp[variant + "_lag_beta_residual_price"] = comp.price - increments.beta
        comp.to_parquet(output / "paths" / (case["id"] + "_5m.parquet"))
        days = comp.resample("D").sum(min_count=1)
        for column in comp.columns:
            if "lag_beta" in column:
                missing_days = comp[column].isna().resample("D").max()
                days.loc[missing_days, column] = float("nan")
        for clock, d in days.iterrows():
            daily_rows.append(
                {
                    "id": case["id"],
                    "date": clock.isoformat(),
                    **{k: float(v) if pd.notna(v) else None for k, v in d.items()},
                }
            )
        loss = (-comp.net.clip(upper=0)).sum()
        weekly = comp.net.groupby(comp.index.strftime("%G-W%V")).sum().sort_values()
        row["worst_net_economic_weeks"] = [
            {"week": str(week), "net_pnl": float(value)} for week, value in weekly.iloc[:5].items()
        ]
        row["negative_5m_net_increment_mass"] = float(loss)
        row["top5_negative_week_loss_share"] = (
            float(-weekly.iloc[:5].clip(upper=0).sum() / (-weekly.clip(upper=0).sum()))
            if (weekly < 0).any()
            else None
        )
        price_series[case["id"]] = comp.price.resample("D").sum()
        active_series[case["id"]] = (curve.set_index("date").gross_exposure > 0).resample("D").max()
        rows.append(row)
        print(
            {
                "account_attributed": case["id"],
                "price": rawprice,
                "unit_btc_relative": row["beta_sources"]["daily_archive"][
                    "unit_btc_relative_price"
                ],
                "beta_relative": row["beta_sources"]["daily_archive"][
                    "lag_beta_btc_relative_price"
                ],
                "missing_beta_entries": row["beta_sources"]["daily_archive"][
                    "unknown_beta_entry_count"
                ],
            },
            flush=True,
        )
    correlations = []
    for i, left in enumerate(rows):
        for right in rows[i + 1 :]:
            a, b = left["id"], right["id"]
            mask = active_series[a] & active_series[b]
            x, y = price_series[a].loc[mask], price_series[b].loc[mask]
            corr = float(x.corr(y)) if len(x) > 2 and x.std() > 1e-15 and y.std() > 1e-15 else None
            correlations.append(
                {
                    "left": a,
                    "right": b,
                    "joint_active_days": int(mask.sum()),
                    "price_pnl_dollar_correlation": corr,
                    "different_gross_risk_not_adjusted": True,
                }
            )
    save(output / "account_attribution.json", rows)
    save(output / "campaign_attribution.json", campaigns)
    save(output / "daily_attribution.json", daily_rows)
    save(output / "joint_active_correlations.json", correlations)
    save(
        output / "summary.json",
        {
            "existing_accounts": len(rows),
            "source_beta_variants": list(betas),
            "new_financial_accounts": 0,
            "new_economic_variants": 0,
            "maximum_reconciliation_error_usdt": maxerror,
            "price_negative_accounts": sum(r["price_component"] < 0 for r in rows),
            "unit_btc_relative_negative_accounts": sum(
                r["beta_sources"]["daily_archive"]["unit_btc_relative_price"] < 0 for r in rows
            ),
            "lag_beta_sources": {
                variant: {
                    "complete_accounts": sum(
                        r["beta_sources"][variant]["lag_beta_btc_relative_price"] is not None
                        for r in rows
                    ),
                    "negative_complete_accounts": sum(
                        r["beta_sources"][variant]["lag_beta_btc_relative_price"] is not None
                        and r["beta_sources"][variant]["lag_beta_btc_relative_price"] < 0
                        for r in rows
                    ),
                    "unknown_entry_count": sum(
                        r["beta_sources"][variant]["unknown_beta_entry_count"] for r in rows
                    ),
                }
                for variant in betas
            },
            "third_party_review_performed": False,
            "causal_attribution_proven": False,
            "funded_hedge_simulated": False,
            "profit_goal_met": False,
        },
    )
    save(
        output / "completion.json",
        {
            "manifest": {
                str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()
            },
            "new_financial_accounts": 0,
        },
    )
    print(
        {"completed": str(output), "accounts": len(rows), "maximum_reconciliation_error": maxerror},
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
