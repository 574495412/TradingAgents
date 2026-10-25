"""Describe independent cold-start windows without stitching their NAV or trade samples."""

import math
import statistics

from research_tools.factor_lab.io import read


def aggregate(rows, run):
    trades = []
    for row in rows:
        trades.extend(read(run / "accounts" / row["id"] / "trades.json"))
    positive = math.fsum(max(t["net_pnl"], 0) for t in trades)
    negative = -math.fsum(min(t["net_pnl"], 0) for t in trades)
    m = [r["metrics"] for r in rows]
    returns = [x["net_return"] for x in m]
    sharpes = [x["sharpe"] for x in m if x["sharpe"] is not None]
    return {
        "windows": len(rows),
        "positive_windows": sum(r > 0 for r in returns),
        "mean_window_return": statistics.mean(returns),
        "median_window_return": statistics.median(returns),
        "minimum_window_return": min(returns),
        "worst_observed_drawdown": max(x["max_drawdown"] for x in m),
        "median_window_sharpe": statistics.median(sharpes) if sharpes else None,
        "complete_trades": len(trades),
        "complete_trade_win_rate": sum(t["net_pnl"] > 0 for t in trades) / len(trades)
        if trades
        else None,
        "pooled_complete_profit_factor": positive / negative if negative else None,
        "closed_net_pnl": math.fsum(t["net_pnl"] for t in trades),
        "mean_beta": statistics.mean(x["beta"] for x in m if x["beta"] is not None),
        "mean_turnover": statistics.mean(x["turnover"] for x in m),
        "mean_open_pnl_with_exit_reserve": statistics.mean(
            x["open_pnl_with_exit_reserve"] for x in m
        ),
        "mean_fees": statistics.mean(x["fees"] for x in m),
        "mean_slippage_cost": statistics.mean(x["slippage_cost"] for x in m),
        "scope": "equal-size independent cold-start windows; arithmetic mean, not compounded or annualized; pooled trade counts are descriptive, not independent observations",
    }


def compare(parent):
    c = read(parent / "catalog.json")
    rows = c["accounts"]
    groups = []
    for cost in (3, 10):
        for preset in ("full", "technical", "training_factor8", "long_only"):
            selected = [
                r
                for r in rows
                if r["phase"] != "capacity" and r["preset"] == preset and r["slippage_bps"] == cost
            ]
            groups.append(
                {
                    "preset": preset,
                    "total_cost_bps_per_side": 5 + cost,
                    **aggregate(selected, parent),
                }
            )
    factors = []
    for f in c["factors"]:
        a, b = f.get("training_rank_ic"), f.get("validation_rank_ic")
        if a is not None and b is not None and a * b > 0:
            factors.append(
                {
                    "id": f["id"],
                    "name": f["name"],
                    "family": f["family"],
                    "training_rank_ic": a,
                    "validation_rank_ic": b,
                    "direction": 1 if a > 0 else -1,
                    "coverage": f["validation_coverage"],
                    "evidence": "same-sign train/2025validation association, not single-factor account alpha",
                }
            )
    factors.sort(key=lambda x: -abs(x["validation_rank_ic"]))
    # The technical/full models differ by macro/news and predeclared news interactions.
    ablation = []
    for w in c["windows"]:
        full = next(
            r
            for r in rows
            if r["window"] == w["id"]
            and r["phase"] != "capacity"
            and r["preset"] == "full"
            and r["slippage_bps"] == 3
        )
        technical = next(
            r
            for r in rows
            if r["window"] == w["id"]
            and r["phase"] != "capacity"
            and r["preset"] == "technical"
            and r["slippage_bps"] == 3
        )
        pf = next(
            r
            for r in c["prediction_evaluations"]
            if r["window"] == w["id"] and r["model"] == "full_24h"
        )
        pt = next(
            r
            for r in c["prediction_evaluations"]
            if r["window"] == w["id"] and r["model"] == "technical_24h"
        )
        ablation.append(
            {
                "window": w["id"],
                "net_return_increment": full["metrics"]["net_return"]
                - technical["metrics"]["net_return"],
                "mse_relative_improvement": 1 - pf["mse"] / pt["mse"],
                "rank_ic_increment": None
                if pf["rank_ic"] is None or pt["rank_ic"] is None
                else pf["rank_ic"] - pt["rank_ic"],
            }
        )
    return {
        "schema": "factor-lab-comparison.v1",
        "groups": groups,
        "windows": c["windows"],
        "account_rows": rows,
        "stable_direction_factors": factors,
        "macro_news_ablation": ablation,
        "selection_diagnostics": "full model has strongest measured potential among active strategies, but stress-cost average return and completed-trade economics are not robust",
        "limitations": c["limitations"],
        "production_eligible": False,
        "orders_enabled": False,
    }
