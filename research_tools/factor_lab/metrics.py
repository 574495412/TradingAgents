"""Explicit metric conventions: crypto365, daily marked NAV, zero risk-free rate."""

import math

import numpy as np
import pandas as pd

METRIC_DEFINITIONS = {
    "benchmark_return": "BTC completed5m price return; excludes fees/funding and is not an executable benchmark account",
    "net_return": "ending marked equity /initial -1, including accrued funding and reserved exit costs",
    "cagr": "(ending/initial)^(365/calendar_days)-1; short-window extrapolation, not forecast",
    "sharpe": "mean UTC daily NAV return /sample standard deviation *sqrt365; risk-free0",
    "sortino": "mean daily NAV return /RMS(min(return,0))*sqrt365; threshold0",
    "calmar": "CAGR /maximum observed drawdown",
    "max_drawdown": "maximum1-NAV/running NAV peak; sampled5m closes, not intrabar worst case",
    "recovery_days": "longest calendar duration below a previous sampled equity high, includes unrecovered tail",
    "profit_factor": "sum positive COMPLETE lifecycle PnL /absolute sum negative COMPLETE lifecycle PnL",
    "win_rate": "positive COMPLETE lifecycle PnL /complete lifecycle count; open trades excluded",
    "payoff_ratio": "mean complete winner /absolute mean complete loser",
    "expectancy": "mean net PnL per COMPLETE lifecycle, USDT",
    "turnover": "sum absolute filled notional /mean sampled equity over the period",
    "beta": "sample daily covariance(strategy,BTC buy-hold)/sample BTC return variance",
    "alpha_annual": "365*(mean strategy daily return-beta*mean benchmark daily return); descriptive regression intercept",
    "information_ratio": "mean daily strategy-minus-benchmark /sample tracking error *sqrt365",
    "tracking_error": "sample daily standard deviation of strategy-minus-benchmark *sqrt365",
    "var95_daily": "negative5th percentile UTC daily return, floored0; historical",
    "cvar95_daily": "negative mean daily returns at/below5th percentile, floored0; historical",
    "rank_ic": "mean same-clock Spearman correlation of factor/prediction and forward labels across six assets",
    "ic": "mean same-clock Pearson correlation across available assets, minimum4 and nonconstant inputs",
    "ic_ir": "mean clock-level IC /sample std; not annualized and not proof of independent samples",
    "direction_accuracy": "fraction sign(prediction)==sign(forward label); excludes zero actual labels",
    "auc": "rank-U area under ROC using continuous prediction and binary positive-return label; ties averaged",
    "mse": "mean squared raw forward return error; includes baseline-zero comparison",
    "mae": "mean absolute raw forward return error",
}


def ratio(a, b):
    return float(a / b) if b is not None and np.isfinite(b) and abs(b) > 1e-15 else None


def prediction_metrics(score, label):
    joined = pd.concat([score.rename("score"), label.rename("label")], axis=1).dropna()
    if not len(joined):
        return {"rows": 0, "mse": None, "rank_ic": None, "direction_accuracy": None}
    sx = joined.score.unstack("symbol")
    sy = joined.label.unstack("symbol")

    def correlations(a, b):
        valid = a.notna() & b.notna()
        a = a.where(valid)
        b = b.where(valid)
        ac = a.sub(a.mean(axis=1), axis=0)
        bc = b.sub(b.mean(axis=1), axis=0)
        denominator = np.sqrt(ac.pow(2).sum(axis=1) * bc.pow(2).sum(axis=1))
        value = (ac * bc).sum(axis=1) / denominator.replace(0, np.nan)
        return value.loc[valid.sum(axis=1) >= 4].dropna().to_numpy()

    ic = correlations(sx, sy)
    rank = correlations(sx.rank(axis=1, method="average"), sy.rank(axis=1, method="average"))
    actual = joined.label.ne(0)
    pos = joined.label.gt(0)
    n1 = int(pos.sum())
    n0 = len(pos) - n1
    ranks = joined.score.rank(method="average")
    auc = ratio(float(ranks[pos].sum()) - n1 * (n1 + 1) / 2, n1 * n0)
    return {
        "rows": len(joined),
        "mse": float(((joined.score - joined.label) ** 2).mean()),
        "mae": float((joined.score - joined.label).abs().mean()),
        "zero_baseline_mse": float(joined.label.pow(2).mean()),
        "ic": float(np.mean(ic)) if len(ic) else None,
        "rank_ic": float(np.mean(rank)) if len(rank) else None,
        "ic_ir": ratio(np.mean(ic), np.std(ic, ddof=1)) if len(ic) > 1 else None,
        "direction_accuracy": float(
            (np.sign(joined.score[actual]) == np.sign(joined.label[actual])).mean()
        ),
        "auc": auc,
        "ic_clock_count": len(ic),
    }


def portfolio_metrics(curve, trades, events, initial, benchmark):
    nav = curve.set_index("date").equity
    daily = nav.loc[(nav.index.hour == 0) & (nav.index.minute == 0)]
    dr = daily.pct_change(fill_method=None).dropna()
    days = (nav.index[-1] - nav.index[0]).total_seconds() / 86400
    net = float(nav.iloc[-1] / initial - 1)
    cagr = (nav.iloc[-1] / initial) ** (365 / days) - 1 if nav.iloc[-1] > 0 and days > 0 else None
    dd = 1 - nav / nav.cummax()
    downside = np.sqrt(np.mean(np.minimum(dr, 0) ** 2))
    benchmark = benchmark.reindex(daily.index)
    br = benchmark.pct_change(fill_method=None).dropna()
    pair = pd.concat([dr.rename("strategy"), br.rename("benchmark")], axis=1).dropna()
    beta = ratio(pair.strategy.cov(pair.benchmark), pair.benchmark.var()) if len(pair) > 1 else None
    active = pair.strategy - pair.benchmark
    longest = 0.0
    peak = nav.iloc[0]
    peak_clock = nav.index[0]
    for clock, value in nav.items():
        if value >= peak:
            peak = value
            peak_clock = clock
        else:
            longest = max(longest, (clock - peak_clock).total_seconds() / 86400)
    pnl = np.array([t["net_pnl"] for t in trades])
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    quantile = float(dr.quantile(0.05)) if len(dr) else None
    below = dr.loc[dr <= quantile] if quantile is not None else []
    return {
        "net_return": net,
        "net_pnl": float(nav.iloc[-1] - initial),
        "ending_equity": float(nav.iloc[-1]),
        "cagr": float(cagr) if cagr is not None else None,
        "calendar_days": days,
        "annual_volatility": float(dr.std(ddof=1) * math.sqrt(365)) if len(dr) > 1 else None,
        "sharpe": ratio(dr.mean() * math.sqrt(365), dr.std(ddof=1)),
        "sortino": ratio(dr.mean() * math.sqrt(365), downside),
        "calmar": ratio(cagr, float(dd.max())) if cagr is not None else None,
        "max_drawdown": float(dd.max()),
        "recovery_days": longest,
        "closed_trades": len(pnl),
        "win_rate": float(np.mean(pnl > 0)) if len(pnl) else None,
        "profit_factor": ratio(float(wins.sum()), float(-losses.sum())),
        "payoff_ratio": ratio(float(wins.mean()), float(-losses.mean()))
        if len(wins) and len(losses)
        else None,
        "expectancy": float(pnl.mean()) if len(pnl) else None,
        "closed_net_pnl": float(pnl.sum()),
        "beta": beta,
        "alpha_annual": float(365 * (pair.strategy.mean() - beta * pair.benchmark.mean()))
        if beta is not None
        else None,
        "information_ratio": ratio(active.mean() * math.sqrt(365), active.std(ddof=1)),
        "tracking_error": float(active.std(ddof=1) * math.sqrt(365)) if len(active) > 1 else None,
        "benchmark_return": float(benchmark.iloc[-1] / benchmark.iloc[0] - 1),
        "var95_daily": max(0.0, -quantile) if quantile is not None else None,
        "cvar95_daily": max(0.0, -float(np.mean(below))) if len(below) else None,
        "fees": math.fsum(e.get("fee", 0) for e in events),
        "funding_net": math.fsum(e["cash_delta"] for e in events if e["type"] == "funding"),
        "slippage_cost": math.fsum(e.get("slippage_cost", 0) for e in events),
        "turnover": math.fsum(e.get("notional", 0) for e in events) / float(nav.mean()),
        "mean_gross_exposure": float((curve.gross_exposure / curve.equity).mean()),
        "max_gross_exposure": float((curve.gross_exposure / curve.equity).max()),
        "open_pnl_with_exit_reserve": float(nav.iloc[-1] - curve.cash.iloc[-1]),
        "sampled_return_frequency": "UTC daily,365; short representative windows, zero risk-free rate",
    }
