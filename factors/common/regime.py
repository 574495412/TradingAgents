"""Same-clock market-state conditioning, without future cross-sectional information."""

REGIME_COLUMNS = ["benchmark_trend", "benchmark_mom7", "market_breadth", "relative_mom7"]


def condition_market(x, benchmark="BTCUSDT"):
    f = x.copy()
    b = x.xs(benchmark, level="symbol")
    clock = x.index.get_level_values("time")
    breadth = x.trend_consensus.gt(0).where(x.trend_consensus.notna()).groupby(level="time").mean()
    f["benchmark_trend"] = b.trend_consensus.reindex(clock).to_numpy()
    f["benchmark_mom7"] = b.mom_z168.reindex(clock).to_numpy()
    f["market_breadth"] = breadth.reindex(clock).to_numpy()
    f["relative_mom7"] = f.mom_z168 - f.benchmark_mom7
    return f
