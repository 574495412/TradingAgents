"""Crypto adapter; importing legacy formulas does not load optional panel dependencies."""


def calculate(market, events=None, funding=None, benchmark_symbol="BTCUSDT"):
    from factors.common.panel import calculate_market

    return calculate_market(market, events, funding, "crypto", benchmark_symbol)
