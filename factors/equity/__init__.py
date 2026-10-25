"""Equity adapter for the shared market-calendar-neutral factor engine."""


def calculate(market, events=None, benchmark_symbol=None):
    from factors.common.panel import calculate_market

    return calculate_market(market, events, asset_class="equity", benchmark_symbol=benchmark_symbol)
