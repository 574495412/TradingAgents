"""Stable metadata for market-only factors used by Weijie.

The catalog is intentionally free of strategy state, labels, symbol allowlists,
and execution code.  Live logging, backtests, reports, and factor diagnostics
can therefore share the same keys without importing one another.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MarketFactor:
    """One interpretable market-input family used by live and backtest."""

    key: str
    columns: tuple[str, ...]
    purpose: str


@dataclass(frozen=True)
class SignalRoute:
    """One asset-scoped lifecycle that consumes the market factors."""

    key: str
    symbols: tuple[str, ...]
    timeframes: tuple[str, ...]
    mechanisms: tuple[str, ...]


V237_MARKET_FACTORS = (
    MarketFactor(
        key="trend",
        columns=(
            "ema21_slope3_pct",
            "tf15_ret16",
            "tf1h_ema21_slope3_pct",
            "tf1h_ret4",
            "tf1h_ret12",
            "adx14",
        ),
        purpose="Separate continuation, pullback, and exhausted trend regimes.",
    ),
    MarketFactor(
        key="momentum",
        columns=(
            "ret1",
            "ret3",
            "ret20",
            "ret60",
            "ret144",
            "ret288",
            "tf15_ret4",
            "rsi14",
            "tf1h_rsi14",
        ),
        purpose="Measure impulse, reversal, continuation, and momentum decay.",
    ),
    MarketFactor(
        key="volatility",
        columns=("atr14_pct", "ret1_atr", "ret3_atr", "range_atr"),
        purpose="Normalize moves across symbols and volatility regimes.",
    ),
    MarketFactor(
        key="price_structure",
        columns=("donchian20_pos", "donchian60_pos", "close_pos"),
        purpose="Identify breakouts, pullbacks, rejection, and range location.",
    ),
    MarketFactor(
        key="liquidity_flow",
        columns=(
            "quote_volume_ratio_med30",
            "quote_volume_rank",
            "taker_buy_ratio",
            "trades",
        ),
        purpose="Distinguish liquid confirmation, absorption, and seller control.",
    ),
    MarketFactor(
        key="cross_sectional",
        columns=(
            "ret20_rank",
            "ret60_rank",
            "market_up_breadth20",
            "market_down_breadth20",
        ),
        purpose="Measure relative leadership, lag, and market-wide participation.",
    ),
    MarketFactor(
        key="market_regime",
        columns=(
            "trend_strength",
            "liquidity_condition",
            "crowding_score",
            "reversal_probability",
            "market_regime_score",
        ),
        purpose="Replace symbol-and-clock filters with one reusable continuation regime.",
    ),
)


V237_SIGNAL_ROUTES = (
    SignalRoute(
        key="market_regime_engine",
        symbols=(
            "ETHUSDT",
            "BCHUSDT",
            "CFXUSDT",
            "DOGEUSDT",
            "ETCUSDT",
            "FILUSDT",
            "LTCUSDT",
            "SOLUSDT",
            "PAXGUSDT",
            "SKHYNIXUSDT",
            "SNDKUSDT",
        ),
        timeframes=("5m", "15m", "1h"),
        mechanisms=(
            "trending",
            "exhaustion",
            "liquidation",
            "accumulation",
            "recovery",
        ),
    ),
)

# Cross-asset campaigns use dimensionless inputs so one definition can be
# evaluated on crypto, commodity, and exchange-listed TradFi contracts.  Each
# entry documents the market hypothesis rather than a fitted symbol or date.
CROSS_ASSET_MARKET_FACTORS = (
    MarketFactor(
        key="cross_asset_market_structure",
        columns=(
            "normalized_return_3",
            "normalized_return_12",
            "normalized_return_36",
            "normalized_return_144",
            "normalized_return_288",
            "range_location_288",
        ),
        purpose=(
            "Describe impulse, exhaustion, recovery, and range location with "
            "scale-free returns that transfer across contract price levels."
        ),
    ),
    MarketFactor(
        key="cross_asset_group_breadth",
        columns=(
            "group_mean_return_36",
            "group_mean_return_144",
            "negative_return_breadth",
            "group_size",
        ),
        purpose=(
            "Require sector participation so a single noisy contract cannot "
            "masquerade as a reusable market regime."
        ),
    ),
    MarketFactor(
        key="cross_asset_liquidity_flow",
        columns=(
            "rolling_quote_volume_24h",
            "quote_volume_ratio",
            "signed_seller_pressure",
            "taker_buy_sell_ratio_1h",
            "taker_buy_sell_ratio_3h",
        ),
        purpose=(
            "Separate liquid confirmation, absorption, and crowded flow from "
            "moves that are too thin to execute consistently."
        ),
    ),
    MarketFactor(
        key="cross_asset_positioning",
        columns=(
            "oi_change_3h_pct",
            "oi_change_12h_pct",
            "oi_change_24h_pct",
        ),
        purpose=(
            "Distinguish leverage expansion from deleveraging using only "
            "closed exchange positioning observations."
        ),
    ),
    MarketFactor(
        key="cross_asset_volatility_normalization",
        columns=("atr14_pct",),
        purpose=(
            "Express entries, adds, targets, and hard stops in volatility "
            "units so lifecycle distances remain reusable across symbols."
        ),
    ),
    MarketFactor(
        key="cross_asset_lifecycle",
        columns=(
            "entry_atr_pct",
            "holding_bars",
            "add_count",
            "best_favorable_price",
        ),
        purpose=(
            "Record strategy-owned inventory state used for adds, reductions, "
            "targets, stops, and time exits; it is a control, not a label."
        ),
    ),
)


NON_FACTOR_CONTROLS = (
    "closed_bar_only",
    "strategy_owned_position_state",
    "reentry_cooldown",
    "factor_specific_margin_budget",
    "shared_account_margin_cap",
    "leverage_and_liquidation_guard",
    "exchange_stop_protection",
    "market_data_coverage_gate",
)


def factor_keys() -> tuple[str, ...]:
    """Return stable factor keys for operator displays and tests."""

    return tuple(factor.key for factor in V237_MARKET_FACTORS)


def cross_asset_factor_keys(*, operation_type: str) -> tuple[str, ...]:
    """Return stable factor provenance for one cross-asset order.

    Opening and adding orders are caused by market observations plus
    volatility-normalized sizing.  Reduction and closing orders are caused by
    the strategy-owned lifecycle.  The split keeps order audits honest and
    avoids claiming that every available factor triggered every order.
    """

    operation = str(operation_type).upper()
    if operation in {"OPEN", "ADD"}:
        return tuple(
            factor.key
            for factor in CROSS_ASSET_MARKET_FACTORS
            if factor.key != "cross_asset_lifecycle"
        )
    if operation in {"REDUCE", "CLOSE"}:
        return (
            "cross_asset_volatility_normalization",
            "cross_asset_lifecycle",
        )
    message = f"unsupported cross-asset operation type: {operation}"
    raise ValueError(message)
