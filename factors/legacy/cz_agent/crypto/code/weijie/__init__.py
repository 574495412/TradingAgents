"""Reusable causal factor modules preserved from the retired strategy."""

from factors.crypto.code.weijie.live_factor_catalog import (
    CROSS_ASSET_MARKET_FACTORS,
    MarketFactor,
    cross_asset_factor_keys,
)
from factors.crypto.code.weijie.signal_templates import (
    build_close_factor_signals,
    build_open_factor_signals,
)

__all__ = [
    "CROSS_ASSET_MARKET_FACTORS",
    "MarketFactor",
    "build_close_factor_signals",
    "build_open_factor_signals",
    "cross_asset_factor_keys",
]
