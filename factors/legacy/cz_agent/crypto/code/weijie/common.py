"""Shared factor helpers for Weijie autonomous research."""

from __future__ import annotations

import numpy as np
import pandas as pd

PRECIOUS_METAL_BASE_ASSETS = frozenset(
    {
        "GOLD",
        "PAXG",
        "XAG",
        "XAU",
        "XAUT",
    },
)

TOKENIZED_EQUITY_BASE_ASSETS = frozenset(
    {
        "AAPL",
        "AMZN",
        "COIN",
        "GOOG",
        "GOOGL",
        "HOOD",
        "META",
        "MSFT",
        "MSTR",
        "NDX",
        "NVDA",
        "QQQ",
        "SKHYNIX",
        "SNDK",
        "SPX",
        "SPY",
        "TSLA",
    },
)

STOCK_GOLD_BASE_ASSETS = PRECIOUS_METAL_BASE_ASSETS | TOKENIZED_EQUITY_BASE_ASSETS


def clean_symbol(value: object) -> str:
    return str(value or "").upper().replace("/", "").replace("-", "").strip()


def weijie_asset_group(value: object) -> str:
    """Map the tradable universe into the three leader-style asset classes."""

    symbol = clean_symbol(value)
    if symbol == "ETHUSDT":
        return "eth"
    base_asset = symbol
    for quote_asset in ("USDT", "USDC", "USD"):
        if base_asset.endswith(quote_asset):
            base_asset = base_asset[: -len(quote_asset)]
            break
    if base_asset in STOCK_GOLD_BASE_ASSETS:
        return "stock_gold"
    return "meme"


def weijie_market_prior_group(value: object) -> str:
    """Map symbols to reusable market-behavior groups for learned priors."""

    symbol = clean_symbol(value)
    base_asset = symbol
    for quote_asset in ("USDT", "USDC", "USD"):
        if base_asset.endswith(quote_asset):
            base_asset = base_asset[: -len(quote_asset)]
            break
    if base_asset in TOKENIZED_EQUITY_BASE_ASSETS:
        return "tradfi_equity"
    if base_asset in PRECIOUS_METAL_BASE_ASSETS:
        return "precious_metal_defensive"
    if base_asset in {"BTC", "ETH", "SOL", "BNB", "BCH", "LTC"}:
        return "large_cap_momentum"
    if base_asset in {"DOGE", "SHIB", "PEPE", "BONK", "WIF", "FLOKI"}:
        return "meme_momentum"
    return "liquid_alt_momentum"


def direction_from_action(action: object) -> str:
    text = str(action or "").upper().strip()
    if text.endswith("LONG"):
        return "LONG"
    if text.endswith("SHORT"):
        return "SHORT"
    return ""


def finite_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return out


def pct_change(current: pd.Series, previous: pd.Series) -> pd.Series:
    current_num = pd.to_numeric(current, errors="coerce")
    previous_num = pd.to_numeric(previous, errors="coerce").replace(0.0, np.nan)
    return ((current_num / previous_num) - 1.0) * 100.0


def shanghai_timestamp_ms(ts: pd.Timestamp) -> int:
    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("Asia/Shanghai")
    else:
        stamp = stamp.tz_convert("Asia/Shanghai")
    return int(stamp.tz_convert("UTC").timestamp() * 1000)
