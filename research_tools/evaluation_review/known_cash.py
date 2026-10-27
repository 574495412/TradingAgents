"""Hand-computable synthetic controls, never market profitability evidence."""

import importlib.util
import sys
from decimal import Decimal
from pathlib import Path

import pandas as pd


def load_engine():
    root = Path(__file__).resolve().parents[3] / "freqtrade"
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location(
        "review_sparse_calendar_v2", root / "research/crypto/sparse_calendar_v2.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs(side, exit_reference, slippage_bps, funding_rate="0"):
    clock = pd.date_range(
        "2024-12-31 23:00", "2025-01-01 02:00", freq="5min", inclusive="left", tz="UTC"
    )
    prices = pd.Series(100.0, index=clock)
    prices.loc["2025-01-01T00:10Z":] = float(exit_reference)
    bars = {
        "X": pd.DataFrame(
            {
                "open": prices,
                "high": prices,
                "low": prices,
                "close": prices,
                "quote_volume": 1e8,
                "price_usable": True,
                "execution_liquidity_usable": True,
            }
        )
    }
    funding = {
        "X": pd.DataFrame(
            {"funding_rate": [float(funding_rate)], "mark": [100.0]},
            index=pd.to_datetime(["2025-01-01T00:30Z"], utc=True),
        )
    }
    hours = pd.date_range("2025-01-01", "2025-01-01 02:00", freq="h", inclusive="left", tz="UTC")
    scores = pd.DataFrame(
        {"target_X": [0.5 * side, 0.0], "risk_scale": 1.0, "rebalance": True}, index=hours
    )
    config = {
        "capital": 10000,
        "fee_bps": 5,
        "slippage_bps": slippage_bps,
        "participation": 0.01,
        "gross": 0.8,
        "max_positions": 1,
        "rebalance_hours": 24,
        "threshold": 0.0032,
        "stop_loss": 0.08,
        "drawdown_halt": 0.15,
    }
    return bars, funding, scores, config


def expected_cash(side, exit_reference, slippage_bps, funding_rate="0"):
    # Desired quantity is exactly 0.5 * 0.8 * 10000 / 100 = 40.
    # It is below the independent gross limit, so fee reserves cannot resize it.
    d = Decimal(side)
    q = Decimal(40)
    slip = Decimal(slippage_bps) / 10000
    fee = Decimal(5) / 10000
    entry = Decimal(100) * (1 + d * slip)
    exit_price = Decimal(str(exit_reference)) * (1 - d * slip)
    price_pnl = d * q * (exit_price - entry)
    commission = q * (entry + exit_price) * fee
    funding = -d * q * Decimal(100) * Decimal(funding_rate)
    return {
        "quantity": q,
        "entry_price": entry,
        "exit_price": exit_price,
        "commission": commission,
        "funding_cash": funding,
        "closed_cash": Decimal(10000) + price_pnl - commission + funding,
    }


def run_case(engine, side, exit_reference, slippage_bps, funding_rate="0"):
    b, f, s, c = inputs(side, exit_reference, slippage_bps, funding_rate)
    account = engine.replay(b, f, s, c, "2025-01-01", "2025-01-01 02:00")
    return account, expected_cash(side, exit_reference, slippage_bps, funding_rate)


def halted_recovery_case(engine):
    b, f, s, c = inputs(1, 50, 3)
    b["X"].loc["2025-01-01T00:20Z":, ["open", "high", "low", "close"]] = 120.0
    s["target_X"] = 0.5
    return engine.replay(b, f, s, c, "2025-01-01", "2025-01-01 02:00")
