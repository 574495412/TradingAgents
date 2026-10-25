"""Causal daily cash accounts with board lots, next-open fills and dividend receivables."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def signals(frame, candidate, legacy=None):
    """Close decisions; caller only executes the previous session's decision."""
    close = frame.close
    if candidate == "trend_20_60":
        fast, slow = close.rolling(20).mean(), close.rolling(60).mean()
        return pd.DataFrame({"enter": fast > slow, "exit": fast <= slow}, index=frame.index)
    if candidate == "rsi2_recovery":
        delta = close.diff()
        up = delta.clip(lower=0).ewm(alpha=0.5, adjust=False, min_periods=2).mean()
        down = (-delta.clip(upper=0)).ewm(alpha=0.5, adjust=False, min_periods=2).mean()
        rsi = 100 * up / (up + down)
        return pd.DataFrame(
            {
                "enter": (close > close.rolling(200).mean()) & (rsi < 10),
                "exit": (rsi > 60) | (close < close.rolling(200).mean()),
            },
            index=frame.index,
        )
    if candidate == "legacy_ema_daily":
        score = legacy(frame).copy()
        # The old implementation checked total frame length, so explicitly mask
        # its first 79 rows to make prefix and full-history decisions identical.
        score.iloc[:79] = 0.0
        return pd.DataFrame({"enter": score >= 0.5, "exit": score <= 0}, index=frame.index)
    if candidate == "buy_hold":
        return pd.DataFrame({"enter": True, "exit": False}, index=frame.index)
    if candidate == "cash":
        return pd.DataFrame({"enter": False, "exit": True}, index=frame.index)
    raise ValueError("unknown frozen candidate")


def account(frames, calendar, decisions, spec, *, start, end, cost_bps):
    """One market/currency, long-only, settled-cash proxy, no leverage or same-bar fill.

    Dividends become nonspendable receivables on ex-date; payment dates are unknown.
    This conservatively forbids reinvesting those amounts for the entire study.
    """
    initial = float(spec["initial_cash"])
    rate = cost_bps / 10000
    if not math.isfinite(rate) or rate < 0 or initial <= 0:
        raise ValueError("invalid cash or costs")
    minimum = spec["minimum_fee"]
    if not math.isfinite(minimum) or minimum < 0 or not 0 <= spec["dividend_haircut"] <= 1:
        raise ValueError("invalid minimum cost or dividend haircut")
    specs = {s["symbol"]: s for s in spec["symbols"]}
    if not specs or set(specs) != set(frames) or set(frames) != set(decisions):
        raise ValueError("account universe mismatch")
    if any(
        isinstance(s["lot_size"], bool) or not isinstance(s["lot_size"], int) or s["lot_size"] <= 0
        for s in specs.values()
    ):
        raise ValueError("board lots must be positive integers")
    if any(not f.index.as_unit("ns").equals(calendar.index.as_unit("ns")) for f in frames.values()):
        raise ValueError("cannot skip missing account marks")
    for symbol, decision in decisions.items():
        if (
            not decision.index.equals(calendar.index)
            or decision[["enter", "exit"]].isna().any().any()
        ):
            raise ValueError(f"unknown decision: {symbol}")
    window = calendar.loc[
        (calendar.index >= pd.Timestamp(start)) & (calendar.index < pd.Timestamp(end))
    ]
    if window.empty:
        raise ValueError("empty account window")
    cash, receivables, fees = initial, 0.0, 0.0
    positions, trades, curve, fills = {}, [], [], []
    peak, max_risk_dd, max_close_dd = initial, 0.0, 0.0

    def fee(quantity, price):
        return max(minimum, quantity * price * rate)

    def close(symbol, price, stamp, reason):
        nonlocal cash, fees
        p = positions.pop(symbol)
        charge = fee(p["quantity"], price)
        cash += p["quantity"] * price - charge
        fees += charge
        trade = {
            **p,
            "symbol": symbol,
            "direction": 1,
            "exit_ms": stamp,
            "exit_price": price,
            "exit_fee": charge,
            "reason": reason,
            "net_pnl": p["quantity"] * (price - p["entry_price"])
            - p["entry_fee"]
            - charge
            + p["dividends"],
        }
        trades.append(trade)
        fills.append(
            {
                "symbol": symbol,
                "clock": stamp,
                "side": "SELL",
                "quantity": p["quantity"],
                "price": price,
                "fee": charge,
            }
        )

    first_open = int(window.iloc[0]["open"].timestamp() * 1000)
    curve.append({"clock": first_open - 1, "equity": initial, "drawdown": 0.0})
    for day, session in window.iterrows():
        index = calendar.index.get_loc(day)
        opened = int(session["open"].timestamp() * 1000)
        closed = int(session["close"].timestamp() * 1000)
        # Entitlement belongs to shares held before today's open, including
        # those sold at today's open; an ex-date buyer receives no distribution.
        for symbol, p in positions.items():
            dividend = (
                p["quantity"] * frames[symbol].at[day, "dividend"] * (1 - spec["dividend_haircut"])
            )
            receivables += dividend
            p["dividends"] += dividend
        prior = calendar.index[index - 1] if index else None
        if prior is not None:
            for symbol in list(positions):
                if bool(decisions[symbol].at[prior, "exit"]):
                    close(
                        symbol, float(frames[symbol].at[day, "open"]), opened, "previous_close_exit"
                    )
            # Stable universe order; fixed equal capital sleeves cannot use
            # future returns or borrow the other sleeve's unused allocation.
            for symbol, symbol_spec in specs.items():
                if symbol in positions or not bool(decisions[symbol].at[prior, "enter"]):
                    continue
                row = frames[symbol].loc[day]
                price, lot = float(row.open), symbol_spec["lot_size"]
                budget = min(cash, initial / len(specs))
                quantity = math.floor(budget / (price * lot * (1 + rate))) * lot
                while quantity > 0 and quantity * price + fee(quantity, price) > budget + 1e-9:
                    quantity -= lot
                if quantity <= 0:
                    continue
                charge = fee(quantity, price)
                cash -= quantity * price + charge
                fees += charge
                positions[symbol] = {
                    "quantity": quantity,
                    "entry_ms": opened,
                    "entry_price": price,
                    "entry_fee": charge,
                    "dividends": 0.0,
                }
                fills.append(
                    {
                        "symbol": symbol,
                        "clock": opened,
                        "side": "BUY",
                        "quantity": quantity,
                        "price": price,
                        "fee": charge,
                    }
                )
        equity = (
            cash
            + receivables
            + sum(p["quantity"] * frames[s].at[day, "close"] for s, p in positions.items())
        )
        # The sum of every held symbol's daily low is a conservative envelope,
        # not a synchronized intraday portfolio observation or executable price.
        low_equity = (
            cash
            + receivables
            + sum(p["quantity"] * frames[s].at[day, "low"] for s, p in positions.items())
        )
        max_risk_dd = max(max_risk_dd, 1 - low_equity / peak)
        peak = max(peak, equity)
        close_dd = 1 - equity / peak
        max_close_dd = max(max_close_dd, close_dd)
        curve.append({"clock": closed, "equity": equity, "drawdown": close_dd})
    # Mark remaining shares to close, retaining unrealized PnL. Never force a
    # retrospective closing-auction fill just because the sample ends.
    ending = curve[-1]["equity"]
    net = np.array([t["net_pnl"] for t in trades], dtype=float)
    gross_profit, gross_loss = net[net > 0].sum(), -net[net < 0].sum()
    daily = pd.Series([r["equity"] for r in curve]).pct_change().dropna()
    std = daily.std(ddof=1)
    downside = np.sqrt(np.mean(np.minimum(daily, 0) ** 2))
    years = (window.index[-1] - window.index[0]).days / 365.25
    annual = (ending / initial) ** (1 / years) - 1 if years > 0 and ending > 0 else None
    metrics = {
        "net_return": ending / initial - 1,
        "net_pnl": ending - initial,
        "realized_closed_net_pnl": float(net.sum()),
        "ending_equity": ending,
        "win_rate": float((net > 0).mean()) if len(net) else None,
        "closed_trades": len(trades),
        "open_positions": len(positions),
        "max_drawdown": max_risk_dd,
        "close_max_drawdown": max_close_dd,
        "profit_factor": float(gross_profit / gross_loss) if gross_loss else None,
        "profit_factor_status": "finite"
        if gross_loss
        else "no_closed_losses"
        if len(net)
        else "no_closed_trades",
        "fees": fees,
        "cash": cash,
        "dividend_receivables": receivables,
        "sharpe_zero_rf_252": float(daily.mean() / std * np.sqrt(252)) if std > 0 else None,
        "sortino_zero_rf_252": float(daily.mean() / downside * np.sqrt(252))
        if downside > 0
        else None,
        "calmar_close": annual / max_close_dd if annual is not None and max_close_dd > 0 else None,
        "annualized_return": annual,
        "mean_closed_pnl": float(net.mean()) if len(net) else None,
        "trades_per_session": len(net) / len(window),
        "sessions": len(window),
        "unrealized_pnl": ending - initial - float(net.sum()),
    }
    assert cash >= -1e-7
    return {
        "metrics": metrics,
        "equity_curve": curve,
        "trades": trades,
        "fills": fills,
        "positions": positions,
    }
