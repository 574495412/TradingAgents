"""Causal trend mechanisms with actual aggressor volume where the feed supplies it.

OHLCV and aggressor fields of a bar starting at t are first usable at t+1h.
This module does not change the frozen74-column panel used by older studies.
"""

import numpy as np
import pandas as pd

PRICE_COLUMNS = [
    "trend_consensus",
    "mom_z168",
    "mom_z504",
    "mom_z1512",
    "efficiency168",
    "channel480",
    "return_z24",
    "vol24",
    "vol168",
    "volume_impulse24",
    "atr24",
]
FLOW_COLUMNS = [
    "aggressor24",
    "aggressor72",
    "aggressor_surprise24",
    "absorption24",
    "flow_trend_agreement",
    "funding_last",
]
RULES = [
    "slow_trend",
    "flow_trend",
    "breakout_acceptance",
    "pullback_continuation",
    "absorption_release",
    "funding_trend",
]
CATALOG = [
    {
        "id": "trend_consensus",
        "name": "多尺度波动归一趋势",
        "definition": "mean tanh(log(close/lag_n)/(std168(log-return)*sqrt(n))), n168/504/1512; completed hourly bars",
        "asset_classes": ["crypto", "equity"],
    },
    {
        "id": "aggressor24",
        "name": "真实主动买卖净额24期",
        "definition": "sum24(2*taker_buy_quote_volume-quote_volume)/sum24(quote_volume)",
        "asset_classes": ["crypto", "equity_with_aggressor_feed"],
    },
    {
        "id": "aggressor72",
        "name": "真实主动买卖净额72期",
        "definition": "sum72(2*taker_buy_quote_volume-quote_volume)/sum72(quote_volume)",
        "asset_classes": ["crypto", "equity_with_aggressor_feed"],
    },
    {
        "id": "aggressor_surprise24",
        "name": "主动成交异常",
        "definition": "(aggressor24-prior168mean)/prior168sample_std; clipped +/-5",
        "asset_classes": ["crypto", "equity_with_aggressor_feed"],
    },
    {
        "id": "absorption24",
        "name": "资金流与价格响应偏离",
        "definition": "aggressor_surprise24-return_z24; positive flow without commensurate price response is only an absorption proxy",
        "asset_classes": ["crypto", "equity_with_aggressor_feed"],
    },
    {
        "id": "flow_trend_agreement",
        "name": "主动资金流趋势确认",
        "definition": "trend_consensus * clip(aggressor_surprise24,-3,3); interaction, not independent alpha",
        "asset_classes": ["crypto", "equity_with_aggressor_feed"],
    },
]


def features(frame, funding=None, asset_class="crypto"):
    """Crypto uses real continuous bars; equities supply observed session-hour bars.

    Equity windows count observations and are not literal crypto calendar days.
    Daily crypto rule_scores are not an equity strategy adapter.
    """
    if asset_class not in ("crypto", "equity"):
        raise ValueError("Unknown asset class")
    if (
        not isinstance(frame.index, pd.DatetimeIndex)
        or frame.index.tz is None
        or frame.index.has_duplicates
        or not frame.index.is_monotonic_increasing
    ):
        raise ValueError("Supply unique ordered timezone-aware market bars")
    required = ["open", "high", "low", "close", "volume", "quote_volume"]
    if not set(required).issubset(frame):
        raise ValueError("Supply real OHLCV and quote turnover")
    aggregation = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "quote_volume": "sum",
    }
    if "taker_buy_quote_volume" in frame:
        aggregation["taker_buy_quote_volume"] = "sum"
    h = (
        frame.resample("h").agg(aggregation)
        if asset_class == "crypto"
        else frame[list(aggregation)].copy()
    )
    x = h.shift(1)
    c = x.close
    returns = np.log(c).diff()
    vol = returns.rolling(168).std(ddof=1).replace(0, np.nan)
    f = pd.DataFrame(index=h.index)
    for n in (168, 504, 1512):
        f[f"mom_z{n}"] = (np.log(c / c.shift(n)) / (vol * np.sqrt(n))).clip(-5, 5)
    f["trend_consensus"] = np.tanh(f[["mom_z168", "mom_z504", "mom_z1512"]]).mean(
        axis=1, skipna=False
    )
    f["efficiency168"] = (c - c.shift(168)) / c.diff().abs().rolling(168).sum().replace(0, np.nan)
    f["prior_high480"] = x.high.shift(1).rolling(480).max()
    f["prior_low480"] = x.low.shift(1).rolling(480).min()
    f["channel480"] = (
        2 * (c - f.prior_low480) / (f.prior_high480 - f.prior_low480).replace(0, np.nan) - 1
    ).clip(-3, 3)
    f["return_z24"] = (np.log(c / c.shift(24)) / (vol * np.sqrt(24))).clip(-5, 5)
    f["return_z4"] = np.log(c / c.shift(4)) / (vol * 2)
    f["vol24"] = returns.rolling(24).std(ddof=1)
    f["vol168"] = vol
    f["volume_impulse24"] = x.quote_volume.rolling(24).sum() / (
        24 * x.quote_volume.shift(1).rolling(168).mean()
    ).replace(0, np.nan)
    previous = x.close.shift(1)
    tr = pd.concat(
        [x.high - x.low, (x.high - previous).abs(), (x.low - previous).abs()], axis=1
    ).max(axis=1)
    f["atr24"] = tr.rolling(24).mean() / c
    f["close"] = c
    f["ema168"] = c.ewm(span=168, adjust=False, min_periods=168).mean()
    f["quote_day28"] = x.quote_volume.rolling(672).mean() * 24
    for n in (24, 72):
        f[f"aggressor{n}"] = (
            (2 * x.taker_buy_quote_volume - x.quote_volume).rolling(n).sum()
            / x.quote_volume.rolling(n).sum().replace(0, np.nan)
            if "taker_buy_quote_volume" in x
            else np.nan
        )
    prior_flow = f.aggressor24.shift(1)
    f["aggressor_surprise24"] = (
        (f.aggressor24 - prior_flow.rolling(168).mean())
        / prior_flow.rolling(168).std(ddof=1).replace(0, np.nan)
    ).clip(-5, 5)
    f["absorption24"] = (f.aggressor_surprise24 - f.return_z24).clip(-5, 5)
    f["flow_trend_agreement"] = f.trend_consensus * f.aggressor_surprise24.clip(-3, 3)
    if funding is None:
        f["funding_last"] = np.nan
    else:
        settled = funding.funding_rate.copy()
        settled.index = settled.index + pd.Timedelta(minutes=5)
        f["funding_last"] = settled.reindex(f.index, method="ffill")
    return f, h.open


def rule_scores(f, rule):
    """Daily state transitions; max7day score lease and one-day rearm gap.

    This is a signal state, never a claim about actual entry or ownership. The
    downstream account separately owns fills, capacity, stops and realized PnL.
    """
    if rule not in RULES:
        raise ValueError("Unknown frozen mechanism")
    daily = f.loc[f.index.hour == 0]
    output = pd.Series(0.0, index=daily.index, name=rule)
    side, age = 0, 0
    for clock, row in daily.iterrows():
        if (
            not np.isfinite(row[PRICE_COLUMNS].to_numpy(dtype=float)).all()
            or row.quote_day28 < 50000000
        ):
            side, age = 0, 0
            continue
        trend = row.trend_consensus
        direction = int(np.sign(trend))
        flow = row.aggressor_surprise24
        if rule == "slow_trend":
            entry = abs(trend) >= 0.25
            hold = abs(trend) >= 0.10 and direction == side
        elif rule == "flow_trend":
            entry = abs(trend) >= 0.25 and flow * direction > 0.5
            hold = direction == side and flow * side > -0.5
        elif rule == "breakout_acceptance":
            direction = (
                1 if row.close > row.prior_high480 else -1 if row.close < row.prior_low480 else 0
            )
            entry = direction != 0 and flow * direction >= 0
            hold = (row.close - row.ema168) * side > 0
        elif rule == "pullback_continuation":
            entry = (
                abs(trend) >= 0.25
                and -0.8 < row.return_z24 * direction < 0
                and flow * direction > 0.25
            )
            hold = direction == side and row.return_z24 * side < 1
        elif rule == "absorption_release":
            entry = (
                abs(trend) >= 0.20
                and row.absorption24 * direction > 1
                and flow * direction > 0.5
                and row.return_z4 * direction > 0
            )
            hold = direction == side and flow * side >= 0
        else:
            entry = abs(trend) >= 0.25 and row.funding_last * direction < 0.0001
            hold = direction == side and row.funding_last * side < 0.0003
        if side:
            if not hold or age >= 7:
                side, age = 0, 0
                continue
            age += 1
        elif entry:
            side, age = direction, 1
        output.loc[clock] = side * 0.02 * max(abs(trend), 0.25) if side else 0.0
    return output.reindex(f.index, method="ffill").fillna(0)


def lease_model_score(score):
    """The same seven-day maximum score lease for learned and price controls."""
    daily = score.loc[score.index.hour == 0]
    out = daily.copy()
    side, age = 0, 0
    for clock, value in daily.items():
        direction = int(np.sign(value)) if np.isfinite(value) and abs(value) > 0.0032 else 0
        if side and (direction != side or age >= 7):
            side, age = 0, 0
            out.loc[clock] = 0
            continue
        if direction:
            side = direction
            age += 1
            out.loc[clock] = value
        else:
            side, age = 0, 0
            out.loc[clock] = 0
    return out.reindex(score.index, method="ffill").fillna(0)
