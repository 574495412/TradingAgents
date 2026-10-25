"""Past-only daily state hypotheses, usable across supplied market calendars.

Inputs contain complete observed daily bars; each output is shifted to the next
observed decision session. Crypto callers aggregate UTC days first. No future
return labels, retrospective regime dates, model fitting or news synthesis.
"""

import numpy as np
import pandas as pd

STATE_FEATURES = ["trend_z", "efficiency", "volatility_ratio", "breadth", "eligible_count"]


def classify_market(frames, parameters, benchmark="BTCUSDT"):
    if benchmark not in frames or not frames:
        raise ValueError("Benchmark daily observations required")
    p = parameters
    index = frames[benchmark].index
    if (
        not isinstance(index, pd.DatetimeIndex)
        or not index.is_monotonic_increasing
        or not index.is_unique
    ):
        raise ValueError("Unique ordered observed-session calendar required")
    if index.tz is None:
        raise ValueError("Explicit market timezone required")
    c = frames[benchmark].close
    ret = c.pct_change(fill_method=None)
    vol = ret.rolling(p["volatility_window"], min_periods=p["volatility_window"]).std(ddof=1)
    historic = vol.shift(1).rolling(
        p["volatility_history"], min_periods=p["volatility_history_min"]
    )
    high_vol = historic.quantile(p["high_volatility_quantile"])
    momentum = c / c.shift(p["momentum_window"]) - 1
    slow = c.rolling(p["slow_ma_window"], min_periods=p["slow_ma_window"]).mean()
    travelled = (
        c.diff().abs().rolling(p["efficiency_window"], min_periods=p["efficiency_window"]).sum()
    )
    efficiency = (c - c.shift(p["efficiency_window"])).abs() / travelled.replace(0, np.nan)
    quote = pd.DataFrame(
        {
            s: f.quote_volume.reindex(index)
            .rolling(p["turnover_window"], min_periods=p["turnover_window"])
            .mean()
            for s, f in sorted(frames.items())
        }
    )
    eligible = (quote >= p["minimum_daily_quote"]) & (
        quote.rank(axis=1, ascending=False, method="first") <= p["maximum_eligible"]
    )
    relative = pd.DataFrame(
        {
            s: f.close.reindex(index) / f.close.reindex(index).shift(p["momentum_window"]) - 1
            for s, f in sorted(frames.items())
        }
    )
    valid = eligible & relative.notna()
    count = valid.sum(axis=1)
    breadth = relative.gt(0).astype(float).where(valid).sum(axis=1, min_count=1) / count.replace(
        0, np.nan
    )
    features = pd.DataFrame(
        {
            "trend_z": momentum / (vol.replace(0, np.nan) * np.sqrt(p["momentum_window"])),
            "efficiency": efficiency,
            "volatility_ratio": vol / high_vol.replace(0, np.nan),
            "breadth": breadth,
            "eligible_count": count.astype(float),
        },
        index=index,
    )
    state = pd.Series("range", index=index, dtype=object)
    strong = (features.trend_z.abs() >= p["trend_z_min"]) & (
        features.efficiency >= p["efficiency_min"]
    )
    state.loc[strong & (momentum > 0) & (c > slow)] = "trend_up"
    state.loc[strong & (momentum < 0) & (c < slow)] = "trend_down"
    state.loc[features.volatility_ratio > 1] = "volatility_breakout"
    state.loc[count < p["minimum_eligible"]] = "liquidity_stress"
    complete = features.notna().all(axis=1) & slow.notna() & np.isfinite(features).all(axis=1)
    state.loc[~complete] = "unknown"
    # A complete bar at session d is unavailable to a session-d open decision.
    out = features.shift(1)
    out["state"] = state.shift(1, fill_value="unknown")
    return out, eligible.shift(1, fill_value=False)


def event_risk(decisions, events, hours=24):
    """Post-release coverage only; False means no covered FOMC event, not no news."""
    result = pd.Series(False, index=decisions)
    if decisions.tz is None or not decisions.is_monotonic_increasing or not decisions.is_unique:
        raise ValueError("Unique ordered timezone-aware decisions required")
    for event in events:
        published = pd.Timestamp(event["published_at"])
        available = pd.Timestamp(event["available_at"])
        if published.tzinfo is None or available.tzinfo is None or available < published:
            raise ValueError("Event publication and availability must be causal")
        if event.get("source_kind") != "official_fomc_release":
            raise ValueError("Unverified event kind")
        result.loc[
            (decisions >= available) & (decisions < available + pd.Timedelta(hours=hours))
        ] = True
    return result
