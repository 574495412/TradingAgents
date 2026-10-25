"""Independent mathematical adapters for pinned public factor definitions.

Upstream expressions/parameters and deviations are registered by the study.
No third-party trading program, installer or configuration is executed here.
Crypto windows count actual UTC days; stocks supply observed session-day bars.
"""

import numpy as np
import pandas as pd

PRICE_RULES = ["pst_ewmac32", "pst_accel4", "pst_breakout10"]
QLIB_RULES = ["qlib_corr20", "qlib_resi20"]


def daily(frame, asset_class="crypto"):
    if asset_class == "equity":
        return frame.copy()
    if asset_class != "crypto":
        raise ValueError("Unknown market calendar")
    return frame.resample("D").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "quote_volume": "sum",
        }
    )


def formulas(d):
    c = d.close
    raw_vol = c.diff().ewm(span=35, min_periods=10, adjust=True).std().clip(lower=1e-10)
    floor = raw_vol.rolling(500, min_periods=100).quantile(0.05)
    if len(floor):
        floor.iloc[0] = 0.0
    vol = pd.Series(np.maximum(raw_vol, floor.ffill()), index=c.index)

    def ewmac(n):
        return (
            c.ewm(span=n, min_periods=1).mean() - c.ewm(span=n * 4, min_periods=1).mean()
        ) / vol.ffill()

    f = pd.DataFrame(index=d.index)
    f["pst_ewmac32"] = ewmac(32)
    short = ewmac(4)
    f["pst_accel4"] = short - short.shift(4)
    hi = c.rolling(10, min_periods=5).max()
    lo = c.rolling(10, min_periods=5).min()
    f["pst_breakout10"] = (
        (40 * (c - (hi + lo) / 2) / (hi - lo).replace(0, np.nan)).ewm(span=2, min_periods=1).mean()
    )
    f["qlib_corr20"] = c.rolling(20, min_periods=20).corr(np.log(d.volume + 1))
    # Endpoint OLS residual divided by current close, as Qlib RESI20 specifies.
    axis = np.arange(20, dtype=float)
    axis -= axis.mean()

    def endpoint(y):
        slope = float(axis @ y / (axis @ axis))
        return y[-1] - (y.mean() + slope * axis[-1])

    f["qlib_resi20"] = c.rolling(20, min_periods=20).apply(endpoint, raw=True) / c
    f["relative_vol"] = vol / c
    f["momentum30"] = c / c.shift(30) - 1
    f["momentum21"] = c / c.shift(21) - 1
    f["momentum14"] = c / c.shift(14) - 1
    f["above100"] = c > c.rolling(100, min_periods=100).mean()
    f["turnover28"] = d.quote_volume.rolling(28, min_periods=28).mean()
    return f


def capped_momentum(values, cap=0.4):
    positive = values.clip(lower=1e-9)
    weights = positive / positive.sum()
    if len(weights) * cap < 1:
        return weights
    for _ in range(8):
        above = weights > cap + 1e-12
        if not above.any():
            break
        excess = float((weights.loc[above] - cap).sum())
        weights.loc[above] = cap
        below = weights < cap - 1e-12
        total = weights.loc[below].sum()
        if total <= 0:
            break
        weights.loc[below] += excess * weights.loc[below] / total
    return weights


def books(frames, rates, normalization, directions):
    symbols = sorted(frames)
    features = {s: formulas(daily(frames[s])) for s in symbols}
    index = features[symbols[0]].index
    quote = pd.DataFrame({s: f.turnover28 for s, f in features.items()})
    eligible = (quote >= 50000000) & (quote.rank(axis=1, ascending=False, method="first") <= 30)
    # All rows below are computed on completed day t, then shifted to decision day t+1.
    scores = {
        k: pd.DataFrame(0.0, index=index, columns=symbols)
        for k in PRICE_RULES
        + QLIB_RULES
        + [
            "sentinel_champion",
            "sentinel_carry21",
            "sentinel_carry14_buffer",
            "liquid_long_control",
        ]
    }
    volatility = {k: pd.DataFrame({s: f.relative_vol for s, f in features.items()}) for k in scores}
    for k in PRICE_RULES:
        for s in symbols:
            scores[k][s] = (features[s][k] / normalization[k]).clip(-3, 3) * 0.01
    for k in QLIB_RULES:
        raw = pd.DataFrame({s: features[s][k] * directions[k] for s in symbols})
        scores[k] = (raw.where(eligible).rank(axis=1, pct=True) - 0.5) * 0.04
    funding = pd.DataFrame(
        {s: rates[s].funding_rate.resample("D").sum(min_count=1) for s in symbols}
    ).reindex(index)
    carry_mean = funding.rolling(5, min_periods=5).mean()
    for k in ("sentinel_carry21", "sentinel_carry14_buffer"):
        volatility[k].loc[:, :] = 1.0
    volatility["sentinel_champion"].loc[:, :] = np.nan
    for clock in index:
        names = [s for s in symbols if bool(eligible.loc[clock, s])]
        if len(names) >= 10 and carry_mean.loc[clock, names].notna().all():
            for key, lookback in (("sentinel_carry21", 21), ("sentinel_carry14_buffer", 14)):
                mom = pd.Series({s: features[s].loc[clock, f"momentum{lookback}"] for s in names})
                if mom.isna().any():
                    continue
                # Original sorting uses stable ordinal ranks; alphabetical input breaks ties.
                ranking = mom.rank(method="first") + (-carry_mean.loc[clock, names]).rank(
                    method="first"
                )
                ordered = ranking.sort_values(ascending=False, kind="stable").index.tolist()
                if key == "sentinel_carry21":
                    for s in ordered[:5]:
                        scores[key].loc[clock, s] = 1.0
                    for s in ordered[-5:]:
                        scores[key].loc[clock, s] = -1.0
                else:
                    for rank, s in enumerate(ordered):
                        scores[key].loc[clock, s] = len(ordered) - 0.5 - rank
                    # Positive ordered values encode ordering; actual held-side retention lives
                    # in the replay engine, not a pretend signal-owned inventory.
        liquid = quote.loc[clock, names].sort_values(ascending=False, kind="stable").index[:4]
        scores["liquid_long_control"].loc[clock, liquid] = 1.0
        if bool(features["BTCUSDT"].loc[clock, "above100"]):
            mom = pd.Series({s: features[s].loc[clock, "momentum30"] for s in names}).dropna()
            picked = mom.loc[mom > 0].sort_values(ascending=False, kind="stable").iloc[:5]
            if len(picked):
                w = capped_momentum(picked)
                scores["sentinel_champion"].loc[clock, w.index] = 1.0
                volatility["sentinel_champion"].loc[clock, w.index] = 1 / w
    for key in scores:
        scores[key] = scores[key].where(eligible, 0.0).shift(1)
        volatility[key] = volatility[key].shift(1)
    return scores, volatility, features, eligible.shift(1, fill_value=False)
