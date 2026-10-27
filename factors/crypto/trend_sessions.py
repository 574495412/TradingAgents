"""Crypto calendar/identity adapter for the unchanged source EWMAC forecast.

Not a new economic factor. Invalid days split state; no prices or volatility are
filled across such splits. Stock sessions require their separate calendar adapter.
"""

import numpy as np
import pandas as pd

from factors.common.github_formulas import formulas

FEATURES = ["pst_ewmac32", "relative_vol", "consecutive_observations"]


def qualified_features(frame, start, minimum=180, episode_start=None):
    if (
        frame.index.tz is None
        or frame.index.has_duplicates
        or not frame.index.is_monotonic_increasing
    ):
        raise ValueError("Qualified source index must be aware, unique and ordered")
    first = pd.Timestamp(start)
    if first.tzinfo is None:
        first = first.tz_localize("UTC")
    calendar = pd.date_range(first, frame.index[-1] + pd.Timedelta(days=1), freq="D")
    q = frame.reindex(calendar)
    valid = q.price_liquidity_feature_usable.eq(True) & q.volume.gt(0) & q.quote_volume.gt(0)
    if frame.index[0] in valid.index:
        valid.loc[frame.index[0]] = False
    if episode_start is not None:
        valid.loc[valid.index < pd.Timestamp(episode_start)] = False
    result = pd.DataFrame(np.nan, index=calendar, columns=FEATURES)
    groups = (~valid).cumsum()
    for _, labels in valid.loc[valid].groupby(groups.loc[valid]).groups.items():
        segment = q.loc[labels]
        if len(segment) < minimum:
            continue
        f = formulas(segment)
        age = np.arange(1, len(segment) + 1)
        result.loc[labels, ["pst_ewmac32", "relative_vol"]] = f[["pst_ewmac32", "relative_vol"]]
        result.loc[labels, "consecutive_observations"] = age
        result.loc[labels[age < minimum], ["pst_ewmac32", "relative_vol"]] = np.nan
    return result.shift(1)
