"""A source-derived daily long index; no fees, fills or customer-cash certification."""

import numpy as np
import pandas as pd


def daily_long_index_returns(daily, funding):
    """Reinvest daily price PnL and actual reported mark-times-rate funding.

    Each day's reference quantity is one divided by the preceding daily close.
    A day with no reported event is unknown, rather than assumed to cost zero.
    The index is an economic signal input and is not an executable net account.
    """
    if funding.index.has_duplicates or not funding.index.is_monotonic_increasing:
        raise ValueError("Funding records must be unique and in exact time order")
    if daily.index.has_duplicates or not daily.index.is_monotonic_increasing:
        raise ValueError("Daily observations must be unique and sorted")
    if str(daily.index.tz) != "UTC" or str(funding.index.tz) != "UTC":
        raise ValueError("Explicit UTC inputs required")
    calendar = pd.date_range(daily.index[0], daily.index[-1], freq="D")
    f = daily.reindex(calendar)
    valid = (
        f.price_liquidity_feature_usable.eq(True)
        & f.volume.gt(0)
        & f.quote_volume.gt(0)
        & np.isfinite(f.close)
        & f.close.gt(0)
    )
    valid.iloc[0] = False
    close = f.close.where(valid)
    rate_ok = np.isfinite(funding.funding_rate)
    mark_ok = np.isfinite(funding.mark) & funding.mark.gt(0)
    events = pd.DataFrame(
        {
            "cash_per_unit": funding.mark * funding.funding_rate,
            "valid": rate_ok & mark_ok,
            "events": 1,
        },
        index=funding.index,
    )
    grouped = (
        events.groupby(events.index.floor("D"))
        .agg(
            cash_per_unit=("cash_per_unit", "sum"),
            all_event_marks_valid=("valid", "all"),
            reported_events=("events", "sum"),
        )
        .reindex(calendar)
    )
    previous = close.shift(1)
    usable = (
        valid
        & valid.shift(1, fill_value=False)
        & grouped.reported_events.ge(1)
        & grouped.all_event_marks_valid.eq(True)
    )
    result = pd.DataFrame(index=calendar)
    result["price_return"] = close / previous - 1
    result["funding_return_paid"] = grouped.cash_per_unit / previous
    result["economic_return"] = (result.price_return - result.funding_return_paid).where(usable)
    result["reported_events"] = grouped.reported_events
    result["usable"] = usable & np.isfinite(result.economic_return) & result.economic_return.gt(-1)
    result["economic_return"] = result.economic_return.where(result.usable)
    return result
