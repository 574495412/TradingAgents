"""Frozen, monthly, long cash adaptations on an ex-ante disclosed stock cohort."""

import pandas as pd

from factors.common.total_return import gross_return_level
from tradingagents.research.equities.pit_data import asof_annual


def targets(frames, calendar, identities, annual, plan):
    if len(identities) != plan["universe"]["size"]:
        raise ValueError("Selected historical members cannot be dropped")
    symbols = [r["symbol"] for r in identities]
    if symbols != list(frames):
        raise ValueError("Frozen identity order differs from prices")
    # Calendar, rather than data truncation, determines the true month-end.
    month_ends = calendar.index.to_series().groupby(calendar.index.to_period("M")).last()
    levels = pd.DataFrame({s: gross_return_level(f.close, f.dividend) for s, f in frames.items()})
    monthly = levels.reindex(month_ends)
    mom = monthly.shift(1) / monthly.shift(12) - 1
    policies = plan["policies"]
    if policies != [
        "disclosed_equal",
        "french_mom_12_2",
        "gross_profitability",
        "stock_blend_equal",
    ]:
        raise ValueError("Unregistered policy family")
    books = {k: pd.DataFrame(0.0, index=levels.index, columns=symbols) for k in policies}
    states = pd.Series("universe_not_yet_disclosed", index=levels.index)
    provenance = []
    known = max(pd.Timestamp(r["available_at"]) for r in identities)
    gross, top = plan["gross_limit"], plan["selected_stocks"]
    for day in month_ends:
        if day not in levels.index or calendar.at[day, "close"] < known:
            continue
        decisions = {k: dict.fromkeys(symbols, 0.0) for k in policies}
        decisions["disclosed_equal"] = dict.fromkeys(symbols, gross / len(symbols))
        winners = mom.loc[day].dropna()
        winners = sorted(
            ((s, float(v)) for s, v in winners.items() if v > 0), key=lambda x: (-x[1], x[0])
        )[:top]
        for s, _ in winners:
            decisions["french_mom_12_2"][s] = gross / top
        profitable = []
        for item in identities:
            rec = asof_annual(annual, item["cik"], calendar.at[day, "close"])
            if rec:
                provenance.append(
                    {
                        "decision_day": str(day.date()),
                        "decision_close": calendar.at[day, "close"].isoformat(),
                        "symbol": item["symbol"],
                        **rec,
                    }
                )
                if rec["gross_profitability"] > 0:
                    profitable.append((item["symbol"], rec["gross_profitability"]))
        for s, _ in sorted(profitable, key=lambda x: (-x[1], x[0]))[:top]:
            decisions["gross_profitability"][s] = gross / top
        decisions["stock_blend_equal"] = {
            s: (decisions["french_mom_12_2"][s] + decisions["gross_profitability"][s]) / 2
            for s in symbols
        }
        for k in policies:
            books[k].loc[day] = pd.Series(decisions[k])
        states.loc[day] = "disclosed_stock_monthly"
    # Carry only decisions actually reached. All earlier dates remain zero.
    decisions = set(month_ends)
    for book in books.values():
        book.loc[~book.index.isin(decisions)] = float("nan")
        book.ffill(inplace=True)
        book.fillna(0.0, inplace=True)
    states = (
        states.where(states != "universe_not_yet_disclosed")
        .ffill()
        .fillna("universe_not_yet_disclosed")
    )
    return books, states, provenance
