"""Date/role contracts for author-published realized crypto risk factors.

The public LTW workbook is a weekly return benchmark, not coin-level market-cap
inputs or a timestamped prediction. Source date labels never establish when the
value was known. Stocks require their own benchmark and trading calendar.
"""

from datetime import date

from factors.common.year52_momentum import week_end_labels


def week_label(key):
    text = str(key)
    if len(text) != 6 or not text.isdigit():
        raise ValueError("Expected source YYYYWW key, not an Excel serial or ISO week")
    year, week = int(text[:4]), int(text[4:])
    if not 2 <= year <= 9998 or not 1 <= week <= 52:
        raise ValueError("Invalid source52week key")
    return week_end_labels(year)[week - 1]


def window_coverage(keys, start, end_exclusive):
    start, end = date.fromisoformat(start), date.fromisoformat(end_exclusive)
    if start >= end:
        raise ValueError("Empty/reversed date interval")
    available = [week_label(key) for key in keys]
    if len(available) != len(set(available)):
        raise ValueError("Duplicate author benchmark week")
    expected = [
        label
        for year in range(start.year, end.year + 1)
        for label in week_end_labels(year)
        if start <= label < end
    ]
    matched = sorted(set(available).intersection(expected))
    return {
        "expected_nominal_week_labels": len(expected),
        "overlap_rows": len(matched),
        "first_overlap_label": matched[0].isoformat() if matched else None,
        "last_overlap_label": matched[-1].isoformat() if matched else None,
        "all_nominal_week_labels_covered": bool(expected) and len(matched) == len(expected),
        "publication_time_or_return_units_certified": False,
    }


def require_reference_use(role, *, units_verified=False, coverage=None):
    if role == "calendar_check":
        return
    if role != "ex_post_attribution":
        raise ValueError(
            "Realized factor returns cannot supply raw caps, identities or live signals"
        )
    if not units_verified:
        raise ValueError("Source return units must be verified before attribution")
    if not coverage or not coverage["all_nominal_week_labels_covered"]:
        raise ValueError("No complete comparable factor/strategy observation window")
