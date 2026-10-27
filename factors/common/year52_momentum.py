"""Year-52-week momentum formation mathematics, with explicit input gates.

Source mapping: Borri/Liu/Tsyvinski/Wu, arXiv2510.14435v4, sections2/3.2.
Provider observation-date labels define groups; they do NOT determine an order
time. Availability, tradable identities and the execution clock need separate
evidence. No returns, winsorized performance, costs or profitability are implied.
"""

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from fractions import Fraction


def week_end_labels(year):
    """51 seven-date groups, then the remaining8/9 provider-date labels."""
    if not 2 <= year <= 9998:
        raise ValueError("Unsupported year")
    first = date(year, 1, 1)
    return [first + timedelta(days=7 * w - 1) for w in range(1, 52)] + [date(year, 12, 31)]


def momentum_labels(formation_label):
    label = date.fromisoformat(str(formation_label))
    ends = week_end_labels(label.year - 1) + week_end_labels(label.year)
    if label not in ends:
        raise ValueError("Formation date is not a source calendar week-end label")
    position = ends.index(label)
    return tuple(ends[position - 2 : position + 1])


def _positive(value):
    try:
        number = Decimal(str(value))
    except (ValueError, InvalidOperation) as error:
        raise ValueError("Missing/nonfinite source input") from error
    if not number.is_finite() or number <= 0:
        raise ValueError("Missing/nonpositive source input")
    return Fraction(number)


def two_week_momentum(prices_by_label, formation_label):
    labels = momentum_labels(formation_label)
    # Require the intermediate endpoint as well: a missing period cannot be
    # turned into an apparently complete two-week sample by endpoint arithmetic.
    prices = [_positive(prices_by_label.get(label.isoformat())) for label in labels]
    return prices[-1] / prices[0] - 1


def _instant(value):
    instant = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if instant.tzinfo is None:
        raise ValueError("Timezone-aware knowledge/decision time required")
    return instant.astimezone(UTC)


def formation_book(records, formation_label, decision_at, *, account_gross=Fraction(4, 5)):
    """Value-weight outer quintiles, rejecting unknown identity/availability.

    The original spread has +1long/-1short (gross2), not a cash-funded100%
    account return. Account weights are scaled by account_gross/2. Exact rational
    weights are returned. Deterministic lexical tie resolution and integer-rank
    quintiles are declared local conventions, not an exact author-code replica.
    """
    decision = _instant(decision_at)
    label = date.fromisoformat(str(formation_label)).isoformat()
    momentum_labels(label)
    if decision.date().isoformat() < label or not 0 < account_gross <= 1:
        raise ValueError("Invalid decision date or risk budget")
    candidates, rejected, seen_ids, seen_identities = [], [], set(), set()
    for row in records:
        provider_id = row.get("provider_id")
        try:
            known = _instant(row["known_at"])
            if known > decision:
                rejected.append({"id": provider_id, "reason": "not_known_at_decision"})
                continue
            if row.get("formation_label") != label:
                raise ValueError("Mixed formation labels")
            if not provider_id or not row.get("native_identity"):
                raise ValueError("Missing stable provider/native identity")
            if row.get("identity_verified_asof") is not True:
                raise ValueError("Historical identity not established")
            if row.get("eligible_asof") is not True:
                raise ValueError("Tradability/qualification not established")
            if (
                row.get("is_stablecoin_asof") is not False
                or row.get("is_wrapped_asof") is not False
            ):
                raise ValueError("Excluded or unknown token classification")
            if row.get("observed_days_asof", 0) < 30:
                raise ValueError("Fewer than30 observed days")
            market_cap = _positive(row.get("market_cap_usd"))
            _positive(row.get("volume_usd"))
            if market_cap < 1000000:
                raise ValueError("Below source market-cap floor")
            momentum = two_week_momentum(row["prices_by_label"], label)
        except (KeyError, ValueError, TypeError) as error:
            rejected.append({"id": provider_id, "reason": str(error)})
            continue
        if provider_id in seen_ids or row["native_identity"] in seen_identities:
            raise ValueError("Ambiguous duplicate identity at the same formation")
        seen_ids.add(provider_id)
        seen_identities.add(row["native_identity"])
        candidates.append({"id": provider_id, "cap": market_cap, "momentum": momentum})
    if len(candidates) < 10:
        return {
            "status": "insufficient_verified_inputs",
            "eligible": len(candidates),
            "spread_weights": {},
            "account_weights": {},
            "rejected": rejected,
        }
    ordered = sorted(candidates, key=lambda row: (row["momentum"], row["id"]))
    buckets = [
        [row for rank, row in enumerate(ordered) if 5 * rank // len(ordered) == q] for q in range(5)
    ]
    spread = {}
    for group, sign in [(buckets[0], -1), (buckets[4], 1)]:
        total_cap = sum(row["cap"] for row in group)
        for row in group:
            spread[row["id"]] = sign * row["cap"] / total_cap
    account = {symbol: weight * account_gross / 2 for symbol, weight in spread.items()}
    assert sum(spread.values()) == 0 and sum(abs(w) for w in spread.values()) == 2
    assert sum(account.values()) == 0 and sum(abs(w) for w in account.values()) == account_gross
    labels = momentum_labels(label)
    return {
        "status": "formation_math_only",
        "eligible": len(candidates),
        "momentum_endpoint_labels": [d.isoformat() for d in labels],
        "lookback_elapsed_days": (labels[-1] - labels[0]).days,
        "quintile_counts": [len(g) for g in buckets],
        "spread_weights": spread,
        "account_weights": account,
        "rejected": rejected,
        "source_spread_gross": 2,
        "account_gross": account_gross,
        "author_code_exact_replication": False,
        "economic_account_verified": False,
    }
