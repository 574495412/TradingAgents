"""Low-complexity autonomous cross-sectional factor for non-ETH contracts.

The policy intentionally has no leader-event input, symbol allowlist, clock
exception, or per-symbol parameter.  Every eligible contract is ranked with
the same closed-bar market formula.
"""

# Breadth construction, eligibility, and ranking intentionally share one
# causal pass in replay and live execution.
# ruff: noqa: PLR0915

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

POLICY_ID = "weijie_non_eth_cross_sectional_reversal_short_v2"


@dataclass(frozen=True, slots=True)
class NonEthLowComplexityFactorPolicy:
    decision_interval_minutes: int = 30
    minimum_quote_volume_24h: float = 50_000_000.0
    minimum_positive_breadth36: float = 0.20
    maximum_activity_ratio1: float = 4.0
    maximum_nr36: float = 5.0
    maximum_nr288: float = 5.0
    maximum_ret36_pct: float = 10.0
    maximum_flow12: float = 0.20
    minimum_score: float = 3.0
    return_36_weight: float = 0.5
    activity_penalty: float = 0.5
    event_capacity: int = 1


def generate_non_eth_low_complexity_signals(
    market: pd.DataFrame,
    *,
    activation_time: object,
    decision_end: object,
    policy: NonEthLowComplexityFactorPolicy | None = None,
) -> pd.DataFrame:
    """Rank mature, liquid reversal candidates using closed public bars only."""

    cfg = policy or NonEthLowComplexityFactorPolicy()
    output = _empty_signals()
    output.attrs["policy_id"] = POLICY_ID
    if market.empty:
        return output
    required = {
        "time",
        "symbol",
        "close",
        "q288",
        "ret36",
        "nr3",
        "nr12",
        "nr36",
        "nr288",
        "activity_ratio1",
        "flow12",
    }
    missing = sorted(required.difference(market.columns))
    if missing:
        raise ValueError(
            "low-complexity non-ETH market missing: " + ",".join(missing),
        )
    if cfg.decision_interval_minutes <= 0 or 60 % cfg.decision_interval_minutes:
        message = "decision_interval_minutes must divide one hour"
        raise ValueError(message)
    if cfg.event_capacity <= 0:
        message = "event_capacity must be positive"
        raise ValueError(message)
    if (
        cfg.minimum_quote_volume_24h <= 0.0
        or not 0.0 <= cfg.minimum_positive_breadth36 <= 1.0
        or cfg.maximum_activity_ratio1 <= 0.0
        or cfg.maximum_nr36 <= 0.0
        or cfg.maximum_nr288 <= 0.0
        or cfg.maximum_ret36_pct <= 0.0
        or not -1.0 <= cfg.maximum_flow12 <= 1.0
        or cfg.return_36_weight < 0.0
        or cfg.activity_penalty < 0.0
    ):
        message = "invalid low-complexity non-ETH factor policy"
        raise ValueError(message)

    activation = _utc_timestamp(activation_time)
    endpoint = _utc_timestamp(decision_end)
    rows = market.loc[:, sorted(required)].copy()
    rows["time"] = pd.to_datetime(rows["time"], utc=True, errors="raise")
    rows["symbol"] = rows["symbol"].map(_clean_symbol)
    numeric = [
        "close",
        "q288",
        "ret36",
        "nr3",
        "nr12",
        "nr36",
        "nr288",
        "activity_ratio1",
        "flow12",
    ]
    for column in numeric:
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    breadth_universe = rows.loc[
        rows["symbol"].ne("ETHUSDT") & rows["q288"].ge(cfg.minimum_quote_volume_24h) & rows["nr288"].notna(),
        ["time", "nr36"],
    ]
    positive_breadth36 = (
        breadth_universe.groupby("time", sort=True)["nr36"]
        .agg(lambda value: float(value.gt(0.0).mean()))
        .rename("positive_breadth36")
    )
    rows = rows.merge(
        positive_breadth36,
        left_on="time",
        right_index=True,
        how="left",
        validate="many_to_one",
    )
    rows["signal_time"] = rows["time"] + pd.Timedelta(minutes=5)
    rows = rows.loc[
        rows["signal_time"].between(activation, endpoint)
        & rows["time"].dt.minute.mod(cfg.decision_interval_minutes).eq(0)
        & rows["time"].dt.second.eq(0)
        & rows["symbol"].ne("ETHUSDT")
        & rows["close"].gt(0.0)
        & rows["nr288"].notna()
        & rows["q288"].ge(cfg.minimum_quote_volume_24h)
        & rows["positive_breadth36"].ge(
            cfg.minimum_positive_breadth36,
        )
        & rows["nr12"].gt(0.0)
        & rows["nr36"].gt(0.0)
        & rows["nr36"].le(cfg.maximum_nr36)
        & rows["nr288"].le(cfg.maximum_nr288)
        & rows["ret36"].le(cfg.maximum_ret36_pct)
        & rows["nr3"].le(0.0)
        & rows["activity_ratio1"].le(cfg.maximum_activity_ratio1)
        & rows["flow12"].le(cfg.maximum_flow12)
    ].copy()
    if rows.empty:
        return output

    rows["score"] = rows["nr12"] + cfg.return_36_weight * rows["nr36"] - cfg.activity_penalty * rows["activity_ratio1"]
    rows = rows.loc[rows["score"].ge(cfg.minimum_score)].copy()
    if rows.empty:
        return output
    rows = rows.sort_values(
        ["signal_time", "score", "q288", "symbol"],
        ascending=[True, False, False, True],
        kind="stable",
    )
    rows = rows.groupby("signal_time", sort=True).head(
        cfg.event_capacity,
    )
    rows["candidate_id"] = [
        f"{POLICY_ID}|{timestamp.value}|{symbol}"
        for timestamp, symbol in zip(
            rows["signal_time"],
            rows["symbol"],
            strict=True,
        )
    ]
    rows["signal_id"] = rows["candidate_id"]
    rows["bar_time"] = rows["time"]
    rows["direction"] = "SHORT"
    rows["action"] = "OPEN_SHORT"
    rows["operation_type"] = "OPEN"
    rows["lifecycle_role"] = "ROOT"
    rows["signal_price"] = rows["close"]
    rows["reason"] = "breadth_gated_reversal_short"
    rows["source_factor"] = POLICY_ID
    rows["policy_id"] = POLICY_ID
    rows["market_only"] = True
    rows["leader_direction_used"] = False
    result = rows.loc[:, output.columns].reset_index(drop=True)
    result.attrs["policy_id"] = POLICY_ID
    return result


def _empty_signals() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "candidate_id",
            "signal_id",
            "signal_time",
            "bar_time",
            "symbol",
            "direction",
            "action",
            "operation_type",
            "lifecycle_role",
            "signal_price",
            "score",
            "positive_breadth36",
            "reason",
            "source_factor",
            "policy_id",
            "market_only",
            "leader_direction_used",
        ],
    )


def _utc_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _clean_symbol(value: object) -> str:
    return str(value or "").upper().replace("/", "").replace("-", "").strip()


__all__ = [
    "POLICY_ID",
    "NonEthLowComplexityFactorPolicy",
    "generate_non_eth_low_complexity_signals",
]
