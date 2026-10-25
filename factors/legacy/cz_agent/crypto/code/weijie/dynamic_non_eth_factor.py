"""Causal local-state factors for market-discovered non-ETH contracts."""

from __future__ import annotations

# ruff: noqa: C901, FBT003
from collections.abc import Collection, Mapping
from dataclasses import dataclass

import pandas as pd

POLICY_ID = "weijie_dynamic_non_eth_autonomous_transition_v3"

REQUIRED_COLUMNS = frozenset(
    {
        "bar_time",
        "symbol",
        "close",
        "normalized_return_3",
        "normalized_return_12",
        "normalized_return_36",
        "normalized_return_144",
        "normalized_return_288",
        "quote_volume_ratio",
        "range_location_288",
        "signed_seller_pressure",
        "subtype",
    },
)


@dataclass(frozen=True, slots=True)
class DynamicNonEthLocalStateConfig:
    """Round, cross-asset thresholds for sparse post-shock transitions."""

    deep_drawdown_return_288: float = -20.0
    stabilization_minimum_return_3: float = -1.0
    stabilization_maximum_range_location: float = 0.40
    downside_return_144: float = -8.0
    downside_return_12: float = -1.5
    downside_return_3: float = -0.5
    recovery_minimum_return_12: float = 0.0
    recovery_minimum_return_36: float = 0.0
    recovery_maximum_seller_pressure: float = -0.20
    seeded_recovery_minimum_return_12: float = 2.0
    seeded_recovery_minimum_return_36: float = 1.0
    seeded_recovery_minimum_return_3: float = 1.5
    seeded_recovery_minimum_return_144: float = -8.0
    downside_minimum_quote_ratio: float = 1.0
    direction_rearm_hours: float = 12.0
    minimum_observation_bars_after_activation: int = 1


def generate_dynamic_non_eth_local_state_signals(
    feature_rows: pd.DataFrame,
    *,
    symbol_activation_ms: Mapping[str, int],
    seeded_symbols: Collection[str] = (),
    config: DynamicNonEthLocalStateConfig | None = None,
) -> pd.DataFrame:
    """Generate symbol-gated OPEN signals from closed market features."""

    cfg = config or DynamicNonEthLocalStateConfig()
    missing = sorted(REQUIRED_COLUMNS.difference(feature_rows.columns))
    if missing:
        raise ValueError("dynamic non-ETH feature rows missing: " + ",".join(missing))
    if feature_rows.empty or not symbol_activation_ms:
        return _empty_signals()

    market = feature_rows.copy()
    market["bar_time"] = pd.to_datetime(
        market["bar_time"],
        utc=True,
        errors="raise",
    )
    market["symbol"] = (
        market["symbol"]
        .astype(str)
        .str.upper()
        .str.replace("/", "", regex=False)
        .str.replace("-", "", regex=False)
    )
    activation = {
        str(symbol).upper().replace("/", "").replace("-", ""): pd.Timestamp(
            int(timestamp_ms),
            unit="ms",
            tz="UTC",
        )
        for symbol, timestamp_ms in symbol_activation_ms.items()
    }
    seeded = {
        str(symbol).upper().replace("/", "").replace("-", "")
        for symbol in seeded_symbols
    }
    market["activation_time"] = market["symbol"].map(activation)
    market = market.loc[
        market["activation_time"].notna()
        & market["bar_time"].ge(market["activation_time"])
    ].copy()
    if market.empty:
        return _empty_signals()

    market = build_dynamic_non_eth_local_states(
        market,
        seeded_symbols=seeded,
        config=cfg,
    )
    long_state = market["long_state"]
    short_state = market["short_state"]
    records: list[dict[str, object]] = []
    for action, family, reason, condition in (
        (
            "OPEN_LONG",
            "post_shock_stabilization_long",
            "深度回撤后短周期停止恶化且仍处低位",
            long_state,
        ),
        (
            "OPEN_SHORT",
            "downside_continuation_short",
            "中长周期弱势中的短周期再次下破",
            short_state,
        ),
    ):
        ordered = market.sort_values(["symbol", "bar_time"], kind="stable")
        discovery_eligible = (
            ordered["autonomous_discovery_eligible"].fillna(False).astype(bool)
            if "autonomous_discovery_eligible" in ordered
            else pd.Series(True, index=ordered.index)
        )
        market_eligible = (
            ordered["eligible"].fillna(False).astype(bool)
            if "eligible" in ordered
            else pd.Series(True, index=ordered.index)
        )
        ordered["condition_active"] = condition.loc[ordered.index].astype(bool)
        ordered["discovery_eligible"] = discovery_eligible
        ordered["market_eligible"] = market_eligible
        for _symbol, group in ordered.groupby("symbol", sort=True):
            last_signal: pd.Timestamp | None = None
            prior_condition = False
            observed_bars = 0
            for row in group.itertuples(index=False):
                signal_time = pd.Timestamp(row.bar_time)
                if not (
                    bool(row.discovery_eligible)
                    and bool(row.market_eligible)
                ):
                    # Live discovery removes a flat symbol and discards its
                    # activation timestamp. Re-entering the roster therefore
                    # starts a new observation epoch. Historical replay must
                    # reset the state here as well; otherwise roster re-entry
                    # itself can be mistaken for a fresh factor transition.
                    prior_condition = False
                    observed_bars = 0
                    continue
                active_condition = bool(row.condition_active)
                observed_bars += 1
                if observed_bars <= max(
                    1,
                    int(cfg.minimum_observation_bars_after_activation),
                ):
                    # Joining the market-attention roster is not an entry
                    # event. Establish the current state first so an old,
                    # already-active condition cannot fire merely because the
                    # symbol has just become observable.
                    prior_condition = active_condition
                    continue
                transition_candidate = active_condition and not prior_condition
                prior_condition = active_condition
                if not transition_candidate:
                    continue
                if (
                    last_signal is not None
                    and signal_time - last_signal
                    < pd.Timedelta(hours=float(cfg.direction_rearm_hours))
                ):
                    continue
                records.append(
                    {
                        "signal_time": signal_time,
                        "symbol": str(row.symbol),
                        "action": action,
                        "operation_type": "OPEN",
                        "signal_price": float(row.close),
                        "factor_family": family,
                        "reason": reason,
                        "policy_id": POLICY_ID,
                        "market_only": True,
                        "leader_direction_used": False,
                    },
                )
                last_signal = signal_time
    if not records:
        return _empty_signals()
    output = pd.DataFrame.from_records(records).sort_values(
        ["signal_time", "symbol", "action"],
        kind="stable",
    )
    output["signal_id"] = [
        (
            f"{POLICY_ID}|{pd.Timestamp(row.signal_time).value}|"
            f"{row.symbol}|{row.action}|{row.factor_family}"
        )
        for row in output.itertuples(index=False)
    ]
    return output.reset_index(drop=True)


def build_dynamic_non_eth_local_states(
    feature_rows: pd.DataFrame,
    *,
    seeded_symbols: Collection[str] = (),
    config: DynamicNonEthLocalStateConfig | None = None,
) -> pd.DataFrame:
    """Attach reusable market-only long/short state flags to feature rows."""

    cfg = config or DynamicNonEthLocalStateConfig()
    missing = sorted(REQUIRED_COLUMNS.difference(feature_rows.columns))
    if missing:
        raise ValueError("dynamic non-ETH feature rows missing: " + ",".join(missing))
    market = feature_rows.copy()
    seeded = {
        str(symbol).upper().replace("/", "").replace("-", "")
        for symbol in seeded_symbols
    }
    stabilization = (
        market["normalized_return_288"].le(cfg.deep_drawdown_return_288)
        & market["normalized_return_3"].ge(cfg.stabilization_minimum_return_3)
        & market["range_location_288"].le(
            cfg.stabilization_maximum_range_location,
        )
    )
    recovery_confirmation = (
        (
            market["normalized_return_12"].ge(
                cfg.recovery_minimum_return_12,
            )
            & market["normalized_return_36"].ge(
                cfg.recovery_minimum_return_36,
            )
        )
        | market["signed_seller_pressure"].le(
            cfg.recovery_maximum_seller_pressure,
        )
    )
    unseeded_long_state = stabilization & recovery_confirmation
    short_state = (
        (
            market["normalized_return_12"].le(-3.0)
            & market["normalized_return_36"].le(-4.0)
        )
        | (
            market["normalized_return_144"].le(cfg.downside_return_144)
            & market["normalized_return_3"].le(cfg.downside_return_3)
            & (
                market["normalized_return_12"].le(-3.0)
                | market["quote_volume_ratio"].ge(
                    cfg.downside_minimum_quote_ratio,
                )
            )
        )
    )
    seeded_mask = market["symbol"].isin(seeded)
    seeded_tradfi = (
        seeded_mask
        & market["subtype"].astype(str).eq("TradFi")
    )
    seeded_recovery = (
        (
            market["normalized_return_3"].ge(
                cfg.seeded_recovery_minimum_return_3,
            )
            & market["normalized_return_12"].ge(
                cfg.seeded_recovery_minimum_return_12,
            )
        )
        | (
            market["normalized_return_12"].ge(
                cfg.seeded_recovery_minimum_return_12,
            )
            & market["normalized_return_36"].ge(
                cfg.seeded_recovery_minimum_return_36,
            )
        )
    ) & market["normalized_return_144"].ge(
        cfg.seeded_recovery_minimum_return_144,
    )
    # Symbols already present when the replay window opens can be in the
    # recovery phase of a shock that started before the local 24-hour feature
    # horizon.  Requiring the same fresh ``return_288 <= -20`` condition would
    # suppress the first causal recovery transition for those symbols.
    long_state = (
        (~seeded_mask & unseeded_long_state)
        | (seeded_tradfi & seeded_recovery)
    )
    short_state = short_state & (~seeded_mask | seeded_tradfi)
    market["long_state"] = long_state.astype(bool)
    market["short_state"] = short_state.astype(bool)
    tradfi = market["subtype"].astype(str).eq("TradFi")
    market["strong_recovery_state"] = (
        market["normalized_return_288"].le(cfg.deep_drawdown_return_288)
        & market["normalized_return_36"].ge(0.0)
        & market["signed_seller_pressure"].le(-0.50)
        & market["range_location_288"].le(
            cfg.stabilization_maximum_range_location,
        )
        & (
            ~tradfi
            | market["normalized_return_144"].ge(
                cfg.seeded_recovery_minimum_return_144,
            )
        )
    ).astype(bool)
    return market


def _empty_signals() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "signal_id",
            "signal_time",
            "symbol",
            "action",
            "operation_type",
            "signal_price",
            "factor_family",
            "reason",
            "policy_id",
            "market_only",
            "leader_direction_used",
        ],
    )


__all__ = [
    "POLICY_ID",
    "DynamicNonEthLocalStateConfig",
    "build_dynamic_non_eth_local_states",
    "generate_dynamic_non_eth_local_state_signals",
]
