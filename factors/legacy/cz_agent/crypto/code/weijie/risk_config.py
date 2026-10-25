"""Risk sizing for Weijie research and the V192 market runtime.

``resolve_dynamic_risk_sizing`` is the only production sizing entry: it uses
equity, closed-bar ATR, signal quality and a small risk configuration.  The
walk-forward profile helpers below remain offline research utilities and are
never imported by the V192 live policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from factors.crypto.code.weijie.common import weijie_asset_group
from factors.crypto.code.weijie.event_features import event_session_family

RISK_PROFILE_COLUMNS = [
    "scope",
    "asset_group",
    "symbol",
    "action",
    "reason_family",
    "session_family",
    "entry_phase",
    "source_priority",
    "sample_count",
    "median_quantity",
    "median_notional_usdt",
    "p75_notional_usdt",
    "p90_notional_usdt",
    "max_notional_usdt",
    "median_leverage",
    "median_risk_basis_usdt",
    "p75_risk_basis_usdt",
    "p90_risk_basis_usdt",
    "max_risk_basis_usdt",
    "source",
]


@dataclass(frozen=True)
class RiskSizingDecision:
    quantity: float
    notional_usdt: float
    leverage: float
    risk_basis_usdt: float
    margin_mode: str
    sizing_reason: str
    margin_pct: float = 0.0
    risk_budget_pct: float = 0.0
    risk_budget_usdt: float = 0.0
    stop_distance_pct: float = 0.0
    atr_pct: float = 0.0


def resolve_dynamic_risk_sizing(
    *,
    symbol: str,
    action: str,
    account_equity_usdt: float,
    price: float,
    atr_pct: float,
    position_lot_count: int,
    quality_multiplier: float,
    risk_budget_pct: float,
    atr_stop_multiplier: float,
    min_stop_distance_pct: float,
    max_stop_distance_pct: float,
    max_single_margin_pct: float,
    max_leverage: float,
    position_lot_risk_multiplier: float = 0.65,
    execution_margin_stop_pct: float = 0.0,
    target_margin_pct: float = 0.0,
) -> RiskSizingDecision:
    """Size one position from equity, closed-bar ATR and a fixed loss budget.

    The output is scale-free: ``margin_pct`` is later applied to the current
    live equity and to the timestamped backtest equity by the same account
    ledger.  No historical position or leader sizing record is consulted.
    """

    equity = _finite_float(account_equity_usdt, 0.0)
    price_value = _finite_float(price, 0.0)
    atr_value = _finite_float(atr_pct, 0.0)
    if equity <= 0.0 or price_value <= 0.0 or atr_value <= 0.0:
        return RiskSizingDecision(0.0, 0.0, 1.0, 0.0, "CROSSED", "dynamic_risk:invalid_market_input")

    min_stop = max(0.05, _finite_float(min_stop_distance_pct, 0.80))
    max_stop = max(min_stop, _finite_float(max_stop_distance_pct, 4.00))
    stop_distance = float(
        np.clip(
            atr_value * max(0.10, _finite_float(atr_stop_multiplier, 2.20)),
            min_stop,
            max_stop,
        ),
    )
    lot_multiplier = float(
        np.clip(
            _finite_float(position_lot_risk_multiplier, 0.65),
            0.25,
            2.00,
        ),
    ) ** max(0, int(position_lot_count))
    quality = float(np.clip(_finite_float(quality_multiplier, 1.0), 0.75, 1.25))
    effective_risk_pct = max(0.01, _finite_float(risk_budget_pct, 0.35)) * quality * lot_multiplier
    risk_budget = equity * effective_risk_pct / 100.0

    configured_leverage_cap = max(1.0, _finite_float(max_leverage, 1.0))
    stop_margin_loss_cap = 45.0
    stop_safe_leverage = max(1.0, np.floor(stop_margin_loss_cap / stop_distance))
    leverage = float(min(configured_leverage_cap, stop_safe_leverage))

    margin_cap = equity * max(0.10, _finite_float(max_single_margin_pct, 3.0)) / 100.0
    target_margin = equity * max(0.0, _finite_float(target_margin_pct, 0.0)) / 100.0
    if target_margin > 0.0:
        notional = min(target_margin, margin_cap) * leverage
    else:
        notional = min(risk_budget / (stop_distance / 100.0), margin_cap * leverage)
    margin = notional / leverage
    margin_pct = margin / equity * 100.0
    actual_risk_budget = notional * stop_distance / 100.0
    quantity = notional / price_value
    execution_margin_stop = max(
        0.0,
        _finite_float(execution_margin_stop_pct, 0.0),
    )
    execution_stop_distance = (
        execution_margin_stop / leverage
        if execution_margin_stop > 0.0 and leverage > 0.0
        else stop_distance
    )
    asset_group = weijie_asset_group(symbol)
    return RiskSizingDecision(
        quantity=float(quantity),
        notional_usdt=float(notional),
        leverage=leverage,
        risk_basis_usdt=float(margin),
        margin_mode="CROSSED",
        sizing_reason=(
            f"dynamic_atr_risk:{asset_group};action={str(action).upper()};"
            f"atr={atr_value:.4f}%;sizing_stop={stop_distance:.4f}%;"
            f"execution_stop={execution_stop_distance:.4f}%;risk={actual_risk_budget:.4f}U"
        ),
        margin_pct=float(margin_pct),
        risk_budget_pct=float(actual_risk_budget / equity * 100.0),
        risk_budget_usdt=float(actual_risk_budget),
        stop_distance_pct=float(execution_stop_distance),
        atr_pct=atr_value,
    )


def build_walk_forward_risk_profile(
    open_ops: pd.DataFrame,
    close_ops: pd.DataFrame,
    open_matches: pd.DataFrame,
) -> pd.DataFrame:
    """Learn pre-eval risk sizing priors from matched train labels."""

    samples = _build_train_risk_samples(open_ops, close_ops, open_matches)
    return _build_risk_profile_from_samples(samples)


def build_public_operation_risk_profile(operations: pd.DataFrame) -> pd.DataFrame:
    """Learn incremental OPEN/ADD margin priors from pre-evaluation public fills."""

    if operations is None or operations.empty:
        return pd.DataFrame(columns=RISK_PROFILE_COLUMNS)
    frame = operations.copy()
    frame["事件时间"] = pd.to_datetime(frame.get("事件时间"), errors="coerce")
    frame["事件类型"] = frame.get("事件类型", pd.Series(dtype=str)).astype(str).str.upper().str.strip()
    frame["标的"] = frame.get("标的", pd.Series(dtype=str)).astype(str).str.upper().str.strip()
    frame["方向"] = frame.get("方向", pd.Series(dtype=str)).astype(str).str.strip()
    for column in ("成交价", "成交数量", "杠杆", "增量保证金估算"):
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    frame = frame[
        frame["事件类型"].isin({"OPEN", "ADD", "FLIP_OPEN"})
        & frame["事件时间"].notna()
        & frame["标的"].astype(bool)
        & frame["方向"].isin({"做多", "做空"})
        & frame["成交价"].gt(0.0)
        & frame["成交数量"].gt(0.0)
        & frame["杠杆"].gt(0.0)
    ].copy()
    if frame.empty:
        return pd.DataFrame(columns=RISK_PROFILE_COLUMNS)
    notional = frame["成交价"] * frame["成交数量"]
    risk_basis = frame["增量保证金估算"].where(
        frame["增量保证金估算"].gt(0.0),
        notional / frame["杠杆"],
    )
    samples = pd.DataFrame(
        {
            "symbol": frame["标的"],
            "asset_group": frame["标的"].map(weijie_asset_group),
            "action": np.where(frame["方向"].eq("做多"), "OPEN_LONG", "OPEN_SHORT"),
            "reason_family": "*",
            "session_family": frame["事件时间"].map(event_session_family),
            "entry_phase": np.where(frame["事件类型"].eq("ADD"), "ADD", "OPEN"),
            "quantity": frame["成交数量"],
            "open_price": frame["成交价"],
            "notional_usdt": notional,
            "leverage": frame["杠杆"],
            "risk_basis_usdt": risk_basis,
            "source": "public_trade_history_operation",
        },
    )
    return _build_risk_profile_from_samples(samples, operation_only=True)


def _build_risk_profile_from_samples(
    samples: pd.DataFrame,
    *,
    operation_only: bool = False,
) -> pd.DataFrame:
    if samples is None or samples.empty:
        return pd.DataFrame(columns=RISK_PROFILE_COLUMNS)
    rows: list[dict[str, Any]] = []
    operation_group_specs = [
        ("symbol_action_entry_phase_session", ["symbol", "action", "entry_phase", "session_family"]),
        ("symbol_action_entry_phase", ["symbol", "action", "entry_phase"]),
        ("asset_group_action_entry_phase", ["asset_group", "action", "entry_phase"]),
        ("asset_group_entry_phase", ["asset_group", "entry_phase"]),
    ]
    group_specs = operation_group_specs if operation_only else [
        ("symbol_action_family_session", ["symbol", "action", "reason_family", "session_family"]),
        ("symbol_action_family", ["symbol", "action", "reason_family"]),
        ("symbol_action", ["symbol", "action"]),
        ("symbol", ["symbol"]),
        ("asset_group_action_family", ["asset_group", "action", "reason_family"]),
        ("asset_group_action", ["asset_group", "action"]),
        ("asset_group", ["asset_group"]),
        ("action_family", ["action", "reason_family"]),
        ("action", ["action"]),
        ("global", []),
    ]
    for scope, columns in group_specs:
        if columns:
            grouped = samples.groupby(columns, dropna=False, sort=True)
            for key, group in grouped:
                values = key if isinstance(key, tuple) else (key,)
                payload = dict(zip(columns, values, strict=False))
                rows.append(_risk_profile_row(group, scope=scope, payload=payload))
        else:
            rows.append(_risk_profile_row(samples, scope=scope, payload={}))

    profile = pd.DataFrame(rows, columns=RISK_PROFILE_COLUMNS)
    return profile.sort_values(
        ["scope", "symbol", "action", "reason_family", "session_family", "entry_phase"],
    ).reset_index(drop=True)


def resolve_risk_sizing(
    risk_profile: pd.DataFrame,
    *,
    symbol: str,
    action: str,
    reason_family: str,
    signal_time: pd.Timestamp,
    price: float,
    signal_score: float | None = None,
    event_score: float | None = None,
    event_pressure: float | None = None,
    ret20: float | None = None,
    ret60: float | None = None,
    ret144: float | None = None,
    ret288: float | None = None,
    don: float | None = None,
    taker: float | None = None,
    seller: float | None = None,
    qvr: float | None = None,
    rsi: float | None = None,
    adx: float | None = None,
    position_lot_count: int = 0,
    non_anchor_leverage_cap: float | None = 8.5,
    anchor_short_leverage_cap: float | None = None,
) -> RiskSizingDecision:
    """Resolve quantity/notional/leverage for a live signal from learned priors."""

    clean_symbol = str(symbol or "").upper().strip()
    clean_action = str(action or "").upper().strip()
    family = str(reason_family or "").strip()
    asset_group = weijie_asset_group(clean_symbol)
    signal_ts = pd.Timestamp(signal_time) if pd.notna(signal_time) else pd.NaT
    session = event_session_family(signal_ts) if pd.notna(signal_ts) else ""
    entry_phase = "ADD" if int(position_lot_count) > 0 else "OPEN"
    price_value = _finite_float(price, np.nan)
    context = {
        "symbol": clean_symbol,
        "action": clean_action,
        "reason_family": family,
        "asset_group": asset_group,
        "session_family": session,
        "entry_phase": entry_phase,
        "hour": float(signal_ts.hour) if pd.notna(signal_ts) else np.nan,
        "signal_score": _finite_float(signal_score, np.nan),
        "event_score": _finite_float(event_score, np.nan),
        "event_pressure": _finite_float(event_pressure, np.nan),
        "ret20": _finite_float(ret20, np.nan),
        "ret60": _finite_float(ret60, np.nan),
        "ret144": _finite_float(ret144, np.nan),
        "ret288": _finite_float(ret288, np.nan),
        "don": _finite_float(don, np.nan),
        "taker": _finite_float(taker, np.nan),
        "seller": _finite_float(seller, np.nan),
        "qvr": _finite_float(qvr, np.nan),
        "rsi": _finite_float(rsi, np.nan),
        "adx": _finite_float(adx, np.nan),
    }
    if risk_profile.empty or not np.isfinite(price_value) or price_value <= 0.0:
        return RiskSizingDecision(0.0, 0.0, 20.0, 0.0, "CROSSED", "no_train_risk_profile")

    for scope, matcher in (
        (
            "symbol_action_entry_phase_session",
            {
                "symbol": clean_symbol,
                "action": clean_action,
                "entry_phase": entry_phase,
                "session_family": session,
            },
        ),
        (
            "symbol_action_entry_phase",
            {"symbol": clean_symbol, "action": clean_action, "entry_phase": entry_phase},
        ),
        (
            "asset_group_action_entry_phase",
            {"asset_group": asset_group, "action": clean_action, "entry_phase": entry_phase},
        ),
        ("asset_group_entry_phase", {"asset_group": asset_group, "entry_phase": entry_phase}),
        ("action_entry_phase", {"action": clean_action, "entry_phase": entry_phase}),
        ("entry_phase", {"entry_phase": entry_phase}),
        (
            "symbol_action_family_session",
            {
                "symbol": clean_symbol,
                "action": clean_action,
                "reason_family": family,
                "session_family": session,
            },
        ),
        ("symbol_action_family", {"symbol": clean_symbol, "action": clean_action, "reason_family": family}),
        ("symbol_action", {"symbol": clean_symbol, "action": clean_action}),
        ("symbol", {"symbol": clean_symbol}),
        (
            "asset_group_action_family",
            {"asset_group": asset_group, "action": clean_action, "reason_family": family},
        ),
        ("asset_group_action", {"asset_group": asset_group, "action": clean_action}),
        ("asset_group", {"asset_group": asset_group}),
        ("action_family", {"action": clean_action, "reason_family": family}),
        ("action", {"action": clean_action}),
        ("global", {}),
    ):
        row = _first_profile_match(risk_profile, scope=scope, matcher=matcher)
        if row is None:
            continue
        decision = _decision_from_profile_row(
            row,
            price=price_value,
            scope=scope,
            asset_group=asset_group,
            symbol=clean_symbol,
            action=clean_action,
            non_anchor_leverage_cap=non_anchor_leverage_cap,
            anchor_short_leverage_cap=anchor_short_leverage_cap,
        )
        decision = _maybe_adjust_contextual_sizing(
            decision,
            risk_profile,
            context=context,
            price=price_value,
        )
        if decision.quantity > 0.0 and decision.notional_usdt > 0.0 and decision.risk_basis_usdt > 0.0:
            return decision

    return RiskSizingDecision(0.0, 0.0, 20.0, 0.0, "CROSSED", "unresolved_train_risk_profile")


def _build_train_risk_samples(
    open_ops: pd.DataFrame,
    close_ops: pd.DataFrame,
    open_matches: pd.DataFrame,
) -> pd.DataFrame:
    if open_ops.empty:
        return pd.DataFrame()
    opens = open_ops.copy()
    opens["op_id"] = opens.get("op_id", pd.Series(dtype=str)).astype(str)
    opens["label_id"] = opens.get("label_id", pd.Series(dtype=str)).astype(str)
    opens["event_time"] = pd.to_datetime(opens.get("event_time"), errors="coerce")
    opens["avg_price"] = pd.to_numeric(opens.get("avg_price"), errors="coerce")
    opens["qty"] = pd.to_numeric(opens.get("qty"), errors="coerce")
    opens["leverage"] = pd.to_numeric(opens.get("leverage"), errors="coerce")
    opens["notional_usdt"] = opens["qty"] * opens["avg_price"]

    close_basis = _close_risk_basis_by_label(close_ops)
    match_by_op = _open_match_by_op(open_matches)
    rows: list[dict[str, Any]] = []
    for _, row in opens.iterrows():
        symbol = str(row.get("symbol") or "").upper().strip()
        action = str(row.get("action") or "").upper().strip()
        qty = _finite_float(row.get("qty"), 0.0)
        price = _finite_float(row.get("avg_price"), np.nan)
        notional = _finite_float(row.get("notional_usdt"), np.nan)
        leverage = _finite_float(row.get("leverage"), np.nan)
        if not symbol or action not in {"OPEN_LONG", "OPEN_SHORT"} or qty <= 0.0 or notional <= 0.0:
            continue
        if not np.isfinite(leverage) or leverage <= 0.0:
            leverage = 20.0
        label_id = str(row.get("label_id") or "")
        risk_basis = _finite_float(close_basis.get(label_id), np.nan)
        if not np.isfinite(risk_basis) or risk_basis <= 0.0:
            risk_basis = notional / leverage
        match = match_by_op.get(str(row.get("op_id") or ""))
        reason = str(match.get("reason") or "") if match else ""
        reason_family = reason.split(";", 1)[0] if reason else "*"
        signal_time = pd.Timestamp(match.get("signal_time")) if match and pd.notna(match.get("signal_time")) else row.get("event_time")
        session = event_session_family(pd.Timestamp(signal_time)) if pd.notna(signal_time) else ""
        rows.append(
            {
                "symbol": symbol,
                "asset_group": weijie_asset_group(symbol),
                "action": action,
                "reason_family": reason_family or "*",
                "session_family": session,
                "entry_phase": "OPEN",
                "quantity": qty,
                "open_price": price,
                "notional_usdt": notional,
                "leverage": leverage,
                "risk_basis_usdt": risk_basis,
                "source": "matched_train_open_factor" if match else "train_open_label_fallback",
            },
        )
    return pd.DataFrame(rows)


def _close_risk_basis_by_label(close_ops: pd.DataFrame) -> dict[str, float]:
    if close_ops.empty or "label_id" not in close_ops:
        return {}
    closes = close_ops.copy()
    closes["label_id"] = closes["label_id"].astype(str)
    closes["pnl"] = pd.to_numeric(closes.get("pnl"), errors="coerce")
    closes["risk_roi_ratio"] = pd.to_numeric(closes.get("risk_roi_ratio"), errors="coerce")
    closes["qty"] = pd.to_numeric(closes.get("qty"), errors="coerce")
    closes["avg_price"] = pd.to_numeric(closes.get("avg_price"), errors="coerce")
    closes["leverage"] = pd.to_numeric(closes.get("leverage"), errors="coerce")
    out: dict[str, float] = {}
    for _, row in closes.iterrows():
        pnl = _finite_float(row.get("pnl"), np.nan)
        roi_ratio = _finite_float(row.get("risk_roi_ratio"), np.nan)
        basis = np.nan
        if np.isfinite(pnl) and np.isfinite(roi_ratio) and abs(roi_ratio) > 1e-12:
            basis = pnl / roi_ratio
        if not np.isfinite(basis) or basis <= 0.0:
            qty = _finite_float(row.get("qty"), np.nan)
            price = _finite_float(row.get("avg_price"), np.nan)
            leverage = _finite_float(row.get("leverage"), np.nan)
            if qty > 0.0 and price > 0.0 and leverage > 0.0:
                basis = qty * price / leverage
        if np.isfinite(basis) and basis > 0.0:
            out[str(row.get("label_id"))] = float(basis)
    return out


def _open_match_by_op(matches: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if matches.empty or "op_id" not in matches:
        return {}
    frame = matches[matches.get("status", pd.Series(dtype=str)).astype(str).eq("MATCH")].copy()
    return {str(row.get("op_id")): dict(row) for _, row in frame.iterrows()}


def _risk_profile_row(group: pd.DataFrame, *, scope: str, payload: dict[str, Any]) -> dict[str, Any]:
    quantity = pd.to_numeric(group.get("quantity"), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    notional = pd.to_numeric(group.get("notional_usdt"), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    leverage = pd.to_numeric(group.get("leverage"), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    risk_basis = pd.to_numeric(group.get("risk_basis_usdt"), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    asset = str(payload.get("asset_group") or "")
    symbol = str(payload.get("symbol", "") or "")
    if not asset or asset == "*":
        asset = weijie_asset_group(symbol) if symbol and symbol != "*" else "*"
    return {
        "scope": scope,
        "asset_group": asset or "*",
        "symbol": symbol or "*",
        "action": str(payload.get("action", "*") or "*"),
        "reason_family": str(payload.get("reason_family", "*") or "*"),
        "session_family": str(payload.get("session_family", "*") or "*"),
        "entry_phase": str(payload.get("entry_phase", "*") or "*"),
        "source_priority": _risk_source_priority(group),
        "sample_count": len(group),
        "median_quantity": _finite_float(quantity.median(), 0.0) if not quantity.empty else 0.0,
        "median_notional_usdt": _finite_float(notional.median(), 0.0) if not notional.empty else 0.0,
        "p75_notional_usdt": _finite_float(notional.quantile(0.75), 0.0) if not notional.empty else 0.0,
        "p90_notional_usdt": _finite_float(notional.quantile(0.90), 0.0) if not notional.empty else 0.0,
        "max_notional_usdt": _finite_float(notional.max(), 0.0) if not notional.empty else 0.0,
        "median_leverage": _finite_float(leverage.median(), 20.0) if not leverage.empty else 20.0,
        "median_risk_basis_usdt": _finite_float(risk_basis.median(), 0.0) if not risk_basis.empty else 0.0,
        "p75_risk_basis_usdt": _finite_float(risk_basis.quantile(0.75), 0.0) if not risk_basis.empty else 0.0,
        "p90_risk_basis_usdt": _finite_float(risk_basis.quantile(0.90), 0.0) if not risk_basis.empty else 0.0,
        "max_risk_basis_usdt": _finite_float(risk_basis.max(), 0.0) if not risk_basis.empty else 0.0,
        "source": ",".join(sorted(set(group.get("source", pd.Series(dtype=str)).astype(str))))[:160],
    }


def _first_profile_match(
    profile: pd.DataFrame,
    *,
    scope: str,
    matcher: dict[str, str],
) -> pd.Series | None:
    frame = profile[profile.get("scope", pd.Series(dtype=str)).astype(str).eq(scope)].copy()
    if frame.empty:
        return None
    for column, value in matcher.items():
        if column not in frame:
            return None
        frame = frame[frame[column].astype(str).eq(str(value))]
        if frame.empty:
            return None
    frame["sample_count"] = pd.to_numeric(frame["sample_count"], errors="coerce").fillna(0)
    if "source_priority" in frame:
        frame["source_priority"] = pd.to_numeric(frame["source_priority"], errors="coerce").fillna(9)
    else:
        frame["source_priority"] = 9
    return frame.sort_values(["source_priority", "sample_count"], ascending=[True, False]).iloc[0]


def _risk_source_priority(group: pd.DataFrame) -> int:
    sources = set(group.get("source", pd.Series(dtype=str)).astype(str))
    if any("public_trade_history_operation" in source for source in sources):
        return 0
    if any("matched_train_open_factor" in source for source in sources):
        return 1
    return 2


def _decision_from_profile_row(
    row: pd.Series,
    *,
    price: float,
    scope: str,
    asset_group: str,
    symbol: str,
    action: str,
    non_anchor_leverage_cap: float | None,
    anchor_short_leverage_cap: float | None,
) -> RiskSizingDecision:
    leverage = _finite_float(row.get("median_leverage"), 20.0)
    if not np.isfinite(leverage) or leverage <= 0.0:
        leverage = 20.0
    cap = _production_leverage_cap(
        asset_group,
        symbol,
        action=action,
        configured_cap=non_anchor_leverage_cap,
        anchor_short_cap=anchor_short_leverage_cap,
    )
    cap_reason = ""
    if np.isfinite(cap) and cap > 0.0 and leverage > cap:
        leverage = float(cap)
        cap_reason = f";lev_cap={cap:g}x"
    risk_basis = _finite_float(row.get("median_risk_basis_usdt"), np.nan)
    notional = _finite_float(row.get("median_notional_usdt"), np.nan)
    if (not np.isfinite(risk_basis) or risk_basis <= 0.0) and np.isfinite(notional) and notional > 0.0:
        risk_basis = notional / leverage
    if cap_reason and np.isfinite(notional) and notional > 0.0:
        risk_basis = notional / leverage
    elif (np.isfinite(risk_basis) and risk_basis > 0.0 and cap_reason) or ((not np.isfinite(notional) or notional <= 0.0) and np.isfinite(risk_basis) and risk_basis > 0.0):
        notional = risk_basis * leverage
    if not np.isfinite(notional) or notional <= 0.0 or not np.isfinite(risk_basis) or risk_basis <= 0.0:
        return RiskSizingDecision(0.0, 0.0, leverage, 0.0, "CROSSED", f"{scope}:invalid_profile_row")
    quantity = notional / float(price)
    return RiskSizingDecision(
        quantity=float(quantity),
        notional_usdt=float(notional),
        leverage=float(leverage),
        risk_basis_usdt=float(risk_basis),
        margin_mode="CROSSED",
        sizing_reason=f"{scope}:median_train_risk{cap_reason}",
    )


def _maybe_adjust_contextual_sizing(
    decision: RiskSizingDecision,
    risk_profile: pd.DataFrame,
    *,
    context: dict[str, float | str],
    price: float,
) -> RiskSizingDecision:
    symbol = str(context.get("symbol") or "")
    if symbol != "ETHUSDT" or decision.notional_usdt <= 0.0:
        return decision
    target_notional, reason = _eth_contextual_notional(risk_profile, context, decision.notional_usdt)
    if not np.isfinite(target_notional) or target_notional <= 0.0:
        return decision
    notional = float(target_notional)
    leverage = float(decision.leverage) if decision.leverage > 0.0 else 30.0
    quantity = notional / float(price)
    risk_basis = notional / leverage
    return RiskSizingDecision(
        quantity=float(quantity),
        notional_usdt=notional,
        leverage=leverage,
        risk_basis_usdt=float(risk_basis),
        margin_mode=decision.margin_mode,
        sizing_reason=f"{decision.sizing_reason};{reason}",
    )


def _eth_contextual_notional(
    risk_profile: pd.DataFrame,
    context: dict[str, float | str],
    current_notional: float,
) -> tuple[float, str]:
    action = str(context.get("action") or "")
    family = str(context.get("reason_family") or "")
    session = str(context.get("session_family") or "")
    hour = _finite_float(context.get("hour"), np.nan)
    ret20 = _finite_float(context.get("ret20"), np.nan)
    ret60 = _finite_float(context.get("ret60"), np.nan)
    ret144 = _finite_float(context.get("ret144"), np.nan)
    ret288 = _finite_float(context.get("ret288"), np.nan)
    don = _finite_float(context.get("don"), np.nan)
    taker = _finite_float(context.get("taker"), np.nan)
    seller = _finite_float(context.get("seller"), np.nan)
    qvr = _finite_float(context.get("qvr"), np.nan)
    rsi = _finite_float(context.get("rsi"), np.nan)
    adx = _finite_float(context.get("adx"), np.nan)
    event_score = _finite_float(context.get("event_score"), np.nan)
    signal_score = _finite_float(context.get("signal_score"), np.nan)

    broad_session = _first_profile_match(
        risk_profile,
        scope="symbol_action_family_session",
        matcher={
            "symbol": "ETHUSDT",
            "action": action,
            "reason_family": "*",
            "session_family": session,
        },
    )
    broad_action = _first_profile_match(
        risk_profile,
        scope="symbol_action",
        matcher={"symbol": "ETHUSDT", "action": action},
    )
    family_profile = _first_profile_match(
        risk_profile,
        scope="symbol_action_family",
        matcher={"symbol": "ETHUSDT", "action": action, "reason_family": family},
    )
    broad_session_or_action = broad_session if broad_session is not None else broad_action

    def profile_value(row: pd.Series | None, field: str) -> float:
        if row is None:
            return np.nan
        return _finite_float(row.get(field), np.nan)

    def choose(row: pd.Series | None, field: str, tag: str, *, floor: bool = True) -> tuple[float, str]:
        value = profile_value(row, field)
        if not np.isfinite(value) or value <= 0.0:
            return np.nan, ""
        if floor:
            value = max(float(current_notional), float(value))
        return float(value), tag

    if action == "OPEN_LONG":
        if family == "eth_anchor_long":
            flow_led_anchor = (
                np.isfinite(hour)
                and 20 <= int(hour) <= 22
                and np.isfinite(ret20)
                and -0.10 <= ret20 <= 0.60
                and np.isfinite(ret60)
                and -0.60 <= ret60 <= 0.10
                and np.isfinite(ret144)
                and -0.80 <= ret144 <= 0.10
                and np.isfinite(ret288)
                and -0.30 <= ret288 <= 0.50
                and np.isfinite(don)
                and 0.70 <= don <= 0.90
                and np.isfinite(taker)
                and taker >= 0.62
                and np.isfinite(qvr)
                and 1.0 <= qvr <= 3.0
                and np.isfinite(rsi)
                and 56.0 <= rsi <= 62.0
                and np.isfinite(adx)
                and 15.0 <= adx <= 25.0
            )
            if flow_led_anchor:
                value, tag = choose(
                    broad_session_or_action,
                    "p75_notional_usdt",
                    "eth_tail=p75_flow_led_anchor",
                )
                if tag:
                    return value, tag
        if family == "range_reclaim_long":
            range_night_chop = (
                np.isfinite(hour)
                and int(hour) in {19, 21}
                and np.isfinite(ret288)
                and ret288 < 1.0
                and np.isfinite(rsi)
                and rsi < 62.0
            )
            range_low_adx_chop = (
                np.isfinite(adx)
                and adx < 18.0
                and np.isfinite(qvr)
                and qvr < 2.0
                and np.isfinite(ret60)
                and ret60 < 0.10
            )
            if session == "regular":
                value, tag = choose(
                    broad_session_or_action,
                    "max_notional_usdt",
                    "eth_tail=cap_max_regular_range",
                    floor=False,
                )
            elif session == "night_trend":
                value, tag = choose(
                    broad_session_or_action,
                    "p75_notional_usdt",
                    "eth_tail=cap_p75_night_range",
                    floor=False,
                )
            else:
                value, tag = choose(broad_action, "p90_notional_usdt", "eth_tail=cap_p90_range", floor=False)
            if tag:
                if range_night_chop:
                    return max(float(value) * 0.25, 1000.0), f"{tag};eth_tail=throttle_range_night_chop"
                if range_low_adx_chop:
                    return max(float(value) * 0.35, 1000.0), f"{tag};eth_tail=throttle_range_low_adx_chop"
                return value, tag
        if family == "core_trend_long":
            weak_session_core_trend = (
                np.isfinite(hour)
                and int(hour) in {5, 6, 16, 23}
                and np.isfinite(qvr)
                and qvr < 3.0
                and np.isfinite(ret144)
                and ret144 < 3.0
            )
            if weak_session_core_trend:
                return float(current_notional), "eth_tail=throttle_weak_session_core_trend"
            low_adx_failed_core_trend = (
                np.isfinite(adx)
                and adx < 15.0
                and np.isfinite(qvr)
                and qvr < 3.0
                and np.isfinite(ret60)
                and ret60 < 1.0
                and np.isfinite(ret144)
                and ret144 < 0.0
            )
            low_commitment_late_core = (
                np.isfinite(qvr)
                and qvr < 1.20
                and np.isfinite(ret60)
                and ret60 >= 2.50
                and np.isfinite(rsi)
                and rsi < 62.0
                and np.isfinite(adx)
                and adx >= 30.0
            )
            if low_adx_failed_core_trend or low_commitment_late_core:
                reason = (
                    "eth_tail=throttle_core_low_adx_failed"
                    if low_adx_failed_core_trend
                    else "eth_tail=throttle_core_low_commitment_late"
                )
                return max(float(current_notional) * 0.15, 1000.0), reason
            fragile_early_tail = (
                np.isfinite(hour)
                and int(hour) in {1, 2, 5, 6, 17, 23}
                and np.isfinite(ret60)
                and ret60 < 2.20
                and np.isfinite(qvr)
                and qvr < 5.0
                and np.isfinite(ret144)
                and ret144 < 4.0
            )
            overheat_low_qvr_tail = (
                np.isfinite(hour)
                and int(hour) in {0, 1, 5, 6, 15, 17, 21, 23}
                and np.isfinite(rsi)
                and rsi >= 69.0
                and np.isfinite(qvr)
                and qvr < 5.0
                and np.isfinite(ret60)
                and ret60 < 2.50
            )
            single_bar_spike_tail = (
                np.isfinite(qvr)
                and qvr >= 20.0
                and np.isfinite(rsi)
                and rsi >= 78.0
                and np.isfinite(adx)
                and adx < 30.0
            )
            core_tail_throttle = fragile_early_tail or overheat_low_qvr_tail or single_bar_spike_tail
            overextended_low_commitment = (
                np.isfinite(event_score)
                and event_score < 0.05
                and np.isfinite(ret60)
                and ret60 >= 4.5
                and np.isfinite(ret288)
                and ret288 >= 6.0
                and (not np.isfinite(qvr) or qvr < 1.5)
            )
            if overextended_low_commitment:
                return float(current_notional), "eth_tail=guard_overextended_core_trend"
            low_liquidity_overextended_core = (
                np.isfinite(ret60)
                and ret60 >= 4.0
                and np.isfinite(ret288)
                and ret288 >= 6.0
                and np.isfinite(qvr)
                and qvr < 1.50
                and np.isfinite(rsi)
                and rsi >= 68.0
                and np.isfinite(adx)
                and adx >= 45.0
            )
            if low_liquidity_overextended_core:
                return float(current_notional), "eth_tail=guard_low_liquidity_overextended_core"
            if (
                (np.isfinite(don) and don >= 0.80)
                or (np.isfinite(qvr) and qvr >= 2.0)
                or (np.isfinite(signal_score) and signal_score >= 4.5)
            ):
                value, tag = choose(broad_session_or_action, "p90_notional_usdt", "eth_tail=p90_core_trend")
                if tag:
                    if core_tail_throttle:
                        if fragile_early_tail:
                            reason = "eth_tail=throttle_core_fragile_tail"
                        elif overheat_low_qvr_tail:
                            reason = "eth_tail=throttle_core_overheat_low_qvr"
                        else:
                            reason = "eth_tail=throttle_core_single_bar_spike"
                        return max(float(value) * 0.35, 1000.0), f"{tag};{reason}"
                    return value, tag
        if family == "capitulation_absorption_long":
            high_qvr_false_absorption = (
                np.isfinite(qvr)
                and qvr >= 8.0
                and np.isfinite(rsi)
                and rsi <= 30.0
                and np.isfinite(adx)
                and adx < 32.0
                and np.isfinite(ret60)
                and ret60 <= -1.0
                and np.isfinite(ret288)
                and ret288 > -0.50
            )
            if high_qvr_false_absorption:
                return max(float(current_notional) * 0.08, 1000.0), "eth_tail=throttle_cap_high_qvr_false_absorption"
            weak_midday_capitulation = (
                np.isfinite(hour)
                and int(hour) in {9, 11, 14, 15, 17, 19, 23}
                and np.isfinite(qvr)
                and qvr < 8.0
            )
            if weak_midday_capitulation:
                cap = profile_value(broad_action, "median_notional_usdt")
                if np.isfinite(cap) and cap > 0.0:
                    return min(float(current_notional), float(cap)), "eth_tail=throttle_weak_midday_capitulation"
                return float(current_notional), "eth_tail=throttle_weak_midday_capitulation"
            weak_false_flush = (
                np.isfinite(qvr)
                and qvr < 1.20
                and np.isfinite(adx)
                and adx >= 35.0
                and np.isfinite(ret288)
                and ret288 > -0.20
            )
            midday_low_qvr_flush = (
                np.isfinite(hour)
                and int(hour) in {1, 4, 9, 10, 14, 15, 17, 19, 23}
                and np.isfinite(qvr)
                and qvr < 3.0
                and np.isfinite(ret288)
                and ret288 > -3.0
            )
            if weak_false_flush or midday_low_qvr_flush:
                reason = "eth_tail=throttle_cap_false_flush" if weak_false_flush else "eth_tail=throttle_cap_midday_low_qvr"
                return max(float(current_notional) * 0.35, 1000.0), reason
            if (
                session == "night_trend"
                and np.isfinite(ret60)
                and ret60 <= -1.0
                and np.isfinite(rsi)
                and rsi <= 35.0
            ):
                cap = profile_value(broad_action, "median_notional_usdt")
                if np.isfinite(cap) and cap > 0.0:
                    return min(float(current_notional), float(cap)), "eth_tail=cap_low_bounce"
            if session in {"regular", "asia_morning"} and (
                (np.isfinite(ret60) and ret60 >= 0.0)
                or (np.isfinite(qvr) and qvr >= 1.5)
                or (np.isfinite(event_score) and event_score >= 0.15)
            ):
                value, tag = choose(broad_session_or_action, "max_notional_usdt", "eth_tail=max_capitulation")
                if tag:
                    return value, tag
            if session == "night_trend" and np.isfinite(qvr) and qvr >= 4.0 and ret60 > -1.5:
                value, tag = choose(broad_session_or_action, "p75_notional_usdt", "eth_tail=p75_night_capitulation")
                if tag:
                    return value, tag

    if action == "OPEN_SHORT" and family == "eth_anchor_exhaustion_short":
        late_blowoff_short = (
            np.isfinite(ret60)
            and ret60 >= 4.50
            and np.isfinite(ret288)
            and ret288 >= 6.0
            and np.isfinite(rsi)
            and rsi >= 78.0
        )
        climax_exhaustion_short = (
            not late_blowoff_short
            and np.isfinite(rsi)
            and rsi >= 78.0
            and np.isfinite(ret60)
            and 0.80 <= ret60 <= 3.50
            and np.isfinite(qvr)
            and qvr >= 12.0
            and np.isfinite(adx)
            and adx >= 30.0
        )
        trend_reversal_exhaustion_short = (
            not late_blowoff_short
            and np.isfinite(event_score)
            and event_score >= 0.18
            and np.isfinite(ret60)
            and 1.50 <= ret60 <= 3.80
            and np.isfinite(rsi)
            and rsi >= 63.0
            and np.isfinite(adx)
            and adx >= 32.0
        )
        if climax_exhaustion_short:
            value, tag = choose(family_profile, "p90_notional_usdt", "eth_tail=p90_climax_exhaustion_short")
            if tag:
                return value, tag
        if trend_reversal_exhaustion_short:
            value, tag = choose(family_profile, "p90_notional_usdt", "eth_tail=p90_trend_reversal_exhaustion_short")
            if tag:
                return value, tag
        underheated_exhaustion = (
            np.isfinite(rsi)
            and rsi < 65.0
            and np.isfinite(ret60)
            and ret60 < 1.50
            and np.isfinite(qvr)
            and qvr >= 3.0
        )
        if underheated_exhaustion:
            return max(float(current_notional) * 0.35, 1000.0), "eth_tail=throttle_underheated_exhaustion_short"

    if action == "OPEN_SHORT" and family == "weak_24h_bounce_short":
        low_trend_bounce_short = (
            np.isfinite(adx)
            and adx < 15.0
            and np.isfinite(rsi)
            and rsi < 62.0
            and np.isfinite(ret144)
            and ret144 <= -1.50
        )
        if low_trend_bounce_short:
            return max(float(current_notional) * 0.35, 1000.0), "eth_tail=throttle_low_trend_bounce_short"

    if action == "OPEN_SHORT" and family == "core_downtrend_short":
        bad_hour_downtrend = (
            np.isfinite(hour)
            and int(hour) in {3, 7, 8, 12, 13, 14, 17, 20}
            and not (
                int(hour) == 14
                and np.isfinite(ret20)
                and ret20 >= 2.0
                and np.isfinite(ret288)
                and ret288 <= -5.0
                and np.isfinite(adx)
                and adx >= 35.0
            )
            and not (int(hour) == 0 and np.isfinite(ret60) and ret60 <= -3.0)
        )
        low_commitment_downtrend = (
            np.isfinite(seller)
            and seller < 0.55
            and np.isfinite(qvr)
            and qvr < 5.0
            and np.isfinite(adx)
            and adx < 32.0
            and not (
                np.isfinite(hour)
                and int(hour) == 0
                and np.isfinite(ret60)
                and ret60 <= -3.0
            )
        )
        if bad_hour_downtrend or low_commitment_downtrend:
            reason = "eth_tail=throttle_downtrend_bad_hour" if bad_hour_downtrend else "eth_tail=throttle_downtrend_low_commitment"
            return max(float(current_notional) * 0.35, 1000.0), reason
        weak_hour_downtrend_tail = (
            np.isfinite(hour)
            and int(hour) in {1, 5, 10, 16, 19, 21, 22}
            and (not np.isfinite(event_score) or event_score < 0.35)
        )
        if weak_hour_downtrend_tail:
            return float(current_notional), "eth_tail=throttle_weak_hour_downtrend_short"
        weak_rebound_short = (
            np.isfinite(ret20)
            and ret20 >= 2.0
            and (not np.isfinite(qvr) or qvr < 1.5)
            and (not np.isfinite(event_score) or event_score < 0.20)
        )
        low_adx_shallow_break = (
            np.isfinite(adx)
            and adx < 20.5
            and (not np.isfinite(ret60) or ret60 > -2.0)
            and (not np.isfinite(qvr) or qvr < 4.0)
        )
        if weak_rebound_short or low_adx_shallow_break:
            tag = "eth_tail=guard_weak_rebound_short" if weak_rebound_short else "eth_tail=guard_low_adx_break"
            return float(current_notional), tag
        strong_downtrend_tail = (
            (np.isfinite(event_score) and event_score >= 0.20 and np.isfinite(ret60) and ret60 <= -3.0)
            or (
                np.isfinite(ret288)
                and ret288 <= -5.0
                and (
                    (np.isfinite(seller) and seller >= 0.70)
                    or (np.isfinite(qvr) and qvr >= 2.0)
                    or (np.isfinite(event_score) and event_score >= 0.20)
                )
            )
            or (
                np.isfinite(seller)
                and seller >= 0.70
                and np.isfinite(qvr)
                and qvr >= 2.0
                and np.isfinite(adx)
                and adx >= 20.5
            )
        )
        if strong_downtrend_tail:
            value, tag = choose(broad_session_or_action, "max_notional_usdt", "eth_tail=max_downtrend_short")
            if tag:
                return value, tag
        value, tag = choose(broad_action, "p75_notional_usdt", "eth_tail=p75_downtrend_short")
        if tag:
            return value, tag

    return np.nan, ""


def _production_leverage_cap(
    asset_group: str,
    symbol: str,
    *,
    action: str = "",
    configured_cap: float | None = 8.5,
    anchor_short_cap: float | None = None,
) -> float:
    asset = str(asset_group or "")
    if not asset or asset == "*":
        asset = weijie_asset_group(symbol)
    if asset == "eth":
        cap = _finite_float(anchor_short_cap, np.nan)
        if str(action or "").upper().strip() == "OPEN_SHORT" and np.isfinite(cap) and cap > 0.0:
            return cap
        return np.nan
    cap = _finite_float(configured_cap, np.nan)
    return cap if np.isfinite(cap) and cap > 0.0 else np.nan


def _finite_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return out
