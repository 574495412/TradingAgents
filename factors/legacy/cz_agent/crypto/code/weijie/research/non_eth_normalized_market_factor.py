"""只用闭合市场数据运行非 ETH 波动率归一化反趋势研究。"""

# ruff: noqa: RUF002

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from factors.crypto.code.weijie import market_features as market_features_module
from factors.crypto.code.weijie.large_cap.countertrend import (
    CountertrendFactorPolicy,
    VolatilityNormalizedCountertrendPolicy,
    evaluate_volatility_normalized_countertrend,
)
from factors.crypto.code.weijie.large_cap.countertrend import (
    scoring as countertrend_scoring_module,
)
from factors.crypto.code.weijie.large_cap.countertrend import (
    volatility_normalized as normalized_lifecycle_module,
)
from factors.crypto.code.weijie.large_cap.reversal_lifecycle import (
    CONTEXT_SYMBOLS,
    build_unified_regime_market_panel,
)
from factors.crypto.code.weijie.large_cap.reversal_lifecycle import (
    market as lifecycle_market_module,
)

NON_ETH_NORMALIZED_MARKET_FACTOR_POLICY_ID = (
    "weijie_volatility_normalized_countertrend_cross_asset_v2"
)


@dataclass(frozen=True)
class NonEthNormalizedMarketFactorReplay:
    """一个冻结参数版本的跨标的市场动作与完整持仓。"""

    policy_id: str
    policy_spec: Mapping[str, object]
    positions: pd.DataFrame
    execution_actions: pd.DataFrame
    summaries: pd.DataFrame


def run_non_eth_normalized_market_factor_replay(
    raw_frames: Mapping[str, pd.DataFrame],
    *,
    target_symbols: tuple[str, ...],
    execution_start: object,
    factor_policy: CountertrendFactorPolicy | None = None,
    lifecycle_policy: VolatilityNormalizedCountertrendPolicy | None = None,
) -> NonEthNormalizedMarketFactorReplay:
    """运行统一无量纲参数，不接受带单标签、标的特例或收益目标。"""

    targets = _normalized_non_eth_targets(target_symbols)
    frames = {
        _clean_symbol(symbol): frame.copy()
        for symbol, frame in raw_frames.items()
        if frame is not None and not frame.empty
    }
    missing_context = sorted(set(CONTEXT_SYMBOLS).difference(frames))
    if missing_context:
        message = "normalized non-ETH replay missing context: "
        raise ValueError(message + ", ".join(missing_context))
    missing_targets = sorted(set(targets).difference(frames))
    if missing_targets:
        message = "normalized non-ETH replay missing targets: "
        raise ValueError(message + ", ".join(missing_targets))
    start = _utc_timestamp(execution_start)
    factor = factor_policy or CountertrendFactorPolicy()
    lifecycle = (
        lifecycle_policy or VolatilityNormalizedCountertrendPolicy()
    )
    policy_spec = _frozen_policy_spec(
        factor_policy=factor,
        lifecycle_policy=lifecycle,
    )

    position_frames: list[pd.DataFrame] = []
    action_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for target in targets:
        required_context = tuple(dict.fromkeys((*CONTEXT_SYMBOLS, target)))
        panel = build_unified_regime_market_panel(
            frames,
            target_symbol=target,
            required_context_symbols=required_context,
        )
        panel = panel.loc[
            panel["countertrend_data_valid"].astype(bool)
        ].copy()
        result = evaluate_volatility_normalized_countertrend(
            panel,
            factor_policy=factor,
            lifecycle_policy=lifecycle,
            execution_start=start,
            close_open_at_window_end=False,
        )
        actions = _namespace_identifiers(
            result.actions,
            target_symbol=target,
        )
        positions = _namespace_identifiers(
            result.positions,
            target_symbol=target,
        )
        if not actions.empty:
            actions["policy_id"] = (
                NON_ETH_NORMALIZED_MARKET_FACTOR_POLICY_ID
            )
            actions["evidence_kind"] = (
                "forward_market_only_volatility_normalized"
            )
            action_frames.append(actions)
        if not positions.empty:
            positions["policy_id"] = (
                NON_ETH_NORMALIZED_MARKET_FACTOR_POLICY_ID
            )
            positions["evidence_kind"] = (
                "forward_market_only_volatility_normalized"
            )
            positions["strategy_evidence_kind"] = (
                "market_only_research_replay"
            )
            position_frames.append(positions)
        summary_rows.append(
            {
                "symbol": target,
                "policy_id": NON_ETH_NORMALIZED_MARKET_FACTOR_POLICY_ID,
                "closed_position_count": len(positions),
                "action_count": len(actions),
                "open_action_count": (
                    int(actions["operation_type"].astype(str).eq("OPEN").sum())
                    if not actions.empty
                    else 0
                ),
                "net_point_return_pct": (
                    float(
                        pd.to_numeric(
                            positions["net_return_pct"],
                            errors="coerce",
                        ).sum(),
                    )
                    if not positions.empty
                    else 0.0
                ),
                "market_only": True,
                "point_research_only": True,
                "status": "REPLAYED",
            },
        )
    return NonEthNormalizedMarketFactorReplay(
        policy_id=NON_ETH_NORMALIZED_MARKET_FACTOR_POLICY_ID,
        policy_spec=policy_spec,
        positions=(
            pd.concat(position_frames, ignore_index=True)
            if position_frames
            else pd.DataFrame()
        ),
        execution_actions=(
            pd.concat(action_frames, ignore_index=True)
            if action_frames
            else pd.DataFrame()
        ),
        summaries=pd.DataFrame.from_records(summary_rows),
    )


def _frozen_policy_spec(
    *,
    factor_policy: CountertrendFactorPolicy,
    lifecycle_policy: VolatilityNormalizedCountertrendPolicy,
) -> dict[str, object]:
    spec = {
        "spec_schema_version": 1,
        "factor_policy": asdict(factor_policy),
        "lifecycle_policy": asdict(lifecycle_policy),
        "required_context_symbols": list(CONTEXT_SYMBOLS),
        "entry_basis": (
            "ATR-normalized return impulse, cross-sectional rank, breadth, "
            "RSI/ADX, volatility/range and quote-volume expansion"
        ),
        "lifecycle_basis": (
            "entry-time ATR distances with broad economic risk bounds"
        ),
        "runtime_input_boundary": (
            "closed OHLCV only; no leader events, labels, ROI or symbol rules"
        ),
        "close_open_at_window_end": False,
        "runtime_source_sha256": _runtime_source_sha256(),
    }
    try:
        return json.loads(
            json.dumps(
                spec,
                allow_nan=False,
                ensure_ascii=True,
                sort_keys=True,
            ),
        )
    except (TypeError, ValueError) as exc:
        message = "normalized non-ETH policy spec must be JSON serializable"
        raise ValueError(message) from exc


def _runtime_source_sha256() -> str:
    source_paths = (
        Path(__file__),
        Path(str(market_features_module.__file__)),
        Path(str(lifecycle_market_module.__file__)),
        Path(str(countertrend_scoring_module.__file__)),
        Path(str(normalized_lifecycle_module.__file__)),
    )
    digest = hashlib.sha256()
    for source_path in source_paths:
        digest.update(source_path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _namespace_identifiers(
    frame: pd.DataFrame,
    *,
    target_symbol: str,
) -> pd.DataFrame:
    output = frame.copy()
    if output.empty:
        return output
    prefix = f"{_clean_symbol(target_symbol)}|"
    for column in ("chain_id", "signal_id"):
        if column not in output:
            continue
        values = output[column]
        populated = values.notna() & values.astype(str).ne("")
        output.loc[populated, column] = (
            prefix + values.loc[populated].astype(str)
        )
    return output


def _normalized_non_eth_targets(
    target_symbols: tuple[str, ...],
) -> tuple[str, ...]:
    targets = tuple(
        sorted(
            {
                _clean_symbol(symbol)
                for symbol in target_symbols
                if _clean_symbol(symbol)
            },
        ),
    )
    if not targets:
        raise ValueError("normalized non-ETH replay requires target symbols")
    if "ETHUSDT" in targets:
        raise ValueError("normalized non-ETH replay must not include ETHUSDT")
    return targets


def _clean_symbol(value: object) -> str:
    return (
        str(value or "")
        .upper()
        .replace("/", "")
        .replace("-", "")
        .strip()
    )


def _utc_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("Asia/Shanghai").tz_convert("UTC")
    return timestamp.tz_convert("UTC")


__all__ = [
    "NON_ETH_NORMALIZED_MARKET_FACTOR_POLICY_ID",
    "NonEthNormalizedMarketFactorReplay",
    "run_non_eth_normalized_market_factor_replay",
]
