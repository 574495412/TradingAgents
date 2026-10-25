"""按 lot 审计反转生命周期动作的库存变化和配对完整性。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import numpy as np
import pandas as pd

_OPEN: Final = "OPEN"
_CLOSE: Final = "CLOSE"
_DIRECTIONS: Final = {"LONG", "SHORT"}
_EPSILON: Final = 1e-9
_ACTION_PARTS: Final = 2

InventoryKey = tuple[str, str]


def classify_lot_inventory_lifecycle(
    actions: pd.DataFrame,
    *,
    initial_inventory_units: Mapping[InventoryKey, float] | None = None,
    default_units: float = 1.0,
) -> pd.DataFrame:
    """为每条动作标注执行前后库存及其生命周期阶段。"""

    return _classify_lot_inventory_lifecycle(
        actions,
        initial_inventory_units=initial_inventory_units,
        default_units=default_units,
        reject_orphan_close=True,
    )


def audit_lot_inventory_lifecycle(
    actions: pd.DataFrame,
    *,
    initial_inventory_units: Mapping[InventoryKey, float] | None = None,
    default_units: float = 1.0,
) -> pd.DataFrame:
    """汇总库存动作并报告越界、孤立减仓和未闭合持仓。"""

    return _classify_lot_inventory_lifecycle(
        actions,
        initial_inventory_units=initial_inventory_units,
        default_units=default_units,
        reject_orphan_close=False,
    )


def _classify_lot_inventory_lifecycle(  # noqa: PLR0915
    actions: pd.DataFrame,
    *,
    initial_inventory_units: Mapping[InventoryKey, float] | None,
    default_units: float,
    reject_orphan_close: bool,
) -> pd.DataFrame:
    """按时间顺序推进单个 lot 的 OPEN、ADD、REDUCE 和 CLOSE。"""

    _validate_default_units(default_units)
    required = {"signal_time", "signal_id", "symbol", "action"}
    missing = sorted(required.difference(actions.columns))
    if missing:
        message = "inventory lifecycle actions missing columns: " + ", ".join(missing)
        raise ValueError(message)

    frame = actions.copy()
    if frame.empty:
        frame["source_operation_type"] = pd.Series(dtype="object")
        frame["operation_type"] = pd.Series(dtype="object")
        frame["inventory_units"] = pd.Series(dtype="float64")
        frame["inventory_before_units"] = pd.Series(dtype="float64")
        frame["inventory_after_units"] = pd.Series(dtype="float64")
        frame["inventory_integrity_valid"] = pd.Series(dtype="bool")
        return frame

    frame["signal_time"] = _utc_times(frame["signal_time"])
    parsed = frame["action"].map(_parse_action)
    frame["_event_kind"] = parsed.map(lambda value: value[0])
    frame["_direction"] = parsed.map(lambda value: value[1])
    frame["_source_order"] = np.arange(len(frame), dtype="int64")
    frame["_event_priority"] = frame["_event_kind"].map({_CLOSE: 0, _OPEN: 1})
    frame["inventory_units"] = _inventory_units(frame, default_units=default_units)
    if "operation_type" in frame:
        frame["source_operation_type"] = frame["operation_type"].astype(str)
    else:
        frame["source_operation_type"] = ""

    frame = frame.sort_values(
        ["signal_time", "_event_priority", "_source_order"],
        kind="stable",
    ).reset_index(drop=True)
    inventory = _initial_inventory(initial_inventory_units)
    operation_types: list[str] = []
    before_units: list[float] = []
    after_units: list[float] = []
    integrity_valid: list[bool] = []
    for _, row in frame.iterrows():
        key = (str(row["symbol"]), str(row["_direction"]))
        before = float(inventory.get(key, 0.0))
        units = float(row["inventory_units"])
        if row["_event_kind"] == _OPEN:
            operation_type = "OPEN" if before <= _EPSILON else "ADD"
            after = before + units
        else:
            if before <= _EPSILON:
                if not reject_orphan_close:
                    operation_types.append("INVALID_CLOSE")
                    before_units.append(before)
                    after_units.append(before)
                    integrity_valid.append(False)
                    continue
                message = (
                    "inventory lifecycle encountered orphan close: "
                    f"signal_id={row['signal_id']}, key={key}"
                )
                raise ValueError(message)
            after = max(0.0, before - units)
            operation_type = "CLOSE" if after <= _EPSILON else "REDUCE"
        inventory[key] = after
        operation_types.append(operation_type)
        before_units.append(before)
        after_units.append(after)
        integrity_valid.append(True)

    frame["operation_type"] = operation_types
    frame["inventory_before_units"] = before_units
    frame["inventory_after_units"] = after_units
    frame["inventory_integrity_valid"] = integrity_valid
    return frame.drop(
        columns=["_event_kind", "_direction", "_source_order", "_event_priority"],
    )


def _utc_times(values: pd.Series) -> pd.Series:
    """将动作时间转换为严格 UTC 时间序列。"""

    times = pd.to_datetime(values, errors="coerce")
    if bool(times.isna().any()):
        message = "inventory lifecycle actions contain invalid signal_time"
        raise ValueError(message)
    if times.dt.tz is None:
        times = times.dt.tz_localize("Asia/Shanghai")
    return times.dt.tz_convert("UTC")


def _parse_action(value: object) -> tuple[str, str]:
    """从标准动作文本解析操作类型和多空方向。"""

    action = str(value).upper().strip()
    pieces = action.split("_")
    if (
        len(pieces) != _ACTION_PARTS
        or pieces[0] not in {_OPEN, _CLOSE}
        or pieces[1] not in _DIRECTIONS
    ):
        message = f"unsupported inventory lifecycle action: {value}"
        raise ValueError(message)
    return pieces[0], pieces[1]


def _inventory_units(frame: pd.DataFrame, *, default_units: float) -> pd.Series:
    """读取动作仓位单位，并对缺失值使用安全默认值。"""

    if "position_fraction" not in frame:
        return pd.Series(default_units, index=frame.index, dtype="float64")
    units = pd.to_numeric(frame["position_fraction"], errors="coerce").fillna(default_units)
    invalid = ~np.isfinite(units) | units.le(0.0)
    if bool(invalid.any()):
        signal_ids = frame.loc[invalid, "signal_id"].astype(str).tolist()
        message = f"inventory lifecycle units must be positive: {signal_ids}"
        raise ValueError(message)
    return units.astype("float64")


def _initial_inventory(
    values: Mapping[InventoryKey, float] | None,
) -> dict[InventoryKey, float]:
    """按方向初始化零库存账本。"""

    inventory: dict[InventoryKey, float] = {}
    for raw_key, raw_units in (values or {}).items():
        if len(raw_key) != _ACTION_PARTS:
            message = f"invalid inventory lifecycle key: {raw_key}"
            raise ValueError(message)
        symbol, direction = str(raw_key[0]), str(raw_key[1]).upper()
        if direction not in _DIRECTIONS:
            message = f"invalid inventory lifecycle direction: {direction}"
            raise ValueError(message)
        units = float(raw_units)
        if not np.isfinite(units) or units < 0.0:
            message = f"invalid inventory lifecycle units: {raw_units}"
            raise ValueError(message)
        inventory[(symbol, direction)] = units
    return inventory


def _validate_default_units(value: float) -> None:
    """_validate_default_units：校验输入结构、时间顺序与数值边界。"""

    units = float(value)
    if not np.isfinite(units) or units <= 0.0:
        message = "inventory lifecycle default_units must be positive"
        raise ValueError(message)


__all__ = [
    "InventoryKey",
    "audit_lot_inventory_lifecycle",
    "classify_lot_inventory_lifecycle",
]
