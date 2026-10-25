"""固定 Alpha158 研究的24账户薄适配，只读取冻结预测白名单。"""

import argparse
import gzip
import importlib.metadata
import json
import shutil
import sys
import time
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd

from ..alpha158_study import (
    END_MS,
    FEATURES,
    HOUR_MS,
    IDENTITY,
    METHODS,
    MODELS,
    PHASES,
    PROFILE,
    REFITS,
    TRAINING_START_MS,
)
from ..artifacts import read, save, sha, verify_receipts
from ..regression_run import ms
from .paired_forecast import drive_accounts, select_pairs
from .search import funding_source_path, gate, load_funding, save_account
from .signals import MINUTE, PROTOCOL

COSTS = (3, 10, 11)
PREDICTION_COLUMNS = (
    *IDENTITY,
    "coefficient_mode",
    *(f"prediction_{m}" for m in MODELS),
    "common_member_gate",
)


def validate_predictions(predictions, universe):
    """验证两方式全部原小时三元身份；未用标签或完整未来路径筛成员。"""
    if set(predictions) != set(PREDICTION_COLUMNS) or predictions.columns.has_duplicates:
        raise ValueError("账户只能加载明确的预测及因果人口白名单")
    if predictions[[*IDENTITY, "coefficient_mode", "common_member_gate"]].isna().any().any():
        raise ValueError("账户身份或因果共同人口标志未知")
    if predictions.duplicated([*IDENTITY, "coefficient_mode"]).any() or set(
        predictions.coefficient_mode
    ) != set(METHODS):
        raise ValueError("预测身份重复或方式不完整")
    if any(not isinstance(value, (bool, np.bool_)) for value in predictions.common_member_gate):
        raise ValueError("共同人口标志必须是明确布尔值")
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
        for value in predictions.clock_ms
    ):
        raise ValueError("预测身份时钟必须是整数毫秒")
    clocks = np.arange(
        TRAINING_START_MS + 5 * MINUTE, ms(PHASES["reserved_later_day"]["end"]), HOUR_MS
    )
    if set(predictions.symbol) != set(universe) or len(predictions) != 2 * len(universe) * len(
        clocks
    ):
        raise ValueError("预测没有完整保留两方式原全集小时")
    for (_, symbol), frame in predictions.groupby(["coefficient_mode", "symbol"], sort=False):
        if (
            not np.array_equal(np.sort(frame.clock_ms.to_numpy()), clocks)
            or not frame.asset_class.eq(universe[symbol]["asset_class"]).all()
        ):
            raise ValueError("预测小时或资产类别与原人口不符")
    if predictions.groupby(list(IDENTITY)).common_member_gate.nunique().gt(1).any():
        raise ValueError("两系数方式不能改变因果共同人口")
    for model in MODELS:
        if not pd.api.types.is_numeric_dtype(
            predictions[f"prediction_{model}"]
        ) or pd.api.types.is_bool_dtype(predictions[f"prediction_{model}"]):
            raise ValueError("预测分数必须是实数数值列")
        scores = predictions[f"prediction_{model}"].to_numpy(dtype=float)
        if np.isinf(scores).any() or np.isfinite(scores[~predictions.common_member_gate]).any():
            raise ValueError("预测含无穷数或越过共同人口门")
    counts = predictions.groupby(
        ["coefficient_mode", "asset_class", "clock_ms"]
    ).common_member_gate.sum()
    if ((counts > 0) & (counts < 20)).any():
        raise ValueError("共同人口门不能保留少于20成员的截面")


def build_schedules(predictions, universe):
    """全部身份先验证，再按固定账户区间排除训练及原后续日。"""
    validate_predictions(predictions, universe)
    first = REFITS[0]
    inside = predictions[(predictions.clock_ms >= first) & (predictions.clock_ms < END_MS)]
    assets = sorted({row["asset_class"] for row in universe.values()})
    if len(assets) != 2:
        raise ValueError("固定24账户必须保留两个资产类别")
    cases = {
        f"{method}:{asset}:20m:{model}:{cost}bps": (method, asset, model, cost)
        for method in METHODS
        for asset in assets
        for model in MODELS
        for cost in COSTS
    }
    if len(cases) != 24:
        raise ValueError("预定案例必须恰为24条")
    schedules, statuses = {}, []
    for (method, asset, clock), frame in inside.groupby(
        ["coefficient_mode", "asset_class", "clock_ms"], sort=True
    ):
        common = frame[frame.common_member_gate]
        for model in MODELS:
            score = f"prediction_{model}"
            complete = len(common) >= 20 and np.isfinite(common[score]).all()
            selected = select_pairs(common, score) if complete else []
            schedules[method, asset, model, int(clock)] = selected
            statuses.append(
                {
                    "method": method,
                    "asset_class": asset,
                    "model": model,
                    "clock_ms": int(clock),
                    "common_members": len(common),
                    "selected": selected,
                    "status": "INSUFFICIENT_COMMON_MEMBERS"
                    if len(common) < 20
                    else "INCOMPLETE_COMMON_PREDICTIONS"
                    if not complete
                    else "PAIRED_SIGNAL"
                    if selected
                    else "NO_PAIRED_THRESHOLD_SIGNAL",
                }
            )
    expected = len(METHODS) * len(assets) * len(MODELS) * ((END_MS - first) // HOUR_MS)
    if len(schedules) != expected:
        raise ValueError("账户调度不能省略无交易或失败小时")
    return (
        cases,
        schedules,
        {
            "input_rows": len(predictions),
            "account_rows": len(inside),
            "training_rows_excluded": int((predictions.clock_ms < first).sum()),
            "later_day_rows_excluded": int((predictions.clock_ms >= END_MS).sum()),
            "schedules": statuses,
        },
    )


def _sources(source, market, funding):
    """资金费和分钟市场必须与原748来源链、研究预测完全一致。"""
    verify_receipts(source)
    parent = read(source / "protocol.json")
    if (
        parent.get("schema") != "ta-alpha158-ridge-study.v1"
        or parent.get("account_family") != "alpha158_ridge"
        or parent.get("alpha158_profile") != PROFILE
        or tuple(parent["features"]) != FEATURES
        or {key: tuple(value) for key, value in parent["models"].items()} != MODELS
        or parent["phases"] != PHASES
        or tuple(parent["refit_times_ms"]) != REFITS
        or len(parent["universe"]) != 748
        or any(
            parent[k] is not False
            for k in ("orders_enabled", "leader_labels_read", "production_eligible")
        )
    ):
        raise ValueError("账户只能使用完整固定Alpha158研究及原748人口")
    label_root = Path(parent["label_source"]).resolve()
    label_protocol = label_root / "protocol.json"
    if parent["input_identities"].get(str(label_protocol)) != sha(label_protocol):
        raise ValueError("原标签来源元数据不等于预测的冻结来源")
    original = read(label_protocol)
    manifest, capture = read(market / "manifest.json"), read(funding / "capture_plan.json")
    first, last = REFITS[0], END_MS
    if (
        manifest["contracts"] != parent["universe"]
        or original["universe"] != parent["universe"]
        or original.get("schema") != "ta-native-settlement-unwind-study.v1"
        or any(original[k] is not False for k in ("orders_enabled", "leader_labels_read"))
        or manifest["market_panel_contract"]["timestamp_semantics"] != "CLOSE_TIME"
        or any(manifest[k] is not False for k in ("orders_enabled", "leader_labels_read"))
        or Path(capture["panel"]).resolve() != market
        or capture["manifest_sha256"] != sha(market / "manifest.json")
        or ms(capture["start"]) > first
        or ms(capture["end"]) < last
        or set(capture["funding_symbols"]) != set(parent["universe"])
        or any(capture[k] is not False for k in ("orders_enabled", "leader_labels_read"))
    ):
        raise ValueError("市场、资金费覆盖或原全集身份不同")
    identities = {
        str(source / name): sha(source / name)
        for name in ("protocol.json", "receipts.json", "predictions.parquet")
    }
    identities[str(label_protocol)] = sha(label_protocol)
    identities[str(market / "manifest.json")] = sha(market / "manifest.json")
    for name in ("capture_plan.json", "funding_manifest.json", "funding_events.json"):
        path = funding / name
        if original["input_identities"].get(str(path)) != sha(path):
            raise ValueError("账户资金费不同于原九项特征及标签来源")
        identities[str(path)] = sha(path)
    funding_manifest = read(funding / "funding_manifest.json")
    for leaf in funding_manifest["leaves"]:
        path = funding_source_path(funding, leaf)
        if sha(path) != leaf["sha256"]:
            raise ValueError("原始资金费身份改变")
        identities[str(path)] = leaf["sha256"]
    for record in manifest["daily_files"]:
        if record["last_close_ms"] < first or record["first_close_ms"] > last:
            continue
        path = (market / record["path"]).resolve()
        if not path.is_relative_to(market) or sha(path) != record["sha256"]:
            raise ValueError("分钟市场路径或原文身份不符")
        identities[str(path)] = record["sha256"]
    return parent, manifest, identities


def market_frames(market, manifest):
    """只加载分钟开收及可用性；完整时间轴由共享驱动器再次核验。"""
    start, end = (pd.Timestamp(value, unit="ms", tz="UTC") for value in (REFITS[0], END_MS))
    for record in sorted(manifest["daily_files"], key=lambda row: row["first_close_ms"]):
        if record["last_close_ms"] < REFITS[0] or record["first_close_ms"] > END_MS:
            continue
        data = pd.read_parquet(
            market / record["path"],
            columns=["open", "close", "eligible", "observed"],
            filters=[("bar_time", ">=", start), ("bar_time", "<=", end)],
        )
        for at, rows in data.groupby(level="bar_time", sort=True):
            current = rows.droplevel("bar_time")
            if (
                at.value % 1_000_000
                or current.index.has_duplicates
                or set(current.index) != set(manifest["contracts"])
            ):
                raise ValueError("分钟市场必须完整保留原748身份")
            if (
                not pd.api.types.is_bool_dtype(current.eligible)
                or not pd.api.types.is_bool_dtype(current.observed)
                or current[["eligible", "observed"]].isna().any().any()
            ):
                raise ValueError("市场可用性必须是明确布尔标志")
            yield (
                int(at.value // 1_000_000),
                dict(
                    zip(
                        current.index,
                        zip(
                            current.open,
                            current.close,
                            current.eligible,
                            current.observed,
                            strict=True,
                        ),
                        strict=True,
                    )
                ),
            )


def _save_ledgers(output, cases, accounts, invalid):
    """正常账本复用原保存器；无曲线/失败账户不伪造零收益指标。"""
    for key in cases:
        path = output / (key.replace(":", "_") + ".json.gz")
        if path.exists():
            continue
        account = accounts.get(key)
        if account is not None and key not in invalid and account.curve:
            save_account(path, account)
        else:
            ledger = {
                "case": cases[key],
                "execution_started": account is not None,
                "numeric_conclusions_allowed": False,
                "metrics": None,
                "ending_state": None
                if account is None
                else {
                    "cash": account.cash,
                    "positions": account.positions,
                    "pending": account.pending,
                    "closes": account.closes,
                    "halted": account.halted,
                    "peak": account.peak,
                    "maximum_drawdown": account.max_dd,
                },
                **{
                    name: getattr(account, field) if account is not None else []
                    for name, field in (
                        ("trades", "trades"),
                        ("fills", "fills"),
                        ("funding", "funding_ledger"),
                        ("curve", "curve"),
                        ("declines", "declines"),
                    )
                },
            }
            with gzip.open(path, "xt", encoding="utf-8") as stream:
                json.dump(ledger, stream, ensure_ascii=False, allow_nan=False)


def run(source, market_panel, funding_root, output):
    """先封存规则、源码和精确消费来源，再调度全部24条连续账户。"""
    source, market, funding, output = (
        Path(p).resolve() for p in (source, market_panel, funding_root, output)
    )
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in (source, market, funding)):
        raise ValueError("新账户不能覆盖或包含原来源")
    parent, manifest, identities = _sources(source, market, funding)
    cases = {
        f"{method}:{asset}:20m:{model}:{cost}bps": (method, asset, model, cost)
        for method in METHODS
        for asset in sorted({r["asset_class"] for r in parent["universe"].values()})
        for model in MODELS
        for cost in COSTS
    }
    if len(cases) != 24:
        raise ValueError("必须恰为24账户")
    package = Path(__file__).resolve().parents[3]
    paths = sorted(
        set(package.rglob("*.py")) | set((package / "research/quant/plans").glob("*.md"))
    )
    frozen = {str(path): sha(path) for path in paths}
    output.mkdir(parents=True, exist_ok=False)
    save(
        output / "protocol.json",
        {
            "schema": "ta-alpha158-ridge-account.v1",
            "created_ms": time.time_ns() // 1_000_000,
            "start_ms": REFITS[0],
            "end_ms": END_MS,
            "cases": cases,
            "prediction_columns_loaded": PREDICTION_COLUMNS,
            "future_labels_loaded": False,
            "source_sha256": frozen,
            "input_identities": identities,
            "predicted_return_threshold": 0.0016,
            "holding_minutes": 20,
            "timed_exit_at_close": True,
            "take_profit_fraction": None,
            "account_rules": {
                key: PROTOCOL[key]
                for key in (
                    "initial_equity",
                    "position_fraction",
                    "max_positions",
                    "decision_latency_ms",
                    "max_signal_age_ms",
                    "fee_bps",
                    "maximum_account_drawdown",
                    "minimum_trade_notional",
                )
            },
            "funding_entry_same_clock_goes_to_old_inventory": True,
            "funding_exit_same_clock_excluded_from_close_exit": True,
            "helper_reuse_scope": "Account/Audit/开盘全批次取消/资金费/保存/经济gate；settlement.blocks仅通用连续账本分块汇总，未调用H01的run/决策/系数",
            "quantity_is_continuous": True,
            "all_or_none_available_open_batch": True,
            "independent_oos": False,
            "real_time_forward_required": False,
            "orders_enabled": False,
            "leader_labels_read": False,
            "production_eligible": False,
            "runtime": {
                "python": sys.version,
                "libraries": {
                    name: importlib.metadata.version(name)
                    for name in ("numpy", "pandas", "pyarrow")
                },
            },
        },
    )
    accounts, invalid, stage = {}, dict.fromkeys(cases), "freeze_source"
    try:
        for path in paths:
            target = output / "source" / path.relative_to(package.parent)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        stage = "prediction_schedule"
        predictions = pd.read_parquet(
            source / "predictions.parquet", columns=list(PREDICTION_COLUMNS)
        )
        actual_cases, schedules, schedule_report = build_schedules(predictions, parent["universe"])
        if actual_cases != cases:
            raise ValueError("调度案例与预先封存的24账户不符")
        save(output / "decisions.json", schedule_report)
        stage = "minute_accounts"
        accounts, result = drive_accounts(
            cases,
            schedules,
            market_frames(market, manifest),
            load_funding(funding),
            (*REFITS, END_MS),
        )
        invalid = result["invalid_accounts"]
        _save_ledgers(output, cases, accounts, invalid)
        if result["global_failure"]:
            save(output / "partial_result.json", result)
            raise ValueError("分钟输入失败，所有账户数字失效")
        stage = "verify_after"
        verify_receipts(source)
        if any(sha(path) != identity for path, identity in (identities | frozen).items()):
            raise ValueError("账户计算期间来源或源码发生改变")
        result.update(
            schema="ta-alpha158-ridge-account-result.v1",
            economic_conditions_by_account={
                key: gate(value) for key, value in result["metrics"].items()
            },
            stable_profitability_proven=False,
            orders_enabled=False,
            production_eligible=False,
            future_labels_loaded=False,
            independent_oos=False,
            case_execution_status={
                key: "INVALID_ACCOUNT" if key in invalid else "FINANCIALLY_VERIFIED"
                for key in cases
            },
        )
        save(output / "result.json", result)
    except Exception as error:
        save(
            output / "failure.json",
            {
                "stage": stage,
                "type": type(error).__name__,
                "reason": str(error),
                "numeric_conclusions_allowed": False,
            },
        )
        raise
    finally:
        _save_ledgers(output, cases, accounts, invalid)
        save(
            output / "receipts.json",
            {
                str(path.relative_to(output)): sha(path)
                for path in output.rglob("*")
                if path.is_file()
            },
        )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "market-panel", "funding-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.source, args.market_panel, args.funding_root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
