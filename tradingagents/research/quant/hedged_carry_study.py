"""运行原生H04的冻结回归；单元标签不作为连续账户利润。"""

from __future__ import annotations

import argparse
import math
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..hedged_capability import END, START, HedgedSourceCapability, verified_hedged_capabilities
from .artifacts import read, save, sha, verify_receipts
from .factors.hedged_carry import DAY, HOUR, SLIPPAGES, samples
from .regression import bh_adjust, correlation, daily_stat

ROOT = Path(__file__).resolve().parents[3]
CUTOFF = int(pd.Timestamp("2026-09-01", tz="UTC").timestamp() * 1000)
HYPOTHESIS = "think_alpha_hypothesis_04"
INPUTS = {
    "panel": (
        "data/research/hedged_majors_20261017_v001",
        "95196a829e5996ecb042f01d516daff1b21727c74a30c4f7d0e05e2495dcb1dd",
    ),
    "audit": (
        "results/research/hedged_majors_source_audit_20261017_v002",
        "8caa4c2f99d7489de8d66fff0c55ba5ce3bfc49df221c4ed9f529ee02b566a0b",
    ),
    "native": (
        "results/research/native_hedged_review_20261017_v002",
        "16d349fe8bc3788642ec2e2775cc06152514800426ef0f14a0bf80625f8d58e6",
    ),
}


def fit_ols(train, days):
    """含截距OLS与按日回归得分HAC；空日保持日历位置。"""
    if len(train) < 3:
        return None
    x, y = train.x.to_numpy(), train.y.to_numpy()
    centered = x - x.mean()
    denominator = float(centered @ centered)
    if denominator <= 0:
        return None
    beta = float(centered @ (y - y.mean()) / denominator)
    alpha = float(y.mean() - beta * x.mean())
    error = y - alpha - beta * x
    daily = pd.Series(centered * error, index=train.day).groupby(level=0).sum()
    scores = np.asarray([daily.get(day, 0.0) for day in days])
    meat = float(scores @ scores)
    for lag in range(1, 4):
        meat += 2 * (1 - lag / 4) * float(scores[lag:] @ scores[:-lag])
    se = math.sqrt(max(0.0, meat)) / denominator if len(daily) >= 5 else None
    z = beta / se if se is not None and se > 0 else None
    return {
        "alpha": alpha,
        "beta": beta,
        "training_mean": float(y.mean()),
        "training_x_mean": float(x.mean()),
        "training_n": len(train),
        "daily_scores": dict(zip(days, scores.tolist(), strict=True)),
        "valid_training_days": len(daily),
        "beta_standard_error": se,
        "beta_z": z,
        "beta_p": math.erfc(abs(z) / math.sqrt(2)) if z is not None else None,
        "quintile_cuts": np.quantile(x, [0.2, 0.4, 0.6, 0.8], method="linear").tolist(),
    }


def evaluate(frame, *, start=START, cutoff=CUTOFF, end=END):
    """全部六种情景的十八项检验一并校正，不根据结果缩小家族。"""
    train_days = (
        pd.date_range(
            pd.Timestamp(start, unit="ms", tz="UTC"),
            pd.Timestamp(cutoff - DAY, unit="ms", tz="UTC"),
            freq="D",
        )
        .strftime("%Y-%m-%d")
        .tolist()
    )
    valid_days = (
        pd.date_range(
            pd.Timestamp(cutoff, unit="ms", tz="UTC"),
            pd.Timestamp(end - DAY, unit="ms", tz="UTC"),
            freq="D",
        )
        .strftime("%Y-%m-%d")
        .tolist()
    )
    frame = frame.copy()
    frame["prediction"] = np.nan
    frame["training_mean_prediction"] = np.nan
    frame["quintile"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    cases, tests = {}, []
    for (symbol, slip), group in frame.groupby(["symbol", "slippage_bps"], sort=True):
        identity = f"{symbol}_{slip}bps"
        complete = np.isfinite(group[["x", "y"]]).all(axis=1)
        train = group[complete & group.phase.eq("training")]
        fit = fit_ols(train, train_days)
        known = group[np.isfinite(group.x)]
        if fit is not None:
            frame.loc[known.index, "prediction"] = fit["alpha"] + fit["beta"] * known.x
            frame.loc[known.index, "training_mean_prediction"] = fit["training_mean"]
            frame.loc[known.index, "quintile"] = (
                np.searchsorted(fit["quintile_cuts"], known.x.to_numpy(), side="left") + 1
            )
        group = frame.loc[group.index]
        valid = group[group.phase.eq("validation")]
        usable = valid[np.isfinite(valid[["x", "y", "prediction"]]).all(axis=1)]
        records = usable.to_dict("records")
        mean_gain = daily_stat(
            records,
            lambda r: (
                (r["y"] - r["training_mean_prediction"]) ** 2 - (r["y"] - r["prediction"]) ** 2
            ),
            valid_days,
        )
        zero_gain = daily_stat(
            records, lambda r: r["y"] ** 2 - (r["y"] - r["prediction"]) ** 2, valid_days
        )
        quintiles = []
        for number in range(1, 6):
            members = valid[valid.quintile.eq(number).fillna(False)]
            labels = members[np.isfinite(members.y)]
            quintiles.append(
                {
                    "quintile": number,
                    "feature_members": len(members),
                    "known_labels": len(labels),
                    "unknown_labels": len(members) - len(labels),
                    "mean_unit_net_return": float(labels.y.mean()) if len(labels) else None,
                }
            )
        transitions, changed = 0, 0
        prev = None
        for r in valid.sort_values("clock_ms").to_dict("records"):
            if (
                prev is not None
                and r["clock_ms"] - prev["clock_ms"] == HOUR
                and pd.notna(r["quintile"])
                and pd.notna(prev["quintile"])
            ):
                transitions += 1
                changed += int(r["quintile"] != prev["quintile"])
            prev = r
        pred_known = valid[np.isfinite(valid.prediction)]
        case = {
            "symbol": symbol,
            "slippage_bps": int(slip),
            "fit": fit,
            "scheduled_rows": len(group),
            "phase_counts": group.phase.value_counts().to_dict(),
            "feature_status_counts": group.feature_status.value_counts().to_dict(),
            "label_status_counts": group.label_status.value_counts().to_dict(),
            "validation_scheduled": len(valid),
            "validation_usable": len(usable),
            "mse_gain_vs_training_mean": mean_gain,
            "mse_gain_vs_zero": zero_gain,
            "raw_temporal_ic": correlation(usable.x, usable.y),
            "raw_temporal_rank_ic": correlation(usable.x, usable.y, rank=True),
            "prediction_temporal_rank_ic": correlation(usable.prediction, usable.y, rank=True),
            "quintiles": quintiles,
            "adjacent_transitions": transitions,
            "adjacent_quintile_change_fraction": changed / transitions if transitions else None,
            "validation_predicted_entries_16bps": int(pred_known.prediction.ge(0.0016).sum()),
            "validation_max_prediction": float(pred_known.prediction.max())
            if len(pred_known)
            else None,
            "validation_min_prediction": float(pred_known.prediction.min())
            if len(pred_known)
            else None,
            "descriptive_mean_unit_return": float(usable.y.mean()) if len(usable) else None,
            "economic_account_evaluated": False,
            "production_eligible": False,
        }
        cases[identity] = case
        for name, p, direction in (
            (
                "training_negative_beta",
                fit["beta_p"] if fit else None,
                fit is not None and fit["beta"] < 0,
            ),
            (
                "validation_mse_vs_training_mean",
                mean_gain["p_asymptotic_two_sided"],
                mean_gain["mean"] is not None and mean_gain["mean"] > 0,
            ),
            (
                "validation_mse_vs_zero",
                zero_gain["p_asymptotic_two_sided"],
                zero_gain["mean"] is not None and zero_gain["mean"] > 0,
            ),
        ):
            tests.append(
                {"case": identity, "test": name, "p": p, "direction_pass": bool(direction)}
            )
    adjusted = bh_adjust([r["p"] if r["p"] is not None else 1.0 for r in tests])
    for test, q in zip(tests, adjusted, strict=True):
        test["q"] = q
        test["passed"] = bool(test["p"] is not None and q <= 0.05 and test["direction_pass"])
    for identity, case in cases.items():
        case["predictive_screen_passed"] = all(t["passed"] for t in tests if t["case"] == identity)
    return {
        "cases": cases,
        "primary_tests": tests,
        "primary_bh_family_size": len(tests),
        "economic_account_evaluated": False,
        "production_eligible": False,
        "independent_oos": False,
    }, frame


def run(output):
    """先验证并冻结原生决定、数据与代码，再执行；失败也封存。"""
    roots = {key: ROOT / name for key, (name, _) in INPUTS.items()}
    for key, root in roots.items():
        if sha(root / "receipts.json") != INPUTS[key][1]:
            raise ValueError("H04输入身份变化")
        verify_receipts(root)
        if (root / "failure.json").exists():
            raise ValueError("失败输入不能运行")
    capability = HedgedSourceCapability(
        schema_version="ta-hedged-source-capability.v1",
        panel_root=str(roots["panel"]),
        panel_receipts_sha256=INPUTS["panel"][1],
        audit_root=str(roots["audit"]),
        audit_receipts_sha256=INPUTS["audit"][1],
    )
    verified = verified_hedged_capabilities(capability)
    native = read(roots["native"] / "result.json")
    if (
        native["selected_hypothesis_ids"] != [HYPOTHESIS]
        or native.get("hedged_capabilities_verified") != verified
        or len(native["calls"]) != 9
    ):
        raise ValueError("未绑定H04真实原生选择和同源两腿能力")
    output = Path(output).resolve()
    package = ROOT / "tradingagents"
    if any(
        output.is_relative_to(p) or p.is_relative_to(output) for p in (*roots.values(), package)
    ):
        raise ValueError("输出不能覆盖来源或代码")
    paths = sorted(
        set(package.rglob("*.py")) | set((package / "research/quant/plans").glob("*.md"))
    )
    frozen = {str(p): sha(p) for p in paths}
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    protocol = {
        "schema": "ta-native-hedged-carry-study.v1",
        "hypothesis_id": HYPOTHESIS,
        "start_ms": START,
        "cutoff_ms": CUTOFF,
        "end_exclusive_ms": END,
        "input_receipts": {k: INPUTS[k][1] for k in INPUTS},
        "source_sha256": frozen,
        "primary_bh_family_size": 18,
        "slippages_bps": list(SLIPPAGES),
        "all_history_previously_viewed": True,
        "independent_oos": False,
        "economic_account_evaluated": False,
        "leader_labels_read": False,
        "orders_enabled": False,
        "production_eligible": False,
        "research_owner": "TradingAgents",
    }
    save(output / "protocol.json", protocol)
    try:
        for p in paths:
            target = output / "source" / p.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)
            if sha(target) != frozen[str(p)]:
                raise ValueError("冻结副本身份不符")
        minutes = pq.read_table(roots["panel"] / "minutes.parquet").to_pandas()
        events = read(roots["panel"] / "funding_events.json")
        frame = samples(minutes, events, start=START, cutoff=CUTOFF, end=END)
        result, frame = evaluate(frame)
        if (
            len(frame) != 8784
            or len(result["cases"]) != 6
            or len(result["primary_tests"]) != 18
            or frame.duplicated(["symbol", "clock_ms", "slippage_bps"]).any()
        ):
            raise ValueError("声明人口或检验家族不守恒")
        pq.write_table(
            pa.Table.from_pandas(frame, preserve_index=False),
            output / "samples.parquet",
            compression="zstd",
        )
        result.update(
            status="COMPLETED_DESCRIPTIVE_H04",
            rows=len(frame),
            elapsed_seconds=time.monotonic() - started,
        )
        if verified_hedged_capabilities(capability) != verified:
            raise ValueError("运行中来源能力改变")
        for key, root in roots.items():
            verify_receipts(root)
            if sha(root / "receipts.json") != INPUTS[key][1]:
                raise ValueError("运行中输入身份改变")
        if any(sha(p) != checksum for p, checksum in frozen.items()):
            raise ValueError("运行中源码改变")
        save(output / "result.json", result)
        return result
    except Exception as error:
        save(output / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        save(
            output / "receipts.json",
            {
                str(p.relative_to(output)): sha(p)
                for p in sorted(output.rglob("*"))
                if p.is_file() and p.name != "receipts.json"
            },
        )


def main(argv=None):
    """统一入口仅支持独立新目录，不接交易接口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.output)
    print(
        {
            "rows": result["rows"],
            "cases": len(result["cases"]),
            "screen_passes": sum(c["predictive_screen_passed"] for c in result["cases"].values()),
            "production_eligible": False,
        },
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
