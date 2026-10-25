"""Publish actual MAX/momentum controls, continuous losses and measured beta."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.regime_program.reporting import partitions

NAMES = {
    "max28": "MAX28极端收益延续",
    "max28_btc_beta": "MAX28＋BTC估计对冲",
    "momentum28_control": "普通28日动量对照",
    "momentum28_btc_beta_control": "普通动量＋BTC估计对冲",
    "cash": "原封存现金对照",
}


def actual_beta(curve, events, bars, features):
    times = pd.DatetimeIndex(curve.date)
    beta = np.zeros(len(times))
    for s, b in bars.items():
        delta = np.zeros(len(times))
        for e in events:
            if e["symbol"] == s and e["type"] in ["entry", "exit"]:
                delta[times.get_loc(pd.Timestamp(e["time"]))] += (
                    e["side"] * e["quantity"] * (1 if e["type"] == "entry" else -1)
                )
        quantity = np.cumsum(delta)
        if not np.any(quantity):
            continue
        prior = features.xs(s).beta60.reindex(times, method="ffill").to_numpy()
        prices = b.reindex(times - pd.Timedelta(minutes=5)).to_numpy()
        if not np.isfinite(prior).all() or not np.isfinite(prices).all():
            raise ValueError("Actual inventory beta history missing")
        beta += quantity * prices * prior / curve.equity.to_numpy()
    active = curve.gross_exposure.to_numpy() > 1e-8
    return {
        "mean_absolute_inventory_estimated_beta": float(np.abs(beta).mean()),
        "mean_absolute_inventory_estimated_beta_when_active": float(np.abs(beta[active]).mean())
        if active.any()
        else None,
        "maximum_absolute_inventory_estimated_beta": float(np.abs(beta).max()),
        "active_5m_intervals": int(active.sum()),
        "note": "Actual filled inventory times prior60day estimated beta, not target neutrality and not future realised beta. Daily realised BTC regression is separately reported in metrics.beta.",
    }


def publish(run, restart, output):  # noqa: C901
    receipt, plan, source = (
        read(run / "completion.json"),
        read(run / "plan.json"),
        read(run / "summary.json"),
    )
    if (
        sha(run / "summary.json") != receipt["summary_sha256"]
        or sha(run / "plan.json") != receipt["plan_sha256"]
        or not source["audit"]["passed"]
    ):
        raise ValueError("Account summary or audit changed")
    for path, digest in plan["account_source_hashes"].items():
        if sha(Path(path)) != digest:
            raise ValueError("Execution source changed after account run")
    check = read(restart / "audit.json")
    if (
        not check["passed"]
        or check["run_completion_sha256"] != sha(run / "completion.json")
        or len(check["checks"]) != 6
        or not all(r["passed"] for r in check["checks"])
    ):
        raise ValueError("Actual process continuation incomplete")
    states_root = Path("results/research/regime_states_20261025_v002")
    for name, digest in read(states_root / "completion.json")["manifest"].items():
        if Path(name).name != name or sha(states_root / name) != digest:
            raise ValueError("Causal state source changed")
    states = pd.read_parquet(states_root / "states_daily.parquet")
    flags = pd.read_parquet(states_root / "event_risk_hourly.parquet").covered_fomc_post_release
    inputs = Path(plan["inputs"])
    features = pd.read_parquet(inputs / "features_daily.parquet")
    data = Path(plan["data"])
    symbols = read(data / "prepare_plan.json")["symbols"]
    bars = {
        s: pd.read_parquet(data / (s + "_5m.parquet"), columns=["close"]).close for s in symbols
    }
    rows, details, curves, bindings = [], {}, {}, {}
    for row in source["account_rows"]:
        folder = Path(row["source_run"]) / "accounts" / row["source_account_id"]
        if sha(folder / "result.json") != row["result_sha256"]:
            raise ValueError("Account identity changed")
        account = read(folder / "result.json")
        if account["metrics"] != row["metrics"]:
            raise ValueError("Account metrics changed")
        for name, digest in account["source_hashes"].items():
            if Path(name).name != name or sha(folder / name) != digest:
                raise ValueError("Account ledger changed")
        curve = pd.read_parquet(folder / "equity_5m.parquet")
        events = read(folder / "events.json")
        detail = partitions(curve, account["trades"], states, flags, row["capital"])
        for group in ["regime", "covered_fomc"]:
            if (
                abs(
                    sum(
                        v["net_pnl_contribution"]
                        for v in detail["state_contributions"]
                        if v["group"] == group
                    )
                    - row["capital"] * row["metrics"]["net_return"]
                )
                > 1e-5
            ):
                raise ValueError("State contribution does not reconcile")
        yearly = [r["net_return"] for r in detail["periods"] if r["kind"] == "year"]
        if abs(float(np.prod([1 + r for r in yearly])) - 1 - row["metrics"]["net_return"]) > 1e-10:
            raise ValueError("Annual returns do not compound to account")
        settled = sum(p["net_pnl"] for p in account["open_positions"].values())
        open_total = settled + row["metrics"]["open_pnl_with_exit_reserve"]
        if (
            abs(
                row["metrics"]["closed_net_pnl"]
                + open_total
                - row["capital"] * row["metrics"]["net_return"]
            )
            > 1e-5
        ):
            raise ValueError("Closed and open campaign PnL differs")
        if not row["reused_reference"]:
            audit_path = folder / "independent_audit.json"
            a = read(audit_path)
            if not a["passed"] or a["result_sha256"] != row["result_sha256"]:
                raise ValueError("Financial audit binding changed")
            bindings[str(audit_path)] = sha(audit_path)
        detail["actual_beta"] = actual_beta(curve, events, bars, features)
        reserves = sum(
            p["q"]
            * bars[s].loc[pd.Timestamp(row["end"], tz="UTC") - pd.Timedelta(minutes=5)]
            * (account["config"]["fee_bps"] + account["config"]["slippage_bps"])
            / 10000
            for s, p in account["open_positions"].items()
        )
        m = row["metrics"]
        price_pnl = (
            row["capital"] * m["net_return"]
            - m["funding_net"]
            + m["fees"]
            + m["slippage_cost"]
            + reserves
        )
        detail["same_executed_path_attribution"] = {
            "pre_fee_funding_slippage_price_pnl": price_pnl,
            "funding_net": m["funding_net"],
            "fees": m["fees"],
            "slippage_cost": m["slippage_cost"],
            "open_exit_reserve": reserves,
            "note": "Accounting decomposition of the actual executed path, not a replay with costs disabled. Stops and fill paths can differ between costs.",
        }
        details[row["id"]] = detail
        rows.append(
            {
                **row,
                "open_campaign_total_net_pnl_with_exit_reserve": open_total,
                "actual_beta": detail["actual_beta"],
            }
        )
        curves[row["id"]] = curve.set_index("date")
    comparisons = []
    for row in rows:
        if row["reused_reference"] or row["candidate"] not in ["max28", "max28_btc_beta"]:
            continue
        control = (
            "momentum28_btc_beta_control"
            if row["candidate"] == "max28_btc_beta"
            else "momentum28_control"
        )
        match = next(
            (
                r
                for r in rows
                if r["candidate"] == control
                and r["phase"] == row["phase"]
                and r["capital"] == row["capital"]
                and r["slippage_bps"] == row["slippage_bps"]
            ),
            None,
        )
        if match is None:
            continue
        x, y = curves[row["id"]], curves[match["id"]]
        dr = pd.concat(
            [
                x.equity.resample("D").first().pct_change(fill_method=None).rename("max"),
                y.equity.resample("D").first().pct_change(fill_method=None).rename("control"),
            ],
            axis=1,
        ).dropna()
        active = x.gross_exposure.resample("D").max().shift(1).reindex(dr.index).gt(
            1e-8
        ) & y.gross_exposure.resample("D").max().shift(1).reindex(dr.index).gt(1e-8)
        common = dr.loc[active]
        corr = common["max"].corr(common.control) if len(common) >= 20 else None
        comparisons.append(
            {
                "account": row["id"],
                "control": match["id"],
                "return_difference_percentage_points": 100
                * (row["metrics"]["net_return"] - match["metrics"]["net_return"]),
                "joint_active_days": int(active.sum()),
                "joint_active_daily_correlation": float(corr)
                if corr is not None and pd.notna(corr)
                else None,
                "note": "Matched initial budget, opportunity set and engine; realised stops, inventory, exposure and cost paths differ. Difference is descriptive and does not prove an independent alpha mechanism.",
            }
        )
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        **source,
        "schema": "max-effect-publication.v1",
        "study_kind": "max_effect",
        "study_description": "公开来源MAX28正向延续，与同期限普通动量及各自BTC估计对冲比较；周频调仓，实际毫秒资金费",
        "candidate_names": NAMES,
        "account_rows": rows,
        "details": details,
        "control_comparisons": comparisons,
        "actual_accounts": source["new_accounts"],
        "reused_reference_accounts": 6,
        "selection": {**source["selection"], "research_followups": []},
        "restart_checks": check["checks"],
        "restart_note": "四个高成本连续账户非空库存断点，以及两个MAX账户跨年断点，共6次实际退出进程后读取磁盘JSON恢复；净值、账本、请求和最终状态逐项一致。",
        "model_comparison": {
            "available": False,
            "note": "本轮未重训神经网络；MAX与普通动量属于相关价格延续假设，未取得独立收益源证据。",
        },
        "execution": "真实五分钟成交量×1%容量代理；周一及冷启动调仓，退出确认后下一根才可反向。先提交请求、再结算区间内真实资金费、再五分钟收盘成交；NAV含退出成本准备。交易收盘价估值不等同交易所mark净值或清算认证。",
        "changes_vs_previous": "新版本引擎仅用于本轮4个同风险对照；保留旧研究。全部78745条资金费有真实mark与毫秒时间，不再采用小时开盘mark代理。",
        "not_evaluated_reason": "两个MAX连续账户均未满足两成本正收益、正闭合PnL且不停机条件，未追加10万/100万容量账户；空白不表示容量为零。",
        "next_action": "先解释2025风险停机与2026冷启动差异，不事后筛状态、调阈值或拼接账户。未证明两个独立机制前不继续复杂化分配器。",
        "actual_beta_note": "目标估计beta接近零不保证实际中性；实际库存估计暴露和日收益BTC回归分开展示。",
        "reporting_source_hashes": {
            str(Path(__file__)): sha(Path(__file__)),
            str(restart / "audit.json"): sha(restart / "audit.json"),
            str(states_root / "completion.json"): sha(states_root / "completion.json"),
            **bindings,
        },
    }
    save(output / "summary.json", summary)
    save(
        output / "completion.json",
        {
            "summary_sha256": sha(output / "summary.json"),
            "audit_passed": True,
            "run_completion_sha256": sha(run / "completion.json"),
            "restart_audit_sha256": sha(restart / "audit.json"),
        },
    )
    print(
        {
            "published": True,
            "new_accounts": source["new_accounts"],
            "rows": len(rows),
            "comparisons": len(comparisons),
        },
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--restart", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    publish(a.run, a.restart, a.output)
