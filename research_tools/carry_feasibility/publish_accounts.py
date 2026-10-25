"""Publish sealed segregated accounts, complete costs and sparse-cycle limits."""

import argparse
import csv
import math
import shutil
from pathlib import Path

import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.regime_program.reporting import partitions


def cost_attribution(folder, row, spot, perpetual):
    result = read(folder / "result.json")
    curve = pd.read_parquet(folder / "equity_5m.parquet")
    events = read(folder / "events.json")
    raw_spot_cash = raw_perp_realised = net_slip = 0.0
    shorts = {s: {"q": 0.0, "entry": 0.0} for s in spot}
    spotq = dict.fromkeys(spot, 0.0)
    for e in events:
        if e["type"] not in ["entry", "exit"]:
            continue
        s, q, market = e["symbol"], e["quantity"], e["market"]
        px = float(
            (spot if market == "spot" else perpetual)[s].loc[
                pd.Timestamp(e["time"]) - pd.Timedelta(minutes=5), "close"
            ]
        )
        net_slip += q * abs(e["price"] - px)
        if market == "spot":
            raw_spot_cash += q * px * (-1 if e["type"] == "entry" else 1)
            spotq[s] += q * (1 if e["type"] == "entry" else -1)
        elif e["type"] == "entry":
            p = shorts[s]
            p["entry"] = (p["entry"] * p["q"] + q * px) / (p["q"] + q)
            p["q"] += q
        else:
            p = shorts[s]
            raw_perp_realised += q * (p["entry"] - px)
            p["q"] -= q
    at = pd.Timestamp(row["end"], tz="UTC") - pd.Timedelta(minutes=5)
    mark_root = Path(read(folder.parents[1] / "plan.json")["mark_data"])
    floating = 0.0
    for s, p in shorts.items():
        raw_spot_cash += spotq[s] * float(spot[s].loc[at, "close"])
        mark = float(pd.read_parquet(mark_root / f"{s}_mark_5m.parquet").loc[at, "close"])
        floating += p["q"] * (p["entry"] - mark)
    price_pnl = raw_spot_cash + raw_perp_realised + floating
    fees = math.fsum(e.get("fee", 0) for e in events)
    funding = math.fsum(e["cash_delta"] for e in events if e["type"] == "funding")
    reserve = float(curve.exit_reserve.iloc[-1])
    expected = price_pnl + funding - fees - net_slip - reserve
    actual = float(curve.equity.iloc[-1] - row["capital"])
    if abs(expected - actual) > max(1e-7, row["capital"] * 1e-10):
        raise ValueError("Price/funding/all-cost attribution does not reconcile")
    return {
        "price_pnl_at_reference_net_base": price_pnl,
        "funding_net": funding,
        "actual_fees": fees,
        "net_base_execution_slippage": net_slip,
        "exit_cost_reserve": reserve,
        "net_pnl": actual,
        "reconciliation_error": abs(expected - actual),
        "gross_order_slippage_proxy_in_raw_metrics": result["metrics"]["slippage_cost"],
        "note": "Net-base slippage plus actual base-fee valuation avoids double counting slippage on buy-side fee coins. Raw gross-order slippage remains a distinct diagnostic; account NAV is unchanged.",
    }


def publish(run, docs):
    receipt = read(run / "completion.json")
    if (
        sha(run / "summary.json") != receipt["summary_sha256"]
        or sha(run / "plan.json") != receipt["plan_sha256"]
    ):
        raise ValueError("Run completion changed")
    raw, plan = read(run / "summary.json"), read(run / "plan.json")
    if not raw["audit"]["passed"] or not all(r["passed"] for r in raw["restart_checks"]):
        raise ValueError("Financial or continuation audit incomplete")
    for p, digest in plan["source_hashes"].items():
        if sha(Path(p)) != digest:
            raise ValueError("Frozen run source changed " + p)
    docs.mkdir(parents=True, exist_ok=False)
    spot = {
        s: pd.read_parquet(Path(plan["spot_data"]) / f"{s}_spot_5m.parquet")
        for s in plan["symbols"]
    }
    perpetual = {
        s: pd.read_parquet(Path(plan["perpetual_data"]) / f"{s}_5m.parquet")
        for s in plan["symbols"]
    }
    rows, details = [], {}
    for row in raw["account_rows"]:
        folder = run / "accounts" / row["id"]
        a = read(folder / "result.json")
        if sha(folder / "result.json") != row["result_sha256"]:
            raise ValueError("Account result changed")
        for name, digest in a["source_hashes"].items():
            if Path(name).name != name or sha(folder / name) != digest:
                raise ValueError("Account ledger changed")
        audit = read(folder / "independent_audit.json")
        if audit != {
            k: v
            for k, v in next(x for x in raw["audits"] if x["id"] == row["id"]).items()
            if k != "id"
        }:
            raise ValueError("Account independent audit changed")
        attribution = cost_attribution(folder, row, spot, perpetual)
        rows.append(
            {
                **row,
                "cost_attribution": attribution,
                "source_run": str(run),
                "source_account_id": row["id"],
            }
        )
        details[row["id"]] = {
            **read(folder / "periods.json"),
            "lifecycle_uncertainty": row["lifecycle_uncertainty"],
            "cost_attribution": attribution,
            "open_net_pnl_with_exit_reserve": audit["open_net_pnl_with_exit_reserve"],
            "wallets_at_end": a["wallets"],
            "risk_flags": a["flags"],
        }
    parent = Path("results/research/regime_accounts_20261026_v001")
    prior = read(parent / "summary.json")
    prior_receipt = read(parent / "completion.json")
    if (
        sha(parent / "summary.json") != prior_receipt["summary_sha256"]
        or not prior["audit"]["passed"]
    ):
        raise ValueError("Cash reference seal changed")
    states_root = Path("results/research/regime_states_20261025_v002")
    states = pd.read_parquet(states_root / "states_daily.parquet")
    event_flags = pd.read_parquet(
        states_root / "event_risk_hourly.parquet"
    ).covered_fomc_post_release
    for r in prior["account_rows"]:
        if (
            r["candidate"] != "cash"
            or r["capital"] != 10000
            or r["phase"] not in ["screen2025", "diagnostic2026", "continuous2025_2026"]
        ):
            continue
        folder = parent / "accounts" / r["id"]
        result = read(folder / "result.json")
        if any(sha(folder / n) != h for n, h in result["source_hashes"].items()):
            raise ValueError("Cash ledger changed")
        curve = pd.read_parquet(folder / "equity_5m.parquet")
        if not (curve.equity == 10000).all() or result["trades"]:
            raise ValueError("Reference is not flat non-interest cash")
        identity = "reference_" + r["id"]
        row = {
            **r,
            "id": identity,
            "flags": {},
            "reused_reference": True,
            "source_run": str(parent),
            "source_account_id": r["id"],
            "result_sha256": sha(folder / "result.json"),
            "cost_attribution": {"net_base_execution_slippage": 0, "exit_cost_reserve": 0},
        }
        rows.append(row)
        details[identity] = partitions(curve, [], states, event_flags, 10000)
    if len(rows) - len(raw["account_rows"]) != 6:
        raise ValueError("Require six matched cash reference rows")
    previous = Path("results/research/spot_carry_accounts_20261026_v001")
    reproduced = []
    for r in raw["reproduced_prior_attempt"]:
        old = previous / "accounts" / r["id"] / "independent_audit.json"
        reproduced.append(
            {
                **{k: v for k, v in r.items() if k != "previous_attempt_audit_passed"},
                "prior_case_audit_passed": old.exists() and read(old)["passed"],
                "prior_batch_completed": False,
            }
        )
    summary = {
        "schema": "segregated-spot-carry-publication.v1",
        "title": "现货＋永续 · 分账账户验证",
        "account_returns_computed": True,
        "actual_accounts": len(raw["account_rows"]),
        "reused_reference_accounts": 6,
        "account_rows": rows,
        "details": details,
        "candidate_names": {
            "fixed_hold": "固定对冲持有",
            "cost_guarded": "成本条件持有",
            "cash": "现金（封存对照）",
        },
        "module_assessments": raw["module_assessments"],
        "audit": raw["audit"],
        "restart_checks": raw["restart_checks"],
        "reproduced_prior_attempt": reproduced,
        "profit_goal_met": False,
        "portfolio_actual": None,
        "mechanism_count": 1,
        "capacity_selection": raw["capacity_selection"],
        "independent_oos": False,
        "production_eligible": False,
        "orders_enabled": False,
        "cost_note": "现货佣金10bps、永续5bps；每腿每边滑点3／10bps。初始总敞口约0.5，保留旧基线风险预算，不与0.8敞口策略直接作同风险增量宣称。",
        "boundary_note": "曲线记录新一轮区间开始动作之前的边界状态；初始现金调拨计入初始点，后续空仓调拨明记下一5m估值点生效。实际调拨时间和资金费结算毫秒另行保留。",
        "next_action": "固定持有有小幅正收益，但只有2个完整周期；条件持有未提高净收益，停止同组窗口/门槛搜索。10万出现终端部分退出，100万失败对冲明显增多；先核查容量退出机制和历史股票/永续主线中的独立经济来源，不以加杠杆补20%目标。",
    }
    save(docs / "summary.json", summary)
    shutil.copyfile(run / "plan.json", docs / "execution_plan.json")
    save(docs / "mark_data_summary.json", read(Path(plan["mark_data"]) / "summary.json"))
    save(
        docs / "provenance.json",
        {
            "run_completion_sha256": sha(run / "completion.json"),
            "publication_source_sha256": sha(Path(__file__)),
            "mark_completion_sha256": sha(Path(plan["mark_data"]) / "completion.json"),
            "raw_run": str(run),
            "prior_batch_raw_reproduction_boolean_note": "raw previous_attempt_audit_passed=false described incomplete batch; corrected per-case audit presence in publication without changing raw receipts",
        },
    )
    columns = [
        "phase",
        "candidate",
        "capital",
        "slippage_bps",
        "net_return",
        "cagr",
        "win_rate",
        "max_drawdown",
        "sharpe",
        "closed_trades",
        "matched_closed_pairs",
        "fees",
        "funding_net",
        "net_base_slippage",
        "open_pnl",
        "mean_gross_exposure",
        "hedge_failures",
        "risk_halted",
        "reused_reference",
    ]
    with (docs / "accounts.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for r in rows:
            m = r["metrics"]
            writer.writerow(
                {
                    **{
                        k: r[k]
                        for k in [
                            "phase",
                            "candidate",
                            "capital",
                            "slippage_bps",
                            "risk_halted",
                            "reused_reference",
                        ]
                    },
                    **{
                        k: m.get(k)
                        for k in [
                            "net_return",
                            "cagr",
                            "win_rate",
                            "max_drawdown",
                            "sharpe",
                            "closed_trades",
                            "matched_closed_pairs",
                            "fees",
                            "funding_net",
                            "mean_gross_exposure",
                        ]
                    },
                    "net_base_slippage": r["cost_attribution"]["net_base_execution_slippage"],
                    "open_pnl": m.get("open_pnl_with_exit_reserve", 0),
                    "hedge_failures": r["capacity"].get("hedge_failures", 0),
                }
            )
    print(
        {
            "publication_prepared": True,
            "new_accounts": summary["actual_accounts"],
            "references": 6,
            "maximum_attribution_error": max(
                r["cost_attribution"].get("reconciliation_error", 0) for r in rows
            ),
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--docs", type=Path, required=True)
    args = parser.parse_args()
    publish(args.run, args.docs)
