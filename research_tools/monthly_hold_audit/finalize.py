"""Finalize preserved accounts after auditing exposure summation at binary64 precision.

No financial account is rerun. Economic NAV/cash/events/state must remain exact;
both exposure series must independently meet a derived floating-sum error bound.
"""

import argparse
import copy
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.factor_lab.study import executor
from research_tools.max_effect.run_accounts import verify_manifest
from research_tools.trend_pool_audit.run import load
from research_tools.trend_stop_audit.run import paired_uncertainty


def audit_exposures(full, resumed, checkpoint, events, bars):
    assert full.index.equals(resumed.index)
    positions = {s: {"q": p["q"], "side": p["side"]} for s, p in checkpoint["positions"].items()}
    prices = {
        s: f.close.reindex(full.index - pd.Timedelta(minutes=5)).to_numpy(dtype=float)
        for s, f in bars.items()
    }
    changes = [e for e in events if e["type"] != "funding"]
    pointer = 0
    maximum_pair = {"gross_exposure": 0.0, "net_exposure": 0.0}
    maximum_canonical = dict.fromkeys(maximum_pair, 0.0)
    maximum_bound_ratio = 0.0
    u = np.finfo(float).eps / 2
    for i, clock in enumerate(full.index):
        while pointer < len(changes) and pd.Timestamp(changes[pointer]["time"]) <= clock:
            e = changes[pointer]
            s = e["symbol"]
            if e["type"] == "entry":
                if s not in positions:
                    positions[s] = {"q": 0.0, "side": e["side"]}
                assert positions[s]["side"] == e["side"]
                positions[s]["q"] += e["quantity"]
            else:
                positions[s]["q"] -= e["quantity"]
                if positions[s]["q"] < 1e-12:
                    del positions[s]
            pointer += 1
        terms = []
        for s, p in positions.items():
            price = prices[s][i]
            assert math.isfinite(price) and price > 0
            terms.append(p["side"] * p["q"] * price)
        gross = math.fsum(abs(v) for v in terms)
        net = math.fsum(terms)
        # Standard summation bound, plus correctly rounded fsum reference error.
        # Gross sum is the sum of absolute terms, so this also bounds cancellation
        # in net exposure; no observed profit or return selects the tolerance.
        n = len(terms)
        gamma = (n + 1) * u / (1 - (n + 1) * u)
        bound = gamma * gross
        for field, canonical in [("gross_exposure", gross), ("net_exposure", net)]:
            a, b = float(full[field].iloc[i]), float(resumed[field].iloc[i])
            err = max(abs(a - canonical), abs(b - canonical))
            assert err <= bound if bound else err == 0, (clock, field, err, bound)
            maximum_pair[field] = max(maximum_pair[field], abs(a - b))
            maximum_canonical[field] = max(maximum_canonical[field], err)
            if bound:
                maximum_bound_ratio = max(maximum_bound_ratio, err / bound)
    assert pointer == len(changes)
    return {
        "passed": True,
        "rows": len(full),
        "maximum_pair_difference_usdt": maximum_pair,
        "maximum_reference_difference_usdt": maximum_canonical,
        "maximum_fraction_of_derived_sum_bound": maximum_bound_ratio,
        "method": "Rebuild inventory from checkpoint and exact child fills; price every held quantity from the retained preceding actual5m close; use math.fsum for canonical gross and net. Require each full/resumed exposure within gamma_(N+1)*gross with u=2^-53, including cancellation in net; no arbitrary relaxed NAV tolerance.",
        "scope": "Exposure arithmetic only. NAV,cash,events,requests,positions and final risk state are required exactly equal.",
    }


def finalize(run, output):
    output.mkdir(parents=True, exist_ok=False)
    plan = read(run / "plan.json")
    assert read(run / "failure.json")["stage"] == "exact_exposure_restore_comparison"
    verify_manifest(run)
    for name, digest in plan["source_hashes"].items():
        assert sha(Path(name)) == digest, name
    target = Path(plan["targets"])
    reference = Path(plan["reference_run"])
    _, bars, rates, lifecycle = load(target)
    sys.path.insert(0, str(Path("../freqtrade").resolve()))
    engine_path = Path("../freqtrade/research/crypto/monthly_hold_v1.py").resolve()
    engine = executor(str(engine_path))
    bridges = []
    for cost in plan["costs_slippage_bps"]:
        identity = f"bridge_enabled_monthly_tsmom_cap10000_slip{cost}"
        original = f"daily_archive_monthly_tsmom_cap10000_slip{cost}"
        a, b = run / "accounts" / identity, reference / "accounts" / original
        pd.testing.assert_frame_equal(
            pd.read_parquet(a / "equity_5m.parquet"),
            pd.read_parquet(b / "equity_5m.parquet"),
            check_exact=True,
        )
        for field in ["events", "requests", "trades"]:
            assert read(a / (field + ".json")) == read(b / (field + ".json"))
        state = read(a / "resume_state.json")
        prior = read(b / "resume_state.json")
        normalized = copy.deepcopy(state)
        normalized["engine_version"] = prior["engine_version"]
        assert normalized == prior
        assert read(a / "result.json")["metrics"] == read(b / "result.json")["metrics"]
        bridges.append(
            {
                "id": identity,
                "original": original,
                "exact_full_curve_ledger_requests_inventory_and_risk": True,
                "only_engine_identity_state_differs": True,
            }
        )
    identity = "disabled_monthly_tsmom_cap10000_slip10"
    folder = run / "accounts" / identity
    reg = read(run / "restart_registration.json")
    checkpoint = read(run / "restart_checkpoint.json")
    child = read(run / "restart_worker.json")
    full = pd.read_parquet(folder / "equity_5m.parquet").set_index("date")
    suffix = full.loc[pd.Timestamp(reg["split"]) :]
    resumed = pd.read_parquet(run / "restart_worker_curve.parquet").set_index("date")
    pd.testing.assert_frame_equal(
        suffix[["equity", "cash"]], resumed[["equity", "cash"]], check_exact=True
    )
    assert child["resume_state"] == read(folder / "resume_state.json")
    assert child["positions"] == child["resume_state"]["positions"]
    assert child["pid"] != reg["parent_pid"]
    events = read(folder / "events.json")
    assert [e for e in events if pd.Timestamp(e["time"]) > pd.Timestamp(reg["split"])] == child[
        "events"
    ]
    requests = read(folder / "requests.json")
    assert requests[-len(child["requests"]) :] == child["requests"]
    assert checkpoint["cash"] == full.loc[pd.Timestamp(reg["split"]), "cash"]
    exposure = audit_exposures(suffix, resumed, checkpoint, child["events"], bars)
    rejection = []
    scores = pd.read_parquet(target / reg["target_file"])
    for config, state, expected in [
        ({**reg["config"], "price_stop_enabled": True}, checkpoint, "configuration"),
        (reg["config"], {**checkpoint, "engine_version": "calendar-sparse.v2"}, "version"),
    ]:
        try:
            engine.replay(
                bars,
                rates,
                scores,
                config,
                reg["split"],
                plan["window"]["end"],
                initial_state=state,
                lifecycle=lifecycle,
            )
        except ValueError as error:
            assert expected in str(error)
            rejection.append(str(error))
        else:
            raise AssertionError("Mismatched policy resume accepted")
    rows = []
    for profile in plan["profiles"]:
        for cost in plan["costs_slippage_bps"]:
            row = read(run / "accounts" / f"disabled_{profile}_cap10000_slip{cost}" / "row.json")
            assert row["financial_audit_passed"] and row["price_stop_disabled_verified"]
            rows.append(row)
    comparisons = []
    for cost in plan["costs_slippage_bps"]:
        a = next(r for r in rows if r["profile"] == "monthly_tsmom" and r["slippage_bps"] == cost)
        b = next(r for r in rows if r["profile"] == "magnitude_long" and r["slippage_bps"] == cost)
        aa, bb = [pd.read_parquet(run / "accounts" / r["id"] / "equity_5m.parquet") for r in [a, b]]
        original = reference / "accounts" / f"daily_archive_monthly_tsmom_cap10000_slip{cost}"
        comparisons.append(
            {
                "slippage_bps": cost,
                "disabled_trend_minus_long_net_pnl": a["metrics"]["net_pnl"]
                - b["metrics"]["net_pnl"],
                "disabled_minus_enabled_trend_net_pnl": a["metrics"]["net_pnl"]
                - read(original / "result.json")["metrics"]["net_pnl"],
                "direction_paired_uncertainty": paired_uncertainty(bb, aa),
                "policy_paired_uncertainty": paired_uncertainty(
                    pd.read_parquet(original / "equity_5m.parquet"), aa
                ),
            }
        )
    assert all(r["metrics"]["net_pnl"] < 0 for r in rows)
    originals = list(rows)
    for row in originals:
        rows.append(
            {
                **row,
                "id": "reconciled_" + row["id"],
                "variant": "daily_with_complete_5m_reconciliation",
                "reused_account": True,
                "source_account_id": row["id"],
            }
        )
    case = read(run / "selected_case.json")
    assert case["selected_known_case_not_strategy_account"] and not case["residual_positions"]
    save(output / "plan.json", plan)
    save(output / "exposure_summation_audit.json", exposure)
    restart = {
        "passed": True,
        "actual_new_process": True,
        "split": reg["split"],
        "parent_pid": reg["parent_pid"],
        "worker_pid": child["pid"],
        "exact_full_NAV_cash_ledger_requests_inventory_and_risk": True,
        "exposure_bit_exact": False,
        "exposure_independently_within_binary64_sum_bound": True,
        "wrong_flag_and_engine_state_rejected": rejection,
        "new_financial_replays": 0,
    }
    save(output / "restart_verification.json", restart)
    save(
        output / "summary.json",
        {
            "schema": "monthly-hold-finalized.v1",
            "account_rows": rows,
            "bridge_replays": bridges,
            "actual_new_policy_accounts": 4,
            "selected_case_paths": 1,
            "source_alias_cases": 4,
            "actual_financial_computations": 7,
            "comparisons": comparisons,
            "capacity_triggered": False,
            "actual_capacity_accounts": 0,
            "actual_process_restart": restart,
            "financial_failures": [],
            "recovered_checker_failure": "Original exact fullframe restore assertion rejected exposure-only floating summation; preserved financial files not rerun. Exact economic state and independent binary64 exposure bounds now verified.",
            "profit_goal_met": False,
            "external_review": False,
            "independent_unseen_evidence": False,
            "orders_enabled": False,
            "production_eligible": False,
        },
    )
    save(
        output / "verification.json",
        {
            "passed": True,
            "source_sha256": sha(Path(__file__)),
            "preserved_run_completion_sha256": sha(run / "completion.json"),
            "engine_sha256": sha(engine_path),
            "no_financial_replay": True,
            "original_cash_and_stops_audits_passed": True,
        },
    )
    save(
        output / "completion.json",
        {
            "passed": True,
            "manifest": {q.name: sha(q) for q in output.iterdir() if q.is_file()},
            "profit_goal_met": False,
        },
    )
    print(
        {
            "passed": True,
            "new_policy_accounts": 4,
            "capacity_triggered": False,
            "restart": restart,
            "exposure": exposure,
        },
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    finalize(a.run, a.output)
