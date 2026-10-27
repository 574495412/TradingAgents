"""Separate calendar/rank and scalar BTC-state checks; internal, not third party."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest
from research_tools.sparse_market.audit_final_cohort import prepare_raw
from research_tools.sparse_market.identity import crypto_context


def audit(inputs, output):
    if output.exists():
        raise FileExistsError("Preserve executed audit")
    verify_manifest(inputs)
    plan = read(inputs / "plan.json")
    for filename, digest in plan["source_hashes"].items():
        assert sha(Path(filename)) == digest, filename
    p = plan["state_parameters"]
    cohort = Path(plan["cohort"])
    registry = read(cohort / "registry.json")
    versions = prepare_raw(read(cohort / "plan.json"), cohort)
    errors = dict.fromkeys(
        ["trend_z", "efficiency", "volatility_ratio", "breadth", "eligible_count"], 0.0
    )
    ranks = labels = 0
    for variant, raw in versions.items():
        vectors = {}
        for s, f in raw.items():
            dates = pd.date_range(f.index[0], f.index[-1] + pd.Timedelta(days=1), freq="D")
            q = f.reindex(dates)
            valid = (
                q.price_liquidity_feature_usable.eq(True) & q.volume.gt(0) & q.quote_volume.gt(0)
            ).to_numpy(copy=True)
            valid[0] = False
            close = q.close.to_numpy(dtype=float).copy()
            close[~valid] = np.nan
            vectors[s] = (
                dates,
                close,
                q.quote_volume.to_numpy(dtype=float),
                np.r_[0, valid.cumsum()],
            )
        bd, btc, _, _ = vectors["BTCUSDT"]
        ret = np.r_[np.nan, btc[1:] / btc[:-1] - 1]
        vols = np.full(len(ret), np.nan)
        for k in range(p["volatility_window"] - 1, len(ret)):
            r = ret[k - p["volatility_window"] + 1 : k + 1]
            if np.isfinite(r).all():
                vols[k] = np.std(r, ddof=1)
        states = pd.read_parquet(inputs / (variant + "_states.parquet"))
        members = {r["decision"]: r for r in read(inputs / (variant + "_members.json"))}
        for clock, actual in states.iterrows():
            end = bd.searchsorted(clock)
            market = btc[end - 61 : end]
            mret = market[1:] / market[:-1] - 1
            candidates = []
            for s, (dates, close, quote, cumulative) in vectors.items():
                if clock < dates[0] or clock > dates[-1]:
                    continue
                c = crypto_context(registry, s, clock)
                if c["blocked"]:
                    continue
                at = dates.searchsorted(clock)
                start = (
                    dates.searchsorted(pd.Timestamp(c["feature_start"]))
                    if c["feature_start"]
                    else 0
                )
                if cumulative[at] - cumulative[start] < 180 or at - start < 61:
                    continue
                history = close[at - 61 : at]
                if (
                    len(history) != 61
                    or not np.isfinite(history).all()
                    or not np.isfinite(market).all()
                ):
                    continue
                variance = np.var(mret, ddof=1)
                if not variance > 1e-12:
                    continue
                q = float(np.mean(quote[at - 28 : at]))
                if q >= p["minimum_daily_quote"]:
                    candidates.append((s, q))
            selected = [
                s
                for s, _ in sorted(candidates, key=lambda r: (-r[1], r[0]))[: p["maximum_eligible"]]
            ]
            stored = members[clock.isoformat()]
            assert stored["selected"] == selected and stored["eligible_before_rank"] == len(
                candidates
            )
            ranks += 1
            d = end - 1
            momentum = btc[d] / btc[d - p["momentum_window"]] - 1
            vol = vols[d]
            prior = vols[max(0, d - p["volatility_history"]) : d]
            finite = prior[np.isfinite(prior)]
            high = (
                np.quantile(finite, p["high_volatility_quantile"])
                if len(finite) >= p["volatility_history_min"]
                else np.nan
            )
            path = btc[d - p["efficiency_window"] : d + 1]
            travelled = np.abs(np.diff(path)).sum()
            efficiency = abs(path[-1] - path[0]) / travelled if travelled else np.nan
            slow = np.mean(btc[d - p["slow_ma_window"] + 1 : d + 1])
            positive = []
            for s in selected:
                dates, close, _, _ = vectors[s]
                at = dates.searchsorted(clock)
                positive.append(close[at - 1] / close[at - 1 - p["momentum_window"]] - 1 > 0)
            calc = {
                "trend_z": momentum / (vol * np.sqrt(p["momentum_window"])),
                "efficiency": efficiency,
                "volatility_ratio": vol / high,
                "breadth": float(np.mean(positive)) if positive else np.nan,
                "eligible_count": float(len(positive)),
            }
            state = "range"
            strong = abs(calc["trend_z"]) >= p["trend_z_min"] and efficiency >= p["efficiency_min"]
            if strong and momentum > 0 and btc[d] > slow:
                state = "trend_up"
            if strong and momentum < 0 and btc[d] < slow:
                state = "trend_down"
            if calc["volatility_ratio"] > 1:
                state = "volatility_breakout"
            if len(positive) < p["minimum_eligible"]:
                state = "liquidity_stress"
            if not np.isfinite([*calc.values(), slow]).all():
                state = "unknown"
            for k, value in calc.items():
                if np.isnan(value):
                    assert np.isnan(actual[k])
                else:
                    errors[k] = max(errors[k], abs(value - actual[k]))
            assert state == actual.state, (variant, clock, state, actual.state, calc)
            labels += 1
        original = pd.read_parquet(
            Path(plan["targets"]) / (variant + "_monthly_tsmom_targets.parquet")
        )
        conditional = pd.read_parquet(inputs / f"{variant}_conditional_trend_targets.parquet")
        long = pd.read_parquet(inputs / f"{variant}_conditional_long_targets.parquet")
        monthlies = pd.date_range(
            plan["window"]["start"], plan["window"]["end"], freq="MS", tz="UTC", inclusive="left"
        )
        active = states.state.reindex(monthlies).isin(plan["gate"]["active_states"])
        active_hours = active.reindex(original.index, method="ffill")
        expected = original.copy()
        columns = list(original.filter(like="target_").columns)
        expected.loc[~active_hours, columns] = 0.0
        pd.testing.assert_frame_equal(expected, conditional, check_exact=True)
        expected.loc[:, columns] = expected[columns].abs()
        pd.testing.assert_frame_equal(expected, long, check_exact=True)
    assert max(errors.values()) < 1e-9 and labels == ranks == 1216
    result = {
        "schema": "monthly-state-domain-internal-audit.v1",
        "passed": True,
        "inputs_completion_sha256": sha(inputs / "completion.json"),
        "source_sha256": sha(Path(__file__)),
        "independent_daily_PIT_rank_cases": ranks,
        "scalar_BTC_state_cases": labels,
        "maximum_feature_errors": errors,
        "exact_monthly_gate_and_magnitude_controls": True,
        "source_equivalence": read(inputs / "summary.json")["source_target_equivalence"],
        "shared_dependencies": "Raw parsing/reconciliation and source-bound identity policy are shared; membership calendar/rank and BTC numeric state formulas are separately recomputed.",
        "new_state_parameters": 0,
        "new_financial_accounts": 0,
        "third_party_review": False,
        "independent_oos": False,
        "profit_goal_met": False,
    }
    save(output, result)
    print(result, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    audit(a.inputs, a.output)
