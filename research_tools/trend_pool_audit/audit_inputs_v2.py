"""Independent NumPy EWM/variance, membership and risk-weight reconstruction."""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest
from research_tools.sparse_market.audit_final_cohort import prepare_raw
from research_tools.sparse_market.identity import crypto_context

START = pd.Timestamp("2023-08-01T00:00:00Z")


def scalar_features(frame, episode_start=None):
    calendar = pd.date_range(START, frame.index[-1] + pd.Timedelta(days=1), freq="D")
    q = frame.reindex(calendar)
    valid = (
        q.price_liquidity_feature_usable.eq(True) & q.volume.gt(0) & q.quote_volume.gt(0)
    ).to_numpy(copy=True)
    if frame.index[0] in calendar:
        valid[calendar.get_loc(frame.index[0])] = False
    if episode_start:
        valid[calendar < pd.Timestamp(episode_start)] = False
    close = q.close.to_numpy(dtype=float)
    out = np.full((len(calendar), 2), np.nan)
    begin = 0
    while begin < len(calendar):
        if not valid[begin]:
            begin += 1
            continue
        end = begin
        while end < len(calendar) and valid[end]:
            end += 1
        prices = close[begin:end]
        fast_num = fast_den = slow_num = slow_den = 0.0
        w = w2 = sx = sx2 = 0.0
        deviations = np.full(len(prices), np.nan)
        for i, price in enumerate(prices):
            fast_num = price + (31 / 33) * fast_num
            fast_den = 1 + (31 / 33) * fast_den
            slow_num = price + (127 / 129) * slow_num
            slow_den = 1 + (127 / 129) * slow_den
            if i:
                change = price - prices[i - 1]
                decay = 34 / 36
                w, w2 = 1 + decay * w, 1 + decay * decay * w2
                sx, sx2 = change + decay * sx, change * change + decay * sx2
                if i >= 10:
                    variance = max(0.0, (sx2 - sx * sx / w) / (w - w2 / w))
                    deviations[i] = max(math.sqrt(variance), 1e-10)
            if i < 179:
                continue
            history = deviations[max(0, i - 499) : i + 1]
            floor = float(np.quantile(history[np.isfinite(history)], 0.05))
            vol = max(deviations[i], floor)
            decision = begin + i + 1
            if decision < len(calendar):
                out[decision] = [(fast_num / fast_den - slow_num / slow_den) / vol, vol / price]
        begin = end
    return pd.DataFrame(out, index=calendar, columns=["raw", "relative_vol"])


def old_members(raw, symbols, decision):
    eligible = []
    for s in symbols:
        if s == "BTCUSDT":
            continue
        frame = raw[s]
        cal = pd.date_range(frame.index[0], frame.index[-1] + pd.Timedelta(days=1), freq="D")
        f = frame.reindex(cal)
        valid = (
            f.price_liquidity_feature_usable.eq(True) & f.volume.gt(0) & f.quote_volume.gt(0)
        ).to_numpy(copy=True)
        valid[0] = False
        end = cal.searchsorted(decision)
        if valid[:end].sum() < 180 or end < 61 or not valid[end - 61 : end].all():
            continue
        market = raw["BTCUSDT"]
        mi = market.index.searchsorted(decision)
        mclose = market.close.iloc[mi - 61 : mi].to_numpy(dtype=float)
        mv = (
            market.price_liquidity_feature_usable.eq(True)
            & market.volume.gt(0)
            & market.quote_volume.gt(0)
        ).iloc[mi - 61 : mi]
        if (
            len(mclose) != 61
            or not mv.all()
            or np.var(mclose[1:] / mclose[:-1] - 1, ddof=1) <= 1e-12
        ):
            continue
        quote = float(np.mean(f.quote_volume.iloc[end - 28 : end]))
        if quote >= 50_000_000:
            eligible.append((s, quote))
    return [s for s, _ in sorted(eligible, key=lambda r: (-r[1], r[0]))[:30]]


def weights(rows, scale):
    good = [
        s
        for s, row in rows.items()
        if abs(np.clip(row.raw / scale, -3, 3)) > 0.32 and np.isfinite(row.relative_vol)
    ]
    selected = sorted(good, key=lambda s: (-abs(np.clip(rows[s].raw / scale, -3, 3)), s))[:10]
    if not selected:
        return {}
    desired = {
        s: 0.16
        / math.sqrt(365)
        / max(rows[s].relative_vol, 0.001)
        / len(selected)
        * np.clip(rows[s].raw / scale * 10, -20, 20)
        / 10
        for s in selected
    }
    gross = math.fsum(abs(v) for v in desired.values())
    shrink = min(1.0, 0.8 / gross) if gross else 1.0
    return {s: float(v * shrink / 0.8) for s, v in desired.items() if v}


def audit(inputs, output):
    verify_manifest(inputs)
    if output.exists():
        raise FileExistsError("Preserve input review")
    plan = read(inputs / "plan.json")
    for path, digest in plan["source_hashes"].items():
        assert sha(Path(path)) == digest, path
    for r in read(inputs / "execution_bindings.json").values():
        assert sha(Path(r["bar_path"])) == r["bar_sha256"]
        assert sha(Path(r["funding_path"])) == r["funding_sha256"]
    cohort = Path(plan["inputs"]["cohort"])
    inherited_path = Path(str(cohort) + "_independent_audit.json")
    inherited = read(inherited_path)
    assert inherited["passed"] and inherited["cohort_completion_sha256"] == sha(
        cohort / "completion.json"
    )
    assert inherited["source_sha256"] == sha(
        Path("research_tools/sparse_market/audit_final_cohort.py")
    )
    registry = read(cohort / "registry.json")
    assert not [
        r
        for key in ["cards", "launch_cards", "type_cards"]
        for r in registry[key]
        if r["symbol"] in plan["old_symbols"]
    ]
    versions = prepare_raw(read(cohort / "plan.json"), cohort)
    anchors = read(inputs / "anchors.json")
    error = 0.0
    groups = {}
    membership_cases = 0
    qualified_cases = 0
    for variant, raw in versions.items():
        refs = {
            pd.Timestamp(r["decision"]): r["selected"]
            for r in read(cohort / (variant + "_cohort.json"))
        }
        own = {t: old_members(raw, plan["old_symbols"], t) for t in refs}
        cache = {}
        for r in [x for x in anchors if x["variant"] == variant]:
            day, member_at = pd.Timestamp(r["decision"]), pd.Timestamp(r["membership_at"])
            expected = own[member_at] if r["domain"] == "fixed40" else refs[member_at]
            assert r["members"] == expected
            membership_cases += 1
            rows = {}
            for s in expected:
                context = crypto_context(registry, s, day)
                if context["blocked"]:
                    continue
                key = (s, context["feature_start"])
                if key not in cache:
                    cache[key] = scalar_features(raw[s], context["feature_start"])
                f = cache[key]
                if day in f.index and np.isfinite(f.loc[day].to_numpy(dtype=float)).all():
                    rows[s] = f.loc[day]
            assert set(rows) == set(r["qualified_forecasts"]), (variant, r["domain"], day)
            expected_book = (
                weights(rows, plan["forecast_normalization"]) if len(expected) >= 10 else {}
            )
            actual = r["book"]
            assert set(expected_book) == set(actual), (day, expected_book, actual)
            maximum = max([abs(expected_book[s] - actual[s]) for s in actual] or [0.0])
            error = max(error, maximum)
            if maximum > 1e-8:
                raise ValueError("Independent EWM/variance/risk book differs")
            groups.setdefault((variant, r["domain"]), {})[day] = expected_book
            qualified_cases += len(rows)
    cells = 0
    for (variant, domain), books in groups.items():
        actual = pd.read_parquet(inputs / f"{variant}_{domain}_targets.parquet")
        expected = (
            pd.DataFrame.from_dict(books, orient="index")
            .reindex(columns=plan["symbols"])
            .fillna(0.0)
        )
        expected = expected.reindex(actual.index, method="ffill")
        array = actual[["target_" + s for s in plan["symbols"]]].to_numpy(copy=True)
        assert np.max(abs(array - expected.to_numpy(copy=True))) <= 1e-8
        assert actual.risk_scale.eq(1.0).all() and actual.rebalance.eq(actual.index.hour == 0).all()
        assert actual.index.equals(
            pd.date_range(
                plan["window"]["start"], plan["window"]["end"], freq="h", inclusive="left", tz="UTC"
            )
        )
        cells += len(actual)
    equality = {}
    for domain in ["fixed40", "pit"]:
        a = pd.read_parquet(inputs / f"daily_archive_{domain}_targets.parquet")
        b = pd.read_parquet(
            inputs / f"daily_with_complete_5m_reconciliation_{domain}_targets.parquet"
        )
        equality[domain] = a.equals(b)
    result = {
        "schema": "trend-pool-independent-input-review.v1",
        "passed": True,
        "inputs_completion_sha256": sha(inputs / "completion.json"),
        "source_sha256": sha(Path(__file__)),
        "inherited_full_cohort_audit_sha256": sha(inherited_path),
        "all89_frozen_membership_clocks_per_source_bound": True,
        "calendar_domain_membership_cases": membership_cases,
        "qualified_daily_forecast_cases": qualified_cases,
        "hourly_source_domain_rows": cells,
        "maximum_numpy_forecast_risk_weight_error": error,
        "source_paired_targets_exactly_equal": equality,
        "financial_accounts_computed": 0,
        "profit_goal_met": False,
        "limits": "Internal mathematical review; same rawdata and provisional announcement availability. No independent source truth, thirdparty review, OOS or search correction.",
    }
    save(output, result)
    print(result, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    audit(a.inputs, a.output)
