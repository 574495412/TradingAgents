"""Observed research lineage and illustrative power; no new strategy accounts.

Run directly with scipy available. Observed local artifacts cannot certify the
complete search history, independent trials, or historical human observations.
"""

import argparse
import csv
import hashlib
import json
import math
import platform
from collections import defaultdict
from datetime import UTC, datetime
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import scipy
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.stats import chi2, nct, norm

ROOT = Path(__file__).resolve().parents[3]
PLAN = ROOT / "docs/research/equities_depth_review_20261026/selection-power-plan.json"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def curve_key(curve, currency):
    initial = Fraction(Decimal(str(curve[0]["equity"])))
    points = [(r["clock"], str(Fraction(Decimal(str(r["equity"]))) / initial)) for r in curve]
    return hashlib.sha256(json.dumps([currency, points]).encode()).hexdigest()


def integrated_survival(cutoff, df, noncentrality):
    # T=(Z+lambda)/sqrt(V/df). Quantile truncation error is <=2e-12.
    lo, hi = chi2.ppf([1e-12, 1 - 1e-12], df) / df
    return quad(
        lambda u: (
            ndtr(noncentrality - cutoff * math.sqrt(u)) * math.exp(chi2.logpdf(df * u, df)) * df
        ),
        lo,
        hi,
        epsabs=1e-11,
        epsrel=1e-11,
    )[0]


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    plan = read(PLAN)
    inputs = {str(PLAN.relative_to(ROOT)): sha(PLAN)}

    def bind(path):
        inputs[str(path.relative_to(ROOT))] = sha(path)
        return read(path)

    with (ROOT / plan["stock_observation_calendar"]).open(newline="") as handle:
        stock_dates = {r["date"] for r in csv.DictReader(handle)}
    inputs[plan["stock_observation_calendar"]] = sha(ROOT / plan["stock_observation_calendar"])
    rows, studies, same_paths = [], [], defaultdict(list)
    earlier_US_dates = set()
    for name in plan["completed_roots"]:
        directory = ROOT / "results/research" / name
        summary = bind(directory / "summary.json")
        protocol = bind(directory / "protocol.json")
        registration_path = directory / "registration.json"
        registration = bind(registration_path) if registration_path.exists() else {}
        selection_path = directory / "selection.json"
        selection = bind(selection_path) if selection_path.exists() else None
        for record in summary["records"]:
            path = directory / record["name"] / "result.json"
            result = bind(path)
            curve = result["account"]["equity_curve"]
            dates = {
                datetime.fromtimestamp(r["clock"] / 1000, UTC).date().isoformat() for r in curve[1:]
            }
            identity = f"{name}/{record['name']}"
            key = curve_key(curve, result["currency"])
            same_paths[key].append(identity)
            if name != plan["completed_roots"][-1] and record["market"] == "US":
                earlier_US_dates |= dates
            rows.append(
                {
                    "study": name,
                    "name": record["name"],
                    "market": record["market"],
                    "policy": record.get("policy", record.get("candidate")),
                    "phase": record.get("phase", "continuous"),
                    "cost_bps": record.get("slippage_bps", record.get("cost_bps")),
                    "currency": result["currency"],
                    "observations": len(curve) - 1,
                    "observed_date_range": [min(dates), max(dates)],
                    "exact_normalized_curve_sha256": key,
                    "forward_OOS_claim": result["independent_forward_oos"],
                    "result_sha256": sha(path),
                }
            )
        studies.append(
            {
                "study": name,
                "account_records": len(summary["records"]),
                "registered_at_or_date": registration.get(
                    "registered_at_utc", protocol.get("frozen_date")
                ),
                "clock_is_external_preregistration_proof": False,
                "declared_history_role": protocol.get("history_role"),
                "declared_forward_OOS": protocol.get("independent_forward_oos"),
                "selection_summary": selection.get("new_selection")
                if isinstance(selection, dict) and "new_selection" in selection
                else summary.get("selection"),
            }
        )
    attempts = []
    for name in plan["incomplete_attempts"]:
        directory = ROOT / "results/research" / name
        registration = bind(directory / "registration.json")
        protocol = bind(directory / "protocol.json")
        archived_fees = directory / "source/tradingagents/research/equities/fees.py"
        old_fees = archived_fees.read_text()
        inputs[str(archived_fees.relative_to(ROOT))] = sha(archived_fees)
        assert (
            sha(archived_fees)
            == registration["source_hashes"]["tradingagents/research/equities/fees.py"]
        )
        attempts.append(
            {
                "study": name,
                "registered_at": registration["registered_at_utc"],
                "account_results_present": len(list(directory.glob("*/result.json"))),
                "states_present": len(list(directory.glob("*/state.json"))),
                "summary_present": (directory / "summary.json").exists(),
                "archived_fee_scope_rejects_non_ETF": 's.get("instrument") != "ETF"' in old_fees,
                "terminal_error_log_available": False,
                "scope": "Incomplete local attempt retained; fee-scope incompatibility is visible, exact terminal exception is not certified from the artifact alone.",
            }
        )
    duplicates = [names for names in same_paths.values() if len(names) > 1]
    lineage = {
        "schema": "equity-observed-selection-lineage.v1",
        "studies": studies,
        "observed_completed_account_records": len(rows),
        "reused_prior_program_account_records": sum(s["account_records"] for s in studies[:2]),
        "current_program_executed_accounts": sum(s["account_records"] for s in studies[2:]),
        "cash_policy_records": sum(r["policy"] == "cash" for r in rows),
        "exact_equal_curve_groups": duplicates,
        "exact_distinct_curve_count": len(same_paths),
        "distinct_curves_are_independent_trials": False,
        "incomplete_attempts": attempts,
        "stock_daily_return_dates": len(stock_dates),
        "stock_dates_also_in_earlier_US_market_results": len(stock_dates & earlier_US_dates),
        "market_date_overlap_proves_prior_stock_specific_viewing": False,
        "all_prior_researcher_choices_and_hypotheses_enumerated": False,
        "independent_trial_count": None,
        "DSR": None,
        "PBO": None,
        "reason": "These are observed artifacts with controls, costs, phases and capital variants. Missing influential choices, security-specific viewing chronology, and test dependence preclude a certified effective trial count or selection-adjusted statistic.",
        "new_economic_accounts": 0,
        "external_attestation": False,
    }
    cfg = plan["power_model"]
    F, alpha, desired = cfg["annualization"], cfg["one_sided_alpha"], cfg["desired_power"]
    scenarios = []
    for null, alternative in cfg["null_alternative_annual_sharpe_pairs"]:

        def power(n, s0=null, s1=alternative):
            critical = nct.ppf(1 - alpha, n - 1, s0 * math.sqrt(n / F))
            return float(nct.sf(critical, n - 1, s1 * math.sqrt(n / F)))

        lo, hi = 2, 100 * F
        assert power(hi) >= desired
        while lo < hi:
            mid = (lo + hi) // 2
            if power(mid) >= desired:
                hi = mid
            else:
                lo = mid + 1
        n = lo
        assert power(n - 1) < desired <= power(n)
        critical = float(nct.ppf(1 - alpha, n - 1, null * math.sqrt(n / F)))
        integrals = {
            "null_tail": integrated_survival(critical, n - 1, null * math.sqrt(n / F)),
            "alternative_tail": integrated_survival(
                critical, n - 1, alternative * math.sqrt(n / F)
            ),
        }
        assert abs(integrals["null_tail"] - alpha) < 2e-9
        assert abs(integrals["alternative_tail"] - power(n)) < 2e-9
        scenarios.append(
            {
                "null_annual_sharpe": null,
                "alternative_annual_sharpe": alternative,
                "minimum_daily_observations": n,
                "illustrative_years_252": n / F,
                "power_at_252_days": power(F),
                "power_at_n_minus_1": power(n - 1),
                "power_at_n": power(n),
                "large_n_z_approx_years": float(
                    (norm.ppf(1 - alpha) + norm.ppf(desired)) ** 2 / (alternative - null) ** 2
                ),
                "quadrature_crosscheck": integrals,
            }
        )
    write(output / "observed-records.json", rows)
    write(output / "lineage.json", lineage)
    write(
        output / "power-scenarios.json",
        {
            "assumptions": cfg,
            "scenarios": scenarios,
            "actual_admission_power_or_required_horizon": None,
            "new_economic_accounts": 0,
            "external_attestation": False,
        },
    )
    source_doc = ROOT / "data/research/equity_selection_power_20261027_v001/scipy-nct.htm"
    inputs[str(source_doc.relative_to(ROOT))] = sha(source_doc)
    write(
        output / "registration.json",
        {
            "inputs_sha256": inputs,
            "analysis_sha256": sha(Path(__file__)),
            "plan_sha256": sha(PLAN),
            "python": platform.python_version(),
            "scipy": scipy.__version__,
            "review_is_retrospective": True,
            "is_external_attestation": False,
        },
    )
    (output / "analysis_source.py").write_bytes(Path(__file__).read_bytes())
    (output / "plan.json").write_bytes(PLAN.read_bytes())
    print(
        json.dumps(
            {
                "records": len(rows),
                "distinct_curves": len(same_paths),
                "cash_records": lineage["cash_policy_records"],
                "stock_date_overlap": lineage["stock_dates_also_in_earlier_US_market_results"],
                "power_scenario_years": [r["illustrative_years_252"] for r in scenarios],
                "DSR": None,
                "new_accounts": 0,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
