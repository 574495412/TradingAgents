"""Reconstruct publisher momentum components, without claiming stock replication.

This standalone module reads frozen public files and uses Decimal arithmetic.
It does not import the application, open accounts or replay financial paths.
"""

import argparse
import csv
import hashlib
import json
import re
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data/research/equity_french_calibration_20261027_v001"
DOC = ROOT / "docs/research/equities_french_calibration_20261027"
LABELS = [
    "SMALL LoPRIOR",
    "ME1 PRIOR2",
    "SMALL HiPRIOR",
    "BIG LoPRIOR",
    "ME2 PRIOR2",
    "BIG HiPRIOR",
]
FILES = {
    "factor": "F-F_Momentum_Factor_CSV.csv",
    "portfolios": "6_Portfolios_ME_Prior_12_2_CSV.csv",
    "size": "ME_Breakpoints_CSV.csv",
    "prior": "Prior_2-12_Breakpoints_CSV.csv",
}
D = Decimal


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def vintage(lines):
    return re.search(r"(\d{6}) CRSP database", lines[0]).group(1)


def previous(month):
    year, number = int(month[:4]), int(month[4:])
    return f"{year - (number == 1):04d}{12 if number == 1 else number - 1:02d}"


def block(lines, heading=None):
    if heading:
        indices = [i for i, line in enumerate(lines) if line.strip() == heading]
        assert len(indices) == 1, heading
        start = indices[0] + 1
    else:
        start = next(i for i, line in enumerate(lines) if line.strip().startswith(",Mom"))
    header = [v.strip() for v in next(csv.reader([lines[start]]))][1:]
    assert header == (LABELS if heading else ["Mom"]), header
    result = {}
    for line in lines[start + 1 :]:
        row = [v.strip() for v in next(csv.reader([line]))]
        if not row or not re.fullmatch(r"\d{6}", row[0]):
            if result:
                break
            continue
        assert len(row) == len(header) + 1 and row[0] not in result
        result[row[0]] = dict(zip(header, row[1:], strict=True))
    months = list(result)
    assert months == sorted(months)
    assert all(previous(b) == a for a, b in zip(months[:-1], months[1:], strict=True))
    return result


def breaks(lines):
    result = {}
    for line in lines:
        row = [v.strip() for v in next(csv.reader([line]))]
        if row and re.fullmatch(r"\d{6}", row[0]):
            assert len(row) == 22 and row[0] not in result
            values = [D(v) for v in row[2:]]
            assert values == sorted(values)
            result[row[0]] = {
                "nyse_count": int(row[1]),
                "p30": str(values[5]),
                "p50": str(values[9]),
                "p70": str(values[13]),
            }
    return result


def momentum(row):
    return (
        D(row["SMALL HiPRIOR"])
        + D(row["BIG HiPRIOR"])
        - D(row["SMALL LoPRIOR"])
        - D(row["BIG LoPRIOR"])
    ) / 2


def half_unit(text):
    return D(5).scaleb(D(text).as_tuple().exponent - 1)


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    plan = read(DOC / "plan.json")
    inputs = {str((DOC / "plan.json").relative_to(ROOT)): sha(DOC / "plan.json")}
    details, summaries = [], []
    for version in ("current", "legacy202412"):
        paths = {key: DATA / version / name for key, name in FILES.items()}
        lines = {key: p.read_text(encoding="utf-8-sig").splitlines() for key, p in paths.items()}
        vintages = {key: vintage(ls) for key, ls in lines.items()}
        for path in paths.values():
            inputs[str(path.relative_to(ROOT))] = sha(path)
            archive = path.with_suffix(".zip")
            receipt = archive.with_suffix(".zip.receipt.json")
            assert sha(archive) == read(receipt)["sha256"]
            inputs[str(archive.relative_to(ROOT))] = sha(archive)
            inputs[str(receipt.relative_to(ROOT))] = sha(receipt)
        factor = block(lines["factor"])
        weighted = block(lines["portfolios"], "Average Value Weighted Returns -- Monthly")
        equal = block(lines["portfolios"], "Average Equal Weighted Returns -- Monthly")
        counts = block(lines["portfolios"], "Number of Firms in Portfolios")
        sizes, priors = breaks(lines["size"]), breaks(lines["prior"])
        summary = {
            "dataset": version,
            "source_crsp_vintages": vintages,
            "same_vintage": len(set(vintages.values())) == 1,
            "factor_range": [min(factor), max(factor)],
            "portfolio_range": [min(weighted), max(weighted)],
            "factor_months": len(factor),
            "portfolio_months": len(weighted),
            "security_level_memberships_available": False,
        }
        if not summary["same_vintage"]:
            summary.update(
                component_check_performed=False,
                component_match_passed=False,
                reason="Different raw CRSP vintages; do not calculate a false replication residual.",
            )
            summaries.append(summary)
            continue
        months = sorted(set(factor) & set(weighted))
        assert set(factor) == set(weighted), "Unexplained same-vintage monthly coverage difference"
        residuals, skipped, control_failures, contracts = (
            [],
            [],
            dict.fromkeys(["reversed_sign", "one_month_shift", "equal_weight"], 0),
            [],
        )
        for month in months:
            row, published = weighted[month], factor[month]["Mom"]
            required = [row[k] for k in LABELS if k not in ["ME1 PRIOR2", "ME2 PRIOR2"]] + [
                published
            ]
            if any(D(v) in [D("-99.99"), D(-999)] for v in required):
                skipped.append({"month": month, "reason": "Publisher missing return sentinel"})
                continue
            expected, reported = momentum(row), D(published)
            bound = min(
                D(plan["tolerance_percent_points_max"]),
                sum((half_unit(v) / 2 for v in required[:-1]), D(0)) + half_unit(published),
            )
            error = expected - reported
            residuals.append(abs(error))
            details.append(
                {
                    "dataset": version,
                    "month": month,
                    "reconstructed_percent": str(expected),
                    "published_percent": published,
                    "residual_percent_points": str(error),
                    "rounding_bound_percent_points": str(bound),
                    "match": abs(error) <= bound,
                }
            )
            controls = {
                "reversed_sign": -expected - reported,
                "equal_weight": momentum(equal[month]) - reported,
            }
            prior = previous(month)
            if prior in factor:
                controls["one_month_shift"] = expected - D(factor[prior]["Mom"])
            for label, residual in controls.items():
                control_failures[label] += abs(residual) > bound
            if (
                plan["primary_comparison_window"][0].replace("-", "")
                <= month
                <= plan["primary_comparison_window"][1].replace("-", "")
            ):
                assert prior in sizes and prior in priors
                count = [int(v) for v in counts[month].values()]
                assert all(n >= 0 for n in count)
                contracts.append(
                    {
                        "return_month": month,
                        "formation_month": prior,
                        "nyse_median_me_million_usd": sizes[prior]["p50"],
                        "nyse_prior_return_p30_percent": priors[prior]["p30"],
                        "nyse_prior_return_p70_percent": priors[prior]["p70"],
                        "published_six_portfolio_stock_counts": count,
                        "identities_reconstructed": False,
                    }
                )
        relevant = [r for r in details if r["dataset"] == version]
        assert len(contracts) == 17
        negatives_rejected = all(n > 0 for n in control_failures.values())
        summary.update(
            component_check_performed=True,
            months_checked=len(relevant),
            skipped=skipped,
            mismatches=sum(not r["match"] for r in relevant),
            max_absolute_residual_percent_points=str(max(residuals)),
            negative_control_mismatch_counts=control_failures,
            negative_controls_rejected=negatives_rejected,
            component_match_passed=all(r["match"] for r in relevant) and negatives_rejected,
            complete_month_contracts=contracts,
        )
        summaries.append(summary)
    result = {
        "schema": "french-publisher-component-calibration.v1",
        "datasets": summaries,
        "calibration_scope": "Publisher portfolio components to published factor only; not security-level formation or independent raw-market reconstruction.",
        "original_stock_universe_replicated": False,
        "existing_long_cash_engine_calibrated_for_original_long_short_strategy": False,
        "user_has_crsp_access": False,
        "broker_authenticated": False,
        "real_execution_verified": False,
        "new_financial_accounts": 0,
        "profit_goal_met": False,
    }
    save(output / "summary.json", result)
    save(output / "monthly-residuals.json", details)
    save(
        output / "registration.json",
        {
            "inputs_sha256": inputs,
            "analysis_sha256": sha(Path(__file__)),
            "independent_external_attestation": False,
        },
    )
    (output / "analysis_source.py").write_bytes(Path(__file__).read_bytes())
    print(
        json.dumps(
            {
                "datasets": [
                    {k: v for k, v in r.items() if k != "complete_month_contracts"}
                    for r in summaries
                ],
                "stock_level_passed": False,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
