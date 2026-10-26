import json, hashlib
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[3]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


a = ROOT / "results/research/equity_arelle_review_20261026_v002"
s = json.loads((a / "summary.json").read_text())
assert (
    s["selected_annual_versions"] == 20
    and s["reviewed_versions"] == 20
    and s["numeric_matches"] == 45
    and s["all_consumed_inputs_match"]
    and s["all_xbrl_validations_passed"]
)
reg = json.loads((a / "registration.json").read_text())
source_checks = {
    f: sha(ROOT / "tradingagents/research/equities" / f) == v
    for f, v in reg["source_hashes"].items()
}
assert all(source_checks.values())
old = json.loads(
    (
        ROOT / "results/research/equities_pit_stocks_20261026_v002/original-annual-audit.json"
    ).read_text()
)
old = {r["accession"]: r for r in old["records"]}
provenance = json.loads(
    (
        ROOT
        / "results/research/equities_pit_stocks_20261026_v002/financial-decision-provenance.json"
    ).read_text()
)
selected = {r["accession"] for r in provenance if "2022-12-30" <= r["decision_day"] < "2024-06-27"}
assert selected == set(reg["selected_accessions"])
roundtrips = 0
clock_checks = 0
for r in s["records"]:
    acc = r["accession"]
    raw = ROOT / "data/research/equity_arelle_20261026_v001/complete-submissions" / f"{acc}.txt"
    assert sha(raw) == r["source_receipt"]["sha256"]
    body = raw.read_bytes()
    ex = json.loads((a / acc / "extraction.json").read_text())
    assert ex["parent_sha256"] == sha(raw)
    for doc in ex["documents"]:
        frozen = a / acc / "documents" / doc["filename"]
        assert (
            hashlib.sha256(body[doc["source_start"] : doc["source_end"]]).hexdigest()
            == doc["sha256"]
            == sha(frozen)
        )
        roundtrips += 1
    payload = json.loads((a / acc / "arelle.json").read_text())
    assert (
        sha(a / acc / "arelle.json") == r["tool_output_sha256"]
        and payload["full_validation_passed"]
        and not payload["validation_log"]["log"]
    )
    available = datetime.fromisoformat(r["raw_acceptance_at_utc"]) + timedelta(hours=1)
    assert available == datetime.fromisoformat(old[acc]["available_at"])
    for obs in provenance:
        if obs["accession"] == acc and "2022-12-30" <= obs["decision_day"] < "2024-06-27":
            assert datetime.fromisoformat(obs["decision_close"]) >= available
            clock_checks += 1
p = ROOT / "results/research/equity_depth_attribution_20261026_v002"
d = json.loads((p / "diagnostics.json").read_text())
at = json.loads((p / "attribution.json").read_text())
rr = json.loads((p / "registration.json").read_text())
assert (
    sha(p / "attribution_source.py")
    == rr["source_sha256"]
    == sha(ROOT / "tradingagents/research/equities/mechanism_attribution.py")
)
assert all(sha(ROOT / f) == v for f, v in rr["inputs_sha256"].items())
xdata = pd.read_csv(p / "paired-daily-returns.csv", index_col="date")
factors = ["SPY_gross_total_return", "QQQ_minus_SPY_gross_return"]
X = np.column_stack([np.ones(len(xdata)), xdata[factors].to_numpy()])
n, k = X.shape
inverse = np.linalg.inv(X.T @ X)
math_checks = {}
for name, y in [
    ("net_account_regression", xdata[d["focal_account"]].to_numpy()),
    (
        "paired_net_difference_regression",
        (xdata[d["focal_account"]] - xdata["US_disclosed_equal_10000_15bps"]).to_numpy(),
    ),
]:
    beta = inverse @ X.T @ y
    score = X * (y - X @ beta)[:, None]
    S = score.T @ score
    for lag in range(1, 6):
        gamma = score[lag:].T @ score[:-lag]
        S += (1 - lag / 6) * (gamma + gamma.T)
    covariance = n / (n - k) * inverse @ S @ inverse
    se = np.sqrt(np.diag(covariance))
    critical = t.ppf(0.975, n - k)
    ci = np.column_stack([beta - critical * se, beta + critical * se])
    reported = d[name]["coefficients"]
    cols = ["intercept_daily_return", *factors]
    error = max(
        np.max(np.abs(beta - [reported[c]["value"] for c in cols])),
        np.max(np.abs(se - [reported[c]["standard_error_hac"] for c in cols])),
        np.max(np.abs(ci - np.array([reported[c]["confidence95"] for c in cols]))),
    )
    assert error < 1e-10
    math_checks[name] = {
        "OLS_and_HAC_Bartlett5_independent_matrix_error": float(error),
        "passed": True,
    }
for name, r in at.items():
    state = json.loads(
        (
            ROOT / "results/research/equities_pit_stocks_20261026_v002" / name / "state.json"
        ).read_text()
    )
    assert (
        abs(
            sum(z["net_marked_pnl"] for z in r["stocks"])
            - (state["equity_curve"][-1]["equity"] - 10000)
        )
        < 1e-7
    )
result = {
    "schema": "equity-depth-verification.v1",
    "role": "Internal provenance/arithmetic/method crosschecks; not external independent attestation or market/return certification",
    "raw_document_byte_roundtrips": roundtrips,
    "decision_availability_checks": clock_checks,
    "active_annual_versions": 20,
    "raw_input_values_matched": 45,
    "full_arelle_xml_validations_passed": 20,
    "new_economic_accounts": 0,
    "source_hash_checks": source_checks,
    "math_crosschecks": math_checks,
    "attributed_accounts": len(at),
    "daily_stock_attribution_rows": len(pd.read_csv(p / "daily-stock-pnl.csv")),
    "external_human_attestation": False,
    "profit_goal_met": False,
}
(ROOT / "docs/research/equities_depth_review_20261026/verification.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print(json.dumps(result))
