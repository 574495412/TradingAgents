"""Conditional module evidence gate, separate from total portfolio objectives."""

import math


def assess_module(cases, protocol, *, domain_registered, sleeve_capacity_passed, cluster_groups):
    """Research qualification is neither portfolio performance nor production approval.

    Caller supplies matched-cost accounts for one preregistered opportunity set.
    Existing frozen legacy verdicts must remain immutable.
    """
    if (
        protocol.get("fixed_annual_hurdle_applies") is not False
        or protocol.get("minimum_cagr") is not None
    ):
        raise ValueError("Portfolio annual hurdle cannot be applied to an individual module")
    result = {
        "status": "invalid_evidence",
        "eligible_for_portfolio_test": False,
        "production_eligible": False,
        "reasons": [],
    }
    if not domain_registered:
        result["reasons"].append("domain_not_registered_before_evaluation")
        return result
    if len(cases) != 2 or {r["slippage_bps"] for r in cases} != {3, 10}:
        result["reasons"].append("two_matched_cost_cases_required")
        return result
    if len({(r["capital"], r["start"], r["end"]) for r in cases}) != 1:
        result["reasons"].append("capital_or_opportunity_window_differs")
        return result
    if any(not r.get("audit_passed") for r in cases):
        result["reasons"].append("financial_audit_unverified")
        return result
    required = ("net_return", "closed_net_pnl", "max_gross_exposure", "closed_trades")
    if any(
        r["metrics"].get(key) is None or not math.isfinite(r["metrics"][key])
        for r in cases
        for key in required
    ):
        result.update(status="evidence_insufficient", reasons=["financial_metric_unavailable"])
        return result
    if any(r["metrics"]["closed_trades"] == 0 for r in cases):
        result.update(status="evidence_insufficient", reasons=["no_completed_lifecycle_evidence"])
        return result
    positive = [
        r["metrics"]["net_return"] > 0 and r["metrics"]["closed_net_pnl"] > 0 for r in cases
    ]
    if not all(positive):
        result.update(
            status="cost_sensitive_current_domain"
            if any(positive)
            else "no_net_edge_current_domain",
            reasons=["nonpositive_marked_or_closed_pnl_after_costs"],
        )
        return result
    if any(r["risk_halted"] or r["metrics"]["max_gross_exposure"] > 1 for r in cases):
        result.update(
            status="risk_failed_current_domain", reasons=["declared_risk_halt_or_marked_leverage"]
        )
        return result
    counts = [
        r["metrics"].get("matched_closed_pairs", r["metrics"]["closed_trades"]) for r in cases
    ]
    if any(n is None or not math.isfinite(n) or n < 0 for n in counts):
        result.update(status="evidence_insufficient", reasons=["lifecycle_evidence_unavailable"])
        return result
    n = min(counts)
    result["minimum_complete_lifecycles"] = n
    if (
        n < protocol["minimum_complete_lifecycles"]
        or cluster_groups is None
        or cluster_groups < protocol["minimum_cluster_groups"]
    ):
        result.update(
            status="evidence_insufficient", reasons=["lifecycle_or_cluster_evidence_insufficient"]
        )
        return result
    if sleeve_capacity_passed is not True:
        result.update(
            status="allocated_sleeve_capacity_unverified",
            reasons=["intended_allocation_capacity_not_proven"],
        )
        return result
    result.update(status="research_qualified_for_portfolio_test", eligible_for_portfolio_test=True)
    return result
