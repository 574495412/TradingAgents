"""验证原生多空研究审查的范围、非有限值拒绝和研究假说约束。"""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tradingagents.research.contracts import digest
from tradingagents.research.strategy_review import (
    FixtureLLM,
    Hypothesis,
    Review,
    ReviewRequest,
    run,
)


def request():
    return ReviewRequest(
        schema_version="ta-factor-review-request.v1",
        evidence={"window": "15 days", "independent_oos": False},
        backend="fixture",
    )


def test_native_researchers_and_review_have_no_production_authority():
    r = request()
    result = run(r)
    assert result["request_sha256"] == digest(r.model_dump())
    assert result["debate"].count("联调替身") >= 2
    assert result["calls"] == [] and result["model_inference_performed"] is False
    assert result["production_eligible"] is False and result["orders_enabled"] is False
    assert result["review"]["production_eligible"] is False


def test_true_production_claim_is_not_valid_structured_output():
    with pytest.raises(ValidationError):
        Review(
            summary="研究",
            evidence_conflicts=[],
            rejected_directions=[],
            hypotheses=[],
            stability_gates=[],
            production_eligible=True,
        )


def test_nonfinite_nested_evidence_is_rejected_before_model_calls():
    r = request().model_copy(update={"evidence": {"nested": [float("nan")]}})
    with pytest.raises(ValueError):
        run(r)


def test_unknown_request_fields_are_rejected():
    with pytest.raises(ValidationError):
        ReviewRequest.model_validate(request().model_dump() | {"orders_enabled": True})


def test_duplicate_hypothesis_identity_is_rejected():
    h = Hypothesis(
        id="same",
        mechanism="机制",
        public_inputs=["公开输入"],
        causal_trigger="已完成信号",
        invalidation="反向证据",
        cost_requirement="计入完整成本",
        frozen_test_design="事前冻结",
        failure_risks=["缺少前向验证"],
    )

    class Duplicate(FixtureLLM):
        def with_structured_output(self, schema):
            return SimpleNamespace(
                invoke=lambda prompt: schema(
                    summary="研究",
                    evidence_conflicts=[],
                    rejected_directions=[],
                    hypotheses=[h, h],
                    stability_gates=[],
                    production_eligible=False,
                )
            )

    with pytest.raises(ValueError, match="编号"):
        run(request(), Duplicate())
