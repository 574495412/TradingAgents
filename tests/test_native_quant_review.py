"""实际原生图的研发选择不能跳过角色、虚构编号或把自由文本当结构化成功。"""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tradingagents.research.native_quant_review import (
    DATA_POLICY,
    INDEX_VISIBILITY,
    METRICS_VISIBILITY,
    NODES,
    PANEL_END,
    PANEL_FALSE_FLAGS,
    PANEL_ROWS_5M,
    PANEL_ROWS_HOURLY,
    PANEL_START,
    POLICY,
    SOURCE_COLUMNS,
    SOURCE_INPUTS,
    NativeQuantRequest,
    SourcePanelCapability,
    run,
)
from tradingagents.research.quant.artifacts import read, save, sha, verify_receipts


class RecordingLLM:
    """确定性替身仅验证图和数据交接，不宣称真实模型推理或盈利。"""

    def __init__(
        self,
        *,
        choice="candidate_one",
        empty=False,
        fail_portfolio=False,
        price=None,
        public_inputs=None,
    ):
        self.calls, self.choice, self.empty, self.fail_portfolio, self.price = (
            [],
            choice,
            empty,
            fail_portfolio,
            price,
        )
        self.public_inputs = ["minute_close"] if public_inputs is None else public_inputs

    def invoke(self, prompt):
        self.calls.append({"kind": "text", "prompt": prompt})
        return SimpleNamespace(content="只依据封存证据讨论候选，尚无盈利验证。")

    def with_structured_output(self, schema):
        def invoke(prompt):
            self.calls.append({"kind": schema.__name__, "prompt": prompt})
            if schema.__name__ == "Review":
                return schema(
                    summary="人工联调假说，不能用于经济结论。",
                    evidence_conflicts=[],
                    rejected_directions=[],
                    stability_gates=[],
                    production_eligible=False,
                    hypotheses=[]
                    if self.empty
                    else [
                        {
                            "id": "candidate_one",
                            "mechanism": "人工联调机制",
                            "public_inputs": self.public_inputs,
                            "causal_trigger": "人工联调公式",
                            "invalidation": "联调不验证盈利",
                            "cost_requirement": "完整成本",
                            "frozen_test_design": "保持原时间隔离",
                            "failure_risks": ["人工样例"],
                        }
                    ],
                )
            if schema.__name__ == "ResearchPlan":
                return schema(
                    recommendation="Hold", rationale="证据有限", strategic_actions="仅启动量化实验"
                )
            if schema.__name__ == "TraderProposal":
                assert isinstance(prompt, list)
                return schema(
                    action="Hold",
                    reasoning="研发篮子无报价",
                    entry_price=None,
                    stop_loss=None,
                    position_sizing=None,
                )
            if schema.__name__ == "QuantPortfolioDecision":
                if self.fail_portfolio:
                    raise RuntimeError("人工结构化失败")
                return schema(
                    rating="Hold",
                    executive_summary="仅批准待验证的实验",
                    investment_thesis="仍需成本检验",
                    price_target=self.price,
                    time_horizon=None,
                    hypothesis_decisions=[]
                    if self.empty
                    else [
                        {"hypothesis_id": self.choice, "decision": "test", "reason": "人工联调选择"}
                    ],
                )
            raise AssertionError(schema)

        return SimpleNamespace(invoke=invoke)


def request():
    return NativeQuantRequest(
        schema_version="ta-native-quant-review-request.v1",
        evidence={"fixture": True},
        backend="fixture",
    )


def test_real_native_downstream_graph_hands_off_proposals_and_preserves_scope():
    llm, stages = RecordingLLM(), []
    result = run(request(), llm, lambda node, update, calls: stages.append((node, update, calls)))
    assert tuple(result["native_node_route"]) == NODES
    assert len(result["native_downstream_nodes"]) == 8
    assert len(llm.calls) == 9
    assert result["selected_hypothesis_ids"] == ["candidate_one"]
    assert result["custom_analyst_nodes"] == ["Market Analyst"]
    assert result["model_inference_performed"] is result["economic_increment_demonstrated"] is False
    assert result["real_time_forward_required"] is result["orders_enabled"] is False
    assert result["standard_tradingagentsgraph_propagate_used"] is False
    assert set(result["native_structured_contracts"]) == {
        "ResearchPlan",
        "TraderProposal",
        "PortfolioDecision",
    }
    assert all(len(calls) == 1 for _, _, calls in stages)
    assert "candidate_one" in result["stages"]["Market Analyst"]["market_report"]


def test_no_new_hypothesis_is_a_valid_recorded_result():
    result = run(request(), RecordingLLM(empty=True))
    assert result["selected_hypothesis_ids"] == []
    assert result["analyst_review"]["hypotheses"] == []
    assert len(result["native_node_route"]) == 9


@pytest.mark.parametrize("failure", ["invented_id", "structured_fallback", "invented_price"])
def test_invalid_native_handoff_cannot_become_an_approved_experiment(failure):
    llm = RecordingLLM(
        choice="invented" if failure == "invented_id" else "candidate_one",
        fail_portfolio=failure == "structured_fallback",
        price=1000.0 if failure == "invented_price" else None,
    )
    with pytest.raises(ValueError):
        run(request(), llm)


def test_fixture_requires_explicit_fake_and_oversized_evidence_never_runs_a_model():
    with pytest.raises(ValueError, match="替身"):
        run(request())
    oversized = request()
    oversized.evidence = {"text": "x" * 24_001}
    llm = RecordingLLM()
    with pytest.raises(ValueError, match="大小"):
        run(oversized, llm)
    assert not llm.calls


def test_explicit_larger_evidence_envelope_retains_legacy_default_and_hard_maximum():
    req = request()
    original = req.model_dump()
    assert "evidence_limit_bytes" not in original
    req.evidence = {"text": "x" * 25000}
    req.evidence_limit_bytes = 32000
    assert req.model_dump()["evidence_limit_bytes"] == 32000
    result = run(req, RecordingLLM())
    assert result["production_eligible"] is False
    assert len(result["calls"]) == 9
    req.evidence = {"text": "x" * 32001}
    llm = RecordingLLM()
    with pytest.raises(ValueError, match="大小"):
        run(req, llm)
    assert llm.calls == []
    with pytest.raises(ValueError):
        NativeQuantRequest.model_validate({**original, "evidence_limit_bytes": 64000})
    req.evidence_limit_bytes = 64000
    with pytest.raises(ValueError, match="固定档位"):
        run(req, llm)
    assert llm.calls == []


@pytest.mark.parametrize("where", ["copy", "inference"])
def test_source_copy_or_model_failure_still_seals_existing_evidence(tmp_path, monkeypatch, where):
    from tradingagents.research import native_quant_review as native

    path, output = tmp_path / "request.json", tmp_path / "result"
    save(path, request().model_dump())
    if where == "copy":
        original = Path.open

        def fail_copy(self, mode="r", *args, **kwargs):
            if mode == "xb" and self.suffix == ".py":
                raise OSError("人工源码复制失败")
            return original(self, mode, *args, **kwargs)

        monkeypatch.setattr(Path, "open", fail_copy)
    else:
        monkeypatch.setattr(
            native,
            "run",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("人工推理失败")),
        )
    with pytest.raises((OSError, RuntimeError)):
        native.execute(path, output)
    verify_receipts(output)
    assert read(output / "failure.json")["orders_enabled"] is False
    assert read(output / "model_attempts.json") == []
    assert not (output / "result.json").exists()


def source_panel(tmp_path):
    """仅造元数据和字节身份，不伪造真实Parquet人口或历史首次可见证明。"""
    root = tmp_path / "source_panel"
    root.mkdir()
    protocol = {
        "schema": "ta-basis-oi-source-panel.v1",
        "symbols": [f"S{index}" for index in range(748)],
        "start_ms": PANEL_START,
        "end_exclusive_ms": PANEL_END,
        "rows_5m": PANEL_ROWS_5M,
        "rows_hourly": PANEL_ROWS_HOURLY,
        "metrics_visibility": METRICS_VISIBILITY,
        "metrics_delay_ms": 600000,
        "index_visibility": INDEX_VISIBILITY,
        "unknowns_preserved": True,
        **dict.fromkeys(PANEL_FALSE_FLAGS, False),
    }
    save(root / "protocol.json", protocol)
    save(
        root / "result.json",
        {
            "status": "COMPLETED_WITH_UNKNOWN",
            "protocol_sha256": sha(root / "protocol.json"),
            "rows_5m": PANEL_ROWS_5M,
            "rows_hourly": PANEL_ROWS_HOURLY,
            **dict.fromkeys(PANEL_FALSE_FLAGS, False),
        },
    )
    save(
        root / "coverage.json",
        {
            "schema": "ta-native-source-input-coverage.v1",
            "fields": {
                name: {
                    "source_column": name,
                    "known_rows": 20,
                    "unknown_rows": PANEL_ROWS_5M - 20,
                    "time_basis": INDEX_VISIBILITY
                    if name.startswith("index_1m_")
                    else METRICS_VISIBILITY,
                    "lag_ms": 0 if name.startswith("index_1m_") else 600000,
                    "scope": "panel_5m",
                    # 额外来源说明不得沿能力摘要意外传到模型。
                    "private_local_path": str(root),
                }
                for name in SOURCE_COLUMNS
            },
            "full_axes": {"source_only": True},
        },
    )
    (root / "synthetic_raw.bin").write_bytes("仅验证不可变来源字节".encode())
    # 成功合成用例只替换独立footer读取；下方另以真实小Parquet验证拒绝边界。
    for name in ("panel_5m.parquet", "panel_hourly.parquet"):
        (root / name).write_bytes(b"synthetic-footer-only")
    return root


@pytest.fixture
def synthetic_panel_footer(monkeypatch):
    from tradingagents.research import native_quant_review as native

    def footer(path):
        return {
            "num_rows": PANEL_ROWS_5M if path.name == "panel_5m.parquet" else PANEL_ROWS_HOURLY,
            "columns": [("symbol", "string"), ("clock_ms", "int64")]
            + [(name, "string") for name in SOURCE_COLUMNS]
            + [(name + "_value_status", "string") for name in SOURCE_COLUMNS],
        }

    monkeypatch.setattr(native, "panel_footer", footer)


def reseal_capability(root, inputs=("index_1m_close", "sum_open_interest")):
    """合成来源可以重新登记字节，真实封存结果从不经此测试函数修改。"""
    receipt = root / "receipts.json"
    receipt.write_text(
        json.dumps({p.name: sha(p) for p in root.iterdir() if p.is_file() and p != receipt}),
        encoding="utf-8",
    )
    return SourcePanelCapability(
        schema_version="ta-native-source-capability.v1",
        panel_root=str(root),
        receipts_sha256=sha(receipt),
        protocol_sha256=sha(root / "protocol.json"),
        result_sha256=sha(root / "result.json"),
        coverage_sha256=sha(root / "coverage.json"),
        declared_inputs=list(inputs),
        metrics_visibility=METRICS_VISIBILITY,
        metrics_delay_ms=600000,
        index_visibility=INDEX_VISIBILITY,
    )


def rewrite_json(path, transform):
    value = read(path)
    transform(value)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def test_legacy_optional_absence_preserves_dump_digest_and_exact_prompt():
    from tradingagents.research.contracts import digest

    old = {
        "schema_version": "ta-native-quant-review-request.v1",
        "evidence": {"fixture": True},
        "model": None,
        "timeout_seconds": 240,
        "backend": "fixture",
    }
    req = NativeQuantRequest.model_validate(old)
    assert req.model_dump() == json.loads(req.model_dump_json()) == old
    assert NativeQuantRequest.model_validate({**old, "data_capabilities": None}).model_dump() == old
    assert (
        NativeQuantRequest.model_validate({**old, "hedged_capabilities": None}).model_dump() == old
    )
    assert digest(req.model_dump()) == digest(old)
    # 与扩展前v002的封存POLICY原文字节一致，不从当前常量循环定义预期。
    assert (
        hashlib.sha256(POLICY.encode()).hexdigest()
        == "1989631532ee9f813ead857af2eebc9f0de845d2dee81c28886088724c6e91da"
    )
    assert DATA_POLICY in POLICY
    llm = RecordingLLM()
    result = run(req, llm)
    assert "data_capabilities_verified" not in result
    assert (
        llm.calls[0]["prompt"]
        == POLICY
        + '\n公开量化证据：\n{"fixture": true}'
        + "\n请作为量化证据分析师先给出最多两项待检验假说。不要预授盈利资格。"
    )


def test_real_spot_inputs_require_explicit_verified_domain_and_keep_all_nine_roles(monkeypatch):
    from tradingagents.research import native_quant_review as native
    from tradingagents.research.hedged_capability import HedgedSourceCapability

    req = request()
    llm = RecordingLLM(public_inputs=["spot_minute_close", "minute_close"])
    with pytest.raises(ValueError, match="未声明"):
        run(req, llm)
    assert len(llm.calls) == 1
    req.hedged_capabilities = HedgedSourceCapability(
        schema_version="ta-hedged-source-capability.v1",
        panel_root="/UNUSED_PRIVATE_PANEL",
        panel_receipts_sha256="a" * 64,
        audit_root="/UNUSED_PRIVATE_AUDIT",
        audit_receipts_sha256="b" * 64,
    )
    summary = {"declared_inputs": ["spot_minute_close", "minute_close"], "rows": 4}
    # 完整来源语义另由test_hedged_capability验证，此处只验证原生图交接。
    monkeypatch.setattr(native, "verified_hedged_capabilities", lambda cap: summary)
    llm = RecordingLLM(public_inputs=["spot_minute_close", "minute_close"])
    result = run(req, llm)
    assert len(llm.calls) == 9 and tuple(result["native_node_route"]) == NODES
    assert result["hedged_capabilities_verified"] == summary
    assert "UNUSED_PRIVATE" not in llm.calls[0]["prompt"]
    assert "等基础数量与等名义金额" in llm.calls[0]["prompt"]
    req.data_capabilities = SourcePanelCapability(
        schema_version="ta-native-source-capability.v1",
        panel_root="/UNUSED_OI",
        receipts_sha256="c" * 64,
        protocol_sha256="d" * 64,
        result_sha256="e" * 64,
        coverage_sha256="f" * 64,
        declared_inputs=["index_1m_close"],
        metrics_visibility=METRICS_VISIBILITY,
        metrics_delay_ms=600000,
        index_visibility=INDEX_VISIBILITY,
    )
    with pytest.raises(ValueError, match="人口"):
        NativeQuantRequest.model_validate(req.model_dump())
    with pytest.raises(ValueError, match="人口"):
        run(req, llm)


def test_verified_source_opens_only_declared_raw_fields_and_keeps_native_roles(
    tmp_path, synthetic_panel_footer
):
    root = source_panel(tmp_path)
    req = request()
    req.data_capabilities = reseal_capability(root)
    llm = RecordingLLM(public_inputs=["minute_close", "index_1m_close", "sum_open_interest"])
    result = run(req, llm)
    assert tuple(result["native_node_route"]) == NODES and len(llm.calls) == 9
    assert result["native_downstream_nodes"] == list(NODES[1:])
    verified = result["data_capabilities_verified"]
    assert verified["unknowns_preserved"] and verified["symbols_count"] == 748
    assert verified["metrics_delay_ms"] == 600000
    assert verified["field_coverage"]["sum_open_interest"]["unknown_rows"] == PANEL_ROWS_5M - 20
    assert verified["historical_first_publication_time_proven"] is False
    assert verified["oi_stock_economic_clock_proven"] is False
    assert verified["basis_generated"] is False
    assert verified["panel_footers"]["panel_5m.parquet"]["num_rows"] == PANEL_ROWS_5M
    prompts = "\n".join(str(call["prompt"]) for call in llm.calls)
    assert str(root) not in prompts and "private_local_path" not in prompts
    assert "H01结算松动" in prompts and "换名" in prompts and "翻向" in prompts
    assert "同钟、同单位配对" in prompts and "不是独立逐行原文重建" in prompts


@pytest.mark.parametrize(
    "problem",
    [
        "receipt",
        "failed",
        "subset718",
        "unknowns",
        "scope",
        "lag",
        "coverage_lag",
        "count",
        "empty_input",
        "basis_generated",
        "models_loaded",
        "extra_file",
    ],
)
def test_invalid_capabilities_never_call_any_model(tmp_path, synthetic_panel_footer, problem):
    root = source_panel(tmp_path)
    if problem == "failed":
        rewrite_json(root / "result.json", lambda d: d.update(status="FAILED_INTEGRITY"))
    elif problem == "subset718":
        rewrite_json(root / "protocol.json", lambda d: d.update(symbols=d["symbols"][:718]))
    elif problem == "unknowns":
        rewrite_json(root / "protocol.json", lambda d: d.update(unknowns_preserved=False))
    elif problem == "scope":
        rewrite_json(root / "protocol.json", lambda d: d.update(start_ms=PANEL_START + 86_400_000))
    elif problem == "lag":
        rewrite_json(root / "protocol.json", lambda d: d.update(metrics_delay_ms=0))
    elif problem in {"coverage_lag", "count", "empty_input"}:

        def change(d):
            field = d["fields"]["sum_open_interest"]
            if problem == "coverage_lag":
                field["lag_ms"] = 0
            elif problem == "count":
                field["known_rows"] += 1
            else:
                field.update(known_rows=0, unknown_rows=PANEL_ROWS_5M)

        rewrite_json(root / "coverage.json", change)
    elif problem in {"basis_generated", "models_loaded"}:
        rewrite_json(root / "result.json", lambda d: d.update({problem: True}))
    # 同步protocol引用，使测试真正触发语义校验而不是顺便被旧摘要挡住。
    rewrite_json(
        root / "result.json", lambda d: d.update(protocol_sha256=sha(root / "protocol.json"))
    )
    req = request()
    req.data_capabilities = reseal_capability(root)
    if problem == "receipt":
        req.data_capabilities.receipts_sha256 = "0" * 64
    if problem == "extra_file":
        (root / "unsealed.txt").write_text("不在收据", encoding="utf-8")
    llm = RecordingLLM(public_inputs=["sum_open_interest"])
    with pytest.raises(ValueError):
        run(req, llm)
    assert llm.calls == []


@pytest.mark.parametrize(
    "input_name", ["basis", "premium_index", "index_1m_high", "funding_carry_24h"]
)
def test_derived_or_unapproved_raw_names_cannot_be_declared(tmp_path, input_name):
    root = source_panel(tmp_path)
    cap = reseal_capability(root).model_dump()
    cap["declared_inputs"] = [input_name]
    with pytest.raises(ValueError):
        SourcePanelCapability.model_validate(cap)


@pytest.mark.parametrize("input_name", ["basis", "sum_open_interest_value"])
def test_valid_panel_does_not_unlock_unlisted_hypothesis_inputs(
    tmp_path, synthetic_panel_footer, input_name
):
    root = source_panel(tmp_path)
    req = request()
    req.data_capabilities = reseal_capability(root, inputs=("index_1m_close",))
    llm = RecordingLLM(public_inputs=[input_name])
    with pytest.raises(ValueError, match="未声明"):
        run(req, llm)
    assert len(llm.calls) == 1


def test_evidence_text_does_not_grant_new_capabilities():
    req = request()
    req.evidence["available_inputs"] = ["sum_open_interest"]
    with pytest.raises(ValueError, match="未声明"):
        run(req, RecordingLLM(public_inputs=["sum_open_interest"]))


def test_changed_source_after_analyst_cannot_complete_review(tmp_path, synthetic_panel_footer):
    root = source_panel(tmp_path)
    req = request()
    req.data_capabilities = reseal_capability(root)
    llm = RecordingLLM(public_inputs=["sum_open_interest"])

    def mutate(node, update, calls):
        if node == "Market Analyst":
            (root / "synthetic_raw.bin").write_bytes(b"changed")

    with pytest.raises(ValueError, match="封存身份"):
        run(req, llm, mutate)
    assert len(llm.calls) == 9


def test_execute_cannot_write_inside_new_panel(tmp_path):
    from tradingagents.research.native_quant_review import execute

    root = source_panel(tmp_path)
    req = request()
    req.data_capabilities = reseal_capability(root)
    path = tmp_path / "request.json"
    save(path, req.model_dump())
    output = root / "must_not_write"
    with pytest.raises(ValueError, match="新来源面板"):
        execute(path, output)
    assert not output.exists()


@pytest.mark.parametrize("change", ["duplicate", "delay", "visibility"])
def test_capability_declaration_cannot_repeat_inputs_or_choose_a_new_lag(tmp_path, change):
    root = source_panel(tmp_path)
    data = reseal_capability(root).model_dump()
    if change == "duplicate":
        data["declared_inputs"] = ["sum_open_interest", "sum_open_interest"]
    elif change == "delay":
        data["metrics_delay_ms"] = 300000
    else:
        data["metrics_visibility"] = "historically_observed"
    with pytest.raises(ValueError):
        SourcePanelCapability.model_validate(data)


def test_execute_rejects_failed_capability_before_model_construction_and_seals_failure(
    tmp_path, monkeypatch, synthetic_panel_footer
):
    from tradingagents.research import native_quant_review as native

    root = source_panel(tmp_path)
    # 成功字样不能掩盖同一封存目录里的真实失败记录。
    save(root / "failure.json", {"type": "SyntheticFailure"})
    req = request()
    req.backend = "codex"
    req.data_capabilities = reseal_capability(root)
    path, output = tmp_path / "request.json", tmp_path / "failed_review"
    save(path, req.model_dump())

    def forbidden_constructor(*args, **kwargs):
        raise AssertionError("来源验证前不能构建模型客户端")

    monkeypatch.setattr(native, "CodexLLM", forbidden_constructor)
    with pytest.raises(ValueError, match="失败面板"):
        native.execute(path, output)
    verify_receipts(output)
    assert read(output / "model_attempts.json") == []
    assert read(output / "failure.json")["orders_enabled"] is False
    assert not (output / "result.json").exists()


def test_all_seven_verified_raw_inputs_can_be_explicitly_declared(tmp_path, synthetic_panel_footer):
    root = source_panel(tmp_path)
    req = request()
    req.data_capabilities = reseal_capability(root, inputs=SOURCE_INPUTS)
    llm = RecordingLLM(public_inputs=list(SOURCE_INPUTS))
    result = run(req, llm)
    assert result["data_capabilities_verified"]["declared_inputs"] == list(SOURCE_INPUTS)
    assert len(llm.calls) == 9


@pytest.mark.parametrize("problem", ["rows", "missing_status", "float_value", "float_clock"])
def test_real_parquet_footer_rejects_small_or_wrong_typed_panel_before_model(tmp_path, problem):
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = source_panel(tmp_path)
    columns = {"symbol": pa.array(["S0"]), "clock_ms": pa.array([PANEL_START], type=pa.int64())}
    columns.update({name: pa.array([None], type=pa.string()) for name in SOURCE_COLUMNS})
    columns.update({name + "_value_status": pa.array(["UNKNOWN"]) for name in SOURCE_COLUMNS})
    if problem == "missing_status":
        del columns["sum_open_interest_value_status"]
    elif problem == "float_value":
        columns["index_1m_close"] = pa.array([1.0])
    elif problem == "float_clock":
        columns["clock_ms"] = pa.array([float(PANEL_START)])
    for name in ("panel_5m.parquet", "panel_hourly.parquet"):
        pq.write_table(pa.table(columns), root / name)
    req = request()
    req.data_capabilities = reseal_capability(root)
    llm = RecordingLLM(public_inputs=["sum_open_interest"])
    expected = "行数" if problem == "rows" else "类型"
    with pytest.raises(ValueError, match=expected):
        run(req, llm)
    assert llm.calls == []


def test_metadata_only_directory_cannot_be_declared_as_a_source_panel(
    tmp_path, synthetic_panel_footer
):
    root = source_panel(tmp_path)
    (root / "panel_5m.parquet").unlink()
    req = request()
    req.data_capabilities = reseal_capability(root)
    llm = RecordingLLM(public_inputs=["sum_open_interest"])
    with pytest.raises(ValueError, match="缺少实际"):
        run(req, llm)
    assert llm.calls == []
