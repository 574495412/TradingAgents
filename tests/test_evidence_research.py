"""验证因果证据边界、真实原生研究节点以及 Codex 子进程的失败语义。"""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tradingagents.research import codex
from tradingagents.research.contracts import Assessment, Request, digest
from tradingagents.research.worker import FixtureLLM, run


@pytest.fixture
def payload():
    text = "已结算资金费率为 0.0001；这不是下一次结算的预告。"
    return {
        "schema_version": "ta-evidence-request.v1",
        "mode": "single",
        "backend": "fixture",
        "model": None,
        "timeout_seconds": 30,
        "snapshot": {
            "symbol": "BTCUSDT",
            "as_of_ms": 10000,
            "features_available_at_ms": 9000,
            "valid_until_ms": 20000,
            "availability_basis": "observed",
            "source_sha256": "a" * 64,
            "data_gaps": ["没有新闻档案"],
            "features": {"momentum": 0.3, "flow": 0.2, "crowding": 0.0, "activity": 0.8},
            "evidence": [
                {
                    "id": "funding-1",
                    "kind": "funding",
                    "source_url": "https://example.org/public/funding",
                    "published_at_ms": 8000,
                    "available_at_ms": 9000,
                    "availability_basis": "observed",
                    "expires_at_ms": 18000,
                    "text": text,
                    "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "funding_rate": 0.0001,
                }
            ],
        },
    }


@pytest.mark.parametrize("mode", ["single", "multi"])
def test_fixture_uses_shared_contract_and_native_nodes(payload, mode):
    payload["mode"] = mode
    response = run(Request.model_validate(payload))
    assert response["assessment"]["event_score"] == pytest.approx(-0.1)
    assert response["backend"] == "fixture" and response["calls"] == []
    assert bool(response["debate"]) == (mode == "multi")
    assert not response["orders_enabled"] and not response["economic_performance_evaluated"]
    assert response["request_sha256"] == digest(payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("available_at_ms", 10001),
        ("expires_at_ms", 10000),
        ("text_sha256", "0" * 64),
        ("funding_rate", float("nan")),
    ],
)
def test_reject_future_stale_or_corrupt_evidence(payload, field, value):
    payload["snapshot"]["evidence"][0][field] = value
    with pytest.raises(ValidationError):
        Request.model_validate(payload)


def test_unknown_fields_and_duplicates_rejected(payload):
    payload["snapshot"]["leader_trade"] = "must not enter research"
    with pytest.raises(ValidationError):
        Request.model_validate(payload)
    del payload["snapshot"]["leader_trade"]
    payload["snapshot"]["evidence"] *= 2
    with pytest.raises(ValidationError):
        Request.model_validate(payload)


def test_unknown_model_citations_are_not_accepted(payload):
    parsed = Request.model_validate(payload)
    llm = FixtureLLM(parsed.snapshot)
    llm.with_structured_output = lambda schema: SimpleNamespace(
        invoke=lambda prompt: Assessment(
            event_score=0.5,
            rationale="示例",
            hypothesis="示例",
            invalidation="示例",
            evidence_ids=["invented"],
            missing_data=[],
        )
    )
    with pytest.raises(ValueError, match="unknown"):
        run(parsed, llm)


def test_codex_runs_in_empty_directory_without_user_tools(monkeypatch):
    def fake_process(command, prompt, timeout, cwd):
        assert "--ignore-user-config" in command and "--ephemeral" in command
        assert "read-only" in command and 'web_search="disabled"' in command
        assert "features.shell_tool=false" in command and "features.plugins=false" in command
        assert "features.multi_agent=false" in command
        assert (
            "memories.use_memories=false" in command
            and "memories.generate_memories=false" in command
        )
        assert timeout == 15 and prompt == "仅分析传入证据"
        assert not list(Path(cwd).iterdir())
        Path(command[command.index("--output-last-message") + 1]).write_text("完成")
        return json.dumps(
            {"type": "turn.completed", "usage": {"input_tokens": 9, "output_tokens": 2}}
        )

    monkeypatch.setattr(codex, "run_process", fake_process)
    llm = codex.CodexLLM(timeout_seconds=15)
    assert llm.invoke("仅分析传入证据").content == "完成"
    assert llm.calls[0]["usage"]["input_tokens"] == 9


def test_codex_tool_activity_fails_closed(monkeypatch):
    monkeypatch.setattr(
        codex,
        "run_process",
        lambda *args: json.dumps({"type": "item.completed", "item": {"type": "command_execution"}}),
    )
    with pytest.raises(RuntimeError, match="non-inference"):
        codex.CodexLLM().invoke("示例")


def test_codex_empty_answer_is_failure(monkeypatch):
    monkeypatch.setattr(codex, "run_process", lambda *args: "")
    with pytest.raises(RuntimeError, match="final answer"):
        codex.CodexLLM().invoke("示例")


def test_strict_schema_recurses():
    schema = codex.strict_schema(Assessment.model_json_schema())
    assert set(schema["required"]) == set(schema["properties"])
    assert schema["additionalProperties"] is False


def test_recovered_cli_errors_require_a_successful_final_turn(monkeypatch):
    def recovered(command, prompt, timeout, cwd):
        Path(command[command.index("--output-last-message") + 1]).write_text("完成")
        return "\n".join(
            json.dumps(e)
            for e in [
                {"type": "item.completed", "item": {"type": "error", "message": "可恢复诊断"}},
                {"type": "error", "message": "重试"},
                {"type": "turn.completed", "usage": {"output_tokens": 2}},
            ]
        )

    monkeypatch.setattr(codex, "run_process", recovered)
    llm = codex.CodexLLM()
    assert llm.invoke("示例").content == "完成"
    assert llm.calls[0]["recovered_diagnostic_events"] == 2


def test_failed_cli_turn_cannot_use_an_answer_file(monkeypatch):
    def failed(command, prompt, timeout, cwd):
        Path(command[command.index("--output-last-message") + 1]).write_text("旧结果")
        return json.dumps({"type": "turn.failed"})

    monkeypatch.setattr(codex, "run_process", failed)
    with pytest.raises(RuntimeError, match="turn failed"):
        codex.CodexLLM().invoke("示例")


def test_system_proxy_is_inherited_without_trading_credentials(monkeypatch):
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.delenv("https_proxy", raising=False)
    monkeypatch.setenv("EXCHANGE_API_SECRET", "test-only-secret")
    monkeypatch.setattr(codex, "getproxies", lambda: {"https": "http://127.0.0.1:7897"})
    env = codex.inference_environment()
    assert env["HTTPS_PROXY"] == "http://127.0.0.1:7897"
    assert "EXCHANGE_API_SECRET" not in env
