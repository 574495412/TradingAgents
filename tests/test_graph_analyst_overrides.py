"""替换证据入口时，原生研究、交易与风控流程仍须完整运行。"""

from types import SimpleNamespace

import pytest

from tradingagents.graph import setup
from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.graph.propagation import Propagator


class OfflineModel:
    """只记录提示词并返回固定文本，不具备联网或工具调用能力。"""

    def __init__(self):
        self.prompts = []

    def with_structured_output(self, schema):
        raise NotImplementedError

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return SimpleNamespace(content="**Rating**: Hold\n封存证据不足以支持交易。")


def graph_setup():
    model = OfflineModel()
    graph = setup.GraphSetup(model, model, ConditionalLogic(), max_tool_rounds=1)
    return graph, model


def initial_state():
    return Propagator().create_initial_state(
        "SEALED_QUANT_PORTFOLIO",
        "2026-10-14",
        asset_type="crypto",
        instrument_context="封存量化组合；仅审查已给出的研究证据。",
    )


def forbidden(*args, **kwargs):
    raise AssertionError("替代分析师不得创建原分析师或股票工具节点")


def test_override_runs_every_native_downstream_role_without_stock_tools(monkeypatch):
    """节点顺序来自真实图运行，角色函数使用原实现。"""
    graph, model = graph_setup()
    monkeypatch.setattr(setup, "create_market_analyst", forbidden)
    monkeypatch.setattr(setup, "_analyst_graph", forbidden)
    monkeypatch.setattr(setup, "ToolNode", forbidden)
    report = "封存证据编号 Q-001：完整成本后收益为负，禁止认定稳定。"

    workflow = graph.setup_graph(
        ["market"], analyst_overrides={"market": lambda state: {"market_report": report}}
    )
    compiled = workflow.compile()
    updates = list(compiled.stream(initial_state(), stream_mode="updates"))

    assert [name for update in updates for name in update] == [
        "Market Analyst",
        "Bull Researcher",
        "Bear Researcher",
        "Research Manager",
        "Trader",
        "Aggressive Analyst",
        "Conservative Analyst",
        "Neutral Analyst",
        "Portfolio Manager",
    ]
    final = updates[-1]["Portfolio Manager"]
    assert final["final_rating"] == "Hold"
    assert final["risk_debate_state"]["count"] == 3
    assert len(model.prompts) == 8
    assert report in model.prompts[0]
    assert report in model.prompts[1]
    trader_prompt = model.prompts[3]
    assert isinstance(trader_prompt, list)
    assert report in trader_prompt[1]["content"]


@pytest.mark.parametrize("key", ["news", "unknown"])
def test_override_must_belong_to_selected_analysts(monkeypatch, key):
    graph, _ = graph_setup()
    monkeypatch.setattr(setup, "create_bull_researcher", forbidden)
    with pytest.raises(ValueError, match="已选分析师"):
        graph.setup_graph(["market"], analyst_overrides={key: lambda state: {}})


def test_override_must_be_callable(monkeypatch):
    graph, _ = graph_setup()
    monkeypatch.setattr(setup, "create_bull_researcher", forbidden)
    with pytest.raises(TypeError, match="可调用"):
        graph.setup_graph(["market"], analyst_overrides={"market": object()})


@pytest.mark.parametrize("overrides", [None, {}])
def test_default_path_still_uses_original_analyst_factory_and_subgraph(monkeypatch, overrides):
    graph, _ = graph_setup()

    def original_agent(state):
        return {"market_report": "原分析师报告"}

    factory_calls, subgraph_calls = [], []

    def factory(model):
        factory_calls.append(model)
        return original_agent

    def subgraph(spec, agent, rounds):
        subgraph_calls.append((spec.key, agent, rounds))
        return agent

    monkeypatch.setattr(setup, "create_market_analyst", factory)
    monkeypatch.setattr(setup, "_analyst_graph", subgraph)
    graph.setup_graph(["market"], analyst_overrides=overrides).compile()
    assert factory_calls == [graph.quick_thinking_llm]
    assert subgraph_calls == [("market", original_agent, 1)]


def test_override_and_original_analysts_share_existing_parallel_barrier(monkeypatch):
    graph, model = graph_setup()
    monkeypatch.setattr(setup, "create_market_analyst", forbidden)
    monkeypatch.setattr(
        setup,
        "create_sentiment_analyst",
        lambda llm: lambda state: {"sentiment_report": "原情绪入口的封存报告"},
    )
    subgraphs = []

    def subgraph(spec, agent, rounds):
        subgraphs.append(spec.key)
        return agent

    monkeypatch.setattr(setup, "_analyst_graph", subgraph)
    compiled = graph.setup_graph(
        ["market", "social"],
        analyst_overrides={"market": lambda state: {"market_report": "自定义量化证据"}},
    ).compile()
    compiled.invoke(initial_state())

    assert subgraphs == ["social"]
    assert "自定义量化证据" in model.prompts[0]
    assert "原情绪入口的封存报告" in model.prompts[0]
