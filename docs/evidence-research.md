# Codex 离线证据研究

本 fork 新增 `tradingagents.research`，通过已登录的 Codex CLI 分析调用方提供的公开证据。
它输出可以检验的因子、假设、失效条件和来源引用。量化回归、数据准备及账户对照已由本仓库的 [研究模块](research/README.md) 持有。
模型会话不读取交易账户、带单员标签、项目源码或全局交易记忆；研究接口不提供订单操作。

## 安装与调用

需要 Python 3.11 以上，以及支持 `exec --ignore-user-config --ephemeral --output-schema` 的 Codex CLI。
先用 `codex login status` 确认登录。无需把 ChatGPT 登录信息转换成 API 密钥。
本机已登录不保证额度和网络始终可用，调用失败会返回非零退出码。
子进程沿用环境变量或 macOS 中已经配置的系统代理，不修改全局网络设置。

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m tradingagents.research.worker --describe
.venv/bin/python -m tradingagents.research.worker < request.json > response.json
```

标准输入只接收一份 JSON，标准输出只返回一份 JSON。错误响应的版本为
`ta-evidence-error.v1`，不会生成替代的交易因子。完整输入类型见
`tradingagents/research/contracts.py`。模型调用只发送传入的公开快照和研究提示词。

请求骨架：

```json
{
  "schema_version": "ta-evidence-request.v1",
  "mode": "multi",
  "backend": "codex",
  "model": null,
  "timeout_seconds": 180,
  "snapshot": {
    "symbol": "BTCUSDT",
    "as_of_ms": 1785546060000,
    "features_available_at_ms": 1785546060000,
    "availability_basis": "assumed_publication",
    "valid_until_ms": 1785567660000,
    "features": {"momentum": 0.2, "flow": 0.1, "crowding": 0.0, "activity": 1.0},
    "evidence": [],
    "data_gaps": ["没有事件证据；事件因子必须为零"],
    "source_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  }
}
```

上面的哈希仅为格式示例，调用方必须填入实际数据来源的哈希。
`backend=fixture` 只做确定性联调，不能作为模型有效性证据。
未显式指定 `model` 时，只读取本地 Codex 配置中的模型名；不会加载其中的 MCP、插件或通知配置。
为了实验可复核，调用方应在实验开始前把模型名解析并固定下来。

## 单模型与多角色

- `single`：直接对同一公开快照生成严格结构化的研究结果。
- `multi`：复用本仓库原生 Bull Researcher、Bear Researcher，然后由证据裁决节点生成相同结构的结果。
- 每个角色使用独立的临时 Codex 会话；辩论内容由程序显式传递。首版固定一轮辩论，模型调用次数分别为 1 和 3。
- 结果包含 `baseline_score`、`rule_event_score`，便于调用方建立数值基线和简单规则对照。
- 不启用原版的在线分析员工具、历史复盘记忆或 `TradingAgentsGraph.propagate()` 的行情抓取路径。

`event_score` 在 −1 至 1 之间，仅表示带方向的研究因子；它不是胜率、置信概率、杠杆或仓位比例。
非零结果必须引用请求内的证据 ID。新证据、重复 ID、未来可用时间、过期信息、非有限数值和未知字段均被拒绝。
所有原始提示词、回答、用量和耗时保存在响应的 `calls` 中，供调用方封存；不能用文字报告代替实际账户收益。

## 信息可用时间

证据必须同时包含发布时间、首次可用时间、失效时间、原文、原文 SHA-256 和公开来源地址。
`availability_basis=observed` 表示调用方有当时采集证据；`assumed_publication` 表示历史公开时间假设。
协议只能核验调用方声明的时序和哈希，不能独立证明一个网页在过去的内容或发布时间。
历史模型可能已经知道后续走势，因此即使输入时间过滤通过，也不能把回顾实验宣称为独立样本外。

Codex 在空目录、只读沙箱和临时会话中运行。禁用网页、shell、MCP 配置、插件、应用和子代理工具；
同时显式关闭 Codex 自身的记忆读取和记忆生成，避免历史会话影响回顾实验。
检测到工具活动会拒绝结果。来源文本始终当作数据。此接口不构成对模型训练知识的隔离保证。

## 验证

```sh
.venv/bin/python -m pip install pytest
.venv/bin/python -m pytest tests/test_evidence_research.py -q
```

测试覆盖时间边界、来源哈希、引用约束、原生多空节点、子进程配置和失败处理。
真实 Codex 调用应使用明确的公开样本单独验证，不在常规测试中消耗模型额度。

官方接口依据：[Codex 非交互调用](https://learn.chatgpt.com/docs/non-interactive-mode)、
[Codex 配置参考](https://learn.chatgpt.com/docs/config-file/config-reference)。

## 策略证据审查与新假说

`python -m tradingagents.research.strategy_review` 接收 `ta-factor-review-request.v1`，
复用原生多空研究员审查调用方提供的公开统计，再返回有限数量的待冻结假说。
每个假说包含因果可用输入、触发／失效规则、完整成本要求、检验设计与失败风险。
原始提示词、辩论、结构化结果、用量和源码身份都保存在响应中。
研究窗口以输入证据为准；模板角色的 BUY／HOLD／SELL 不作执行指令。
此接口的 `production_eligible` 始终为 false，稳定性和运行资格由独立账户与前向验证决定。

```json
{
  "schema_version": "ta-factor-review-request.v1",
  "backend": "codex",
  "model": null,
  "timeout_seconds": 240,
  "evidence": {
    "window": "调用方声明的真实研究区间",
    "funding_included": true,
    "independent_oos": false,
    "accounts": []
  }
}
```
