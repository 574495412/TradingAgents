# TradingAgents 接入验证记录

验证日期：2026-10-15。接入与可复核实验链路已完成，尚未证明找到可盈利策略。
本文只采用最终源码对应的 `codex_public_003` 与 `fixture_004`，不混用早期联调输出。

## 开发范围

- TradingAgents fork：Codex CLI 适配、公开证据协议、原生多空研究节点、结构化裁决与调用记录。
- CZ-Agent：公开归档转换、隔离解释器桥接、四组对照、统一连续账户、成本压力测试及封存复核。
- 新增代码注释和文档字符串均使用中文；不修改原项目已有英文注释。
- 已有 Codex 登录可直接使用；未生成 API 密钥。没有操作生产服务、交易账户或带单员标签。

TradingAgents 本地提交：`6aec61012a74623ef8f80aca530765281968a9e7`，分支 `codex/codex-evidence-research`。
CZ-Agent 改动保留在现有 `feature/trade-2.0` 分支，最终实验源码身份如下：

| 身份 | SHA-256 |
| --- | --- |
| CZ 研究代码 | `008d6d3ad048ed817f1addf412bf58878644f270d8b121cf37f19286819b3510` |
| TradingAgents Python 源码 | `5e4c30444e828f106f4c4514e63ba955ac77386f92845e15be9f509fb61f9150` |
| 实验数据文件 | `343f9adb36dd13726663ff208381d612b0979919e7b3a89a307806e6143ec037` |

## 测试与复核

相关测试合计 **175 项通过**，不是两个仓库的全量测试声明：

- TradingAgents：153 项，包含新增 15 项，以及结构化节点、提示词、持仓上下文、评分完整性和数据配置回归。
- CZ-Agent：22 项，包含时间因果、资金费归属、完整成本、共享容量、满仓空头滑点、输入不可变性、恢复和篡改拒绝。
- TradingAgents 新文件通过仓库配置的 Ruff 检查；CZ 新文件通过 `E9,F,I` 检查；两边格式与差异空白检查通过。
- 最终真实实验：重建全部 8 个账户，校验 108 个文件；恢复命令成功且无新增模型调用。
- 最终合成实验：重建全部 8 个账户，校验 110 个文件；每组覆盖 3 笔完整交易及资金费记账，仅作联调。

```sh
# 在 TradingAgents 目录运行相关回归测试。
../CZ-Agent/.venv-tradingagents-lab/bin/python -m pytest \
  tests/test_evidence_research.py tests/test_structured_agents.py \
  tests/test_structured_agent_prompts.py tests/test_portfolio_context.py \
  tests/test_rating_integrity.py tests/test_dataflows_config.py -q

# 在 CZ-Agent 目录运行账户与实验测试。
.venv-tradingagents-lab/bin/python -m pytest \
  test/test_tradingagents_lab.py -q --override-ini addopts=''
```

## 真实 Codex 小样本

数据为 BTCUSDT 的 2026-08-01 至 2026-08-03（UTC，结束时间不含）的 48 根小时行情，
只有 2 个决策快照。输入仅包含量价与已结算资金费，没有新闻档案和真实持仓拥挤度。
历史信息可用时间采用公开时间加 60 秒的假设；这不是当时实际采集延迟的证明。

模型继承本机配置 `gpt-6-astra`，运行时为 Python 3.11.14、Codex CLI 0.162.0-alpha.17.2、
TradingAgents 0.6.0、Pydantic 2.14.1、LangChain Core 1.6.10、LangGraph 1.2.15。
显式关闭模型会话的记忆读取、记忆生成和工具能力。每个快照分别运行一次单模型和一轮多空辩论，
总计 8 次真实模型调用。封存提示词、回答、来源引用、用量与耗时。

初始权益 1,000 USDT，单仓目标 25%，最大四仓；基础单边手续费 5bps、滑点 3bps，
压力滑点 10bps。本轮通过事前协议将统一决策延迟预算设为 600 秒，四组完全相同；
仓位、退出和因子阈值未按本轮收益调整。

| 组别 | 基础净收益 | 基础收盘最大回撤 | 完整交易 | 胜率 | 压力净收益 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 数值基线 | 0.0000% | 0.0000% | 0 | 无交易 | 0.0000% |
| 固定规则 | -0.0611% | 0.0705% | 1 | 0% | -0.0962% |
| 单模型 | 0.0000% | 0.0000% | 0 | 无交易 | 0.0000% |
| 原生多空研究 | 0.0000% | 0.0000% | 0 | 无交易 | 0.0000% |

固定规则基础账户最终权益为 999.388755 USDT，手续费约 0.250181 USDT，资金费为零。
模型组给出的事件因子均为零：单次已结算资金费无法证明异常拥挤或未来价格方向。
本轮没有交易的组不能被解释为具有更高胜率，也不能从一笔亏损判定固定规则长期无效。

整轮研究约 157.08 秒；单模型每个快照约 15.62～15.81 秒，三次调用的多空研究约 61.60～64.04 秒。
CLI 报告累计输入 92,139 tokens，其中缓存输入 8,320；输出 2,382 tokens。
缓存输入不与总输入相加。没有将订阅额度虚构成美元费用，推理金额成本仍为空。

## 结果位置与复核命令

本地结果保存在仓库忽略的研究产物目录，不随源码提交：

- `artifacts/tradingagents_lab_20261015/codex_public_003/result.json`：最终真实模型比较。
- 同目录 `freeze.json`、`sources/`、`responses/` 和八个账户文件：冻结身份、源码副本与完整证据。
- `artifacts/tradingagents_lab_20261015/fixture_004/result.json`：最终合成联调。

在 CZ-Agent 目录执行：

```sh
.venv-tradingagents-lab/bin/python -m cz.research.tradingagents_lab verify \
  --run artifacts/tradingagents_lab_20261015/codex_public_003

.venv-tradingagents-lab/bin/python -m cz.research.tradingagents_lab run \
  --dataset artifacts/tradingagents_lab_20261015/btc_public.json \
  --output artifacts/tradingagents_lab_20261015/codex_public_003 \
  --tradingagents-root ../TradingAgents \
  --tradingagents-python .venv-tradingagents-lab/bin/python \
  --backend codex \
  --protocol artifacts/tradingagents_lab_20261015/codex_pilot_protocol.json --resume
```

修改源码或运行库后不能把新版本冒充成原实验恢复；原实验的源码副本仍保留在 `sources/`。

## 策略结论边界

当前证据只支持接入、真实推理和账户复核完成。两天、两个快照不足以判断策略增益；
历史模型知识、信息可用时间假设、缺少事件档案、连续数量与固定滑点均限制收益结论。
所有结果保持 `independent_oos=false`、`production_eligible=false`。

下一阶段需要先固定候选规则，再采集带真实可用时间的前向事件、持仓和行情证据，
继续用同一四组账户比较净收益、胜率、回撤及推理成本。当前版本没有自动参数搜索或实盘上线功能。
