# TradingAgents × CZ 独立研究实验

本模块在 CZ-Agent 内提供公开数据转换、隔离进程桥接、四组对照和连续账户评价。
TradingAgents fork 提供 Codex 推理与原生多空研究角色。两边使用版本化 JSON 交换数据，
CZ 主环境不需要导入 LangChain、LangGraph 或交易运行时。

本机接入、测试与真实 Codex 小样本结果见 [验证记录](VALIDATION.md)。

## 安装

在 CZ-Agent 根目录执行，假定 TradingAgents fork 位于相邻目录：

```sh
python3.11 -m venv .venv-tradingagents-lab
.venv-tradingagents-lab/bin/python -m pip install -e ../TradingAgents pytest
codex login status
```

现有 Codex 登录供推理使用。默认只继承其模型名，模型与两边源码、关键运行库版本一起冻结。
不把本地交易配置、凭证或研究源码发送给模型。安装环境已加入忽略规则。

## 快速自检

```sh
mkdir -p artifacts/tradingagents_lab_demo
.venv-tradingagents-lab/bin/python -m cz.research.tradingagents_lab demo-data \
  --output artifacts/tradingagents_lab_demo/input.json
.venv-tradingagents-lab/bin/python -m cz.research.tradingagents_lab run \
  --dataset artifacts/tradingagents_lab_demo/input.json \
  --output artifacts/tradingagents_lab_demo/fixture_001 \
  --tradingagents-root ../TradingAgents \
  --tradingagents-python .venv-tradingagents-lab/bin/python --backend fixture
```

`fixture` 是确定性替身，合成数据和替身结果都不是策略收益证据。
输出目录必须不存在。模型失败时不会把结果改成 HOLD，也不会只比较幸存样本。
成功响应逐条封存；中断后在原命令追加 `--resume`，同一源码、数据和协议下可复用已有响应。

## 使用 CZ 已有公开归档

```sh
.venv-tradingagents-lab/bin/python -m cz.research.tradingagents_lab export \
  --archive artifacts/dual_alpha_long_horizon_20261015/public_v1 \
  --symbols BTCUSDT --start 2026-08-01 --end 2026-08-03 --every-hours 24 \
  --output artifacts/tradingagents_lab_demo/btc_public.json
.venv-tradingagents-lab/bin/python -m cz.research.tradingagents_lab run \
  --dataset artifacts/tradingagents_lab_demo/btc_public.json \
  --output artifacts/tradingagents_lab_demo/codex_001 \
  --tradingagents-root ../TradingAgents \
  --tradingagents-python .venv-tradingagents-lab/bin/python --backend codex
```

导出器验证原始行情 ZIP 与资金费 JSON 的封存哈希，要求完整的 24 小时预热和连续小时行情。
每条历史行情或资金费采用公开时刻加 60 秒的可用时间假设，并明确标为 `assumed_publication`。
这不是当时真实采集延迟的证明，也未独立证明资金费记录完整。
默认输入只有量价与已结算资金费，没有新闻档案、真实持仓拥挤度或带单员标签。

可以通过 `export --evidence events.json` 增加真实公开事件。文件是 JSON 数组，每项为
TradingAgents `Evidence` 协议加 `symbol` 字段，必须提供来源地址、原文哈希、
`published_at_ms`、`available_at_ms`、`expires_at_ms`、`availability_basis` 等字段。
导出器只向每个决策快照加入当时可用且未过期的证据；不得把今天看到的内容伪装成历史采集。

## 固定的四组比较

| 组别 | 决策依据 |
| --- | --- |
| baseline | 动量、主动成交流、拥挤度三个输入的等权均值 |
| fixed_rule | 同一基线加固定词表或资金费规则 |
| single | 同一基线加单次 Codex 证据因子 |
| multi | 同一基线加 TradingAgents 多空辩论后的证据因子 |

首版事件因子权重固定为 0.5，绝对分数至少 0.1 才开仓。四组共用相同数据、
有效期、300 秒决策延迟预算、仓位与退出规则；实际模型耗时超过预算就拒绝整轮评价。
这些是公开、固定的新研究基线，不代表完整 Omofun、DAmorne 或两位带单员的策略表现。
历史导出器的拥挤度缺失时显式填中性零，并写入缺失字段；不能宣称覆盖了该因子的真实信息。

账户初始权益 1,000 USDT，单仓目标名义金额 25%，最多四仓，总开仓名义敞口不超过一倍权益；
基础单边手续费 5bps、滑点 3bps，压力滑点 10bps。最多持有四根 K 线；
10% 收盘回撤触发下一可成交开盘退出。每个信号只开一次，不能凭旧信号无限续开。
决策完成后才可在后续开盘成交；边界资金费先由旧持仓结算，再处理当时开平仓。
期末按事前指定的终点平仓并计退出成本，逐根收盘计算含浮亏权益。
连续数量、固定滑点及 10 USDT 研究最小金额都尚未验证真实交易所历史手数与成交限制。

需要调整研究协议时，通过 `--protocol protocol.json` 提供 `account.Protocol` 的字段。
协议在模型推理前冻结；改变协议、数据、源码、运行库或模型后须开始新实验。
不同候选不能共享现金，也不能拼接各自盈利交易。首版不包含自动参数搜索或自动上线。

## 输出与复核

- `freeze.json`：事前冻结的输入身份、源码身份、模型、运行库与实验协议。
- `sources/`：本轮实际使用的 CZ 研究代码和 TradingAgents Python 源码副本。
- `responses/`：请求、完整回答、原生辩论、每次调用提示词、令牌用量与耗时。
- `decisions.json`：四组实际使用的结构化决策。
- `*_base.json`、`*_stress.json`：八个连续账户的逐笔、资金费、权益与指标。
- `result.json`：四组经济对照、来源哈希、推理成本信息及结论边界。

```sh
.venv-tradingagents-lab/bin/python -m cz.research.tradingagents_lab verify \
  --run artifacts/tradingagents_lab_demo/codex_001
.venv-tradingagents-lab/bin/python -m pytest test/test_tradingagents_lab.py -q --override-ini addopts=''
```

复核不调用模型：检查文件哈希、原输入和输出的绑定，再重建信号与八个账户。
每个账户还逐根验证已实现、未实现、手续费和资金费的经济恒等式。
模型自报信心不计入胜率；没有交易时胜率为空。推理令牌和耗时单列，订阅额度不虚构成美元成本。

所有当前结果标记 `independent_oos=false`、`production_eligible=false`。
历史模型知识、信息发布时间假设、少量联调样本及收盘回撤口径都限制结论。
后续应冻结候选，再采集真实前向证据评价净收益、胜率、回撤和推理成本；本模块不会操作实盘服务。
