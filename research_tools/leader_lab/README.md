# Leader Lab：输入带单员，寻找可验证的行为与规则

独立入口不导入TradingAgents应用启动，也不加载其`.env`。当前可运行入口有两级：

1. `profile`：输入币安带单URL／ID，或本地登记名称，自动获取可公开读取的已平仓历史，或读取保留缓存／CSV；归纳仓位分档、双向重叠、同向重开与反例。公开接口不可用、分页不完整或身份冲突均明确失败，不转成零交易。
2. `learn`：对完整可观察的市场—动作数据学习浅层可解释树，按时间与episode清除跨段样本，验证段选择后锁定候选，留出段分别做动作模仿评分与自主双向共享资金回放。导出纯JSON规则，可脱离scikit-learn执行。闭仓汇总不自动补成HOLD标签。

直接运行而不带参数时，交互终端会提示输入带单员。仅`profile`不需要额外第三方库：

```sh
python3 -m research_tools.leader_lab
python3 -m research_tools.leader_lab profile --leader 'DAmorné'
python3 -m research_tools.leader_lab profile --leader 'https://www.binance.com/en/copy-trading/lead-details/5219911688676500225'
python3 -m research_tools.leader_lab profile --leader 5219911688676500225 --source exported_positions.csv
```

也可以直接用`python3 scripts/leader_lab.py profile --leader 'DAmorné'`，不必安装整个TradingAgents应用。已登记的Damorne名称与ID指向同一保留缓存，报告明确这是历史数据，不伪装成实时采集。其他名称通过公开昵称搜索，仅接受唯一精确匹配；重名时要求链接／ID。未知ID只调用固定币安公共历史端点；`--refresh`显式跳过本地缓存。无需私钥、登录Cookie或下单权限。其他平台目前可导入规范CSV，不冒称已实现所有交易所采集。

CSV仓位列为`position_id,symbol,direction,opened_ms,closed_ms,entry_price,exit_price,quantity,max_quantity,leverage,reported_pnl`，时间戳为UTC毫秒。也支持币安原始`data.list`响应、位置列表以及本项目的`closed_positions[].raw`恢复格式。公开原始PNL保留原字段含义，不能从闭仓和推算账户收益／夏普／回撤。

## 动作学习的数据契约

安装本目录`requirements.txt`到独立研究环境。一个观察表必须包含：

| 列 | 含义 |
| --- | --- |
| `leader_id` | 每行相同、与输入带单员ID一致 |
| `timestamp` | 决策时钟，带时区ISO时间 |
| `available_at` | 本行全部特征最晚可用时刻，不晚于决策 |
| `action_end` | 动作标签观察区间终点；用于切分时清除跨段标签 |
| `observed` | true/false：确实观察到该动作；未知不变成HOLD |
| `campaign_id` | 非HOLD动作的episode身份，用于清除跨段同episode |
| `symbol` | 交易品种身份，不进入模型特征 |
| `action` | HOLD、OPEN_LONG、OPEN_SHORT、CLOSE_LONG、CLOSE_SHORT之一 |
| `open,mark` | 本分钟执行开盘代理与mark观察值，正数 |
| `funding_rate` | 本分钟资金费事件；非结算分钟显式0，不能用0填未知 |
| `x_*` | 决策时已知的有限数值特征；禁止明显未来结果、身份或标签字段 |

第一版要求完整的1分钟品种／时钟网格，不插值。时间顺序60/20/20，只有明确观察到的动作参与模仿；训练和验证清除跨边界标签及episode。模型搜索仅深度3／5、最小叶10、随机种子0，按验证macroF1选择，留出结果不参与选择。

允许的自身状态特征为`x_long_open,x_short_open,x_long_net_roi,x_short_net_roi,x_margin_free_fraction,x_long_age_minutes,x_short_age_minutes`。模仿评分可使用记录中的专家状态，必须标为逐步模仿；自主回放会移除这些专家状态并由自身账本重新计算，决不读取专家动作、episode或未来平仓。

```sh
python3 -m research_tools.leader_lab learn \
  --leader 5219911688676500225 --panel observed_decisions.csv \
  --capital 1000 --stake 20 --leverage 2 --fee-bps 5 --slippage-bps 3
```

经济输入是显式研究假设，不是从带单员还原的资金参数。回放允许同币多空独立持有，使用同一钱包及保证金池；动作在下一个完整分钟开盘代理成交。资金费先对已持有仓位结算，再执行当分钟待成交动作。净值包含费用、提供的资金费、终端浮亏和退出费准备，不强制平掉终端持仓。提供资金费的完整性及原始特征生成因果性还需各自数据源审计。没有强平、盘口排队或真实滑点证据，所有结果均不直接晋级实盘。

## 输出与复现

每次输出独立新目录：源文件摘要、数据质量、行为或学习报告、JSON候选、候选规则文本、逐步模仿混淆矩阵、独立双向完整仓位、分钟权益与自主决策。保留失败状态，拒绝覆写。报告明确阶段、缺失资料、规则反例和是否只做描述性研究，不输出无依据的“量化概率”或“复刻成功率”。

该工具不加载任何交易私钥，不发送订单。`production_eligible=false`固定保留；动作模仿、规则相似与独立收益是不同证据。

## 成熟组件接入位置

- PySR：后续将可解释树扩展为受复杂度约束的公式候选；当前未声称已集成Julia搜索。
- HumanCompatibleAI/imitation：动作及状态足够时接BC／AIRL；当前浅树基线用于先检验可识别性，未安装或运行其RL算法。
- NautilusTrader：生产级事件回放与HEDGING OMS适配候选。当前实现是独立、可核对的研究账本，不冒称Nautilus运行结果或已接入交易所。

最新市场调研与星数快照见仓库`docs/research/leader_lab_market_review_20261024.md`。

## Damorne完整研究记录

本轮已补到545条公开已执行订单汇总、43天成交／mark分钟数据与资金费，并完成订单链核对、8候选冻结回放及另一个预挂止盈假设。两批目前均未取得实盘资格；高胜率不等于正净收益。运行入口为`python3 /Users/chrise/GitHub/TradingAgents/scripts/damorne_pipeline.py --help`。

具体规则、实际结果、失败记录、独立回测及单步当前行情paper命令见`docs/research/damorne_pipeline_20261024.md`。当前paper只供诊断，无真实下单路径；补跑历史不计作实时前向表现。

针对性检查：`python -m pytest --confcutdir=tests/leader_lab -q tests/leader_lab`。此独立测试集不加载应用级密钥环境，也不代表全仓库测试。

后续首仓／盈利重开分离、20币独立币池、19／32因子净收益筛选、8组加减仓风险对照和资金费预算修复，见`docs/research/damorne_entry_scale_20261024.md`。统一入口新增`mechanisms`、`payoff`、`risk-ablation`、`derivatives`、`funding-budget`，均需新的输出目录。本轮49项检查通过；351笔固定诊断账户的大小仓贡献仍均为负，32因子模型也未达到入场门槛，无实盘准入。

最新GitHub／X参考核对、12组合筛选、16币跨所成本及双原生币钱包回放见`docs/research/external_contract_strategies_20261024.md`。入口增加`external-contracts`、`carry-capture`、`carry-cost`、`carry-account`。58个不同检查已在相应研究／指标环境通过。日内大小仓仍亏；跨所72小时连续账户仅1笔微利，168小时4笔合计亏损，原有现金汇率重估单列，不计策略利润。所有版本仍未取得实盘资格。
