# 输入带单员的策略发现工具：市场调研与已实现入口

2026-10-24。目标是输入带单员名称／链接／ID后，自动获取可读证据、归纳规律、生成可解释候选，并用独立双向账户验证。**“对方是量化”不自动保证成交记录能唯一还原源码；产品应输出可验证的候选与证据阶段。**本次已实现可运行的Leader Lab入口，而非仅提出方案。

## 市面上能复用什么

星数通过GitHub仓库API在本次调研查询，不是盈利或复刻成功的指标。

| 方案 | 本次星数 | 官方说明所支持的用途 | 在此产品中的位置 |
| --- | ---: | --- | --- |
| [PySR](https://github.com/astroautomata/PySR) | 4,115 | 搜索可解释符号表达式 | 公式候选引擎；需要针对状态／动作任务设置目标与复杂度约束 |
| [imitation](https://github.com/HumanCompatibleAI/imitation) | 1,810 | BC、GAIL、AIRL等模仿／奖励学习 | 有足够完整轨迹时的动作模仿后端；仓库API最后推送为2025-01-07，接入需检查依赖兼容性 |
| [NautilusTrader](https://github.com/nautechsystems/nautilus_trader) | 30,678 | 确定性事件驱动交易引擎 | 后续完整执行与回放后端；[HEDGING OMS](https://nautilustrader.io/docs/latest/concepts/positions/)允许同一品种多个独立position ID |
| [QuantAnalyzer](https://strategyquant.com/quantanalyzer/) | 商业产品 | 分析现有交易结果、资金管理、Monte Carlo、What-If和组合比较 | 复盘分析参考，官方定位并非从交易流水恢复原策略 |
| [Strategy X-Ray手册](https://strategyx-ray.com/admin/static/downloads/StrategyXRay-UserManual-en.html) | 商业服务 | MT5连接、实时监控、EA／Magic维度统计与组合回放 | 监控分析参考；所查手册未提供可靠的一键恢复原算法证据 |

另查了QuantConnect/Lean（22,177星）与vectorbt（9,504星）仓库元数据，未在本轮集成。它们的星数不能代替带单员复刻的任务验证。无连字符的`strategyxray.com`无法通过本次网页工具访问，不把第三方帖子中的宣传当作已验证产品能力。

本次调查未核实到“只给任意带单员名字即可可靠恢复其原算法并复现收益”的成熟产品。这不排除市场上存在相关服务；调查所核实的成熟能力集中在交易分析、可解释模型、模仿学习和执行回放，适合组成统一入口。

## 已经实现的第一版

代码：[research_tools/leader_lab](../../research_tools/leader_lab/README.md)。独立启动器：[scripts/leader_lab.py](../../scripts/leader_lab.py)。不导入应用的`.env`启动，不需要交易私钥或Cookie，不发送订单；学习依赖放在独立研究环境，没有改变现有Freqtrade环境。

```sh
python3 /Users/chrise/GitHub/TradingAgents/scripts/leader_lab.py
```

终端会提示输入名称／链接／ID。已登记Damorne名称可使用保留缓存；想明确拉取当前可公开访问的历史，用：

```sh
python3 /Users/chrise/GitHub/TradingAgents/scripts/leader_lab.py profile \
  --leader 'https://www.binance.com/en/copy-trading/lead-details/5219911688676500225' \
  --refresh
```

其他昵称走公开查询，只接受唯一精确匹配；重名时要求链接或ID。也支持明确输入的币安JSON或规范CSV。当前自动在线适配是币安，其他平台尚未实现完整在线采集。

第一版已包含：

1. **获取与校验**：限固定公开端点、完整分页、重复／冲突ID拒绝、成功但不完整的响应拒绝、来源摘要与时间记录、失败保留；名称／URL／ID统一为组合身份。
2. **规律归纳**：同币反向仓位重叠、描述性仓位分档、同向重开、此前记录盈亏与价格关系；同时输出反例。给定12小时空仓划分只是明确研究假设，不冒充带单员真实生命周期参数。跨长空窗的新episode不混成同episode重开。
3. **规则学习**：完整可观察动作表可训练浅层决策树，导出可独立执行的JSON规则；按时间60/20/20切分，清除跨段episode与标签区间，仅按验证macroF1选择深度3／5，留出结果不参与选择。
4. **自主双向回放**：同币多空独立持有、一个共享钱包和保证金池、下一分钟开盘代理执行、明确交易费／滑点、提供的资金费、终端浮盈亏与退出费准备。自身仓位、持仓时长、已知收益和可用保证金由自身账本计算，不把带单员状态或动作作为交易信号。
5. **分别给结论**：逐步动作模仿分数与自主账户收益／完整净胜率／PF／观察回撤／日夏普代理分开；闭仓汇总不会被伪造为完整动作表或HOLD负样本。

当前实际学习后端是可解释浅树，实际回放后端是独立研究账本。PySR、imitation和Nautilus是已核实的扩展组件，**未声称已安装、集成或运行这三个引擎**。研究回放还没有盘口队列、真实冲击及强平模型，不能直接生成实盘资格。

## 已执行证据

- 22项针对性检查通过：双向独立持仓、共享保证金、费用／资金费、身份与分页、跨段清除、未来时钟拒绝、无观察HOLD拒绝、JSON规则导出一致，以及自主回放不读取专家动作／库存。
- 用真实Damorne URL从CZ-Agent工作目录启动成功，旧缓存归并175条闭仓汇总；入口能处理该目录默认Python3.9，并自动使用此工具独立环境。
- 实际公开接口请求成功，完整分页取得267条闭仓汇总；本轮新取数据与旧缓存分别保存，没有覆盖原始记录。它们仍不是完整订单或全部未成交动作。
- 合成已知量化规则的完整流程通过：留出动作macroF1为1.0，导出规则逐行与训练模型一致，独立双向账本现金核对通过。这是`SYNTHETIC_KNOWN_POLICY`工程测试，**不是Damorne策略恢复或真实收益证据**。
- 旧的启动缺依赖／Python版本预检失败保留；合成v002原Markdown未突出合成标签，已保留范围说明与执行源码快照。最终v003在规则、评分卡及报告中均明确数据类型。

最终目录为`results/research/leader_lab_20261024/`下的`damorne_public_profile_v003/`及`synthetic_recovery_v003/`。公开原始分页在`binance_public_probe_v001/`，旧缓存结果在`damorne_cache_v002/`。完整输入、源码和环境身份见`delivery_receipt.json`。

## Damorne目前的证据阶段

当前267条公开闭仓汇总只能支持行为归纳。首次选币、挂单与取消、大额风险准入、完整仓位规模决策、当时权益及真实保护单仍未被恢复。没有把267条闭仓位置当成267个独立机会，也没有训练一个伪造的“所有其他分钟都不交易”模型。

双向持仓已由研究账本支持，框架限制不会被当成策略不可研究的结论。接下来可以为完整观察轨迹接上更丰富的规则或模仿学习后端；对资料不足的带单员，工具保留候选规律、反例与明确缺口。原有0/79历史自主对齐失败及旧Freqtrade筛选结果没有改判。
