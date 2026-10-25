# GitHub、X 合约策略与因子复核

本轮已完成公开代码与作者原文核对、12组有风险边界的策略组合回放，以及跨所合约资金费成本筛选。当前没有满足大小仓分别扣费盈利、足够交易数和风险门槛的候选，继续保留研究状态。公开项目的实现成熟度与本账户盈利能力分别记录。

## 来源与采用范围

2026-10-24通过GitHub公共API读取仓库星数、当前提交和许可证，随后按提交摘要下载相关源文件，避免搜索缓存和浮动master版本混淆。元数据见`repository_snapshots.json`，下载文件摘要及404记录见`source_manifest.json`。本轮没有执行下载的程序、安装其依赖或接入其下单模块。

| 项目 | API星数快照 | 本轮用途 |
| --- | ---: | --- |
| [Hummingbot](https://github.com/hummingbot/hummingbot) | 20,662 | 阅读Bollinger、MACD+Bollinger、Supertrend控制器及跨所资金费脚本 |
| [Jesse](https://github.com/jesse-ai/jesse) | 8,694 | 交易框架备选，仅核对仓库元数据，未宣称已跑其原生策略 |
| [Freqtrade官方策略库](https://github.com/freqtrade/freqtrade-strategies) | 5,589 | 阅读合约ADX/SMA与三重Supertrend规则 |
| [NostalgiaForInfinity](https://github.com/iterativv/NostalgiaForInfinity) | 3,494 | 仓库版本与活跃性复核，本轮未重跑既有NFI实验 |
| [Passivbot](https://github.com/enarjord/passivbot) | 2,142 | 阅读EMA价格带、入场、加减仓及敞口机制；仅采用有边界的EMA偏离恢复假设 |
| [alpha-rptr](https://github.com/TheFourGreatErrors/alpha-rptr) | 704 | 读取README与策略目录；猜测的Turtle／Surfer路径实际404，不当成已实现参考 |

Passivbot参考提交为`772abadc10a5bbf258a84aa8209a9084699fa1bd`；Hummingbot为`9af100d6822da7d2d0291a906c730ef172284ee2`；Freqtrade策略库为`f3340ce11f5bdf62f598522e64d1f5638eaa13f5`。星数是访问时快照，不用于替代经济验证。

X文本抓取最初返回403，随后通过应用内浏览器实际读取以下公开原文，未登录、发帖或联系作者：

- [Fede Cardoso，2024-04-24资金费套利讨论](https://x.com/cardosofede/status/1783149025513836562)：介绍Binance与Hyperliquid之间的永续合约资金费差，链接Hummingbot实现及说明。官方[发布文章](https://hummingbot.substack.com/p/hummingbot-newsletter-april-2024)也指向该讨论。
- [HangukQuant，2024-07-02资金费策略记录](https://x.com/HangukQuant/status/1808020040027197444)：作者描述双交易所、多标的自动资金费套利，但图表幅度隐藏、记录仅18天。没有把其估算年化或图形当作本项目已核实收益。

这两条线索支持研究跨所永续—永续组合，而不是把现货收益或单所资金费替代为合约套利。公开原文观察范围单独保存为`x_primary_observations.json`。

## 12个冻结组合

沿用此前按历史成交额登记的20币主池，以及43天成交、mark、资金费缓存。小仓4个规则族、大仓3个规则族，事前固定笛卡尔积12组合，没有按本轮收益挑币或搜索参数：

| 仓位档 | 规则族及固定参数 |
| --- | --- |
| 小仓 | Hummingbot布林3分钟100期、2倍标准差，增加已闭合小时ADX≤25；MACD21/42/9确认的同布林条件；EMA60/120/240分钟偏离0.3%后恢复；Supertrend3分钟20期、4倍ATR、距离小于1% |
| 大仓 | Supertrend15分钟20期、4倍ATR；官方ADX/SMA1小时14/12/48、ADX>30并交叉；官方三重Supertrend1小时已发布buy/sell参数字典 |

这些是**规则适配实验，不是原项目完整策略的原生收益复刻**。共用研究钱包、费用和保护退出替换原控制器执行器；EMA恢复条件及ADX区间过滤是明确提出的假设。原Passivbot逆势递增加仓未搬入本轮首仓筛选。Hummingbot代码期待的pandas_ta列名与本地版本不同，按指标字段含义提取列，指标内核和版本记录在协议中，未声称上游完整运行时一致。

指标实际使用现有Freqtrade环境的pandas_ta 0.3.16、TA-Lib 0.8.2和technical包。高周期K线严格等到完整闭合才映射到分钟决策；当前未闭合小时不进入趋势或区间判断。没有引入带单员动作、私有状态或未来标签。

经济假设：共享本金10,000 USDT，小／大仓初始保证金50／500，2倍杠杆，每边5bps手续费加3bps执行成本准备，实际资金费另计。小仓目标0.9%、保护距离0.7%、最长120分钟；大仓目标2.5%、保护距离1.2%、最长720分钟。没有加仓和分批减仓；资金费结算后收紧保护价的修复开启。亏后24小时冷却、完整扣费盈利后2分钟允许重开等条件保持固定。

验证日期为10月7—14日。12组合净收益范围−2.254869%至−0.467948%，全部未通过两档净贡献与已闭仓贡献分别为正、交易数、利润因子及回撤门槛。因此按预设规则仅保留验证排序第一名作为诊断，不能因其亏得少而晋级。

## 固定诊断账户的实际结果

验证排序选出的组合为`pb-ema-recovery__ft-adxsma1h`，随后固定规则回顾评估10月14—24日：

| 指标 | 小仓档 | 大仓档 |
| --- | ---: | ---: |
| 完整平仓数 | 297 | 5 |
| 扣费净贡献 | −41.2388 USDT | −21.9192 USDT |
| 完整交易净胜率 | 42.76% | 40.00% |
| 平均盈利／亏损 | +0.6446／−0.7241 USDT | +9.4648／−13.6162 USDT |
| 最差单笔净亏损 | −1.0546 USDT | −13.6492 USDT |

全账户302个完整仓位，终端0持仓、0待成交报价；净收益−0.631579%、净胜率42.7152%、利润因子0.61476、分钟观察mark开盘回撤0.927904%。手续费与执行成本准备共54.5541 USDT，资金费净额−0.03123 USDT。两档净贡献均为负，大仓5笔还不足评估门槛15笔。没有用终端浮盈抵消已实现亏损。

再增加每边2／5bps执行成本，净收益分别−0.767964%／−0.972542%，两档仍亏。已实际调用Freqtrade原生权益指标函数：Sharpe−10.19191、Sortino−13.85715、Calmar−158.73548；只有10个日收益，属于短窗诊断。指标源码SHA256和完整权益输入在`screen_v001`中保留。独立研究账本不等于原生Freqtrade策略回测。

两次小仓亏损超过初始保护预算，跳空／穿价损失保留，没有伪装成严格止损上限。当前目录与选币截面仍有回顾性限制，盘口队列、真实冲击、强平、跨所余额管理及大资金容量均未验证。本轮历史日期已查看，不能称为新前向或独立样本外收益。

## 跨所资金费研究

从Hyperliquid官方公开接口取得2026-09-25至10-24的小时成交K线及资金费，从已验证的Binance源读取对应分钟行情、资金费，并用公共连接器补齐USDCUSDT小时价格。这里的现货报价只用于结算币换算背景，不是现货策略。

初始BTC／ETH／SOL各有697根含终点的小时K线和696笔小时资金费。资金费真实时间位于整点后0—139毫秒；最初精确整点检查拒绝这些记录，修正后仅用小时归属核对覆盖，保留原始时间用于可用性判断。记录缺失不补零，不把本小时结算值提前放进本小时信号。

初筛事前固定72／168小时两种持有期限，使用截止决策前5分钟的完整24小时已知资金费差，假设差值持续，比较两腿开平四次成交的32bps成本及16bps额外缓冲，另限制已知跨所价格基差与USDCUSDT偏离均不超过1%。所有bps均按单腿名义本金归一化。

三大币每种期限671个小时背景，均未通过成本门槛。以已知24小时费差外推7天，预估毛资金费差最高约25.4707bps，低于32bps成本预算；这是预估成本筛选，**不是未来已实现收益、交易笔数或完整资金账户回测**。

扩展研究使用原20币名单与Hyperliquid公开永续目录的完整交集16币，登记文件为`carry_expansion_universe.json`，不按观察到的资金费或收益筛币。扩展采集与成本结果保留在独立目录，失败版本不覆写；最终覆盖和筛选结果另见本报告末尾的扩展记录。

## 复跑与检查

策略入口是`external_contract_screen`，成本入口是`cross_venue_carry_screen`；统一启动器已加入`external-contracts`和`carry-cost`。每次输出必须是新目录。

```sh
cd /Users/chrise/GitHub/TradingAgents
/Users/chrise/GitHub/freqtrade/.venv/bin/python -m research_tools.leader_lab.external_contract_screen prepare \
  results/research/external_contracts_20261024_v001/screen_review_v001 \
  --parent results/research/damorne_entry_scale_20261024_v001 \
  --root results/research/external_contracts_20261024_v001

python3 scripts/damorne_pipeline.py external-contracts study \
  results/research/external_contracts_20261024_v001/screen_review_v001

python3 scripts/damorne_pipeline.py carry-cost \
  results/research/external_contracts_20261024_v001 \
  --parent results/research/damorne_entry_scale_20261024_v001 \
  --output results/research/external_contracts_20261024_v001/carry_cost_review_v001
```

检查结果：独立研究环境57项通过、1项因缺少pandas_ta跳过；指标环境单独运行相关6项全部通过，涵盖被跳过的指标因果性检查，共58个不同检查在所需环境通过。曾在Freqtrade环境误跑完整学习测试，11项因该环境未安装sklearn而失败；按环境分工重跑后通过，没有修改无关依赖来掩盖问题。Ruff通过。固定诊断账户中途JSON恢复后，完整账本和指标与连续运行精确一致。

来源、冻结协议、原始数据、完整结果和限制汇总见`results/research/external_contracts_20261024_v001/delivery_receipt.json`。没有开启真实订单、部署服务器或把未合格规则接入生产。

## 扩展后的完整成本与双腿账户记录

最终16币全部补齐，数据在`carry_expansion_v003`：BTC、ETH、SOL、ZEC、LINK、XRP、NEAR、XLM、HBAR、DOGE、HYPE、PUMP、ONDO、ICP、SUI、WLD；合计11,152根含终点小时K线、11,136笔小时资金费，另保留697根USDCUSDT小时换算报价。采集进度独占写入错误和一次ZEC握手超时的目录保留，后续只重取缺失来源。

扩展时发现HYPE、PUMP、ONDO等Binance合约按4小时结算，不能沿用三大币的每24小时3笔假设。已用决策前72小时的实际结算间隔判断1／2／4／8小时周期，再要求过去24小时笔数与连续性一致。最初错误排除及空汇总失败保留在`carry_expanded_cost_v001`，修复后使用`carry_expanded_cost_v002`，没有把未知资金费补零。

16币、两种期限共21,472个小时背景，118个通过预估成本门槛：ICP的7天36个、PUMP的7天4个、WLD的3天11个／7天59个、XLM的7天8个。它们是相关的重复小时背景，**不等于118笔交易或独立盈利样本**。

随后冻结72／168小时两种双腿账户规则，独立钱包预存USDT与USDC各占初始价值50%，每条套利用相同基础币数量、一边多一边空；初始每腿名义本金约为账户10%，2倍杠杆、最多3组。根据信号在下个小时开盘后1秒代理成交，先处理真实时间在此前的资金费，再执行新仓，避免新仓领取入场前的结算。每腿每次8bps成本纳入，按每所余额分别检查保证金预算；观察到单组损失达到本金0.2%时排队下一小时退出。这些均是回放假设，不是已验证的真实同步成交或强平模型。

验证10-07至10-14：72小时版零交易；168小时版零完整平仓、终端有1组持仓，含浮亏净收益−0.862628%。按预设验证排序固定72小时作为诊断版，在10-14至10-24仍零交易。该窗口账户账面增加1.8986 USDT全部来自原有USDC现金汇率变化，**交易净利为0**。1,000和100,000本金冷启动对照也零交易，其汇率重估同样不算策略盈利。

为检查已观察持仓的完整生命周期，还分别从最早可用信号2026-09-26 01:00 UTC连续运行至10-24，两个期限各自保持一个完整账户，未拼接验证／评估冷启动：

| 连续账户 | 完整双腿套利组 | 完整套利净胜率 | 扣费交易净损益，排除现金汇率重估 | 终端持仓 |
| --- | ---: | ---: | ---: | ---: |
| 72小时 | 1 | 100%，仅1笔 | +0.064873 USDT | 0 |
| 168小时 | 4 | 25% | −3.069728 USDT | 0 |

这些结果不能支持高胜率或稳定盈利结论。双腿资金流水分别以USDT／USDC对账，闭仓交易、终端未平仓、原有现金和已实现USDC盈余的后续汇率变化分开核对；没有把货币升值包装成交易优势。两个连续账户均在实际非空持仓处进行JSON断点恢复，完整账本与得分精确一致。

Hyperliquid历史mark仍缺失，其资金费名义金额与未实现损益采用对应小时成交开盘价代理；Binance使用实际资金费记录mark及分钟mark，结算币换算使用实际USDCUSDT观察价。缺少真实双腿执行、历史保证金／强平与足够独立完整交易证据，两个版本均为诊断研究，`production_eligible=false`。它们也没有与亏损的小仓日内策略合并为一个盈利组合。

复跑双腿与连续账户：

```sh
python3 /Users/chrise/GitHub/TradingAgents/scripts/damorne_pipeline.py carry-account \
  /Users/chrise/GitHub/TradingAgents/results/research/external_contracts_20261024_v001 \
  --parent /Users/chrise/GitHub/TradingAgents/results/research/damorne_entry_scale_20261024_v001 \
  --output /Users/chrise/GitHub/TradingAgents/results/research/external_contracts_20261024_v001/carry_account_review_v001
```
