# GitHub公开策略：先复现，再优化

当前主线已执行切换：暂停自研因子、模型、新策略及新的风险覆盖实验，优先对齐GitHub原版收益与账户口径。历史实验和失败完整保留。下一份策略须先取得原版合同并复现，才进入优化。

## 当前已完成什么

NFI v18.0.146的原版72笔交易、217笔成交保持逐字段一致。作者与本地同口径总收益均为81.8864%，已平仓最大回撤均为1.3912%，原生已平仓夏普均为36.7738。[逐字段口径对齐](nfi_author_metric_alignment.json)已从作者与本地原始JSON重新核对，未重跑金融账户。

全持仓5分钟参考回撤13.6107%、日净值夏普7.7421另列，不能用它们替换作者指标后判定“没复现”。这些都是2025年1月的已知历史结果。

公开来源已固定commit并归档：一份895行Freqtrade策略测量账表、另一份416份报告卡的策略集、GeneTrader的原始报告及对应策略文件。详细登记24份候选/基线资料，尚未完成对新候选的本地经济复刻，尚无共同条件冠军；覆盖数量不等于独立策略数量。

## 收益领先线索及取证状态

| 对象 | 已取得原件 | 当前结论 |
| --- | --- | --- |
| NOTankAi_15_Cleaned_v2 | 第三方报告、冻结账表、当前作者源码 | 披露数字在已取Apex账表中最高；原测量源码指纹、数据与策略输入版本未对齐，不能当成已验证收益领先者 |
| GeneTrader_gen10_1734895087_6007 | 作者文本报告与对应策略文件 | 报告本金300 USDT、期末995.233 USDT；不同周期与市场，未与NFI混排。原始逐笔导出、执行配置与精确引擎仍待取证 |
| ClucHAnix_5m_old / CombinedBinHClucAndMADV5 | 第三方报告、账表及来源路径 | 保留为复现线索；第三方筛选门槛不自动成为我们的策略准入标准 |
| CacheCarti策略集 | 406行索引、完整416文件树计数、12份报告卡 | 索引遗漏passed目录10项；这10份卡片的披露OOS均为负，“passed”标签不能替代净收益。缺具体起止时间与完整引擎合同 |
| NostalgiaForInfinityX8 | 作者原生账户、数据/引擎/配置、本地原生账户 | 原版报告已复现；只有一个已对齐基线时，不宣布跨策略最高 |

[候选登记与全部原始数字](candidate_registry.json)区分第三方测量、作者报告、本地已复现；所有未取得字段保留未知。这里的“披露最高”只是取证优先级，不是策略业绩排名，也不是GitHub全网最高。

## 为什么最高数字尚不能直接复刻

Apex报告记录的测量代码指纹为`590bf74986c5`，当前公开harness为`df39e7b6b43f`；已检查可见历史中9个harness版本，未找到对应指纹。公开setup脚本能解释1000 USDT、8槽、现货、unlimited配置，但不能冒充当次实际执行配置。原生订单导出、原数据hash和被测策略commit未取齐。

对NOTankAi当前作者源码的一个固定合成价格案例，保持决策前250根收盘价完全不变，只追加5根未来价格，`Minima Full Send`入场分支从false变为true，原因是全数组局部极值检测。这个[因果性反例](notank_current_source_counterexample.json)证明当前源码该分支依赖未来；没有运行完整策略、没有生成收益，也没有修改作者逻辑。它不能追溯证明第三方报告实际用了同一源码。报告称未声明启动期、当前文件声明200，同样是待解释冲突，不能据此直接断言版本或检查器哪一方有错。

这属于复刻前的来源和因果性检查，不是自研因子或修补策略。不会先“修好”源码再冒充复现原版。

## 后续按此顺序执行

1. 补齐披露收益领先候选的源码commit/许可证、实际引擎与依赖、数据身份、配置、原报告及完整成交账本。保留缺口，不能自行补参数后称原版。
2. 用原版本金、币池、周期、仓位、复利、成本、杠杆和退出规则复现。原始数字、已平仓指标、全持仓净值分别记录。先解释差异，不加本地0.8x、8%止损或15%停机来裁定原策略。
3. 对复现通过的候选冻结共同比较合同。不同周期、市场、资本分母和杠杆分层列示，扣费账户净收益是主要排序对象；同时报告胜率、回撤、夏普、资金占用、分段表现和容量边界。尚未完成共同条件，不宣布冠军。
4. 确定已复现且可比的收益领先策略后，才以其原版作为冻结基线登记优化，逐项证明增量。当前未启动新策略优化。

本轮只完成主线变更、口径核对与第一轮来源取证；“找到并复刻最高收益策略”的整体工作仍未完成。长期、冻结后未见、外部复核和真实执行容量均未建立，不生成实盘资格。

## 原始来源

- [Apex冻结账表](https://github.com/Apex-prim/strategy-audit/blob/c83edf2a7ba13733addfd39caed71dfed120fce0/LEDGER.csv)、[最高披露项报告](https://github.com/Apex-prim/strategy-audit/blob/c83edf2a7ba13733addfd39caed71dfed120fce0/corpus/NOTankAi_15_Cleaned_v2.md)。
- [NOTankAi当前冻结源码](https://github.com/TheoBrigitte/freqtrade/blob/c9def379b1036064d592c3a0518a07811f5a9a76/strategies/notankai/NOTankAi_15_Cleaned_v2.py#L331)。
- [GeneTrader作者报告](https://github.com/imsatoshi/GeneTrader/blob/20053ba673a58976d5f5ab05cb2ec09018cbfd96/daily_results/20241223/gen10/results.txt)。
- [CacheCarti报告卡目录](https://github.com/CacheCarti/Crypto-Strategies/tree/f8e32599c192b823dbe7aea77a2b9341c6114527/cards)。
- [Freqtrade官方收益定义](https://github.com/freqtrade/freqtrade/blob/develop/docs/backtesting.md)：期末权益相对初始本金的总收益，不等于每笔百分比之和；官方当前文档仅解释口径，不替代NFI固定2026.9源码。
