# 基差／OI来源面板草案

文档性质：本页保留首次实现前的来源设计与当时调查边界。后续面板、全原文审计、独立逐格复核及原生H02选择已完成，当前状态见[实现验证](basis_oi_validation_20261017.md)。下文草案措辞不作为最新运行状态；旧封存计划与结果不回写。

## 本轮先交付什么

建议先制作独立的**index／metrics来源面板**，保留全748人口及完整17日格点。当前不生成basis、premium、OI变化、标准化、排名、拥挤度或交易信号。下一步由TradingAgents原生研究节点基于干净的可用输入审查新机制，再登记候选和量化评分；本草案不先替其挑公式或窗口。

新来源区间是UTC `2026-09-28 00:00:00`至`2026-10-15 00:00:00`右开，即Sep28至Oct14全部17日。它与旧Aug／Sep的909人口研究区间不同，不能把旧标签、成功成员或收益结论搬进本轮输入。

旧价格资料本身不能作为同一区间的完整配对证明。九月档案可以成为补源，但尚需明确跨来源连接及冲突审计，因此本次先不输出价差列；相关边界在后文列明。

## 来源身份与已知状态

主来源为`results/research/profit_factor_search_20261015/uniform_inputs_recovery_v132`，已记录全25432个依赖，正好`748×17×2`，未因结果筛选。来源收据SHA为`659674a5d2b6e8be62ef1ba9594bdcf6dbfd54be87e87dedffd0062174110d20`；[protocol](../../results/research/profit_factor_search_20261015/uniform_inputs_recovery_v132/protocol.json) SHA为`fe78a80bfb00c3ca98b824d67b6f6e3801217732e453570222964af39836407c`，[result](../../results/research/profit_factor_search_20261015/uniform_inputs_recovery_v132/result.json) SHA为`6205bd888d01997d34b399981295f017abe1ca4389ae344e95ff4584b2743543`。

已完成的[输入就绪调查](../../results/research/basis_oi_input_readiness_20261017_v001/result.json)及其[收据](../../results/research/basis_oi_input_readiness_20261017_v001/receipts.json)绑定相同来源，收据SHA为`f806b9e4db19c63709e70fc30892b842af619b7eb6a2e4dba8c71fbed70ad7f9`。以下是该调查已有事实，本草案没有重新计算市场效果：

| 维度 | 已封存状态 | 面板含义 |
| --- | --- | --- |
| 全部依赖 | 25078个VERIFIED、349个PUBLIC_404、5个FAILED／UNKNOWN | 349和5都保留为对应kind／day的未知，不填零，不换成功子集 |
| index profile | 12449个C、12个G | C仅计数、端点、无重复的profile条件，不能代替逐字段原文检查；G保留已观察记录和具体缺口 |
| metrics profile | 10139个C、2478个G | 不能把2478个不齐profile整日删掉，也不能把C当作所有字段和经济时间均已证明 |
| 预定原文样本 | BTC、ETH、牛来、AAOI，三日、两类，共24依赖 | 牛来三日index404仍保留，未换样；样本不能代表全人口数值齐全 |
| metrics原序 | 已读12包各有288个不同五分钟点，但原行序均有逆序 | 不能按文件行号做lag／rolling；须按解析后的原时间稳定排序 |

人口唯一来源为该终态protocol的依赖身份集合，必须恰为748个符号、17个日期、两种kind的笛卡尔积；不得取718个成功符号或只留index／metrics交集。资产类型和晚期静态资格可另带元数据及其SHA，未知类型保持UNKNOWN；来源面板不按可交易资格删行，也不声称晚期快照是历史PTI证明。

实际读取应使用v132的`record.file`相对于v132根的本地档案，并核对记录SHA与封存收据。`frozen_inputs`中的旧CZ-Agent绝对路径仅保留历史来历，不作为可变的运行时回退路径。两个冻结manifest分别为[index](../../results/research/profit_factor_search_20261015/uniform_inputs_recovery_v132/frozen_inputs/indexPriceKlines_manifest.json) SHA `4c65f33b22e2eb862342e85fbe42e1e5760bcdb8c20b7847cd34ac3e8fb2a0c4`与[metrics](../../results/research/profit_factor_search_20261015/uniform_inputs_recovery_v132/frozen_inputs/metrics_manifest.json) SHA `ffee7ebda879a20df21900c769b36f5f057c341e5e226c04210bb9063cb4d0b2`；不能据其旧path悄悄换读取目标。

## 固定格点及边界

定义知识快照时钟`clock_ms`，均为整数UTC毫秒，范围`[Sep28 00:00, Oct15 00:00)`：

- 五分钟表：每个符号4896个时钟，共**3662208行**，包括所有未知值。
- 小时表：每个符号408个HH:00时钟，共**305184行**；直接投影五分钟表中相同HH:00行，不聚合出新的小时均值、OI求和或流量因子。
- 原记录表另保留所有源时刻及延迟后落在面板区间外的记录，不把末日记录提前挪回区间以补齐数值。

面板身份为`(symbol,clock_ms)`，资产类为独立元数据，不能用其UNKNOWN状态使符号消失。两张表固定人口和日历，字段缺失不改变成员。没有任何target、未来价格、成交路径或模型列参与此过程。

## index价格时钟

原CSV的`open_time`是毫秒bar起点，完成边界仅为`open_time+60000`。采集profile的index最小／最大时间已经按完成边界描述，不能把profile时钟再加一分钟；转换以原CSV的原始open_time为准。

在快照t，只取**恰好于t完成**的那条一分钟index bar，即原open_time为`t−60000`。建议列名保留`index_1m_open/high/low/close`，并保存`index_bar_open_ms`和`index_completed_ms`，避免把五分钟抽样的一分钟价格叫成五分钟OHLC。首个时钟Sep28 00:00所需的Sep27最后一分钟不在本次来源里，保留UNKNOWN；不从Sep28第一条bar向后回填。

官方Index Price Kline将第5、7、8、9、10、11位置标为Ignore。就绪样本中所谓volume／quote／taker栏为零占位，count为60也不是可用的真实成交笔数。因此index提供价格，**不提供成交量、流动性或主动买卖流**；占位栏只留原文审计，不进入数值面板。[官方UM Index Price Kline](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data#index-price-klinecandlestick-data)

index完成时间是本轮研究的可见时钟假设，仍不证明当年的首次公开或收到时刻。它也不等同于合约成交价、mark price或premiumIndex。

## metrics原字段与统一可见规则

仅保留已有六个原字段，不派生因子：

| 原字段 | 原文类别与限制 |
| --- | --- |
| `sum_open_interest` | OI存量原值；经济采样时钟及真实首次可见时刻仍未知 |
| `sum_open_interest_value` | OI价值原值；不当成成交金额、流动性或账户现金 |
| `count_toptrader_long_short_ratio` | 原账户比字段，不改造成持仓数量 |
| `sum_toptrader_long_short_ratio` | 原持仓比字段，不合并前一账户比 |
| `count_long_short_ratio` | 原全市场账户比，不以index占位字段补值 |
| `sum_taker_long_short_vol_ratio` | 原主动买卖量比，不能直接当T时点已经发生的独立量 |

保留原始`create_time`文本。已读样本没有时区标记，UTC是当前档案协议解释，不是原文自带证据；解析结果另存`raw_T_ms`、`timezone_assumption=UTC`和解析状态。每条记录必须落在本文件日内的五分钟原格点，否则作为越界／离格原记录保留，不能自动四舍五入到近邻时钟。

统一采用**`research_visible_at_ms=raw_T_ms+10min`**。快照t只连接恰好`raw_T=t−10min`的源记录，六字段同一规则，不对OI、账户比或流量分别搜索最佳lag。不做向前／向后填充；目标原钟缺失时，该快照对应字段仍UNKNOWN。最初00:00和00:05没有区间前记录，仍占行且未知；最后原钟23:50和23:55延迟后可能落到Oct15，原记录仍保存，但不提前显示在Oct14。

理由是既有v121前缀调查中，171符号、1807份metrics的流量对照更贴T后五分钟；436178个共同量值的随后解释相关约0.999969，之前解释约0.000790。这是既有诊断，不是本草案新增lag试验，也不证明全748一致、OI存量的经济时刻或首次公开时间。T+10min是额外留缓冲的保守研究假设，不能命名为“已证明因果可见”。

当前官方REST将OI统计的timestamp描述为区间结束，将taker买卖量timestamp描述为区间开始；这说明不同字段的时间含义需要区分，**不能据REST当前说明直接证明档案`create_time`对全部六列的历史映射**。原始时刻、研究可见假设、经济时间和首次发布时间分别记录。[官方UM市场数据：OI与Taker](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data)

原数值以Decimal文本保留，不先降到float丢精度；未来数值视图可另存有限float并记录转换。真实源零保留为观察到的零，不能将缺值、404或冲突填成零。无需按统计分位裁尾；量纲／数量乘数未获证据时显式标记，不擅自跨合约归一化。

## 排序、重复、缺口与逐字段状态

读取顺序不能改变语义。先保存档案SHA、CSV member、原行号、原始时钟文本和原值，再按解析时钟稳定排序；Unicode符号和ZIP名称编码保留原文证据，不能因解码困难换成别的符号。

同一`(symbol,kind,raw_timestamp)`发生重复时，不采用首条／末条优先、均值或任意去重：

- 所有重复记录的某字段均可解析且Decimal值相同，允许保留该字段数值，同时保存重复次数和所有来源行。
- 任一重复值冲突、空值或不可解析，受影响字段置UNKNOWN并留全部冲突原文；其他有一致证据的字段可保留，不能让一列冲突静默污染或删除整个人口。
- 符号身份、时钟解析或文件边界发生冲突时，整条原记录不能进入正常格点，仍在异常表保留。SHA、路径逃逸或封存集合变化属于完整性失败，不能降格成普通缺数据后继续宣称成功。

profile G不是整日无效。按原文把每个有效时钟的各字段放回完整日历，缺哪一钟／哪一列就标哪一格UNKNOWN；不压缩时间、前填或改变lag。349个404和5个失败分别保存原status及原因，不当成下架证明或数值零。

建议状态拆为三层：`source_status`说明VERIFIED／404／FAILED；每字段`value_status`说明KNOWN_RAW_VALUE／MISSING_CLOCK／EMPTY／INVALID／DUPLICATE_CONFLICT／NOT_APPLICABLE；`time_status`说明INDEX_COMPLETION_ASSUMPTION／METRICS_T_PLUS_10M_ASSUMPTION及OI经济时间、首次发布时间未知。一个笼统`known=true`不足以代表以上三种证据。

## 合约分钟价格的实际边界

本次只读核对了已有清单及源绑定，没有重新扫描全量价格原文或证明跨来源数值相等：

| 来源 | 实际能说明什么 | 尚缺什么 |
| --- | --- | --- |
| `data/research/damorne_15d_alignment_20261014/dynamic_market_panel/manifest.json` | 748身份、11968份raw；范围Sep28 **19:38**至Oct14 00:00右开，Sep28每符号仅262分钟暖机。manifest SHA `1ef30a09a5f801cdc0a6a10cde4632f3f59dc2a7b5df52774910f3b75e5c5cad` | 不含Sep28整日，也不含Oct14整日；其旧CZ-Agent路径须映射到TradingAgents中已证明同SHA的副本 |
| `settlement_unwind_20261016_v001/protocol.json`的原price输入 | 明确绑定前述11968份和`unseen_20261014_v002/raw`的748份Oct14补充，共12716；[H01收据](../../results/research/settlement_unwind_20261016_v001/receipts.json) SHA `0de16a4e4c741b9ddc239f7477a7f50c0bd6f4bda6b04bde941f339862d782d3` | 增补了Oct14，但仍没有Sep28全天；这里只读取输入身份，不加载H01标签或收益 |
| `data/research/historical_market_20261017_v002`的九月1m档案 | 原748均在909采集人口里，九月739个VERIFIED、9个NOT_LISTED，其中703个profile为整月43200行；源收据SHA `21624a01c5e1247dc4797fb020ccc4f94ba404fea654f89589784d5d16cfa880` | 可作为Sep28候选补源，36个VERIFIED并非整月满行；尚未对全部交叠时钟逐键／值／单位核定冲突，也不提供10月来源 |

因此不能称“已有一套全748、同17日、逐钟已验证的合约／index配对”。全748身份乘17日是输出人口要求，不要求每格都有数。下一阶段可以仅按已绑定旧REST原文加Oct14补充附接一个有限的合约价源：Sep28大部缺口明确UNKNOWN、原748和17日全部保留，**无需为了首日数值齐全强并Aug／Sep来源**。需要连续窗口的具体要求由后续原生假说提出并预先登记，不能据结果好坏裁掉Sep28。

如后续另行登记使用九月档案补源，须先声明各源首末SHA、完整日历、精确完成边界、重叠一致性与冲突UNKNOWN、合约／index价格单位及符号映射，并保留缺行。不得按后来价差或收益选择重叠源；这一补源不是当前source-only面板的前置要求。

在这一步完成前，本次index／metrics面板**不生成价差字段**，也不以旧小时close、其他合约或指数自身补合约价格。后续输入版本明确绑定价格来源后，必须使用同symbol、同一分钟完成边界、两侧均有独立有限价格和单位证据的原文；任一侧未知，派生字段未知，人口仍不变。具体basis定义、方向、窗口和假说留待原生研究审查及前瞻登记。这些本地研究步骤已在用户目标范围内，无需为常规来源绑定另行索要许可。

## 最小交付契约

建议新来源产物采用`ta-basis-oi-source-panel.v1`，保存：

- `protocol.json`：原748人口、完整17日、两种输出时钟、统一T+10min假设、来源收据及实际执行依赖SHA；声明无标签、无模型、无收益、无订单、无leader标签、无新假说、无lag选择，生产资格为false。
- `source_records`及异常／冲突表：原身份、时间、Decimal原值、源状态、档案和member SHA／原行号；不读取模型或旧样本的target。
- `panel_5m.parquet`与`panel_hourly.parquet`：固定全部身份、index一分钟价格点、六metrics原值、各字段的源／值／时钟状态和可追溯引用。小时表是同钟投影，不能重新“补齐”五分钟未知。
- `coverage.json`：按symbol、日期、kind、原钟、可见钟、字段和原因给出守恒计数，完整列出349个404、5个失败及profile缺口；不评分、不筛选所谓高质量赢家。
- `result.json`和`receipts.json`：允许“完成且保留未知”的终态；完整性失败在新目录封存，首末核验实际消费来源，不原位改成功。现阶段本草案没有创建这些产物。

最小实现只需档案记录读取、纯格点构造与状态连接，不做通用表达式解释器、特征框架或图表。逐symbol／day流式读取并写列式数据，避免同时展开25432包；原始档案已有冻结来源，不在builder里临时联网补失败。

有意义的验证应覆盖全身份行数、首尾及跨日T+10min、原逆序扰动后结果一致、相同／冲突duplicate、逐列空值和真实零、404／失败格点、index占位列拒入、小时投影等值，以及未来原记录扰动不改变较早可见面板。将全部输出时钟和状态计数与原生日期／字段期望格核对，不能仅核选中的四符号样本。

面板清理完成后才更新TradingAgents原生研究的可用输入契约，明确哪些是原始来源、哪些可见时钟是假设、哪些机制数据仍缺。此步提供新的审查材料，不提供盈利结论，也不把持续执行记账工程当作寻找新alpha的替代。

官方字段复核日期：2026-10-17。链接使用USD-M市场数据文档，旧单接口URL当前重定向整Market Data目录；文档的当前REST说明不替代本批历史档案时钟及完整性证据。
