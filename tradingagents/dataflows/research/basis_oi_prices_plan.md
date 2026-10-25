# 原748同钟合约价格来源附接计划

本计划登记纯来源整理，不计算basis、变化率、因子、标签、模型、交易或收益；首次真实构建由根任务复核后启动。保持原748身份和完整17日分钟格点，不按价格完整、资格或旧研究结果缩小人口。

唯一入口是`results/research/settlement_unwind_20261016_v001/protocol.json`，SHA固定为`70a9bd0ccd476d0b72f5c57e7cf1ed9c576b93ab541a3e4130b7f6bb0718e5f8`。只消费其中universe和input_identities，不读取其标签、result、预测、账户或收益。市场依赖必须正好为12716份：原748每个符号的16份`data/research/damorne_15d_alignment_20261014/dynamic_market_panel/raw/{symbol}/{day}.json`与一份`results/research/profit_factor_search_20261015/unseen_20261014_v002/raw/{symbol}/2026-10-14.json`。路径必须是当前TradingAgents所有权树中的规范文件，无CZ回退或symlink。其他H01依赖不消费。

固定快照是UTC `[2026-09-28 00:00,2026-10-15 00:00)`内的24480个一分钟完成边界，每个符号均占行，共18311040行。原row的open_time加60000才是完成边界；close_time必须为open_time+59999。首日原来源从19:38起，因而首完成边界为19:39；更早格点保持UNKNOWN。最后原open_time=Oct14 23:59的row完成于END，不提前放回面板；原row仍在SHA绑定的原文件中。无需重复制全部原文，不拼Aug／Sep归档、不前填、不后填、不改日历。

原JSON schema为`damorne-dynamic-public-kline-day.v1`。先核整个文件SHA、symbol、start/end与rows_sha256。12位置分别是open_time、open、high、low、close、base volume、close_time、quote volume、trade count、taker-buy base volume、taker-buy quote volume、Ignore。输出九原数值字段：`contract_1m_open/high/low/close`、`base_volume`、`quote_volume`、`trade_count`、`taker_buy_base_volume`、`taker_buy_quote_volume`。末位Ignore不当作经济量。原数值文本仅以精确Decimal规范文本表示，不先转float；原拼写可按文件SHA和rows数组内1起始行号重读。已观察到的真实零量保留0，空、非数、负量和非正价格各字段UNKNOWN。OHLC已知值的上下界关系矛盾时四价全UNKNOWN，保留经济键和异常，不能猜哪一价正确。

原rows顺序不决定时序。按完成边界稳定分组，等值Decimal重复可保留数值及全部原引用；同键某字段任一空／非法／不一致，则仅该字段DUPLICATE_CONFLICT。重复不采用首末优先或均值；顺序、不同精度文本或未来row变化不能改过去值。时钟不是整分钟、close_time错误、文件日或声明区间之外的row只进入异常审计，不四舍五入，不挪到正常钟。档案身份、rows内容摘要、规范路径及执行源变化为完整性失败；可识别的文件结构问题记录PARSE_FAILED并保留该文件对应未知分钟，不能吞掉编程异常。

`observed`仅表示存在合法时钟的原row，不表示九字段齐全、历史可交易或首次公开时间已证明。来源状态、各字段value_status、完成时间假设及首次公开未知分别保存。快照t仅取恰好于t完成的原一分钟bar，不取最近值。五分钟／小时若输出，只是相同完成钟的投影，不合成OHLC或均值。

晚期静态元数据来自同一固定protocol的原universe，另列asset_class、captured_status、onboard_ms、delivery_ms及`historical_pti_proven=false`。`late_static_qualified_at_open`仅按原bar open与这两个晚期日期比较，并标明是假设；未知日期为UNKNOWN。无论资格如何，价格原值和所有身份仍保留，未来delivery不能用于提前避单或改价格。

API：`bind_inputs(protocol_path)->BoundPrices`固定并首验全部12716原文；`parse_raw(payload,symbol,day)->dict`仅解析一个日包；`minute_rows(symbol,parsed_sources,metadata,clocks)->iterator`仅精确连接并保留未知；`run(protocol_path,output)->result`以逐符号／日批次写`minutes.parquet`，另存protocol、实际input_identities、source_profiles、anomalies、coverage、result和receipts。输出schema为`ta-basis-oi-price-source.v1`。执行仅依赖本模块、本计划与artifacts工具，首末均验SHA，完整首末再验所有12716原文与源protocol；中断或失败只封存在新目录，不覆盖。

首次构建前将执行源码、本计划和artifacts工具按SHA复制到新目录，复制后核验，保证未提交代码仍可复算。输出不能包含、覆盖或嵌套进原价格来源、H01封存目录或包源码。结束另核实际Parquet footer行数，每25符号报告处理进度；这些是来源整理记录，不是收益。

内存上限按单符号24480分钟的规范原记录加1440输出批次设计，不保留全748数据；按天写Arrow。实际分钟行数、各字段已知／未知状态、observed、原记录／重复／异常／区间外完成记录、按日／符号人口守恒全部封存。实际时间、峰值RSS与输出大小由首次构建如实记录，不把估算当测量。

合成验证覆盖原12位置／精确Decimal／零量／非法值、完成钟及首尾跨日、重复和冲突、缺格不填、晚期资格不删人口、未来扰动不改过去、价格与量无float替代、原文或rows摘要改变拒绝、12716对应身份结构及首末哈希、输出缺口与状态守恒、失败封存。测试不读取真实标签、收益或未来模型结果，不联网，不触及生产。
