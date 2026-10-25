# TradingAgents 量化研究

2026-10-25 当前版本：[统一因子、宏观新闻、预测与HTML组合](factor_lab_20261025/README.md)。74活跃因子、12模型、64账户，统一源码在 `factors/`；本次用户已授权扩展原两候选范围及整理提交。旧主线协议／失败记录保留历史。

2026-10-25 当前主线：[流动性充足币种的中低频趋势／动量](crypto_trend_mainline_20261025.md)。已运行48个成本／资金／时段账户，训练选择现金；先验证有效收益源再扩展。当前优先级见 [主线清单](crypto_trend_mainline_policy.json)，此前分散研究保留为历史与后续队列。

本 fork 持有 DAmorne／Omofun 因子研究的代码、数据、实验和研究文档。CZ-Agent 保留生产策略、风控、订单与状态管理；本目录不向实盘下单。

2026-10-24 新增[港美股研究入口与首轮结果](equities_hkus_20261024.md)：基于 CZ-Agent、TradingAgents、Freqtrade 的现有职责，完成长桥只读适配及 108 个股票 ETF 现金账户，分别使用 USD/HKD。首轮训练选择均为 cash；长桥鉴权及实盘接入尚未验证。

最新进展：[原生H04对冲净现金流回归](hedged_carry_validation_20261017.md)已完成8784行标签、六情景回归及独立复核；完整筛查通过零、预测入场零，尚无可上线新策略。

## 目录职责

| 位置 | 内容 |
| --- | --- |
| `tradingagents/dataflows/research` | 公开数据采集、归档复用、来源合并、分钟与持仓时钟校验 |
| `tradingagents/dataflows/research/historical_capture.py` | 固定909人口的八、九月分钟／资金费原文、官方校验和REST现金字段采集 |
| `tradingagents/dataflows/research/historical_funding.py` | 显式Regular／Special视图、完整月度对账及缺源未知状态 |
| `tradingagents/dataflows/research/historical_panel.py` | 全909身份的两月小时特征和独立标签，保留静态资格假设 |
| `tradingagents/dataflows/research/spot_archives.py` | 论文先验固定BTC／ETH现货月档案与微秒质量核验，仅补源不算收益 |
| `tradingagents/research/quant` | 因子回归、IC／Rank IC、训练冻结预测、控制项和逐因子剔除 |
| `tradingagents/research/quant/factors` | 新研究因子公式，窗口和缺失规则独立于未来标签 |
| `tradingagents/research/quant/factors/alpha158.py` | 固定Qlib公式的158项分钟适配，完整窗口与未知状态明确保留 |
| `tradingagents/research/quant/ridge.py` | 成熟共同人口岭回归、训练内标准化及严格时间内折选参 |
| `tradingagents/research/quant/alpha158_study.py` | 九项基线与固定158项增强的同人口增量回归 |
| `tradingagents/research/quant/position_tilt_study.py` | 原生H02的60分钟固定岭回归、共同人口对照与两项主检验 |
| `tradingagents/research/quant/open_interest_flow_study.py` | 原生H03持仓扩张×滞后成交方向，含主效应的交互增量及完整覆盖比较 |
| `tradingagents/dataflows/research/hedged_market.py` | 真实现货／永续原分钟及实际结算事件连接，来源阶段不生成收益 |
| `tradingagents/research/hedged_capability.py` | 绑定独立来源审计的两腿输入能力，会计mark不能进入预测特征 |
| `tradingagents/research/quant/hedged_carry_study.py` | 原生H04两腿净现金目标OLS、18项统一BH及完整六成本情景 |
| `tradingagents/research/quant/extended_funding_study.py` | 原四小时资金费家族的固定／每周重训与24项声明检验 |
| `tradingagents/research/quant/extended_funding_diagnostics.py` | 同人口12分数的描述诊断，完整保留UNKNOWN身份与缺失标签 |
| `tradingagents/research/quant/factor_diagnostics.py` | 保留同值的五层标签、IC／ICIR、精确日历换手及秩自相关，独立于成本账户 |
| `tradingagents/research/quant/reference` | 独立持有的历史策略特征研究副本、原始分钟读取与面板构建 |
| `tradingagents/research/quant/backtesting` | 连续模拟账户、资金费、容量、退出、恢复与前向诊断 |
| `tradingagents/research/quant/backtesting/paired_forecast.py` | 显式预测调度的共享分钟驱动器，账户和风险跨重训连续保留 |
| `tradingagents/research/quant/backtesting/alpha158.py` | 预测列白名单薄适配，完整保留24成本账户及无交易／失败状态 |
| `tradingagents/research/quant/backtesting/extended_funding.py` | 全909身份分钟映射与48条四小时连续成本账户，现金未知使账户失效 |
| `tradingagents/research/quant/backtesting/capacity_sources.py` | 冻结规则来源、事前60分钟容量预算与完整人口分钟映射 |
| `tradingagents/research/quant/backtesting/capacity_account.py` | 共同缩量、精确数量网格、持续部分退出及残仓失效账本 |
| `tradingagents/research/quant/backtesting/capacity.py` | 全部48容量账户编排，独立保存有效前缀和反事实后缀 |
| `tradingagents/research/quant/comparison` | 纯量化、固定规则、单模型与原生多空研究的八个基础／压力账户 |
| `tradingagents/research/native_quant_review.py` | 量化证据分析师接入原生研究、交易及风险链，逐假说决定是否检验 |
| `tradingagents/research/quant/plans` | 随安装包保存的冻结研究设计 |
| `tests/test_quant_*`、`tests/test_research_*` | 研究测试，沿用仓库 pytest 和 Ruff 约定 |
| `data/research` | 本地原始与派生输入，Git 忽略 |
| `results/research` | 本地实验、不可变运行包和封存证据，Git 忽略 |

研究公式不导入 `cz`，纯回归不加载 LangChain／LangGraph 或调用模型。生产公式继续由 CZ-Agent 持有；研究副本的来源记录在 `docs/quant-migration-origin.json`，后续研究修改在本仓库独立维护。

## 安装与命令

在独立 Python 3.11 以上研究环境安装，不给 CZ 交易环境增加量化依赖：

```sh
python -m pip install -e '.[quant,dev]'
python -m tradingagents.research.quant --help

# 从完整封存面板重算固定五因子与两控制项。
python -m tradingagents.research.quant regression \
  --source results/research/profit_factor_search_20261015/quant_regression_v093 \
  --output results/research/my_regression

# 同一面板、人口和固定训练斜率上的七模型增量比较。
python -m tradingagents.research.quant incremental \
  --source results/research/my_regression \
  --output results/research/my_incremental

# 用另一套名次与线性代数实现独立复核。
python -m tradingagents.research.quant audit \
  --source results/research/my_regression \
  --output results/research/my_independent_audit

# 固定三个价格响应候选，比较九模型和三个预测期限。
tradingagents-quant response \
  --source results/research/profit_factor_search_20261015/quant_regression_v093 \
  --output results/research/my_response_study

# 仅将已结算普通资金费加入原七特征；特殊股息留在现金会计。
tradingagents-quant funding \
  --source results/research/my_response_study \
  --funding-root results/research/profit_factor_search_20261015/public_v001 \
  --output results/research/my_funding_study

# 用冻结训练预测运行完整连续账户，保留全部成本版本。
tradingagents-quant forecast \
  --source results/research/my_response_study \
  --market-panel data/research/dual_alpha_native_fusion_20261015/full_omofun_market_001 \
  --funding-root results/research/profit_factor_search_20261015/public_v001 \
  --output results/research/my_forecast_accounts \
  --extra-slippage-bps 11

python -m tradingagents.research.quant comparison --help
python -m tradingagents.dataflows.research.uniform_inputs --help
python -m tradingagents.dataflows.research.historical_capture --help
python -m tradingagents.research.quant.backtesting.paper_live --help
python -m tradingagents.research.quant.backtesting.capacity --help
```

命令中的历史面板已经使用，重算不会产生新样本外。输出目录必须不存在；不覆盖原结果。回归输出保留面板、输入身份和覆盖记录，可直接接续增量与独立复核。

历史 `cz-*` 数据 schema 保留用于辨认原契约，不能把 schema 前缀当作当前所有者。新编排与对照实验明确记录 TradingAgents 所有权。旧实验复核须使用其封存源码和依赖，不能用当前版本冒充原运行恢复。

CZ中没有调用方的旧研究转发入口已清理。历史冻结包独立保留原代码，新研究统一使用本仓库入口；数据路径兼容别名不意味着旧CZ模块命令仍然存在。

## 数据转交

首轮实际转交五个研究目录，约 19 GB；随后补充转交三组早期实验数据，全部八组映射见 [存储清单](storage-map.json)，清理范围见 [CZ轻量化收尾](cz_cleanup_20261016.md)。使用同一文件系统的目录移动，CZ 原数据路径仅保留指向本仓库的符号链接；没有另存一份活动数据库。旧绝对路径继续服务历史收据与原冻结进程，字节、原采集时刻及历史失败不回写。新任务直接使用本仓库路径。

原始、派生数据和不可变源码归档均留在本机，不加入 Git 源码提交。数据转交的后置脚本曾发生路径类型错误，目录转交已经完成且六个研究进程在 `finally` 中恢复；原失败保留，随后独立核对全部 12,716 份回归原文，共 2,253,386,900 字节，均与原封存 SHA 一致。实际短暂停顿时长没有持久记录，不能报告精确值或据此声称连续无缺口。

原全天模拟此前已经出现中断；本次迁移没有抹平缺口，没有改变原结束时刻、持仓、触发、费用和风险规则。仍运行的是原冻结诊断包；本版尚未发布新的前向研究或生产服务。

## 研究标准与当前结论

借鉴项目和已实现／待实现的边界见 [参考实现](references.md)。模型负责提出可检验的机制、输入、公式与失效条件；数值计算负责验证。没有新候选或新经济证据时，不再通过重复多空审查冒充研究进展。

本版包含现有五因子回归、七模型增量和连续扣费账户，后续家族已加入固定边界扩展重训及独立因子诊断。五分位、IC／ICIR、精确日历换手等借鉴公开定义，在本仓库独立实现；不等于接入Qlib或Alphalens引擎。通用因子表达式注册、任意家族重训调度和实体／日期匿名化尚未实现。

迁移不改变策略结论：既有五因子成本敏感性和模型验证尚未得到可晋级结果。已核验的基差候选短前缀也亏损，尚有库存、缺失估值和退出延迟；没有证明稳定盈利或生产资格。不得将 IC、探索性显著性、测试数量、迁移成功或模型调用次数等同于盈利。

迁移后的价格响应研究、完整账户及追加压力结果见 [本轮研究验证](response_validation_20261016.md)。48个原账户完整重放后与追加压力运行中的对应账户一致；唯一两档盈利的组合，在单边滑点增加1bps后转负，尚不满足稳定性或接入要求。

后续 [已结算资金费研究](settled_funding_validation_20261016.md) 出现加密240分钟双因子历史经济候选，三档成本均盈利，但预测增量不可靠、压力胜率下界及实际执行证据不足。`forecast --source results/research/my_funding_study` 自动执行该家族预先声明的全部72账户及三档成本，其余市场与资金费参数同上；不以候选历史盈利替代上线条件。

用户于2026-10-16取消强制未来两段三十日前向，当前以[历史时间隔离与滚动训练计划](../../tradingagents/research/quant/plans/historical_rolling_plan.md)验收，真实前向观察可选。对上述资金费来源增加 `--historical-rolling`，即可在四个预定边界扩展训练，同时保持全部72账户的本金、持仓和风险连续；历史已参与研发，不称新增独立样本外。旧三十日统计工具保留为可选复核。

[本版历史滚动验证](historical_rolling_validation_20261016.md)已完成72条连续账户及201项相关测试。双因子压力收益约+1.38%，但只有8批次且集中于首块，尚无账户通过原项目经济筛查；不设最低研发天数，达标即推进接入。

随后实际运行量化证据分析师及原生下游八角色，组合经理选择结算退出假说H01。它是原生图加本fork量化扩展，不是未经修改的标准`propagate`入口。九次真实调用完成后，按其设计执行新时钟下四模型回归和全部48连续账户；八项预测误差增量均为负、12项统计和48条账户经济筛查均未通过，H01不晋级。完整流程、全部成本结果与可复算命令见[原生研究链与H01验证](native_quant_validation_20261017.md)。新入口为`settlement`与`settlement-accounts`，不把原生讨论或测试通过当作盈利成果。

纳入失败证据后的九节点复审没有选择新候选。下一步扩大已冻结候选的历史证据；固定人口、归档容量、资金费现金及可交易资格的区别见[历史扩展说明](historical_expansion_20261017.md)。数据采集尚不产生新增收益或上线结论。

随后完成固定Alpha158分钟适配的全部748合约真实特征、同人口Ridge比较及24条连续成本账户。TradFi固定方式MSE改善约9.16%、四项BH q约0.00750，但滚动方式未通过统计筛查；增强模型12条均无交易，全部24条经济筛查未通过，不能据此称新盈利策略。覆盖损失、全部四比较、账户和复算入口见[Alpha158首版结果](alpha158_validation_20261017.md)。

八、九月扩展已完成全909身份小时双表、显式Regular／Special视图、46日原240分钟家族回归及12分数描述诊断。双因子对同基线四组误差改善均负；固定Crypto单加日变化虽有约0.00547%误差改善、q约0.03155，其实际账户亏损。原固定滑点账户滚动Crypto双因子10bps的+17.63%仅作历史参考，其整分钟成交额桥接已暴露容量不足；来源、统计和旧版全部案例见[两个月资金费续检](extended_funding_validation_20261017.md)。

随后按事前10%容量、离散数量和持续部分退出完成全部48案及独立原文审计：45条在冻结代理内完整、3条固定Crypto双因子因XRP期末残仓INVALID。完整账户为27有交易、18无交易，10正、17负、18零；滚动Crypto双因子三档净+0.392554%／+0.109257%／+0.067479%，10bps有86腿24批、Wilson下界37.4454%、PF 1.04621，未过原条件，没有家族三档全过。该档181个fill中24个超过事后实际分钟quote量10%、SUN一笔达8.691倍；前一完成分钟预算仍不证明真实可成交。后续reduceOnly官方豁免未计入V1，不能回写计划或把失效改为成功。完整16组×三档及审计边界见[容量首版结果](capacity_validation_20261017.md)，当前不接入实盘。

[后续撮合设计](capacity_matching_design_20261017.md)仅为未实施草案，尚无新实验结果；不能据草案声称已有真实撮合能力。策略选单和提交额度仍不得用未来量提前筛选，本轮静态晚期规则与历史mark代理的限制保留。

指数／metrics新来源已完成全原文数值审计、3662208行五分钟与305184行小时面板、独立36622080字段逐格复核。另独立生成18311040分钟合约价身份，未知保留。新九节点原生审查选择的H02头部持仓／账户比偏离机制已完成固定60分钟岭回归及独立重建：加密日均MSE改善为−9.77e−10、系数负；TradFi为+4.96e−10但q约0.786，两资产均未通过。全部研究属于9月28日至10月14日的已看历史，尚无新增盈利或上线结论；过程与边界见[新来源验证](basis_oi_validation_20261017.md)及[H02完整结果](position_tilt_validation_20261017.md)。新命令为`python -m tradingagents.research.quant position-tilt --output results/research/新的独立目录`。

H02失败后的九节点复审选择H03持仓扩张×滞后成交流，固定S=max(小时OI对数变化,0)×结束于t−10分钟的60分钟成交流。B3对含F/G主效应的B2、现金及B1对照已全部完成；加密误差略改善但q约0.173、系数与RankIC为负，TradFi误差变差，均未通过。全量特征、十二模型及1077120个预测独立复算通过；没有运行H03成本账户或授予盈利资格，见[H03完整结果](open_interest_flow_validation_20261017.md)。新入口为`open-interest-flow`，旧H02默认结果精确复现。并行价格来源连接保留完整305184小时身份，未生成basis，历史单位与发布时间仍缺证。

H03后一次九节点复审未提出新假说，没有重复调用。价格连接独立全量复核已通过，旧基差精确规则已加入避免换名重测。随后按BIS公开研究先验在新样本读取前固定BTC／ETH，取得八、九月四份真实现货月档，共175680分钟，官方SHA及独立全原文／微秒月历复核通过。当前仅补源，尚未构建对冲现金流标签或收益；见[复审与现货来源](post_h03_source_expansion_20261017.md)。该可行性范围不删除旧748／909人口，不从中筛赢家；新入口为`python -m tradingagents.dataflows.research.spot_archives`。

## 开发约定

沿用仓库的 Python 包、pytest 和 Ruff 配置。数据采集放在 `dataflows/research`，研究公式放在 `quant/factors`，回归和研究编排放在 `quant`，账户与执行诊断放在 `quant/backtesting`；不在 CZ 新增研究引擎。新增代码注释、文档字符串和研究说明使用中文。

每次研究先冻结公式、成员、时段、成本和评估方法，再计算结果。实验必须新建输出目录，保留失败与来源收据。源码、计划、测试和说明保留在工作区供审阅，依用户最新要求不自动提交；原文、面板、账户账本及环境留在本地忽略目录。测试只能证明对应实现行为，盈利与上线结论须另有历史经济、实际执行及运行证据。
