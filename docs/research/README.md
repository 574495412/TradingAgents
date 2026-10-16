# TradingAgents 量化研究

本 fork 持有 DAmorne／Omofun 因子研究的代码、数据、实验和研究文档。CZ-Agent 保留生产策略、风控、订单与状态管理；本目录不向实盘下单。

## 目录职责

| 位置 | 内容 |
| --- | --- |
| `tradingagents/dataflows/research` | 公开数据采集、归档复用、来源合并、分钟与持仓时钟校验 |
| `tradingagents/research/quant` | 因子回归、IC／Rank IC、训练冻结预测、控制项和逐因子剔除 |
| `tradingagents/research/quant/factors` | 新研究因子公式，窗口和缺失规则独立于未来标签 |
| `tradingagents/research/quant/reference` | 独立持有的历史策略特征研究副本、原始分钟读取与面板构建 |
| `tradingagents/research/quant/backtesting` | 连续模拟账户、资金费、容量、退出、恢复与前向诊断 |
| `tradingagents/research/quant/comparison` | 纯量化、固定规则、单模型与原生多空研究的八个基础／压力账户 |
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
python -m tradingagents.research.quant.backtesting.paper_live --help
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

本版包含现有五因子回归、七模型增量和连续扣费账户。通用因子表达式注册、滚动训练调度、实体／日期匿名化与完整 Alphalens 式分层换手报告仍需后续开发，不能称已经接入这些外部引擎。

迁移不改变策略结论：既有五因子成本敏感性和模型验证尚未得到可晋级结果。已核验的基差候选短前缀也亏损，尚有库存、缺失估值和退出延迟；没有证明稳定盈利或生产资格。不得将 IC、探索性显著性、测试数量、迁移成功或模型调用次数等同于盈利。

迁移后的价格响应研究、完整账户及追加压力结果见 [本轮研究验证](response_validation_20261016.md)。48个原账户完整重放后与追加压力运行中的对应账户一致；唯一两档盈利的组合，在单边滑点增加1bps后转负，尚不满足稳定性或接入要求。

后续 [已结算资金费研究](settled_funding_validation_20261016.md) 出现加密240分钟双因子历史经济候选，三档成本均盈利，但预测增量不可靠、压力胜率下界及实际执行证据不足。`forecast --source results/research/my_funding_study` 自动执行该家族预先声明的全部72账户及三档成本，其余市场与资金费参数同上；不以候选历史盈利替代上线条件。

用户于2026-10-16取消强制未来两段三十日前向，当前以[历史时间隔离与滚动训练计划](../../tradingagents/research/quant/plans/historical_rolling_plan.md)验收，真实前向观察可选。对上述资金费来源增加 `--historical-rolling`，即可在四个预定边界扩展训练，同时保持全部72账户的本金、持仓和风险连续；历史已参与研发，不称新增独立样本外。旧三十日统计工具保留为可选复核。

[本版历史滚动验证](historical_rolling_validation_20261016.md)已完成72条连续账户及201项相关测试。双因子压力收益约+1.38%，但只有8批次且集中于首块，尚无账户通过原项目经济筛查；不设最低研发天数，达标即推进接入。

## 开发约定

沿用仓库的 Python 包、pytest 和 Ruff 配置。数据采集放在 `dataflows/research`，研究公式放在 `quant/factors`，回归和研究编排放在 `quant`，账户与执行诊断放在 `quant/backtesting`；不在 CZ 新增研究引擎。新增代码注释、文档字符串和研究说明使用中文。

每次研究先冻结公式、成员、时段、成本和评估方法，再计算结果。实验必须新建输出目录，保留失败与来源收据。源码、计划、测试和说明提交到 Git；原文、面板、账户账本及环境留在本地忽略目录。测试只能证明对应实现行为，盈利与上线结论须另有历史经济、实际执行及运行证据。
