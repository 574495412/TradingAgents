# 统一因子库

这是股票与加密研究因子的统一研发入口。新因子只在此处实现和登记，组合、预测、数据采集、账户与 UI 不再各维护一份公式。

```text
factors/
├── catalog.json           # 74个活跃因子的唯一ID、公式、类别、时序和适用市场
├── origins.json           # 195份历史源码/元数据的原位置与SHA256
├── common/panel.py        # 跨市场OHLCV、趋势、技术、风险、流动性、横截面及交互
├── equity/                # 股票适配：交易日历不填充，排除加密专属资金费
├── crypto/quant/          # 已迁入的资金费、OI、对冲现金流、Alpha158等研究公式
├── macro/news.py          # 官方事件来源、公布时间、词典分数与5分钟可用延迟
├── reference/             # 已迁入的独立原始指标和横截面公式
└── legacy/                # 既有CZ/Freqtrade研究与执行版本的来源一致参考
```

`catalog.json` 的74列是本轮活跃实现，包含不同窗口及交互，不能解释成74个独立alpha。`origins.json` 是195份历史文件的索引，文件数不是因子数。历史复刻因子、策略规则与纯价格因子分别保留来源，不把旧研究的失败改成通过。

## 统一调用

以下导入在TradingAgents工程或其独立研究环境中使用；CZ生产工程不直接导入相邻研究包。历史参考源码保留原字节与来源摘要，局部空白检查例外仅适用于 `legacy/`，新研发源码继续检查。

```python
# Stock input: timezone-aware, unique, ordered real session bars, no weekend fill.
from factors.equity import calculate
frames, source, definitions = calculate(
    {"SPY": spy_bars, "QQQ": qqq_bars},
    events=official_events,
    benchmark_symbol="SPY",
)

# Crypto input: the same shared calculation plus known settled funding.
from factors.crypto import calculate
frames, source, definitions = calculate(
    crypto_bars,
    events=official_events,
    funding=settled_rate_series,
    benchmark_symbol="BTCUSDT",
)
```

行情每行须有 `open/high/low/close/volume`，索引是带时区的bar开始时间。当前行只使用前一根已完成bar；窗口长度表示实际观测bar数。真实 `quote_volume` 可选，缺少时流动性值保留未知，不用价格×成交量伪装成交易所成交额。股票停市日不生成假bar，拆股、股息、交易日历与252日风险频率由股票数据／账户适配负责；加密账户使用365日。

股票适配当前输出69个共同因子，排除4个资金费列和1个资金费交互项。无新闻输入时新闻列保留未知；新闻只能在官方公布加声明延迟后匹配，不提前看到声明内容。当前22份FOMC声明是官方宏观新闻样本，不代表完整商业新闻覆盖，也没有分析师一致预期差数据。

## 目录迁移与执行边界

原 `tradingagents/research/quant/factors` 与 `reference/factors` 只保留导入桥；实际公式已迁入此目录。`research_tools/factor_lab/features.py`、`news.py` 也只做适配转发，预测与组合从这里取得因子定义。

CZ-Agent 的 `factors/stock`、`factors/crypto` 当前被生产适配器直接按路径使用，保留为既有执行版本；统一研发库保存字节一致的研究参考和来源摘要。它们不是第二个新因子研发入口，也不通过跨仓库软链接或导入研究环境改变运行中的策略。以后提升执行版本时须显式冻结并同步对应快照。

Freqtrade 的历史冻结实验与其旧源码身份保留，不通过修改旧收据伪造新结构下的历史重跑。新组合回放只读取上游有时间戳的因子／预测分数，不重新实现因子。

安装研究依赖：`pip install -e '.[factors]'`。完整因子工作台由 `research_tools/factor_lab` 编排；它不导入TradingAgents应用启动器或加载`.env`。

## 事后组合研究

`portfolio/recipes.py` 保存已有因子ID的组合配方与选择来源，不增加独立因子数。2026-10-25等权6因子方案属于已观察历史后的假设，未通过净收益筛选。64组对比、滚动训练及全部226账户的结果见 [比较报告](../docs/research/factor_lab_review_20261025/README.md)。

## 真实资金流与状态因子研发

`common/mechanisms.py` 新增6个机制定义，登记在 `catalog.json.discovery_registry`；`common/regime.py` 提供同一时点的BTC趋势、市场宽度及相对动量条件。原74列冻结面板保持原定义，不把新增研发项自动加入旧模型。126账户结果及拒绝原因见 [研发报告](../docs/research/factor_discovery_20261025/README.md)。

加密五分钟数据先聚合并滞后一根小时bar；股票传入 `features(session_hour_bars, asset_class="equity")`，保留原交易日历，窗口按观测bar计数。加密日历7/21/63天不可原样解释为股票天数，日UTC策略规则不作为股票策略适配器。真实主动成交字段缺失则保持未知，不用涨跌符号代替。

## GitHub来源复现

协整币对形成期的共同数学定义集中在 `portfolio/cointegrated_pairs.py`，协议在 `research_tools/regime_program/pair_plan.json`。它只用形成时刻之前的观测价格、双向EG及多重校正、Johansen秩和成本空间筛选，不读取账户收益排序。股票可以复用按真实交易日排列的数学输入，当前半衰期天数、流动性门槛与交易规则只验证加密日历，不能直接声称股票账户有效。第三轮已完成26个双腿成本／容量账户，滚动方案只有5个匹配周期，仍属证据不足；见[完整账户报告](../docs/research/regime_program/pair_accounts_20261026/README.md)，[形成期快照](../docs/research/regime_program/pair_formation_20261026/README.md)作为前一阶段保留。原74列面板不变。

`common/github_formulas.py` 保存PST的3个日频规则、Qlib的2个具体价量公式及Sentinel组合映射；固定源码commit与研究状态登记在 `catalog.json.github_research_registry`。这是来源明确的数学适配，不能继承原股票／期货工作流的收益声明；5个公式和3个组合不等于8个独立盈利因子。13个仓库筛选、40币数据质量及46个完整账户见 [GitHub复现报告](../docs/research/github_factor_study_20261025/README.md)。

`portfolio/module_domains.json` 集中登记7张条件性模块／风控卡；这属于适用域与失效假设，不增加已验证因子数。模块、配置和总账户分别按v002协议验收，不要求每个策略全年20%或覆盖所有行情。已完成其中5类既有策略的10个条件账户，未取得双成本净优势；见[完整结果](../docs/research/regime_program/domain_accounts_20261026/README.md)，[预登记与解释](../docs/research/regime_program/module_domains_20261026/README.md)保留原版本。

`crypto/spot_perpetual_carry.py` 保存既有资金费机制的一次固定成本条件对照，另有固定持有基线；不改原74列，也不计作两个独立收益源。真实现货库存与永续钱包由Freqtrade研究引擎分别核算；20成本／区间／容量账户已独立重建，4次持仓检查点经新进程恢复一致，但只有2/7个完整周期，仍证据不足。见[分账账户结果](../docs/research/regime_program/spot_carry_accounts_20261026/README.md)。股票不能直接使用加密资金费与交易日历规则。
