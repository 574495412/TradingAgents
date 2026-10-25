# 三项目统一量化结构

最新持续研发目标：[多行情、多策略任务协议](docs/research/regime_program/README.md)，唯一目标配置为 `docs/research/regime_program/targets.json`。先网络检索／固定版本复现，再针对失效机制自研；目标制定、策略实现、收益达标分开记录。旧实验计划与失败结论不更改。

```text
TradingAgents/                         # 研发与验证
├── factors/                          # 统一因子源码、跨市场适配、公式/时序/来源登记
├── research_tools/factor_lab/         # 特征组装、冻结训练、预测、代表性窗口和组合编排
├── research_tools/factor_lab_audit.py # 独立现金/净值/资金费/容量与训练复核
├── data/research/                     # 本地原始/派生输入（不提交大数据）
├── results/research/                  # 不可覆盖实验、账户与审计（不提交大数据）
└── docs/research/                     # 可版本化协议、结果摘要、交接

freqtrade/                            # 加密执行研究
├── research/crypto/factor_portfolio.py # 因子组合的成本/资金费/容量/持仓回放
├── research/crypto/trend_mainline/     # 前版趋势基线及历史证据
└── user_data/crypto_quant/            # 本地数据、原生/研究账户与不可变运行包

CZ-Agent/                             # 研究界面与独立生产职责
├── web/cockpit/src/FactorLab.jsx       # 在现有界面中选因子、调权、运行及比较
├── web/cockpit/scripts/factor_lab_service.py # 本机研究API，固定研究worker
├── factors/                          # 已有生产执行快照；研发源见TradingAgents/factors
└── ops/quant_research_mainline.json    # 现行职责与最新研究入口
```

2026-10-25用户将原两候选限制更新为更高维度因子、宏观新闻、预测能力和HTML量化组合。当前允许按新协议研发，不再用旧版“不扩展”限制阻止本次授权；旧失败、旧账户和旧假设不追溯改判。

目标是快速、可复现地得到量化结果。训练使用2024年，后续每个可用季度固定取前28天，覆盖七个窗口，不按候选收益选时段。短窗口风险比率是描述性年化，不把窗口拼接成连续账户，也不声称是未经查看的独立样本外。

本次三个仓库各形成一个本地提交。代码、协议、测试及已有工程改动可版本化；数据库、行情大文件、运行日志、凭据和研究虚拟环境不纳入提交。提交不等于推送、部署或实盘升级。
