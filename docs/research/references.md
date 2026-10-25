# 量化研究参考与采用范围

核对日期：2026-10-16。用户提供的近似拼写与功能最接近 TraderHarness，本版以它作为参考，另核对 Qlib 和 Alphalens 的官方代码。没有直接复制外部项目代码或继承其示例收益。

| 项目及原始来源 | 可借鉴内容 | 本版落实与边界 |
| --- | --- | --- |
| [TraderHarness 架构](https://github.com/HephaestLab/TraderHarness/blob/3e18369da5a2060f9c65b27c2a63f99edd93e167/docs/architecture.md) | 时点可见性、单一路径撮合、请求指纹和确定性回放 | 研究数据与账户归 TradingAgents，四组独立账户使用相同资金、成本和容量；缓存请求绑定快照、模型与源码身份 |
| [量化研究员卡片](https://github.com/HephaestLab/TraderHarness/blob/3e18369da5a2060f9c65b27c2a63f99edd93e167/traderharness/agents/builtin/quant-researcher.json) | 可计算公式、完整横截面、覆盖与风险暴露 | 固定共同人口，未来标签不改变特征排名；控制项和逐因子剔除一起报告。该卡片本身不是回归算法或盈利证明 |
| [TraderHarness 回放](https://github.com/HephaestLab/TraderHarness/blob/3e18369da5a2060f9c65b27c2a63f99edd93e167/traderharness/trajectory/replay.py) | 改动请求或缺失回放时明确失败 | 本地四组实验验证请求、响应、源码、决策与账户；不以模型失败后的虚构 HOLD 代替失败记录 |
| [Qlib 记录与评估](https://qlib.readthedocs.io/en/stable/component/recorder.html)、[IC 实现](https://github.com/microsoft/qlib/blob/main/qlib/contrib/eva/alpha.py) | 分离预测、信号统计与账户回测；IC／Rank IC 与风险、成本分列 | 已有 IC、Rank IC、日均统计、固定 3 日 HAC、探索性 BH、固定训练模型后段 R²；保持各指标口径，不造通用合成分数 |
| [Alphalens 因子评估](https://github.com/quantopian/alphalens/blob/master/alphalens/performance.py) | 因子 IC、分层前瞻收益、换手和秩自相关 | 当前保留同一已知人口的两端组标签诊断；完整分层、换手与自相关报告尚未实现 |

TraderHarness 当前维护的数据与撮合规则主要面向 A 股。其交易时段、整手、涨跌停、年化和示例基准不适用于直接评价永续合约。本研究继续使用公开合约资格、资金费、双向持仓、真实接收时钟、完整成本和连续账户，不套用外部回测标题数字。

实体与日期遮罩值得后续补充，但不能保证历史模型没有记忆。当前已使用的历史回归明确标为回顾数据，重算不能改称新增独立样本外。用户已取消强制真实前向等待：历史按时间隔离、滚动训练及完整成本连续账户是当前验收路线，不设最低研究天数。若另行声称新独立样本外，仍需说明候选冻结和未使用数据的真实边界。

2026-10-17再次核对[Qlib因子评估原始实现](https://github.com/microsoft/qlib/blob/main/qlib/contrib/eva/alpha.py)及[Alphalens原始实现](https://github.com/quantopian/alphalens/blob/master/alphalens/performance.py)：IC／RankIC、分位收益、分位换手和秩自相关用于描述不同性质，不能合成一个自动上线分数。分位成员须在读取未来收益前确定；相同因子值不按品种名字拆成不同尾组。换手与持久性明确真实日历时距，缺期不压缩成相邻。原始／去截面均值的标签诊断与成本后连续账户收益分别列示。

机制参考也核对了[加密货币风险与收益](https://www.nber.org/papers/w24877)及[加密货币共同风险因子](https://www.nber.org/papers/w25882)。这些原始研究涉及加密动量、市场及规模等解释，不能把其样本结论直接移植为本项目的永续合约日内盈利证据；成交量也不等同于流通市值。新机制仍须给出当前可用输入上的准确公式，并与已测短期动量、主动流、冲击残差及资金费因子区分。

本轮优先完成研究与执行解耦、数据所有权、回归重算和行为回归。后续新增因子应先登记定义、来源可用时间、训练／验证区间、全部比较与成本，再运行；回归结果不能自行触发 CZ 实盘。
