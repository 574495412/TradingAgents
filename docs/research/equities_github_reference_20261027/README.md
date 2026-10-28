# 港美股：先与 GitHub 原策略收益对齐，再复用和优化

2026-10-27。**后续用户已纠正为成熟方案直接采用优先**，当前执行见 [adoption.md](adoption.md)。以下为纠正前已完成的收益复现审计记录，保留事实与冻结计划；其中“精确对齐后才采用/优化”的流程已由最新指示替代。

原轮次：**暂停港美股自研与参数优化**。当前顺序为：固定外部版本 → 对齐作者收益口径 → 原引擎和独立复刻核对 → 单列本地可执行条件 → 再考虑优化。原研究结题、24次账户预算、cash选择及失败记录完整保留；这次是用户新授权的外部复现范围，没有改动历史结果。

本轮实际完成：10个GitHub仓库固定commit/文件SHA；美股1个策略由作者原引擎及独立实现各跑1条路径；港股4条作者公开净值曲线复算。没有搜索新参数、训练新模型、运行本地适配账户或下单。未认证“全GitHub最高收益”，也未把不同区间、费用、杠杆或短期模拟账户年化混成统一榜单。

## 当前优先复现对象

|市场及对象|作者声明|本地已核到的范围|尚未完成|
|---|---|---|---|
|美股 [CazSyd/IBS-Strategy](https://github.com/CazSyd/IBS-Strategy/tree/c56cbdff2f18bc8ae7d7a2e2f2defe8243de1df5)，TQQQ IBS 0.132/0.965|2010-02至2026-07，年化59.3%，回撤56.8%；未扣交易费用/税，参数在样本内挑选|2010-02-11至2026-07-31，新固定行情下原引擎和独立复刻均为**59.6751%年化、56.8420%回撤**|作者原始行情快照、精确截止日和headline现金利息设置缺失；原59.3%仍未精确对齐；本地费用/真实执行未评估|
|港股 [Vectaix-Finance](https://github.com/Asher-S-Wu/Vectaix-Finance/tree/a65dc5c3494d59a784567f71caf042640280fb15)，small LightGBM|2024-02-01至2026-09-16，年化35.56%，回撤20.10%，每侧0.25%费率|644日公开NAV复算为**35.5564%年化、20.1000%回撤、122.0919%累计收益**，和作者表内数值相符|只完成产物算术核对；未重训、未重建全期选股/交易；原产物execution_validated=false|

这是本次有限来源集中，剔除明显非账户收益公式后优先核对的长历史高收益声明。港股116%的配对声称和美股76.66%的定投声称并未隐藏，原值及不参与账户CAGR排名的原因见下表。FinRL短期paper年化62.16%独立列示，未当长期历史策略第一名。

**美股同输入逐日对齐的证据：**4,142日现金、持股数量、净值最大绝对差均为0；177笔已闭合交易及1笔未闭合持仓逐项相同；截断未来数据后，历史前缀保持一致。作者引擎及独立实现均使用前一日IBS、下一交易日开盘成交、10000 USD初始资金、整数复权价格单位、闲置现金利息0。复权OHLC由当前Yahoo原始字节的AdjClose/Close构造，不是作者当时保存的数据。TQQQ本身有3倍日杠杆，本次账户没有额外借款。整数复权单位不认证历史真实股数/分红现金/整股交易。

与作者59.3%的差值为+0.3751个百分点，超过事前0.1个百分点容差；没有调截止日期、参数或现金利息去凑数。原数据/设置不完整使headline精确复现仍未通过，不能把原引擎与复刻相同误报成原历史数据已复原。当前待优化字段保持false。

**港股产物复算的证据：**4模型全部通过事前误差限；年化采用末NAV相对原始本金1、958自然日、365.25日年。回撤把买入前NAV=1纳入高水位，不能把第一天已经扣费的0.9982当作新本金。small LightGBM仍含169个参考估值日、1个未平仓及1个待卖持仓，采用复权分数单位；这些不是已验证可成交收益。0.50%费率情景31.41%是作者另表声明，本轮未从stress逐日曲线复算。

## 检索与原始声明审查

以下是作者口径，尚不能跨行直接比较。完整commit、许可、窗口及证据见 [catalog.json](catalog.json) 和 [sources.json](sources.json)。

|仓库|原始收益声明|原始回撤声明|本轮核查结果|
|---|---:|---:|---|
|swhy0913/Pairs-Trading-with-DTW-Clustering-and-OU-Process|116.41%算术年化|80.52%累计PnL落差|非共同本金净值CAGR；缺配对元数据且测试全窗OLS；未全策略复现|
|ZhongHong98/DCA-VS-Trend-Filtered-Strategy-Backtest-Python|TQQQ定投76.66%|81.56%|每月1000 USD转入后，期末总资产仍除最初1000 USD；未消除资金流，不作为策略CAGR|
|AI4Finance-Foundation/FinRL-Trading|paper62.16%；历史25.85%|paper12.22%；历史38.95%|前者仅2025-10至2026-03模拟账户，总收益19.76%；不是已核验实盘或长期CAGR|
|CazSyd/IBS-Strategy|59.3%|56.8%|同固定输入原引擎与独立复刻通过；历史headline快照仍缺|
|NivEyal/TQQQ-Strategy|README52.6%|61.7%|已提交对齐比较CSV为47.2420%，全历史审计CSV为46.9596%且窗口不同；README与产物尚未对齐|
|Asher-S-Wu/Vectaix-Finance|港股35.56%|20.10%|公开曲线计算通过；全链条复现待输入|
|HaomingJue/lev_etf_long_term_strategy|walkforward34.8%|同口径本轮未核|另一个全历史maxdd50口径含合成历史且回撤51.8%；不把该回撤移植至walkforward|
|justin10moter-ux/tqqq-trend-following|33.60%|54.9%|作者历史声明，本轮未重跑；live-since文字未当真实账户收益|
|davidwang0116/Quant-QQQ-QLD-TQQQ-SMA200Timing-MixedPosition-Backtest|全历史28.53%；近10年39.28%|76.37%；69.92%|全历史含上市前合成TQQQ；不同窗口分开保留|
|amit943c/MultiFactorStockRS|26.6%|27.4%|作者明示样本内、当前约100股票子集、无walkforward；本轮未重跑|

116%来源的代码问题可直接定位：notebook第13单元对传入的整个价格窗口回归，测试调用也会看到测试全窗；第14单元把不同配对的PnL直接加总，年化为mean×252，回撤为累计PnL减累计高水位，而非净值除高水位。作者输出仍保留，未替它生成没有共同本金的“纠正CAGR”。README的1,131日与notebook数据摘要的1,231日也需核对。

定投问题可定位到MonthlyInvestment的AddAmount，以及OnEndOfAlgorithm用equity_curve[-1]/initial_cash计算total_return和CAGR。逐日收益也未剔除注资。正确TWR/IRR需要完整现金流时间及净值记录，目前未给出，保持null。NivEyal产物差异另有窗口与执行实现差异，本轮未以单个CSV反推哪个README数字为真。

## 后续准入顺序与缺失输入

1. 美股先补IBS59.3%对应的原始OHLC快照、精确起止交易日及运行参数（含现金利息/成交模式）。只核对登记参数，不扫描日期或搜索阈值。已具备“原引擎+独立复刻”的同数据基线。
2. 港股先补35.56%对应的历史股票池/行情、模型和训练配置、全期月末分数、完整逐笔成交/现金/估值状态，再重放。公开performance_curve.csv本身不补足这些输入。
3. 原始收益通过后，再单独建立实际可交易的本地账户：保留原始结果，逐项展示股息/复权、整手、费用、成交与杠杆适用性的影响，不先改80%仓位或股票池再说已复现。
4. 最后才优化。原收益未对齐、来源仅有曲线或原始口径错误时，暂停优化并保留差异。原50%年化/15%回撤挑战仍未通过；IBS高年化伴随56.84%回撤，不得填入已验证挑战账户。

当前仓库API未检测到IBS及Vectaix仓库许可；本轮仅在本机私下审查/研究原文件，没有把第三方代码入库分发或连接实盘。NivEyal为MIT、FinRL为Apache-2.0，许可信息已固定；正式复用仍按所选仓库具体许可处理。原数据字节、外部完整源码与大型账本依仓库约定位于被忽略的data/research和results/research。

## 运行与证据

本次结果目录：`results/research/equities_github_reference_20261027_v001/`。原始行情JSON和仓库文件含SHA清单；运行前逐文件核验，登记plan、所有清单、行情及复现脚本hash后才执行。结果包含两引擎逐日路径、完整交易和因果前缀检查。[verification.json](verification.json) 为紧凑验证收据；[claim-audit.json](claim-audit.json) 为原始声明公式核查；[plan.json](plan.json) 为计算前冻结范围。

从TradingAgents根目录，先运行capture_sources.py、capture_details.py、capture_market.py固定来源（现有数据目录不覆盖）；已有快照可直接复算：

```sh
/Users/chrise/GitHub/CZ-Agent/.venv-tradingagents-lab/bin/python \
  research_tools/github_equity_reference/reproduce.py \
  --output results/research/新的唯一复现目录
python3 research_tools/github_equity_reference/audit_claims.py
```

复算仅需pandas/numpy；不加载.env，不导入应用交易配置，不执行notebook、优化器、下载模块或交易模块。源引擎只载入已审阅的backtest.py和metrics.py。完整复现结论仍待原始输入，工程对齐、产物复算和真实收益分别保留。
