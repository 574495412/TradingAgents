# 复核入口与保留范围

本轮为当前本地研究证据的有限复核。未修改交易信号、参数、旧经济门槛或生产配置；未运行远程策略、未读取实盘账单。395份带账户标记JSON未直接归一，主要包含计划、聚合摘要和原生ZIP镜像；它们不是395个新策略。完整缺口保留在归一化文件的`unrecognized_account_marker_files`与原始清单。

## 固定输入

- 初始研究界面快照：`data/research/existing_strategy_comparison_20261028_v001/catalog_snapshot.json`。
- 三项目文件及目录别名清单：`data/research/existing_strategy_inventory_20261028_v001/inventory.json`。扫描匹配50,822份证据文件；JSON保存源码路径与哈希，原生ZIP在归一化阶段保存哈希。不是策略计数。
- 最终账户：`results/research/existing_strategy_review_20261028_v005/accounts.json`。
- 可比组及审计：同目录`comparison.json`。
- 84条旧曲线：`results/research/existing_strategy_review_20261028_v003/legacy_curves.json`。
- 原样回放登记：Freqtrade的`research/crypto/existing_strategy_review_plan_v001.json`。原始计划、数据与341份引擎核心身份保持不变；结果在`user_data/crypto_quant/existing_strategy_review_20261028_v001/`。

上述大文件位于本地数据目录；Git提交保留工具、精简结果、原合同及哈希索引。缺本地原件的另一台机器需要按索引取得数据，不能只凭本报告重新生成数值。

## 重算与验算

在TradingAgents根目录，用Freqtrade已有Python环境运行：

```sh
/Users/chrise/GitHub/freqtrade/.venv/bin/python -m research_tools.existing_strategy_review.verify
/Users/chrise/GitHub/freqtrade/.venv/bin/python -m pytest research_tools/existing_strategy_review/test_normalize.py -q
```

`verify`重新哈希归一化账户及曲线的原件、原生核心和既有策略源码，检验本金收益恒等式、7次原样回放及原训练交易/订单一致性，并更新`verification.json`。订单现金公式由归一化工具的Decimal逻辑另算；它是独立于引擎报表的计算，不是第三方人审。

重新解析时必须使用固定清单、指定全新输出目录，保留v001–v005诊断产物：

```sh
/Users/chrise/GitHub/freqtrade/.venv/bin/python -m research_tools.existing_strategy_review.normalize \
  --inventory data/research/existing_strategy_inventory_20261028_v001/inventory.json \
  --additional-root /Users/chrise/GitHub/freqtrade/user_data/crypto_quant/existing_strategy_review_20261028_v001 \
  --output results/research/existing_strategy_review_audit_rebuild
/Users/chrise/GitHub/freqtrade/.venv/bin/python -m research_tools.existing_strategy_review.analyze \
  --accounts results/research/existing_strategy_review_audit_rebuild/accounts.json \
  --legacy results/research/existing_strategy_review_20261028_v003/legacy_curves.json \
  --output results/research/existing_strategy_review_audit_rebuild
```

原生补测控制器在Freqtrade根目录使用同一Python环境运行，输出目录必须全新：

```sh
.venv/bin/python -m research.crypto.review_existing_accounts \
  --plan research/crypto/existing_strategy_review_plan_v001.json \
  --output user_data/crypto_quant/existing_strategy_review_audit_replay
```

该操作会回放相同历史，不产生独立未见证据。`publish.py`固定读取本次v005快照，只适用于更新本次报告；新研究须另登记来源与输出目录。

## 指标边界

净收益为结算权益／明确本金−1；胜率按完整交易，零成交保持空值。原生报告的已平仓夏普/回撤原样保留，另算全持仓保存净值的观察回撤和日夏普。加密用365，港美股用252；同一时刻结算前后路径均保留计算回撤，日/月末取最后结算值。原生起始点延后不超过一小时且存在明确本金时补入已知初始现金，不能把首条扣费后权益当本金。仅有完整月末端点才计算月收益。

8bps中的额外3bps是费用等价代理，不能证明可成交滑点、盘口冲击或容量。不同频率保存的小时mark、5分钟参考价格和已平仓回撤不可互称。可比组允许原策略仓位预算、交易频率及槽位不同，因此不代表相同实际风险；无共同合同不排名。未验证容量、第三方复核、真正未见和实盘，总收益目标保持未完成。
