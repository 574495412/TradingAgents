# 历史资金费显式类型研究视图

本视图只读取已封存的 `historical_market_20261017_v002`，不修改采集器或封存结果，不发网络请求、不调用模型、不计算收益或账户、不自动提交。它不是交易许可或历史完整性证明。

## 固定契约

入口为 `historical_funding.run(source, output)`，命令行使用 `--source` 和 `--output`。完整保留来源909个符号、2026年8/9月1818个符号月份，不按当前状态、覆盖、收益或是否可交易删成员。输出 `protocol.json`、`coverage.json`、`typed_cash_events.jsonl`、`rejected_events.jsonl`、`errors.json`、`result.json`、`receipts.json`，另封存输入收据与实际执行源码。未知月份仍在coverage中。

运行先冻结来源收据/计划/原对账表的身份和实际模块、分页helper、本文，再流式核对来源完整文件集合与所有SHA。实际读取的路径须在源目录中、被源收据绑定且属于对应任务，拒绝绝对路径、路径逃逸、符号链接和错符号/错时钟来源。末验固定输入及实际执行源码；并发改动不得被记为成功。全局身份错误保留失败结果及本轮收据，不发布可信现金事件。单个符号或月份语义错误保留该项未知及原文引用，其他月份继续，不能要求全909就绪才能保存质量诊断。

## 类型及原文核验

Regular/Special只能取REST显式 `rateType`。每个事件须有原始整数毫秒fundingTime、正确symbol、有限Decimal fundingRate、有限且正的Decimal markPrice，原毫秒（包括整点后1ms）和所有额外原字段保持不变。不从三字段月包默认Regular，也不把档案中的Special偷偷转入普通因子。

REST从任务记录绑定的原始成功响应及收据离线重放固定分页helper，逐次核对公开请求的symbol/startTime/endTime/limit=1000、原文SHA及请求顺序。重放完整事件必须与规范events JSON严格相同，不能只相信后者。归档须重新核官方UTF-8摘要/文件名/ZIP唯一CSV成员/三字段，并与规范funding_rows JSON逐字段一致。

去重不使用(symbol,time)，Regular与Special同钟均保留。重复(symbol,time,type)不静默覆盖；不同费率/mark或无法解释的多重性须明确标冲突，保留原事件和未知状态。无法完成类型或原文核验的规范事件单独进入rejected记录并保留原来源引用，不混入可信typed事件。typed事件保留原REST dict、原序号及来源事件文件SHA；显式类型已知也不自动代表现金历史独立完整或历史可交易。

## 月度覆盖

`coverage`原样保留来源Regular对账状态及数量，原1416 MATCHED、41 MISMATCH、361 UNPROVEN_MISSING_SOURCE不得改写。新增比较只按原毫秒和精确Decimal费率的多重集合，对照月包与REST Regular+Special；计算归档独有/REST独有的观测差异，完整非空且相同才记MATCHED。

已知41差异月的43条归档额外记录与REST显式Special对应，该事实另列，不把原Regular比较改绿，也不将月包称为纯Regular。若同钟异类型、重复或标记价冲突，保留全部原事件，单独降低类型身份质量状态。

缺档、缺REST、API空或当月任一比较侧为空均为UNKNOWN；观测行数0只表示本次响应/文件没有行，绝不成为已知零现金。每个coverage条目分别报告原对账、合并对账、显式类型和冲突状态。`API_RANGE_VERIFIED`仅表示接口请求范围与分页协议假设，归档相等加请求范围仍不等独立完整性、历史首次发布或逐日可交易资格。全局 `independent_completeness_proven` 与 `historical_eligibility_proven` 始终为false。

## 验证

仅用合成档案测试原1416/41/361语义、Regular+Special多重集合、同钟异类型、重复同类型冲突、未知类型、非有限费率/非正mark、原文与规范JSON差异、UTC月边界及+1ms、错源/路径逃逸、并发源码变化和部分失败仍封存。根代理审查后再运行真实视图；本次实现与测试不读取真实资金费计算视图。
