"""Publish formation evidence with explicit uncomputed financial metrics."""

import argparse
import shutil
from pathlib import Path

from research_tools.factor_lab.io import read, save, sha


def publish(root, audit_path, output):
    output.mkdir(parents=True, exist_ok=False)
    receipt = read(root / "completion.json")
    for name, digest in receipt["manifest"].items():
        if Path(name).name != name or sha(root / name) != digest:
            raise ValueError("Formation receipt changed")
    audit = read(audit_path)
    if not audit.get("passed") or audit["completion_sha256"] != sha(root / "completion.json"):
        raise ValueError("Independent formation audit missing or mismatched")
    plan = read(root / "plan.json")
    sources = Path(plan["source_pack"])
    if (
        sha(sources / "source_review.json")
        != read(sources / "completion.json")["source_review_sha256"]
    ):
        raise ValueError("Source receipt changed")
    for record in read(sources / "source_review.json")["records"]:
        if sha(Path(record["local"])) != record["sha256"]:
            raise ValueError("Source file changed")
    summary = read(root / "summary.json")
    summary.update(
        stage="formation_only",
        title="第三轮：协整币对价差回归",
        account_returns_computed=False,
        net_return=None,
        net_win_rate=None,
        max_drawdown=None,
        sharpe=None,
        capacity_usdt=None,
        audit=audit,
        source_records=len(read(sources / "source_review.json")["records"]),
        next_action="固定两套形成期，不追加参数网格；下一步实施双腿独立成交、未对冲库存退出和配对生命周期账本，再按两档成本完成连续账户与容量检验。",
    )
    save(output / "summary.json", summary)
    for source, name in [
        (audit_path, "independent_audit.json"),
        (root / "plan.json", "plan.json"),
        (root / "completion.json", "formation_completion.json"),
        (sources / "source_review.json", "source_review.json"),
    ]:
        shutil.copyfile(source, output / name)
    rows = []
    for profile in summary["profile"]:
        pairs = "、".join(profile["selected"]) or "现金"
        label = (
            "2024固定形成期"
            if profile["file"] == "fixed_formation.json"
            else profile["fit_at"][:7] + "月度形成期"
        )
        rows.append(f"|{label}|{profile['eligible_names']}|{profile['directional_tests']}|{pairs}|")
    text = (
        """# 协整币对研究：形成期筛选完成，账户收益尚未验证

2026-10-26。多行情研发第三轮，收益目标继续进行。前两轮的价格规则和仓位修正没有通过经济验收，本轮转向币对价差回归。筛选前已登记 [plan.json](plan.json)；本页是形成期与数值核验，不能作为盈利、容量或生产资格证明。

## 当前结果

固定2024形成期选出IOTAUSDT/SANDUSDT。180日月度滚动在2025-01、2026-02、03、04、05有入选组合，其他15个月现金。部分币对跨月重复，不算多个独立收益源。固定方案与滚动方案全部保留，没有按后续收益反选。

|形成期|当时符合流动性与历史长度的币数|双向检验数|不重复占用币种的入选币对|
|---|---:|---:|---|
"""
        + "\n".join(rows)
        + "\n\n"
        + f"累计21次形成期拟合、{summary['directional_tests']:,}次双向EG检验、{summary['unknown_tests']}个未知检验。每个形成期分别做BH校正；累计次数不是独立样本数，也未消除历史反复研究的多重搜索偏差。原始4625组币对记录中4603组未通过双向FDR，22组通过其余形成条件；贪心排除共享资产后留下8组币对月份/固定期记录。\n\n"
        + """## 方法与时间边界

- 每次只读取形成时刻之前已经完整关闭的UTC日线；日成交额过去28日均值至少5000万USDT，最多30币、每币至少120天历史。当天与未来价格、成交额变化不能改变过去形成结果。
- 对数价格两个方向分别调用正确的Engle–Granger协整检验，固定1阶滞后与截距；对全部声明的双向检验做5% BH，再要求Johansen恰好1个协整关系、beta在0.25至4、价格相关至少0.7、残差半衰期0.125至7天。
- 2倍训练残差标准差除以(1+beta)须超过0.6%，即两倍压力往返成本/单位总敞口；不是收益保证，也不是把两腿各当全部本金。按最差双向q、半衰期和币名排序，最多5组、币种不重复。
- 交易规则已登记为每4小时检查、2σ入场、0.5σ退出、4σ止损、最多7天；形成后系数不使用未来数据重定中心，月度换模型必须先处理旧库存。规则尚待双腿账户执行验证。

## 独立核验与警告

全部4625组币对、9250个p/q值重新计算；22次Johansen检验用独立协方差构造和对称广义特征值解重新核对。原库出现complex转real警告，复核全部特征值虚部为0，trace最大差异约6.57e-13，秩判断一致，不能只忽略警告。8个入选回归的独立beta最大差约1.22e-15。真实数据3个截断/当前及未来扰动检查通过。完整记录见 [independent_audit.json](independent_audit.json)。

运行使用Python3.12、statsmodels0.14.5、pandas2.2.3、NumPy2.5.4与SciPy1.18.1。statsmodels版本不兼容工程pandas3，因此研究路径单独复用内置pandas2.2.3，未替换生产依赖。完整原始结果在TradingAgents `results/research/pair_formation_20261026_v001`，本次快照摘要见 [summary.json](summary.json)。

## 来源与采用范围

固定7份源码/许可记录，摘要见 [source_review.json](source_review.json)。

- [Jansen配对方法](https://github.com/stefan-jansen/machine-learning-for-trading/blob/30db4f00946ce91fbc98fef18582b8cbc3fd8afd/09_model_based_features/14_panel_features.py)：MIT；只参考协整、形成期与回归速度，ETF示例不证明本币对盈利。
- [je-suis-tm配对回测](https://github.com/je-suis-tm/quant-trading/blob/611b73f2c3f577ac5b28aaa19ac8c43d3236c7a5/Pair%20trading%20backtest.py)：Apache2；发现用全样本最高价分配股数及普通残差ADF等假设，不复用其收益或仓位。
- [statsmodels 0.14.5](https://github.com/statsmodels/statsmodels/tree/1107ea567121b80f90562d6c72085eed7e882113)：BSD3，归档正确协整检验实现。官方文档域超时后改为同版官方GitHub源码/文档字符串，未完成v001/v002目录保留，不伪造成功。
- [加密币对研究论文](https://arxiv.org/abs/2109.10662)、[Copula与协整论文](https://arxiv.org/abs/2305.06961)：本轮查阅摘要作为机制与成交敏感性线索，未复现整篇方法与作者收益。

## 尚未完成

扣费收益、净胜率、回撤、夏普与容量均为空，不能填0或沿用第二轮数值作为币对结果。下一步在Freqtrade研究目录实现双腿独立次根5分钟成交、实际资金费、失败对冲退出、完整配对生命周期与JSON恢复审计；随后在最多26账户预算内对比2025、2026冷启动、连续608日及10万/100万容量。统计关系通过而执行/成本失败的币对仍应淘汰。

40币来自事后选定池，存在幸存者偏差；2025/2026历史已经查看，不称新的独立样本外。当前无盘口历史，未来容量只能先报告成交额参与率代理。股票可以复用统一目录的数学定义，但股票交易日历、股息/拆股、借券成本与账户仍需另行验证。
"""
    )
    (output / "README.md").write_text(text)
    files = sorted(p.name for p in output.iterdir() if p.is_file())
    save(
        output / "completion.json",
        {
            "manifest": {n: sha(output / n) for n in files},
            "source_formation_completion_sha256": sha(root / "completion.json"),
            "audit_passed": True,
            "account_returns_computed": False,
            "profit_verified": False,
        },
    )
    print({"published": str(output), "financial_metrics": "not_computed"}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    publish(args.root.resolve(), args.audit.resolve(), args.output.resolve())
