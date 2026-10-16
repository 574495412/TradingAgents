"""统一量化入口；重型计算、账户对照和复核均在 TradingAgents 中执行。"""

import argparse
import sys
from pathlib import Path


def main(argv=None):
    """按子命令延迟导入，纯量化重算不加载模型或 CZ-Agent。"""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "funding":
        from .funding_study import main as funding

        return funding(argv[1:])
    if argv and argv[0] == "comparison":
        from .comparison.__main__ import main as comparison

        return comparison(argv[1:])
    if argv and argv[0] == "forecast":
        from .backtesting.forecast import main as forecast

        return forecast(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("regression", "从封存面板重算IC、回归和固定训练预测"),
        ("incremental", "控制项基准与逐因子剔除检验"),
        ("audit", "独立重算封存回归结果"),
        ("response", "冻结价格响应候选并执行共同人口回归"),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--source", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    sub.add_parser("comparison", help="四组独立账户比较；使用 comparison --help 查看参数")
    sub.add_parser("forecast", help="固定训练预测的连续账户；使用 forecast --help 查看参数")
    sub.add_parser("funding", help="已结算资金费增量回归；使用 funding --help 查看参数")
    args = parser.parse_args(argv)
    if args.command == "regression":
        from .study import run
    elif args.command == "incremental":
        from .incremental import run
    elif args.command == "response":
        from .response_study import run
    else:
        from .regression_audit import run
    result = run(args.source, args.output)
    print(
        {
            "output": str(args.output),
            "research_owner": "TradingAgents",
            "groups": len(result.get("groups", {})),
            "production_eligible": False,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
