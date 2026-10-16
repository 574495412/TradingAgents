"""命令行入口：导出公开数据、运行四组研究、复核封存账户。"""

import argparse
import sys
from pathlib import Path

from .account import Protocol
from .common import read, save
from .dataset import demo_dataset, public_export
from .experiment import run, verify


def main(argv=None):
    parser = argparse.ArgumentParser(description="TradingAgents 四组量化对照研究")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo-data", help="生成明确标记的合成联调数据")
    demo.add_argument("--output", type=Path, required=True)
    export = sub.add_parser("export", help="从已有官方历史归档导出公开输入")
    export.add_argument("--archive", type=Path, required=True)
    export.add_argument("--symbols", required=True)
    export.add_argument("--start", required=True)
    export.add_argument("--end", required=True)
    export.add_argument("--every-hours", type=int, default=24)
    export.add_argument("--evidence", type=Path, help="带 symbol 字段的公开证据 JSON 数组")
    export.add_argument("--output", type=Path, required=True)
    execute = sub.add_parser("run", help="冻结并运行四组对照")
    execute.add_argument("--dataset", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--tradingagents-root", type=Path, required=True)
    execute.add_argument("--tradingagents-python", type=Path, required=True)
    execute.add_argument("--backend", choices=("fixture", "codex"), default="fixture")
    execute.add_argument("--model", default=None)
    execute.add_argument(
        "--protocol", type=Path, help="事前固定的账户协议 JSON，可省略字段使用默认值"
    )
    execute.add_argument("--timeout-seconds", type=int, default=180)
    execute.add_argument("--resume", action="store_true")
    check = sub.add_parser("verify", help="无模型调用地复核八个基础/压力账户")
    check.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "demo-data":
        save(args.output, demo_dataset())
        print(f"合成数据已保存：{args.output}")
    elif args.command == "export":
        dataset = public_export(
            args.archive,
            args.symbols.split(","),
            args.start,
            args.end,
            args.every_hours,
            args.evidence,
        )
        save(args.output, dataset)
        print(
            f"已导出 {len(dataset['bars'])} 根 K 线、{len(dataset['snapshots'])} 个快照：{args.output}"
        )
    elif args.command == "run":
        result = run(
            args.dataset,
            args.output,
            args.tradingagents_root,
            args.tradingagents_python,
            args.backend,
            args.model,
            protocol=Protocol(**read(args.protocol)) if args.protocol else None,
            timeout_seconds=args.timeout_seconds,
            resume=args.resume,
        )
        print(f"四组研究完成，账户复核通过。结果：{args.output / 'result.json'}")
        print(
            f"真实模型推理：{result['model_inference_performed']}；独立样本外：False；生产资格：False"
        )
    else:
        print(verify(args.run))
    return 0


if __name__ == "__main__":
    sys.exit(main())
