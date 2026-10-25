import argparse
import json
from pathlib import Path

from .compose import compose
from .io import read
from .study import train


def main():
    p = argparse.ArgumentParser(
        description="Registered factor research and interactive composition, no trading"
    )
    p.add_argument("action", choices=["study", "compose"])
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--config", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if a.action == "study":
        result = train(a.run.resolve())
    else:
        if not a.config or not a.output:
            p.error("compose requires --config and --output")
        result = compose(a.run.resolve(), read(a.config), a.output.resolve())
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
