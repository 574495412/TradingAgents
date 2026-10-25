import hashlib
import json
from pathlib import Path


def read(path):
    def invalid(value):
        raise ValueError(f"Nonfinite JSON: {value}")

    return json.loads(Path(path).read_text(), parse_constant=invalid)


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
