"""验证量化研究独立运行、冻结阶段和结果接续，不使用模型或网络。"""

import ast
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradingagents.research.quant import study
from tradingagents.research.quant.artifacts import read, save, sha, verify_receipts
from tradingagents.research.quant.regression import FEATURES


def panel():
    """人工预测因子仅用于验证时序边界，不充当真实市场收益证据。"""
    generator = np.random.default_rng(1729)
    parts = []
    for phase, day in (("training", "2026-10-01"), ("validation", "2026-10-02")):
        frame = pd.DataFrame(generator.normal(size=(40, len(FEATURES))), columns=FEATURES)
        frame["symbol"] = [f"A{i:03}" for i in range(40)]
        frame["asset_class"] = "人工类别"
        frame["clock_ms"] = pd.Timestamp(day, tz="UTC").value // 1_000_000
        for horizon in (20, 60):
            frame[f"target_{horizon}"] = frame["return_3m"] * 0.01
            frame[f"phase_{horizon}"] = phase
        parts.append(frame)
    return pd.concat(parts, ignore_index=True), {
        "phases": {
            "training": {"start": "2026-10-01T00:00:00Z", "end": "2026-10-02T00:00:00Z"},
            "validation": {"start": "2026-10-02T00:00:00Z", "end": "2026-10-03T00:00:00Z"},
        },
        "leader_labels_read": False,
        "orders_enabled": False,
    }


def test_replay_seals_self_contained_panel_and_never_grants_production(tmp_path, monkeypatch):
    samples, protocol = panel()
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    save(source / "protocol.json", protocol)
    save(source / "input_identities.json", {})
    save(source / "coverage.json", {"人工样本": True})
    (source / "hourly_samples.parquet").write_bytes("人工测试面板，实际数据另行端到端复核".encode())
    save(source / "receipts.json", {p.name: sha(p) for p in source.iterdir()})
    before = read(source / "receipts.json")
    # 常规测试隔离文件格式库；真实 Parquet 与完整历史数据在迁移验收中验证。
    monkeypatch.setattr(study.pd, "read_parquet", lambda _: samples)
    monkeypatch.setattr(study.importlib.metadata, "version", lambda _: "人工运行库身份")
    result = study.run(source, output)
    verify_receipts(output)
    assert result["research_owner"] == "TradingAgents"
    assert result["production_eligible"] is result["independent_oos"] is False
    for name, checksum in before.items():
        assert sha(source / name) == checksum
    assert sha(output / "hourly_samples.parquet") == sha(source / "hourly_samples.parquet")
    assert read(output / "protocol.json")["phases"] == protocol["phases"]
    with pytest.raises(FileExistsError):
        study.run(source, output)


@pytest.mark.parametrize(
    "damage", ["duplicate", "future", "mixed", "hidden", "overlap", "labels", "naive"]
)
def test_bad_phase_or_identity_is_rejected(damage):
    samples, protocol = panel()
    protocol = deepcopy(protocol)
    if damage == "duplicate":
        samples = pd.concat([samples, samples.iloc[:1]], ignore_index=True)
    elif damage == "future":
        samples.loc[:39, "clock_ms"] += 23 * 3_600_000
    elif damage == "mixed":
        samples.loc[0, "phase_20"] = "validation"
    elif damage == "hidden":
        samples.loc[:39, "phase_20"] = "excluded_boundary"
    elif damage == "overlap":
        protocol["phases"]["training"]["end"] = "2026-10-02T01:00:00Z"
    elif damage == "labels":
        protocol["leader_labels_read"] = True
    else:
        protocol["phases"]["training"]["start"] = "2026-10-01"
    with pytest.raises(ValueError):
        study.validate_panel(samples, protocol)


def test_quant_engine_imports_neither_cz_nor_model_frameworks(tmp_path):
    root = Path(__file__).resolve().parents[1]
    code = """
import importlib.abc
import sys
sys.path.insert(0, sys.argv[1])
class Refuse(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in {'cz', 'langchain', 'langchain_core', 'langgraph'}:
            raise AssertionError('量化核心不应加载交易项目或模型框架：' + name)
sys.meta_path.insert(0, Refuse())
from tradingagents.research.quant import incremental, linear_family, response_study, study
from tradingagents.dataflows.research import holdout, uniform_inputs
from tradingagents.research.quant.backtesting import forecast, paper_live
assert len(incremental.MODELS) == 7
assert study.evaluate.__module__ == 'tradingagents.research.quant.regression'
assert len(response_study.MODELS) == 9
assert len(forecast.TESTED_MODELS) == 4
assert linear_family.evaluate.__module__ == 'tradingagents.research.quant.linear_family'
"""
    subprocess.run(
        [sys.executable, "-I", "-c", code, str(root)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    for path in (root / "tradingagents/research/quant").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("cz.")
            elif isinstance(node, ast.Import):
                assert not any(name.name.split(".")[0] == "cz" for name in node.names)


def test_sealed_input_rejects_extra_files_and_symlink_escape(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    (root / "escaped.json").symlink_to(outside)
    save(root / "receipts.json", {"escaped.json": sha(outside)})
    with pytest.raises(ValueError):
        verify_receipts(root)
