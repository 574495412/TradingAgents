"""Preserve invalid self-referential metadata and add corrected identity receipts."""

from pathlib import Path

from research_tools.factor_lab.io import read, save, sha


def repair(root):
    previous = root / "completion.json"
    prior = read(previous)
    assert prior["manifest"]["completion.json"] == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    for relative, digest in prior["manifest"].items():
        if relative != "completion.json":
            assert sha(root / relative) == digest, relative
    save(root / "manifest_repair.json", {
        "previous_receipt_sha256": sha(previous),
        "previous_receipt_defect": "The temporary sealing expression opened completion.json before building the manifest, so it included an empty-file self hash.",
        "original_bytes_preserved": True,
        "original_data_and_financial_accounts_unchanged": True,
        "corrected_receipt": "completion_v2.json",
        "repair_source_sha256": sha(Path(__file__)),
    })
    manifest = {str(p.relative_to(root)): sha(p) for p in root.rglob("*") if p.is_file()}
    assert "completion_v2.json" not in manifest
    save(root / "completion_v2.json", {
        "passed": prior.get("passed", True),
        "manifest": manifest,
        "identity_seal_only": True,
        "reporting_defect_repaired_by_other_directory": prior.get("replaced_by"),
        "original_bytes_preserved": True,
    })
    print({"root": str(root), "files_verified": len(manifest), "previous_bytes_preserved": True})


if __name__ == "__main__":
    for folder in [
        "data/research/monthly_tsmom_early_funding_20261027_v001",
        "data/research/monthly_tsmom_case_crosscheck_20261027_v001",
        "docs/research/regime_program/monthly_tsmom_20261027",
    ]:
        repair(Path(folder))
