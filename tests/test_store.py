from pathlib import Path

from mg2si.data.store import remove_legacy_csvs


def test_cleanup_only_removes_known_derived_csvs(tmp_path: Path):
    (tmp_path / "bo_old.csv").write_text("x\n1\n", encoding="utf-8")
    (tmp_path / "mobo_old.csv").write_text("x\n1\n", encoding="utf-8")
    (tmp_path / "keep.csv").write_text("x\n1\n", encoding="utf-8")
    removed = remove_legacy_csvs(tmp_path)
    assert removed == ["bo_old.csv", "mobo_old.csv"]
    assert (tmp_path / "keep.csv").exists()
