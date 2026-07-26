from pathlib import Path

import pandas as pd

from mg2si.config import PROJECT_ROOT, load_config


def validate_workbook(path: Path, required_sheets: list[str]) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Configured workbook does not exist: {path}")
    missing = sorted(set(required_sheets) - set(pd.ExcelFile(path).sheet_names))
    if missing:
        raise ValueError(f"{path.name} is missing configured sheets: {missing}")


def resolve_sources(root: Path = PROJECT_ROOT) -> tuple[Path, Path]:
    sources = load_config("data_sources.yaml")["sources"]
    material = root / sources["material_workbook"]["path"]
    biology = root / sources["biology_workbook"]["path"]
    validate_workbook(material, sources["material_workbook"]["required_sheets"])
    validate_workbook(biology, sources["biology_workbook"]["required_sheets"])
    return material, biology

