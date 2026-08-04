from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "configs"
DATA_ROOT = PROJECT_ROOT / "data"
PROCESSED_ROOT = DATA_ROOT / "processed"
REPORT_ROOT = PROJECT_ROOT / "docs" / "assets"


def load_config(name: str, root: Path | None = None) -> dict[str, Any]:
    path = (root or CONFIG_ROOT) / name
    if not path.exists():
        raise FileNotFoundError(f"Missing configuration: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return data
