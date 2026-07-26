from pathlib import Path

import yaml

from mg2si.config import CONFIG_ROOT


def load_aliases(path: Path | None = None) -> dict[str, list[str]]:
    with (path or CONFIG_ROOT / "aliases.yaml").open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return {alias: list(metadata.get("candidates", [])) for alias, metadata in raw.get("aliases", {}).items()}

