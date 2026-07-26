from hashlib import sha256
from pathlib import Path

import pandas as pd


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def workbook_manifest(path: Path, schema_version: str) -> dict:
    return {"path": str(path), "sha256": file_sha256(path), "sheet_names": pd.ExcelFile(path).sheet_names, "schema_version": schema_version}

