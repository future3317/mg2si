from __future__ import annotations

from pathlib import Path
import sqlite3

import pandas as pd

from mg2si.config import PROJECT_ROOT


DEFAULT_DATABASE = PROJECT_ROOT / "data" / "processed" / "mg2si.sqlite"


def connect(path: Path = DEFAULT_DATABASE) -> sqlite3.Connection:
    if not path.exists():
        raise FileNotFoundError(f"Database does not exist: {path}. Run `mg2si ingest` first.")
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def read_training_view(path: Path = DEFAULT_DATABASE) -> pd.DataFrame:
    with connect(path) as connection:
        return pd.read_sql_query("SELECT * FROM bo_training", connection)


def write_table(frame: pd.DataFrame, table: str, path: Path = DEFAULT_DATABASE) -> None:
    with connect(path) as connection:
        frame.to_sql(table, connection, if_exists="replace", index=False)


def remove_legacy_csvs(root: Path = PROJECT_ROOT) -> list[str]:
    patterns = (
        "bo_*.csv",
        "mobo_*.csv",
        "model_evaluation.csv",
        "sample_id_mapping_review.csv",
    )
    targets: set[Path] = set()
    resolved_root = root.resolve()
    for pattern in patterns:
        targets.update(path for path in root.glob(pattern) if path.is_file())
    removed = []
    for path in sorted(targets):
        if path.resolve().parent != resolved_root:
            raise ValueError(f"Refusing to remove file outside project root: {path}")
        path.unlink()
        removed.append(path.name)
    return removed

