from dataclasses import asdict, dataclass
from pathlib import Path
import sqlite3

import pandas as pd

from mg2si.config import load_config


@dataclass
class QualityIssue:
    severity: str
    check: str
    evidence: str
    impact: str


def validate_dataset(path: Path) -> dict:
    frame = pd.read_csv(path, low_memory=False)
    schema = load_config("schema.yaml")
    issues: list[QualityIssue] = []
    missing = sorted(set(schema["required_model_context"]) - set(frame.columns))
    if missing:
        issues.append(QualityIssue("critical", "required_columns", str(missing), "Model scope cannot be defined."))
    for target in ("y_tumor_viability_pct", "y_normal_viability_pct"):
        if target in frame:
            values = pd.to_numeric(frame[target], errors="coerce").dropna()
            invalid = int((~values.between(0, 100)).sum())
            if invalid:
                issues.append(QualityIssue("high", f"{target}_range", f"{invalid} rows outside 0..100", "Invalid response scale."))
    if "normal_cell_line" in frame:
        rate = float(frame["normal_cell_line"].notna().mean())
        if rate < 0.8:
            issues.append(QualityIssue("high", "normal_cell_line_completeness", f"{rate:.1%} populated", "Safety model scope is ambiguous."))
    if "material_id" in frame:
        rate = float(frame["material_id"].notna().mean())
        if rate < 0.8:
            issues.append(QualityIssue("high", "material_mapping_coverage", f"{rate:.1%} mapped", "Process-state-biology joins lose evidence."))
    state = [column for column in schema["roles"]["material_state_z"] if column in frame]
    state_coverage = {column: float(frame[column].notna().mean()) for column in state}
    if state and max(state_coverage.values(), default=0.0) < 0.2:
        issues.append(QualityIssue("high", "material_state_coverage", "all state fields below 20%", "Two-stage model is not currently identifiable."))
    grain = ["experiment_id", "concentration_ppm"]
    if all(column in frame for column in grain):
        duplicate_rows = int(frame.duplicated(grain, keep=False).sum())
        if duplicate_rows:
            issues.append(QualityIssue("medium", "experiment_concentration_duplicates", f"{duplicate_rows} affected rows", "Replicates require an explicit replicate key."))
    severities = {issue.severity for issue in issues}
    status = "blocked" if "critical" in severities else ("not_model_ready" if "high" in severities else "ok")
    return {
        "status": status,
        "dataset": str(path),
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "intended_grain": "material/batch x experiment x cell-line pair x concentration",
        "state_coverage": state_coverage,
        "issues": [asdict(issue) for issue in issues],
    }


def validate_database(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Database does not exist: {path}")
    with sqlite3.connect(path) as connection:
        issues = pd.read_sql_query("SELECT * FROM quality_issue ORDER BY issue_id", connection)
        tables = pd.read_sql_query(
            "SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view') ORDER BY type, name",
            connection,
        )
        counts = {}
        for table in ("material", "bioassay", "bioassay_source_audit", "source_record", "source_conflict", "supplement_process_observation", "supplement_material_lineage", "supplement_index_reference", "material_input_coverage", "bioassay_condition_summary", "data_quality_profile", "prospective_design_space", "prospective_design_space_summary"):
            counts[table] = int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
        eligible = connection.execute(
            "SELECT SUM(model_eligible_direct), SUM(model_eligible_material) FROM bioassay"
        ).fetchone()
    severities = set(issues["severity"])
    status = "blocked" if "critical" in severities else ("not_model_ready" if "high" in severities else "ok")
    return {
        "status": status,
        "database": str(path),
        "objects": tables.to_dict("records"),
        "row_counts": counts,
        "model_eligible_direct": int(eligible[0] or 0),
        "model_eligible_material": int(eligible[1] or 0),
        "issues": issues.to_dict("records"),
    }
