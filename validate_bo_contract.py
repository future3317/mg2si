"""Validate the local, generated BO artifacts without exposing research data.

This is intentionally a schema/constraint check rather than a statistical test.
It catches the failure mode where a candidate looks numerically valid but lacks
an executable material-processing branch.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent


def read_required(name: str) -> pd.DataFrame:
    path = ROOT / name
    if not path.exists():
        raise FileNotFoundError(f"Missing generated artifact: {name}")
    return pd.read_csv(path, encoding="utf-8-sig")


def find_column(frame: pd.DataFrame, *names: str) -> str | None:
    normalized = {str(column).strip().lower(): str(column) for column in frame.columns}
    for name in names:
        if name.lower() in normalized:
            return normalized[name.lower()]
    return None


def validate_branch_contract(candidates: pd.DataFrame) -> None:
    branch_column = find_column(candidates, "workflow_branch")
    synthesis_column = find_column(candidates, "synthesis_required")
    if branch_column is None or synthesis_column is None:
        raise ValueError(
            "Candidate output must contain workflow_branch and synthesis_required"
        )

    branches = set(candidates[branch_column].dropna().astype(str))
    invalid = branches - {"synthetic", "commercial"}
    if invalid:
        raise ValueError(f"Unsupported workflow branches: {sorted(invalid)}")

    for _, row in candidates.iterrows():
        branch = str(row[branch_column])
        required = int(row[synthesis_column])
        expected = 1 if branch == "synthetic" else 0
        if required != expected:
            raise ValueError(
                f"Branch mismatch at candidate row {_}: "
                f"{branch} requires synthesis_required={expected}"
            )


def validate_required_process_fields(candidates: pd.DataFrame) -> None:
    columns = {str(column).strip().lower() for column in candidates.columns}
    groups = {
        "milling": ("milling", "ball_mill", "ball_to_material"),
        "particle_size": ("particle_size", "milled_size", "grain_size", "dls_size"),
    }
    missing = [
        label
        for label, tokens in groups.items()
        if not any(any(token in column for token in tokens) for column in columns)
    ]
    if missing:
        raise ValueError(
            "Candidate output is missing executable process fields: "
            + ", ".join(missing)
        )


def main() -> None:
    joint = read_required("bo_joint_dataset.csv")
    features = read_required("bo_features.csv")
    targets = read_required("bo_targets.csv")
    candidates = read_required("mobo_demo_virtual_candidates.csv")

    if len(features) != len(targets):
        raise ValueError("bo_features.csv and bo_targets.csv have different row counts")
    if len(joint) == 0 or len(features) == 0 or len(candidates) == 0:
        raise ValueError("BO artifacts must not be empty")

    validate_branch_contract(candidates)
    validate_required_process_fields(candidates)

    print(
        {
            "joint_rows": len(joint),
            "feature_rows": len(features),
            "target_rows": len(targets),
            "candidate_rows": len(candidates),
            "candidate_branches": candidates["workflow_branch"]
            .value_counts(dropna=False)
            .to_dict(),
            "status": "ok",
        }
    )


if __name__ == "__main__":
    main()
