from itertools import product
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from mg2si.config import load_config


def factor_levels(branch: str) -> dict[str, list]:
    """Return the prospective discrete levels for a workflow branch."""
    config = load_config("design_space.yaml")
    if branch not in {"synthetic", "commercial"}:
        raise ValueError(f"Unsupported workflow branch: {branch}")
    branch_config = config[branch]
    levels = dict(branch_config.get("factor_levels", {}))
    for field, choices in branch_config.get("categorical", {}).items():
        levels.setdefault(field, choices)
    for field in branch_config.get("forbidden", []):
        levels.pop(field, None)
    if not levels:
        raise ValueError(f"No factor levels configured for {branch}")
    if any(not values for values in levels.values()):
        raise ValueError(f"Empty factor level list in {branch}")
    return levels


def full_factorial_size(branch: str) -> int:
    levels = factor_levels(branch)
    size = 1
    for values in levels.values():
        size *= len(values)
    return size


def _unrank(index: int, sizes: list[int]) -> list[int]:
    coordinates = [0] * len(sizes)
    remainder = index
    for position in range(len(sizes) - 1, -1, -1):
        remainder, coordinates[position] = divmod(remainder, sizes[position])
    return coordinates


def enumerate_design_space(branch: str, max_points: int | None = 10000, seed: int = 20260726) -> pd.DataFrame:
    """Enumerate or deterministically sample the configured full-factorial space.

    The configuration defines the complete Cartesian product.  When the product
    is larger than ``max_points``, this function emits a reproducible subset and
    records the full count in every row rather than pretending the table is
    complete.
    """
    config = load_config("design_space.yaml")
    branch_config = config[branch]
    levels = factor_levels(branch)
    fields = list(levels)
    sizes = [len(levels[field]) for field in fields]
    total = full_factorial_size(branch)
    if max_points is None or total <= max_points:
        indexes = np.arange(total, dtype=np.int64)
    else:
        rng = np.random.default_rng(seed)
        indexes = np.sort(rng.choice(total, size=max_points, replace=False))

    rows = []
    for index in indexes.tolist():
        coordinates = _unrank(int(index), sizes)
        row = {field: levels[field][coordinate] for field, coordinate in zip(fields, coordinates)}
        for field in branch_config.get("forbidden", []):
            row[field] = np.nan
        row.update({
            "candidate_id": f"prospective_{branch}_{int(index):010d}",
            "candidate_index": int(index),
            "workflow_branch": branch,
            "synthesis_required": int(branch == "synthetic"),
            "candidate_source": "prospective_full_factorial",
            "candidate_space_type": "full_factorial" if total <= len(indexes) else "full_factorial_sample",
            "full_factorial_count": int(total),
            "candidate_space_complete": bool(total <= len(indexes)),
            "synthesis_parameter_applicability": "applicable" if branch == "synthetic" else "not_applicable",
            "not_applicable_fields": ",".join(branch_config.get("forbidden", [])),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def build_prospective_design_space(
    database: Path,
    branch: str | None = None,
    max_points: int = 10000,
    seed: int = 20260726,
) -> dict:
    branches = [branch] if branch else ["synthetic", "commercial"]
    frames = [enumerate_design_space(item, max_points=max_points, seed=seed) for item in branches]
    candidates = pd.concat(frames, ignore_index=True)
    summary_rows = []
    for item in branches:
        total = full_factorial_size(item)
        generated = int((candidates["workflow_branch"] == item).sum())
        summary_rows.append({
            "workflow_branch": item,
            "full_factorial_count": total,
            "generated_count": generated,
            "complete_materialization": int(total <= max_points),
            "factor_fields": ",".join(factor_levels(item)),
            "space_type": "prospective_full_factorial",
        })
    summary = pd.DataFrame(summary_rows)
    with sqlite3.connect(database) as connection:
        candidates.to_sql("prospective_design_space", connection, if_exists="replace", index=False)
        summary.to_sql("prospective_design_space_summary", connection, if_exists="replace", index=False)
    return {
        "status": "ok",
        "branches": branches,
        "generated_points": int(len(candidates)),
        "full_factorial_counts": {item: full_factorial_size(item) for item in branches},
        "complete_materialization": all(full_factorial_size(item) <= max_points for item in branches),
        "output": f"{database}#prospective_design_space",
    }


def sample_design_space(branch: str, size: int, seed: int = 20260726) -> pd.DataFrame:
    config = load_config("design_space.yaml")
    if branch not in {"synthetic", "commercial"}:
        raise ValueError(f"Unsupported workflow branch: {branch}")
    branch_config = config[branch]
    rng = np.random.default_rng(seed)
    continuous = branch_config["continuous"]
    rows = np.empty((size, len(continuous)))
    for index, (_, bounds) in enumerate(continuous.items()):
        strata = (rng.permutation(size) + rng.random(size)) / size
        rows[:, index] = bounds[0] + strata * (bounds[1] - bounds[0])
    frame = pd.DataFrame(rows, columns=list(continuous))
    for field, choices in branch_config.get("categorical", {}).items():
        frame[field] = rng.choice(choices, size=size)
    for field in branch_config.get("forbidden", []):
        frame[field] = np.nan
    frame["workflow_branch"] = branch
    frame["synthesis_required"] = int(branch == "synthetic")
    frame["candidate_id"] = [f"candidate_{branch}_{index:04d}" for index in range(size)]
    return frame
