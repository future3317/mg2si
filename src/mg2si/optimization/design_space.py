import numpy as np
import pandas as pd

from mg2si.config import load_config


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

