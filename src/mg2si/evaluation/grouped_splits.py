import numpy as np
from sklearn.model_selection import GroupKFold


def grouped_splits(groups, folds: int = 5):
    values = np.asarray(groups)
    unique = np.unique(values.astype(str))
    if len(unique) < 2:
        raise ValueError("At least two non-overlapping groups are required.")
    splitter = GroupKFold(n_splits=min(folds, len(unique)))
    indices = np.arange(len(values))
    return list(splitter.split(indices, groups=values))


def assert_no_group_leakage(groups, splits) -> None:
    values = np.asarray(groups).astype(str)
    for train, test in splits:
        overlap = set(values[train]) & set(values[test])
        if overlap:
            raise AssertionError(f"Group leakage detected: {sorted(overlap)[:3]}")

