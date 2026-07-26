from mg2si.evaluation.grouped_splits import assert_no_group_leakage, grouped_splits


def test_group_split_has_no_leakage():
    groups = ["a", "a", "b", "b", "c", "c"]
    splits = grouped_splits(groups, folds=3)
    assert_no_group_leakage(groups, splits)

