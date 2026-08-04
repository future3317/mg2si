from mg2si.optimization.real_space import candidate_status


def test_real_candidate_status_prioritizes_evidence_before_prediction():
    assert candidate_status(True, True, 250, True) == "already_measured"
    assert candidate_status(False, False, 125, False) == "screening_anchor_required"
    assert candidate_status(False, True, 250, False) == "await_feature_completion"
    assert candidate_status(False, False, 250, True) == "model_supported"
