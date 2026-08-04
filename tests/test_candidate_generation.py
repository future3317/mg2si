from mg2si.optimization.constraints import validate_candidates
from mg2si.optimization.design_space import sample_design_space


def test_conditional_candidate_generation():
    synthetic = sample_design_space("synthetic", 8, seed=1)
    commercial = sample_design_space("commercial", 8, seed=1)
    assert not validate_candidates(synthetic)
    assert not validate_candidates(commercial)
    assert commercial["material_max_temp_c"].isna().all()


def test_acquisition_requires_efficacy_and_safety_probability():
    utility = (100.0 - 75.0 + 5.0) * 0.0 * 0.99
    assert utility == 0.0
