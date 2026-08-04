from mg2si.optimization.constraints import validate_candidates
from mg2si.optimization.design_space import enumerate_design_space, full_factorial_size


def test_prospective_space_reports_full_factorial_size_and_branch_conditionality():
    synthetic = enumerate_design_space("synthetic", max_points=32, seed=7)
    commercial = enumerate_design_space("commercial", max_points=32, seed=7)

    assert len(synthetic) == 32
    assert len(commercial) == 32
    assert synthetic["full_factorial_count"].eq(full_factorial_size("synthetic")).all()
    assert commercial["full_factorial_count"].eq(full_factorial_size("commercial")).all()
    assert commercial["material_max_temp_c"].isna().all()
    assert commercial["synthesis_parameter_applicability"].eq("not_applicable").all()
    assert not validate_candidates(synthetic)
    assert not validate_candidates(commercial)

