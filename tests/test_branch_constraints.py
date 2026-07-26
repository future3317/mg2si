import numpy as np
import pandas as pd

from mg2si.optimization.constraints import validate_candidates


def test_commercial_forbids_shs_fields():
    frame = pd.DataFrame([{"workflow_branch": "commercial", "synthesis_required": 0, "material_milling_cycle_time": 60, "material_ball_to_material_ratio": 10, "material_max_temp_c": np.nan}])
    assert validate_candidates(frame) == []
    frame.loc[0, "material_max_temp_c"] = 700
    assert any("forbidden" in error for error in validate_candidates(frame))

