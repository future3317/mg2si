import pandas as pd

from mg2si.config import load_config


def validate_candidates(frame: pd.DataFrame) -> list[str]:
    design = load_config("design_space.yaml")
    errors = []
    for branch in ("synthetic", "commercial"):
        subset = frame[frame["workflow_branch"].eq(branch)]
        if subset.empty:
            continue
        if not subset["synthesis_required"].eq(int(branch == "synthetic")).all():
            errors.append(f"{branch}: synthesis_required contradiction")
        config = design[branch]
        for field in config.get("required", []):
            if field not in subset or subset[field].isna().any():
                errors.append(f"{branch}: required field missing: {field}")
        for field in config.get("forbidden", []):
            if field in subset and subset[field].notna().any():
                errors.append(f"{branch}: forbidden field populated: {field}")
        for field, bounds in config["continuous"].items():
            if field in subset:
                values = pd.to_numeric(subset[field], errors="coerce").dropna()
                if not values.between(bounds[0], bounds[1]).all():
                    errors.append(f"{branch}: {field} outside configured bounds")
    return errors

