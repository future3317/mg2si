import pandas as pd


def target_profile(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in candidates.columns:
        if column.startswith("predicted_material_") and column.endswith("_mean"):
            base = column.removeprefix("predicted_").removesuffix("_mean")
            rows.append({"material_indicator": base, "recommended_lower": candidates.get(f"predicted_{base}_lower", pd.Series(dtype=float)).min(), "recommended_upper": candidates.get(f"predicted_{base}_upper", pd.Series(dtype=float)).max(), "evidence": "posterior_from_project_data"})
    return pd.DataFrame(rows)

