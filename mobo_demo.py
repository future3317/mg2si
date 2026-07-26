from pathlib import Path

import numpy as np
import pandas as pd
from MultiBgolearn import bgo


ROOT = Path(__file__).resolve().parent
JOINT_PATH = ROOT / "bo_joint_dataset.csv"
OBSERVED_PATH = ROOT / "mobo_demo_observed.csv"
VIRTUAL_MATRIX_PATH = ROOT / "mobo_demo_virtual_space.csv"
VIRTUAL_RAW_PATH = ROOT / "mobo_demo_virtual_candidates.csv"
RECOMMENDATION_PATH = ROOT / "mobo_demo_recommendation.csv"


SYNTHESIS_NUMERIC = [
    "material_mg_mass_mg", "material_si_mass_mg", "material_molar_ratio_Mg_to_Si",
    "material_vacuum_cycle", "material_vacuum_time_min", "material_initial_pressure_atm",
    "material_max_temp_c", "material_hold_time_min",
]
PROCESS_NUMERIC = [
    "is_milled", "milled_size_nm", "pvp_mw", "material_to_pvp_ratio", "concentration_ppm",
    "material_ball_to_material_ratio", "material_milling_cycle_time", "material_pvp_mw",
    "material_pvp_material_to_pvp_ratio", "material_ultrasonic_time_h",
]
PROCESS_CATEGORICAL = [
    "post_treatment", "material_milling_mode", "material_post_treatment",
]
MATERIAL_CATEGORICAL = [
    "material_synthesis_method", "material_material_source", "material_protective_gas", "material_is_layered",
]


def encode_features(train, candidates, numeric_features, categorical_features):
    train_frame = train[numeric_features + categorical_features].copy()
    candidate_frame = candidates[numeric_features + categorical_features].copy()
    for column in numeric_features:
        train_frame[column] = pd.to_numeric(train_frame[column], errors="coerce")
        candidate_frame[column] = pd.to_numeric(candidate_frame[column], errors="coerce")
        train_frame[column + "__missing"] = train_frame[column].isna().astype(float)
        candidate_frame[column + "__missing"] = candidate_frame[column].isna().astype(float)
        non_missing = train_frame[column].dropna()
        fill_value = non_missing.median() if len(non_missing) else 0.0
        train_frame[column] = train_frame[column].fillna(fill_value)
        candidate_frame[column] = candidate_frame[column].fillna(fill_value)
    for column in categorical_features:
        train_frame[column] = train_frame[column].fillna("__missing__").astype(str)
        candidate_frame[column] = candidate_frame[column].fillna("__missing__").astype(str)
    combined = pd.concat([train_frame, candidate_frame], ignore_index=True)
    combined = pd.get_dummies(combined, columns=categorical_features, dtype=float)
    return combined.iloc[: len(train_frame)].reset_index(drop=True), combined.iloc[len(train_frame):].reset_index(drop=True)


def build_candidates(data):
    material_profiles = data[data["material_id"].notna()].copy()
    material_profiles = material_profiles.sort_values(["material_id", "workflow_branch"]).drop_duplicates(["material_id", "workflow_branch"])

    process_columns = [
        "is_milled", "milled_size_nm", "pvp_mw", "material_to_pvp_ratio", "post_treatment",
        "material_milling_mode", "material_post_treatment", "material_ball_to_material_ratio",
        "material_milling_cycle_time", "material_ultrasonic_time_h",
    ]
    process_profiles = data[process_columns].drop_duplicates().copy()
    process_profiles = process_profiles[process_profiles["is_milled"].notna() & process_profiles["post_treatment"].notna()]

    material_columns = [
        "material_id", "material_parent_id", "material_synthesis_method", "material_material_source",
        "material_is_layered", "material_protective_gas", "material_milling_mode", "material_post_treatment",
        *SYNTHESIS_NUMERIC, "material_ball_to_material_ratio", "material_milling_cycle_time",
        "material_pvp_mw", "material_pvp_material_to_pvp_ratio", "material_ultrasonic_time_h",
    ]
    rows = []
    for material_number, (_, material_row) in enumerate(material_profiles.iterrows()):
        branch = material_row["workflow_branch"]
        if branch not in {"synthetic", "commercial"}:
            continue
        if branch == "synthetic" and material_row[SYNTHESIS_NUMERIC].isna().any():
            continue
        for process_number, (_, process_row) in enumerate(process_profiles.iterrows()):
            for concentration in [125, 250, 500]:
                candidate = {column: material_row.get(column, np.nan) for column in material_columns}
                for column in process_columns:
                    value = process_row.get(column, np.nan)
                    if pd.notna(value):
                        candidate[column] = value
                candidate["workflow_branch"] = branch
                candidate["synthesis_required"] = 1 if branch == "synthetic" else 0
                candidate["synthesis_feature_status"] = "applicable" if branch == "synthetic" else "not_applicable"
                candidate["concentration_ppm"] = concentration
                if branch == "commercial":
                    for column in SYNTHESIS_NUMERIC + ["material_synthesis_method", "material_protective_gas"]:
                        candidate[column] = np.nan
                candidate["candidate_id"] = f"virtual_{branch}_{material_number:03d}_{process_number:03d}_{concentration}ppm"
                candidate["candidate_source"] = "conditional_material_branch_x_process_levels_x_concentration"
                rows.append(candidate)
    candidates = pd.DataFrame(rows)
    if candidates.empty:
        raise RuntimeError("没有生成候选，请检查材料映射和工艺字段")
    candidates = candidates[
        candidates["is_milled"].eq(1)
        & candidates["material_ball_to_material_ratio"].notna()
        & candidates["material_milling_cycle_time"].notna()
        & candidates["milled_size_nm"].notna()
    ].copy()
    candidates = candidates.drop_duplicates(
        ["workflow_branch", "material_id", "is_milled", "milled_size_nm", "pvp_mw", "material_to_pvp_ratio", "post_treatment", "concentration_ppm"]
    ).reset_index(drop=True)
    return candidates


def main():
    data = pd.read_csv(JOINT_PATH, low_memory=False)
    required_columns = {
        "workflow_branch", "synthesis_required", "synthesis_feature_status",
        "y_tumor_viability_pct", "y_normal_viability_pct", "tumor_cell_line",
        "material_id", *SYNTHESIS_NUMERIC, *PROCESS_NUMERIC,
    }
    missing_columns = sorted(required_columns - set(data.columns))
    if missing_columns:
        raise ValueError(
            "bo_joint_dataset.csv is not the branch-aware rebuild output; "
            f"missing columns: {missing_columns}"
        )
    data = data[
        data["y_tumor_viability_pct"].notna()
        & data["y_normal_viability_pct"].notna()
        & data["workflow_branch"].isin(["synthetic", "commercial"])
    ].copy()
    data["objective_tumor_kill_pct"] = 100.0 - data["y_tumor_viability_pct"]
    data["objective_normal_viability_pct"] = data["y_normal_viability_pct"]

    huh7 = data[data["tumor_cell_line"].eq("Huh-7")].copy()
    model_data = huh7 if len(huh7) >= 40 else data
    model_scope = "Huh-7" if len(huh7) >= 40 else "all_available_cell_lines"
    candidates = build_candidates(data)

    design_numeric = PROCESS_NUMERIC + SYNTHESIS_NUMERIC + ["synthesis_required"]
    design_categorical = PROCESS_CATEGORICAL + MATERIAL_CATEGORICAL + ["workflow_branch", "synthesis_feature_status"]
    design_fingerprint = design_numeric + design_categorical
    observed_fingerprints = set(model_data[design_fingerprint].fillna("__NA__").astype(str).agg("|".join, axis=1))
    candidate_fingerprints = candidates[design_fingerprint].fillna("__NA__").astype(str).agg("|".join, axis=1)
    candidates = candidates[~candidate_fingerprints.isin(observed_fingerprints)].reset_index(drop=True)
    if candidates.empty:
        raise RuntimeError("候选空间全部与历史设计点重复")

    # Branch constraints are explicit tests, not implicit missing-value behavior.
    assert set(candidates["workflow_branch"]) <= {"synthetic", "commercial"}
    assert (candidates.loc[candidates["workflow_branch"] == "synthetic", "synthesis_required"] == 1).all()
    assert (candidates.loc[candidates["workflow_branch"] == "commercial", "synthesis_required"] == 0).all()
    assert candidates.loc[candidates["workflow_branch"] == "synthetic", SYNTHESIS_NUMERIC].notna().all().all()
    assert candidates.loc[candidates["workflow_branch"] == "commercial", SYNTHESIS_NUMERIC].isna().all().all()
    milled = candidates["is_milled"].eq(1)
    assert milled.all()
    assert candidates.loc[milled, ["milled_size_nm", "material_ball_to_material_ratio", "material_milling_cycle_time"]].notna().all().all()

    requested_numeric = PROCESS_NUMERIC + SYNTHESIS_NUMERIC + ["synthesis_required"]
    numeric_features = [column for column in requested_numeric if data[column].notna().any() or candidates[column].notna().any()]
    categorical_features = design_categorical
    rng = np.random.RandomState(20260726)
    permutation = rng.permutation(len(model_data))
    train_size = max(40, int(len(model_data) * 0.75))
    train_data = model_data.iloc[permutation[:train_size]].reset_index(drop=True)
    train_x, virtual_x = encode_features(train_data, candidates, numeric_features, categorical_features)
    observed_dataset = train_x.copy()
    observed_dataset["objective_tumor_kill_pct"] = train_data["objective_tumor_kill_pct"]
    observed_dataset["objective_normal_viability_pct"] = train_data["objective_normal_viability_pct"]
    observed_dataset.to_csv(OBSERVED_PATH, index=False, encoding="utf-8-sig")
    virtual_x.to_csv(VIRTUAL_MATRIX_PATH, index=False, encoding="utf-8-sig")
    candidates.to_csv(VIRTUAL_RAW_PATH, index=False, encoding="utf-8-sig")

    _, improvements, recommended_index = bgo.fit(
        str(OBSERVED_PATH), str(VIRTUAL_MATRIX_PATH), object_num=2,
        max_search=True, method="EHVI", assign_model="GaussianProcess", bootstrap=5, batch_size=1,
    )
    recommended_index = int(np.asarray(recommended_index).reshape(-1)[0])
    selected = candidates.iloc[recommended_index].to_dict()
    selected.update({
        "recommended_virtual_index": recommended_index,
        "acquisition_value": float(np.asarray(improvements).reshape(-1)[recommended_index]),
        "model_scope": model_scope,
        "objective_1": "tumor_kill_pct_maximize",
        "objective_2": "normal_viability_pct_maximize",
        "candidate_is_historical_observation": 0,
    })
    pd.DataFrame([selected]).to_csv(RECOMMENDATION_PATH, index=False, encoding="utf-8-sig")
    print({
        "model_scope": model_scope,
        "complete_target_rows": int(len(model_data)),
        "observed_rows": int(len(train_data)),
        "generated_virtual_candidates": int(len(candidates)),
        "candidate_branch_counts": candidates["workflow_branch"].value_counts().to_dict(),
        "numeric_features_used": numeric_features,
        "recommended_candidate_id": selected["candidate_id"],
        "recommendation_file": str(RECOMMENDATION_PATH),
    })


if __name__ == "__main__":
    main()
