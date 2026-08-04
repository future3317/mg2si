from __future__ import annotations

from math import erf, sqrt
from pathlib import Path
import re
import sqlite3

import numpy as np
import pandas as pd

from mg2si.config import load_config
from mg2si.models.botorch_surrogate import BotorchSurrogate


def _normal_cdf(values: np.ndarray) -> np.ndarray:
    return np.vectorize(lambda value: 0.5 * (1.0 + erf(value / sqrt(2.0))))(values)


def _parent_id(material_id: str, fraction_parents: dict[str, str]) -> str:
    if material_id in fraction_parents:
        return fraction_parents[material_id]
    match = re.match(r"^(MS-\d{6}-SHS)\s+(top|down|mid)$", material_id, flags=re.IGNORECASE)
    return match.group(1) if match else material_id


def _is_synthetic(material_id: str, known_synthetic: set[str], fraction_parents: dict[str, str]) -> bool:
    return material_id in known_synthetic or _parent_id(material_id, fraction_parents) in known_synthetic or "-SHS" in material_id


def candidate_status(is_observed: bool, has_anchor: bool, concentration: float, model_supported: bool) -> str:
    if is_observed:
        return "already_measured"
    if model_supported:
        return "model_supported"
    if concentration == 125:
        return "screening_anchor_required"
    if has_anchor:
        return "await_feature_completion"
    return "await_125ppm_anchor"


def _read_database(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with sqlite3.connect(path) as connection:
        training = pd.read_sql_query("SELECT * FROM bo_training", connection)
        material = pd.read_sql_query("SELECT * FROM material", connection)
        fractions = pd.read_sql_query("SELECT * FROM supplement_particle_fraction", connection)
    return training, material, fractions


def explore_real_space(
    dataset: Path,
    branch: str,
    tumor_cell_line: str,
    normal_cell_line: str,
) -> dict:
    if dataset.suffix.lower() not in {".sqlite", ".db"}:
        raise ValueError("Real candidate exploration requires the local SQLite database.")
    training, material, fractions = _read_database(dataset)
    scoped = training[
        training["workflow_branch"].eq(branch)
        & training["tumor_cell_line"].eq(tumor_cell_line)
        & training["normal_cell_line"].eq(normal_cell_line)
        & training["model_eligible_direct"].eq(1)
    ].dropna(subset=["y_tumor_viability_pct", "y_normal_viability_pct"]).copy()
    if scoped.empty:
        raise ValueError("No complete direct observations exist in the requested scope.")

    fraction_parents = dict(zip(fractions["material_id"], fractions["material_parent_id"])) if not fractions.empty else {}
    known_synthetic = set(scoped["material_id"].dropna().astype(str)) | set(scoped["material_parent_id"].dropna().astype(str))
    material = material[material["material_id"].astype(str).map(lambda value: _is_synthetic(value, known_synthetic, fraction_parents))].copy()
    material["candidate_material_id"] = material["material_id"].astype(str)
    material["candidate_parent_id"] = material["candidate_material_id"].map(lambda value: _parent_id(value, fraction_parents))

    schema = load_config("schema.yaml")["roles"]
    variable_fields = []
    bounds: dict[str, tuple[float, float]] = {}
    for field in schema["controllable_x"]:
        if field not in scoped:
            continue
        values = pd.to_numeric(scoped[field], errors="coerce").dropna()
        if values.nunique() >= 2:
            variable_fields.append(field)
            bounds[field] = (float(values.min()), float(values.max()))
    if "concentration_ppm" not in variable_fields:
        raise ValueError("Concentration must have at least two observed levels for real-space exploration.")

    feature_columns = [field.removeprefix("material_") for field in variable_fields if field.startswith("material_")]
    candidates = material[["candidate_material_id", "candidate_parent_id"] + feature_columns].copy()
    candidates = candidates.rename(columns={column: f"material_{column}" for column in feature_columns})
    doses = np.array(sorted(pd.to_numeric(scoped["concentration_ppm"], errors="coerce").dropna().unique()), dtype=float)
    candidates = candidates.merge(pd.DataFrame({"concentration_ppm": doses}), how="cross")
    candidates["workflow_branch"] = branch
    candidates["synthesis_required"] = int(branch == "synthetic")
    candidates["tumor_cell_line"] = tumor_cell_line
    candidates["normal_cell_line"] = normal_cell_line
    candidates["candidate_id"] = [f"real_{material_id}_{int(dose)}ppm" for material_id, dose in zip(candidates["candidate_material_id"], candidates["concentration_ppm"])]

    observed = set(zip(scoped["material_id"].fillna(""), scoped["concentration_ppm"].astype(float)))
    anchored = set(scoped.loc[scoped["concentration_ppm"].eq(125), "material_id"].dropna().astype(str))
    candidates["observed_in_scope"] = [
        (material_id, float(dose)) in observed
        for material_id, dose in zip(candidates["candidate_material_id"], candidates["concentration_ppm"])
    ]
    candidates["has_125ppm_anchor"] = candidates["candidate_material_id"].isin(anchored)

    reasons = []
    supported = []
    for _, row in candidates.iterrows():
        row_reasons = []
        for field in variable_fields:
            value = pd.to_numeric(pd.Series([row.get(field)]), errors="coerce").iloc[0]
            lower, upper = bounds[field]
            if pd.isna(value):
                row_reasons.append(f"missing:{field}")
            elif value < lower or value > upper:
                row_reasons.append(f"outside:{field}")
        reasons.append(";".join(row_reasons) if row_reasons else "in_training_domain")
        supported.append(not row_reasons)
    candidates["domain_status"] = reasons
    candidates["model_supported"] = supported
    candidates["recommendation_status"] = [
        candidate_status(observed, anchor, dose, support)
        for observed, anchor, dose, support in zip(
            candidates["observed_in_scope"],
            candidates["has_125ppm_anchor"],
            candidates["concentration_ppm"],
            candidates["model_supported"],
        )
    ]

    candidates["predicted_tumor_viability_mean"] = np.nan
    candidates["predicted_normal_viability_mean"] = np.nan
    candidates["probability_efficacy"] = np.nan
    candidates["probability_safety"] = np.nan
    candidates["acquisition_value"] = np.nan
    candidates["target_feasibility"] = "not_predicted"
    predict_mask = candidates["model_supported"] & ~candidates["observed_in_scope"]
    if predict_mask.any():
        tumor = BotorchSurrogate().fit(scoped[variable_fields], scoped["y_tumor_viability_pct"])
        normal = BotorchSurrogate().fit(scoped[variable_fields], scoped["y_normal_viability_pct"])
        tumor_mean, tumor_std = tumor.predict(candidates.loc[predict_mask, variable_fields])
        normal_mean, normal_std = normal.predict(candidates.loc[predict_mask, variable_fields])
        candidates.loc[predict_mask, "predicted_tumor_viability_mean"] = tumor_mean
        candidates.loc[predict_mask, "predicted_normal_viability_mean"] = normal_mean
        objective = load_config("objectives.yaml")
        candidates.loc[predict_mask, "probability_efficacy"] = _normal_cdf(
            (float(objective["efficacy"]["threshold_pct"]) - tumor_mean) / np.maximum(tumor_std, 1e-6)
        )
        candidates.loc[predict_mask, "probability_safety"] = _normal_cdf(
            (normal_mean - float(objective["safety"]["threshold_pct"])) / np.maximum(normal_std, 1e-6)
        )
        candidates.loc[predict_mask, "acquisition_value"] = (
            (100.0 - tumor_mean + tumor_std)
            * candidates.loc[predict_mask, "probability_efficacy"]
            * candidates.loc[predict_mask, "probability_safety"]
        )
        candidates.loc[predict_mask, "target_feasibility"] = np.where(
            (candidates.loc[predict_mask, "probability_efficacy"] >= 0.5)
            & (candidates.loc[predict_mask, "probability_safety"] >= float(objective["safety"]["minimum_probability"])),
            "joint_target_plausible",
            "below_joint_target",
        )

    priority = {
        "model_supported": 0,
        "screening_anchor_required": 1,
        "await_125ppm_anchor": 2,
        "await_feature_completion": 3,
        "already_measured": 4,
    }
    candidates["priority_order"] = candidates["recommendation_status"].map(priority)
    candidates = candidates.sort_values(["priority_order", "acquisition_value", "candidate_id"], ascending=[True, False, True], na_position="last").reset_index(drop=True)
    summary = pd.DataFrame([
        {"metric": "scope_rows", "value": int(len(scoped))},
        {"metric": "real_materials", "value": int(material["candidate_material_id"].nunique())},
        {"metric": "real_candidate_points", "value": int(len(candidates))},
        {"metric": "variable_model_fields", "value": ",".join(variable_fields)},
        {"metric": "model_supported_unmeasured", "value": int((candidates["recommendation_status"] == "model_supported").sum())},
        {"metric": "joint_target_plausible", "value": int((candidates["target_feasibility"] == "joint_target_plausible").sum())},
        {"metric": "screening_anchor_required", "value": int((candidates["recommendation_status"] == "screening_anchor_required").sum())},
        {"metric": "await_125ppm_anchor", "value": int((candidates["recommendation_status"] == "await_125ppm_anchor").sum())},
    ])
    with sqlite3.connect(dataset) as connection:
        candidates.to_sql("real_candidate_space", connection, if_exists="replace", index=False)
        summary.to_sql("real_candidate_space_summary", connection, if_exists="replace", index=False)
    return {
        "status": "ok",
        "scope_rows": int(len(scoped)),
        "real_materials": int(material["candidate_material_id"].nunique()),
        "real_candidate_points": int(len(candidates)),
        "variable_model_fields": variable_fields,
        "model_supported_unmeasured": int((candidates["recommendation_status"] == "model_supported").sum()),
        "joint_target_plausible": int((candidates["target_feasibility"] == "joint_target_plausible").sum()),
        "screening_anchor_required": int((candidates["recommendation_status"] == "screening_anchor_required").sum()),
        "await_125ppm_anchor": int((candidates["recommendation_status"] == "await_125ppm_anchor").sum()),
        "output": f"{dataset}#real_candidate_space",
    }
