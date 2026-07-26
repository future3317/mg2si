from math import erf, sqrt
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from mg2si.config import load_config
from mg2si.models.botorch_surrogate import BotorchSurrogate
from mg2si.models.material_centered import MaterialCenteredModel
from mg2si.optimization.constraints import validate_candidates
from mg2si.optimization.design_space import sample_design_space


def _normal_cdf(values):
    return np.vectorize(lambda value: 0.5 * (1.0 + erf(value / sqrt(2.0))))(values)


def _read_training(dataset: Path) -> pd.DataFrame:
    if dataset.suffix.lower() in {".sqlite", ".db"}:
        with sqlite3.connect(dataset) as connection:
            return pd.read_sql_query("SELECT * FROM bo_training", connection)
    return pd.read_csv(dataset, low_memory=False)


def _save_recommendations(frame: pd.DataFrame, dataset: Path, output: Path | None) -> str:
    if dataset.suffix.lower() in {".sqlite", ".db"}:
        with sqlite3.connect(dataset) as connection:
            frame.to_sql("recommendation", connection, if_exists="replace", index=False)
        return f"{dataset}#recommendation"
    if output is None:
        raise ValueError("CSV input requires an explicit output path.")
    frame.to_csv(output, index=False, encoding="utf-8-sig")
    return str(output)


def recommend(dataset: Path, output: Path | None, branch: str, tumor_cell_line: str, normal_cell_line: str, allow_direct_baseline: bool = False) -> dict:
    data = _read_training(dataset)
    objective = load_config("objectives.yaml")
    experiment = load_config("experiment.yaml")
    schema = load_config("schema.yaml")["roles"]
    scoped = data[
        data["workflow_branch"].eq(branch)
        & data["tumor_cell_line"].eq(tumor_cell_line)
        & data["normal_cell_line"].eq(normal_cell_line)
    ].copy()
    if "model_eligible_direct" in scoped:
        scoped = scoped[scoped["model_eligible_direct"].eq(1)]
    scoped = scoped.dropna(subset=["y_tumor_viability_pct", "y_normal_viability_pct"])
    minimum_rows = int(experiment["model"]["minimum_rows"])
    if len(scoped) < minimum_rows:
        result = {"status": "insufficient_evidence", "reason": f"{len(scoped)} complete rows; need {minimum_rows}", "scope": [branch, tumor_cell_line, normal_cell_line]}
        _save_recommendations(pd.DataFrame([result]), dataset, output)
        return result
    candidates = sample_design_space(branch, int(experiment["recommendation"]["pool_size"]), int(experiment["model"]["random_seed"]))
    candidates["tumor_cell_line"] = tumor_cell_line
    candidates["normal_cell_line"] = normal_cell_line
    exposure = pd.to_numeric(scoped["exposure_time_h"], errors="coerce").dropna() if "exposure_time_h" in scoped else pd.Series(dtype=float)
    candidates["exposure_time_h"] = exposure.median() if len(exposure) else np.nan
    x_fields = [
        field for field in schema["controllable_x"]
        if field in candidates and field in scoped and pd.to_numeric(scoped[field], errors="coerce").notna().any()
    ]
    if not x_fields:
        result = {"status": "insufficient_evidence", "reason": "No controllable field is populated in the selected scope", "scope": [branch, tumor_cell_line, normal_cell_line]}
        _save_recommendations(pd.DataFrame([result]), dataset, output)
        return result
    z_fields = [field for field in schema["material_state_z"] if field in scoped]
    context_fields = [field for field in ("concentration_ppm", "exposure_time_h") if field in candidates and field in scoped]
    material_model = MaterialCenteredModel(int(experiment["model"]["minimum_state_rows"])).fit(scoped, x_fields, z_fields, context_fields)
    model_path = "process_to_state_to_biology"
    if material_model.ready:
        prediction = material_model.predict(candidates, context_fields, int(experiment["model"]["posterior_samples"]), int(experiment["model"]["random_seed"]))
        tumor_mean, tumor_std = prediction["y_tumor_viability_pct"]["mean"], prediction["y_tumor_viability_pct"]["std"]
        normal_mean, normal_std = prediction["y_normal_viability_pct"]["mean"], prediction["y_normal_viability_pct"]["std"]
        for state, draws in prediction["state"].items():
            candidates[f"predicted_{state}_mean"] = draws.mean(axis=0)
            candidates[f"predicted_{state}_lower"] = np.quantile(draws, 0.1, axis=0)
            candidates[f"predicted_{state}_upper"] = np.quantile(draws, 0.9, axis=0)
    elif allow_direct_baseline:
        model_path = "direct_process_to_biology_transitional_baseline"
        if experiment["model"]["backend"] == "botorch":
            surrogate_type = BotorchSurrogate
        else:
            from mg2si.models.direct_baseline import DirectBaseline
            surrogate_type = DirectBaseline
        scoped[x_fields] = scoped[x_fields].apply(pd.to_numeric, errors="coerce")
        tumor = surrogate_type().fit(scoped[x_fields], scoped["y_tumor_viability_pct"])
        normal = surrogate_type().fit(scoped[x_fields], scoped["y_normal_viability_pct"])
        tumor_mean, tumor_std = tumor.predict(candidates[x_fields])
        normal_mean, normal_std = normal.predict(candidates[x_fields])
    else:
        result = {"status": "insufficient_evidence", "reason": "process-state and state-biology stages do not meet coverage thresholds", "scope": [branch, tumor_cell_line, normal_cell_line]}
        _save_recommendations(pd.DataFrame([result]), dataset, output)
        return result
    safe_threshold = float(objective["safety"]["threshold_pct"])
    efficacy_threshold = float(objective["efficacy"]["threshold_pct"])
    candidates["predicted_tumor_viability_mean"] = tumor_mean
    candidates["predicted_tumor_viability_lower"] = tumor_mean - 1.645 * tumor_std
    candidates["predicted_tumor_viability_upper"] = tumor_mean + 1.645 * tumor_std
    candidates["predicted_normal_viability_mean"] = normal_mean
    candidates["predicted_normal_viability_lower"] = normal_mean - 1.645 * normal_std
    candidates["predicted_normal_viability_upper"] = normal_mean + 1.645 * normal_std
    candidates["probability_efficacy"] = _normal_cdf((efficacy_threshold - tumor_mean) / np.maximum(tumor_std, 1e-6))
    candidates["probability_safety"] = _normal_cdf((normal_mean - safe_threshold) / np.maximum(normal_std, 1e-6))
    candidates["acquisition_value"] = (100.0 - tumor_mean + tumor_std) * candidates["probability_safety"]
    candidates["model_path"] = model_path
    candidates["out_of_domain"] = 0
    errors = validate_candidates(candidates)
    if errors:
        raise ValueError(errors)
    ranked = candidates.sort_values("acquisition_value", ascending=False).head(int(experiment["recommendation"]["batch_size"])).copy()
    ranked["recommendation_role"] = experiment["recommendation"]["roles"][: len(ranked)]
    destination = _save_recommendations(ranked, dataset, output)
    return {"status": "ok", "model_path": model_path, "scope_rows": int(len(scoped)), "candidate_pool": int(len(candidates)), "recommendations": int(len(ranked)), "output": destination}
