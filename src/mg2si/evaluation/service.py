"""Evaluation services used by the CLI and reporting workflow."""

from pathlib import Path
import sqlite3

import pandas as pd

from mg2si.evaluation.grouped_splits import assert_no_group_leakage, grouped_splits
from mg2si.evaluation.metrics import regression_metrics
from mg2si.models.direct_baseline import DirectBaseline


def evaluate_database(path: Path, output: Path) -> dict:
    """Run grouped baseline evaluation and persist the report."""
    if path.suffix.lower() in {".sqlite", ".db"}:
        with sqlite3.connect(path) as connection:
            frame = pd.read_sql_query("SELECT * FROM bo_training", connection)
    else:
        frame = pd.read_csv(path, low_memory=False)
    if "model_eligible_direct" in frame:
        frame = frame[frame["model_eligible_direct"].eq(1)].copy()

    reports = []
    features = [
        field for field in (
            "material_molar_ratio_Mg_to_Si",
            "material_max_temp_c",
            "material_hold_time_min",
            "material_initial_pressure_atm",
            "material_milling_cycle_time",
            "material_ball_to_material_ratio",
            "pvp_mw",
            "material_to_pvp_ratio",
            "concentration_ppm",
        ) if field in frame
    ]
    for scope, scoped in frame.groupby(
        ["workflow_branch", "tumor_cell_line", "normal_cell_line"], dropna=False
    ):
        scope_label = "|".join(str(value) for value in scope)
        scoped = scoped.dropna(subset=["y_tumor_viability_pct", "y_normal_viability_pct"])
        scope_features = [field for field in features if scoped[field].notna().any()]
        group_field = (
            "material_parent_id"
            if "material_parent_id" in scoped and scoped["material_parent_id"].nunique() >= 2
            else "experiment_id"
        )
        groups = scoped[group_field].fillna(scoped["experiment_id"]).astype(str)
        if len(scoped) < 20 or groups.nunique() < 2 or not scope_features:
            reports.append({
                "scope": scope_label,
                "status": "insufficient_evidence",
                "rows": len(scoped),
                "usable_features": len(scope_features),
            })
            continue

        splits = grouped_splits(groups, 5)
        assert_no_group_leakage(groups, splits)
        for target in ("y_tumor_viability_pct", "y_normal_viability_pct"):
            observed, predicted = [], []
            for train, test in splits:
                fold_features = [
                    field for field in scope_features
                    if scoped.iloc[train][field].notna().any()
                ]
                model = DirectBaseline().fit(
                    scoped.iloc[train][fold_features],
                    scoped.iloc[train][target],
                )
                mean, _ = model.predict(scoped.iloc[test][fold_features])
                observed.extend(scoped.iloc[test][target])
                predicted.extend(mean)
            reports.append({
                "scope": scope_label,
                "target": target,
                "group": group_field,
                "rows": len(scoped),
                **regression_metrics(observed, predicted),
            })

    report_frame = pd.DataFrame(reports)
    if path.suffix.lower() in {".sqlite", ".db"}:
        with sqlite3.connect(path) as connection:
            report_frame.to_sql("model_evaluation", connection, if_exists="replace", index=False)
        destination = f"{path}#model_evaluation"
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        report_frame.to_csv(output, index=False, encoding="utf-8-sig")
        destination = str(output)
    return {"status": "ok", "evaluations": len(reports), "output": destination}
