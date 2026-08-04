"""Print a compact, reproducible audit of the rebuilt SQLite database."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3

import pandas as pd

from mg2si.data.store import DEFAULT_DATABASE


def audit(path: Path) -> dict:
    with sqlite3.connect(path) as connection:
        materials = pd.read_sql_query("SELECT material_id, product_stage, material_registry_status, source_mapping_status FROM material", connection)
        parameters = pd.read_sql_query("SELECT product_stage, parameter_name, COUNT(*) AS n FROM supplement_process_observation GROUP BY product_stage, parameter_name ORDER BY product_stage, parameter_name", connection)
        lineage = pd.read_sql_query("SELECT child_material_id, parent_material_id, transformation_type, mapping_status FROM supplement_material_lineage ORDER BY child_material_id", connection)
        quality = pd.read_sql_query("SELECT * FROM data_quality_profile ORDER BY metric", connection)
        condition_summary = pd.read_sql_query("SELECT * FROM bioassay_condition_summary ORDER BY workflow_branch, concentration_ppm", connection)
        input_coverage = pd.read_sql_query("SELECT * FROM material_input_coverage ORDER BY field", connection)
        manifests = pd.read_sql_query("SELECT source_file, source_sheet, rows, columns FROM source_manifest ORDER BY source_file, source_sheet", connection)
        index_references = pd.read_sql_query("SELECT exp_index, tace_sample_id_raw, sample_stage, material_id, COUNT(*) AS concentration_rows FROM bioassay WHERE exp_index IN ('S20','S21','S22','S23','S24') GROUP BY exp_index, tace_sample_id_raw, sample_stage, material_id ORDER BY exp_index", connection)
        particle_fractions = pd.read_sql_query("SELECT material_id, material_parent_id, fraction_label, size_lower_nm, size_upper_nm, size_basis, index_ids, assay_status, mapping_status FROM supplement_particle_fraction ORDER BY material_id", connection)
    return {
        "material_stage_counts": materials["product_stage"].value_counts(dropna=False).to_dict(),
        "source_derived_materials": materials[materials["material_registry_status"].eq("source_derived_needs_confirmation")].to_dict("records"),
        "process_parameter_counts": parameters.to_dict("records"),
        "lineage": lineage.to_dict("records"),
        "quality_profile": quality.to_dict("records"),
        "condition_summary": condition_summary.to_dict("records"),
        "material_input_coverage": input_coverage.to_dict("records"),
        "source_manifest_rows": manifests.to_dict("records"),
        "index_references": index_references.to_dict("records"),
        "particle_fractions": particle_fractions.to_dict("records"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    args = parser.parse_args()
    print(json.dumps(audit(args.database), ensure_ascii=False, indent=2, default=str))
