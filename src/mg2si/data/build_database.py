from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha1
import json
import os
from pathlib import Path
import sqlite3
from typing import Any

import numpy as np
import pandas as pd

from mg2si.config import PROJECT_ROOT, load_config
from mg2si.data.store import DEFAULT_DATABASE
from mg2si.data.supplemental_materials import load_supplemental_materials
from mg2si.io.excel_reader import resolve_sources
from mg2si.io.source_manifest import file_sha256


def _source_readers():
    from mg2si.io.biology_reader import melt, read_sheet
    from mg2si.io.material_reader import (
        choose_material_id,
        infer_parent_id,
        make_long_cell_table,
        read_material_master,
        read_tace_summary,
    )
    return read_material_master, read_tace_summary, make_long_cell_table, choose_material_id, infer_parent_id, read_sheet, melt


def _json_value(value: Any) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return str(value) if not isinstance(value, (str, int, float, bool)) else value


def _source_tables(paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifests = []
    records = []
    schema_version = load_config("data_sources.yaml")["schema_version"]
    for path in paths:
        digest = file_sha256(path)
        workbook = pd.ExcelFile(path)
        for sheet_name in workbook.sheet_names:
            raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
            nonempty = raw.dropna(how="all")
            manifests.append({
                "source_file": path.name,
                "source_file_hash": digest,
                "source_sheet": sheet_name,
                "rows": int(raw.shape[0]),
                "columns": int(raw.shape[1]),
                "schema_version": schema_version,
            })
            if nonempty.empty:
                continue
            header_values = [_json_value(value) for value in nonempty.iloc[0].tolist()]
            headers = []
            seen: dict[str, int] = {}
            for index, value in enumerate(header_values):
                base = str(value).strip() if value not in (None, "") else f"column_{index + 1}"
                seen[base] = seen.get(base, 0) + 1
                headers.append(base if seen[base] == 1 else f"{base}__{seen[base]}")
            for row_index, row in nonempty.iterrows():
                values = [_json_value(value) for value in row.tolist()]
                payload = {headers[index]: value for index, value in enumerate(values) if value is not None}
                records.append({
                    "source_record_id": f"{digest[:12]}:{sheet_name}:{int(row_index) + 1}",
                    "source_file": path.name,
                    "source_file_hash": digest,
                    "source_sheet": sheet_name,
                    "source_row": int(row_index) + 1,
                    "record_role": "header" if row_index == nonempty.index[0] else ("declared_range" if payload.get(headers[0]) == "范围" else "data"),
                    "payload_json": json.dumps(payload, ensure_ascii=False, sort_keys=True),
                })
    return pd.DataFrame(manifests), pd.DataFrame(records)


def _measurement_id(prefix: str, values: list[Any]) -> str | float:
    normalized = ["<missing>" if pd.isna(value) else str(value) for value in values]
    if all(value == "<missing>" for value in normalized):
        return np.nan
    return f"{prefix}_{sha1('|'.join(normalized).encode('utf-8')).hexdigest()[:16]}"


def _first_value(series: pd.Series) -> Any:
    for value in series:
        if value is None or pd.isna(value):
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _register_biology_materials(material: pd.DataFrame, bioassay: pd.DataFrame) -> pd.DataFrame:
    """Register every canonical biology sample and preserve source-derived process fields."""
    frame = material.copy().set_index("material_id", drop=False)
    text_fields = {
        "material_id", "material_source", "synthesis_method", "product_stage",
        "post_treatment", "pvp_mw_raw", "pvp_ratio_material_to_pvp_raw",
        "particle_distribution", "milling_mode", "material_registry_status",
        "source_mapping_status", "material_parent_id", "note",
    }
    for field in text_fields.intersection(frame.columns):
        frame[field] = frame[field].astype(object)
    stage_map = {
        "原料": "raw_material",
        "半成品I": "intermediate",
        "半成品II": "intermediate",
        "半成品": "intermediate",
        "成品": "finished_product",
    }
    for sample_id, rows in bioassay.dropna(subset=["tace_sample_id"]).groupby("tace_sample_id", sort=True):
        sample_id = str(sample_id).strip()
        if not sample_id:
            continue
        if sample_id not in frame.index:
            record = {column: np.nan for column in frame.columns}
            record.update({
                "material_id": sample_id,
                "has_sample_info": 1,
                "material_registry_status": "source_derived",
                "source_mapping_status": "confirmed_biology_identifier",
                "material_parent_id": sample_id,
                "note": "Registered from the primary biology workbook; parent lineage remains self until a confirmed parent is available.",
            })
            frame.loc[sample_id] = record

        source = _first_value(rows.get("material_source", pd.Series(dtype=object)))
        workflow = _first_value(rows.get("workflow_branch", pd.Series(dtype=object)))
        stage = _first_value(rows.get("sample_stage", pd.Series(dtype=object)))
        post_treatment = _first_value(rows.get("post_treatment", pd.Series(dtype=object)))
        pvp_mw_raw = _first_value(rows.get("pvp_mw_raw", pd.Series(dtype=object)))
        pvp_mw = _first_value(rows.get("pvp_mw", pd.Series(dtype=float)))
        pvp_ratio_raw = _first_value(rows.get("pvp_ratio_raw", pd.Series(dtype=object)))
        pvp_ratio = _first_value(rows.get("material_to_pvp_ratio", pd.Series(dtype=float)))
        particle_raw = _first_value(rows.get("milled_size_nm_raw", pd.Series(dtype=object)))
        updates = {
            "material_source": source,
            "synthesis_method": "commercial_purchase" if workflow == "commercial" else "biology_source_derived",
            "product_stage": stage_map.get(str(stage), None),
            "post_treatment": post_treatment,
            "pvp_mw_raw": pvp_mw_raw,
            "pvp_mw": pvp_mw,
            "pvp_ratio_material_to_pvp_raw": pvp_ratio_raw,
            "pvp_material_to_pvp_ratio": pvp_ratio,
            "particle_distribution": particle_raw,
            "milling_mode": "reported_milled" if pd.to_numeric(rows.get("is_milled"), errors="coerce").fillna(0).gt(0).any() else None,
            "has_sample_info": 1,
        }
        for field, value in updates.items():
            if field not in frame.columns or value is None or pd.isna(value):
                continue
            existing = frame.at[sample_id, field]
            if existing is None or pd.isna(existing) or (isinstance(existing, str) and not existing.strip()):
                frame.at[sample_id, field] = value
        if "material_parent_id" in frame.columns and (
            frame.at[sample_id, "material_parent_id"] is None or pd.isna(frame.at[sample_id, "material_parent_id"])
        ):
            frame.at[sample_id, "material_parent_id"] = sample_id
    return frame.reset_index(drop=True)


def _promote_process_observations(
    material: pd.DataFrame,
    observations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Promote unique, material-specific protocol values while retaining document-scope evidence."""
    frame = material.copy().set_index("material_id", drop=False)
    process = observations.copy()
    if process.empty:
        process["mapping_scope"] = pd.Series(dtype=object)
        return frame.reset_index(drop=True), process

    known = set(frame.index.astype(str))
    material_specific = process["material_id"].notna() & process["material_id"].astype(str).isin(known)
    process.loc[material_specific, "mapping_status"] = "confirmed_source_row"
    process.loc[~material_specific, "mapping_status"] = "document_scope"
    process["mapping_scope"] = np.where(material_specific, "material", "source_document")

    field_map = {
        "ball_to_material_ratio": ("ball_to_material_ratio", "numeric"),
        "initial_pressure_atm": ("initial_pressure_atm", "numeric"),
        "milling_total_runtime": ("milling_cycle_time", "numeric"),
        "pvp_mw": ("pvp_mw", "numeric"),
        "material_to_pvp_ratio": ("pvp_material_to_pvp_ratio", "numeric"),
        "ultrasonic_time": ("ultrasonic_time_h", "numeric"),
        "protective_atmosphere": ("protective_gas", "raw"),
        "post_treatment_solvent": ("post_treatment", "raw"),
        "post_treatment_solvent_system": ("post_treatment", "raw"),
    }
    scoped = process[material_specific].copy()
    for (material_id, parameter_name), group in scoped.groupby(["material_id", "parameter_name"], dropna=False):
        if parameter_name not in field_map or material_id not in frame.index:
            continue
        target, value_kind = field_map[parameter_name]
        if target not in frame.columns:
            continue
        source_column = "value_numeric" if value_kind == "numeric" else "value_raw"
        values = group[source_column].dropna().drop_duplicates().tolist()
        if len(values) != 1:
            continue
        existing = frame.at[material_id, target]
        if existing is None or pd.isna(existing) or (isinstance(existing, str) and not existing.strip()):
            frame.at[material_id, target] = values[0]
    return frame.reset_index(drop=True), process


def _feature_mapping_tables(
    material: pd.DataFrame,
    process_observations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    process_columns = {
        "molar_ratio_Mg_to_Si": ("Mg_to_Si_molar_ratio", None),
        "max_temp_c": ("synthesis_temperature", "degC"),
        "hold_time_min": ("hold_time", "min"),
        "vacuum_cycle": ("vacuum_cycle", "count"),
        "vacuum_time_min": ("vacuum_time", "min"),
        "protective_gas": ("protective_gas", None),
        "initial_pressure_atm": ("initial_pressure", "atm"),
        "milling_mode": ("milling_mode", None),
        "ball_to_material_ratio": ("ball_to_material_ratio", "ratio"),
        "milling_cycle_time": ("milling_time", "min"),
        "post_treatment": ("post_treatment", None),
        "ultrasonic_time_h": ("ultrasonic_time", "h"),
        "pvp_mw": ("pvp_mw", "g/mol"),
        "pvp_material_to_pvp_ratio": ("material_to_pvp_ratio", "ratio"),
    }
    characterization_columns = {
        "xrd_match": "structure",
        "mg2si_purity_pct": "structure",
        "grain_size_nm": "structure",
        "particle_distribution": "particle",
        "hrtem_lattice": "structure",
        "saed": "structure",
        "dls_size_nm": "particle",
        "pdi": "dispersion",
        "zeta_potential_mv": "dispersion",
        "xps_mg_1s": "surface_chemistry",
        "xps_si_2p": "surface_chemistry",
        "xps_o_1s": "surface_chemistry",
        "mg_si_atom_ratio": "surface_chemistry",
        "mgo_ratio": "surface_chemistry",
        "siox_ratio": "surface_chemistry",
        "oxide_thickness": "surface_chemistry",
        "defect_state": "surface_chemistry",
        "other_surface_features": "surface_chemistry",
    }
    process_rows: list[dict[str, Any]] = []
    for row in material.to_dict("records"):
        for field, (parameter, unit) in process_columns.items():
            value = row.get(field)
            if value is None or pd.isna(value):
                continue
            numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            process_rows.append({
                "material_id": row["material_id"],
                "material_parent_id": row.get("material_parent_id"),
                "product_stage": row.get("product_stage"),
                "parameter_name": parameter,
                "value_numeric": None if pd.isna(numeric) else float(numeric),
                "value_raw": str(value),
                "unit": unit,
                "mapping_status": row.get("source_mapping_status") or "material_master",
                "mapping_scope": "material",
                "source_file": "canonical_material",
                "source_record_id": None,
            })
    if not process_observations.empty:
        for row in process_observations.to_dict("records"):
            process_rows.append({
                "material_id": row.get("material_id"),
                "material_parent_id": row.get("material_parent_id"),
                "product_stage": row.get("product_stage"),
                "parameter_name": row.get("parameter_name"),
                "value_numeric": row.get("value_numeric"),
                "value_raw": row.get("value_raw"),
                "unit": row.get("unit"),
                "mapping_status": row.get("mapping_status"),
                "mapping_scope": row.get("mapping_scope"),
                "source_file": row.get("source_file"),
                "source_record_id": row.get("source_record_id"),
            })
    process_mapping = pd.DataFrame(process_rows).drop_duplicates()

    characterization_rows: list[dict[str, Any]] = []
    for row in material.to_dict("records"):
        for field, category in characterization_columns.items():
            value = row.get(field)
            if value is None or pd.isna(value):
                continue
            numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            characterization_rows.append({
                "material_id": row["material_id"],
                "material_parent_id": row.get("material_parent_id"),
                "feature_name": field,
                "feature_category": category,
                "value_numeric": None if pd.isna(numeric) else float(numeric),
                "value_raw": str(value),
                "mapping_status": row.get("source_mapping_status") or "material_master",
                "source_file": "canonical_material",
            })
    return process_mapping, pd.DataFrame(characterization_rows)


def _build_quality_issues(
    material: pd.DataFrame,
    bioassay: pd.DataFrame,
    audit: pd.DataFrame,
    conflicts: pd.DataFrame,
) -> pd.DataFrame:
    state_fields = [
        "mg2si_purity_pct", "grain_size_nm", "dls_size_nm", "pdi",
        "zeta_potential_mv", "mg_si_atom_ratio", "mgo_ratio", "siox_ratio", "oxide_thickness",
    ]
    explicit_normal = float(bioassay["normal_cell_line_status"].eq("explicit").mean())
    mapped = float(bioassay["material_id"].notna().mean())
    state_coverage = {field: float(material[field].notna().mean()) for field in state_fields if field in material}
    tumor = pd.to_numeric(bioassay["y_tumor_viability_pct"], errors="coerce")
    normal = pd.to_numeric(bioassay["y_normal_viability_pct"], errors="coerce")
    reused = int(bioassay.groupby("normal_measurement_group_id", dropna=True)["cell_record_id"].nunique().max() or 0)
    issues = [
        {
            "issue_id": "DQ001",
            "severity": "high",
            "check_name": "normal_cell_identity",
            "evidence": f"{explicit_normal:.1%} of concentration rows have an explicit normal cell line",
            "impact": "Safety responses without an explicit cell identity cannot define a formal model scope.",
            "action": "Populate the normal-cell field in the TACE summary; do not infer THLE from the HELE column label.",
        },
        {
            "issue_id": "DQ002",
            "severity": "high",
            "check_name": "material_mapping_coverage",
            "evidence": f"{mapped:.1%} of concentration rows map to the material workbook",
            "impact": "Unmapped biology rows cannot support process-state-biology inference.",
            "action": "Approve aliases and parent/child sample relationships in configs/aliases.yaml.",
        },
        {
            "issue_id": "DQ003",
            "severity": "high",
            "check_name": "material_state_coverage",
            "evidence": json.dumps(state_coverage, ensure_ascii=False, sort_keys=True),
            "impact": "The process-to-state stage is not identifiable with the current workbook.",
            "action": "Populate purity, grain size, DLS, PDI, Zeta and surface chemistry fields.",
        },
        {
            "issue_id": "DQ004",
            "severity": "high",
            "check_name": "summary_subtable_conflicts",
            "evidence": f"{len(conflicts)} concentration points differ between the summary and stage subtable; 297/297 subtable points already exist in the summary",
            "impact": "Appending stage sheets would duplicate evidence; silently overwriting would lose source disagreement.",
            "action": "Use 数据汇总 as canonical and resolve rows recorded in source_conflict.",
        },
        {
            "issue_id": "DQ005",
            "severity": "high",
            "check_name": "shared_normal_measurements",
            "evidence": f"A normal-cell measurement group is reused by up to {reused} concentration rows",
            "impact": "Treating reused safety values as independent observations understates uncertainty.",
            "action": "Split tumor and normal observations or group validation by normal_measurement_group_id.",
        },
        {
            "issue_id": "DQ006",
            "severity": "high",
            "check_name": "response_semantics",
            "evidence": "The material characterization sheet labels >100 values as tumor kill rate while TACE labels the modeled endpoint as viability.",
            "impact": "Mixing kill and viability directions would reverse the optimization target.",
            "action": "Exclude material-sheet kill columns until the metric owner confirms their definition.",
        },
        {
            "issue_id": "DQ007",
            "severity": "medium",
            "check_name": "above_control_viability",
            "evidence": f"{int((tumor > 100).sum())} tumor and {int((normal > 100).sum())} normal observations exceed 100% of control",
            "impact": "These may represent proliferation or assay normalization and should not be clipped automatically.",
            "action": "Retain values with above_control_reference flags; confirm assay normalization and plausible upper bound.",
        },
    ]
    return pd.DataFrame(issues)


def build_database(path: Path = DEFAULT_DATABASE) -> dict[str, Any]:
    (
        read_material_master,
        read_tace_summary,
        make_long_cell_table,
        choose_material_id,
        infer_parent_id,
        read_sheet,
        melt,
    ) = _source_readers()
    material_path, biology_path = resolve_sources(PROJECT_ROOT)
    material = read_material_master(material_path)
    supplements = load_supplemental_materials(PROJECT_ROOT, material)
    material = supplements["material"]
    tace = read_tace_summary(biology_path)
    bioassay = make_long_cell_table(tace)
    material = _register_biology_materials(material, bioassay)
    material, promoted_process = _promote_process_observations(
        material,
        supplements["process_observations"],
    )
    supplements["material"] = material
    supplements["process_observations"] = promoted_process

    material_ids = set(material["material_id"].dropna().astype(str))
    parent_lookup = (
        material.set_index("material_id")["material_parent_id"].dropna().astype(str).to_dict()
        if "material_parent_id" in material
        else {}
    )
    mapping_keys = bioassay[["tace_sample_id_raw", "tace_sample_id", "sample_stage"]].drop_duplicates()
    mapping_rows = []
    for _, row in mapping_keys.iterrows():
        selected, mapping_type, basis, candidates = choose_material_id(row["tace_sample_id_raw"], row["sample_stage"], material_ids)
        canonical_id = str(row["tace_sample_id"]).strip() if pd.notna(row["tace_sample_id"]) else None
        if selected is None and canonical_id in material_ids:
            selected = canonical_id
            mapping_type = "biology_registry_exact"
            basis = "canonical_tace_sample_id_registered_from_primary_biology_workbook"
            candidates = canonical_id
        mapping_rows.append({
            **row.to_dict(),
            "material_id": selected,
            "mapping_type": mapping_type,
            "mapping_basis": basis,
            "mapping_candidates": candidates,
            "material_parent_id": parent_lookup.get(selected) or infer_parent_id(selected, material_ids),
        })
    mapping = pd.DataFrame(mapping_rows)
    bioassay = bioassay.merge(mapping, how="left", on=["tace_sample_id_raw", "tace_sample_id", "sample_stage"])

    has_normal = bioassay[["y_normal_viability_pct"]].notna().any(axis=1)
    explicit_normal = bioassay["normal_cell_line"].notna()
    bioassay["normal_cell_line_status"] = np.select(
        [explicit_normal, has_normal],
        ["explicit", "unresolved_HELE_column_header"],
        default="not_measured",
    )
    bioassay.loc[has_normal & ~explicit_normal, "normal_cell_line"] = "UNRESOLVED_HELE_HEADER"

    normal_group_by_record = {}
    tumor_group_by_record = {}
    for row_number, row in tace.iterrows():
        record_id = f"TACE_summary_row_{row_number + 2}"
        normal_group_by_record[record_id] = _measurement_id("normal", [
            row.get("safety_assay_date"), row.get("normal_cell_line"),
            row.get("y_normal_500ppm"), row.get("y_normal_250ppm"), row.get("y_normal_125ppm"),
        ])
        tumor_group_by_record[record_id] = _measurement_id("tumor", [
            row.get("kill_assay_date"), row.get("tumor_cell_line"), row.get("exp_index"), row.get("repeat_batch"),
            row.get("y_tumor_500ppm"), row.get("y_tumor_250ppm"), row.get("y_tumor_125ppm"),
        ])
    bioassay["normal_measurement_group_id"] = bioassay["cell_record_id"].map(normal_group_by_record)
    bioassay["tumor_measurement_group_id"] = bioassay["cell_record_id"].map(tumor_group_by_record)

    tumor = pd.to_numeric(bioassay["y_tumor_viability_pct"], errors="coerce")
    normal = pd.to_numeric(bioassay["y_normal_viability_pct"], errors="coerce")
    bioassay["tumor_response_flag"] = np.select([tumor < 0, tumor > 200, tumor > 100], ["invalid_negative", "implausible_above_200", "above_control_reference"], default="ok")
    bioassay["normal_response_flag"] = np.select([normal < 0, normal > 200, normal > 100], ["invalid_negative", "implausible_above_200", "above_control_reference"], default="ok")
    plausible = tumor.between(0, 200) & normal.between(0, 200)
    bioassay["model_eligible_direct"] = (
        bioassay["target_complete"].eq(1)
        & bioassay["normal_cell_line_status"].eq("explicit")
        & plausible
    ).astype(int)
    bioassay["model_eligible_material"] = (
        bioassay["model_eligible_direct"].eq(1) & bioassay["material_id"].notna()
    ).astype(int)

    workbook = pd.ExcelFile(biology_path)
    source_tables = []
    for index, sheet_name in enumerate(workbook.sheet_names):
        if sheet_name == "备注":
            continue
        source_tables.append(read_sheet(biology_path, sheet_name, index == 0))
    source_rows = pd.concat(source_tables, ignore_index=True)
    audit = melt(source_rows)
    key_fields = ["tace_sample_id", "exp_index", "repeat_batch", "concentration_ppm"]
    audit["duplicate_group_key"] = audit[key_fields].fillna("").astype(str).agg("|".join, axis=1)
    audit["duplicate_group_count"] = audit["duplicate_group_key"].map(audit["duplicate_group_key"].value_counts())
    summary = audit[audit["is_summary_source"].eq(1)]
    subtables = audit[audit["is_summary_source"].eq(0)]
    conflicts = subtables.merge(summary, how="inner", on=key_fields, suffixes=("_subtable", "_summary"))
    same_tumor = conflicts["y_tumor_viability_pct_subtable"].fillna(-9999).round(8).eq(conflicts["y_tumor_viability_pct_summary"].fillna(-9999).round(8))
    same_normal = conflicts["y_normal_viability_pct_subtable"].fillna(-9999).round(8).eq(conflicts["y_normal_viability_pct_summary"].fillna(-9999).round(8))
    conflicts = conflicts[~(same_tumor & same_normal)].copy()
    conflict_columns = key_fields + [
        "source_sheet_subtable", "source_sheet_summary",
        "y_tumor_viability_pct_subtable", "y_tumor_viability_pct_summary",
        "y_normal_viability_pct_subtable", "y_normal_viability_pct_summary",
    ]
    conflicts = conflicts[conflict_columns]

    manifests, source_records = _source_tables([material_path, biology_path])
    manifests = pd.concat([manifests, supplements["manifests"]], ignore_index=True)
    source_records = pd.concat([source_records, supplements["records"]], ignore_index=True)
    quality_issues = pd.concat(
        [_build_quality_issues(material, bioassay, audit, conflicts), supplements["issues"]],
        ignore_index=True,
        sort=False,
    )
    mapping["cell_point_count"] = mapping["tace_sample_id"].map(bioassay["tace_sample_id"].value_counts())

    input_fields = [
        "molar_ratio_Mg_to_Si", "max_temp_c", "hold_time_min", "initial_pressure_atm",
        "vacuum_cycle", "vacuum_time_min", "protective_gas", "ball_to_material_ratio",
        "milling_cycle_time", "dls_size_nm", "pdi", "zeta_potential_mv", "pvp_mw",
        "pvp_material_to_pvp_ratio", "post_treatment",
    ]
    material_input_coverage = pd.DataFrame([
        {"field": field, "non_null_materials": int(material[field].notna().sum()), "total_materials": int(len(material)), "coverage_rate": float(material[field].notna().mean())}
        for field in input_fields if field in material
    ])
    condition_summary = (
        bioassay.groupby(["workflow_branch", "concentration_ppm", "tumor_cell_line", "normal_cell_line"], dropna=False)
        .agg(
            rows=("cell_record_id", "size"),
            materials=("material_id", "nunique"),
            tumor_viability_mean=("y_tumor_viability_pct", "mean"),
            tumor_viability_std=("y_tumor_viability_pct", "std"),
            normal_viability_mean=("y_normal_viability_pct", "mean"),
            normal_viability_std=("y_normal_viability_pct", "std"),
        )
        .reset_index()
    )
    duplicate_key = ["material_id", "experiment_id", "concentration_ppm", "tumor_cell_line", "normal_cell_line", "y_tumor_viability_pct", "y_normal_viability_pct"]
    exact_duplicate_rows = int(bioassay.duplicated(duplicate_key, keep=False).sum())
    data_quality_profile = pd.DataFrame([
        {"metric": "source_files_ingested", "value_numeric": float(len(supplements["manifests"])), "details": "All non-primary xlsx/docx files recursively discovered."},
        {"metric": "source_records_ingested", "value_numeric": float(len(source_records)), "details": "Every non-empty workbook row and Word paragraph/table row is retained."},
        {"metric": "exact_duplicate_bioassay_rows", "value_numeric": float(exact_duplicate_rows), "details": "Duplicate check at material x experiment x concentration x cell-pair x target grain."},
        {"metric": "unmapped_bioassay_rows", "value_numeric": float(bioassay["material_id"].isna().sum()), "details": "Rows requiring sample alias or parent-child confirmation."},
        {"metric": "above_control_tumor_rows", "value_numeric": float((tumor > 100).sum()), "details": "Retained as above-control observations; not clipped."},
        {"metric": "above_control_normal_rows", "value_numeric": float((normal > 100).sum()), "details": "Retained as above-control observations; not clipped."},
    ])
    material_process_mapping, material_characterization_mapping = _feature_mapping_tables(
        material,
        supplements["process_observations"],
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".sqlite.tmp")
    if temporary.exists():
        temporary.unlink()
    connection = sqlite3.connect(temporary)
    try:
        material.to_sql("material", connection, if_exists="replace", index=False)
        bioassay.to_sql("bioassay", connection, if_exists="replace", index=False)
        audit.to_sql("bioassay_source_audit", connection, if_exists="replace", index=False)
        mapping.to_sql("sample_mapping", connection, if_exists="replace", index=False)
        conflicts.to_sql("source_conflict", connection, if_exists="replace", index=False)
        manifests.to_sql("source_manifest", connection, if_exists="replace", index=False)
        source_records.to_sql("source_record", connection, if_exists="replace", index=False)
        supplements["links"].to_sql("supplement_material_link", connection, if_exists="replace", index=False)
        supplements["process_observations"].to_sql("supplement_process_observation", connection, if_exists="replace", index=False)
        material_process_mapping.to_sql("material_process_mapping", connection, if_exists="replace", index=False)
        material_characterization_mapping.to_sql("material_characterization_mapping", connection, if_exists="replace", index=False)
        supplements["particle_fractions"].to_sql("supplement_particle_fraction", connection, if_exists="replace", index=False)
        supplements["lineage"].to_sql("supplement_material_lineage", connection, if_exists="replace", index=False)
        supplements["index_references"].to_sql("supplement_index_reference", connection, if_exists="replace", index=False)
        supplements["issues"].to_sql("supplement_quality_issue", connection, if_exists="replace", index=False)
        material_input_coverage.to_sql("material_input_coverage", connection, if_exists="replace", index=False)
        condition_summary.to_sql("bioassay_condition_summary", connection, if_exists="replace", index=False)
        data_quality_profile.to_sql("data_quality_profile", connection, if_exists="replace", index=False)
        quality_issues.to_sql("quality_issue", connection, if_exists="replace", index=False)
        pd.DataFrame([{
            "pipeline_version": "0.3.0",
            "built_at_utc": datetime.now(timezone.utc).isoformat(),
            "material_source_hash": file_sha256(material_path),
            "biology_source_hash": file_sha256(biology_path),
            "supplement_source_count": int(len(supplements["manifests"])),
        }]).to_sql("pipeline_run", connection, if_exists="replace", index=False)
        material_columns = [column for column in material.columns if column != "material_id"]
        material_select = ", ".join(
            f'm."{column}" AS "material_{column}"' for column in material_columns
        )
        connection.execute("DROP VIEW IF EXISTS bo_training")
        connection.execute(
            f'CREATE VIEW bo_training AS SELECT b.*, m."material_id" AS "material_material_id"'
            + (f", {material_select}" if material_select else "")
            + ' FROM bioassay b LEFT JOIN material m ON b."material_id" = m."material_id"'
        )
        connection.execute("CREATE INDEX idx_bioassay_scope ON bioassay(workflow_branch, tumor_cell_line, normal_cell_line)")
        connection.execute("CREATE INDEX idx_bioassay_material ON bioassay(material_id, material_parent_id)")
        connection.execute("CREATE INDEX idx_source_record_sheet ON source_record(source_file, source_sheet, source_row)")
        connection.commit()
    finally:
        connection.close()
    os.replace(temporary, path)
    return {
        "status": "ok",
        "database": str(path),
        "material_rows": int(len(material)),
        "bioassay_rows": int(len(bioassay)),
        "source_audit_rows": int(len(audit)),
        "source_records": int(len(source_records)),
        "source_conflicts": int(len(conflicts)),
        "quality_issues": int(len(quality_issues)),
        "supplement_links": int(len(supplements["links"])),
        "supplement_process_observations": int(len(supplements["process_observations"])),
        "material_process_mappings": int(len(material_process_mapping)),
        "material_characterization_mappings": int(len(material_characterization_mapping)),
        "supplement_particle_fractions": int(len(supplements["particle_fractions"])),
        "supplement_lineage": int(len(supplements["lineage"])),
        "supplement_index_references": int(len(supplements["index_references"])),
        "supplement_quality_issues": int(len(supplements["issues"])),
        "model_eligible_direct": int(bioassay["model_eligible_direct"].sum()),
        "model_eligible_material": int(bioassay["model_eligible_material"].sum()),
    }
