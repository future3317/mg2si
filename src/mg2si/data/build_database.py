from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha1
import json
import os
from pathlib import Path
import sqlite3
import sys
from typing import Any

import numpy as np
import pandas as pd

from mg2si.config import PROJECT_ROOT, load_config
from mg2si.data.store import DEFAULT_DATABASE
from mg2si.io.excel_reader import resolve_sources
from mg2si.io.source_manifest import file_sha256


def _legacy_readers():
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from export_tace_subtables import melt, read_sheet
    from rebuild_material_cell_dataset import (
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
    ) = _legacy_readers()
    material_path, biology_path = resolve_sources(PROJECT_ROOT)
    material = read_material_master(material_path)
    tace = read_tace_summary(biology_path)
    bioassay = make_long_cell_table(tace)

    material_ids = set(material["material_id"].dropna().astype(str))
    mapping_keys = bioassay[["tace_sample_id_raw", "tace_sample_id", "sample_stage"]].drop_duplicates()
    mapping_rows = []
    for _, row in mapping_keys.iterrows():
        selected, mapping_type, basis, candidates = choose_material_id(row["tace_sample_id"], row["sample_stage"], material_ids)
        mapping_rows.append({
            **row.to_dict(),
            "material_id": selected,
            "mapping_type": mapping_type,
            "mapping_basis": basis,
            "mapping_candidates": candidates,
            "material_parent_id": infer_parent_id(selected, material_ids),
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
    quality_issues = _build_quality_issues(material, bioassay, audit, conflicts)
    mapping["cell_point_count"] = mapping["tace_sample_id"].map(bioassay["tace_sample_id"].value_counts())

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
        quality_issues.to_sql("quality_issue", connection, if_exists="replace", index=False)
        pd.DataFrame([{
            "pipeline_version": "0.3.0",
            "built_at_utc": datetime.now(timezone.utc).isoformat(),
            "material_source_hash": file_sha256(material_path),
            "biology_source_hash": file_sha256(biology_path),
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
        "model_eligible_direct": int(bioassay["model_eligible_direct"].sum()),
        "model_eligible_material": int(bioassay["model_eligible_material"].sum()),
    }

