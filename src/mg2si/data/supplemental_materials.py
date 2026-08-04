from __future__ import annotations

from hashlib import sha1
import json
from pathlib import Path
import re
from typing import Any
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np
import pandas as pd

from mg2si.config import CONFIG_ROOT, load_config
from mg2si.io.source_manifest import file_sha256


SOURCE_RECORD_COLUMNS = [
    "source_record_id",
    "source_file",
    "source_file_hash",
    "source_sheet",
    "source_row",
    "record_role",
    "payload_json",
]
LINK_COLUMNS = [
    "source_file",
    "source_sheet",
    "source_row",
    "source_material_id_raw",
    "material_id",
    "material_parent_id",
    "mapping_type",
    "mapping_status",
    "mapping_basis",
]
PROCESS_COLUMNS = [
    "source_file",
    "material_id",
    "material_parent_id",
    "mapping_status",
    "parameter_name",
    "value_numeric",
    "unit",
    "value_raw",
    "note",
]
ISSUE_COLUMNS = ["issue_id", "severity", "check_name", "evidence", "impact", "action"]
PARTICLE_FRACTION_COLUMNS = [
    "source_file",
    "material_id",
    "material_parent_id",
    "source_material_id_raw",
    "fraction_label",
    "size_lower_nm",
    "size_upper_nm",
    "mapping_status",
    "size_basis",
    "index_ids",
    "assay_status",
    "note",
]
INDEX_REFERENCE_COLUMNS = [
    "index_id", "material_id", "sample_stage", "mapping_status", "assay_scope", "mapping_basis",
]


def _classify_product_stage(material_id: str, row: pd.Series | None = None) -> str:
    text = " ".join(str(value) for value in (row.tolist() if row is not None else []) if value is not None).lower()
    if material_id.endswith("-P") or material_id.endswith("-K60") or "pvp" in text:
        return "finished_product"
    if material_id.endswith("-SHS") or "合成" in text or "烧结" in text or "管式炉" in text:
        return "raw_material"
    return "intermediate"


def _register_source_materials(material: pd.DataFrame, discovered_ids: set[str], id_stage: dict[str, str]) -> pd.DataFrame:
    """Register concrete IDs mentioned in source files without making them BO-ready."""
    frame = material.copy()
    for column, default in {
        "product_stage": None,
        "material_registry_status": "master_or_curated",
        "source_mapping_status": "confirmed",
        "material_parent_id": None,
    }.items():
        if column not in frame.columns:
            frame[column] = default
    existing = set(frame["material_id"].dropna().astype(str))
    template = frame.iloc[0].copy() if not frame.empty else pd.Series(dtype=object)
    new_rows = []
    for material_id in sorted(discovered_ids):
        if material_id in existing:
            continue
        row = template.copy()
        for column in frame.columns:
            row[column] = np.nan
        row["material_id"] = material_id
        row["product_stage"] = id_stage.get(material_id, _classify_product_stage(material_id))
        row["material_registry_status"] = "source_derived_needs_confirmation"
        row["source_mapping_status"] = "needs_confirmation"
        row["has_sample_info"] = 0
        row["note"] = "Concrete identifier found in supplemental source; process evidence retained, material master fields await confirmation."
        new_rows.append(row.to_dict())
    if new_rows:
        frame = pd.concat([frame, pd.DataFrame(new_rows, columns=frame.columns)], ignore_index=True)
    frame["product_stage"] = frame.apply(
        lambda row: row.get("product_stage") if pd.notna(row.get("product_stage")) else _classify_product_stage(str(row.get("material_id", "")), row),
        axis=1,
    )
    return frame


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _append_frames(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    if left.empty:
        return right.copy()
    if right.empty:
        return left.copy()
    return pd.concat([left, right], ignore_index=True)


def _clean(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return None if text in {"", "-", "/", "nan", "None"} else text


def _number(value: Any) -> float | None:
    text = _clean(value)
    if text is None:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(match.group()) if match else None


def normalize_supplemental_material_id(value: Any) -> tuple[str | None, str | None, str | None]:
    """Return canonical id, SHS parent id and optional layer without dropping layer identity."""
    raw = _clean(value)
    if raw is None:
        return None, None, None
    match = re.match(
        r"^(?P<parent>MS-\d{6}-SHS)(?:\s+(?P<layer>top|down|mid))?(?:\s*(?:上层|下层|中层))?$",
        raw,
        flags=re.IGNORECASE,
    )
    if not match:
        return raw, raw, None
    parent = match.group("parent")
    layer = match.group("layer")
    return (f"{parent} {layer}" if layer else parent), parent, layer


def _docx_paragraphs(path: Path) -> list[str]:
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with ZipFile(path) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
    paragraphs = []
    for paragraph in root.findall(".//w:p", namespace):
        text = "".join(node.text or "" for node in paragraph.findall(".//w:t", namespace)).strip()
        if text:
            paragraphs.append(text)
    return paragraphs


def _source_name(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _source_rows(path: Path, root: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[str, int, str]]]:
    """Read all supplemental workbook rows and collect candidate sample ids."""
    digest = file_sha256(path)
    source_file = _source_name(path, root)
    manifests = []
    records = []
    sample_rows: list[tuple[str, int, str]] = []
    workbook = pd.ExcelFile(path)
    for sheet_name in workbook.sheet_names:
        raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
        nonempty = raw.dropna(how="all")
        manifests.append({
            "source_file": source_file,
            "source_file_hash": digest,
            "source_sheet": sheet_name,
            "rows": int(raw.shape[0]),
            "columns": int(raw.shape[1]),
            "schema_version": "1.1-supplemental",
        })
        if nonempty.empty:
            continue
        header = [str(_clean(value) or f"column_{index + 1}") for index, value in enumerate(nonempty.iloc[0])]
        for row_index, row in nonempty.iterrows():
            values = [_clean(value) for value in row]
            payload = {header[index]: value for index, value in enumerate(values) if value is not None}
            source_row = int(row_index) + 1
            role = "header" if row_index == nonempty.index[0] else ("declared_range" if values and values[0] == "范围" else "data")
            records.append({
                "source_record_id": f"{digest[:12]}:{sheet_name}:{source_row}",
                "source_file": source_file,
                "source_file_hash": digest,
                "source_sheet": sheet_name,
                "source_row": source_row,
                "record_role": role,
                "payload_json": json.dumps(payload, ensure_ascii=False, sort_keys=True),
            })
            if role == "data" and values and values[0]:
                sample_rows.append((sheet_name, source_row, values[0]))
    return pd.DataFrame(manifests), pd.DataFrame(records), sample_rows


def _apply_structure_rows(material: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Add only layer-specific samples; keep source values if existing values conflict."""
    workbook = pd.ExcelFile(path)
    structure = pd.read_excel(path, sheet_name=workbook.sheet_names[1])
    frame = material.set_index("material_id", drop=False).copy()
    conflicts: list[dict[str, Any]] = []
    field_positions = {
        "peak_ratio_raw": 3,
        "material_kill_500ppm": 11,
        "material_kill_250ppm": 12,
        "material_kill_125ppm": 13,
    }
    for _, row in structure.iterrows():
        material_id, parent_id, layer = normalize_supplemental_material_id(row.iloc[0])
        if material_id is None:
            continue
        if material_id not in frame.index and layer and parent_id in frame.index:
            derived = frame.loc[parent_id].copy()
            derived["material_id"] = material_id
            derived["layer_position"] = layer
            parent_note = _clean(derived.get("note"))
            derived["note"] = "; ".join(filter(None, [parent_note, f"Layer-specific structure record: {layer}"]))
            for field in field_positions:
                derived[field] = np.nan
            frame.loc[material_id] = derived
        if material_id not in frame.index:
            continue
        if layer and pd.isna(frame.at[material_id, "layer_position"]):
            frame.at[material_id, "layer_position"] = layer
        for field, position in field_positions.items():
            incoming = row.iloc[position]
            incoming = _clean(incoming) if field == "peak_ratio_raw" else _number(incoming)
            if incoming is None:
                continue
            existing = frame.at[material_id, field]
            if pd.isna(existing) or existing is None:
                frame.at[material_id, field] = incoming
            elif str(existing) != str(incoming):
                conflicts.append({
                    "issue_id": f"SUP_CONFLICT_{sha1(f'{material_id}|{field}'.encode()).hexdigest()[:10]}",
                    "severity": "medium",
                    "check_name": "supplement_structure_conflict",
                    "evidence": f"{material_id} {field}: existing={existing}; supplement={incoming}",
                    "impact": "Supplemental value was retained as source evidence but did not overwrite the existing material value.",
                    "action": "Confirm whether measurements are replicates, different layers, or revised values before selecting a canonical value.",
                })
        frame.at[material_id, "has_structure"] = 1
    return frame.reset_index(drop=True), conflicts


def _material_links(
    source_file: str,
    sample_rows: list[tuple[str, int, str]],
    material_ids: set[str],
) -> pd.DataFrame:
    links = []
    for sheet_name, source_row, raw_id in sample_rows:
        material_id, parent_id, layer = normalize_supplemental_material_id(raw_id)
        if material_id in material_ids:
            links.append({
                "source_file": source_file,
                "source_sheet": sheet_name,
                "source_row": source_row,
                "source_material_id_raw": raw_id,
                "material_id": material_id,
                "material_parent_id": parent_id if layer else material_id,
                "mapping_type": "layer_canonicalization" if layer else "exact",
                "mapping_status": "confirmed",
                "mapping_basis": "supplemental_workbook_sample_id",
            })
        else:
            links.append({
                "source_file": source_file,
                "source_sheet": sheet_name,
                "source_row": source_row,
                "source_material_id_raw": raw_id,
                "material_id": None,
                "material_parent_id": parent_id,
                "mapping_type": "unmatched",
                "mapping_status": "needs_confirmation",
                "mapping_basis": "supplemental_workbook_sample_id_not_in_material_master",
            })
    return pd.DataFrame(links, columns=LINK_COLUMNS)


def _docx_sources(
    directory: Path,
    root: Path,
    material_ids: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifests = []
    records = []
    links = []
    process = []
    for path in sorted(directory.glob("*.docx")):
        digest = file_sha256(path)
        source_file = _source_name(path, root)
        paragraphs = _docx_paragraphs(path)
        manifests.append({
            "source_file": source_file,
            "source_file_hash": digest,
            "source_sheet": "document",
            "rows": len(paragraphs),
            "columns": 1,
            "schema_version": "1.1-supplemental",
        })
        document_text = "\n".join(paragraphs)
        for row_number, text in enumerate(paragraphs, start=1):
            records.append({
                "source_record_id": f"{digest[:12]}:document:{row_number}",
                "source_file": source_file,
                "source_file_hash": digest,
                "source_sheet": "document",
                "source_row": row_number,
                "record_role": "document_paragraph",
                "payload_json": json.dumps({"text": text}, ensure_ascii=False),
            })
        for material_id in sorted(material_ids, key=len, reverse=True):
            if material_id not in document_text:
                continue
            uncertain = "260122" in path.name and material_id.startswith("MS-Q-260122")
            links.append({
                "source_file": source_file,
                "source_sheet": "document",
                "source_row": None,
                "source_material_id_raw": material_id,
                "material_id": material_id,
                "material_parent_id": material_id,
                "mapping_type": "document_identifier_mention",
                "mapping_status": "needs_confirmation" if uncertain else "confirmed",
                "mapping_basis": "ambiguous_solvent_output_labels" if uncertain else "exact_identifier_in_protocol",
            })
        if "260110" in path.name and "260116" in path.name:
            protocol_members = [
                material_id
                for material_id in ("MS-251016-Q", "MS-251215-Q", "MS-251220-Q", "MS-251222-Q ADMS")
                if material_id in material_ids and material_id in document_text
            ]
            parameters = [
                ("ball_to_material_ratio", 100.0, "ratio", "100:1", "Document specifies zirconia ball:material = 100:1."),
                ("milling_total_runtime", 800.0, "min", "20 min running × 20 cycles", "Total active milling time."),
                ("milling_run_segment", 20.0, "min", "20 min", "Per-cycle active milling duration."),
                ("milling_pause_segment", 5.0, "min", "5 min", "Per-cycle stop duration."),
                ("milling_cycle_count", 20.0, "count", "20 cycles", "Program cycle count."),
                ("milling_frequency", 30.0, "Hz", "30 Hz", "Program frequency."),
                ("milling_speed", 500.0, "rpm", "500 rpm", "Program speed."),
                ("pre_milling_vacuum", 3.0, "min", "3 min", "Evacuation before loading."),
            ]
            for material_id in protocol_members:
                parent_candidate = material_id.split("-Q")[0] + "-SHS"
                for parameter_name, value, unit, raw, note in parameters:
                    process.append({
                        "source_file": source_file,
                        "material_id": material_id,
                        "material_parent_id": parent_candidate if parent_candidate in material_ids else None,
                        "mapping_status": "confirmed",
                        "parameter_name": parameter_name,
                        "value_numeric": value,
                        "unit": unit,
                        "value_raw": raw,
                        "note": note,
                    })
    return (
        pd.DataFrame(manifests),
        pd.DataFrame(records, columns=SOURCE_RECORD_COLUMNS),
        pd.DataFrame(links, columns=LINK_COLUMNS),
        pd.DataFrame(process, columns=PROCESS_COLUMNS),
    )


def _apply_curated_mappings(material: pd.DataFrame) -> dict[str, Any]:
    """Apply researcher-confirmed mappings that resolve ambiguous protocol labels."""
    config_path = CONFIG_ROOT / "supplemental_mappings.yaml"
    config = load_config(config_path.name)
    digest = file_sha256(config_path)
    source_file = _source_name(config_path, CONFIG_ROOT.parent)
    frame = material.set_index("material_id", drop=False).copy()
    for column in ("product_stage", "material_registry_status", "source_mapping_status", "material_parent_id"):
        if column not in frame.columns:
            frame[column] = pd.Series([None] * len(frame), index=frame.index, dtype="object")
    records = []
    links = []
    process = []
    fractions = []
    index_references = []
    row_number = 0
    for item in config.get("material_records", []):
        row_number += 1
        material_id = item["material_id"]
        if material_id not in frame.index:
            derived = frame.iloc[0].copy()
            for field in frame.columns:
                derived[field] = np.nan
            derived["material_id"] = material_id
            derived["has_sample_info"] = 1
            frame.loc[material_id] = derived
        frame.at[material_id, "product_stage"] = item.get("product_stage", "raw_material")
        frame.at[material_id, "material_source"] = item.get("material_source")
        frame.at[material_id, "synthesis_method"] = item.get("synthesis_method")
        frame.at[material_id, "material_registry_status"] = "researcher_confirmed"
        frame.at[material_id, "source_mapping_status"] = item.get("source_mapping_status", "confirmed")
        frame.at[material_id, "note"] = item.get("note")
        records.append({
            "source_record_id": f"{digest[:12]}:material_records:{row_number}",
            "source_file": source_file,
            "source_file_hash": digest,
            "source_sheet": "material_records",
            "source_row": row_number,
            "record_role": "researcher_confirmed_mapping",
            "payload_json": json.dumps(item, ensure_ascii=False, sort_keys=True),
        })
    for item in config.get("solvent_treatments", []):
        row_number += 1
        material_id = item["material_id"]
        if material_id not in frame.index:
            continue
        frame.at[material_id, "post_treatment"] = item["post_treatment"]
        frame.at[material_id, "ultrasonic_time_raw"] = item["ultrasonic_time_raw"]
        frame.at[material_id, "ultrasonic_time_h"] = float(item["ultrasonic_time_h"])
        records.append({
            "source_record_id": f"{digest[:12]}:solvent_treatments:{row_number}",
            "source_file": source_file,
            "source_file_hash": digest,
            "source_sheet": "solvent_treatments",
            "source_row": row_number,
            "record_role": "researcher_confirmed_mapping",
            "payload_json": json.dumps(item, ensure_ascii=False, sort_keys=True),
        })
        links.append({
            "source_file": source_file,
            "source_sheet": "solvent_treatments",
            "source_row": row_number,
            "source_material_id_raw": material_id,
            "material_id": material_id,
            "material_parent_id": item.get("source_material_id_raw"),
            "mapping_type": "researcher_confirmed_solvent_mapping",
            "mapping_status": "confirmed",
            "mapping_basis": "data_provider_confirmation_2026-07-26",
        })
        process.extend([
            {
                "source_file": source_file,
                "material_id": material_id,
                "material_parent_id": item.get("source_material_id_raw"),
                "mapping_status": "confirmed",
                "parameter_name": "post_treatment_solvent_system",
                "value_numeric": None,
                "unit": None,
                "value_raw": item["post_treatment"],
                "note": "Researcher-confirmed sample-name semantics.",
            },
            {
                "source_file": source_file,
                "material_id": material_id,
                "material_parent_id": item.get("source_material_id_raw"),
                "mapping_status": "confirmed",
                "parameter_name": "ultrasonic_time",
                "value_numeric": float(item["ultrasonic_time_h"]),
                "unit": "h",
                "value_raw": item["ultrasonic_time_raw"],
                "note": "Researcher-confirmed post-treatment duration.",
            },
        ])
    structure_and_biology_fields = [
        "xrd_match", "mg2si_purity_pct", "peak_ratio_raw", "grain_size_nm", "particle_distribution",
        "hrtem_lattice", "saed", "dls_size_nm", "pdi", "zeta_potential_mv", "material_kill_500ppm",
        "material_kill_250ppm", "material_kill_125ppm", "xps_mg_1s", "xps_si_2p", "xps_o_1s",
        "mg_si_atom_ratio", "mgo_ratio", "siox_ratio", "oxide_thickness", "defect_state",
        "other_surface_features", "mg2si_purity_score", "size_score", "dispersion_score",
        "tumor_kill_score", "safety_score", "overall_score", "entered_animal", "quality_note",
        "cell_line", "tumor_cell_type", "normal_cell_type", "treat_conc_ppm", "treat_time_h",
        "ic50_tumor", "ic50_normal", "safety_index", "ros_level", "apoptosis_rate",
    ]
    for item in config.get("particle_fractions", []):
        row_number += 1
        material_id = item["material_id"]
        parent_id = item["material_parent_id"]
        if material_id not in frame.index and parent_id in frame.index:
            derived = frame.loc[parent_id].copy()
            derived["material_id"] = material_id
            derived["material_source"] = item["source_material_id_raw"]
            derived["synthesis_method"] = "SHS + ball milling + differential centrifugation"
            derived["milling_mode"] = "dry_milling"
            derived["ball_mill_ratio_raw"] = "100:1"
            derived["ball_to_material_ratio"] = 100.0
            derived["milling_cycle_time"] = 800.0
            derived["post_treatment"] = "ethanol dispersion + differential centrifugation"
            derived["note"] = item["note"]
            for field in structure_and_biology_fields:
                derived[field] = np.nan
            derived["has_sample_info"] = 1
            derived["has_structure"] = 0
            derived["has_surface_chemistry"] = 0
            derived["has_screening"] = 0
            derived["has_material_biology"] = 0
            frame.loc[material_id] = derived
        if material_id in frame.index:
            frame.at[material_id, "product_stage"] = "intermediate"
            frame.at[material_id, "material_parent_id"] = parent_id
            frame.at[material_id, "material_registry_status"] = "researcher_confirmed"
            frame.at[material_id, "source_mapping_status"] = "confirmed"
            frame.at[material_id, "synthesis_method"] = "commercial raw material + ultrasonication"
            frame.at[material_id, "post_treatment"] = "ultrasonication"
        records.append({
            "source_record_id": f"{digest[:12]}:particle_fractions:{row_number}",
            "source_file": source_file,
            "source_file_hash": digest,
            "source_sheet": "particle_fractions",
            "source_row": row_number,
            "record_role": "researcher_confirmed_mapping",
            "payload_json": json.dumps(item, ensure_ascii=False, sort_keys=True),
        })
        links.append({
            "source_file": source_file,
            "source_sheet": "particle_fractions",
            "source_row": row_number,
            "source_material_id_raw": item["source_material_id_raw"],
            "material_id": material_id,
            "material_parent_id": parent_id,
            "mapping_type": "researcher_confirmed_particle_fraction",
            "mapping_status": "confirmed",
            "mapping_basis": "data_provider_confirmation_2026-07-26",
        })
        fractions.append({
            "source_file": source_file,
            "material_id": material_id,
            "material_parent_id": parent_id,
            "source_material_id_raw": item["source_material_id_raw"],
            "fraction_label": item["fraction_label"],
            "size_lower_nm": item.get("size_lower_nm"),
            "size_upper_nm": item.get("size_upper_nm"),
            "mapping_status": "confirmed",
            "size_basis": item.get("size_basis", "source_document"),
            "index_ids": ",".join(item.get("index_ids", [])),
            "assay_status": item.get("assay_status"),
            "note": item["note"],
        })
    for item in config.get("index_references", []):
        row_number += 1
        index_references.append({
            "index_id": item["index_id"],
            "material_id": item["material_id"],
            "sample_stage": item.get("sample_stage"),
            "mapping_status": item.get("mapping_status", "needs_confirmation"),
            "assay_scope": item.get("assay_scope"),
            "mapping_basis": "data_provider_confirmation_2026-07-26",
        })
        records.append({
            "source_record_id": f"{digest[:12]}:index_references:{row_number}",
            "source_file": source_file,
            "source_file_hash": digest,
            "source_sheet": "index_references",
            "source_row": row_number,
            "record_role": "researcher_confirmed_mapping",
            "payload_json": json.dumps(item, ensure_ascii=False, sort_keys=True),
        })
    manifests = pd.DataFrame([{
        "source_file": source_file,
        "source_file_hash": digest,
        "source_sheet": "researcher_confirmed_mappings",
        "rows": row_number,
        "columns": 1,
        "schema_version": str(config.get("schema_version", "1.0")),
    }])
    return {
        "material": frame.reset_index(drop=True),
        "manifests": manifests,
        "records": pd.DataFrame(records, columns=SOURCE_RECORD_COLUMNS),
        "links": pd.DataFrame(links, columns=LINK_COLUMNS),
        "process_observations": pd.DataFrame(process, columns=PROCESS_COLUMNS),
        "particle_fractions": pd.DataFrame(fractions, columns=PARTICLE_FRACTION_COLUMNS),
        "index_references": pd.DataFrame(index_references, columns=INDEX_REFERENCE_COLUMNS),
        "solvent_mapping_confirmed": bool(config.get("solvent_treatments")),
        "particle_mapping_confirmed": bool(config.get("particle_fractions")),
        "commercial_particle_mapping_confirmed": any(
            str(item.get("material_id", "")).startswith("MS-Q-251124-")
            for item in config.get("particle_fractions", [])
        ),
    }


def load_supplemental_materials(root: Path, material: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Load all non-primary Excel/Word sources and preserve uncertain mappings."""
    result = {
        "material": material.copy(),
        "manifests": _empty(["source_file", "source_file_hash", "source_sheet", "rows", "columns", "schema_version"]),
        "records": _empty(SOURCE_RECORD_COLUMNS),
        "links": _empty(LINK_COLUMNS),
        "process_observations": _empty(PROCESS_COLUMNS),
        "particle_fractions": _empty(PARTICLE_FRACTION_COLUMNS),
        "index_references": _empty(INDEX_REFERENCE_COLUMNS),
        "lineage": _empty(["source_record_id", "source_file", "source_sheet", "source_row", "child_material_id", "parent_material_id", "transformation_type", "mapping_status", "evidence"]),
        "issues": _empty(ISSUE_COLUMNS),
    }

    from mg2si.data.bulk_ingest import bulk_ingest_sources
    from mg2si.io.excel_reader import resolve_sources

    primary_material, primary_biology = resolve_sources(root)
    bulk = bulk_ingest_sources(root, {primary_material, primary_biology})

    # The updated SHS workbook contains layer-specific structure rows.  Apply
    # those values to the canonical material table, while all other workbook
    # cells remain lossless in source_record.
    structure_workbooks = sorted(root.glob("*SHS*(2).xlsx"))
    if structure_workbooks:
        updated, conflicts = _apply_structure_rows(result["material"], structure_workbooks[0])
        result["material"] = updated
        result["issues"] = pd.DataFrame(conflicts, columns=ISSUE_COLUMNS)

    result["manifests"] = pd.concat([result["manifests"], bulk["manifests"]], ignore_index=True)
    result["records"] = pd.concat([result["records"], bulk["records"]], ignore_index=True)
    result["links"] = pd.concat([result["links"], bulk["links"]], ignore_index=True)
    result["process_observations"] = (
        bulk["observations"].copy()
        if result["process_observations"].empty
        else pd.concat([result["process_observations"], bulk["observations"]], ignore_index=True, sort=False)
    )
    result["lineage"] = bulk["lineage"]

    curated = _apply_curated_mappings(result["material"])
    result["material"] = curated["material"]
    for key in ("manifests", "records", "links", "process_observations", "particle_fractions", "index_references"):
        result[key] = _append_frames(result[key], curated[key])

    result["material"] = _register_source_materials(result["material"], bulk["discovered_ids"], bulk["id_stage"])
    if len(result["links"]):
        known = set(result["material"]["material_id"].dropna().astype(str))
        result["links"]["mapping_status"] = result["links"].apply(
            lambda row: "confirmed" if row.get("material_id") in known and row.get("mapping_type") != "bulk_source_identifier" else row.get("mapping_status"),
            axis=1,
        )

    # Source files that contain the same bytes are retained for provenance but
    # reported as duplicates rather than silently merged into one experiment.
    duplicate_hashes = result["manifests"].groupby("source_file_hash")["source_file"].nunique()
    duplicate_hashes = duplicate_hashes[duplicate_hashes > 1]
    for digest, count in duplicate_hashes.items():
        result["issues"] = pd.concat([result["issues"], pd.DataFrame([{
            "issue_id": f"SUP_DUP_{str(digest)[:10]}",
            "severity": "low",
            "check_name": "duplicate_source_files",
            "evidence": f"{int(count)} source paths share file hash {digest}; source evidence is retained once per path.",
            "impact": "Counting source files as independent experiments would inflate evidence volume.",
            "action": "Use source_file_hash to deduplicate provenance when aggregating experiments.",
        }], columns=ISSUE_COLUMNS)], ignore_index=True)

    if not curated["solvent_mapping_confirmed"]:
        result["issues"] = pd.concat([result["issues"], pd.DataFrame([{
            "issue_id": "SUP001",
            "severity": "medium",
            "check_name": "solvent_output_label_ambiguity",
            "evidence": "The MS-Q-260122 protocol labels three solvent outputs inconsistently.",
            "impact": "Post-treatment conditions cannot be safely used as BO features.",
            "action": "Confirm the material-id to solvent-process mapping before promoting these records to BO features.",
        }], columns=ISSUE_COLUMNS)], ignore_index=True)
    if "material_id" in result["particle_fractions"] and any(
        str(value).startswith("MS-Q-251124-")
        for value in result["particle_fractions"]["material_id"].dropna().tolist()
    ):
        result["issues"] = pd.concat([result["issues"], pd.DataFrame([{
            "issue_id": "SUP_SIZE_251124_B",
            "severity": "medium",
            "check_name": "provider_confirmed_size_range_overrides_source_text",
            "evidence": "The TACE source text describes S22/MS-Q-251124-B as 200-500 nm, while the data provider confirmed B as 500-800 nm.",
            "impact": "Using the stale source text would assign the wrong particle-size feature to B.",
            "action": "Use supplement_particle_fraction as the canonical size-range table and retain the TACE text as historical source evidence.",
        }], columns=ISSUE_COLUMNS)], ignore_index=True)
    if "product_stage" in result["process_observations"].columns:
        stage_by_parameter = {
            "post_treatment_solvent_system": "intermediate",
            "ultrasonic_time": "intermediate",
        }
        result["process_observations"]["product_stage"] = result["process_observations"]["product_stage"].fillna(
            result["process_observations"]["parameter_name"].map(stage_by_parameter)
        )
    return result
