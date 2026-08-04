"""Bulk ingestion of every supplemental workbook and Word record.

The primary workbooks are normalized by the legacy readers.  This module keeps
the remaining source material lossless: every non-empty row/block is stored as
source evidence, while recognizable process parameters are also promoted to a
typed observation table.  Observations with uncertain sample mappings remain
auditable and are not silently promoted to BO features.
"""

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

from mg2si.data.supplemental_materials import normalize_supplemental_material_id
from mg2si.io.source_manifest import file_sha256


OBSERVATION_COLUMNS = [
    "source_record_id", "source_file", "source_sheet", "source_row",
    "source_material_id_raw", "material_id", "material_parent_id",
    "mapping_status", "product_stage", "parameter_name", "value_numeric",
    "unit", "value_raw", "confidence", "note",
]
LINEAGE_COLUMNS = [
    "source_record_id", "source_file", "source_sheet", "source_row",
    "child_material_id", "parent_material_id", "transformation_type",
    "mapping_status", "evidence",
]


def _clean(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return None if text in {"", "-", "/", "nan", "None"} else text


def _number(value: str) -> float | None:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", value.replace(",", ""))
    return float(match.group()) if match else None


def _canonical_id(value: str) -> str | None:
    value = _clean(value)
    if not value:
        return None
    value = re.sub(r"^Mg2Si-", "MS-", value, flags=re.IGNORECASE)
    value = value.strip("[](){}，,。；;:：")
    if re.search(r"x{2,}|y{2,}", value, flags=re.IGNORECASE):
        return None
    canonical, _, _ = normalize_supplemental_material_id(value)
    return canonical


def extract_material_ids(text: str) -> list[str]:
    """Extract concrete sample identifiers without treating placeholders as IDs."""
    patterns = [
        r"(?:Mg2Si-)?MS-(?:Q-)?\d{6,8}(?:-[A-Za-z0-9]+)*(?:\s+(?:SHS|Q|ADMS|top|down|mid))?",
        r"(?:Mg2Si-)?MS-Q-\d{6}-[^\s，,。；;:：()\[\]]+",
    ]
    found: list[str] = []
    for pattern in patterns:
        for raw in re.findall(pattern, text, flags=re.IGNORECASE):
            canonical = _canonical_id(raw)
            if canonical and canonical not in found:
                found.append(canonical)
    return found


def _stage(parameter_name: str, text: str) -> str:
    if parameter_name.startswith("pvp_") or "PVP" in text.upper():
        return "finished_product"
    if parameter_name.startswith("synthesis_") or parameter_name.startswith("thermal_"):
        return "raw_material"
    return "intermediate"


def _observations(text: str) -> list[dict[str, Any]]:
    """Parse high-signal process values; retain raw text for every interpretation."""
    text = _clean(text) or ""
    results: list[dict[str, Any]] = []

    def add(name: str, value: float | None, unit: str | None, raw: str, note: str, confidence: str = "medium") -> None:
        results.append({
            "parameter_name": name,
            "value_numeric": value,
            "unit": unit,
            "value_raw": raw,
            "product_stage": _stage(name, text),
            "confidence": confidence,
            "note": note,
        })

    patterns = [
        (r"球料比\s*[:：]?\s*(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)", "ball_to_material_ratio", "ratio", "Ball-to-material ratio."),
        (r"(\d+(?:\.\d+)?)\s*Hz", "milling_frequency", "Hz", "Ball-mill frequency."),
        (r"(\d+(?:\.\d+)?)\s*rpm", "milling_speed", "rpm", "Ball-mill or centrifuge speed."),
        (r"(\d+(?:\.\d+)?)\s*min\s*运行", "milling_run_segment", "min", "Active milling segment."),
        (r"(\d+(?:\.\d+)?)\s*min\s*停止", "milling_pause_segment", "min", "Pause segment."),
        (r"(\d+(?:\.\d+)?)\s*个?循环", "milling_cycle_count", "count", "Milling cycle count."),
        (r"总运行时间\s*(\d+(?:\.\d+)?)\s*min", "milling_total_runtime", "min", "Total active milling time."),
        (r"抽(?:放)?气\s*(\d+(?:\.\d+)?)\s*min", "pre_milling_vacuum", "min", "Evacuation before milling."),
        (r"每罐投料(?:量)?\s*[:：]?\s*(\d+(?:\.\d+)?)\s*g", "feed_mass_per_batch", "g", "Feed mass per milling batch."),
        (r"球磨投料量\s*[:：]?\s*(\d+(?:\.\d+)?)\s*g", "feed_mass_total", "g", "Total material feed reported in protocol."),
        (r"材料.*?PVP.*?(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)", "material_to_pvp_ratio", "ratio", "Material-to-PVP mass ratio."),
        (r"PVP.*?(?:分子量|MW)\s*[:：]?\s*(\d+(?:\.\d+)?)\s*[-~至]\s*(\d+(?:\.\d+)?)", "pvp_mw", "g/mol", "PVP molecular-weight range; numeric value is midpoint."),
        (r"超声(?:约|时间)?\s*(\d+(?:\.\d+)?)\s*(h|小时|min|分钟)", "ultrasonic_time", None, "Ultrasonication duration."),
        (r"(\d+(?:\.\d+)?)\s*mL", "liquid_volume", "mL", "Reported liquid volume."),
        (r"(\d+(?:\.\d+)?)\s*k\s*[,，]?\s*(\d+(?:\.\d+)?)\s*min", "centrifuge_speed", "rpm", "Centrifuge speed; k means 1000 rpm."),
        (r"(?:负压|压力).*?(\d+(?:\.\d+)?)\s*[-~至]\s*(\d+(?:\.\d+)?)\s*atm", "initial_pressure_atm", "atm", "Reported pressure range; numeric value is midpoint."),
    ]
    for pattern, name, unit, note in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            raw = match.group(0)
            numbers = [_number(value) for value in match.groups() if _number(value) is not None]
            if name == "ball_to_material_ratio" and len(numbers) >= 2:
                value = numbers[0] / numbers[1] if numbers[1] else None
            elif name == "pvp_mw" and len(numbers) >= 2:
                value = float(np.mean(numbers[:2]))
            elif name == "initial_pressure_atm" and len(numbers) >= 2:
                value = float(np.mean(numbers[:2]))
            elif name == "centrifuge_speed" and len(numbers) >= 1:
                value = numbers[0] * 1000.0
            else:
                value = numbers[0] if numbers else None
            if name == "ultrasonic_time" and len(match.groups()) >= 2 and match.group(2).lower() in {"min", "分钟"}:
                value = value / 60.0 if value is not None else None
                unit = "h"
            add(name, value, unit, raw, note)

    for pattern, name, note in [
        (r"气氛\s*[:：]?\s*([^，,；;。]+)", "protective_atmosphere", "Protective atmosphere description."),
        (r"温控程序\s*[:：]?\s*([^，,；;。]+)", "thermal_program", "Thermal program retained as raw protocol text."),
        (r"(?:使用|采用)([^，,；;。]*(?:乙醇|氨水)[^，,；;。]*)", "post_treatment_solvent", "Post-treatment solvent system."),
    ]:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            add(name, None, None, match.group(0), note, confidence="low")
    return results


def _docx_blocks(path: Path) -> list[tuple[str, str]]:
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with ZipFile(path) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
    blocks: list[tuple[str, str]] = []
    for paragraph in root.findall(".//w:p", namespace):
        text = "".join(node.text or "" for node in paragraph.findall(".//w:t", namespace)).strip()
        if text:
            blocks.append(("document_paragraph", text))
    for table in root.findall(".//w:tbl", namespace):
        for row in table.findall(".//w:tr", namespace):
            cells = []
            for cell in row.findall("./w:tc", namespace):
                cells.append("".join(node.text or "" for node in cell.findall(".//w:t", namespace)).strip())
            text = " | ".join(cell for cell in cells if cell)
            if text:
                blocks.append(("document_table_row", text))
    return blocks


def _lineage_for_ids(ids: list[str], source_record_id: str, source_file: str, sheet: str, row: int, text: str) -> list[dict[str, Any]]:
    pairs: list[tuple[str, str, str]] = []
    for child in ids:
        match = re.match(r"MS-Q-260113-([ABC])$", child, flags=re.IGNORECASE)
        if match:
            pairs.append((child, "MS-260110-SHS", "differential_centrifugation_fraction"))
        elif re.match(r"MS-Q-251124-[ABCD]$", child, flags=re.IGNORECASE):
            pairs.append((child, "MS-20251201-商业化", "ultrasonication_size_fraction"))
        elif child == "MS-P-251114":
            pairs.append((child, "MS-Q-251114", "PVP_modification"))
        elif child.endswith("-K60"):
            pairs.append((child, child[:-4].rstrip("-"), "PVP_modification"))
    return [{
        "source_record_id": source_record_id,
        "source_file": source_file,
        "source_sheet": sheet,
        "source_row": row,
        "child_material_id": child,
        "parent_material_id": parent,
        "transformation_type": transform,
        "mapping_status": "confirmed" if child.startswith(("MS-Q-260113-", "MS-Q-251124-")) else "needs_confirmation",
        "evidence": text,
    } for child, parent, transform in pairs]


def bulk_ingest_sources(root: Path, excluded_paths: set[Path]) -> dict[str, pd.DataFrame | set[str] | dict[str, str]]:
    manifests: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    lineage: list[dict[str, Any]] = []
    discovered: set[str] = set()
    id_stage: dict[str, str] = {}
    excluded = {path.resolve() for path in excluded_paths}
    sources = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".xlsx", ".xls", ".docx"} and path.resolve() not in excluded)

    for path in sources:
        digest = file_sha256(path)
        source_file = path.relative_to(root).as_posix()
        if path.suffix.lower() in {".xlsx", ".xls"}:
            blocks: list[tuple[str, str]] = []
            workbook = pd.ExcelFile(path)
            for sheet in workbook.sheet_names:
                raw = pd.read_excel(path, sheet_name=sheet, header=None)
                manifests.append({"source_file": source_file, "source_file_hash": digest, "source_sheet": sheet, "rows": int(raw.shape[0]), "columns": int(raw.shape[1]), "schema_version": "1.2-bulk"})
                for row_number, row in raw.iterrows():
                    values = [_clean(value) for value in row.tolist()]
                    if not any(value is not None for value in values):
                        continue
                    blocks.append((sheet, " | ".join(value for value in values if value is not None)))
        else:
            blocks = [(kind, text) for kind, text in _docx_blocks(path)]
            manifests.append({"source_file": source_file, "source_file_hash": digest, "source_sheet": "document", "rows": len(blocks), "columns": 1, "schema_version": "1.2-bulk"})

        for row_number, (sheet, text) in enumerate(blocks, start=1):
            source_record_id = f"{digest[:12]}:{sheet}:{row_number}"
            ids = extract_material_ids(text)
            discovered.update(ids)
            records.append({
                "source_record_id": source_record_id,
                "source_file": source_file,
                "source_file_hash": digest,
                "source_sheet": sheet,
                "source_row": row_number,
                "record_role": "document_table_row" if sheet == "document_table_row" else "data",
                "payload_json": json.dumps({"text": text, "material_ids": ids}, ensure_ascii=False, sort_keys=True),
            })
            for material_id in ids:
                links.append({
                    "source_file": source_file,
                    "source_sheet": sheet,
                    "source_row": row_number,
                    "source_material_id_raw": material_id,
                    "material_id": material_id,
                    "material_parent_id": None,
                    "mapping_type": "bulk_source_identifier",
                    "mapping_status": "needs_confirmation",
                    "mapping_basis": "identifier_extracted_from_source_record",
                })
                if "PVP" in text.upper() or material_id.endswith("-P") or material_id.endswith("-K60"):
                    id_stage[material_id] = "finished_product"
                elif "烧结" in text or "管式炉" in text or material_id.endswith("-SHS"):
                    id_stage.setdefault(material_id, "raw_material")
                else:
                    id_stage.setdefault(material_id, "intermediate")
            observation_ids = ids or [None]
            for material_id in observation_ids:
                for observation in _observations(text):
                    observations.append({
                        "source_record_id": source_record_id,
                        "source_file": source_file,
                        "source_sheet": sheet,
                        "source_row": row_number,
                        "source_material_id_raw": material_id,
                        "material_id": material_id,
                        "material_parent_id": None,
                        "mapping_status": "needs_confirmation" if material_id else "unmapped",
                        **observation,
                    })
            lineage.extend(_lineage_for_ids(ids, source_record_id, source_file, sheet, row_number, text))

    return {
        "manifests": pd.DataFrame(manifests),
        "records": pd.DataFrame(records),
        "links": pd.DataFrame(links),
        "observations": pd.DataFrame(observations, columns=OBSERVATION_COLUMNS),
        "lineage": pd.DataFrame(lineage, columns=LINEAGE_COLUMNS),
        "discovered_ids": discovered,
        "id_stage": id_stage,
    }
