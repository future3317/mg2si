from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
XLSX_FILES = sorted(ROOT.glob("*.xlsx"))
if len(XLSX_FILES) < 2:
    raise FileNotFoundError("需要两个 Excel 源文件")
MG_PATH, TACE_PATH = XLSX_FILES[0], XLSX_FILES[1]


def clean_text(value):
    if pd.isna(value):
        return np.nan
    text = re.sub(r"\s+", " ", str(value)).strip()
    if text in {"", "-", "—", "–", "/", "\\", "nan", "None"}:
        return np.nan
    return text


def first_number(value):
    value = clean_text(value)
    if pd.isna(value):
        return np.nan
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(match.group(0)) if match else np.nan


def midpoint_number(value):
    value = clean_text(value)
    if pd.isna(value):
        return np.nan
    text = str(value).replace(",", "")
    numbers = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)]
    if not numbers:
        return np.nan
    if len(numbers) >= 2 and any(mark in text for mark in ["-", "~", "至", "到"]):
        return float(np.mean(numbers[:2]))
    return numbers[0]


def parse_date(value):
    value = clean_text(value)
    if pd.isna(value):
        return np.nan
    parsed = pd.to_datetime(value, errors="coerce")
    return parsed.strftime("%Y-%m-%d") if not pd.isna(parsed) else np.nan


def parse_yes_no(value):
    value = clean_text(value)
    if pd.isna(value):
        return np.nan
    if str(value) in {"是", "有", "Y", "yes", "Yes", "1"}:
        return 1
    if str(value) in {"否", "无", "N", "no", "No", "0"}:
        return 0
    return np.nan


def parse_pvp_mw(value):
    value = clean_text(value)
    if pd.isna(value):
        return np.nan
    number = first_number(value)
    if pd.isna(number):
        return np.nan
    upper = str(value).upper()
    if "W" in upper:
        return number * 10000.0
    if "K" in upper:
        return number * 1000.0
    if "M" in upper:
        return number * 1000000.0
    return number


def parse_ratio(value):
    value = clean_text(value)
    if pd.isna(value):
        return np.nan
    numbers = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", str(value))]
    if len(numbers) >= 2 and numbers[1] != 0:
        return numbers[0] / numbers[1]
    return first_number(value)


def parse_pressure(value):
    value = clean_text(value)
    if pd.isna(value):
        return np.nan
    if "常压" in str(value):
        return 1.0
    return midpoint_number(value)


def parse_hours(value):
    value = clean_text(value)
    if pd.isna(value):
        return np.nan
    text = str(value).lower()
    numbers = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)]
    if not numbers:
        return np.nan
    if "min" in text and "h" not in text:
        return numbers[0] / 60.0
    if len(numbers) >= 2 and any(mark in text for mark in ["-", "~", "至", "到"]):
        return float(np.mean(numbers[:2]))
    return numbers[0]


def normalize_material_id(value):
    value = clean_text(value)
    if pd.isna(value):
        return np.nan
    text = str(value)
    text = re.sub(r"\([^)]*\)|（[^）]*）", "", text)
    text = re.sub(r"\s+S\d+.*$", "", text)
    text = re.sub(r"\s+(top|down|mid|上层|下层|中层)$", "", text, flags=re.IGNORECASE)
    return text.strip()


def read_material_master(path):
    xl = pd.ExcelFile(path)
    sample_raw = pd.read_excel(path, sheet_name=xl.sheet_names[0], header=None)
    sample = sample_raw.iloc[1:].copy()
    sample.columns = [
        "material_id", "sample_date", "material_source", "synthesis_method", "color",
        "is_layered", "layer_position", "note", "mg_mass_mg", "si_mass_mg",
        "molar_ratio_Mg_to_Si", "crucible_area", "vacuum_cycle", "vacuum_time_min",
        "protective_gas", "initial_pressure_atm_raw", "max_temp_c", "hold_time_min",
        "milling_mode", "ball_mill_ratio_raw", "milling_cycle_time", "pvp_mw_raw",
        "pvp_ratio_material_to_pvp_raw", "post_treatment", "ultrasonic_time_raw",
    ]
    for column in sample.columns:
        sample[column] = sample[column].map(clean_text)
    sample["material_id"] = sample["material_id"].map(normalize_material_id)
    sample = sample[sample["material_id"].notna() & (sample["material_id"] != "范围")].copy()
    sample["sample_date"] = sample["sample_date"].map(parse_date)
    for column in ["mg_mass_mg", "si_mass_mg", "molar_ratio_Mg_to_Si", "vacuum_cycle", "vacuum_time_min", "max_temp_c", "hold_time_min", "milling_cycle_time"]:
        sample[column] = sample[column].map(midpoint_number)
    sample["initial_pressure_atm"] = sample["initial_pressure_atm_raw"].map(parse_pressure)
    sample["ball_to_material_ratio"] = sample["ball_mill_ratio_raw"].map(parse_ratio)
    sample["pvp_mw"] = sample["pvp_mw_raw"].map(parse_pvp_mw)
    sample["pvp_material_to_pvp_ratio"] = sample["pvp_ratio_material_to_pvp_raw"].map(parse_ratio)
    sample["ultrasonic_time_h"] = sample["ultrasonic_time_raw"].map(parse_hours)
    sample["has_sample_info"] = 1

    specs = [
        (["xrd_match", "mg2si_purity_pct", "peak_ratio_raw", "grain_size_nm", "particle_distribution", "hrtem_lattice", "saed", "dls_size_nm", "pdi", "zeta_potential_mv", "material_kill_500ppm", "material_kill_250ppm", "material_kill_125ppm"], "has_structure"),
        (["xps_mg_1s", "xps_si_2p", "xps_o_1s", "mg_si_atom_ratio", "mgo_ratio", "siox_ratio", "oxide_thickness", "defect_state", "other_surface_features"], "has_surface_chemistry"),
        (["mg2si_purity_score", "size_score", "dispersion_score", "tumor_kill_score", "safety_score", "overall_score", "entered_animal", "quality_note"], "has_screening"),
        (["cell_line", "tumor_cell_type", "normal_cell_type", "treat_conc_ppm", "treat_time_h", "ic50_tumor", "ic50_normal", "safety_index", "ros_level", "apoptosis_rate"], "has_material_biology"),
    ]
    tables = [sample]
    for sheet_name, (names, flag) in zip(xl.sheet_names[1:], specs):
        raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
        tab = raw.iloc[1:, : len(names) + 1].copy()
        tab.columns = ["material_id"] + names
        for column in tab.columns:
            tab[column] = tab[column].map(clean_text)
        tab["material_id"] = tab["material_id"].map(normalize_material_id)
        tab = tab[tab["material_id"].notna()].drop_duplicates("material_id")
        for column in names:
            if column not in {"peak_ratio_raw", "particle_distribution", "hrtem_lattice", "saed", "defect_state", "other_surface_features", "quality_note", "cell_line", "tumor_cell_type", "normal_cell_type"}:
                tab[column] = tab[column].map(midpoint_number)
        tab[flag] = 1
        tables.append(tab)

    master = tables[0]
    for tab in tables[1:]:
        master = master.merge(tab, how="outer", on="material_id", suffixes=("", "_duplicate"))
        duplicate_columns = [c for c in master.columns if c.endswith("_duplicate")]
        master = master.drop(columns=duplicate_columns)
    return master


def read_tace_summary(path):
    xl = pd.ExcelFile(path)
    raw = pd.read_excel(path, sheet_name=xl.sheet_names[0], header=None)
    columns = [
        "tace_sample_id_raw", "exp_index", "sample_stage", "material_source", "is_milled_raw",
        "milled_size_nm_raw", "post_treatment", "pvp_mw_raw", "pvp_ratio_raw", "repeat_batch",
        "kill_assay_date", "tumor_cell_line", "y_tumor_500ppm", "y_tumor_250ppm", "y_tumor_125ppm",
        "selection_cell", "safety_assay_date", "y_normal_500ppm", "y_normal_250ppm", "y_normal_125ppm",
        "y_selectivity_500ppm_source", "y_selectivity_250ppm_source", "y_selectivity_125ppm_source", "remark",
    ]
    data = raw.iloc[1:, :24].copy()
    data.columns = columns
    for column in data.columns:
        data[column] = data[column].map(clean_text)
    data = data[data["tace_sample_id_raw"].notna()].copy()
    data["tace_sample_id"] = data["tace_sample_id_raw"].map(normalize_material_id)
    data["exp_index"] = data["exp_index"].map(clean_text)
    data["sample_stage"] = data["sample_stage"].map(clean_text)
    data["material_source"] = data["material_source"].map(clean_text)
    data["post_treatment"] = data["post_treatment"].map(clean_text)
    data["tumor_cell_line"] = data["tumor_cell_line"].map(clean_text)
    data["remark"] = data["remark"].map(clean_text)
    data["is_milled"] = data["is_milled_raw"].map(parse_yes_no)
    data["milled_size_nm"] = data["milled_size_nm_raw"].map(midpoint_number)
    data["pvp_mw"] = data["pvp_mw_raw"].map(parse_pvp_mw)
    data["material_to_pvp_ratio"] = data["pvp_ratio_raw"].map(parse_ratio)
    data["repeat_batch"] = data["repeat_batch"].map(first_number).astype("Int64")
    data["kill_assay_date"] = data["kill_assay_date"].map(parse_date)
    data["safety_assay_date"] = data["safety_assay_date"].map(parse_date)
    for column in [
        "y_tumor_500ppm", "y_tumor_250ppm", "y_tumor_125ppm", "y_normal_500ppm", "y_normal_250ppm",
        "y_normal_125ppm", "y_selectivity_500ppm_source", "y_selectivity_250ppm_source", "y_selectivity_125ppm_source",
    ]:
        data[column] = data[column].map(first_number)
    return data


def choose_material_id(tace_id, stage, material_ids):
    if pd.isna(tace_id):
        return np.nan, "unmatched", "missing_sample_id", ""
    raw = str(tace_id)
    normalized = normalize_material_id(raw)
    ids = sorted(material_ids)
    exact = [mid for mid in ids if normalized == normalize_material_id(mid)]
    if len(exact) == 1:
        return exact[0], "exact", "normalized_exact_id", ";".join(exact)

    candidates = [mid for mid in ids if normalize_material_id(mid) and normalize_material_id(mid) in normalized]
    if normalized.startswith("MS-"):
        stem_match = re.match(r"(MS-\d{6})", normalized)
        if stem_match:
            stem = stem_match.group(1)
            candidates += [mid for mid in ids if str(mid).startswith(stem + "-")]
    candidates = sorted(set(candidates))
    if not candidates:
        aliases = {
            "MS-251016": ["MS-251016-SHS", "MS-251016-Q"],
            "MS-251215": ["MS-251215-Q"],
            "MS-260514": ["MS-260514-SHS"],
        }
        candidates = [mid for mid in aliases.get(normalized, []) if mid in material_ids]
    if not candidates:
        return np.nan, "unmatched", "no_material_id_candidate", ""

    stage_text = "" if pd.isna(stage) else str(stage)
    non_shs = [mid for mid in candidates if "-SHS" not in str(mid)]
    shs = [mid for mid in candidates if "-SHS" in str(mid)]
    if stage_text == "原料" and shs:
        selected = sorted(shs, key=len)[0]
        basis = "stage_prefers_SHS_parent"
    elif stage_text in {"半成品I", "半成品II", "成品"} and non_shs:
        selected = sorted(non_shs, key=len)[0]
        basis = "stage_prefers_processed_material"
    elif len(candidates) == 1:
        selected = candidates[0]
        basis = "unique_prefix_or_parent_candidate"
    else:
        return np.nan, "ambiguous", "multiple_material_candidates", ";".join(candidates)
    mapping_type = "layer_or_parent_alias" if selected != normalized else "alias"
    return selected, mapping_type, basis, ";".join(candidates)


def infer_parent_id(material_id, material_ids):
    if pd.isna(material_id):
        return np.nan
    material_id = str(material_id)
    if "-SHS" in material_id:
        return material_id
    stem = material_id.split("-Q")[0] if "-Q" in material_id else material_id
    candidates = [mid for mid in material_ids if str(mid).startswith(stem) and "-SHS" in str(mid)]
    return sorted(candidates, key=len)[0] if candidates else np.nan


def make_long_cell_table(data):
    rows = []
    for row_number, row in data.iterrows():
        base = {
            "cell_record_id": f"TACE_summary_row_{row_number + 2}",
            "tace_sample_id_raw": row["tace_sample_id_raw"],
            "tace_sample_id": row["tace_sample_id"],
            "exp_index": row["exp_index"],
            "sample_stage": row["sample_stage"],
            "material_source": row["material_source"],
            "is_milled": row["is_milled"],
            "milled_size_nm_raw": row["milled_size_nm_raw"],
            "milled_size_nm": row["milled_size_nm"],
            "post_treatment": row["post_treatment"],
            "pvp_mw_raw": row["pvp_mw_raw"],
            "pvp_mw": row["pvp_mw"],
            "pvp_ratio_raw": row["pvp_ratio_raw"],
            "material_to_pvp_ratio": row["material_to_pvp_ratio"],
            "repeat_batch": row["repeat_batch"],
            "kill_assay_date": row["kill_assay_date"],
            "safety_assay_date": row["safety_assay_date"],
            "tumor_cell_line": row["tumor_cell_line"],
            "remark": row["remark"],
            "source_sheet": "数据汇总",
        }
        for concentration in [500, 250, 125]:
            suffix = f"{concentration}ppm"
            tumor = row[f"y_tumor_{suffix}"]
            normal = row[f"y_normal_{suffix}"]
            source_selectivity = row[f"y_selectivity_{suffix}_source"]
            calculated = np.nan
            if pd.notna(tumor) and pd.notna(normal) and normal != 100:
                calculated = (100.0 - tumor) / (100.0 - normal)
            flags = []
            if pd.isna(tumor):
                flags.append("missing_tumor_viability")
            if pd.isna(normal):
                flags.append("missing_normal_viability")
            if pd.notna(tumor) and not 0 <= tumor <= 100:
                flags.append("tumor_viability_out_of_0_100")
            if pd.notna(normal) and not 0 <= normal <= 100:
                flags.append("normal_viability_out_of_0_100")
            rows.append({
                **base,
                "concentration_ppm": concentration,
                "y_tumor_viability_pct": tumor,
                "y_normal_viability_pct": normal,
                "y_selectivity_index": calculated,
                "y_selectivity_index_source": source_selectivity,
                "target_complete": int(pd.notna(tumor) and pd.notna(normal)),
                "target_quality_flag": ";".join(flags) if flags else "ok",
            })
    result = pd.DataFrame(rows)
    result["workflow_branch"] = result["material_source"].map(
        lambda x: "synthetic" if isinstance(x, str) and "合成" in x else ("commercial" if isinstance(x, str) and "商业" in x else "unknown")
    )
    result["synthesis_required"] = result["workflow_branch"].map({"synthetic": 1, "commercial": 0})
    result["synthesis_feature_status"] = result["workflow_branch"].map(
        {"synthetic": "applicable", "commercial": "not_applicable", "unknown": "unknown"}
    )
    result["experiment_id"] = (
        result["cell_record_id"].astype(str) + "|" + result["exp_index"].fillna("").astype(str)
        + "|r" + result["repeat_batch"].astype(str)
    )
    return result


def main():
    material = read_material_master(MG_PATH)
    tace = read_tace_summary(TACE_PATH)
    cells = make_long_cell_table(tace)
    material_ids = set(material["material_id"].dropna())

    resolved = cells.apply(lambda row: choose_material_id(row["tace_sample_id"], row["sample_stage"], material_ids), axis=1)
    cells["material_id"] = resolved.map(lambda x: x[0])
    cells["mapping_type"] = resolved.map(lambda x: x[1])
    cells["mapping_basis"] = resolved.map(lambda x: x[2])
    cells["mapping_candidates"] = resolved.map(lambda x: x[3])
    cells["material_parent_id"] = cells["material_id"].map(lambda x: infer_parent_id(x, material_ids))

    material_prefixed = material.add_prefix("material_")
    joint = cells.merge(material_prefixed, how="left", left_on="material_id", right_on="material_material_id")
    feature_columns = [
        "experiment_id", "cell_record_id", "tace_sample_id_raw", "tace_sample_id", "material_id", "material_parent_id",
        "mapping_type", "mapping_basis", "sample_stage", "material_source", "is_milled", "milled_size_nm_raw",
        "milled_size_nm", "post_treatment", "pvp_mw_raw", "pvp_mw", "pvp_ratio_raw", "material_to_pvp_ratio",
        "repeat_batch", "concentration_ppm", "tumor_cell_line",
        "workflow_branch", "synthesis_required", "synthesis_feature_status",
        "material_sample_date", "material_material_source", "material_synthesis_method", "material_is_layered",
        "material_mg_mass_mg", "material_si_mass_mg", "material_molar_ratio_Mg_to_Si", "material_vacuum_cycle",
        "material_vacuum_time_min", "material_initial_pressure_atm", "material_max_temp_c", "material_hold_time_min",
        "material_protective_gas", "material_milling_mode", "material_ball_to_material_ratio",
        "material_milling_cycle_time", "material_pvp_mw", "material_pvp_material_to_pvp_ratio",
        "material_post_treatment", "material_ultrasonic_time_h", "material_grain_size_nm", "material_dls_size_nm",
        "material_pdi", "material_zeta_potential_mv",
    ]
    features = joint[feature_columns].copy()
    target_columns = [
        "experiment_id", "cell_record_id", "tace_sample_id", "material_id", "material_parent_id",
        "mapping_type", "workflow_branch", "synthesis_required", "synthesis_feature_status", "concentration_ppm",
        "y_tumor_viability_pct", "y_normal_viability_pct",
        "y_selectivity_index", "y_selectivity_index_source", "target_complete", "target_quality_flag",
    ]
    targets = joint[target_columns].copy()
    targets["objective_tumor_viability"] = "minimize"
    targets["objective_normal_viability"] = "maximize"

    mapping = cells[["tace_sample_id_raw", "tace_sample_id", "material_id", "material_parent_id", "mapping_type", "mapping_basis", "mapping_candidates", "sample_stage"]].drop_duplicates()
    mapping["cell_row_count"] = mapping["tace_sample_id"].map(cells["tace_sample_id"].value_counts())
    coverage_columns = [
        "material_mg_mass_mg", "material_si_mass_mg", "material_molar_ratio_Mg_to_Si", "material_vacuum_cycle",
        "material_vacuum_time_min", "material_initial_pressure_atm", "material_max_temp_c", "material_hold_time_min",
        "material_ball_to_material_ratio", "material_milling_cycle_time", "material_grain_size_nm", "material_dls_size_nm",
    ]
    coverage = pd.DataFrame({
        "feature": coverage_columns,
        "non_null_rows": [int(joint[c].notna().sum()) for c in coverage_columns],
        "total_cell_rows": [int(len(joint)) for _ in coverage_columns],
        "coverage_rate": [float(joint[c].notna().mean()) for c in coverage_columns],
    })

    material.to_csv(ROOT / "bo_material_master.csv", index=False, encoding="utf-8-sig")
    material.to_csv(ROOT / "bo_meta_sample.csv", index=False, encoding="utf-8-sig")
    cells.to_csv(ROOT / "bo_cell_long.csv", index=False, encoding="utf-8-sig")
    cells.to_csv(ROOT / "bo_experiment_long.csv", index=False, encoding="utf-8-sig")
    features.to_csv(ROOT / "bo_features.csv", index=False, encoding="utf-8-sig")
    targets.to_csv(ROOT / "bo_targets.csv", index=False, encoding="utf-8-sig")
    joint.to_csv(ROOT / "bo_joint_dataset.csv", index=False, encoding="utf-8-sig")
    mapping.to_csv(ROOT / "sample_id_mapping_review.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(ROOT / "bo_feature_coverage.csv", index=False, encoding="utf-8-sig")
    print({
        "material_master_rows": int(len(material)),
        "tace_summary_rows": int(len(tace)),
        "cell_long_rows": int(len(cells)),
        "complete_two_target_rows": int(targets["target_complete"].sum()),
        "mapping_type_counts": mapping["mapping_type"].value_counts(dropna=False).to_dict(),
        "mapped_cell_rows": int(cells["material_id"].notna().sum()),
        "coverage": {c: round(float(joint[c].notna().mean()), 3) for c in coverage_columns},
    })


if __name__ == "__main__":
    main()
