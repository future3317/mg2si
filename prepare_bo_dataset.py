import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
XLSX_FILES = sorted(ROOT.glob("*.xlsx"))
if len(XLSX_FILES) < 2:
    raise FileNotFoundError("需要两个 Excel 源文件")


def clean_text(value):
    if pd.isna(value):
        return np.nan
    text = re.sub(r"\s+", " ", str(value)).strip()
    if text in {"", "-", "—", "–", "/", "\\", "nan", "None"}:
        return np.nan
    return text


def normalize_id(value):
    value = clean_text(value)
    if pd.isna(value):
        return np.nan
    return value


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
    nums = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", str(value).replace(",", ""))]
    if not nums:
        return np.nan
    if len(nums) >= 2 and any(mark in str(value) for mark in ["-", "~", "至", "到"]):
        return float(np.mean(nums[:2]))
    return nums[0]


def parse_date(value):
    value = clean_text(value)
    if pd.isna(value):
        return np.nan
    parsed = pd.to_datetime(value, errors="coerce")
    return parsed.strftime("%Y-%m-%d") if not pd.isna(parsed) else np.nan


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
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", str(value))]
    if len(nums) >= 2 and nums[1] != 0:
        return nums[0] / nums[1]
    return first_number(value)


def parse_hours(value):
    value = clean_text(value)
    if pd.isna(value):
        return np.nan
    text = str(value).lower()
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)]
    if not nums:
        return np.nan
    if "min" in text and "h" not in text:
        return nums[0] / 60.0
    if len(nums) >= 2 and any(mark in text for mark in ["-", "~", "至", "到"]):
        return float(np.mean(nums[:2]))
    return nums[0]


def parse_pressure(value):
    value = clean_text(value)
    if pd.isna(value):
        return np.nan
    if "常压" in str(value):
        return 1.0
    return midpoint_number(value)


def parse_yes_no(value):
    value = clean_text(value)
    if pd.isna(value):
        return np.nan
    if str(value) in {"是", "有", "Y", "yes", "Yes", "1"}:
        return 1
    if str(value) in {"否", "无", "N", "no", "No", "0"}:
        return 0
    return np.nan


def normalize_frame(frame):
    frame = frame.copy()
    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = frame[column].map(clean_text)
    return frame


def read_mg2si_metadata(path):
    xl = pd.ExcelFile(path)
    sample_raw = pd.read_excel(path, sheet_name=xl.sheet_names[0], header=None)
    sample = sample_raw.iloc[1:].copy()
    sample.columns = [
        "sample_id", "sample_date", "material_source", "synthesis_method", "color",
        "is_layered", "layer_position", "note", "mg_mass_mg", "si_mass_mg",
        "molar_ratio_Mg_to_Si", "crucible_area", "vacuum_cycle", "vacuum_time_min",
        "protective_gas", "initial_pressure_atm_raw", "max_temp_c", "hold_time_min",
        "milling_mode", "ball_mill_ratio_raw", "milling_cycle_time", "pvp_mw_raw",
        "pvp_ratio_material_to_pvp_raw", "post_treatment", "ultrasonic_time_raw",
    ]
    sample = normalize_frame(sample)
    sample["sample_id"] = sample["sample_id"].map(normalize_id)
    sample = sample[sample["sample_id"].notna() & (sample["sample_id"] != "范围")].copy()
    sample["sample_date"] = sample["sample_date"].map(parse_date)
    for column in ["mg_mass_mg", "si_mass_mg", "molar_ratio_Mg_to_Si", "vacuum_cycle", "vacuum_time_min", "max_temp_c", "hold_time_min", "milling_cycle_time"]:
        sample[column] = sample[column].map(midpoint_number)
    sample["initial_pressure_atm"] = sample["initial_pressure_atm_raw"].map(parse_pressure)
    sample["ball_to_material_ratio"] = sample["ball_mill_ratio_raw"].map(parse_ratio)
    sample["pvp_mw"] = sample["pvp_mw_raw"].map(parse_pvp_mw)
    sample["pvp_material_to_pvp_ratio"] = sample["pvp_ratio_material_to_pvp_raw"].map(parse_ratio)
    sample["ultrasonic_time_h"] = sample["ultrasonic_time_raw"].map(parse_hours)

    tab_specs = [
        (["xrd_match", "mg2si_purity_pct", "peak_ratio_raw", "grain_size_nm", "particle_distribution", "hrtem_lattice", "saed", "dls_size_nm", "pdi", "zeta_potential_mv", "tumor_kill_500ppm", "tumor_kill_250ppm", "tumor_kill_125ppm"]),
        (["xps_mg_1s", "xps_si_2p", "xps_o_1s", "mg_si_atom_ratio", "mgo_ratio", "siox_ratio", "oxide_thickness", "defect_state", "other_surface_features"]),
        (["mg2si_purity_score", "size_score", "dispersion_score", "tumor_kill_score", "safety_score", "overall_score", "entered_animal", "quality_note"]),
        (["cell_line", "tumor_cell_type", "normal_cell_type", "treat_conc_ppm", "treat_time_h", "ic50_tumor", "ic50_normal", "safety_index", "ros_level", "apoptosis_rate"]),
    ]
    for sheet_name, names in zip(xl.sheet_names[1:], tab_specs):
        tab = pd.read_excel(path, sheet_name=sheet_name, header=None)
        tab = tab.iloc[1:].copy()
        tab = tab.iloc[:, : len(names) + 1]
        tab.columns = ["sample_id"] + names
        tab = normalize_frame(tab)
        tab["sample_id"] = tab["sample_id"].map(normalize_id)
        tab = tab[tab["sample_id"].notna()].drop_duplicates("sample_id")
        for column in names:
            if column not in {"peak_ratio_raw", "particle_distribution", "hrtem_lattice", "saed", "defect_state", "other_surface_features", "quality_note", "cell_line", "tumor_cell_type", "normal_cell_type"}:
                tab[column] = tab[column].map(midpoint_number)
        sample = sample.merge(tab, how="left", on="sample_id", suffixes=("", "_dup"))
    sample = sample.loc[:, ~sample.columns.str.endswith("_dup")]
    return sample


def read_tace_long(path):
    xl = pd.ExcelFile(path)
    rows = []
    columns = [
        "sample_id_raw", "exp_index", "sample_stage", "material_source", "is_milled_raw",
        "milled_size_nm_raw", "post_treatment", "pvp_mw_raw", "pvp_ratio_raw", "repeat_batch",
        "kill_assay_date", "tumor_cell_line", "y_tumor_500ppm", "y_tumor_250ppm", "y_tumor_125ppm",
        "safety_assay_date", "y_normal_500ppm", "y_normal_250ppm", "y_normal_125ppm",
        "y_selectivity_500ppm_source", "y_selectivity_250ppm_source", "y_selectivity_125ppm_source", "remark",
    ]
    for sheet_name in xl.sheet_names[1:-1]:
        raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
        data = raw.iloc[1:, :23].copy()
        data.columns = columns
        data = normalize_frame(data)
        data["source_sheet"] = sheet_name
        data = data[data["sample_id_raw"].notna()].copy()
        rows.append(data)
    data = pd.concat(rows, ignore_index=True)
    data["sample_id"] = data["sample_id_raw"].map(normalize_id)
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

    base_key = (
        data["source_sheet"].fillna("").astype(str) + "|" + data["exp_index"].fillna("").astype(str)
        + "|" + data["sample_id"].fillna("").astype(str) + "|r" + data["repeat_batch"].astype(str)
    )
    data["experiment_id"] = base_key + "|n" + data.groupby(base_key).cumcount().astype(str)
    long_rows = []
    for _, row in data.iterrows():
        for concentration in [500, 250, 125]:
            suffix = str(concentration) + "ppm"
            tumor = row["y_tumor_" + suffix]
            normal = row["y_normal_" + suffix]
            source_selectivity = row["y_selectivity_" + suffix + "_source"]
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
            if pd.notna(normal) and normal == 100:
                flags.append("selectivity_denominator_zero")
            long_rows.append({
                "experiment_id": row["experiment_id"], "sample_id": row["sample_id"], "sample_id_raw": row["sample_id_raw"],
                "exp_index": row["exp_index"], "sample_stage": row["sample_stage"], "material_source": row["material_source"],
                "is_milled": row["is_milled"], "milled_size_nm_raw": row["milled_size_nm_raw"], "milled_size_nm": row["milled_size_nm"],
                "post_treatment": row["post_treatment"], "pvp_mw_raw": row["pvp_mw_raw"], "pvp_mw": row["pvp_mw"],
                "pvp_ratio_raw": row["pvp_ratio_raw"], "material_to_pvp_ratio": row["material_to_pvp_ratio"],
                "repeat_batch": row["repeat_batch"], "kill_assay_date": row["kill_assay_date"], "safety_assay_date": row["safety_assay_date"],
                "tumor_cell_line": row["tumor_cell_line"], "remark": row["remark"], "source_sheet": row["source_sheet"],
                "concentration_ppm": concentration, "y_tumor_viability_pct": tumor, "y_normal_viability_pct": normal,
                "y_selectivity_index": calculated, "y_selectivity_index_source": source_selectivity,
                "target_complete": int(pd.notna(tumor) and pd.notna(normal)), "target_quality_flag": ";".join(flags) if flags else "ok",
            })
    return pd.DataFrame(long_rows)


def resolve_meta_sample_id(sample_id, meta_ids):
    if pd.isna(sample_id):
        return np.nan, "unmatched", "missing_sample_id"
    sample_id = str(sample_id)
    if sample_id in meta_ids:
        return sample_id, "exact", "normalized_exact_id"
    explicit_aliases = {
        "MS-251016": "MS-251016-Q",
        "MS-251215": "MS-251215-Q",
        "MS-251215 S41工艺": "MS-251215-Q",
    }
    if sample_id in explicit_aliases and explicit_aliases[sample_id] in meta_ids:
        return explicit_aliases[sample_id], "parent_alias", "explicit_process_parent_alias"
    candidates = sorted([meta_id for meta_id in meta_ids if str(meta_id) in sample_id])
    if len(candidates) == 1:
        return candidates[0], "parent_alias", "metadata_id_is_contained_in_stage_id"
    return np.nan, "unmatched", "no_unique_mapping_rule"


def main():
    meta = read_mg2si_metadata(XLSX_FILES[0])
    long = read_tace_long(XLSX_FILES[1])

    meta_ids = set(meta["sample_id"].dropna())
    resolved = long["sample_id"].map(lambda x: resolve_meta_sample_id(x, meta_ids))
    long["meta_source_sample_id"] = resolved.map(lambda x: x[0])
    long["mapping_type"] = resolved.map(lambda x: x[1])
    long["mapping_basis"] = resolved.map(lambda x: x[2])
    long["meta_match_status"] = long["mapping_type"].map(
        {"exact": "exact_match", "parent_alias": "parent_alias_match", "unmatched": "no_match"}
    )
    meta_counts = long.groupby("sample_id", dropna=False).agg(
        stage_row_count=("experiment_id", "size"), experiment_count=("experiment_id", "nunique"),
    ).reset_index()
    mapping_detail = long[["sample_id", "meta_source_sample_id", "mapping_type", "mapping_basis", "meta_match_status"]].drop_duplicates("sample_id")
    mapping = meta_counts.merge(mapping_detail, how="left", on="sample_id")

    meta_prefixed = meta.add_prefix("meta_")
    joint = long.merge(meta_prefixed, how="left", left_on="meta_source_sample_id", right_on="meta_sample_id")
    feature_columns = [
        "experiment_id", "sample_id", "exp_index", "sample_stage", "material_source", "is_milled",
        "milled_size_nm_raw", "milled_size_nm", "post_treatment", "pvp_mw_raw", "pvp_mw",
        "pvp_ratio_raw", "material_to_pvp_ratio", "repeat_batch", "concentration_ppm", "tumor_cell_line",
        "meta_source_sample_id", "mapping_type", "mapping_basis", "meta_match_status",
        "meta_sample_date", "meta_material_source", "meta_synthesis_method", "meta_is_layered",
        "meta_mg_mass_mg", "meta_si_mass_mg", "meta_molar_ratio_Mg_to_Si", "meta_vacuum_cycle",
        "meta_vacuum_time_min", "meta_initial_pressure_atm", "meta_max_temp_c", "meta_hold_time_min",
        "meta_protective_gas", "meta_milling_mode", "meta_ball_to_material_ratio", "meta_milling_cycle_time",
        "meta_pvp_mw", "meta_pvp_material_to_pvp_ratio", "meta_post_treatment", "meta_ultrasonic_time_h",
    ]
    features = joint[feature_columns].copy()
    target_columns = [
        "experiment_id", "sample_id", "exp_index", "repeat_batch", "concentration_ppm",
        "y_tumor_viability_pct", "y_normal_viability_pct", "y_selectivity_index", "y_selectivity_index_source",
        "target_complete", "target_quality_flag", "meta_source_sample_id", "mapping_type", "meta_match_status",
    ]
    targets = joint[target_columns].copy()
    targets["objective_tumor_viability"] = "minimize"
    targets["objective_normal_viability"] = "maximize"

    # Keep this generic normalized export separate from the canonical
    # conditional dataset produced by rebuild_material_cell_dataset.py.
    # Both scripts used to write bo_joint_dataset.csv, so running the full
    # pipeline silently replaced the branch-aware input expected by mobo_demo.
    meta.to_csv(ROOT / "bo_prepared_meta_sample.csv", index=False, encoding="utf-8-sig")
    long.to_csv(ROOT / "bo_prepared_experiment_long.csv", index=False, encoding="utf-8-sig")
    features.to_csv(ROOT / "bo_prepared_features.csv", index=False, encoding="utf-8-sig")
    targets.to_csv(ROOT / "bo_prepared_targets.csv", index=False, encoding="utf-8-sig")
    joint.to_csv(ROOT / "bo_prepared_joint_dataset.csv", index=False, encoding="utf-8-sig")
    mapping.to_csv(ROOT / "bo_prepared_sample_id_mapping_review.csv", index=False, encoding="utf-8-sig")
    bo_feature_columns = [
        "is_milled", "milled_size_nm", "pvp_mw", "material_to_pvp_ratio", "repeat_batch", "concentration_ppm",
        "meta_mg_mass_mg", "meta_si_mass_mg", "meta_molar_ratio_Mg_to_Si", "meta_vacuum_cycle",
        "meta_vacuum_time_min", "meta_initial_pressure_atm", "meta_max_temp_c", "meta_hold_time_min",
        "meta_ball_to_material_ratio", "meta_milling_cycle_time", "meta_pvp_mw",
        "meta_pvp_material_to_pvp_ratio", "meta_ultrasonic_time_h",
    ]
    coverage = pd.DataFrame({
        "feature": bo_feature_columns,
        "non_null_rows": [int(joint[column].notna().sum()) for column in bo_feature_columns],
        "total_rows": [int(len(joint)) for _ in bo_feature_columns],
        "coverage_rate": [float(joint[column].notna().mean()) for column in bo_feature_columns],
        "source": ["TACE" if not column.startswith("meta_") else "Mg2Si_metadata" for column in bo_feature_columns],
    })
    coverage.to_csv(ROOT / "bo_prepared_feature_coverage.csv", index=False, encoding="utf-8-sig")
    mapping_counts = mapping["mapping_type"].value_counts(dropna=False).to_dict()
    synth_columns = [
        "meta_molar_ratio_Mg_to_Si", "meta_vacuum_cycle", "meta_vacuum_time_min",
        "meta_initial_pressure_atm", "meta_max_temp_c", "meta_hold_time_min",
    ]
    synth_coverage = {column: round(float(joint[column].notna().mean()), 3) for column in synth_columns}
    print({
        "meta_rows": int(len(meta)), "experiment_rows": int(len(long) // 3), "long_rows": int(len(long)),
        "feature_rows": int(len(features)), "target_rows": int(len(targets)),
        "mapping_sample_id_counts": mapping_counts,
        "synthesis_feature_coverage": synth_coverage,
        "complete_two_target_rows": int(targets["target_complete"].sum()),
    })


if __name__ == "__main__":
    main()
