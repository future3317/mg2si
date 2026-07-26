from pathlib import Path
import re

import numpy as np
import pandas as pd
from mg2si.io.excel_reader import resolve_sources
from mg2si.io.parsers import parse_pvp_mw as parse_pvp_mw_strict


ROOT = Path(__file__).resolve().parent
_, TACE_PATH = resolve_sources(ROOT)


def clean(value):
    if pd.isna(value):
        return np.nan
    text = re.sub(r"\s+", " ", str(value)).strip()
    return np.nan if text in {"", "-", "—", "–", "/", "\\", "nan", "None"} else text


def first_number(value):
    value = clean(value)
    if pd.isna(value):
        return np.nan
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(match.group(0)) if match else np.nan


def midpoint(value):
    value = clean(value)
    if pd.isna(value):
        return np.nan
    nums = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", str(value))]
    if len(nums) >= 2 and any(x in str(value) for x in ["-", "~", "至", "到"]):
        return float(np.mean(nums[:2]))
    return nums[0] if nums else np.nan


def parse_yes_no(value):
    value = clean(value)
    if pd.isna(value):
        return np.nan
    if str(value) in {"是", "有", "Y", "yes", "Yes", "1"}:
        return 1
    if str(value) in {"否", "无", "N", "no", "No", "0"}:
        return 0
    return np.nan


def parse_date(value):
    value = clean(value)
    if pd.isna(value):
        return np.nan
    parsed = pd.to_datetime(value, errors="coerce")
    return parsed.strftime("%Y-%m-%d") if not pd.isna(parsed) else np.nan


def parse_pvp_mw(value):
    return parse_pvp_mw_strict(value)


def parse_ratio(value):
    value = clean(value)
    if pd.isna(value):
        return np.nan
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", str(value))]
    return nums[0] / nums[1] if len(nums) >= 2 and nums[1] else first_number(value)


def normalize_id(value):
    value = clean(value)
    if pd.isna(value):
        return np.nan
    text = re.sub(r"\([^)]*\)|（[^）]*）", "", str(value))
    text = re.sub(r"\s+S\d+.*$", "", text)
    text = re.sub(r"\s+(top|down|mid|上层|下层|中层)$", "", text, flags=re.IGNORECASE)
    return text.strip()


def read_sheet(path, sheet_name, is_summary):
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    if is_summary:
        names = [
            "tace_sample_id_raw", "exp_index", "sample_stage", "material_source", "is_milled_raw",
            "milled_size_nm_raw", "post_treatment", "pvp_mw_raw", "pvp_ratio_raw", "repeat_batch",
            "kill_assay_date", "tumor_cell_line", "y_tumor_500ppm", "y_tumor_250ppm", "y_tumor_125ppm",
            "selection_cell", "safety_assay_date", "y_normal_500ppm", "y_normal_250ppm", "y_normal_125ppm",
            "y_selectivity_500ppm_source", "y_selectivity_250ppm_source", "y_selectivity_125ppm_source", "remark",
        ]
        data = raw.iloc[1:, :24].copy()
    else:
        names = [
            "tace_sample_id_raw", "exp_index", "sample_stage", "material_source", "is_milled_raw",
            "milled_size_nm_raw", "post_treatment", "pvp_mw_raw", "pvp_ratio_raw", "repeat_batch",
            "kill_assay_date", "tumor_cell_line", "y_tumor_500ppm", "y_tumor_250ppm", "y_tumor_125ppm",
            "safety_assay_date", "y_normal_500ppm", "y_normal_250ppm", "y_normal_125ppm",
            "y_selectivity_500ppm_source", "y_selectivity_250ppm_source", "y_selectivity_125ppm_source", "remark",
        ]
        data = raw.iloc[1:, :23].copy()
    data.columns = names
    for column in data.columns:
        data[column] = data[column].map(clean)
    data = data[data["tace_sample_id_raw"].notna()].copy()
    data["tace_sample_id"] = data["tace_sample_id_raw"].map(normalize_id)
    data["is_milled"] = data["is_milled_raw"].map(parse_yes_no)
    data["milled_size_nm"] = data["milled_size_nm_raw"].map(midpoint)
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
    data["source_sheet"] = sheet_name
    data["is_summary_source"] = int(is_summary)
    return data


def melt(data):
    rows = []
    for row_number, row in data.iterrows():
        for concentration in [500, 250, 125]:
            suffix = f"{concentration}ppm"
            tumor = row[f"y_tumor_{suffix}"]
            normal = row[f"y_normal_{suffix}"]
            selectivity = row[f"y_selectivity_{suffix}_source"]
            flags = []
            if pd.isna(tumor):
                flags.append("missing_tumor_viability")
            if pd.isna(normal):
                flags.append("missing_normal_viability")
            rows.append({
                "source_record_id": f"{row['source_sheet']}_row_{row_number + 2}",
                "source_sheet": row["source_sheet"],
                "is_summary_source": row["is_summary_source"],
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
                "normal_cell_line": row.get("selection_cell", np.nan),
                "remark": row["remark"],
                "concentration_ppm": concentration,
                "y_tumor_viability_pct": tumor,
                "y_normal_viability_pct": normal,
                "y_selectivity_index_source": selectivity,
                "target_complete": int(pd.notna(tumor) and pd.notna(normal)),
                "target_quality_flag": ";".join(flags) if flags else "ok",
            })
    return pd.DataFrame(rows)


def main():
    xl = pd.ExcelFile(TACE_PATH)
    tables = []
    for index, sheet in enumerate(xl.sheet_names):
        if sheet == "备注":
            continue
        tables.append(read_sheet(TACE_PATH, sheet, index == 0))
    row_tables = pd.concat(tables, ignore_index=True)
    long = melt(row_tables)
    long["duplicate_group_key"] = (
        long["tace_sample_id"].fillna("").astype(str) + "|" + long["exp_index"].fillna("").astype(str)
        + "|" + long["repeat_batch"].astype(str) + "|" + long["concentration_ppm"].astype(str)
    )
    long["duplicate_group_count"] = long["duplicate_group_key"].map(long["duplicate_group_key"].value_counts())
    summary = long[long["is_summary_source"] == 1].copy()
    subtables = long[long["is_summary_source"] == 0].copy()
    registry = row_tables.groupby(["source_sheet", "is_summary_source"], dropna=False).size().reset_index(name="row_count")
    registry["long_cell_point_count"] = registry["row_count"] * 3
    registry["duplicate_group_count"] = registry["source_sheet"].map(long.groupby("source_sheet")["duplicate_group_count"].apply(lambda x: int((x > 1).sum())).to_dict())
    long.to_csv(ROOT / "bo_cell_all_sources_long.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(ROOT / "bo_cell_summary_long.csv", index=False, encoding="utf-8-sig")
    subtables.to_csv(ROOT / "bo_cell_subtables_long.csv", index=False, encoding="utf-8-sig")
    registry.to_csv(ROOT / "bo_cell_source_registry.csv", index=False, encoding="utf-8-sig")
    print({
        "source_sheets": registry[["source_sheet", "row_count"]].to_dict("records"),
        "all_source_long_rows": int(len(long)),
        "summary_long_rows": int(len(summary)),
        "subtable_long_rows": int(len(subtables)),
        "duplicate_long_rows_marked": int((long["duplicate_group_count"] > 1).sum()),
    })


if __name__ == "__main__":
    main()
