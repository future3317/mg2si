import argparse
import json
from pathlib import Path
import sqlite3

import pandas as pd

from mg2si.config import PROJECT_ROOT
from mg2si.data.build_database import build_database
from mg2si.data.quality import validate_database, validate_dataset
from mg2si.data.store import DEFAULT_DATABASE, remove_legacy_csvs
from mg2si.optimization.constraints import validate_candidates
from mg2si.optimization.recommend import recommend


def _print(payload) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def command_evaluate(path: Path, output: Path) -> dict:
    from mg2si.evaluation.grouped_splits import assert_no_group_leakage, grouped_splits
    from mg2si.evaluation.metrics import regression_metrics
    from mg2si.models.direct_baseline import DirectBaseline

    if path.suffix.lower() in {".sqlite", ".db"}:
        with sqlite3.connect(path) as connection:
            frame = pd.read_sql_query("SELECT * FROM bo_training", connection)
    else:
        frame = pd.read_csv(path, low_memory=False)
    if "model_eligible_direct" in frame:
        frame = frame[frame["model_eligible_direct"].eq(1)].copy()
    reports = []
    features = [field for field in ("material_molar_ratio_Mg_to_Si", "material_max_temp_c", "material_hold_time_min", "material_initial_pressure_atm", "material_milling_cycle_time", "material_ball_to_material_ratio", "pvp_mw", "material_to_pvp_ratio", "concentration_ppm") if field in frame]
    for scope, scoped in frame.groupby(["workflow_branch", "tumor_cell_line", "normal_cell_line"], dropna=False):
        scope_label = "|".join(str(value) for value in scope)
        scoped = scoped.dropna(subset=["y_tumor_viability_pct", "y_normal_viability_pct"])
        scope_features = [field for field in features if scoped[field].notna().any()]
        group_field = "material_parent_id" if "material_parent_id" in scoped and scoped["material_parent_id"].nunique() >= 2 else "experiment_id"
        groups = scoped[group_field].fillna(scoped["experiment_id"]).astype(str)
        if len(scoped) < 20 or groups.nunique() < 2 or not scope_features:
            reports.append({"scope": scope_label, "status": "insufficient_evidence", "rows": len(scoped), "usable_features": len(scope_features)})
            continue
        splits = grouped_splits(groups, 5)
        assert_no_group_leakage(groups, splits)
        for target in ("y_tumor_viability_pct", "y_normal_viability_pct"):
            observed, predicted = [], []
            for train, test in splits:
                fold_features = [field for field in scope_features if scoped.iloc[train][field].notna().any()]
                model = DirectBaseline().fit(scoped.iloc[train][fold_features], scoped.iloc[train][target])
                mean, _ = model.predict(scoped.iloc[test][fold_features])
                observed.extend(scoped.iloc[test][target])
                predicted.extend(mean)
            reports.append({"scope": scope_label, "target": target, "group": group_field, "rows": len(scoped), **regression_metrics(observed, predicted)})
    report_frame = pd.DataFrame(reports)
    if path.suffix.lower() in {".sqlite", ".db"}:
        with sqlite3.connect(path) as connection:
            report_frame.to_sql("model_evaluation", connection, if_exists="replace", index=False)
        destination = f"{path}#model_evaluation"
    else:
        report_frame.to_csv(output, index=False, encoding="utf-8-sig")
        destination = str(output)
    return {"status": "ok", "evaluations": len(reports), "output": destination}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mg2si")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest")
    sub.add_parser("build-dataset")
    sub.add_parser("clean-derived")
    quality = sub.add_parser("validate-data")
    quality.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    evaluate.add_argument("--output", type=Path, default=PROJECT_ROOT / "model_evaluation.csv")
    rec = sub.add_parser("recommend")
    rec.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    rec.add_argument("--output", type=Path)
    rec.add_argument("--branch", choices=["synthetic", "commercial"], required=True)
    rec.add_argument("--tumor-cell-line", required=True)
    rec.add_argument("--normal-cell-line", required=True)
    rec.add_argument("--allow-direct-baseline", action="store_true")
    validate = sub.add_parser("validate-candidates")
    validate.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    validate.add_argument("--input", type=Path)
    sub.add_parser("replay")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command in {"ingest", "build-dataset"}:
        _print(build_database(DEFAULT_DATABASE))
    elif args.command == "clean-derived":
        removed = remove_legacy_csvs(PROJECT_ROOT)
        _print({"status": "ok", "removed_count": len(removed), "removed": removed})
    elif args.command == "validate-data":
        _print(validate_database(args.database))
    elif args.command == "evaluate":
        _print(command_evaluate(args.database, args.output))
    elif args.command == "recommend":
        _print(recommend(args.database, args.output, args.branch, args.tumor_cell_line, args.normal_cell_line, args.allow_direct_baseline))
    elif args.command == "validate-candidates":
        if args.input:
            frame = pd.read_csv(args.input, low_memory=False)
        else:
            with sqlite3.connect(args.database) as connection:
                frame = pd.read_sql_query("SELECT * FROM recommendation", connection)
        errors = validate_candidates(frame) if "workflow_branch" in frame else ["recommendation file has no candidates"]
        _print({"status": "ok" if not errors else "blocked", "errors": errors})
        if errors:
            raise SystemExit(1)
    elif args.command == "replay":
        _print({"status": "insufficient_evidence", "reason": "A dated multi-round experiment ledger is required for retrospective replay."})


if __name__ == "__main__":
    main()
