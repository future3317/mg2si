import argparse
import json
from pathlib import Path
import sqlite3

import pandas as pd

from mg2si.config import PROJECT_ROOT
from mg2si.data.build_database import build_database
from mg2si.data.quality import validate_database
from mg2si.data.store import DEFAULT_DATABASE, remove_legacy_csvs
from mg2si.evaluation.service import evaluate_database
from mg2si.optimization.constraints import validate_candidates
from mg2si.optimization.design_space import build_prospective_design_space
from mg2si.optimization.real_space import explore_real_space
from mg2si.optimization.recommend import recommend


def _print(payload) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mg2si")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest")
    sub.add_parser("ingest-all-sources")
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
    real_space = sub.add_parser("explore-real-space")
    real_space.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    real_space.add_argument("--branch", choices=["synthetic", "commercial"], required=True)
    real_space.add_argument("--tumor-cell-line", required=True)
    real_space.add_argument("--normal-cell-line", required=True)
    prospective = sub.add_parser("enumerate-design-space")
    prospective.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    prospective.add_argument("--branch", choices=["synthetic", "commercial"])
    prospective.add_argument("--max-points", type=int, default=10000)
    prospective.add_argument("--seed", type=int, default=20260726)
    validate = sub.add_parser("validate-candidates")
    validate.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    validate.add_argument("--input", type=Path)
    sub.add_parser("replay")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command in {"ingest", "ingest-all-sources", "build-dataset"}:
        _print(build_database(DEFAULT_DATABASE))
    elif args.command == "clean-derived":
        removed = remove_legacy_csvs(PROJECT_ROOT)
        _print({"status": "ok", "removed_count": len(removed), "removed": removed})
    elif args.command == "validate-data":
        _print(validate_database(args.database))
    elif args.command == "evaluate":
        _print(evaluate_database(args.database, args.output))
    elif args.command == "recommend":
        _print(recommend(args.database, args.output, args.branch, args.tumor_cell_line, args.normal_cell_line, args.allow_direct_baseline))
    elif args.command == "explore-real-space":
        _print(explore_real_space(args.database, args.branch, args.tumor_cell_line, args.normal_cell_line))
    elif args.command == "enumerate-design-space":
        _print(build_prospective_design_space(args.database, args.branch, args.max_points, args.seed))
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
