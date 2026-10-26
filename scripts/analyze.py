#!/usr/bin/env python3
"""Build the result tables of chapter 6 from the experiment CSVs.

    # Tables 6.1 and 6.2 (replay stability), optionally as LaTeX
    python scripts/analyze.py replays --attacks robustness_attacks.csv --safe robustness_safe.csv --latex

    # Tables 6.3 and 6.3bis (tail-risk), with the paired safe-vs-attacks test
    python scripts/analyze.py tail-risk --attacks evaluator_attacks.csv --safe evaluator_safe.csv

    # Tables 6.4 to 6.6 (Investigator), one --judge per model
    python scripts/analyze.py investigator \\
        --judge Qwen3-14B investigator_attacks_qwen3_14b.csv investigator_safe_qwen3_14b.csv \\
        --judge Gemma-3-12B investigator_attacks_gemma3_12b.csv investigator_safe_gemma3_12b.csv

    # Pilot study (repository Investigator)
    python scripts/analyze.py pilot-detection pilot_attacks.csv pilot_safe.csv

Every command prints its tables; --save PREFIX also writes PREFIX_<table>.csv.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prompt_forensics.experiments import analysis  # noqa: E402


def replays(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    tables = {}
    for mode, path, number in (("attacks", args.attacks, "6_1"), ("safe", args.safe, "6_2")):
        if path:
            tables[f"table_{number}_{mode}"] = analysis.replay_tables(
                pd.read_csv(path), args.threshold, complete_only=args.complete_only)
    if args.latex:
        for name, table in tables.items():
            print(analysis.replay_tables_latex(table, f"Replay stability ({name}).", f"tab:{name}"))
    return tables


def tail_risk(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    attacks = pd.read_csv(args.attacks)
    tables = {f"attacks_{k}": v for k, v in analysis.tail_risk_tables(attacks, args.pool_size).items()}
    tables["attacks_by_pool_size"] = analysis.tail_risk_by_pool_size(attacks, args.pool_size)
    if args.safe:
        safe = pd.read_csv(args.safe)
        tables.update({f"safe_{k}": v for k, v in analysis.tail_risk_tables(safe, args.pool_size).items()})
        tables["safe_by_pool_size"] = analysis.tail_risk_by_pool_size(safe, args.pool_size)
        paired = analysis.paired_tail_risk(attacks, safe, args.pool_size)
        print("\n=== paired safe vs attacks (McNemar) ===\n" + json.dumps(paired, indent=2, default=float))
        tables["paired"] = pd.DataFrame([paired])
    return tables


def investigator(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    judges = {name: (pd.read_csv(attacks), pd.read_csv(safe)) for name, attacks, safe in args.judge}
    tables = analysis.investigator_tables(judges, args.threshold)
    for name, (attacks, safe) in judges.items():
        tables[f"score_bands_{name}"] = analysis.score_distribution(pd.concat([attacks, safe]))
    return tables


def pilot_detection(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    df = pd.concat([pd.read_csv(path) for path in args.csv], ignore_index=True)
    per_judge, per_family = analysis.pilot_detection_metrics(df)
    return {"per_judge": per_judge, "per_family": per_family}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--save", metavar="PREFIX", help="write every table to PREFIX_<table>.csv")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("replays", help="Tables 6.1 and 6.2")
    p.add_argument("--attacks")
    p.add_argument("--safe")
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--complete-only", action="store_true", help="drop pools with failed replays")
    p.add_argument("--latex", action="store_true", help="also print LaTeX tables")
    p.set_defaults(func=replays)

    p = sub.add_parser("tail-risk", help="Tables 6.3 and 6.3bis")
    p.add_argument("--attacks", required=True)
    p.add_argument("--safe")
    p.add_argument("--pool-size", type=int, default=10)
    p.set_defaults(func=tail_risk)

    p = sub.add_parser("investigator", help="Tables 6.4, 6.5 and 6.6")
    p.add_argument("--judge", nargs=3, action="append", required=True,
                   metavar=("NAME", "ATTACKS_CSV", "SAFE_CSV"))
    p.add_argument("--threshold", type=float, default=0.6)
    p.set_defaults(func=investigator)

    p = sub.add_parser("pilot-detection", help="pilot study precision/recall")
    p.add_argument("csv", nargs="+")
    p.set_defaults(func=pilot_detection)

    args = parser.parse_args()
    tables = args.func(args)
    with pd.option_context("display.width", 220, "display.max_columns", None):
        for name, table in tables.items():
            print(f"\n=== {name} ===\n{table.to_string(index=False)}")
            if args.save:
                table.to_csv(f"{args.save}_{name}.csv", index=False)


if __name__ == "__main__":
    main()
