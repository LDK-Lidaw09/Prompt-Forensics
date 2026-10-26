#!/usr/bin/env python3
"""Run the LLM-as-Investigator over a replay file (thesis sections 5.4 and 6.2).

One run = one judge model on one run mode. The thesis ran five judges on the
attacks and safe replay files (serve one judge at a time with vLLM):

    python scripts/run_investigator.py \\
        --replays results/replay_outputs_attacks.jsonl \\
        --judge Qwen/Qwen3-14B \\
        --output-csv results/investigator_attacks_qwen3_14b.csv --resume
    python scripts/run_investigator.py \\
        --replays results/replay_outputs_safe.jsonl \\
        --judge Qwen/Qwen3-14B \\
        --output-csv results/investigator_safe_qwen3_14b.csv --resume

Then build Tables 6.4 to 6.6 with ``scripts/analyze.py investigator``.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prompt_forensics.experiments.investigator import InvestigatorRunConfig, run_investigator  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replays", required=True, help="replay JSONL written by run_replays.py")
    parser.add_argument("--judge", required=True, help="judge model id")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-jsonl", help="optional detailed copy of every row")
    parser.add_argument("--scenario-root", help="default: SCENARIO_ROOT")
    parser.add_argument("--max-tokens", type=int, default=640)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--repetition-penalty", type=float, default=1.3)
    parser.add_argument("--dataset-max-chars", type=int, default=1000,
                        help="context excerpt shown to the judge is cut to this length")
    parser.add_argument("--disable-guided-json", action="store_true")
    parser.add_argument("--guided-backend", default="outlines")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-pending", type=int, default=64)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--limit", type=int, help="only the first N scenarios (smoke test)")
    parser.add_argument("--resume", action="store_true",
                        help="append to --output-csv and skip scenarios already investigated successfully")
    parser.add_argument("--include-raw", action="store_true", help="keep the judge's raw text in the CSV")
    args = parser.parse_args()

    run_investigator(InvestigatorRunConfig(
        replays=args.replays,
        output_csv=args.output_csv,
        judge_model=args.judge,
        output_jsonl=args.output_jsonl,
        scenario_root=args.scenario_root,
        max_tokens=args.max_tokens,
        seed=args.seed,
        repetition_penalty=args.repetition_penalty,
        guided_json=not args.disable_guided_json,
        guided_backend=args.guided_backend,
        dataset_max_chars=args.dataset_max_chars,
        workers=args.workers,
        max_pending=args.max_pending,
        max_retries=args.max_retries,
        limit=args.limit,
        resume=args.resume,
        include_raw=args.include_raw,
    ))


if __name__ == "__main__":
    main()
