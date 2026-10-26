#!/usr/bin/env python3
"""Judge every replay output with the Evaluator Agent (thesis 5.3, 6.4).

Example (Qwen3-14B served by vLLM, blind judging):

    python scripts/run_evaluator.py --input results/replay_outputs_attacks.jsonl \\
        --output-csv results/evaluator_attacks.csv --judge-model Qwen/Qwen3-14B \\
        --workers 16 --resume

Add --include-metadata to reproduce the prompt used for the thesis runs, which
showed the judge the run mode and attack type.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prompt_forensics.experiments.evaluate import EvaluatorConfig, run_evaluator  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", nargs="+", required=True, help="replay JSONL file(s)")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-jsonl", help="optional detailed output")
    parser.add_argument("--judge-model", help="default: JUDGE_MODEL (Qwen/Qwen3-14B)")
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-max-tokens", type=int, default=512)
    parser.add_argument("--judge-seed", type=int, default=777, help="replay index is added")
    parser.add_argument("--include-metadata", action="store_true",
                        help="show run_mode / attack_type to the judge (original, non-blind prompt)")
    parser.add_argument("--disable-guided-json", action="store_true")
    parser.add_argument("--guided-backend", default="outlines", help="outlines or xgrammar")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-pending", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--run-mode", choices=["attacks", "safe"])
    parser.add_argument("--resume", action="store_true", help="skip outputs already in --output-csv")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--preview-chars", type=int, default=500)
    parser.add_argument("--include-output-text", action="store_true")
    parser.add_argument("--include-raw-judge-text", action="store_true")
    args = parser.parse_args()

    run_evaluator(EvaluatorConfig(
        inputs=args.input,
        output_csv=args.output_csv,
        output_jsonl=args.output_jsonl,
        judge_model=args.judge_model,
        temperature=args.judge_temperature,
        max_tokens=args.judge_max_tokens,
        seed=args.judge_seed,
        include_metadata=args.include_metadata,
        guided_json=not args.disable_guided_json,
        guided_backend=args.guided_backend,
        workers=args.workers,
        max_pending=args.max_pending,
        limit=args.limit,
        run_mode=args.run_mode,
        resume=args.resume,
        max_retries=args.max_retries,
        retry_sleep_s=args.retry_sleep,
        preview_chars=args.preview_chars,
        include_output_text=args.include_output_text,
        include_raw_judge_text=args.include_raw_judge_text,
    ))


if __name__ == "__main__":
    main()
