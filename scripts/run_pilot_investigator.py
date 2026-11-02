#!/usr/bin/env python3
"""Run the repository Investigator on the scenarios (pilot study protocol).

The victim answers each scenario once (cached in --live-cache), then each
judge investigates the interaction. Run the judges one at a time if your GPU
can only hold one model:

    python scripts/run_pilot_investigator.py --run-mode attacks --skip-judging
    python scripts/run_pilot_investigator.py --run-mode attacks --judge Qwen/Qwen3-14B \\
        --output-csv results/investigator_attacks.csv
    python scripts/run_pilot_investigator.py --run-mode safe --judge Qwen/Qwen3-14B \\
        --output-csv results/investigator_safe.csv
    python scripts/analyze.py pilot-detection results/investigator_attacks.csv results/investigator_safe.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prompt_forensics.experiments.pilot_investigator import InvestigatorConfig, run_investigator  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-mode", choices=["attacks", "safe"], required=True)
    parser.add_argument("--judge", action="append", default=[], dest="judges",
                        help="judge model id; repeat for several judges")
    parser.add_argument("--attack-filter")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", help="victim model (default: LIVE_MODEL)")
    parser.add_argument("--output-csv", help="appended to if it exists")
    parser.add_argument("--live-cache", default="results/live_cache.jsonl")
    parser.add_argument("--skip-judging", action="store_true", help="only generate victim answers")
    parser.add_argument("--max-tokens", type=int, default=640, help="judge answer budget")
    parser.add_argument("--disable-thinking", action="store_true",
                        help="send enable_thinking=false (Qwen3 chat template)")
    args = parser.parse_args()

    path = run_investigator(InvestigatorConfig(
        run_mode=args.run_mode,
        judge_models=args.judges,
        attack_filter=args.attack_filter,
        limit=args.limit,
        target_model=args.model,
        output_csv=args.output_csv,
        live_cache=args.live_cache,
        skip_judging=args.skip_judging,
        max_tokens=args.max_tokens,
        disable_thinking=args.disable_thinking,
    ))
    print(f"results: {path}")


if __name__ == "__main__":
    main()
