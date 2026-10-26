#!/usr/bin/env python3
"""Replay every scenario N times and compute Consistency / Fidelity (thesis 5.2, 6.1).

Example (thesis setting, attack prompts, vLLM serving Mistral-7B on port 8000):

    export LIVE_API_BASE=http://127.0.0.1:8000/v1
    python scripts/run_replays.py --run-mode attacks \\
        --output-csv results/replay_robustness_attacks.csv \\
        --output-jsonl results/replay_outputs_attacks.jsonl --resume
"""

import argparse
import sys
from pathlib import Path

# Allow running from a fresh clone without "pip install -e .".
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prompt_forensics.experiments.replays import ReplayConfig, run_replays  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-mode", choices=["attacks", "safe"], required=True)
    parser.add_argument("--attack-filter", help="keep one family, e.g. jailbreak")
    parser.add_argument("--limit", type=int, help="first N scenarios only (for quick tests)")
    parser.add_argument("--num-replays", type=int, default=10)
    parser.add_argument("--model", help="victim model id (default: LIVE_MODEL)")
    parser.add_argument("--temperature", type=float, help="default: LIVE_TEMPERATURE (0.2)")
    parser.add_argument("--max-tokens", type=int, help="default: LIVE_MAX_TOKENS (512)")
    parser.add_argument("--seed-base", type=int, help="replay i uses seed_base + i (default 1234)")
    parser.add_argument("--threshold", type=float, default=0.9, help="similarity threshold")
    parser.add_argument("--cosine-mapping", choices=["shifted", "raw"], default="shifted",
                        help="shifted = (cos+1)/2, as used for the thesis tables; raw = cosine")
    parser.add_argument("--no-bertscore", action="store_true", help="skip BERTScore")
    parser.add_argument("--scenario-root", help="default: SCENARIO_ROOT (data/scenarios)")
    parser.add_argument("--scenario-workers", type=int, default=1)
    parser.add_argument("--replay-workers", type=int, default=10)
    parser.add_argument("--max-inflight-calls", type=int, default=8)
    parser.add_argument("--output-csv")
    parser.add_argument("--output-jsonl")
    parser.add_argument("--resume", action="store_true",
                        help="reuse replays in --output-jsonl and skip complete rows in --output-csv")
    args = parser.parse_args()

    csv_path, jsonl_path = run_replays(ReplayConfig(
        run_mode=args.run_mode,
        attack_filter=args.attack_filter,
        limit=args.limit,
        num_replays=args.num_replays,
        target_model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed_base=args.seed_base,
        threshold=args.threshold,
        cosine_mapping=args.cosine_mapping,
        use_bertscore=not args.no_bertscore,
        scenario_root=args.scenario_root,
        scenario_workers=args.scenario_workers,
        replay_workers=args.replay_workers,
        max_inflight_calls=args.max_inflight_calls,
        resume=args.resume,
        output_csv=args.output_csv,
        output_jsonl=args.output_jsonl,
    ))
    print(f"metrics: {csv_path}\nreplays: {jsonl_path}")


if __name__ == "__main__":
    main()
