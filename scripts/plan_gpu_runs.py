#!/usr/bin/env python3
"""Plan the version 2 GPU runs: deep (100 replays) and shallow (10) pools, cut into shards.

Writes ``<out>/plan.csv`` (one row per prompt) and, for every shard, one
scenario root per run mode and pool size (``shard0/attacks_r100``, ...),
which ``scripts/run_gpu_shard.sh`` replays. The plan is deterministic for a
given dataset and seed, so both GPUs can build it independently.

    python scripts/plan_gpu_runs.py --out results/v2/plan
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Allow running from a fresh clone without "pip install -e .".
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prompt_forensics.experiments.plan import plan_runs, write_plan  # noqa: E402
from prompt_forensics.scenarios import load_scenarios  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario-root", default="data/scenarios_v2")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--deep", type=int, default=5_000, help="prompts replayed --deep-replays times")
    parser.add_argument("--deep-replays", type=int, default=100)
    parser.add_argument("--replays", type=int, default=10, help="replays of every other prompt")
    parser.add_argument("--shards", type=int, default=2, help="number of GPUs")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true", help="replace existing scenario roots")
    args = parser.parse_args()

    scenarios = load_scenarios(args.scenario_root)
    rows = plan_runs(scenarios, args.deep, args.deep_replays, args.replays, args.shards, args.seed)
    roots = write_plan(scenarios, rows, args.out, overwrite=args.overwrite)
    summary = {
        "prompts": len(rows),
        "by_pool_and_mode": {f"{p}/{m}": n for (p, m), n in sorted(Counter((r["pool"], r["run_mode"]) for r in rows).items())},
        "by_shard_and_pool": {f"shard{s}/{p}": n for (s, p), n in sorted(Counter((r["shard"], r["pool"]) for r in rows).items())},
        "replays_per_shard": {f"shard{s}": n for s, n in sorted(Counter(
            r["shard"] for r in rows for _ in range(r["replays"])).items())},
        "scenario_roots": [str(r) for r in roots],
    }
    (Path(args.out) / "plan.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
