#!/usr/bin/env python3
"""Plan the sensitivity runs: temperature test and Investigator without the response.

Writes ``<out>/plan.csv`` (one row per prompt of the ablation sample, with
``temperature_sample`` marking the nested temperature sample) and the
scenario roots ``ablation_attacks``, ``ablation_safe``, ``temperature_attacks``
and ``temperature_safe`` replayed by ``scripts/run_sensitivity.sh``. The
samples are stratified like the deep pools (run mode x family x domain x
split x safe-prompt kind) and deterministic for a given seed.

    python scripts/plan_sensitivity.py --out results/v2/sensitivity/plan
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prompt_forensics.experiments.plan import sensitivity_plan, write_sensitivity_plan  # noqa: E402
from prompt_forensics.scenarios import load_scenarios  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario-root", default="data/scenarios_v2")
    parser.add_argument("--out", required=True)
    parser.add_argument("--ablation", type=int, default=2_000, help="prompts of the Investigator ablation")
    parser.add_argument("--temperature", type=int, default=1_000,
                        help="prompts of the temperature test (drawn from the ablation sample)")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    scenarios = load_scenarios(args.scenario_root)
    rows = sensitivity_plan(scenarios, args.ablation, args.temperature, args.seed)
    roots = write_sensitivity_plan(scenarios, rows, args.out, overwrite=args.overwrite)
    summary = {
        "ablation": dict(Counter(r["run_mode"] for r in rows)),
        "temperature": dict(Counter(r["run_mode"] for r in rows if r["temperature_sample"])),
        "split": dict(Counter(r["split"] for r in rows)),
        "scenario_roots": [str(r) for r in roots],
    }
    (Path(args.out) / "plan.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
