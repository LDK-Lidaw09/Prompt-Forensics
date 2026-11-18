#!/usr/bin/env python3
"""Measures of Experiment B (docs/PROTOCOL_judge_repetition_integrated_path.md).

Reads the outputs of ``scripts/run_integrated_path.py`` and writes in ``--out``:

- ``integrated_path_steps.csv``: whether each step ran for all prompts
  (records, annotations, failures and their causes, verification of the
  unaltered store);
- ``integrated_path_replay_config.csv``: replays whose recorded and used
  configuration differ, by field and origin;
- ``integrated_path_alterations.csv``: T1 to T6, the result expected in the
  protocol and which layer (hash, chain, signature, anchor) reports the problem;
- ``integrated_path_reports.csv``: one row per prompt (unsafe answers among
  the replays, Investigator decision). Descriptive only: the replays are new
  answers, so they are not compared with the result files of the main runs.

    python scripts/analysis_v2/integrated_path.py --run results/v2/repetition/B \\
        --out results/v2/repetition/tables
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import pandas as pd

# First run (docs/PROTOCOL_judge_repetition_integrated_path.md): annotations outside the chain.
EXPECTED = {
    "T1": "detected (response hash, chain)",
    "T2": "detected (Prompt ID, chain)",
    "T3": "detected (chain)",
    "T4": "detected only with the exported anchor",
    "T5": "not detected",
    "T6": "detected only with the exported anchor",
}
# Second run (docs/PROTOCOL_policies_verdict_chain_fidelity.md): chained and signed annotations.
EXPECTED_CHAINED = {
    "T1": "detected (response hash)",
    "T2": "detected (Prompt ID, content hash)",
    "T3": "detected (chain, binding, anchor)",
    "T4": "detected (binding, anchor)",
    "T5": "detected (annotation chain)",
    "T6": "detected (binding, anchor)",
    "T7": "detected (annotation chain)",
    "T8": "detected (binding, anchor)",
    "T9": "detected only with the exported anchor",
}
LAYERS = ("hash", "chain", "signature", "binding", "anchor")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="results/v2/repetition/B")
    parser.add_argument("--out", default="results/v2/repetition/tables")
    args = parser.parse_args()
    run, out = Path(args.run), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    steps = json.loads((run / "steps.json").read_text())
    failures = []
    if (run / "failures.csv").exists():
        with open(run / "failures.csv", newline="", encoding="utf-8") as handle:
            failures = list(csv.DictReader(handle))
    causes = Counter((f["step"], f["error"].split(":")[0]) for f in failures)
    prompts = steps["prompts"]
    rows = [
        {"step": "1 capture", "expected": prompts, "done": steps["captured"]},
        {"step": "2 replays", "expected": prompts * 10, "done": steps["replays"]},
        {"step": "2 replays linked to their source", "expected": steps["replays"],
         "done": steps["replays"] - steps["replays_unlinked"]},
        {"step": "3 evaluator annotations", "expected": steps["replays"],
         "done": steps["annotations"].get("evaluator", 0)},
        {"step": "3 investigator annotations", "expected": steps["captured"],
         "done": steps["annotations"].get("investigator", 0)},
        {"step": "4 reports", "expected": steps["captured"], "done": steps["reports"]},
        {"step": "5 anchor records", "expected": steps["captured"] + steps["replays"],
         "done": steps["anchor"]["records"]},
        {"step": "6 verification problems (unaltered store)", "expected": 0,
         "done": len(steps["verification"]["problems"])},
    ]
    for (step, cause), k in sorted(causes.items()):
        rows.append({"step": f"failure {step}: {cause}", "expected": 0, "done": k})
    pd.DataFrame(rows).to_csv(out / "integrated_path_steps.csv", index=False)

    config = pd.read_csv(run / "replay_config.csv")
    config.to_csv(out / "integrated_path_replay_config.csv", index=False)

    alt = pd.read_csv(run / "alterations.csv")
    expected = EXPECTED_CHAINED if "T7" in set(alt.case) else EXPECTED
    alt.insert(1, "expected", alt.case.map(expected))
    alt["layers_reporting"] = [", ".join(layer for layer in LAYERS if r.get(f"problems_{layer}", 0) > 0) or "none"
                               for _, r in alt.iterrows()]
    alt.to_csv(out / "integrated_path_alterations.csv", index=False)

    reports = [json.loads(line) for line in (run / "reports.jsonl").read_text().splitlines() if line.strip()]
    pd.DataFrame([{k: r[k] for k in ("item", "run_mode", "scenario_id", "replays_judged", "unsafe_answers",
                                     "investigator_decision", "investigator_attack_succeeded")}
                  for r in reports]).to_csv(out / "integrated_path_reports.csv", index=False)
    for name in ("steps", "replay_config", "alterations"):
        print(pd.read_csv(out / f"integrated_path_{name}.csv").to_string())


if __name__ == "__main__":
    main()
