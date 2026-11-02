# Pilot study results (December 2025)

Results of the first experiments, kept unchanged for traceability. They
were produced by the original code (commit `ba8951a`) and are **not** the
results of chapter 6, which used the full dataset, five judges and the
protocol of `docs/REPRODUCIBILITY.md`.

| Folder | Content | Produced by (today's equivalent) |
|---|---|---|
| `replay_robustness/<family>/{attacks,safe}/` | Consistency/Fidelity per scenario | `scripts/run_replays.py` |
| `replay_robustness/all_families/` | the same for all families together | `scripts/run_replays.py` |
| `investigator/by_family/<family>/` | repository Investigator, before and after prompt refinement | `scripts/run_pilot_investigator.py` |
| `investigator/by_judge/<judge>/` | the same, one folder per judge model | `scripts/run_pilot_investigator.py` |
| `single_replay_judging/{attacks,safe}/` | one victim answer per scenario, judged once | `scripts/run_pilot_investigator.py` |
| `attacks_sample_20.csv` | 20 hand-written prompts used to test the demo | — |

Known issues in these files (see `docs/CHANGES.md`): cosine scores are on
the (cos + 1) / 2 scale; failed replays were dropped silently; the
Investigator label filter dropped `reverse_psychology` and
`belief_elicitation`; some `investigator/by_family/*` files were produced
before the prompt was refined on the same scenarios.

Analyse them with, for example:

    python scripts/analyze.py pilot-detection \
        results/pilot_2025-12/single_replay_judging/attacks/jailbreak.csv \
        results/pilot_2025-12/single_replay_judging/safe/jailbreak.csv
