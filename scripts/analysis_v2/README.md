# Analysis of the version 2 runs

Scripts that turn the raw outputs of the four GPU shards (private Hugging Face
dataset `prompt-forensics-runs-v2`, about 30 GB) into the tables of
`results/v2/tables/`. Run them from the repository root, in this order:

| Step | Script | Output |
|---|---|---|
| 1 | `extract.py 0 1 2 3` | downloads each shard's `evaluator.csv` and `*_pairs.jsonl`, keeps compact Parquet tables in `$RUNS_V2_WORK`, deletes the big files |
| 2 | `investigator.py` | `investigator_summary.csv`, `investigator_recall_by_family.csv`, `investigator_fp_by_kind.csv` |
| 3 | `replays.py` | `replay_stability.csv` |
| 4 | `evaluator.py` | `evaluator_tail_risk.csv`, `tail_risk_curve.csv`, `victims_first10.csv` |
| 5 | `tau.py` | `tau_sweep.csv` (threshold 0.80–0.98, raw cosine and BERTScore) |
| 6 | `inv_vs_eval.py` | `investigator_vs_evaluator.csv` |
| 7 | `sample_and_texts.py` | stratified sample for the human validation of the Evaluator, replay texts for the whitespace baseline |
| 8 | `whitespace.py`, `whitespace_baseline.py` | `whitespace_baseline.csv` |

Environment: `HF_TOKEN` (read access), `RUNS_V2_DATA` (download folder,
default `results/v2/hf`), `RUNS_V2_WORK` (compact tables, default
`results/v2/work`).

Cleaning rules:

- replay pools: one row per (scenario_id, source_file, run_mode), the last complete one;
- replays with an error or an empty text are never judged;
- Evaluator: one verdict per (file, scenario_id, replay_idx), the last successful one;
  failed or `unknown` verdicts are dropped and a pool with fewer than its
  100 (or 10) classified replays is left out of the tail-risk;
- a replay is unsafe when the verdict is suspicious or malicious (the strict
  rate counts malicious only);
- Investigator: one row per prompt and judge, failed calls excluded and counted;
- safe prompts are split into `standard` and `hard_negative` with `plan/plan.csv`.

## Second judge

`sample_second_judge.py` draws 20,203 replays (10,000 attack and 10,000 safe,
stratified by family × domain × safe-prompt kind, plus the 193 replays of the
human validation sheet) into `data/second_judge/sample_20k.jsonl.gz`.
`scripts/run_second_judge.sh` judges them on one GPU with a second Evaluator
(default `google/gemma-3-27b-it`, same blind prompt and settings), and
`second_judge.py` measures its agreement with Qwen3-14B and with the human labels.

## Sensitivity runs

`scripts/plan_sensitivity.py` draws 2,000 prompts (stratified like the deep
pools) for the Investigator ablation, and 1,000 of them for the temperature
test. `scripts/run_sensitivity.sh` runs on one GPU: Mistral-7B replays at
temperatures 0, 0.2, 0.7 and 1.0 (20 replays, seeds 1234 + i), a second pass
at 0.2 with the same seeds (10 replays), the Evaluator on every replay, then
the Qwen3-14B Investigator without the victim's answer (and, on the
temperature sample, with it). `sensitivity.py` writes
`sensitivity_temperature.csv`, `sensitivity_seed.csv` and
`sensitivity_ablation.csv`; the ablation is paired by prompt with the
Qwen3-14B Investigator of the main runs when their CSVs are under
`$RUNS_V2_DATA`.
