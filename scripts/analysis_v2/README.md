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
| 9 | `truncation_t02.py <sensitivity folder> results/v2/tables` | answers cut at 512 tokens and their verdicts, on the replays at temperature 0.2 of the sensitivity run: `truncation_t02_by_cut.csv`, `truncation_t02_pools.csv`, `truncation_t02_within_pools.csv` (bootstrap over pools, 10,000 resamples, seed 20261117), and `t02_replays.pkl` for step 10 |
| 10 | `similarity_targeted_test.py t02_replays.pkl results/v2/tables/similarity_targeted_test_pairs.csv` | targeted test of the similarity scores: 30 answers changed by rule (negation, identifier, added record), a fixed refusal, and real pairs of safe and malicious answers (selection seed 7) |
| 11 | `evaluator_check_invented_rule.py <human labels csv> results/v2/tables/evaluator_check_invented_rule.csv` | manual check of the Evaluator under two rules for invented data (stratified bootstrap, 4,000 resamples, seed 20261117) |

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

## Reference distribution (Table 6.3)

`reference_distribution_pairs.csv` holds 5,000 pairs of first answers to
different prompts per run mode and kind (same domain, or drawn at random),
with their cosine and BERTScore; `reference_distribution_summary.csv` gives
the quantiles reported in Table 6.3. The pairs were drawn from 4,000 first
answers (500 per domain and run mode) during the chapter 6 analysis; the
drawing script itself was not archived, so the pairs file is the reference.

## Figures quoted in the thesis

`scripts/check_thesis_numbers.py` compares the figures quoted in chapter 6
with the tables of `results/v2/tables/`. Table 6.5 (answer lengths) and the
failure causes of Table 6.1 come from the raw replay and Investigator files on
the Hugging Face dataset and have no table here.
