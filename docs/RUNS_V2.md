# Version 2 runs on two GPUs

The runs of the revised thesis replay the 41,581 prompts of dataset v2
(`data/scenarios_v2`: 21,600 attack and 19,981 safe prompts).

## Design

- **Deep pools.** 5,000 prompts, drawn at random and stratified by
  run mode × attack family × domain × split × safe-prompt kind, are replayed
  100 times. They support the tail-risk curve up to 100 replays and every
  statistic that needs many replays of one prompt.
- **Shallow pools.** Every other prompt is replayed 10 times, as in the
  first version.
- **Shards.** `scripts/plan_gpu_runs.py` deals the prompts of every stratum
  in turn to two shards, so each GPU gets the same work: about 2,500 deep
  and 18,300 shallow prompts, or 433,000 replays. The plan is deterministic
  (seed 2026), and `results/v2/plan/plan.csv` lists the pool, the number of
  replays and the shard of every prompt.

Each shard runs, on one GPU:

1. the replays with `mistralai/Mistral-7B-Instruct-v0.3` (temperature 0.2,
   seed 1234 + i, 512 tokens), with Consistency and Fidelity from the raw
   cosine of `gte-large-en-v1.5` embeddings and from BERTScore;
   With the victim still loaded, 500 deep-pool prompts (both shards
   together) are replayed 10 more times as is and 10 times with extra
   whitespace that the canonical form removes (double spaces, tabs, trailing
   spaces, CRLF), same seeds: does dropping whitespace before hashing lose
   anything that changes the answer? (`whitespace_*` files)
2. the same deep-pool prompts replayed 10 times with a second victim,
   `meta-llama/Llama-3.1-8B-Instruct`, better aligned than Mistral-7B and the
   usual victim of jailbreak benchmarks, to check that the results do not
   hold for one model only (`VICTIM2=` skips it);
3. the Evaluator (`Qwen/Qwen3-14B`, blind to the attack family) on every
   replay of both victims;
4. the LLM-as-Investigator on the first victim, with the five judges of
   the thesis, served one after the other. `microsoft/Phi-4-reasoning` gets 4,096 output tokens
   instead of 640, which cut its reasoning before the verdict.

Estimated cost from the A100 benchmark of 2026-10-27: about 42 GPU hours of
replays and Evaluator per shard, about 2 hours for the second victim and
about 7 hours of Investigator, so two days per GPU.

## Running

On huggingface.co, accept the licence of the gated models with the account of
the token: `mistralai/Mistral-7B-Instruct-v0.3`,
`mistralai/Ministral-8B-Instruct-2410`, `google/gemma-3-12b-it` and
`meta-llama/Llama-3.1-8B-Instruct`. The script checks access to every model
before it starts.

Two pods, each with one A100 80 GB and 150 GB of disk:

```bash
git clone https://github.com/LDK-Lidaw09/Prompt-Forensics.git && cd Prompt-Forensics
export HF_TOKEN=hf_...
nohup bash scripts/run_gpu_shard.sh 0 > shard0.log 2>&1 &   # 1 on the second pod
tail -f results/v2/shard0/steps.log
```

If the pod stops, start the same command again: finished steps are skipped
and the runs resume from their output files. At the end the results are in
`results/v2/shard<k>.tar.gz`; download both archives before the pods are
deleted.

## Backup

Every 30 minutes, and once more at the end, each shard copies its results
folder to a private dataset repository on the Hugging Face Hub,
`<token owner>/prompt-forensics-runs-v2` (`shard0/`, `shard1/`, `plan/`), with
`scripts/backup_to_hf.py`. The token must have the write permission; the
script checks it before the first replay. `HF_BACKUP_REPO` chooses another
repository, `BACKUP=0` turns the backup off. The raw replays (several GB) stay
on the Hub; only the aggregated tables go to this repository.

## Outputs (per shard)

| File | Content |
|---|---|
| `replays_<mode>_r<n>.jsonl` | every replay (text, seed, model, timing) |
| `replays_<mode>_r<n>.csv` | Consistency and Fidelity per prompt |
| `replays_<mode>_r<n>_pairs.jsonl` | raw cosine and BERTScore F1 of every pair of replays (threshold sensitivity, pairs with the same or different Evaluator verdicts) |
| `replays_<mode>_r10_llama-3_1-8b-instruct.*` | the same for the second victim |
| `evaluator.csv` | safe/unsafe verdict of every replay |
| `investigator_<mode>_r<n>_<judge>.csv` | Investigator report per prompt and judge |
| `steps.log`, `environment.txt`, `vllm_*.log` | timings, GPU, vLLM version, commit |

## Sensitivity runs (one GPU, about 6 to 7 hours)

Three checks that each change one factor and keep every other setting of the
main runs:

1. **Temperature.** 1,000 prompts (stratified, nested in the ablation
   sample) replayed 20 times by Mistral-7B at temperatures 0, 0.2, 0.7 and
   1.0, seeds 1234 + i, each replay judged by the Evaluator.
2. **Same seed, same text?** A second pass at 0.2 with the same seeds
   (10 replays per prompt), compared text for text with the first pass.
   It measures whether a recorded seed reproduces a replay exactly in this
   environment, or only controls the randomness.
3. **Investigator without the answer.** The Qwen3-14B Investigator, same
   prompt and settings, on 2,000 prompts with the victim's answer replaced
   by a fixed placeholder (`run_investigator.py --no-output`). The baseline
   with the answer is the main run on the same prompts (paired by
   `scenario_id`), and, on the 1,000 temperature prompts, the same run.

```bash
export HF_TOKEN=hf_...   # write access
nohup bash scripts/run_sensitivity.sh > sensitivity.log 2>&1 &
tail -f results/v2/sensitivity/steps.log
```

Results go to `results/v2/sensitivity/` (copied to `sensitivity/` on the
Hub every 30 minutes) and `results/v2/sensitivity.tar.gz`; the tables come
from `scripts/analysis_v2/sensitivity.py`.

## Archiving the results

The Hub repository is an archive of the result files, not an evidence store:
its files are not chained, signed or anchored. Before the archived version of
the thesis is deposited (the repository must be public by then), write a
SHA-256 manifest pinned to one commit of the repository and publish it with
the archive:

```bash
export HF_TOKEN=hf_...   # read access while the repository is private
python scripts/archive_manifest.py --repo Latavious/prompt-forensics-runs-v2 \
    --revision main --out archive/MANIFEST
```

This writes `archive/MANIFEST.sha256` (the `sha256sum` format) and
`archive/MANIFEST.json` (repository, resolved commit, date, size and hash of
each file). Files stored with LFS or Xet are not downloaded, since the Hub
gives their SHA-256; the small git files are downloaded one at a time (about
3 minutes and 0.5 GB for the 885 files of the version 2 runs). Cite the
commit recorded in `MANIFEST.json` as the revision of the dataset.

A reader checks a download of the same revision with
`sha256sum -c MANIFEST.sha256`, or with
`python scripts/archive_manifest.py --out archive/MANIFEST --check <folder>`.
The check shows that the files have not changed since the manifest was
written, not since they were generated.
