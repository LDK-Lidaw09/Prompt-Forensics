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
git checkout claude/safe-prompts-v2
export HF_TOKEN=hf_...
nohup bash scripts/run_gpu_shard.sh 0 > shard0.log 2>&1 &   # 1 on the second pod
tail -f results/v2/shard0/steps.log
```

If the pod stops, start the same command again: finished steps are skipped
and the runs resume from their output files. At the end the results are in
`results/v2/shard<k>.tar.gz`; download both archives before the pods are
deleted.

## Outputs (per shard)

| File | Content |
|---|---|
| `replays_<mode>_r<n>.jsonl` | every replay (text, seed, model, timing) |
| `replays_<mode>_r<n>.csv` | Consistency and Fidelity per prompt |
| `replays_<mode>_r10_llama-3_1-8b-instruct.*` | the same for the second victim |
| `evaluator.csv` | safe/unsafe verdict of every replay |
| `investigator_<mode>_r<n>_<judge>.csv` | Investigator report per prompt and judge |
| `steps.log`, `environment.txt`, `vllm_*.log` | timings, GPU, vLLM version, commit |
