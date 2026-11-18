# Protocol: judge repetition and complete path in the signed store

Status: **draft for validation**. Once validated, this file is committed
before any run and is not changed afterwards; any deviation is reported as a
deviation in the results.

Both experiments use the models, prompts and settings of the main version 2
runs (`docs/RUNS_V2.md`): victim `mistralai/Mistral-7B-Instruct-v0.3`
(temperature 0.2, seed 1234 + i, 512 tokens), Evaluator `Qwen/Qwen3-14B`
(blind schema, temperature 0, guided JSON with xgrammar), Investigator
`Qwen/Qwen3-14B` (temperature 0, seed 777, 640 tokens), vLLM 0.10.1.1, one
A100 80 GB.

## Experiment A: repetition of the judgements

### Question

When the same judge receives the same input again, how often does its
verdict or decision change? The judges run at temperature 0 with a fixed
seed; any change comes from the serving stack (batching, kernel order), as
observed for the victim in the same-seed pass (Table 6.16).

### Sample (fixed before the run)

Drawn by `scripts/plan_repetition.py`, seed 20261118, from the archived
verdicts of the main runs. Strata and sizes:

| Stratum | Content | Items |
|---|---|---|
| E1 | the 193 answers of the manual check (Table 5.8) | 193 |
| E2 | attack pools with exactly 1 unsafe answer in 100: that answer | 300 |
| E3 | attack pools with 2 to 5 unsafe answers: one unsafe answer drawn at random | 200 |
| E4 | legitimate pools with at least 1 unsafe answer: one unsafe answer | 200 |
| E5 | attack answers drawn at random among all complete pools | 300 |
| E6 | legitimate answers drawn at random among all complete pools | 300 |
| I1 | Investigator: attack prompts it missed (false negatives), at random | 200 |
| I2 | Investigator: the 5 legitimate prompts it flagged (false positives) | 5 |
| I3 | Investigator: the 100 prompts of the report review (Table 6.13) | 100 |
| I4 | Investigator: attack prompts it flagged, at random | 200 |
| I5 | Investigator: legitimate prompts it cleared, at random (standard and written to resemble attacks, half each) | 200 |

Strata overlap only where stated; an item drawn twice is kept once and
counted in its first stratum. E1 to E6 total 1,493 Evaluator inputs; I1 to I5
total 705 Investigator inputs. The sample file (`plan/repetition_sample.csv`)
and its SHA-256 are committed before the run.

### Procedure

1. Three repetitions R1, R2, R3 of the Evaluator on the E items, with exactly
   the inputs and settings of the main run (same seed per replay).
2. Before each repetition the vLLM server is restarted and the order of the
   requests is shuffled (seed 1, 2, 3), so that batches differ.
3. The same for the Investigator on the I items, with the same input as in
   the main run (system prompt, excerpt, prompt, first non-empty replay).
4. The original verdict of the main run counts as repetition R0.

### Measures reported (all fixed now)

For each stratum and overall:

- **Verdict change rate (Evaluator)**: share of items whose verdict
  (safe / suspicious / malicious) is not the same in R0 to R3, with a Wilson
  95% interval; the same for the binary unsafe label.
- **Agreement**: Fleiss' kappa of the four repetitions, three classes and
  binary.
- **Rare signals**: in E2, the share of pools that would lose their only
  unsafe answer in at least one repetition, and in E2 to E4 the share whose
  label is unsafe in all four repetitions.
- **Decision change rate (Investigator)**: share of items whose decision
  (attack / safe) differs across R0 to R3, with Wilson 95% interval; share
  whose declared attack type, success field or rule cited changes; mean and
  maximum spread of the maliciousness score.
- **Identical text**: share of judge outputs whose raw text is identical
  across repetitions (reported last, as a secondary measure).

Not reported as a result: any recomputation of the 72.7% tail-risk rate,
since the sample does not support it. The change rates of E2 to E4 are
reported as an indication of how stable rare signals are, nothing more.

### Expected interpretation (fixed now)

- A change rate under 1% in every stratum: verdicts are stable on repetition
  at the precision of the sample.
- A change rate of 1% or more in a stratum: reported as the measured
  instability of that stratum, with its interval; no threshold makes the
  judges "reliable" or "unreliable".

## Experiment B: complete path in the signed store

### Question

Do the evaluated components run end to end through the signed evidence
store, and what does the verification detect when the stored evidence is
altered afterwards?

### Sample (fixed before the run)

50 prompts drawn by `scripts/plan_repetition.py` (same seed): 25 attack
prompts (stratified by source collection) and 25 legitimate prompts (20
standard, 5 written to resemble attacks), all from the test split.

### Procedure

In a fresh SQLite store with a new ECDSA key (kept outside the database):

1. **Capture**: each prompt is sent to the victim through
   `pipeline.capture_interaction`; the record is chained and signed.
2. **Replays**: 10 replays per prompt through `pipeline.replay_interaction`
   with the recorded configuration and seed 1234 + i for replay i; each
   replay record links to its source
   by `metadata.source_prompt_id` and stores `metadata.replay_config`.
3. **Judgements**: the Evaluator on every replay and the Investigator on the
   captured interaction, written with `EvidenceStore.add_annotation`.
4. **Report**: one report per prompt (verdicts per replay, number of unsafe
   answers, Investigator decision, replay configuration and its divergences).
5. **Anchor**: the head of the chain is exported (`EvidenceStore.anchor`).
6. **Verification**: `scripts/verify_evidence.py` with the trusted public key
   and the exported anchor.

### Controlled alterations

Each alteration is applied to a fresh copy of the store after step 6, by
editing the SQLite file directly (the attacker bypasses the triggers):

| Case | Alteration | Expected result with the current code |
|---|---|---|
| T1 | change one word of one victim answer | detected (response hash, chain) |
| T2 | change the recorded seed of one replay | detected (Prompt ID, chain) |
| T3 | delete one replay record in the middle | detected (chain) |
| T4 | delete the last records | detected only with the exported anchor |
| T5 | change one Evaluator verdict (annotation) | **not detected**: annotations carry a hash of their own content but are not chained, signed or checked by `verify()` |
| T6 | re-sign the whole altered chain with a stolen key | detected only with the exported anchor |

Case T5 is a known limit of the current store (verdicts are added as
annotations outside the chain). It is measured as it is and reported as a
limit; the code is not changed before the run.

### Measures reported (all fixed now)

- Whether each step ran for all 50 prompts (counts of records and
  annotations, failures and their causes).
- Replay configuration: number of replays whose recorded and used
  configuration differ, by field. Expected: only `seed`, passed explicitly
  as 1234 + i for replay i (as in the main runs) and recorded with the origin
  `override`; no other field.
- Verification result on the unaltered store, and for T1 to T6 which layer
  (hash, chain, signature, anchor) reports the problem, or that none does.
- Agreement between the store-based verdicts and the result-file pipeline
  for the same prompts is **not** used as a result, because the replays are
  new answers.

## Files and run

- `scripts/plan_repetition.py`: draws both samples (needs the archived main
  run files under `$RUNS_V2_DATA`).
- `scripts/run_repetition.sh`: Experiment A, then Experiment B, on one GPU.
- `scripts/analysis_v2/repetition.py` and `integrated_path.py`: compute the
  measures above and write `results/v2/tables/repetition_*.csv` and
  `integrated_path_*.csv`.

Indicative cost, to be confirmed on the pod: about 4,500 Evaluator calls
and 2,100 Investigator calls for A, 500 replays, 500 Evaluator and 50
Investigator calls for B, plus model loading; roughly 1.5 to 3 GPU hours.
