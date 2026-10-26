# Changes from the original research code

The original code is preserved in the Git history:

- commit `ba8951a` — the repository before the refactoring (`app.py`,
  `streamlit_app.py`, `run_*.py`, `judge_as_evaluator.py`, ...);
- commit `f0d9ef9` — scripts recovered afterwards and committed unchanged
  under `recovered_scripts/`: the Investigator of section 5.4
  (`run_investigator_dataset.py`), the table scripts
  (`aggregate_replay_robustness.py`, `aggregate_tail_risk*.py`,
  `eval_investigator.py`, `table3_synthesis.py`) and older variants of the
  Evaluator runner and of `app.py`.

This file lists every change that can alter a number, a label or a stored
record. Pure restructuring (modules, names, comments) is not listed.

**Equivalence checks.** On the same inputs, the new code gives exactly the
same values as the original for: Consistency and Fidelity of complete
replay pools (cosine and BERTScore, checked on 50 random pools); Tables
6.1/6.2, 6.3 (overall, by family, by domain, paired test) and 6.4–6.6 (score
threshold rule), checked against the original scripts on synthetic CSVs;
the Investigator prompt and JSON schema (checked on dataset scenarios).

---

## 1. Points that affect what the thesis reports

### 1.1 How Tables 6.4–6.6 classify attacks <a id="decision-rule"></a>

The thesis (5.4.1, 6.2) says the attack/safe decision is the judge's own
`decision` field and "is not derived by thresholding the maliciousness
score". The original `table3_synthesis.py`, which writes the
`*_summary.csv` behind Tables 6.4–6.6, classifies a row as an attack when
`maliciousness_score >= 0.6` and never reads `decision`
(`eval_investigator.py` uses `decision`, but only for one judge at a time).
Table 6.6's "F1 at 0.6" column equals Table 6.4's F1 for every judge, which
is what the threshold rule produces.

`scripts/analyze.py investigator` now prints both rules side by side
(`rule = decision` and `rule = threshold`). If the two differ on the real
CSVs, either the tables or the text of 5.4.1/6.2 must be corrected. The text
also says the empirical boundary between the two decisions sits "closer to
0.4 than to 0.6", which contradicts "this decision empirically aligns with a
0.6 score cutoff" a few pages later.

### 1.2 Cosine scale <a id="cosine-scale"></a>

The original code reported embedding similarity as (cos + 1) / 2, without
saying so. Cosines between answers of the same model are almost always
positive, so this maps them to roughly [0.5, 1] and makes every score look
high: the 0.9 threshold on this scale is a raw cosine of 0.8. The default
(`cosine_mapping = "shifted"`) keeps this mapping so that the thesis numbers
are reproduced, and the raw mean cosine is now also reported
(`*_raw_cosine_mean`). `--cosine-mapping raw` reports the cosine itself.
Section 5.2.2 should state which scale is used.

### 1.3 Failed replays

Original: empty or failed replays were silently removed, then the remaining
outputs were renumbered. When replay 1 had failed, replay 2 silently became
the Fidelity reference, and a pool with fewer than two outputs got scores of
0.0 (indistinguishable from "completely different").

Now: the pool keeps its indices; failed replays are listed in
`missing_indices` and the row has `complete = False`; Fidelity is left empty
when replay 1 is missing; scores that cannot be computed are empty, never 0.0.
The original CSV writer also turned a missing BERTScore into 0.0; it is now
empty and the error is in `bert_error`.

### 1.4 Evaluator Agent sees the attack family

The original Evaluator prompt included the scenario metadata (attack type,
scenario name, run mode), with the instruction not to use it. A judge that
is told "attack_type: jailbreak" is not judging the output alone, which is
what section 5.3 claims. Judging is now **blind by default**;
`--include-metadata` reproduces the thesis prompt exactly. The CSV has a
`judge_blind` column.

### 1.5 Investigator failures counted as "safe"

In `run_investigator_dataset.py`, a judge call that failed or could not be
parsed was written as `decision = "safe"`, `maliciousness_score = 0.0`, and
any unexpected decision value was also turned into `"safe"`. The table
scripts dropped rows with `judge_ok = False`, so failures were excluded,
but their number was never reported and the "per judge: 21,600 attack and
24,000 safe prompts" in the thesis assumes none failed. Now a failure has an
empty `decision` and score, an invalid decision is a failure, and the
tables report `failed_attacks` / `failed_safe` for each judge.

### 1.6 Investigator generation settings depended on the `app.py` version

`run_investigator_dataset.py` asks for `repetition_penalty = 1.3` and a
guided-decoding backend. The repository's `app.py` forwarded the backend but
dropped the repetition penalty; the recovered `app.py` forwarded the
penalty but dropped the backend. Which one was used for the thesis runs is
not recorded. The new client forwards both.

### 1.7 Evidence store

The original store (`dna_store.sqlite`) signed records but wrote them with
`INSERT OR REPLACE` (a record could be overwritten), had no hash chain and
no protection against edits or deletions, and hashed the prompt without the
text normalisation described in 4.2.1. The new store
(`psi_evidence.sqlite`, table `interactions`) implements section 4.2 as
written: canonical payload with normalisation, Prompt ID, content hash,
hash chain, ECDSA signature of each chain link, append-only triggers, and a
`verify()` that detects edited, deleted, reordered or re-signed records.
Section 4.5 (which names `dna_store.sqlite` and `chat_logs`) needs updating.

## 2. Other behaviour changes

- **Model errors raise.** The original client caught every error and
  returned an empty text, so a timeout looked like an empty answer; and an
  unprefixed model id with `LIVE_API_BASE` unset went silently to a
  simulator that echoes the prompt. Now `LLMError` is raised, batch runners
  record the error (`error` field) and the replay counts as missing. The
  simulator only runs for `sim:` model ids.
- **Seed always sent.** The seed was only sent on the vLLM route; it is now
  sent to every OpenAI-compatible backend.
- **Deterministic scenario order.** Scenario files are read in sorted
  order, so `--limit` and resumed runs always see the same scenarios
  (`glob` order is filesystem dependent).
- **Pilot Investigator labels.** The allowed-label list contained
  `"reverse psychology"` and `"belief Elicitation"` while the prompt asked
  for `reverse_psychology` and `belief_elicitation`, so those two families
  were silently dropped from every report. Fixed. This affects the pilot
  results in `results/pilot_2025-12/investigator/`.
- **Pilot Investigator input.** The scenario's system prompt is passed with
  role `system` (it was labelled `user`, so a benign system prompt could be
  "attributed" as malicious). Verdicts are lower-cased before validation
  (`"Malicious"` used to become `unknown`).
- **Score 0.0 kept.** Several judges used `float(x or 0.5)`, which turns a
  legitimate score of 0.0 into 0.5. Scores of 0.0 are now kept; a missing
  score is empty (the pilot Investigator still imputes one from the verdict,
  and says so in `score_imputed`).
- **JSON lists.** A `null` item in a judge's list is dropped instead of
  becoming the string `"None"`.
- **Investigator representative replay.** The Investigator uses replay 0,
  as before, unless it failed; then it uses the first successful replay
  (the original sent an empty output to the judge).
- **Resuming.** Resumed Investigator runs redo scenarios whose previous
  attempt failed (they used to be skipped forever). The table functions keep
  the last row when a scenario or replay appears twice.
- **Retries.** Evaluator and Investigator retry call errors and empty
  answers. A non-empty answer that cannot be parsed is not retried: at
  temperature 0 the judge would repeat it. (One recovered variant of the
  Evaluator runner retried parse errors and repaired malformed JSON; the
  repository version, which uses guided decoding, did not. The latter is
  kept.)
- **Undefined metrics.** Precision, recall or F1 with an empty denominator
  are reported as NaN instead of 0.

## 3. Removed

- The FastAPI endpoint in `app.py` (not described in the thesis).
- The "Judge v1 enriched triptych" view of the UI, which forced the verdict
  to `risky` whatever the judge said.
- The default output-language hint of the UI (now optional, default none).
- Screenshots, Numbers files and SQLite files that were committed with the
  results; the CSV results are kept under `results/pilot_2025-12/`.

## 4. Not found in the original material

- The vLLM version, GPU model and serving command of the thesis runs.
- A script for Table 6.3bis (tail-risk as a function of pool size);
  `analysis.tail_risk_by_pool_size` implements its definition.
- The full result CSVs of chapter 6 (only the pilot study is in the
  repository). They should be archived with the thesis, together with the
  public signing key.
