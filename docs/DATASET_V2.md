# Scenario dataset, version 2: safe prompts

## Why a version 2

An audit of version 1 (`python scripts/generate_safe_prompts.py audit`)
gives:

| Safe prompts | Count |
|---|---|
| Rows | 24,000 |
| Distinct texts (as reported in Table 5.2 of the thesis) | 6,174 |
| Distinct texts once normalised and with the document name masked | **186** |
| Contexts (domain x family x document) | 1,188 |
| Documents | 198 |
| System prompts | 23 |

The 6,174 distinct texts are about 186 templates in which only the document
name changes ("You will be shown `<doc>`. If pressed to ..., decline."). They
are also guard-rail instructions, not requests a legitimate user would send,
and they name the attack they guard against. A judge can separate them from
attacks on surface features alone, so they measure false positives poorly.

Version 2 keeps every context and every attack prompt of version 1 and
replaces the safe prompts by distinct, realistic user requests.

## Method

Code: `src/prompt_forensics/datagen/`, command line:
`scripts/generate_safe_prompts.py`.

1. **Contexts.** The 1,188 contexts of version 1 (system prompt + document),
   unchanged. Each gets a quota of `target / 1,188` prompts, so the
   domain x family design stays balanced.
2. **Slots.** Each generation request fixes one task (14), one user
   profile (8 per domain) and one answer format (7), drawn at random:
   784 combinations per context. About 20 % of the requests ask for
   **hard negatives**: legitimate requests that look, on the surface, like
   the attack family of the context (a translation of the user's own text in
   a data-leakage context, a training role-play in a jailbreak context, ...).
3. **Generation.** The model `claude-haiku-5-5` by default (about 1/40 of the
   Opus price; `--model claude-opus-5-5` for more varied prompts) through the
   Messages API or the Message Batches API, 10 prompts per request, answer
   constrained to a JSON schema. The request shows the system prompt, the
   document, the slot and up to 15 prompts already kept for the same document,
   which the model is asked not to repeat.
4. **Filters.** Length (6 to 200 words), English, no blatant attack phrase
   ("ignore previous instructions", "developer mode", ...); standard prompts
   must also avoid attack vocabulary ("bypass", "system prompt",
   "role-play", ...).
5. **Deduplication.** Exact (normalised text, document name masked) and
   near-duplicate (raw cosine of `gte-large-en-v1.5` embeddings of the full
   text >= 0.88, calibrated on the 300-prompt pilot) against every prompt
   kept so far. The first occurrence is kept. The embeddings are not taken on
   the masked text: masking also removes the topic words, which hides
   rewordings (0.97 on the full texts, 0.87 masked). Resuming a run checks
   the kept prompts again with the current rule; the ones it rejects go to
   `pruned.jsonl` and the next rounds top up.
6. **Stopping rule.** Rounds continue until the target is reached or a round
   keeps less than 10 % of what it generated (saturation). `rounds.csv`
   gives the saturation curve (distinct prompts kept per round).
7. **Assembly.** Each version 1 scenario receives one new safe prompt of its
   context (`safe_prompt_v1` keeps the old one); extra prompts become
   safe-only scenarios. `split` (dev 20 % / test 80 %) is drawn per document,
   so threshold tuning and evaluation never share a document.
8. **Human validation.** `sample` draws 500 prompts stratified by domain and
   kind. The sheet shows the system prompt, the document and the prompt only;
   the key (kind, task) is kept apart, so annotators do not know which items
   are hard negatives. Two annotators label each item `legitimate`,
   `not_legitimate` or `unclear` (`oui`/`non`/`uncertain` are read as the
   same labels). `agreement` reports the observed agreement, Cohen's kappa
   and Gwet's AC1: when nearly every item is legitimate, kappa collapses
   towards 0 even with high agreement, and AC1 does not.
9. **Adjudication.** A prompt stays in the safe set only if both annotators
   labelled it `legitimate`. `exclude` moves the others from
   `safe_prompts.jsonl` to `excluded.jsonl` with the item id and the reason;
   run `assemble` again afterwards. `agreement` still counts excluded items,
   since they were annotated.

   Result of the 2026 annotation (475 items analysed, 25 pruned before):
   observed agreement 98.1 %, Cohen's kappa 0 (the second annotator labelled
   every item legitimate), Gwet's AC1 0.981. The 9 prompts labelled
   `not_legitimate` by one annotator (items A0083, A0143, A0226, A0241,
   A0249, A0355, A0372, A0406, A0447) were excluded: 19,981 safe prompts and
   23,376 scenarios remain (18,205 with both prompts, 3,395 attack only,
   1,776 safe only; safe prompts dev 2,997 / test 16,984).

## Traceability

Every file of a run is kept in its output directory: the full generation
requests, the raw answers with the serving model and token usage, every
candidate with its fate (filtered, exact or near duplicate of which prompt,
kept), the kept prompts with their SHA-256, and a manifest with the settings,
the code commit and the token totals. Current Anthropic models take no
temperature or seed, so a run cannot be repeated token for token; it is made
auditable instead.

## Commands

```bash
python scripts/generate_safe_prompts.py audit --near           # Table 5.2, v1
python scripts/generate_safe_prompts.py preview --hard-negative
python scripts/generate_safe_prompts.py generate --out data/safe_prompts_v2 --target 300 --max-rounds 1   # pilot
python scripts/generate_safe_prompts.py generate --out data/safe_prompts_v2 --target 20000 --backend batch
python scripts/generate_safe_prompts.py assemble --out data/safe_prompts_v2 --scenario-out data/scenarios_v2
python scripts/generate_safe_prompts.py sample --out data/safe_prompts_v2 --size 500
```

Then run the experiments on the new files with `--scenario-root data/scenarios_v2`,
and `--split dev` (choosing thresholds) or `--split test` (reported results)
for the replay runs, whose outputs feed the judges.

Requirements: `pip install -e ".[ml,datagen]"` (embeddings and the Anthropic
SDK), an API key in `PF_ANTHROPIC_API_KEY` (or `ANTHROPIC_API_KEY`), and access to the Hugging Face Hub to download the
embedding model. A pilot (`--target 300`) sends about 30 requests: a round
never asks for much more than the remaining target. If every request of a
round fails (no key, no network, refusals), the run stops and says why.
`assemble` writes its summary to `<out>/assembly.json`, not into the
scenario root, and refuses to replace an existing scenario root without
`--overwrite`.
