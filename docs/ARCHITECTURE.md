# Architecture

## Principles

- **One implementation of every computation.** Hashing, metrics, judge
  prompts and tables live in the `prompt_forensics` package. The scripts and
  the Streamlit app only parse arguments, call the package and display the
  result, so the demo and the experiments cannot drift apart.
- **Failures are data, not zeros.** A failed model call, an unparseable
  judge answer or a missing replay is recorded as such (`None`, `ok = False`,
  `missing_indices`) and excluded explicitly, with its count reported.
- **Settings in one place.** Every default comes from
  `config.get_settings()` (environment variables, see `.env.example`), and
  the defaults are the thesis settings.
- **Testable without a GPU.** Model calls go through `llm.call_llm`, and the
  similarity backends are injectable, so the tests replace both.

## Package map

```
config ──► llm ──────────────► agents.* ──► experiments.* ──► scripts/, app/
   │        │                     ▲              │
   │        └─► (vLLM, LM Studio, │              ├─► experiments.analysis (tables)
   │             OpenAI, Anthropic)              │
   ├─► canonical ─► signing ─► storage ─► pipeline (capture, replay)
   └─► scenarios ─────────────────────────────────┘
       metrics.similarity ─► metrics.robustness ─┘
```

| Module | Role |
|---|---|
| `config` | Frozen `Settings` built from the environment. |
| `llm` | One `call_llm(prompt, model, params)` for all backends; model id prefixes select the backend; errors raise `LLMError`. |
| `jsonparse` | Extracts the first JSON object from a judge answer (fenced, raw, or embedded in prose; `<think>` blocks removed). |
| `canonical` | Text normalisation, canonical JSON, Prompt ID and content hash (4.2.1). |
| `signing` | P-256 key management and signatures (4.2.2). |
| `storage` | Append-only SQLite evidence store with hash chain and verification (4.2.3). |
| `scenarios` | Dataset loading, selection and the exact victim prompt (5.1). |
| `metrics.similarity` | Cached embedding model and BERTScore scorer. |
| `metrics.robustness` | Consistency and Fidelity of one replay pool (4.3.2–4.3.3). |
| `agents.evaluator` | Evaluator Agent: safe/suspicious/malicious verdict on one output (4.3.4). |
| `agents.investigator` | LLM-as-Investigator on a prompt–response pair (4.4, 5.4). |
| `agents.repository_investigator` | Multi-prompt Investigator of the pilot study and the UI. |
| `agents.comparators`, `replay_group`, `self_audit`, `adversary` | Demo-only judges and the toy red-team generator. |
| `pipeline` | Capture of a live interaction and replay of a stored one. |
| `experiments.replays` / `evaluate` / `investigator` | Batch runners, resumable, writing CSV and JSONL. |
| `experiments.analysis` | Tables 6.1–6.6 from the CSVs. |

## Evidence model (thesis 4.2)

Each captured interaction is a payload

```json
{"prompt": "...", "system": null, "model": "...", "params": {"temperature": 0.2, "max_tokens": 512, "seed": 1234},
 "timestamp": "...", "user_id": "...", "session_id": null, "context_hash": null}
```

- **Canonical form**: strings NFC-normalised, line endings to `\n`,
  horizontal whitespace collapsed outside double-quoted spans, lines
  trimmed; `null` fields dropped; keys sorted; compact separators.
- **Prompt ID** = SHA-256 of the canonical payload. It identifies the
  capture *event* (it includes the timestamp and user).
- **Content hash** = SHA-256 of the payload without `timestamp`, `user_id`
  and `session_id`. Two replays of the same input with the same parameters
  share it; this is how replays are linked to their source.
- **Response hash** = SHA-256 of the raw answer (not normalised: the
  answer is evidence and is hashed byte for byte).
- **Chain**: each row stores `prev_chain_hash` and
  `chain_hash = SHA-256(prev, prompt_id, response_hash, record_type, SHA-256(metadata))`;
  the first row chains to 64 zeros.
- **Signature**: `chain_hash` is signed with ECDSA P-256. Because each
  link covers the previous one, a signature vouches for the whole history up
  to that row.
- **Append-only**: SQL triggers refuse `UPDATE` and `DELETE`. A party with
  file access can drop the triggers, so the real guarantee is `verify()`
  against an archived public key, which detects an edited, deleted,
  reordered or re-signed row.

## Replay and stability (thesis 4.3, 5.2)

A replay pool is the same victim input sent *n* = 10 times with seeds
`1234 + i`. For outputs r₁…rₙ (r₁ is the reference):

- Consistency = mean similarity over all pairs (i < j);
- Fidelity = mean similarity of r₂…rₙ to r₁;

each with the embedding cosine (mapped to (c + 1) / 2 by default) and with
BERTScore F1, plus the population standard deviation and the number of
pairs/replays at or above the 0.9 threshold. Missing replays keep their
index and are reported; Fidelity is undefined without r₁.

## Judges

| Agent | Input | Output | Used for |
|---|---|---|---|
| Evaluator | one replay output (blind by default) | verdict, `unsafe`, score, risk types | tail-risk (5.3) |
| Investigator | system prompt + context + attack/safe prompt + one output | `decision`, `maliciousness_score`, families, markers | Tables 6.4–6.6 (5.4) |
| Repository Investigator | several stored prompts + one output | verdict, score, malicious prompt ids | pilot study, UI |

All judges run at temperature 0 with a fixed seed, and the batch judges
use vLLM guided decoding so that their output is valid JSON by construction.
Their prompts are kept verbatim from the thesis runs, because changing a
word would change the results.

## Demo application (thesis 4.5)

`app/streamlit_app.py` is a single file with one function per tab:
Capture, Red team, Replay (single replay with judges, replay pool, across
models), Investigator, Experiments (small runs and the chapter 6 tables) and
Evidence (browse, export, verify). It holds no forensic logic.
