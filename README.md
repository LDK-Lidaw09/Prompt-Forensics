# Prompt Forensics (PSI)

Reference implementation of **PSI (Prompt Scene Investigation)**, the framework
of the M.Sc. thesis *Prompt Forensics: Toward a New Subdomain for Investigating
LLM Interactions* (Lidaw Fabrice Ledjaki, Université de Montréal, 2026).

PSI treats every prompt–response interaction with a large language model as a
forensic artefact:

1. **Capture** — the interaction is serialised canonically, identified by a
   SHA-256 Prompt ID, signed (ECDSA P-256) and appended to a hash-chained,
   append-only evidence store.
2. **Replay** — the prompt is replayed under controlled seeds to measure how
   stable the model's behaviour is (Consistency and Fidelity, with embedding
   cosine and BERTScore) and how often a pool of replays hides an unsafe
   answer (tail-risk).
3. **Investigate** — LLM agents judge each replay (Evaluator Agent) and
   analyse the prompt–response pair for adversarial intent
   (LLM-as-Investigator).

## Where each part of the thesis lives

| Thesis | Code |
|---|---|
| 4.2.1–4.2.2 Canonical serialisation, Prompt ID, signatures | `src/prompt_forensics/canonical.py`, `signing.py` |
| 4.2.3 Append-only evidence store | `src/prompt_forensics/storage.py`, `scripts/verify_evidence.py` |
| 4.3.1 Replay protocol | `src/prompt_forensics/experiments/replays.py`, `scripts/run_replays.py` |
| 4.3.2–4.3.3, 5.2 Consistency and Fidelity | `src/prompt_forensics/metrics/` |
| 4.3.4, 5.3 Evaluator Agent and tail-risk | `agents/evaluator.py`, `experiments/evaluate.py`, `scripts/run_evaluator.py` |
| 4.4, 5.4 LLM-as-Investigator | `agents/investigator.py`, `experiments/investigator.py`, `scripts/run_investigator.py` |
| 4.5 Demo application | `app/streamlit_app.py` |
| 5.1 Dataset (24,000 scenarios) | `data/scenarios/`, `src/prompt_forensics/scenarios.py` |
| Tables 6.1–6.6 | `src/prompt_forensics/experiments/analysis.py`, `scripts/analyze.py` |
| Pilot study (Dec. 2025) | `agents/repository_investigator.py`, `experiments/pilot_investigator.py`, `results/pilot_2025-12/` |

## Repository layout

```
src/prompt_forensics/   the library (every computation lives here)
  agents/               LLM agents: Evaluator, Investigator, demo judges
  metrics/              similarity backends and replay robustness
  experiments/          batch runners and result tables
scripts/                command-line entry points for the experiments
app/streamlit_app.py    demo interface (single file, calls the library only)
data/scenarios/         the dataset: 6 attack families x 4 domains
results/pilot_2025-12/  results of the pilot study
tests/                  unit and end-to-end tests (no GPU, no model server)
docs/                   architecture, reproduction guide, change log
```

## Installation

Python 3.10 or newer.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[ml,ui,dev]"      # or: pip install -r requirements.txt
cp .env.example .env               # then edit the model endpoints
```

The `ml` extra (PyTorch, sentence-transformers, bert-score) is only needed
for the Consistency and Fidelity metrics. Everything else, including the
tests, runs without it.

## Quick start

**Without any model server**, using the built-in simulator (`sim:` model ids
echo the prompt):

```bash
LIVE_MODEL=sim:victim JUDGE_MODEL=sim:judge streamlit run app/streamlit_app.py
pytest
```

**With models served by vLLM** (OpenAI-compatible API on port 8000):

```bash
# 1. Replay pools: 10 replays per scenario, seeds 1234..1243
python scripts/run_replays.py --run-mode attacks --resume \
    --output-csv results/robustness_attacks.csv --output-jsonl results/replays_attacks.jsonl
# 2. Evaluator Agent on every replay
python scripts/run_evaluator.py --input results/replays_attacks.jsonl \
    --output-csv results/evaluator_attacks.csv --resume
# 3. Investigator, one judge at a time
python scripts/run_investigator.py --replays results/replays_attacks.jsonl \
    --judge Qwen/Qwen3-14B --output-csv results/investigator_attacks_qwen3_14b.csv --resume
# (repeat 1-3 with --run-mode safe / the safe files)
# 4. Tables of chapter 6
python scripts/analyze.py tail-risk --attacks results/evaluator_attacks.csv --safe results/evaluator_safe.csv
```

The full protocol, with every parameter, is in
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Model identifiers

A model id may carry a prefix that selects the backend:

| Id | Backend |
|---|---|
| `Qwen/Qwen3-14B` (no prefix) | OpenAI-compatible server at `LIVE_API_BASE` (vLLM) |
| `lmstudio:<name>` | LM Studio at `LMSTUDIO_API_BASE` |
| `openai:<name>` | OpenAI API (`OPENAI_API_KEY`) |
| `anthropic:<name>` | Anthropic API (`ANTHROPIC_API_KEY`) |
| `sim:<anything>` | offline simulator, for tests and demos |

A failed model call raises an error; nothing silently falls back to the
simulator.

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — how the pieces fit together and why.
- [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) — exact commands and settings of every experiment.
- [docs/CHANGES.md](docs/CHANGES.md) — every behaviour change from the original research code, and what it means for the reported numbers.

## Testing

```bash
pytest
```

The suite checks canonicalisation, signatures and tamper detection, the
robustness formulas, judge-output parsing, the result tables (against
hand-computed values) and runs the experiments and the interface end to end
with the simulator.
