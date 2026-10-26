"""Prompt Forensics: reference implementation of the PSI framework.

PSI (Prompt Scene Investigation) is the framework described in the M.Sc. thesis
"Prompt Forensics: Toward a New Subdomain for Investigating LLM Interactions"
(Université de Montréal, 2026). It turns each prompt-response interaction into a
traceable forensic artefact, replays it to measure behavioural stability, and
uses LLM agents to judge the replays and investigate adversarial intent.

Package layout (thesis section in brackets):

- ``config``                 runtime settings read from the environment
- ``canonical``              canonical serialization and Prompt ID [4.2.1, 4.2.2]
- ``signing``                ECDSA P-256 signatures [4.2.2]
- ``storage``                append-only, hash-chained evidence store [4.2.3]
- ``llm``                    one client for vLLM, LM Studio, OpenAI and Anthropic
- ``jsonparse``              robust extraction of JSON from judge outputs
- ``scenarios``              dataset loading and victim prompt construction [5.1]
- ``metrics.similarity``     embedding cosine and BERTScore backends [5.2.2, 5.2.3]
- ``metrics.robustness``     Consistency and Fidelity of a replay pool [5.2]
- ``agents.evaluator``       Evaluator Agent: per-replay safety verdict [4.3.4, 5.3]
- ``agents.investigator``    LLM-as-Investigator, five-judge protocol [4.4, 5.4]
- ``agents.repository_investigator``  pilot-study Investigator (demo UI)
- ``agents.comparators``     baseline-vs-replay judges used by the demo UI
- ``agents.replay_group``    LLM review of a whole replay pool (demo UI)
- ``agents.self_audit``      self-audit of a single output (demo UI)
- ``agents.adversary``       toy red-team generator (demo UI)
- ``pipeline``               live capture of one interaction (demo UI and API)
- ``experiments``            batch runners behind the scripts in ``scripts/``
"""

__version__ = "1.0.0"
