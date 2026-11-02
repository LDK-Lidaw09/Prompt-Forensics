"""LLM agents of the PSI framework.

- :mod:`.evaluator`     Evaluator Agent: safety verdict on one replay output (thesis 4.3.4, 5.3)
- :mod:`.investigator`  LLM-as-Investigator on a prompt-response pair (thesis 4.4, 5.4)
- :mod:`.repository_investigator`  multi-prompt Investigator of the pilot study and demo UI
- :mod:`.comparators`   baseline-vs-replay judges used in the demo UI
- :mod:`.replay_group`  LLM review of a whole replay pool (demo UI)
- :mod:`.self_audit`    self-audit of one output (demo UI)
- :mod:`.adversary`     toy red-team prompt generator (demo UI)
"""
