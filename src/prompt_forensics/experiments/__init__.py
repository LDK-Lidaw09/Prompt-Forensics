"""Batch experiment runners. Each one has a command-line wrapper in ``scripts/``.

- :mod:`.replays`       replay pools and their Consistency / Fidelity (thesis 5.2, 6.1)
- :mod:`.evaluate`      Evaluator Agent over replay outputs (thesis 5.3, 6.4)
- :mod:`.investigator`  LLM-as-Investigator over the replay files (thesis 5.4, 6.2)
- :mod:`.pilot_investigator`  repository Investigator over the scenarios (pilot study)
- :mod:`.analysis`      the tables of chapter 6 (stability, tail-risk, Investigator)
"""
