"""Scenario dataset: loading, filtering and victim prompt construction (thesis 5.1).

The dataset lives in ``data/scenarios/<family>/<domain>_scenarios.json``: 6
attack families x 4 domains, 1,000 scenarios per file. Each scenario has:

- ``id``, ``domain``, ``attack_type``, ``scenario_name``, ``description``
- ``system_prompt``: role and guardrail of the assistant
- ``dataset``: a short fictional document excerpt given as context
- ``safe_prompt``: a benign request (present in all 24,000 scenarios)
- ``attack_prompt``: an adversarial request (present in 21,600 scenarios;
  the last 100 scenarios of each file are safe-only)
- ``benchmark_source``, ``benchmark_id``: provenance of the attack seed

``run_mode`` selects which prompt is used: ``"attacks"`` (attack_prompt) or
``"safe"`` (safe_prompt).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

RUN_MODES = ("attacks", "safe")
# Context fields accepted, in priority order. The current dataset only uses
# "dataset"; the others are kept for older scenario files.
CONTEXT_KEYS = ("dataset", "case_info", "data_snippet", "context")


def normalize_attack_type(value: str | None) -> str:
    """``"Reverse Psychology"`` -> ``"reverse_psychology"``."""
    return (value or "").strip().lower().replace(" ", "_").replace("-", "_")


def scenario_files(root: str | Path) -> list[Path]:
    """All scenario JSON files under ``root``, in a stable (sorted) order."""
    return sorted(Path(root).rglob("*.json"))


def load_scenarios(root: str | Path) -> list[dict[str, Any]]:
    """Load every scenario under ``root``.

    The order is deterministic (files sorted by path, scenarios in file
    order), so ``--limit N`` always selects the same N scenarios.
    Each scenario gets a ``source_file`` field with its file name.
    """
    scenarios: list[dict[str, Any]] = []
    for path in scenario_files(root):
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        items = data["scenarios"] if isinstance(data, dict) and "scenarios" in data else data
        if not isinstance(items, list):
            raise ValueError(f"{path}: expected a list of scenarios")
        for item in items:
            scenario = dict(item)
            scenario.setdefault("source_file", path.name)
            scenarios.append(scenario)
    return scenarios


def scenario_id(scenario: dict[str, Any]) -> str:
    return str(scenario.get("id") or scenario.get("scenario_id") or "unknown_id")


def main_prompt(scenario: dict[str, Any], run_mode: str) -> str:
    """The prompt under test: ``attack_prompt`` or ``safe_prompt``."""
    if run_mode not in RUN_MODES:
        raise ValueError(f"run_mode must be one of {RUN_MODES}, got {run_mode!r}")
    key = "attack_prompt" if run_mode == "attacks" else "safe_prompt"
    return str(scenario.get(key) or "")


def select_scenarios(
    scenarios: Iterable[dict[str, Any]],
    run_mode: str,
    attack_filter: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Keep scenarios that have a prompt for ``run_mode``, then filter and cut.

    ``attack_filter`` matches ``attack_type`` after normalisation, either
    exactly or as a substring (``"injection"`` matches ``"prompt_injection"``).
    """
    selected = [s for s in scenarios if main_prompt(s, run_mode)]
    if attack_filter:
        wanted = normalize_attack_type(attack_filter)
        selected = [s for s in selected if wanted in normalize_attack_type(s.get("attack_type"))]
    if limit is not None and limit > 0:
        selected = selected[:limit]
    return selected


def build_victim_prompt(scenario: dict[str, Any], run_mode: str) -> str:
    """Text sent to the victim model, as one user message.

    Concatenates, separated by blank lines:

    1. the scenario's ``system_prompt``;
    2. its context excerpt (first non-empty field of :data:`CONTEXT_KEYS`);
    3. the prompt under test (attack or safe).

    The system prompt is placed in the user message rather than in a system
    role. This is how the thesis experiments were run, and it must be stated
    as such in the methodology (section 5.2.1).
    """
    parts = []
    if scenario.get("system_prompt"):
        parts.append(str(scenario["system_prompt"]))
    for key in CONTEXT_KEYS:
        if scenario.get(key):
            parts.append(str(scenario[key]))
            break
    prompt = main_prompt(scenario, run_mode)
    if prompt:
        parts.append(prompt)
    return "\n\n".join(parts).strip()
