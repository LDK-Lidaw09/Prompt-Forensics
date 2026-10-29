"""Scenario contexts that the new safe prompts are written for.

A *context* is what the victim model sees around the prompt under test: the
domain, the attack family whose guard-rail is in the system prompt, the
system prompt itself and the document excerpt. Version 1 has 1,188 distinct
contexts (198 documents x the families they appear with). Version 2 keeps
them unchanged, so that safe and attack prompts are still compared in the
same contexts and the attack prompts do not need to be regenerated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from ..canonical import sha256_hex


@dataclass(frozen=True)
class Context:
    context_id: str
    domain: str
    family: str
    system_prompt: str
    document: str

    @property
    def doc_name(self) -> str:
        """Identifier of the document, e.g. ``AUDIT_COMMITTEE_BRIEFING``.

        Version 1 documents start with their identifier on the first line.
        """
        return self.document.strip().split("\n", 1)[0].strip()

    @property
    def doc_key(self) -> str:
        """Groups the contexts that share a document (used for the dev/test split)."""
        return f"{self.domain}/{self.doc_name}"

    def mask_terms(self) -> list[str]:
        """Spellings of the document name, removed before duplicate detection.

        Without this, "Summarize the AR_AGING report" and "Summarize the
        VENDOR_MASTER report" would count as different prompts although they
        are the same request with a different document name.
        """
        name = self.doc_name
        spaced = name.replace("_", " ")
        return [t for t in {name, spaced, spaced.title(), spaced.lower(), spaced.capitalize()} if t]


def context_key(domain: str, family: str, system_prompt: str, document: str) -> str:
    return sha256_hex("\x1f".join([domain, family, system_prompt, document]))[:16]


def scenario_context_id(scenario: dict[str, Any]) -> str:
    return context_key(
        str(scenario.get("domain", "")),
        str(scenario.get("attack_type", "")),
        str(scenario.get("system_prompt", "")),
        str(scenario.get("dataset", "")),
    )


def contexts_from_scenarios(scenarios: Iterable[dict[str, Any]]) -> list[Context]:
    """Distinct contexts, in the order they first appear in the dataset."""
    seen: dict[str, Context] = {}
    for s in scenarios:
        cid = scenario_context_id(s)
        if cid not in seen:
            seen[cid] = Context(
                context_id=cid,
                domain=str(s.get("domain", "")),
                family=str(s.get("attack_type", "")),
                system_prompt=str(s.get("system_prompt", "")),
                document=str(s.get("dataset", "")),
            )
    return list(seen.values())
