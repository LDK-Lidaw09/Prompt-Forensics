"""Regex flags for sensitive content in a model output (demo UI).

A quick, transparent first pass shown next to each answer. It is deliberately
simple and has false positives (any 13-19 digit number looks like a card
number); it is not used in the thesis experiments.
"""

from __future__ import annotations

import re

PATTERNS = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "phone": re.compile(r"(?:\+?\d{1,3}[ -]?)?(?:\(?\d{2,4}\)?[ -]?)?\d{3,4}[ -]?\d{4}"),
    "ipv4": re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
    "secret_like": re.compile(
        r"\b(?:(?:api|secret|token|key)[-_]?(?:id|key|secret)?)[\"'=: ]{1,3}[A-Za-z0-9_\-]{10,}\b", re.I
    ),
    "password_hint": re.compile(r"\bpassword\b\s*[:=]\s*\S+", re.I),
}
LEAK_KEYWORDS = ("password", "secret", "private key", "token", "confidential", "internal")


def detect_sensitive(text: str) -> dict[str, object]:
    """Which patterns match ``text``, plus the leak keywords it contains."""
    text = text or ""
    flags: dict[str, object] = {name: bool(p.search(text)) for name, p in PATTERNS.items()}
    lowered = text.lower()
    flags["keywords"] = [k for k in LEAK_KEYWORDS if k in lowered]
    return flags


def has_sensitive(flags: dict[str, object]) -> bool:
    return any(bool(v) for v in flags.values())
