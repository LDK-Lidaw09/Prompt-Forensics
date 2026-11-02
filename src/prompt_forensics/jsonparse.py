"""Extract the JSON object that a judge model was asked to return.

Judges are instructed to answer with one JSON object, but small models often
wrap it in Markdown fences, add prose around it, or (for reasoning models such
as Qwen3 or Phi-4-reasoning) write a ``<think>...</think>`` block first.

:func:`extract_json_object` handles these cases in a fixed order and returns
``None`` when no JSON object can be recovered. Callers must count ``None`` as a
parse failure and report it, never replace it with a default verdict.
"""

from __future__ import annotations

import json
import re
from typing import Any

_THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_FENCED_JSON = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", flags=re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Remove ``<think>`` blocks. An unclosed block means the answer was cut off."""
    text = _THINK_BLOCK.sub("", text or "")
    if re.search(r"<think>", text, flags=re.IGNORECASE):
        # The model was still reasoning when it hit max_tokens: no answer to parse.
        return ""
    return text.strip()


def _first_balanced_object(text: str) -> dict[str, Any] | None:
    """Return the first ``{...}`` span that parses as a JSON object.

    Scans every opening brace, tracking string literals and escapes so that
    braces inside strings do not end the object early.
    """
    start = text.find("{")
    while start != -1:
        depth, in_string, escaped = 0, False, False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        candidate = json.loads(text[start : index + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(candidate, dict):
                        return candidate
                    break
        start = text.find("{", start + 1)
    return None


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort recovery of the judge's JSON object, or ``None``."""
    cleaned = strip_reasoning(text)
    if not cleaned:
        return None

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    fenced = _FENCED_JSON.search(cleaned)
    if fenced:
        try:
            parsed = json.loads(fenced.group(1))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    return _first_balanced_object(cleaned)


def coerce_float(value: Any, default: float | None = None) -> float | None:
    """Convert a judge score to float. ``0.0`` stays ``0.0``.

    (The original code used ``float(x or 0.5)``, which turned a legitimate score
    of 0.0 into 0.5.)
    """
    if value is None or isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def as_str_list(value: Any) -> list[str]:
    """Normalise a judge field that should be a list of strings."""
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    # null items (JSON null) are dropped rather than turned into "None".
    return [str(v).strip() for v in value if v is not None and str(v).strip()]
