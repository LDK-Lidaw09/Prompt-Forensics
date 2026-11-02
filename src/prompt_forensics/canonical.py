"""Canonical serialization and Prompt ID (thesis sections 4.2.1 and 4.2.2).

An interaction is first encoded as a JSON object (prompt, system instructions,
model, generation parameters, timestamp, contextual identifiers). Before
hashing, the object is normalised so that two representations of the same
interaction always produce the same bytes:

1. every string is converted to Unicode Normalization Form C (NFC);
2. line endings are converted to LF;
3. non-significant whitespace is collapsed: leading and trailing whitespace is
   removed, and runs of spaces or tabs outside double-quoted spans are reduced
   to one space (line breaks are kept, because they can carry meaning);
4. keys whose value is ``None`` are omitted rather than serialised as ``null``,
   so that the presence of a field is itself part of the representation;
5. the result is serialised as UTF-8 JSON with sorted keys and fixed
   separators.

Two identifiers are derived from the canonical bytes:

- the **Prompt ID** hashes the full payload, timestamp and user included. It
  identifies one capture *event* and is unique per submission.
- the **content hash** hashes only what determines the model input (prompt,
  system instructions, model, parameters). Two submissions of the same prompt
  under the same configuration share it, which is what lets replays of the
  same interaction be grouped.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Any, Mapping

# A run of spaces or tabs. Line breaks are deliberately not included.
_HORIZONTAL_WS = re.compile(r"[ \t]+")
# Splits a string into double-quoted spans and the text between them.
_QUOTED_SPAN = re.compile(r'("[^"\n]*")')

# Fields of an interaction payload that do not change what the model receives.
# They are excluded from the content hash.
EVENT_ONLY_FIELDS = frozenset({"timestamp", "user_id", "session_id"})


def normalize_text(text: str) -> str:
    """Apply the text normalisation rules of section 4.2.1 to one string."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse whitespace only outside double-quoted spans, so that quoted
    # content (for example a literal string the user asked about) is kept as is.
    # re.split with a capturing group puts the quoted spans at odd indices.
    parts = _QUOTED_SPAN.split(text)
    text = "".join(
        part if i % 2 == 1 else _HORIZONTAL_WS.sub(" ", part)
        for i, part in enumerate(parts)
    )
    # Remove spaces left at the start or end of each line, then around the text.
    text = "\n".join(line.strip(" \t") for line in text.split("\n"))
    return text.strip()


def normalize_value(value: Any) -> Any:
    """Recursively normalise strings and drop ``None`` values from mappings."""
    if isinstance(value, str):
        return normalize_text(value)
    if isinstance(value, Mapping):
        return {
            unicodedata.normalize("NFC", str(k)): normalize_value(v)
            for k, v in value.items()
            if v is not None
        }
    if isinstance(value, (list, tuple)):
        return [normalize_value(v) for v in value]
    return value


def canonical_json(payload: Mapping[str, Any]) -> str:
    """Return the canonical JSON string of a payload."""
    return json.dumps(
        normalize_value(payload),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def sha256_hex(data: str | bytes) -> str:
    """SHA-256 of a string (encoded as UTF-8) or of raw bytes, as hex."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def build_interaction_payload(
    *,
    prompt: str,
    model: str,
    params: Mapping[str, Any],
    timestamp: str,
    system: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    context_hash: str | None = None,
) -> dict[str, Any]:
    """Assemble the JSON object that represents one captured interaction.

    ``params`` should only hold generation parameters that affect the output
    (temperature, max_tokens, seed, ...).
    """
    return {
        "prompt": prompt,
        "system": system,
        "model": model,
        "params": dict(params),
        "timestamp": timestamp,
        "user_id": user_id,
        "session_id": session_id,
        "context_hash": context_hash,
    }


def prompt_id(payload: Mapping[str, Any]) -> str:
    """Prompt ID: SHA-256 of the canonical payload (unique per capture event)."""
    return sha256_hex(canonical_json(payload))


def content_hash(payload: Mapping[str, Any]) -> str:
    """Hash of the payload without event-only fields (timestamp, user, session)."""
    content = {k: v for k, v in payload.items() if k not in EVENT_ONLY_FIELDS}
    return sha256_hex(canonical_json(content))


def text_hash(text: str) -> str:
    """SHA-256 of a raw text, used for model responses (no normalisation)."""
    return sha256_hex(text or "")
