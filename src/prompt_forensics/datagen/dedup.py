"""Exact and near-duplicate detection for generated prompts.

Two prompts are duplicates when they make the same request, whatever the
document they name. Detection has two levels:

1. **Exact**: same text after normalisation (NFC, whitespace, case,
   punctuation) and after replacing the document name by ``<doc>``.
2. **Near**: cosine similarity of the sentence embeddings (raw cosine, in
   [-1, 1]) at or above a threshold, 0.90 by default. This catches
   paraphrases ("Summarize this report in 3 points" / "Give me a three-point
   summary of the report"), which an exact hash misses.

The index is incremental: candidates are checked, in order, against every
prompt kept so far and against the candidates of the same batch kept before
them. The first occurrence is kept, so the result does not depend on how
the work was split into batches.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from ..canonical import normalize_text

EmbedFn = Callable[[Sequence[str]], np.ndarray]

KEPT = "kept"
EXACT_DUPLICATE = "exact_duplicate"
NEAR_DUPLICATE = "near_duplicate"

_NON_WORD = re.compile(r"[^\w<>]+")


def mask_document(text: str, terms: Sequence[str]) -> str:
    """Replace every spelling of the document name by ``<doc>`` (case-insensitive)."""
    for term in sorted(set(terms), key=len, reverse=True):
        if term:
            text = re.sub(re.escape(term), "<doc>", text, flags=re.IGNORECASE)
    return text


def exact_key(text: str, terms: Sequence[str] = ()) -> str:
    """Key under which two prompts count as the same text."""
    text = mask_document(normalize_text(text).casefold(), terms)
    return " ".join(_NON_WORD.sub(" ", text).split())


@dataclass(frozen=True)
class Verdict:
    status: str                       # kept | exact_duplicate | near_duplicate
    duplicate_of: str | None = None   # id of the kept prompt it duplicates
    similarity: float | None = None   # highest cosine to a kept prompt


def default_embedder(batch_size: int = 64) -> EmbedFn:
    """Sentence embeddings with the model used for the replay metrics (gte-large)."""
    from ..metrics.similarity import get_embedder

    def embed(texts: Sequence[str]) -> np.ndarray:
        vectors = get_embedder().encode(
            list(texts), normalize_embeddings=True, convert_to_numpy=True, batch_size=batch_size
        )
        return np.asarray(vectors, dtype=np.float32)

    return embed


class DedupIndex:
    """Prompts kept so far, with their exact keys and embeddings."""

    def __init__(self, embed: EmbedFn | None, threshold: float = 0.90):
        # embed=None turns near-duplicate detection off (exact only).
        self.embed = embed
        self.threshold = threshold
        self._keys: dict[str, str] = {}
        self._ids: list[str] = []
        self._chunks: list[np.ndarray] = []
        self._matrix: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self._keys)

    def _kept_matrix(self) -> np.ndarray | None:
        if self._chunks:
            parts = ([self._matrix] if self._matrix is not None else []) + self._chunks
            self._matrix = np.vstack(parts)
            self._chunks = []
        return self._matrix

    def add_batch(self, items: Sequence[tuple[str, str]]) -> list[Verdict]:
        """Check ``(item_id, masked_key_text)`` pairs in order; keep the new ones.

        ``masked_key_text`` is the text used for comparison (document name
        already masked). Returns one verdict per item.
        """
        verdicts: list[Verdict | None] = [None] * len(items)
        keys = [exact_key(text) for _, text in items]
        pending: list[int] = []
        batch_keys: dict[str, str] = {}
        for i, ((item_id, _), key) in enumerate(zip(items, keys)):
            first = self._keys.get(key) or batch_keys.get(key)
            if first is not None:
                verdicts[i] = Verdict(EXACT_DUPLICATE, first, 1.0)
            else:
                batch_keys[key] = item_id
                pending.append(i)

        if not pending:
            return [v for v in verdicts if v is not None]

        if self.embed is None:
            for i in pending:
                self._keep(items[i][0], keys[i], None)
                verdicts[i] = Verdict(KEPT)
            return list(verdicts)  # type: ignore[arg-type]

        vectors = self.embed([items[i][1] for i in pending])
        kept = self._kept_matrix()
        old_sims = vectors @ kept.T if kept is not None and len(kept) else None
        new_sims = vectors @ vectors.T
        accepted: list[int] = []  # positions in `pending` kept so far
        for p, i in enumerate(pending):
            best, best_id = -1.0, None
            if old_sims is not None:
                j = int(np.argmax(old_sims[p]))
                best, best_id = float(old_sims[p, j]), self._ids[j]
            for q in accepted:
                if new_sims[p, q] > best:
                    best, best_id = float(new_sims[p, q]), items[pending[q]][0]
            if best_id is not None and best >= self.threshold:
                verdicts[i] = Verdict(NEAR_DUPLICATE, best_id, round(best, 4))
                continue
            accepted.append(p)
            verdicts[i] = Verdict(KEPT, None, round(best, 4) if best_id is not None else None)

        for p in accepted:
            i = pending[p]
            self._keep(items[i][0], keys[i], vectors[p : p + 1])
        return list(verdicts)  # type: ignore[arg-type]

    def _keep(self, item_id: str, key: str, vector: np.ndarray | None) -> None:
        self._keys[key] = item_id
        if vector is not None:
            self._ids.append(item_id)
            self._chunks.append(vector)
