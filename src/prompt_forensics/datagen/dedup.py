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

    def __init__(self, embed: EmbedFn | None, threshold: float = 0.90, chunk_size: int = 1024):
        # embed=None turns near-duplicate detection off (exact only).
        self.embed = embed
        self.threshold = threshold
        # Candidates are compared to the kept prompts by blocks of this size,
        # so memory stays at chunk_size x kept similarities.
        self.chunk_size = chunk_size
        self._keys: dict[str, str] = {}       # exact key of a kept prompt -> its id
        self._redirect: dict[str, str] = {}   # exact key of a rejected prompt -> kept id it duplicates
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

    def _exact_target(self, key: str) -> str | None:
        return self._keys.get(key) or self._redirect.get(key)

    def add_batch(self, items: Sequence[tuple[str, str]]) -> list[Verdict]:
        """Check ``(item_id, masked_key_text)`` pairs in order; keep the new ones.

        ``masked_key_text`` is the text used for comparison (document name
        already masked). Returns one verdict per item. ``duplicate_of`` always
        names a kept prompt.
        """
        verdicts: list[Verdict] = []
        for start in range(0, len(items), self.chunk_size):
            verdicts += self._add_chunk(items[start : start + self.chunk_size])
        return verdicts

    def _add_chunk(self, items: Sequence[tuple[str, str]]) -> list[Verdict]:
        keys = [exact_key(text) for _, text in items]
        vectors = old_sims = new_sims = None
        if self.embed is not None:
            vectors = self.embed([text for _, text in items])
            kept = self._kept_matrix()
            old_sims = vectors @ kept.T if kept is not None and len(kept) else None
            new_sims = vectors @ vectors.T

        verdicts: list[Verdict] = []
        accepted: list[int] = []  # positions in this chunk kept so far
        for i, ((item_id, _), key) in enumerate(zip(items, keys)):
            target = self._exact_target(key)
            if target is not None:
                verdicts.append(Verdict(EXACT_DUPLICATE, target, 1.0))
                continue
            if vectors is None:
                self._keep(item_id, key, None)
                verdicts.append(Verdict(KEPT))
                continue
            best, best_id = -1.0, None
            if old_sims is not None:
                j = int(np.argmax(old_sims[i]))
                best, best_id = float(old_sims[i, j]), self._ids[j]
            for q in accepted:
                if new_sims[i, q] > best:
                    best, best_id = float(new_sims[i, q]), items[q][0]
            if best_id is not None and best >= self.threshold:
                self._redirect[key] = best_id
                verdicts.append(Verdict(NEAR_DUPLICATE, best_id, round(best, 4)))
                continue
            accepted.append(i)
            self._keep(item_id, key, vectors[i : i + 1])
            verdicts.append(Verdict(KEPT, None, round(best, 4) if best_id is not None else None))
        return verdicts

    def _keep(self, item_id: str, key: str, vector: np.ndarray | None) -> None:
        self._keys[key] = item_id
        if vector is not None:
            self._ids.append(item_id)
            self._chunks.append(vector)
