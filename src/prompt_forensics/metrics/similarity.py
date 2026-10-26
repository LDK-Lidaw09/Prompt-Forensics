"""Similarity backends: embedding cosine and BERTScore (thesis 5.2.2 and 5.2.3).

Both models are heavy (PyTorch), so they are imported and loaded lazily, once
per process, the first time a metric is computed. Importing this module is
cheap.

- **Cosine similarity** uses sentence-transformers with
  ``Alibaba-NLP/gte-large-en-v1.5`` (``EMBED_MODEL_NAME``). Embeddings are
  L2-normalised, so the cosine is a dot product, in [-1, 1].
- **BERTScore** uses the ``bert_score`` library with its default backbone for
  English (``roberta-large``, layer 17), no IDF weighting and no baseline
  rescaling, as reported in the thesis. A single ``BERTScorer`` is reused for
  all pairs; the original code reloaded roberta-large for every pair.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from ..config import get_settings


@lru_cache(maxsize=2)
def get_embedder(model_name: str | None = None, device: str | None = None):
    """Load (once) and return the sentence-transformers model."""
    from sentence_transformers import SentenceTransformer

    settings = get_settings()
    return SentenceTransformer(
        model_name or settings.embed_model_name,
        # gte-large-en-v1.5 ships custom modelling code on the Hugging Face Hub.
        trust_remote_code=True,
        device=device or settings.embed_device,
    )


def cosine_matrix(texts: Sequence[str]) -> list[list[float]]:
    """Pairwise cosine similarity of ``texts``, as a nested list in [-1, 1]."""
    from sentence_transformers import util

    settings = get_settings()
    embeddings = get_embedder().encode(
        list(texts),
        normalize_embeddings=True,
        convert_to_tensor=True,
        batch_size=settings.embed_batch_size,
    )
    return util.cos_sim(embeddings, embeddings).tolist()


def cosine_pair(a: str, b: str) -> float:
    """Cosine similarity of two texts, in [-1, 1]."""
    return cosine_matrix([a, b])[0][1]


@lru_cache(maxsize=2)
def get_bert_scorer(lang: str | None = None, model_type: str | None = None):
    """Load (once) and return a ``bert_score.BERTScorer``.

    Returns ``None`` if the ``bert_score`` package is not installed.
    """
    try:
        from bert_score import BERTScorer
    except ImportError:
        return None
    settings = get_settings()
    return BERTScorer(
        lang=lang or settings.bertscore_lang,
        model_type=(model_type or settings.bertscore_model_type) or None,
        rescale_with_baseline=False,
    )


def bertscore(candidates: Sequence[str], references: Sequence[str]) -> list[tuple[float, float, float]]:
    """BERTScore (precision, recall, F1) for each (candidate, reference) pair.

    Raises ``RuntimeError`` if ``bert_score`` is not installed.
    """
    scorer = get_bert_scorer()
    if scorer is None:
        raise RuntimeError("bert_score is not installed")
    if not candidates:
        return []
    precision, recall, f1 = scorer.score(list(candidates), list(references), verbose=False)
    return [(float(p), float(r), float(f)) for p, r, f in zip(precision, recall, f1)]


def bertscore_model_label() -> str:
    """Human-readable name of the BERTScore backbone, for result files."""
    settings = get_settings()
    return settings.bertscore_model_type or f"default for lang={settings.bertscore_lang}"
