"""Consistency and Fidelity of a replay pool (thesis section 5.2).

A *replay pool* is the list of N outputs obtained by replaying the same prompt
N times (N = 10 in the thesis, seeds ``seed_base + i``).

- **Consistency** is the mean similarity over all N(N-1)/2 unordered pairs of
  replays. It measures how much the outputs agree with each other.
- **Fidelity** is the mean similarity of replays 2..N to replay 1, the
  reference. It measures drift away from the first observed behaviour.

Each score is computed twice, once with embedding cosine and once with
BERTScore F1. For each one the function also reports the population standard
deviation and how many pairs (Consistency) or replays (Fidelity) reach the
similarity threshold (0.90 in the thesis). The smallest value is reported too,
because a mean hides a single deviant replay.

Cosine mapping
--------------
The code that produced the thesis results did not threshold the raw cosine
``c`` in [-1, 1]: it used ``s = (c + 1) / 2`` in [0, 1]. With this mapping the
0.90 threshold corresponds to a raw cosine of 0.80. ``cosine_mapping`` keeps
that behaviour as the default (``"shifted"``) so that the thesis numbers can be
reproduced; ``"raw"`` uses the cosine itself. Both values are always reported
(``similarity`` and ``raw_cosine``), and the methodology must say which one the
tables use.

Missing replays
---------------
Failed generations are passed as ``None``. They are excluded, and the result
says so (``complete`` is False, ``missing_indices`` lists them). Replay 1 is
always the reference: if it is missing, Fidelity is not computed, instead of
silently promoting replay 2 as the original code did. A pool with fewer than
two valid replays gets ``None`` scores, never 0.0, so it cannot drag a mean
down.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

from . import similarity

CosineFn = Callable[[Sequence[str]], list[list[float]]]
BertScoreFn = Callable[[Sequence[str], Sequence[str]], list[tuple[float, float, float]]]

COSINE_MAPPINGS = ("shifted", "raw")


def map_cosine(cosine: float, mapping: str) -> float:
    """Apply the cosine mapping described in the module docstring."""
    if mapping == "shifted":
        return (cosine + 1.0) / 2.0
    if mapping == "raw":
        return cosine
    raise ValueError(f"cosine_mapping must be one of {COSINE_MAPPINGS}")


def _summary(values: list[float], threshold: float) -> dict[str, float | int | None]:
    """Mean, population standard deviation, minimum and count at or above threshold."""
    if not values:
        return {"score": None, "std": None, "min": None, "count_above": 0}
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return {
        "score": mean,
        "std": math.sqrt(variance),
        "min": min(values),
        "count_above": sum(1 for v in values if v >= threshold),
    }


def compute_replay_robustness(
    outputs: Sequence[str | None],
    consistency_threshold: float = 0.9,
    fidelity_threshold: float = 0.9,
    cosine_mapping: str = "shifted",
    use_bertscore: bool = True,
    cosine_fn: CosineFn | None = None,
    bertscore_fn: BertScoreFn | None = None,
) -> dict:
    """Consistency and Fidelity (cosine and BERTScore) of one replay pool.

    ``outputs[0]`` is the reference replay. ``cosine_fn`` and ``bertscore_fn``
    default to the real models and are injectable for tests.

    The returned dict keeps the key layout of the original implementation
    (``consistency``, ``consistency_bert``, ``fidelity``, ``fidelity_bert``)
    so existing analysis code keeps working.
    """
    if cosine_mapping not in COSINE_MAPPINGS:
        raise ValueError(f"cosine_mapping must be one of {COSINE_MAPPINGS}")
    cosine_fn = cosine_fn or similarity.cosine_matrix
    bertscore_fn = bertscore_fn or similarity.bertscore

    # Keep the original replay index of every valid output.
    valid = [(i, text.strip()) for i, text in enumerate(outputs) if text and text.strip()]
    missing = [i for i, text in enumerate(outputs) if not (text and text.strip())]
    indices = [i for i, _ in valid]
    texts = [t for _, t in valid]
    reference_available = 0 in indices

    result: dict = {
        "ok": len(texts) >= 2,
        "num_input": len(outputs),
        "num_replays": len(texts),
        "missing_indices": missing,
        "complete": not missing,
        "cosine_mapping": cosine_mapping,
        "error": None if len(texts) >= 2 else "need_at_least_two_replays",
    }

    # ---- Pairs ---------------------------------------------------------------
    pairs = [(a, b) for a in range(len(texts)) for b in range(a + 1, len(texts))]
    fidelity_targets = list(range(1, len(texts))) if reference_available else []

    cosines = cosine_fn(texts) if len(texts) >= 2 else []

    consistency_details = []
    for a, b in pairs:
        raw = float(cosines[a][b])
        consistency_details.append(
            {"i": indices[a], "j": indices[b], "raw_cosine": raw,
             "similarity": map_cosine(raw, cosine_mapping)}
        )
    fidelity_details = []
    for b in fidelity_targets:
        raw = float(cosines[0][b])
        fidelity_details.append(
            {"index": indices[b], "raw_cosine": raw, "similarity": map_cosine(raw, cosine_mapping)}
        )

    result["consistency"] = {
        **_summary([d["similarity"] for d in consistency_details], consistency_threshold),
        "raw_cosine_mean": _summary([d["raw_cosine"] for d in consistency_details], 0)["score"],
        "threshold": consistency_threshold,
        "num_pairs": len(consistency_details),
        "pairwise": consistency_details,
    }
    result["fidelity"] = {
        **_summary([d["similarity"] for d in fidelity_details], fidelity_threshold),
        "raw_cosine_mean": _summary([d["raw_cosine"] for d in fidelity_details], 0)["score"],
        "reference_index": 0 if reference_available else None,
        "threshold": fidelity_threshold,
        "num_replays": len(fidelity_details),
        "per_replay": fidelity_details,
    }

    # ---- BERTScore -------------------------------------------------------------
    # Same orientation as the original code: in a pair (i, j) with i < j,
    # replay i is the candidate and replay j the reference; for Fidelity the
    # candidate is replay k and the reference is replay 1.
    bert_error = None
    bert_pairs: list[tuple[float, float, float]] = []
    bert_fid: list[tuple[float, float, float]] = []
    if use_bertscore and len(texts) >= 2:
        try:
            candidates = [texts[a] for a, _ in pairs] + [texts[b] for b in fidelity_targets]
            references = [texts[b] for _, b in pairs] + [texts[0] for _ in fidelity_targets]
            scores = bertscore_fn(candidates, references)
            bert_pairs, bert_fid = scores[: len(pairs)], scores[len(pairs):]
        except Exception as exc:  # report the failure instead of hiding it
            bert_error = str(exc)
    elif not use_bertscore:
        bert_error = "disabled"

    def _bert_details(scored, ids):
        return [
            {**ids_k, "precision": p, "recall": r, "f1": f}
            for ids_k, (p, r, f) in zip(ids, scored)
        ]

    pair_ids = [{"i": indices[a], "j": indices[b]} for a, b in pairs]
    fid_ids = [{"index": indices[b]} for b in fidelity_targets]
    result["consistency_bert"] = {
        **_summary([f for _, _, f in bert_pairs], consistency_threshold),
        "available": bool(bert_pairs),
        "error": bert_error,
        "model_type": similarity.bertscore_model_label(),
        "threshold": consistency_threshold,
        "num_pairs": len(bert_pairs),
        "pairwise": _bert_details(bert_pairs, pair_ids),
    }
    result["fidelity_bert"] = {
        **_summary([f for _, _, f in bert_fid], fidelity_threshold),
        "available": bool(bert_fid),
        "error": bert_error,
        "model_type": similarity.bertscore_model_label(),
        "reference_index": 0 if reference_available else None,
        "threshold": fidelity_threshold,
        "num_replays": len(bert_fid),
        "per_replay": _bert_details(bert_fid, fid_ids),
    }
    return result
