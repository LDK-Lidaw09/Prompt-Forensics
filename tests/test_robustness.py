import pytest

from prompt_forensics.metrics.robustness import compute_replay_robustness


def fake_cosine(values):
    """Cosine matrix where texts are equal (1.0) or orthogonal (0.0)."""
    def cosine(texts):
        return [[1.0 if a == b else values for b in texts] for a in texts]
    return cosine


def fake_bert(candidates, references):
    return [(1.0, 1.0, 1.0) if c == r else (0.5, 0.5, 0.5) for c, r in zip(candidates, references)]


def run(outputs, **kw):
    return compute_replay_robustness(outputs, cosine_fn=fake_cosine(0.0), bertscore_fn=fake_bert, **kw)


def test_identical_replays_score_one():
    result = run(["a", "a", "a"])
    assert result["consistency"]["score"] == 1.0
    assert result["fidelity"]["score"] == 1.0
    assert result["consistency_bert"]["score"] == 1.0
    assert result["complete"]


def test_shifted_mapping_reproduces_thesis_scale():
    # Orthogonal texts: raw cosine 0 -> (0 + 1) / 2 = 0.5 with the thesis mapping.
    shifted = run(["a", "b"])
    raw = run(["a", "b"], cosine_mapping="raw")
    assert shifted["consistency"]["score"] == pytest.approx(0.5)
    assert raw["consistency"]["score"] == pytest.approx(0.0)
    assert shifted["consistency"]["raw_cosine_mean"] == pytest.approx(0.0)


def test_pairs_and_population_std():
    result = run(["a", "a", "b"])  # pairs: (a,a)=1, (a,b)=.5, (a,b)=.5
    assert result["consistency"]["num_pairs"] == 3
    assert result["consistency"]["score"] == pytest.approx(2 / 3)
    assert result["consistency"]["std"] == pytest.approx(0.2357, abs=1e-4)
    assert result["consistency"]["count_above"] == 1
    assert result["consistency"]["min"] == pytest.approx(0.5)


def test_missing_replays_are_reported_not_scored_as_zero():
    result = run(["a", None, "a", ""])
    assert result["missing_indices"] == [1, 3]
    assert not result["complete"]
    assert result["consistency"]["score"] == 1.0
    assert [d["index"] for d in result["fidelity"]["per_replay"]] == [2]


def test_fidelity_is_undefined_without_replay_one():
    result = run([None, "a", "b"])
    assert result["fidelity"]["score"] is None
    assert result["fidelity"]["reference_index"] is None
    assert result["consistency"]["score"] is not None


def test_bertscore_failure_is_reported():
    def broken(c, r):
        raise RuntimeError("no model")
    result = compute_replay_robustness(["a", "b"], cosine_fn=fake_cosine(0.0), bertscore_fn=broken)
    assert result["consistency_bert"]["score"] is None
    assert result["consistency_bert"]["error"] == "no model"


def test_single_output_is_not_ok():
    result = run(["a"])
    assert not result["ok"] and result["consistency"]["score"] is None
