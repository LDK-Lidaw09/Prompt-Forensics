"""Experiments C (contrasted policies) and D (Fidelity against a recorded answer), with fake models."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "data" / "extension" / "plan"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not (PLAN / "policy_pairs.csv").exists(), reason="plan not drawn")
def test_extension_end_to_end(tmp_path, monkeypatch):
    run = _load("run_extension", ROOT / "scripts" / "run_extension.py")
    from prompt_forensics.agents.evaluator import normalize_verdict
    from prompt_forensics.agents.investigator import normalize_report
    from prompt_forensics.llm import LLMResponse
    import prompt_forensics.metrics.similarity as similarity

    def fake_llm(prompt, model, params):
        return LLMResponse(text=f"answer {params['seed'] % 3} {len(prompt) % 5}", model=model, provider="sim",
                           timing_ms=0, finish_reason="stop")

    def fake_inv(scenario, run_mode, text, judge, **kw):
        # the heuristic "restrictive policy -> attack" that the controls must expose
        forbid = "MUST" in scenario["system_prompt"]
        report = normalize_report({"decision": "attack" if forbid else "safe", "maliciousness_score": 0.5,
                                   "guardrail_violations": ["reproduce word for word"] if forbid else []})
        return report, "{}", 1

    def fake_eval(text, **kw):
        return normalize_verdict({"verdict": "suspicious" if text.startswith("answer 1") else "safe"}), "{}", {}

    monkeypatch.setattr(run, "call_llm", fake_llm)
    monkeypatch.setattr(run, "investigate_pair", fake_inv)
    monkeypatch.setattr(run, "evaluate_output", fake_eval)
    monkeypatch.setattr(similarity, "cosine_matrix",
                        lambda texts: [[1.0 if a == b else 0.8 for b in texts] for a in texts])
    monkeypatch.setattr(similarity, "bertscore", lambda c, r: [(0, 0, 1.0 if a == b else 0.85) for a, b in zip(c, r)])
    for stage in ("policies-victim", "policies-judge", "fidelity-record", "fidelity-replay", "fidelity-judge",
                  "fidelity-score"):
        monkeypatch.setattr(sys, "argv", ["run_extension.py", stage, "--out", str(tmp_path), "--workers", "4"])
        run.main()
    sub = run.substitutes(pd.read_csv(PLAN / "fidelity_sample.csv", dtype=str).to_dict("records"))
    assert all(a != b for a, b in sub.items())

    tables = _load("extension_tables", ROOT / "scripts" / "analysis_v2" / "extension.py")
    monkeypatch.setattr(sys, "argv", ["extension.py", "--run", str(tmp_path), "--out", str(tmp_path / "t")])
    tables.main()
    assert len(pd.read_csv(tmp_path / "C" / "answers.csv")) == 70   # one answer per pair
    same = pd.read_csv(tmp_path / "t" / "policies_same_answer_check.csv").iloc[0]
    assert same.pairs == same.pairs_same_answer == 70
    pairs = pd.read_csv(tmp_path / "t" / "policies_pairs.csv").set_index(["pair_kind", "rule_type"])
    assert pairs.loc[("contrast", "all"), "both_correct"] == 50
    assert pairs.loc[("control_neutral", "all"), "both_correct"] == 0
    assert pairs.loc[("control_neutral", "all"), "decision_changed_with_rule"] == 10
    assert pairs.loc[("control_harmful", "all"), "both_correct"] == 0
    summary = pd.read_csv(tmp_path / "t" / "fidelity_summary.csv").set_index("run_mode")
    assert summary.loc["all", "prompts"] == 200
    assert (tmp_path / "t" / "fidelity_counterexamples.csv").exists()
