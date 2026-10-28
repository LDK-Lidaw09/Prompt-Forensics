"""End-to-end runs of the batch experiments against the offline simulator."""

import json
from collections import Counter

import pandas as pd
import pytest

from prompt_forensics.agents import investigator as investigator_agent
from prompt_forensics.experiments.investigator import InvestigatorRunConfig, run_investigator
from prompt_forensics.experiments.replays import ReplayConfig, run_replays
from prompt_forensics.llm import LLMResponse
from prompt_forensics.metrics import similarity


def fake_metrics(monkeypatch):
    monkeypatch.setattr(similarity, "cosine_matrix", lambda texts: [[1.0] * len(texts) for _ in texts])
    monkeypatch.setattr(similarity, "bertscore", lambda c, r: [(1.0, 1.0, 1.0)] * len(c))


def test_replays_then_investigator(tmp_path, monkeypatch):
    fake_metrics(monkeypatch)
    csv_path, jsonl_path = run_replays(ReplayConfig(
        run_mode="attacks", limit=2, num_replays=3, target_model="sim:victim",
        output_csv=str(tmp_path / "r.csv"), output_jsonl=str(tmp_path / "r.jsonl")), progress=lambda m: None)

    table = pd.read_csv(csv_path)
    assert len(table) == 2 and table["complete"].all()
    assert table["cons_score"].tolist() == [1.0, 1.0]
    lines = [json.loads(line) for line in open(jsonl_path)]
    first_pool = [line for line in lines if line["scenario_id"] == lines[0]["scenario_id"]]
    assert sorted(line["seed"] for line in first_pool) == [1234, 1235, 1236]
    assert table["cons_min"].tolist() == [1.0, 1.0]
    pairs = [json.loads(line) for line in open(tmp_path / "r_pairs.jsonl")]
    assert len(pairs) == 2 and pairs[0]["indices"] == [0, 1, 2]
    assert pairs[0]["cosine"] == [1.0, 1.0, 1.0] and pairs[0]["bert_f1"] == [1.0, 1.0, 1.0]

    answer = json.dumps({"decision": "attack", "maliciousness_score": 0.8, "adversarial_types": [],
                         "guardrail_violations": [], "risk_markers": [], "attack_succeeded": False,
                         "summary": "s"})
    monkeypatch.setattr(investigator_agent, "call_llm",
                        lambda p, m, params: LLMResponse(answer, m, "test", 1))
    out = tmp_path / "inv.csv"
    n = run_investigator(InvestigatorRunConfig(replays=jsonl_path, output_csv=str(out), judge_model="j",
                                               workers=2), progress=lambda m: None)
    rows = pd.read_csv(out)
    assert n == 2 and rows["correct"].tolist() == [1, 1] and rows["gold_label"].eq("attack").all()

    # Resume: nothing left to do.
    assert run_investigator(InvestigatorRunConfig(replays=jsonl_path, output_csv=str(out), judge_model="j",
                                                  resume=True), progress=lambda m: None) == 0


def _plan_scenarios():
    out = []
    for i in range(40):
        out.append({
            "id": f"S{i:03d}", "attack_type": ["jailbreak", "data_leakage"][i % 2],
            "domain": "audit", "split": "test" if i % 4 else "dev",
            "attack_prompt": f"attack {i}" if i % 5 else None,
            "safe_prompt": f"safe {i}" if i % 3 else None,
            "safe_kind": "hard_negative" if i % 7 == 0 else "standard",
        })
    return out


def test_plan_runs_is_stratified_balanced_and_deterministic():
    from prompt_forensics.experiments.plan import plan_runs

    scenarios = _plan_scenarios()
    rows = plan_runs(scenarios, deep=10, deep_replays=100, replays=10, shards=2, seed=1)
    prompts = sum(bool(s["attack_prompt"]) + bool(s["safe_prompt"]) for s in scenarios)
    assert len(rows) == prompts
    assert len({(r["scenario_id"], r["run_mode"]) for r in rows}) == prompts
    deep = [r for r in rows if r["pool"] == "deep"]
    assert len(deep) == 10 and all(r["replays"] == 100 for r in deep)
    assert all(r["replays"] == 10 for r in rows if r["pool"] == "shallow")
    shard_counts = Counter(r["shard"] for r in deep)
    assert abs(shard_counts[0] - shard_counts[1]) <= 2
    assert all(r["safe_kind"] is None for r in rows if r["run_mode"] == "attacks")
    assert rows == plan_runs(scenarios, deep=10, deep_replays=100, replays=10, shards=2, seed=1)
    with pytest.raises(ValueError):
        plan_runs(scenarios, deep=10_000)


def test_write_plan_keeps_only_the_prompt_of_each_run_mode(tmp_path):
    from prompt_forensics.experiments.plan import plan_runs, write_plan
    from prompt_forensics.scenarios import load_scenarios

    scenarios = _plan_scenarios()
    rows = plan_runs(scenarios, deep=6, shards=2)
    roots = write_plan(scenarios, rows, tmp_path)
    assert (tmp_path / "plan.csv").exists()
    total = 0
    for root in roots:
        mode = "attacks" if root.name.startswith("attacks") else "safe"
        other = "safe_prompt" if mode == "attacks" else "attack_prompt"
        loaded = load_scenarios(root)
        assert loaded and all(s[other] is None for s in loaded)
        total += len(loaded)
    assert total == len(rows)


def test_whitespace_perturbation_keeps_the_canonical_text():
    import random

    from prompt_forensics.canonical import normalize_text
    from prompt_forensics.experiments.plan import perturb_whitespace

    rng = random.Random(0)
    for text in ["Summarise the memo in three bullets.", 'Quote "two  spaces" exactly.\nThanks', "x"]:
        out = perturb_whitespace(text, rng)
        assert out != text and normalize_text(out) == normalize_text(text)
    assert '"two  spaces"' in perturb_whitespace('Quote "two  spaces" exactly.', rng)


def test_whitespace_roots_pair_original_and_perturbed(tmp_path):
    from prompt_forensics.experiments.plan import plan_runs, whitespace_sample, write_whitespace_roots
    from prompt_forensics.scenarios import load_scenarios

    scenarios = _plan_scenarios()
    rows = plan_runs(scenarios, deep=8, shards=2)
    sample = whitespace_sample(rows, 4)
    roots = write_whitespace_roots(scenarios, sample, tmp_path)
    orig = {s["id"]: s for r in roots if r.name.endswith("orig") for s in load_scenarios(r)}
    pert = {s["id"]: s for r in roots if r.name.endswith("pert") for s in load_scenarios(r)}
    assert orig.keys() == pert.keys() and len(orig) == 4
