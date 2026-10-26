"""End-to-end runs of the batch experiments against the offline simulator."""

import json

import pandas as pd

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
