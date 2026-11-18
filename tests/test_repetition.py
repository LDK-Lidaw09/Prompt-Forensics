"""Experiment A measures and Experiment B path (protocol of the judge repetition)."""

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rep = _load("repetition_analysis", ROOT / "scripts" / "analysis_v2" / "repetition.py")


def test_wilson_and_kappa():
    lo, hi = rep.wilson(0, 300)
    assert lo == pytest.approx(0, abs=1e-12) and hi == pytest.approx(0.0126, abs=1e-4)
    assert rep.fleiss_kappa([["a"] * 4, ["b"] * 4], ["a", "b"]) == pytest.approx(1.0)
    assert pd.isna(rep.fleiss_kappa([["a"] * 4, ["a"] * 4], ["a", "b"]))


def _write(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_change_rates(tmp_path):
    sample = [
        {"item": "A-E0001", "stratum": "E2", "judge": "evaluator", "run_mode": "attacks", "scenario_id": "S1",
         "replay_idx": "3", "r0_verdict": "suspicious", "r0_unsafe": "True", "r0_raw_json": '{"verdict": "suspicious"}'},
        {"item": "A-E0002", "stratum": "E2", "judge": "evaluator", "run_mode": "attacks", "scenario_id": "S2",
         "replay_idx": "7", "r0_verdict": "malicious", "r0_unsafe": "True", "r0_raw_json": '{"verdict": "malicious"}'},
        {"item": "A-I0001", "stratum": "I1", "judge": "investigator", "run_mode": "attacks", "scenario_id": "S3",
         "replay_idx": "0", "r0_decision": "safe", "r0_maliciousness_score": "0.2", "r0_adversarial_types": "[]",
         "r0_attack_succeeded": "False", "r0_guardrail_violations": "[]", "r0_raw_json": "{}"},
    ]
    fields = sorted({k for r in sample for k in r} | set(rep.__dict__.get("SAMPLE_FIELDS", [])))
    plan = tmp_path / "plan"
    runs = tmp_path / "runs"
    plan.mkdir()
    runs.mkdir()
    _write(plan / "repetition_sample.csv", [{f: r.get(f, "") for f in fields} for r in sample])
    for i, r in enumerate(("R1", "R2", "R3")):
        second = "safe" if r == "R2" else "malicious"     # S2 loses its unsafe label once
        _write(runs / f"evaluator_{r}.csv", [
            {"run_mode": "attacks", "scenario_id": "S1", "replay_idx": "3", "verdict": "suspicious",
             "raw_judge_json": '{"verdict": "suspicious"}', "raw_judge_text": "x", "judge_ok": "True"},
            {"run_mode": "attacks", "scenario_id": "S2", "replay_idx": "7", "verdict": second,
             "raw_judge_json": json.dumps({"verdict": second}), "raw_judge_text": second, "judge_ok": "True"},
        ])
        _write(runs / f"investigator_attacks_{r}.csv", [
            {"run_mode": "attacks", "scenario_id": "S3", "decision": "safe", "maliciousness_score": str(0.2 + 0.1 * i),
             "adversarial_types": "[]", "attack_succeeded": "False", "guardrail_violations": "[]",
             "raw_json": "{}", "raw_text": "y", "judge_ok": "True"},
        ])
    out = tmp_path / "out"
    sys.argv = ["repetition.py", "--plan", str(plan), "--runs", str(runs), "--out", str(out)]
    rep.main()
    ev = pd.read_csv(out / "repetition_evaluator.csv").set_index("stratum")
    assert ev.loc["E2", "verdict_changed"] == 1 and ev.loc["E2", "unsafe_label_changed"] == 1
    assert ev.loc["E2", "verdict_change_rate"] == pytest.approx(0.5)
    rare = pd.read_csv(out / "repetition_rare_signals.csv").set_index("stratum")
    assert rare.loc["E2", "pools_losing_only_unsafe_answer"] == 1
    assert rare.loc["E2", "unsafe_in_all_four"] == 1
    inv = pd.read_csv(out / "repetition_investigator.csv").set_index("stratum")
    assert inv.loc["I1", "decision_changed"] == 0
    assert inv.loc["I1", "score_spread_max"] == pytest.approx(0.2)
    same = pd.read_csv(out / "repetition_identical_text.csv").set_index("judge")
    assert same.loc["evaluator", "raw_text_identical_r1_r3"] == pytest.approx(0.5)


def test_integrated_path_end_to_end(tmp_path, monkeypatch):
    path = _load("run_integrated_path", ROOT / "scripts" / "run_integrated_path.py")
    from prompt_forensics.agents.evaluator import normalize_verdict
    from prompt_forensics.agents.investigator import normalize_report
    from prompt_forensics.scenarios import load_scenarios, scenario_id

    def fake_eval(text, **kw):
        return normalize_verdict({"verdict": "suspicious", "score": 0.6}), "{}", {}

    def fake_inv(scenario, run_mode, text, judge, **kw):
        report = normalize_report({"decision": "attack" if run_mode == "attacks" else "safe",
                                   "maliciousness_score": 0.9, "attack_succeeded": False})
        return report, "{}", 1

    monkeypatch.setattr(path, "evaluate_output", fake_eval)
    monkeypatch.setattr(path, "investigate_pair", fake_inv)
    scen = [s for s in load_scenarios(ROOT / "data" / "scenarios_v2") if s.get("attack_prompt")][:3]
    sample = tmp_path / "sample.csv"
    _write(sample, [{"item": f"B{i:02d}", "stratum": "x", "run_mode": "attacks", "scenario_id": scenario_id(s),
                     "source_file": "", "attack_type": "", "domain": "", "benchmark_source": "", "safe_kind": ""}
                    for i, s in enumerate(scen, start=1)])
    out = tmp_path / "B"
    common = ["--out", str(out), "--sample", str(sample), "--victim", "sim:victim", "--judge", "sim:judge",
              "--workers", "4", "--key", str(tmp_path / "keys" / "k.pem")]
    for stage in ("capture", "judge", "finish"):
        monkeypatch.setattr(sys, "argv", ["run_integrated_path.py", stage, *common])
        path.main()
    steps = json.loads((out / "steps.json").read_text())
    assert steps["captured"] == 3 and steps["replays"] == 30
    assert steps["annotations"]["evaluator"] == 30 and steps["annotations"]["investigator"] == 3
    assert steps["replays_unlinked"] == 0 and steps["verification"]["problems"] == []
    assert not (out / "keys").exists()                     # the key is kept outside the results
    config = pd.read_csv(out / "replay_config.csv")
    assert set(config.field) == {"seed"} and set(config.origin) == {"override"}
    assert config.replays.sum() == 27                      # replay 0 uses the recorded seed 1234
    alt = pd.read_csv(out / "alterations.csv").set_index("case")
    for case in ("T1", "T2", "T3"):
        assert alt.loc[case, "detected"] and alt.loc[case, "detected_without_anchor"]
    for case in ("T4", "T6"):
        assert alt.loc[case, "detected"] and not alt.loc[case, "detected_without_anchor"]
    assert not alt.loc["T5", "detected"]
    table = _load("integrated_path_tables", ROOT / "scripts" / "analysis_v2" / "integrated_path.py")
    monkeypatch.setattr(sys, "argv", ["integrated_path.py", "--run", str(out), "--out", str(tmp_path / "t")])
    table.main()
    assert (tmp_path / "t" / "integrated_path_alterations.csv").exists()
