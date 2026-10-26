"""Result tables, checked on small hand-made CSV contents."""

import math

import pandas as pd
import pytest

from prompt_forensics.experiments import analysis


def test_wilson_interval_known_value():
    low, high = analysis.wilson_interval(5, 10)
    assert low == pytest.approx(0.2366, abs=1e-4) and high == pytest.approx(0.7634, abs=1e-4)
    assert all(math.isnan(x) for x in analysis.wilson_interval(0, 0))


def evaluator_rows(scenario, family, unsafe_at=(), unknown_at=(), k=10, domain="health"):
    return [{"scenario_id": scenario, "replay_idx": i, "attack_type": family, "domain": domain,
             "verdict": "unknown" if i in unknown_at else ("malicious" if i in unsafe_at else "safe"),
             "unsafe": i in unsafe_at, "judge_ok": i not in unknown_at} for i in range(k)]


def test_tail_risk_counts_complete_pools_only():
    df = pd.DataFrame(
        evaluator_rows("a", "jailbreak", unsafe_at=(7,))
        + evaluator_rows("b", "jailbreak")
        + evaluator_rows("c", "jailbreak", unsafe_at=(1,), unknown_at=(2,))   # incomplete
    )
    overall = analysis.tail_risk_tables(df)["overall"].iloc[0]
    assert overall["pools"] == 2 and overall["pools_with_unsafe"] == 1
    assert overall["tail_risk"] == 0.5 and overall["excluded_incomplete"] == 1
    assert overall["unsafe_replay_rate"] == pytest.approx(1 / 20)


def test_tail_risk_grows_with_pool_size():
    df = pd.DataFrame(evaluator_rows("a", "jailbreak", unsafe_at=(7,)) + evaluator_rows("b", "jailbreak"))
    by_m = analysis.tail_risk_by_pool_size(df).set_index("m")["tail_risk"]
    assert by_m[7] == 0.0 and by_m[8] == 0.5 and by_m.is_monotonic_increasing


def test_paired_mcnemar():
    attacks = pd.DataFrame(evaluator_rows("a", "jb", unsafe_at=(0,)) + evaluator_rows("b", "jb", unsafe_at=(0,)))
    safe = pd.DataFrame(evaluator_rows("a", "jb") + evaluator_rows("b", "jb", unsafe_at=(3,)))
    paired = analysis.paired_tail_risk(attacks, safe)
    assert paired["common_scenarios"] == 2
    assert paired["b_safe_clean_attack_unsafe"] == 1 and paired["c_safe_unsafe_attack_clean"] == 0
    assert paired["delta"] == 0.5


def investigator_df(gold, decisions, scores, ok=None):
    ok = ok or [True] * len(decisions)
    return pd.DataFrame({"attack_type": "jailbreak", "gold_label": gold, "decision": decisions,
                         "maliciousness_score": scores, "judge_ok": ok})


def test_investigator_rules_are_reported_separately():
    attacks = investigator_df("attack", ["attack", "attack", "safe", None], [0.9, 0.5, 0.1, None],
                              ok=[True, True, True, False])
    safe = investigator_df("safe", ["safe", "attack"], [0.2, 0.7])
    tables = analysis.investigator_tables({"J": (attacks, safe)})
    t = tables["table_6_4"].set_index("rule")
    # decision field: TP=2 FN=1 FP=1 ; score >= 0.6: TP=1 FN=2 FP=1
    assert (t.loc["decision", "tp"], t.loc["decision", "fn"], t.loc["decision", "fp"]) == (2, 1, 1)
    assert (t.loc["threshold", "tp"], t.loc["threshold", "fn"], t.loc["threshold", "fp"]) == (1, 2, 1)
    assert t.loc["decision", "failed_attacks"] == 1
    recall = tables["table_6_5"].set_index("rule").loc["decision", "recall"]
    assert recall == pytest.approx(2 / 3)


def test_best_threshold_takes_first_maximum():
    attacks = investigator_df("attack", ["attack"] * 2, [0.8, 0.9])
    safe = investigator_df("safe", ["safe"] * 2, [0.1, 0.2])
    threshold, f1 = analysis.best_threshold(attacks, safe)
    assert threshold == 0.21 and f1 == 1.0


def test_replay_tables_deduplicate_and_average():
    rows = [
        {"scenario_id": "a", "source_file": "f", "run_mode": "attacks", "attack_type": "jailbreak",
         "complete": False, "cons_score": 0.1, "fid_score": 0.1, "cons_bert_score": 0.1, "fid_bert_score": 0.1},
        # same scenario, resumed run: replaces the row above
        {"scenario_id": "a", "source_file": "f", "run_mode": "attacks", "attack_type": "jailbreak",
         "complete": True, "cons_score": 0.95, "fid_score": 0.9, "cons_bert_score": 0.8, "fid_bert_score": None},
        {"scenario_id": "b", "source_file": "f", "run_mode": "attacks", "attack_type": "jailbreak",
         "complete": True, "cons_score": 0.85, "fid_score": 0.9, "cons_bert_score": 0.8, "fid_bert_score": 0.9},
    ]
    table = analysis.replay_tables(pd.DataFrame(rows)).set_index("family")
    assert list(table.index) == ["JB", "Micro"]
    assert table.loc["JB", "n"] == 2 and table.loc["JB", "incomplete"] == 0
    assert table.loc["JB", "cons_cos_mean"] == pytest.approx(0.9)
    assert table.loc["JB", "cons_cos_std"] == pytest.approx(0.05)
    assert table.loc["JB", "cons_cos_frac_ge"] == 0.5
    assert table.loc["JB", "fid_bert_mean"] == pytest.approx(0.9)   # missing value ignored
    assert "JB" in analysis.replay_tables_latex(table.reset_index(), "c", "l")
