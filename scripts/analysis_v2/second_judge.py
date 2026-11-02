"""Agreement between the main Evaluator (Qwen3-14B) and the second judge on the 20,203-replay sample,
and of both judges with the human validation labels.

    python scripts/analysis_v2/second_judge.py results/v2/second_judge/evaluator_gemma-3-27b-it.csv \
        --human validation_evaluator_resultats.csv
"""
import argparse, os
import pandas as pd
from sklearn.metrics import cohen_kappa_score

W = os.environ.get("RUNS_V2_WORK", "results/v2/work")
p = argparse.ArgumentParser()
p.add_argument("second_csv")
p.add_argument("--human", help="merged human validation CSV (item key + 'human' column)")
a = p.parse_args()
UNSAFE = {"suspicious", "malicious"}
q = pd.read_parquet(f"{W}/eval_main_clean.parquet",
                    columns=["run_mode", "scenario_id", "replay_idx", "verdict", "victim", "attack_type", "domain", "safe_kind"])
q = q[q.victim == "Mistral-7B"]
g = pd.read_csv(a.second_csv, usecols=["run_mode", "scenario_id", "replay_idx", "verdict", "judge_ok", "line_number"])
g = g.sort_values(["judge_ok", "line_number"]).drop_duplicates(["run_mode", "scenario_id", "replay_idx"], keep="last")
print("second judge rows", len(g), "failed or unknown", int((~g.judge_ok.astype(bool) | g.verdict.eq("unknown")).sum()))
g = g[g.judge_ok.astype(bool) & g.verdict.ne("unknown")]
m = g.merge(q, on=["run_mode", "scenario_id", "replay_idx"], suffixes=("_2", "_q"))
rows = []
def agree(d, label):
    out = {"group": label, "n": len(d)}
    for name, cls in [("unsafe", UNSAFE), ("malicious", {"malicious"})]:
        x, y = d.verdict_q.isin(cls), d.verdict_2.isin(cls)
        out[f"qwen_{name}"] = x.mean(); out[f"second_{name}"] = y.mean()
        out[f"agreement_{name}"] = (x == y).mean()
        out[f"kappa_{name}"] = cohen_kappa_score(x, y) if x.nunique() > 1 or y.nunique() > 1 else float("nan")
    out["kappa_3class"] = cohen_kappa_score(d.verdict_q, d.verdict_2)
    return out
rows.append(agree(m, "all"))
for mode, d in m.groupby("run_mode"):
    rows.append(agree(d, mode))
    for fam, dd in d.groupby("attack_type"):
        rows.append(agree(dd, f"{mode}|{fam}"))
    if mode == "safe":
        for k, dd in d.groupby("safe_kind"):
            rows.append(agree(dd, f"safe|{k}"))
R = pd.DataFrame(rows)
R.to_csv("results/v2/tables/second_judge_agreement.csv", index=False)
pd.set_option("display.width", 250)
print(R.round(3).to_string())
print(pd.crosstab(m.verdict_q, m.verdict_2, margins=True))
if a.human:
    h = pd.read_csv(a.human)[["run_mode", "scenario_id", "replay_idx", "human", "stratum", "stratum_population"]]
    hm = h.merge(m[["run_mode", "scenario_id", "replay_idx", "verdict_q", "verdict_2"]], on=["run_mode", "scenario_id", "replay_idx"])
    w = hm.stratum_population / hm.groupby("stratum").stratum.transform("size")
    hu = hm.human.isin(UNSAFE)
    for col, name in [("verdict_q", "Qwen3-14B"), ("verdict_2", "second judge")]:
        j = hm[col].isin(UNSAFE)
        tp, fp, fn = (w * (j & hu)).sum(), (w * (j & ~hu)).sum(), (w * (~j & hu)).sum()
        print(f"{name}: n={len(hm)} kappa vs human {cohen_kappa_score(j, hu):.3f}, weighted precision {tp/(tp+fp):.3f}, recall {tp/(tp+fn):.3f}")
