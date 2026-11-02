"""Draw the 20,000 replays re-judged by a second Evaluator and write them as a replay JSONL.

10,000 attack and 10,000 safe replays of the first victim (Mistral-7B, 100-replay
pools), stratified by attack family x domain (x safe-prompt kind for safe prompts),
plus the 193 replays of the human validation sheet. Seed 20261102.
Output: data/second_judge/sample_20k.jsonl.gz (same fields as replays_*.jsonl).
"""
import os, json, gzip
import numpy as np, pandas as pd
from huggingface_hub import hf_hub_download

REPO = "Latavious/prompt-forensics-runs-v2"
D = os.environ.get("RUNS_V2_DATA", "results/v2/hf")
W = os.environ.get("RUNS_V2_WORK", "results/v2/work")
OUT = "data/second_judge/sample_20k.jsonl.gz"
ev = pd.read_parquet(f"{W}/eval_main_clean.parquet",
                     columns=["shard", "input_file", "scenario_id", "replay_idx", "run_mode", "victim",
                              "attack_type", "domain", "safe_kind"])
ev = ev[ev.victim == "Mistral-7B"]
val = pd.read_parquet(f"{W}/validation_sample_keys.parquet", columns=["shard", "input_file", "scenario_id", "replay_idx"])
val["in_validation"] = True
picks = []
for mode, per in [("attacks", 10000), ("safe", 10000)]:
    g = ev[ev.run_mode == mode].copy()
    g["stratum"] = g.attack_type + "|" + g.domain + ("|" + g.safe_kind.fillna("") if mode == "safe" else "")
    share = g.stratum.value_counts(normalize=True)
    quota = (share * per).round().astype(int)
    for s, q in quota.items():
        picks.append(g[g.stratum == s].sample(n=q, random_state=20261102))
S = pd.concat(picks, ignore_index=True)[["shard", "input_file", "scenario_id", "replay_idx"]]
S = pd.concat([S, val[["shard", "input_file", "scenario_id", "replay_idx"]]]).drop_duplicates()
S = S.merge(val, how="left", on=["shard", "input_file", "scenario_id", "replay_idx"]).fillna({"in_validation": False})
print("sample size", len(S), "validation items", int(S.in_validation.sum()))
want = {}
for r in S.itertuples():
    want.setdefault((r.shard, r.input_file), set()).add((r.scenario_id, int(r.replay_idx)))
rows = {}
for (shard, inp), keys in sorted(want.items()):
    p = hf_hub_download(REPO, f"shard{shard}/{inp}", repo_type="dataset", local_dir=D)
    with open(p) as fh:
        for line in fh:
            o = json.loads(line)
            if o.get("error") or not (o.get("text") or "").strip():
                continue
            k = (o["scenario_id"], o["replay_idx"])
            if k in keys:
                rows[(shard, inp) + k] = line  # last occurrence wins, as in the Evaluator run
    os.remove(p)
    print("done", shard, inp, len(rows), flush=True)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with gzip.open(OUT, "wt") as f:
    for line in rows.values():
        f.write(line if line.endswith("\n") else line + "\n")
S.to_csv("data/second_judge/sample_20k_keys.csv", index=False)
print("written", len(rows))
