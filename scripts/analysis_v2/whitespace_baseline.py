"""Whitespace experiment against its baseline: the same prompt, same seeds, from the main run.

Needs work/texts_needed.parquet (written by sample_and_texts.py) and the whitespace_*.jsonl files.
Prints the share of identical texts and identical Evaluator verdicts for
main run vs replayed original, and original vs whitespace-perturbed prompt.
"""
import os, json, glob
import pandas as pd
D = os.environ.get("RUNS_V2_DATA", "results/v2/hf")
W = os.environ.get("RUNS_V2_WORK", "results/v2/work")
T = pd.read_parquet(f"{W}/texts_needed.parquet"); T = T[T.for_ws]
base = {(r.scenario_id, int(r.replay_idx)): r.text for r in T.itertuples()}

def load(mode, kind):
    rows = {}
    for f in sorted(glob.glob(f"{D}/shard*/whitespace_{mode}_{kind}.jsonl")):
        for line in open(f):
            o = json.loads(line)
            if o.get("error") or not (o.get("text") or "").strip():
                continue
            rows[(o["scenario_id"], o["replay_idx"])] = o["text"]
    return rows

ev = pd.concat([pd.read_parquet(f"{W}/eval_s{s}.parquet", columns=["input_file", "scenario_id", "replay_idx", "unsafe", "judge_ok", "line_number"]) for s in range(4)])
ev = ev.sort_values(["judge_ok", "line_number"]).drop_duplicates(["input_file", "scenario_id", "replay_idx"], keep="last")
U = {(r.input_file, r.scenario_id, int(r.replay_idx)): r.unsafe for r in ev.itertuples()}
out = []
for mode in ["attacks", "safe"]:
    o, p = load(mode, "orig"), load(mode, "pert")
    ks = set(o) & set(p) & set(base)
    files = {"base": (base, f"replays_{mode}_r100.jsonl"), "orig": (o, f"whitespace_{mode}_orig.jsonl"), "pert": (p, f"whitespace_{mode}_pert.jsonl")}
    for a, b in [("base", "orig"), ("orig", "pert"), ("base", "pert")]:
        (ta, fa), (tb, fb) = files[a], files[b]
        out.append({"mode": mode, "comparison": f"{a} vs {b}", "replays": len(ks),
                    "identical_text": sum(ta[k] == tb[k] for k in ks) / len(ks),
                    "same_verdict": sum(U.get((fa,) + k) == U.get((fb,) + k) for k in ks) / len(ks)})
    sids = {k[0] for k in ks}
    for name, (_, f) in files.items():
        out.append({"mode": mode, "comparison": f"{name}: pools with >=1 unsafe in 10", "replays": len(ks),
                    "identical_text": None,
                    "same_verdict": sum(any(U.get((f, s, i), False) for i in range(10)) for s in sids) / len(sids)})
R = pd.DataFrame(out)
R.to_csv("results/v2/tables/whitespace_baseline.csv", index=False)
print(R.round(4).to_string())
