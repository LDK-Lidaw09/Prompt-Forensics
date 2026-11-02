"""Download the big files of one shard, reduce them to compact tables, delete them."""
import sys, os, json, math, itertools
import numpy as np, pandas as pd
from huggingface_hub import hf_hub_download
REPO="Latavious/prompt-forensics-runs-v2"; DATA=os.environ.get("RUNS_V2_DATA", "results/v2/hf"); OUT=os.environ.get("RUNS_V2_WORK", "results/v2/work")
os.makedirs(OUT,exist_ok=True)
TAUS=[round(0.80+0.01*i,2) for i in range(19)]
def dl(path):
    return hf_hub_download(REPO,path,repo_type="dataset",local_dir=DATA)
def evaluator(s):
    out=f"{OUT}/eval_s{s}.parquet"
    if os.path.exists(out): return pd.read_parquet(out)
    p=dl(f"shard{s}/evaluator.csv")
    cols=["input_file","line_number","scenario_id","source_file","run_mode","replay_idx","target_model","verdict","unsafe","malicious","score","risk_types","judge_ok","output_chars","output_hash"]
    parts=[]
    for ch in pd.read_csv(p,usecols=cols,chunksize=200000,dtype={"verdict":str,"risk_types":str,"unsafe":str,"malicious":str,"judge_ok":str}):
        parts.append(ch)
    d=pd.concat(parts,ignore_index=True)
    raw=len(d)
    for c in ["unsafe","malicious","judge_ok"]:
        d[c]=d[c].astype(str).str.strip().str.lower().isin({"true","1"})
    d["shard"]=s; d["raw_rows"]=raw
    d.to_parquet(out,index=False)
    os.remove(p); return d
def dedup_eval(d):
    d=d.sort_values(["judge_ok","line_number"])
    return d.drop_duplicates(["input_file","scenario_id","replay_idx"],keep="last")
def pairs(s,fname,ev):
    out=f"{OUT}/pairs_s{s}_{fname.replace('.jsonl','')}.parquet"
    if os.path.exists(out): return
    p=dl(f"shard{s}/{fname}")
    inp=fname.replace("_pairs.jsonl",".jsonl")
    e=ev[ev.input_file==inp]
    um={(r.scenario_id,int(r.replay_idx)):(bool(r.unsafe) if r.judge_ok else None) for r in e.itertuples()}
    last={}
    with open(p) as f:
        for line in f:
            o=json.loads(line)
            if not o["cosine"]: continue
            last[(o["scenario_id"],o["source_file"])]=line
    rows=[]
    for (sid,sf),line in last.items():
        o=json.loads(line); idx=o["indices"]; cos=np.array(o["cosine"]); bf=np.array(o["bert_f1"],dtype=float) if o["bert_f1"] else None
        pr=list(itertools.combinations(range(len(idx)),2))
        assert len(pr)==len(cos),(sid,len(pr),len(cos))
        r={"scenario_id":sid,"source_file":sf,"run_mode":o["run_mode"],"target_model":o["target_model"],"n_idx":len(idx),"n_pairs":len(cos),
           "cos_mean":cos.mean(),"cos_min":cos.min(),"cos_p05":np.percentile(cos,5),"cos_median":np.median(cos)}
        for t in TAUS: r[f"cos_ge_{t:.2f}"]=(cos>=t).mean()
        if bf is not None and len(bf)==len(cos):
            r["bert_mean"]=bf.mean(); r["bert_min"]=bf.min()
            for t in TAUS: r[f"bert_ge_{t:.2f}"]=(bf>=t).mean()
        u=np.array([um.get((sid,int(i))) for i in idx],dtype=object)
        known=np.array([x is not None for x in u])
        if known.all():
            ub=u.astype(bool); a=np.array([p_[0] for p_ in pr]); b=np.array([p_[1] for p_ in pr])
            same_s=(~ub[a])&(~ub[b]); same_u=ub[a]&ub[b]; diff=ub[a]!=ub[b]
            r["n_unsafe"]=int(ub.sum())
            r["cos_mean_both_safe"]=cos[same_s].mean() if same_s.any() else np.nan
            r["cos_mean_both_unsafe"]=cos[same_u].mean() if same_u.any() else np.nan
            r["cos_mean_mixed"]=cos[diff].mean() if diff.any() else np.nan
            r["n_pairs_mixed"]=int(diff.sum())
        rows.append(r)
    pd.DataFrame(rows).to_parquet(out,index=False)
    os.remove(p)
if __name__=="__main__":
    for s in map(int,sys.argv[1:]):
        ev=dedup_eval(evaluator(s)); print("eval done",s,len(ev),flush=True)
        for f in ["replays_attacks_r100_pairs.jsonl","replays_safe_r100_pairs.jsonl","replays_attacks_r10_llama-3_1-8b-instruct_pairs.jsonl","replays_safe_r10_llama-3_1-8b-instruct_pairs.jsonl"]:
            pairs(s,f,ev); print("pairs done",s,f,flush=True)
