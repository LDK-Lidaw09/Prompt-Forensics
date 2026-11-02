import pandas as pd, numpy as np, json, os, glob
from huggingface_hub import hf_hub_download
C=os.environ.get("RUNS_V2_WORK", "results/v2/work"); D=os.environ.get("RUNS_V2_DATA", "results/v2/hf"); REPO="Latavious/prompt-forensics-runs-v2"
ev=pd.read_parquet(f"{C}/eval_main_clean.parquet"); ev=ev[ev.victim=="Mistral-7B"]
rng=np.random.default_rng(20261102)
# one replay per pool: shuffle then take first per pool within each stratum
quota={("attacks","safe",None):40,("attacks","suspicious",None):40,("attacks","malicious",None):40,
       ("safe","safe","standard"):15,("safe","safe","hard_negative"):15,
       ("safe","suspicious","standard"):20,("safe","suspicious","hard_negative"):20,("safe","malicious",None):10}
picks=[]
for (mode,verd,kind),q in quota.items():
    g=ev[(ev.run_mode==mode)&(ev.verdict==verd)]
    if kind: g=g[g.safe_kind==kind]
    pop=len(g)
    g=g.sample(frac=1,random_state=int(rng.integers(1e9))).drop_duplicates(["scenario_id"])
    s=g.head(q).copy(); s["stratum"]=f"{mode}|{verd}|{kind or 'all'}"; s["stratum_population"]=pop
    picks.append(s)
S=pd.concat(picks,ignore_index=True)
print(S.groupby("stratum").agg(n=("scenario_id","size"),pop=("stratum_population","first")))
S.to_parquet(f"{C}/validation_sample_keys.parquet",index=False)
want={(r.shard,r.input_file):set() for r in S.itertuples()}
for r in S.itertuples(): want[(r.shard,r.input_file)].add((r.scenario_id,int(r.replay_idx)))
# whitespace baseline: same scenarios, replays 0..9 of the r100 run
ws=set()
for f in glob.glob(f"{D}/shard*/whitespace_*_orig.jsonl"):
    s=int(f.split("shard")[1][0]); mode="attacks" if "attacks" in f else "safe"
    for l in open(f):
        o=json.loads(l); ws.add((s,f"replays_{mode}_r100.jsonl",o["scenario_id"],o["replay_idx"]))
texts=[]
for s in range(4):
    for inp in ["replays_attacks_r100.jsonl","replays_safe_r100.jsonl"]:
        w=want.get((s,inp),set()); wsk={(a,b) for (ss,ii,a,b) in ws if ss==s and ii==inp}
        p=hf_hub_download(REPO,f"shard{s}/{inp}",repo_type="dataset",local_dir=D)
        with open(p) as fh:
            for l in fh:
                o=json.loads(l)
                if o.get("error") or not (o.get("text") or "").strip(): continue
                k=(o["scenario_id"],o["replay_idx"])
                if k in w or k in wsk:
                    texts.append(dict(shard=s,input_file=inp,scenario_id=k[0],replay_idx=k[1],seed=o["seed"],text=o["text"],for_sample=k in w,for_ws=k in wsk))
        os.remove(p); print("done",s,inp,len(texts),flush=True)
T=pd.DataFrame(texts).drop_duplicates(["shard","input_file","scenario_id","replay_idx"],keep="last")
T.to_parquet(f"{C}/texts_needed.parquet",index=False)
print("texts",len(T))
