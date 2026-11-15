import pandas as pd, numpy as np, json, os, glob
from huggingface_hub import hf_hub_download
R="Latavious/prompt-forensics-runs-v2"; W=os.environ.get("WORKDIR","work/")
p=pd.read_parquet('/mnt/project-files/analyse_runs_v2/compact/pools.parquet'); p=p[(p.victim=="Mistral-7B")&p.complete]
rng=np.random.default_rng(20261115)
strata=[("A1 attaque, k=1","attacks",lambda g:g.k==1,40,"unsafe"),
        ("A2 attaque, k=2-5","attacks",lambda g:g.k.between(2,5),30,"unsafe"),
        ("A3 attaque, 1-5 malicious","attacks",lambda g:g.k_strict.between(1,5),30,"malicious"),
        ("L1 legitime, k=1","safe",lambda g:g.k==1,30,"unsafe"),
        ("L2 legitime, k=2-5","safe",lambda g:g.k.between(2,5),20,"unsafe")]
picks=[]; used=set()
for name,mode,f,q,var in strata:
    g=p[(p.run_mode==mode)]; g=g[f(g)]; pop=len(g)
    g=g[~g.scenario_id.isin(used)].sample(frac=1,random_state=int(rng.integers(1e9))).head(q).copy()
    used|=set(g.scenario_id); g["stratum"]=name; g["population"]=pop; g["variant"]=var; picks.append(g)
S=pd.concat(picks,ignore_index=True); print(S.groupby("stratum").agg(n=("k","size"),pop=("population","first")))
keys=set(zip(S.scenario_id,S.run_mode))
# per-replay verdicts for sampled pools
ev=[]
for s in range(4):
    f=hf_hub_download(R,f"shard{s}/evaluator.csv",repo_type="dataset",local_dir=W+"dl")
    for ch in pd.read_csv(f,usecols=["scenario_id","run_mode","replay_idx","target_model","verdict","judge_ok","line_number"],chunksize=500000):
        ch=ch[ch.target_model.astype(str).str.contains("Mistral")]
        ch=ch[[k in keys for k in zip(ch.scenario_id,ch.run_mode)]]
        if len(ch): ev.append(ch.assign(shard=s))
    os.remove(f); print("ev",s,flush=True)
E_=pd.concat(ev).sort_values(["judge_ok","line_number"]).drop_duplicates(["scenario_id","run_mode","replay_idx"],keep="last")
E_.to_parquet(W+"pool_verdicts.parquet")
show=[]
for r in S.itertuples():
    e=E_[(E_.scenario_id==r.scenario_id)&(E_.run_mode==r.run_mode)]
    sel=e[e.verdict.isin(["suspicious","malicious"])] if r.variant=="unsafe" else e[e.verdict=="malicious"]
    for x in sel.sort_values("replay_idx").head(3).itertuples():
        show.append(dict(scenario_id=r.scenario_id,run_mode=r.run_mode,replay_idx=int(x.replay_idx),verdict=x.verdict,shard=x.shard))
Sh=pd.DataFrame(show); print("answers",len(Sh))
want={(a,b,c) for a,b,c in zip(Sh.scenario_id,Sh.run_mode,Sh.replay_idx)}
texts={}
for s in range(4):
    for mode in ["attacks","safe"]:
        f=hf_hub_download(R,f"shard{s}/replays_{mode}_r100.jsonl",repo_type="dataset",local_dir=W+"dl")
        for l in open(f):
            o=json.loads(l); k=(o["scenario_id"],mode,o["replay_idx"])
            if k in want and (o.get("text") or "").strip(): texts[k]=o["text"]
        os.remove(f); print("tx",s,mode,len(texts),flush=True)
Sh["answer"]=[texts.get(k) for k in zip(Sh.scenario_id,Sh.run_mode,Sh.replay_idx)]
sc={}
for f in glob.glob("/home/user/repo/data/scenarios_v2/*/*.json"):
    for o in json.load(open(f)): sc[o["id"]]=o
Sh=Sh.merge(S[["scenario_id","run_mode","stratum","population","variant","k","k_strict","domain","attack_type"]],on=["scenario_id","run_mode"])
Sh["system_prompt"]=[sc[i]["system_prompt"] for i in Sh.scenario_id]
Sh["document"]=[sc[i]["dataset"] for i in Sh.scenario_id]
Sh["prompt"]=[sc[i]["attack_prompt"] if m=="attacks" else sc[i]["safe_prompt"] for i,m in zip(Sh.scenario_id,Sh.run_mode)]
Sh=Sh.sample(frac=1,random_state=7).reset_index(drop=True)
pid={k:f"P{i+1:03d}" for i,k in enumerate(dict.fromkeys(zip(Sh.scenario_id,Sh.run_mode)))}
Sh["pool_id"]=[pid[k] for k in zip(Sh.scenario_id,Sh.run_mode)]
Sh=Sh.sort_values(["pool_id","replay_idx"]).reset_index(drop=True); Sh["item_id"]=[f"R{i+1:03d}" for i in range(len(Sh))]
Sh.to_parquet(W+"pool_sheet_full.parquet")
print("missing texts",Sh.answer.isna().sum()); print("done")
