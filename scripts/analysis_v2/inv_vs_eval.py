import os
import pandas as pd, numpy as np, pickle
C=os.environ.get("RUNS_V2_WORK", "results/v2/work"); T="results/v2/tables"
allj=pickle.load(open(f"{C}/investigators.pkl","rb"))
ev=pd.read_parquet(f"{C}/eval_main_clean.parquet"); ev=ev[ev.victim=="Mistral-7B"]
pools=pd.read_parquet(f"{C}/pools.parquet"); pools=pools[pools.victim=="Mistral-7B"]
NAMES={"qwen3-14b":"Qwen3-14B","gemma-3-12b-it":"Gemma-3-12B","ministral-8b-instruct-2410":"Ministral-8B","llama-3_1-8b-instruct":"Llama-3.1-8B","phi-4-reasoning":"Phi-4-reasoning"}
rows=[]
for j,(a,s) in allj.items():
    a=a[a.judge_ok==True].copy()
    print(NAMES[j],"replay_idx used:",a.replay_idx.value_counts().head(3).to_dict())
    a["inv_succ"]=a.attack_succeeded.astype(str).str.lower().isin(["true","1"])
    m=a.merge(ev[["run_mode","scenario_id","replay_idx","unsafe","strict"]],on=["run_mode","scenario_id","replay_idx"],how="left")
    m=m.merge(pools[["run_mode","scenario_id","k"]],on=["run_mode","scenario_id"],how="left")
    m=m.dropna(subset=["unsafe"])
    u=m.unsafe.astype(bool); i=m.inv_succ
    agree=(u==i).mean()
    po=agree; pe=u.mean()*i.mean()+(1-u.mean())*(1-i.mean()); kappa=(po-pe)/(1-pe)
    rows.append(dict(judge=NAMES[j],n=len(m),inv_success_rate=i.mean(),eval_unsafe_same_replay=u.mean(),agreement=agree,kappa=kappa,
        inv_succ_given_eval_unsafe=i[u].mean(),inv_succ_given_eval_safe=i[~u].mean(),
        pool_any_unsafe=(m.k>0).mean(),inv_succ_given_pool_any=i[m.k>0].mean(),inv_succ_given_pool_none=i[m.k==0].mean()))
R=pd.DataFrame(rows); R.to_csv(f"{T}/investigator_vs_evaluator.csv",index=False)
pd.set_option("display.width",250); print(R.round(3).to_string())
