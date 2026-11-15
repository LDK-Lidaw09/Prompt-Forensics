import pandas as pd, numpy as np, glob, sys
sys.path.insert(0,"/home/user/repo/src")
from prompt_forensics.experiments.analysis import wilson_interval
m=pd.read_parquet("meta.parquet").merge(pd.read_parquet("req.parquet"),on="scenario_id",how="left")
m.loc[m.run_mode=="safe","req"]=None
p=pd.read_parquet('/mnt/project-files/analyse_runs_v2/compact/pools.parquet'); p=p[(p.victim=="Mistral-7B")&p.complete]
p=p.merge(m[["scenario_id","run_mode","doc","context","req","dupgroup","src"]],on=["scenario_id","run_mode"])
def boot(num,den,g,B=2000,seed=0):
    d=pd.DataFrame({"n":num,"d":den,"g":g.values}).groupby("g")[["n","d"]].sum()
    rng=np.random.default_rng(seed); idx=rng.integers(0,len(d),(B,len(d)))
    r=d.n.to_numpy()[idx].sum(1)/d.d.to_numpy()[idx].sum(1); return np.quantile(r,[.025,.975]),len(d)
rows=[]
def add(name,mode,num,den,groups):
    est=num.sum()/den.sum()
    lo,hi=wilson_interval(int(num.sum()),int(den.sum())) if (den==1).all() else (np.nan,np.nan)
    r=dict(measure=name,run_mode=mode,estimate=est,n=int(den.sum()),wilson_lo=lo,wilson_hi=hi)
    pb,_=boot(num,den,pd.Series(range(len(num)))); r.update(prompt_lo=pb[0],prompt_hi=pb[1])
    for gname,g in groups.items():
        (a,b),k=boot(num,den,g); r.update({f"{gname}_lo":a,f"{gname}_hi":b,f"{gname}_clusters":k})
    rows.append(r)
for mode,g in p.groupby("run_mode"):
    groups={"document":g.doc,"context":g.context,"duptext":g.dupgroup}
    if mode=="attacks": groups["source_request"]=g.req
    add("pools >=1 unsafe in 100",mode,(g.k>0).astype(int),pd.Series(1,index=g.index),groups)
    add("pools >=1 malicious in 100",mode,(g.k_strict>0).astype(int),pd.Series(1,index=g.index),groups)
    add("answers unsafe (per replay)",mode,g.k,g.n,groups)
# Investigators
for f,name in [("qwen3-14b","Qwen3-14B"),("gemma-3-12b-it","Gemma-3-12B"),("ministral-8b-instruct-2410","Ministral-8B"),("llama-3_1-8b-instruct","Llama-3.1-8B"),("phi-4-reasoning","Phi-4-reasoning")]:
    d=pd.concat([pd.read_csv(x,usecols=["scenario_id","run_mode","decision","judge_ok"]) for x in glob.glob(f"/home/user/repo/results/v2/hf/shard*/investigator_*_r100_{f}.csv")])
    d=d.sort_values("judge_ok").drop_duplicates(["scenario_id","run_mode"],keep="last"); d=d[d.judge_ok.astype(str).isin(["True","true","1"])]
    d["flag"]=d.decision.astype(str).str.lower().eq("attack").astype(int)
    d=d.merge(m[["scenario_id","run_mode","doc","context","req","dupgroup"]],on=["scenario_id","run_mode"])
    for mode,g in d.groupby("run_mode"):
        groups={"document":g.doc,"context":g.context,"duptext":g.dupgroup}
        if mode=="attacks": groups["source_request"]=g.req
        add(f"{name} {'recall' if mode=='attacks' else 'false-positive rate'}",mode,g.flag,pd.Series(1,index=g.index),groups)
R=pd.DataFrame(rows); R.to_csv("/home/user/repo/results/v2/tables/cluster_intervals.csv",index=False)
pd.set_option("display.width",300); pd.set_option("display.max_columns",30)
c=["measure","run_mode","estimate","wilson_lo","wilson_hi","document_lo","document_hi","context_lo","context_hi","duptext_lo","duptext_hi","source_request_lo","source_request_hi","source_request_clusters"]
print((R[c].set_index(["measure","run_mode"])*100).round(2).to_string())
