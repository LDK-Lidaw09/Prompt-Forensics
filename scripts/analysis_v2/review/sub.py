import pandas as pd, numpy as np
p=pd.read_parquet('/mnt/project-files/analyse_runs_v2/compact/pools.parquet'); p=p[(p.victim=="Mistral-7B")&p.complete]
m=pd.read_parquet("meta.parquet")[["scenario_id","run_mode","doc","src"]]
q=p.merge(m,on=["scenario_id","run_mode"])
def boot(y,g,B=2000,seed=0):
    d=pd.DataFrame({"n":y.values,"g":g.values}).groupby("g").n.agg(["sum","size"])
    idx=np.random.default_rng(seed).integers(0,len(d),(B,len(d)))
    r=d["sum"].to_numpy()[idx].sum(1)/d["size"].to_numpy()[idx].sum(1); return np.quantile(r,[.025,.975])
rows=[]
for mode in ["attacks","safe"]:
    g0=q[q.run_mode==mode]
    for col in ["domain","src","attack_type"]+(["safe_kind"] if mode=="safe" else []):
        for v,g in g0.groupby(col):
            y=(g.k>0).astype(int); lo,hi=boot(y,g.doc)
            rows.append(dict(run_mode=mode,by=col,value=v,n=len(g),rate=y.mean(),doc_lo=lo,doc_hi=hi,docs=g.doc.nunique()))
R=pd.DataFrame(rows); R.to_csv("/home/user/repo/results/v2/tables/cluster_intervals_subgroups.csv",index=False)
print((R.assign(rate=R.rate*100,doc_lo=R.doc_lo*100,doc_hi=R.doc_hi*100)).round(1).to_string())
