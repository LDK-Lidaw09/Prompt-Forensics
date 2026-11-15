import pandas as pd, numpy as np, glob
S="/home/user/repo/results/v2/hf/sensitivity"; A="/tmp/claude-0/-home-user/8c235359-af69-5d63-8d23-b0a5a7435e18/scratchpad/ana2/"
def inv(f):
    d=pd.read_csv(f); d=d.sort_values("judge_ok").drop_duplicates(["scenario_id","run_mode"],keep="last")
    d["ok"]=d.judge_ok.astype(str).isin(["True","true","1"]); d["flag"]=d.decision.astype(str).str.lower().eq("attack")
    return d[["scenario_id","run_mode","ok","flag"]]
wo=pd.concat([inv(f) for f in sorted(glob.glob(S+"/**/investigator_noout_*.csv",recursive=True))])
same=pd.concat([inv(f) for f in sorted(glob.glob(S+"/**/investigator_withanswer_*.csv",recursive=True))])
main=pd.concat([inv(f) for f in sorted(glob.glob("/home/user/repo/results/v2/hf/shard*/investigator_*_r100_qwen3-14b.csv"))])
meta=pd.read_parquet(A+"meta.parquet")[["scenario_id","run_mode","doc"]].merge(pd.read_parquet(A+"req.parquet"),on="scenario_id",how="left")
rows=[]
def boot(diff,g,B=4000,seed=0):
    d=pd.DataFrame({"x":diff.values,"g":g.values}).groupby("g").x.agg(["sum","size"])
    idx=np.random.default_rng(seed).integers(0,len(d),(B,len(d)))
    r=d["sum"].to_numpy()[idx].sum(1)/d["size"].to_numpy()[idx].sum(1); return np.quantile(r,[.025,.975])
for name,w in [("same run",same),("main runs",main)]:
    m=w.merge(wo,on=["scenario_id","run_mode"],suffixes=("_w","_wo")); m=m[m.ok_w&m.ok_wo].merge(meta,on=["scenario_id","run_mode"],how="left")
    for mode,g in m.groupby("run_mode"):
        diff=g.flag_wo.astype(int)-g.flag_w.astype(int)
        r=dict(baseline=name,run_mode=mode,n=len(g),with_=g.flag_w.mean(),without=g.flag_wo.mean(),diff=diff.mean())
        lo,hi=boot(diff,pd.Series(range(len(g)))); r.update(prompt_lo=lo,prompt_hi=hi)
        lo,hi=boot(diff,g.doc); r.update(doc_lo=lo,doc_hi=hi,docs=g.doc.nunique())
        if mode=="attacks":
            lo,hi=boot(diff,g.req.fillna("na")); r.update(src_lo=lo,src_hi=hi,srcs=g.req.nunique())
        rows.append(r)
R=pd.DataFrame(rows); pd.set_option("display.width",250); print((R.set_index(["baseline","run_mode"])).round(4).to_string())
R.to_csv("/home/user/repo/results/v2/tables/sensitivity_ablation_clustered.csv",index=False)
