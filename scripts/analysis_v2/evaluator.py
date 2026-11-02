import os
import pandas as pd, numpy as np, glob, math, sys, pickle
sys.path.insert(0,"src")
from prompt_forensics.experiments.analysis import wilson_interval
C=os.environ.get("RUNS_V2_WORK", "results/v2/work"); T="results/v2/tables"; D=os.environ.get("RUNS_V2_DATA", "results/v2/hf")
plan=pd.read_csv(f"{D}/plan/plan.csv")
shards=[int(x) for x in sys.argv[1:]] or [0,1,2,3]
ev=pd.concat([pd.read_parquet(f"{C}/eval_s{s}.parquet") for s in shards],ignore_index=True)
raw=len(ev)
ev=ev.sort_values(["judge_ok","line_number"]).drop_duplicates(["shard","input_file","scenario_id","replay_idx"],keep="last")
print("raw rows",raw,"after dedup",len(ev),"judge failures",int((~ev.judge_ok).sum()),"unknown",int((ev.verdict=="unknown").sum()))
ev["victim"]=np.where(ev.input_file.str.contains("llama"),"Llama-3.1-8B","Mistral-7B")
ev["exp"]=np.where(ev.input_file.str.startswith("whitespace"),ev.input_file.str.replace(".jsonl",""),"main")
ev["classified"]=ev.judge_ok&(ev.verdict!="unknown")
ev=ev.merge(plan[["scenario_id","run_mode","split","safe_kind","attack_type","domain"]],on=["scenario_id","run_mode"],how="left")
main=ev[(ev.exp=="main")&ev.classified].copy()
main["strict"]=main.verdict.eq("malicious")
main.to_parquet(f"{C}/eval_main_clean.parquet",index=False)
pools=main.groupby(["victim","run_mode","scenario_id"]).agg(n=("unsafe","size"),k=("unsafe","sum"),k_strict=("strict","sum"),
     attack_type=("attack_type","first"),domain=("domain","first"),split=("split","first"),safe_kind=("safe_kind","first")).reset_index()
pools["N"]=np.where(pools.victim=="Mistral-7B",100,10)
pools["complete"]=pools.n==pools.N
pools.to_parquet(f"{C}/pools.parquet",index=False)
print(pools.groupby(["victim","run_mode"]).agg(pools=("n","size"),complete=("complete","sum")))
def ci(k,n):
    lo,hi=wilson_interval(int(k),int(n)); return (k/n if n else math.nan),lo,hi
rows=[]
def summ(g,**lab):
    g=g[g.complete]; n=len(g)
    r=dict(**lab,pools=n)
    for nm,col in [("unsafe","k"),("malicious","k_strict")]:
        k=int((g[col]>0).sum()); p,lo,hi=ci(k,n)
        r[f"tail_{nm}"]=p; r[f"tail_{nm}_lo"]=lo; r[f"tail_{nm}_hi"]=hi
        tot=g.N.sum(); kk=g[col].sum(); p,lo,hi=ci(kk,tot)
        r[f"rate_{nm}"]=p
        r[f"pools_all_{nm}"]=int((g[col]==g.N).sum())
        r[f"median_k_{nm}_given_any"]=g.loc[g[col]>0,col].median() if k else np.nan
    return r
for (v,m),g in pools.groupby(["victim","run_mode"]):
    rows.append(summ(g,victim=v,mode=m,group="all"))
    for k,gg in g.groupby("attack_type"): rows.append(summ(gg,victim=v,mode=m,group=k))
    for k,gg in g.groupby("domain"): rows.append(summ(gg,victim=v,mode=m,group="domain="+k))
    for k,gg in g.groupby("split"): rows.append(summ(gg,victim=v,mode=m,group="split="+k))
    if m=="safe":
        for k,gg in g.groupby("safe_kind"): rows.append(summ(gg,victim=v,mode=m,group=k))
        for (a,k),gg in g.groupby(["attack_type","safe_kind"]): rows.append(summ(gg,victim=v,mode=m,group=f"{a}|{k}"))
R=pd.DataFrame(rows); R.to_csv(f"{T}/evaluator_tail_risk.csv",index=False)
pd.set_option("display.width",250); pd.set_option("display.max_rows",200)
print(R[["victim","mode","group","pools","tail_unsafe","tail_unsafe_lo","tail_unsafe_hi","rate_unsafe","tail_malicious","rate_malicious","pools_all_unsafe","median_k_unsafe_given_any"]].round(4).to_string())
# Mistral, first 10 replays, both victims comparable (same seeds)
m10=main[main.replay_idx<10]
p10=m10.groupby(["victim","run_mode","scenario_id"]).agg(n=("unsafe","size"),k=("unsafe","sum"),ks=("strict","sum")).reset_index()
p10=p10[p10.n==10]
cmp=p10.groupby(["victim","run_mode"]).agg(pools=("k","size"),tail10=("k",lambda x:(x>0).mean()),rate=("k",lambda x:x.sum()/(10*len(x))),tail10_strict=("ks",lambda x:(x>0).mean())).reset_index()
cmp.to_csv(f"{T}/victims_first10.csv",index=False); print(cmp.round(4))
# tail-risk curve m=1..100 Mistral
mm=main[main.victim=="Mistral-7B"]
comp=pools[(pools.victim=="Mistral-7B")&pools.complete][["run_mode","scenario_id"]]
mm=mm.merge(comp,on=["run_mode","scenario_id"])
first=mm[mm.unsafe].groupby(["run_mode","scenario_id"]).replay_idx.min()
firsts=mm[mm.strict].groupby(["run_mode","scenario_id"]).replay_idx.min()
curve=[]
for mode in ["attacks","safe"]:
    n=int((comp.run_mode==mode).sum()); f=first.loc[mode]; fs=firsts.loc[mode] if mode in firsts.index.get_level_values(0) else pd.Series(dtype=float)
    for m in range(1,101):
        k=int((f<m).sum()); p,lo,hi=ci(k,n); ks=int((fs<m).sum())
        curve.append(dict(mode=mode,m=m,pools=n,tail=p,lo=lo,hi=hi,tail_strict=ks/n))
CU=pd.DataFrame(curve); CU.to_csv(f"{T}/tail_risk_curve.csv",index=False)
print(CU[CU.m.isin([1,2,5,10,20,50,100])].round(4).to_string())
# distribution of k among attack pools
for mode in ["attacks","safe"]:
    g=pools[(pools.victim=="Mistral-7B")&(pools.run_mode==mode)&pools.complete]
    print(mode, pd.cut(g.k,[-1,0,1,5,10,25,50,75,99,100]).value_counts(sort=False).to_dict())
# Paired McNemar, same scenario id, Mistral 100
a=pools[(pools.victim=="Mistral-7B")&(pools.run_mode=="attacks")&pools.complete].set_index("scenario_id").k>0
s=pools[(pools.victim=="Mistral-7B")&(pools.run_mode=="safe")&pools.complete].set_index("scenario_id").k>0
cm=a.index.intersection(s.index); b=int((a[cm]&~s[cm]).sum()); c=int((~a[cm]&s[cm]).sum())
print("paired",len(cm),"attack tail",a[cm].mean(),"safe tail",s[cm].mean(),"b",b,"c",c,"chi2",(abs(b-c)-1)**2/(b+c))
