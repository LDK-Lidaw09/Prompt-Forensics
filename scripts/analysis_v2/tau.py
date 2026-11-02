import os
import pandas as pd, numpy as np, glob
C=os.environ.get("RUNS_V2_WORK", "results/v2/work"); T="results/v2/tables"; D=os.environ.get("RUNS_V2_DATA", "results/v2/hf")
plan=pd.read_csv(f"{D}/plan/plan.csv")
fs=sorted(glob.glob(f"{C}/pairs_s*.parquet"))
P=pd.concat([pd.read_parquet(f).assign(victim="Llama-3.1-8B" if "llama" in f else "Mistral-7B") for f in fs],ignore_index=True)
P=P.merge(plan[["scenario_id","run_mode","safe_kind","split"]],on=["scenario_id","run_mode"],how="left")
print(P.groupby(["victim","run_mode"]).agg(pools=("n_idx","size"),full=("n_idx",lambda x:(x==x.max()).mean())))
taus=[c for c in P.columns if c.startswith("cos_ge_")]
rows=[]
for (v,m),g in P.groupby(["victim","run_mode"]):
    for c in taus:
        t=float(c[-4:])
        rows.append(dict(victim=v,mode=m,tau=t,frac_pairs_ge=g[c].mean(),frac_pools_mean_ge=(g.cos_mean>=t).mean(),frac_pools_min_ge=(g.cos_min>=t).mean(),
                         bert_frac_pairs_ge=g[f"bert_ge_{t:.2f}"].mean() if f"bert_ge_{t:.2f}" in g else np.nan,frac_pools_bertmean_ge=(g.bert_mean>=t).mean()))
S=pd.DataFrame(rows); S.to_csv(f"{T}/tau_sweep.csv",index=False)
pd.set_option("display.width",250)
print(S[S.tau.isin([0.8,0.85,0.88,0.9,0.92,0.95,0.98])].round(3).to_string())
# dev-set choice of tau: maximise separation (Youden) between attack and safe pools on pool mean cosine, dev split only, Mistral
mi=P[(P.victim=="Mistral-7B")]
dev=mi[mi.split=="dev"]; test=mi[mi.split=="test"]
best=None
for t in np.arange(0.80,0.99,0.005):
    tpr=(dev[dev.run_mode=="safe"].cos_mean>=t).mean(); fpr=(dev[dev.run_mode=="attacks"].cos_mean>=t).mean()
    if best is None or tpr-fpr>best[1]: best=(t,tpr-fpr)
t=best[0]
print("dev Youden tau (safe vs attack by pool mean cosine)",round(t,3),round(best[1],3),"test: safe>=t",(test[test.run_mode=="safe"].cos_mean>=t).mean(),"attack>=t",(test[test.run_mode=="attacks"].cos_mean>=t).mean())
from sklearn.metrics import roc_auc_score
y=(mi.run_mode=="safe").astype(int); print("AUC pool mean cosine safe vs attacks (Mistral)",roc_auc_score(y,mi.cos_mean))
# similarity by verdict mix
V=mi[mi.run_mode=="attacks"]
print("Mistral attacks: mean cos both-safe %.4f both-unsafe %.4f mixed %.4f"%(V.cos_mean_both_safe.mean(),V.cos_mean_both_unsafe.mean(),V.cos_mean_mixed.mean()))
mix=V[V.n_pairs_mixed>0]
print("pools with mixed verdicts",len(mix),"/",len(V),"; within those: both-safe %.4f both-unsafe %.4f mixed %.4f"%(mix.cos_mean_both_safe.mean(),mix.cos_mean_both_unsafe.mean(),mix.cos_mean_mixed.mean()))
V["any_unsafe"]=V.n_unsafe>0
print(V.groupby("any_unsafe")[["cos_mean","cos_min","cos_ge_0.90"]].mean())
P.to_parquet(f"{C}/pairs_all.parquet",index=False)
