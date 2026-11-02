import os
import pandas as pd, numpy as np, glob
D=os.environ.get("RUNS_V2_DATA", "results/v2/hf"); T="results/v2/tables"
plan=pd.read_csv(f"{D}/plan/plan.csv")
def load(pat):
    d=pd.concat([pd.read_csv(f).assign(shard=int(f.split("shard")[1][0])) for f in sorted(glob.glob(f"{D}/shard*/{pat}"))])
    d=d.sort_values(["complete","num_replays"]).drop_duplicates(["scenario_id","source_file","run_mode"],keep="last")
    return d.merge(plan[["scenario_id","run_mode","split","safe_kind"]],on=["scenario_id","run_mode"],how="left")
out=[]
for victim,suf in [("Mistral-7B","r100.csv"),("Llama-3.1-8B","r10_llama-3_1-8b-instruct.csv")]:
    for mode in ["attacks","safe"]:
        d=load(f"replays_{mode}_{suf}")
        d.to_parquet(f"results/v2/work/replays_{victim}_{mode}.parquet")
        def summ(g,label):
            r=dict(victim=victim,mode=mode,group=label,n=len(g),complete=int(g.complete.sum()),replays=int(g.num_replays.sum()))
            for m in ["cons_score","fid_score","cons_bert_score","fid_bert_score","cons_min","fid_min"]:
                v=g[m].dropna(); r[m+"_mean"]=v.mean(); r[m+"_sd"]=v.std(ddof=0)
            for m in ["cons_score","fid_score","cons_bert_score","fid_bert_score"]:
                r[m+"_frac_ge0.9"]=(g[m]>=0.9).mean()
            r["cons_pairs_ge0.9"]=g.cons_count_above.sum()/g.cons_num_pairs.sum()
            return r
        out.append(summ(d,"all"))
        for k,g in d.groupby("attack_type"): out.append(summ(g,k))
        if mode=="safe":
            for k,g in d.groupby("safe_kind"): out.append(summ(g,k))
        for k,g in d.groupby("domain"): out.append(summ(g,"domain="+k))
R=pd.DataFrame(out); R.to_csv(f"{T}/replay_stability.csv",index=False)
pd.set_option("display.width",250)
print(R[["victim","mode","group","n","complete","replays","cons_score_mean","cons_score_sd","fid_score_mean","cons_bert_score_mean","cons_score_frac_ge0.9","cons_bert_score_frac_ge0.9","cons_pairs_ge0.9","cons_min_mean"]].round(3).to_string())
