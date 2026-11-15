import json, os, numpy as np, pandas as pd
from huggingface_hub import hf_hub_download
R="Latavious/prompt-forensics-runs-v2"; rows=[]; check=[]
for s in range(4):
    for mode in ("attacks","safe"):
        csv=pd.read_csv(hf_hub_download(R,f"shard{s}/replays_{mode}_r100.csv",repo_type="dataset",local_dir="dl"))
        csv=csv.drop_duplicates(["scenario_id","run_mode"],keep="last").set_index("scenario_id")
        p=hf_hub_download(R,f"shard{s}/replays_{mode}_r100_pairs.jsonl",repo_type="dataset",local_dir="dl")
        seen=set()
        for line in open(p):
            o=json.loads(line); sid=o["scenario_id"]
            if sid in seen: continue
            idx=o["indices"]; n=len(idx)
            if n<100: continue
            for key in ("cosine","bertscore_f1","bert_f1","bertscore"):
                if key in o: pass
            c=np.array(o["cosine"]); 
            if len(c)!=n*(n-1)//2: continue
            seen.add(sid)
            M=np.zeros((n,n)); iu=np.triu_indices(n,1); M[iu]=c; M=M+M.T
            F=M.sum(1)/(n-1)
            rows.append((sid,mode,F[0],np.median(F),F.min(),F.max(),(F<F[0]).mean(),F.std()))
            if sid in csv.index and len(check)<50: check.append((F[0],csv.loc[sid].get("fid_score",np.nan) if not isinstance(csv.loc[sid],pd.DataFrame) else np.nan))
        os.remove(p); print(s,mode,len(rows),flush=True)
print("keys in last obj:",list(o.keys()))
print("check F0 vs fid_score:",check[:5])
D=pd.DataFrame(rows,columns=["scenario_id","run_mode","F0","F_median","F_min","F_max","F0_percentile","F_sd"])
D.to_parquet("fidelity_refs.parquet",index=False); print("done",len(D))
