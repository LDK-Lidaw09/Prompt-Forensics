import os
import json,glob,pandas as pd,numpy as np
D=os.environ.get("RUNS_V2_DATA", "results/v2/hf")
def load(mode,kind):
    rows={}
    for f in sorted(glob.glob(f"{D}/shard*/whitespace_{mode}_{kind}.jsonl")):
        for l in open(f):
            o=json.loads(l)
            if o.get("error") or not (o.get("text") or "").strip(): continue
            rows[(o["scenario_id"],o["replay_idx"])]=o["text"]
    return rows
res=[]
for mode in ["attacks","safe"]:
    a,b=load(mode,"orig"),load(mode,"pert")
    common=set(a)&set(b)
    same=sum(a[k]==b[k] for k in common)
    sc=set(k[0] for k in common)
    poolsame=sum(all(a[k]==b[k] for k in common if k[0]==s) for s in sc)
    res.append(dict(mode=mode,pairs=len(common),identical=same,frac=same/len(common),pools=len(sc),pools_all_identical=poolsame))
    o=pd.concat([pd.read_csv(f) for f in glob.glob(f"{D}/shard*/whitespace_{mode}_orig.csv")]).sort_values("complete").drop_duplicates("scenario_id",keep="last")
    p=pd.concat([pd.read_csv(f) for f in glob.glob(f"{D}/shard*/whitespace_{mode}_pert.csv")]).sort_values("complete").drop_duplicates("scenario_id",keep="last")
    m=o.merge(p,on="scenario_id",suffixes=("_o","_p"))
    print(mode,len(m),"cons orig %.4f pert %.4f diff mean %.4f sd %.4f"%(m.cons_score_o.mean(),m.cons_score_p.mean(),(m.cons_score_p-m.cons_score_o).mean(),(m.cons_score_p-m.cons_score_o).std()))
print(pd.DataFrame(res))
