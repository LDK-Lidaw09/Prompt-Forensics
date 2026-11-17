import json,glob,pandas as pd
rows=[]
for f in glob.glob("/home/user/repo/data/scenarios_v2/*/*.json"):
    for s in json.load(open(f)):
        doc=(s.get("dataset") or "").split("\n")[0].strip()
        if s.get("attack_prompt"):
            rows.append(dict(scenario_id=s["id"],run_mode="attacks",doc=doc,context=s.get("context_id"),src=s.get("benchmark_source"),bid=s.get("benchmark_id"),text=s["attack_prompt"],split=s.get("split")))
        if s.get("safe_prompt"):
            rows.append(dict(scenario_id=s["id"],run_mode="safe",doc=doc,context=s.get("context_id"),src=None,bid=None,text=s["safe_prompt"],split=s.get("split"),safe_kind=s.get("safe_kind")))
m=pd.DataFrame(rows).drop_duplicates(["scenario_id","run_mode"])
m["dupgroup"]=m.groupby(["run_mode","text"]).ngroup()
m.to_parquet("meta.parquet",index=False)
print(m.groupby("run_mode").agg(n=("scenario_id","size"),docs=("doc","nunique"),ctx=("context","nunique"),bids=("bid","nunique"),dups=("dupgroup","nunique")))
a=m[m.run_mode=="attacks"]
dev=set(a[a.split=="dev"].bid); t=a[a.split=="test"]
print("test attack prompts whose source item also in dev:",t.bid.isin(dev).mean())
dt=set(a[a.split=="dev"].text); print("test attack texts identical to a dev text:",t.text.isin(dt).mean())
# template: text after removing request? skip
