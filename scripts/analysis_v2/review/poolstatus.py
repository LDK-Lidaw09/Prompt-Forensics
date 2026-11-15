"""Three statuses per checked pool (confirmed / not confirmed / unresolved) and population overlap of the groups."""
import pandas as pd, sys
key, sheet, out = sys.argv[1], sys.argv[2], sys.argv[3]
k = pd.read_csv(key); x = pd.read_excel(sheet, sheet_name="Fiche")
m = k.merge(x[["item_id", "label"]], on="item_id")
m["conf"] = [(l == "malicious") if v == "malicious" else l in ("suspicious", "malicious") for l, v in zip(m.label, m.variant)]
g = m.groupby(["stratum", "pool_id"]).agg(conf=("conf", "any"), read=("item_id", "size"), k=("k", "first"), ks=("k_strict", "first"), var=("variant", "first")).reset_index()
g["flagged"] = [ks if v == "malicious" else kk for kk, ks, v in zip(g.k, g.ks, g["var"])]
g["status"] = ["confirmed" if c else ("not_confirmed" if f <= r else "unresolved") for c, f, r in zip(g.conf, g.flagged, g.read)]
t = pd.crosstab(g.stratum, g.status, margins=True); print(t); t.to_csv(out)
p = pd.read_parquet("/mnt/project-files/analyse_runs_v2/compact/pools.parquet"); p = p[(p.victim == "Mistral-7B") & p.complete]
a = p[p.run_mode == "attacks"]; A3 = a.k_strict.between(1, 5)
print("A3 overlap: k=1", (A3 & (a.k == 1)).sum(), "k=2-5", (A3 & a.k.between(2, 5)).sum(), "k>=6", (A3 & (a.k >= 6)).sum())
