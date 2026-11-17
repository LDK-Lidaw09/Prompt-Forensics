"""Recompute every number of the pool-level check (Section 6.3.5) from the archived labels.

Inputs (repository): results/v2/tables/pool_check_key.csv (sample drawn by poolsheet.py, seed 20261115)
and results/v2/tables/pool_check_labels.csv (the author's labels, one row per answer read).
Output: results/v2/tables/pool_check_status.csv and the figures printed below.
"""
import math, sys
import pandas as pd

T = "results/v2/tables/"
key = sys.argv[1] if len(sys.argv) > 1 else T + "pool_check_key.csv"
lab = sys.argv[2] if len(sys.argv) > 2 else T + "pool_check_labels.csv"

def wilson(x, n, z=1.96):
    p = x / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h

m = pd.read_csv(key).merge(pd.read_csv(lab)[["item_id", "label"]], on="item_id", validate="one_to_one")
m["unsafe"] = m.label.isin(["suspicious", "malicious"])
m["conf"] = [(l == "malicious") if v == "malicious" else u for l, v, u in zip(m.label, m.variant, m.unsafe)]
g = m.groupby(["stratum", "pool_id"]).agg(conf=("conf", "any"), read=("item_id", "size"), k=("k", "first"),
                                          ks=("k_strict", "first"), var=("variant", "first"),
                                          population=("population", "first")).reset_index()
g["flagged"] = [ks if v == "malicious" else kk for kk, ks, v in zip(g.k, g.ks, g["var"])]
g["status"] = ["confirmed" if c else ("not_confirmed" if f <= r else "unresolved")
               for c, f, r in zip(g.conf, g.flagged, g.read)]
rows = []
for s, d in g.groupby("stratum"):
    x, n = int(d.conf.sum()), len(d); lo, hi = wilson(x, n)
    st = d.status.value_counts()
    rows.append(dict(stratum=s, population=int(d.population.iloc[0]), pools=n, answers_read=int(d.read.sum()),
                     confirmed=x, not_confirmed=int(st.get("not_confirmed", 0)), unresolved=int(st.get("unresolved", 0)),
                     share=x / n, wilson_lo=lo, wilson_hi=hi))
out = pd.DataFrame(rows); out.to_csv(T + "pool_check_status.csv", index=False)
pd.set_option("display.width", 200); print(out.round(3).to_string(index=False))
for mode, verdict in [("attacks", "suspicious"), ("attacks", "malicious"), ("safe", "suspicious")]:
    d = m[(m.run_mode == mode) & (m.verdict == verdict)]
    print(f"{mode} answers labelled {verdict} by the Evaluator: {int(d.unsafe.sum())}/{len(d)} judged unsafe")
