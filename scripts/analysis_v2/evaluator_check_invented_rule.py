"""Manual check of the Evaluator (Section 6.3.5) under two rules for invented data.

rule_as_annotated: unsafe = human label suspicious or malicious (labels as recorded).
rule_without_invented: unsafe only if the human label is unsafe AND the answer has no invented data
  (lower bound: an answer with invented data and other harmful content is counted safe).
Weights: stratum population / items drawn in the stratum; 95% interval: stratified bootstrap over items.
Usage: python evaluator_check_invented_rule.py validation_evaluator_resultats_Lys.csv <output csv>
"""
import sys
import numpy as np, pandas as pd

d = pd.read_csv(sys.argv[1])
d['ev'] = d.verdict != 'safe'
d['rule_as_annotated'] = d.human != 'safe'
d['rule_without_invented'] = d.rule_as_annotated & (d.invented == 'non')
d['w'] = d.stratum_population / d.groupby(['run_mode', 'stratum']).item_id.transform('size')

def measures(x, c):
    hu = x[c]; w = x.w
    share = (w * hu).sum() / w.sum()
    rec = (w * (hu & x.ev)).sum() / (w * hu).sum() if hu.any() else np.nan
    prec = (w * (hu & x.ev)).sum() / (w * x.ev).sum()
    return share, rec, prec

rng = np.random.default_rng(20261117)
out = []
print(pd.crosstab([d.run_mode, d.human], d.invented))
for m, g in d.groupby('run_mode'):
    for c in ['rule_as_annotated', 'rule_without_invented']:
        est = measures(g, c)
        boot = []
        for _ in range(4000):
            x = pd.concat([s.sample(len(s), replace=True, random_state=int(rng.integers(1e9))) for _, s in g.groupby('stratum')])
            boot.append(measures(x, c))
        lo, hi = np.nanpercentile(np.array(boot, dtype=float), [2.5, 97.5], axis=0)
        for k, name in enumerate(['share_unsafe_annotation', 'evaluator_recall', 'evaluator_precision']):
            out.append(dict(run_mode=m, rule=c, measure=name, value=est[k], ci_low=lo[k], ci_high=hi[k], items_unsafe=int(g[c].sum())))
o = pd.DataFrame(out); o.to_csv(sys.argv[2], index=False); print(o.round(3).to_string())
