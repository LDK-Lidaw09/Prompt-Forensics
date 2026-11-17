"""Cut answers and verdicts on the temperature-0.2 replays of the sensitivity run.

Inputs (Hugging Face dataset prompt-forensics-runs-v2, folder sensitivity/):
  replays_attacks_t02.jsonl, replays_safe_t02.jsonl  replays at temperature 0.2, 20 per prompt
  evaluator.csv                                       Evaluator verdicts of the sensitivity runs
Outputs: truncation_t02_by_cut.csv, truncation_t02_pools.csv, truncation_t02_within_pools.csv

Usage: python truncation_t02.py <folder with the inputs> <output folder>
"""
import json, sys
import numpy as np, pandas as pd
from tokenizers import Tokenizer

src, out = sys.argv[1], sys.argv[2]
rows = []
for m in ['attacks', 'safe']:
    for line in open(f'{src}/replays_{m}_t02.jsonl'):
        j = json.loads(line)
        rows.append(dict(mode=m, sid=j['scenario_id'], idx=j['replay_idx'], fin=j['finish_reason'], text=j['text']))
r = pd.DataFrame(rows)
ev = pd.read_csv(f'{src}/evaluator.csv', usecols=['input_file', 'scenario_id', 'replay_idx', 'verdict'])
ev = ev[ev.input_file.isin(['replays_attacks_t02.jsonl', 'replays_safe_t02.jsonl'])]
ev['mode'] = np.where(ev.input_file.str.contains('attacks'), 'attacks', 'safe')
d = r.merge(ev[['mode', 'scenario_id', 'replay_idx', 'verdict']].rename(columns={'scenario_id': 'sid', 'replay_idx': 'idx'}),
            on=['mode', 'sid', 'idx'], how='left')
missing = d.verdict.isna().sum()
if missing or len(d) != len(r):
    raise SystemExit(f'{missing} replays without a verdict or duplicated verdicts ({len(d)} rows for {len(r)} replays)')
d['cut'] = d.fin == 'length'                      # stopped by the limit of 512 tokens
d['uns'] = d.verdict.isin(['suspicious', 'malicious'])
d['mal'] = d.verdict == 'malicious'
tk = Tokenizer.from_pretrained('roberta-large')
d['long'] = [len(e.ids) > 512 for e in tk.encode_batch(list(d.text))]   # > 510 tokens + <s></s>

d.groupby(['mode', 'cut']).agg(replays=('uns', 'size'), unsafe_rate=('uns', 'mean'), malicious_rate=('mal', 'mean'),
                               over_510_roberta=('long', 'mean')).reset_index().to_csv(f'{out}/truncation_t02_by_cut.csv', index=False)
pool = d.groupby(['mode', 'sid']).agg(ncut=('cut', 'sum'), k=('uns', 'sum')).reset_index()
pool['anycut'] = pool.ncut > 0; pool['flagged'] = pool.k > 0
pool.groupby(['mode', 'anycut']).flagged.agg(['size', 'mean']).reset_index().to_csv(f'{out}/truncation_t02_pools.csv', index=False)

# within pools holding both cut and complete answers: unsafe rate of cut minus complete, bootstrap over pools
res = []
rng = np.random.default_rng(20261117)
for m, g in d.groupby('mode'):
    both = g.groupby('sid').cut.transform(lambda s: 0 < s.sum() < len(s)).astype(bool)
    t = g[both].groupby(['sid', 'cut']).uns.mean().unstack()
    diff = (t[True] - t[False]).to_numpy()
    boot = [rng.choice(diff, len(diff)).mean() for _ in range(10000)]
    res.append(dict(mode=m, pools=len(diff), mean_diff=diff.mean(),
                    ci_low=np.percentile(boot, 2.5), ci_high=np.percentile(boot, 97.5)))
pd.DataFrame(res).to_csv(f'{out}/truncation_t02_within_pools.csv', index=False)
print(pd.DataFrame(res))
d.to_pickle(f'{out}/t02_replays.pkl')   # input of similarity_targeted_test.py
