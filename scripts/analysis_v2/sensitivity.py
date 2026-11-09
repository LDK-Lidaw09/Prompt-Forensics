"""Tables of the sensitivity runs (scripts/run_sensitivity.sh).

    python scripts/analysis_v2/sensitivity.py [SENSITIVITY_DIR]

SENSITIVITY_DIR defaults to results/v2/sensitivity (the unpacked
results/v2/sensitivity.tar.gz, or the sensitivity/ folder of the Hub copy).
The paired baseline of the ablation comes from the main runs: the Qwen3-14B
Investigator CSVs of the four shards under $RUNS_V2_DATA (default
results/v2/hf); without them that table is skipped.

Writes to results/v2/tables/:

- sensitivity_temperature.csv: per temperature and run mode, the number of
  pools, mean Consistency (raw cosine), share of pools with Consistency
  >= 0.9, share of pools whose replays are all the same text, the per-replay
  unsafe rate, and the share of prompts with at least one unsafe replay
  among 20 (Wilson 95 % interval);
- sensitivity_seed.csv: replays of the first pass at 0.2 and of the second
  pass with the same seeds, compared text for text, on the prompts whose
  replays are all present in both passes; the interval is a bootstrap over
  prompts (replays of one prompt are not independent);
- sensitivity_ablation.csv: recall and false-positive rate of the Qwen3-14B
  Investigator with and without the victim's answer, on the same prompts,
  with McNemar's test (continuity correction) on the paired decisions.

Cleaning rules as in README.md: error or empty replays are dropped, one
Evaluator verdict per (file, scenario, replay), failed Investigator calls
are excluded and counted.
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from prompt_forensics.experiments.analysis import wilson_interval  # noqa: E402

S = Path(sys.argv[1] if len(sys.argv) > 1 else "results/v2/sensitivity")
D = os.environ.get("RUNS_V2_DATA", "results/v2/hf")
T = Path(os.environ.get("TABLES", "results/v2/tables"))
T.mkdir(parents=True, exist_ok=True)


def temp_of(name: str) -> float:
    """replays_attacks_t07.jsonl -> 0.7 (t0 -> 0.0, t10 -> 1.0)."""
    digits = re.search(r"_t(\d+)", name).group(1)
    return float(digits[0] + "." + digits[1:]) if len(digits) > 1 else float(digits)


def read_replays(path: Path) -> pd.DataFrame:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            obj = json.loads(line)
            if obj.get("error") or not (obj.get("text") or "").strip():
                continue
            rows.append({k: obj.get(k) for k in ("scenario_id", "run_mode", "replay_idx", "seed", "text")})
    # A resumed run can write a replay twice: keep the last one.
    return pd.DataFrame(rows).drop_duplicates(["scenario_id", "run_mode", "replay_idx"], keep="last")


def ci(k: int, n: int) -> tuple[float, float, float]:
    lo, hi = wilson_interval(k, n)
    return (k / n if n else math.nan), lo, hi


def cluster_bootstrap(hits: pd.Series, groups: pd.Series, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95 % percentile interval of a rate, resampling whole groups (prompts)."""
    per_group = pd.DataFrame({"hit": hits.astype(float), "g": groups.values}).groupby("g").hit.agg(["sum", "count"])
    if per_group.empty:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_group), size=(n_boot, len(per_group)))
    sums, counts = per_group["sum"].to_numpy()[idx].sum(1), per_group["count"].to_numpy()[idx].sum(1)
    rates = sums / counts
    return float(np.quantile(rates, 0.025)), float(np.quantile(rates, 0.975))


def mcnemar(b: int, c: int) -> tuple[float, float]:
    """Chi-square with continuity correction, (|b - c| - 1)^2 / (b + c), and its p-value (1 df)."""
    if b + c == 0:
        return math.nan, math.nan
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return chi2, math.erfc(math.sqrt(chi2 / 2))


# ---- Temperature -----------------------------------------------------------------------
ev = pd.read_csv(S / "evaluator.csv")
ev = ev[ev.judge_ok.astype(str).isin(["True", "true", "1"]) & ev.verdict.ne("unknown")]
ev = ev.drop_duplicates(["input_file", "scenario_id", "replay_idx"], keep="last")
ev["unsafe"] = ev.unsafe.astype(str).isin(["True", "true", "1"])

rows = []
for csv_path in sorted(S.glob("replays_*_t*.csv")):
    mode, t = csv_path.stem.split("_")[1], temp_of(csv_path.stem)
    pools = pd.read_csv(csv_path).drop_duplicates(["scenario_id", "run_mode"], keep="last")
    pools = pools[pools.complete.astype(str).isin(["True", "true", "1"])]
    texts = read_replays(csv_path.with_suffix(".jsonl"))
    # Only pools with every requested replay present: a pool reduced to one
    # surviving replay would otherwise count as "all identical".
    counts = texts.groupby("scenario_id").replay_idx.nunique()
    complete = counts[counts >= pools.num_requested.max()].index
    same = texts[texts.scenario_id.isin(complete)].groupby("scenario_id").text.nunique().eq(1)
    e = ev[ev.input_file == csv_path.with_suffix(".jsonl").name]
    full = e.groupby("scenario_id").size()
    full = full[full >= pools.num_requested.max()].index  # pools with every replay judged
    e = e[e.scenario_id.isin(full)]
    any_unsafe = e.groupby("scenario_id").unsafe.any()
    p, lo, hi = ci(int(any_unsafe.sum()), len(any_unsafe))
    rows.append({
        "temperature": t, "run_mode": mode, "pools": len(pools),
        "consistency_mean": pools.cons_score.mean(), "consistency_ge_0_9": (pools.cons_score >= 0.9).mean(),
        "all_replays_identical": same.mean(),
        "replays_judged": len(e), "unsafe_per_replay": e.unsafe.mean(),
        "pools_judged": len(any_unsafe), "pools_any_unsafe": int(any_unsafe.sum()),
        "any_unsafe_rate": p, "any_unsafe_lo": lo, "any_unsafe_hi": hi,
    })
temperature = pd.DataFrame(rows).sort_values(["run_mode", "temperature"])
temperature.to_csv(T / "sensitivity_temperature.csv", index=False)
print(temperature.round(4).to_string(index=False))

# ---- Same seed, same text? -----------------------------------------------------------------
rows = []
for second in sorted(S.glob("seedcheck_*_t02.jsonl")):
    mode = second.stem.split("_")[1]
    a = read_replays(S / f"replays_{mode}_t02.jsonl")
    b = read_replays(second)
    m = a.merge(b, on=["scenario_id", "run_mode", "replay_idx", "seed"], suffixes=("_a", "_b"))
    # Keep prompts whose replays 0 .. n-1 of the second pass all have a match.
    expected = int(b.replay_idx.max()) + 1 if len(b) else 0
    matched = m.groupby("scenario_id").replay_idx.nunique()
    m = m[m.scenario_id.isin(matched[matched >= expected].index)]
    exact = m.text_a.eq(m.text_b)
    lo, hi = cluster_bootstrap(exact, m.scenario_id)
    rows.append({"run_mode": mode, "prompts": m.scenario_id.nunique(),
                 "prompts_left_out": int((matched < expected).sum()),
                 "replays_compared": len(m), "identical": int(exact.sum()),
                 "identical_rate": exact.mean() if len(m) else math.nan, "lo_bootstrap": lo, "hi_bootstrap": hi,
                 "pools_all_identical": exact.groupby(m.scenario_id).all().mean()})
seed = pd.DataFrame(rows)
seed.to_csv(T / "sensitivity_seed.csv", index=False)
print(seed.round(4).to_string(index=False))


# ---- Investigator with and without the answer ------------------------------------------------
def investigator(path: str | Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    d = d.sort_values("judge_ok").drop_duplicates(["scenario_id", "run_mode"], keep="last")
    d["ok"] = d.judge_ok.astype(str).isin(["True", "true", "1"])
    d["flag"] = d.decision.astype(str).str.lower().eq("attack")
    return d[["scenario_id", "run_mode", "ok", "flag"]]


def compare(name: str, with_answer: pd.DataFrame, without: pd.DataFrame) -> list[dict]:
    m = with_answer.merge(without, on=["scenario_id", "run_mode"], suffixes=("_with", "_without"))
    # Prompts of the ablation sample with no baseline row (should be 0 for the
    # main runs, which investigated every prompt) are counted, not hidden.
    unmatched = len(without) - len(m) if name.startswith("main") else 0
    failed = int((~m.ok_with | ~m.ok_without).sum())
    m = m[m.ok_with & m.ok_without]
    out = []
    for mode, measure in (("attacks", "recall"), ("safe", "false_positive_rate")):
        g = m[m.run_mode == mode]
        b = int((g.flag_with & ~g.flag_without).sum())   # flagged only with the answer
        c = int((~g.flag_with & g.flag_without).sum())   # flagged only without it
        chi2, pval = mcnemar(b, c)
        row = {"baseline": name, "run_mode": mode, "measure": measure, "n": len(g), "failed_pairs_excluded": failed,
               "prompts_without_baseline": unmatched}
        for cond in ("with", "without"):
            p, lo, hi = ci(int(g[f"flag_{cond}"].sum()), len(g))
            row.update({f"{cond}_answer": p, f"{cond}_lo": lo, f"{cond}_hi": hi})
        row.update({"only_with_answer": b, "only_without_answer": c, "mcnemar_chi2_cc": chi2, "p_value": pval})
        out.append(row)
    return out


rows = []
without = pd.concat([investigator(f) for f in sorted(S.glob("investigator_noout_*.csv"))])
same_run = sorted(S.glob("investigator_withanswer_*.csv"))
if same_run:
    rows += compare("same run, temperature sample", pd.concat([investigator(f) for f in same_run]), without)
main_runs = sorted(glob.glob(f"{D}/shard*/investigator_*_r100_qwen3-14b.csv"))
if main_runs:
    rows += compare("main v2 runs, ablation sample", pd.concat([investigator(f) for f in main_runs]), without)
else:
    print(f"no main-run Investigator CSVs under {D}: paired baseline on the whole ablation sample skipped")
ablation = pd.DataFrame(rows)
ablation.to_csv(T / "sensitivity_ablation.csv", index=False)
print(ablation.round(4).to_string(index=False))
