#!/usr/bin/env python3
"""Measures of Experiment A (docs/PROTOCOL_judge_repetition_integrated_path.md).

Compares, item by item, the original judgement of the main run (R0, in
``repetition_sample.csv``) with the three repetitions R1 to R3 written by
``scripts/run_repetition.sh``, and writes in ``--out``:

- ``repetition_evaluator.csv``: per stratum and overall, verdict change rate
  (three classes and binary unsafe label) with Wilson 95% interval, and
  Fleiss' kappa of the four repetitions;
- ``repetition_rare_signals.csv``: E2, pools that would lose their only unsafe
  answer in at least one repetition; E2 to E4, items unsafe in all four;
- ``repetition_investigator.csv``: per stratum, overall, and on the whole list
  of the report review (``in_review``, whatever stratum counts it), decision change
  rate with Wilson interval, change of declared attack type, success field and
  rule cited, mean and maximum spread of the maliciousness score;
- ``repetition_identical_text.csv``: share of identical judge outputs (raw text
  across R1 to R3, since the main run did not keep the raw text; parsed JSON
  across R0 to R3);
- ``repetition_items.csv``: one row per item with its four judgements.

An item whose judge failed in one repetition is left out of the rates and
counted in ``items_incomplete``.

    python scripts/analysis_v2/repetition.py --plan data/repetition/plan \\
        --runs results/v2/repetition/A --out results/v2/repetition/tables
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

REPS = ("R0", "R1", "R2", "R3")
CLASSES = ("safe", "suspicious", "malicious")


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    p = k / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - half) / (1 + z * z / n)), min(1.0, (centre + half) / (1 + z * z / n))


def fleiss_kappa(labels: list[list], categories) -> float:
    """Fleiss' kappa for items rated by the same number of raters."""
    if not labels:
        return math.nan
    m = len(labels[0])
    counts = [[row.count(c) for c in categories] for row in labels]
    n = len(counts)
    p_items = [(sum(x * x for x in row) - m) / (m * (m - 1)) for row in counts]
    p_cat = [sum(row[j] for row in counts) / (n * m) for j in range(len(categories))]
    p_bar, p_e = sum(p_items) / n, sum(p * p for p in p_cat)
    return math.nan if p_e == 1 else (p_bar - p_e) / (1 - p_e)


def as_bool(value) -> bool | None:
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    return None


def as_list(value) -> tuple:
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError):
        return (str(value),)
    return tuple(sorted(str(x).strip().lower() for x in parsed)) if isinstance(parsed, list) else ()


def canonical(value) -> str | None:
    try:
        return json.dumps(json.loads(value), sort_keys=True)
    except (TypeError, ValueError):
        return None


def load_runs(runs: Path, pattern: str) -> dict[str, pd.DataFrame]:
    out = {}
    for rep in REPS[1:]:
        files = sorted(runs.glob(pattern.format(rep=rep)))
        if files:
            frame = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False) for f in files], ignore_index=True)
            frame = frame[frame.judge_ok.str.lower().isin({"true", "1"})]
            out[rep] = frame
    return out


def rate_row(group: str, items: int, changed: int, prefix: str) -> dict:
    lo, hi = wilson(changed, items)
    return {f"{prefix}_changed": changed, f"{prefix}_change_rate": changed / items if items else math.nan,
            f"{prefix}_ci_low": lo, f"{prefix}_ci_high": hi}


def evaluator(sample: pd.DataFrame, runs: Path, out: Path) -> pd.DataFrame:
    items = sample[sample.judge == "evaluator"].copy()
    key = lambda f: list(zip(f.run_mode, f.scenario_id, f.replay_idx.astype(int)))  # noqa: E731
    items["key"] = key(items)
    items["R0_verdict"], items["R0_json"] = items.r0_verdict, items.r0_raw_json.map(canonical)
    items["R0_text"] = None
    for rep, frame in load_runs(runs, "evaluator_{rep}.csv").items():
        frame = frame.assign(key=key(frame)).drop_duplicates("key", keep="last").set_index("key")
        items[f"{rep}_verdict"] = items.key.map(frame.verdict)
        items[f"{rep}_json"] = items.key.map(frame.raw_judge_json.map(canonical))
        items[f"{rep}_text"] = items.key.map(frame.raw_judge_text)
    for rep in REPS:
        for col in ("verdict", "json", "text"):
            if f"{rep}_{col}" not in items:
                items[f"{rep}_{col}"] = None
    verdicts = items[[f"{r}_verdict" for r in REPS]]
    items["complete"] = verdicts.isin(CLASSES).all(axis=1)
    items["verdict_changed"] = verdicts.nunique(axis=1) > 1
    unsafe = verdicts.isin({"suspicious", "malicious"})
    items["unsafe_changed"] = unsafe.nunique(axis=1) > 1
    items["unsafe_all_four"] = unsafe.all(axis=1)
    items["unsafe_lost_once"] = items.r0_unsafe.map(as_bool) & ~unsafe[[f"{r}_verdict" for r in REPS[1:]]].all(axis=1)
    items["text_identical_r1_r3"] = items[[f"{r}_text" for r in REPS[1:]]].nunique(axis=1, dropna=False) == 1
    items["json_identical_r0_r3"] = items[[f"{r}_json" for r in REPS]].nunique(axis=1, dropna=False) == 1

    rows, rare = [], []
    for group, frame in [*items.groupby("stratum"), ("all", items)]:
        done = frame[frame.complete]
        n = len(done)
        labels = [list(r) for r in done[[f"{x}_verdict" for x in REPS]].itertuples(index=False)]
        binary = [[v != "safe" for v in row] for row in labels]
        rows.append({"stratum": group, "items": len(frame), "items_complete": n, "items_incomplete": len(frame) - n,
                     **rate_row(group, n, int(done.verdict_changed.sum()), "verdict"),
                     **rate_row(group, n, int(done.unsafe_changed.sum()), "unsafe_label"),
                     "fleiss_kappa_3class": fleiss_kappa(labels, CLASSES),
                     "fleiss_kappa_binary": fleiss_kappa(binary, (False, True))})
        if group in {"E2", "E3", "E4"}:
            lost, always = int(done.unsafe_lost_once.sum()), int(done.unsafe_all_four.sum())
            rare.append({"stratum": group, "items_complete": n,
                         "pools_losing_only_unsafe_answer": lost if group == "E2" else None,
                         "share_losing": lost / n if group == "E2" and n else None,
                         "losing_ci_low": wilson(lost, n)[0] if group == "E2" else None,
                         "losing_ci_high": wilson(lost, n)[1] if group == "E2" else None,
                         "unsafe_in_all_four": always, "share_unsafe_all_four": always / n if n else math.nan,
                         "all_four_ci_low": wilson(always, n)[0], "all_four_ci_high": wilson(always, n)[1]})
    pd.DataFrame(rows).to_csv(out / "repetition_evaluator.csv", index=False)
    pd.DataFrame(rare).to_csv(out / "repetition_rare_signals.csv", index=False)
    return items


def investigator(sample: pd.DataFrame, runs: Path, out: Path) -> pd.DataFrame:
    items = sample[sample.judge == "investigator"].copy()
    key = lambda f: list(zip(f.run_mode, f.scenario_id))  # noqa: E731
    items["key"] = key(items)
    items["R0_decision"] = items.r0_decision
    items["R0_types"] = items.r0_adversarial_types.map(as_list)
    items["R0_success"] = items.r0_attack_succeeded.map(as_bool)
    items["R0_rules"] = items.r0_guardrail_violations.map(as_list)
    items["R0_score"] = pd.to_numeric(items.r0_maliciousness_score, errors="coerce")
    items["R0_json"] = items.r0_raw_json.map(canonical)
    items["R0_text"] = None
    for rep, frame in load_runs(runs, "investigator_*_{rep}.csv").items():
        frame = frame.assign(key=key(frame)).drop_duplicates("key", keep="last").set_index("key")
        items[f"{rep}_decision"] = items.key.map(frame.decision)
        items[f"{rep}_types"] = items.key.map(frame.adversarial_types.map(as_list))
        items[f"{rep}_success"] = items.key.map(frame.attack_succeeded.map(as_bool))
        items[f"{rep}_rules"] = items.key.map(frame.guardrail_violations.map(as_list))
        items[f"{rep}_score"] = items.key.map(pd.to_numeric(frame.maliciousness_score, errors="coerce"))
        items[f"{rep}_json"] = items.key.map(frame.raw_json.map(canonical))
        items[f"{rep}_text"] = items.key.map(frame.raw_text)
    for rep in REPS:
        for col in ("decision", "types", "success", "rules", "score", "json", "text"):
            if f"{rep}_{col}" not in items:
                items[f"{rep}_{col}"] = None
    decisions = items[[f"{r}_decision" for r in REPS]]
    items["complete"] = decisions.isin({"attack", "safe"}).all(axis=1)
    for col in ("decision", "types", "success", "rules"):
        items[f"{col}_changed"] = items[[f"{r}_{col}" for r in REPS]].astype(str).nunique(axis=1) > 1
    scores = items[[f"{r}_score" for r in REPS]].astype(float)
    items["score_spread"] = scores.max(axis=1) - scores.min(axis=1)
    items["text_identical_r1_r3"] = items[[f"{r}_text" for r in REPS[1:]]].nunique(axis=1, dropna=False) == 1
    items["json_identical_r0_r3"] = items[[f"{r}_json" for r in REPS]].nunique(axis=1, dropna=False) == 1

    rows = []
    review = items[items.in_review.astype(str).str.lower() == "true"] if "in_review" in items else items.iloc[:0]
    for group, frame in [*items.groupby("stratum"), ("all", items), ("I3 whole review list", review)]:
        done = frame[frame.complete]
        n = len(done)
        row = {"stratum": group, "items": len(frame), "items_complete": n, "items_incomplete": len(frame) - n,
               **rate_row(group, n, int(done.decision_changed.sum()), "decision")}
        for col in ("types", "success", "rules"):
            row[f"{col}_change_rate"] = done[f"{col}_changed"].mean() if n else math.nan
        row["score_spread_mean"] = done.score_spread.mean() if n else math.nan
        row["score_spread_max"] = done.score_spread.max() if n else math.nan
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "repetition_investigator.csv", index=False)
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", default="data/repetition/plan")
    parser.add_argument("--runs", default="results/v2/repetition/A")
    parser.add_argument("--out", default="results/v2/repetition/tables")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sample = pd.read_csv(Path(args.plan) / "repetition_sample.csv", dtype=str, keep_default_na=False)
    ev = evaluator(sample, Path(args.runs), out)
    inv = investigator(sample, Path(args.runs), out)

    identical = []
    for judge, frame in (("evaluator", ev), ("investigator", inv)):
        done = frame[frame.complete]
        identical.append({"judge": judge, "items_complete": len(done),
                          "raw_text_identical_r1_r3": done.text_identical_r1_r3.mean() if len(done) else math.nan,
                          "parsed_json_identical_r0_r3": done.json_identical_r0_r3.mean() if len(done) else math.nan})
    pd.DataFrame(identical).to_csv(out / "repetition_identical_text.csv", index=False)
    keep = ["item", "stratum", "judge", "run_mode", "scenario_id", "replay_idx", "complete"]
    detail = pd.concat([
        ev[keep + [f"{r}_verdict" for r in REPS] + ["verdict_changed", "unsafe_changed"]],
        inv[keep + [f"{r}_decision" for r in REPS] + [f"{r}_score" for r in REPS] + ["decision_changed"]],
    ], ignore_index=True)
    detail.to_csv(out / "repetition_items.csv", index=False)
    print(pd.read_csv(out / "repetition_evaluator.csv").to_string())
    print(pd.read_csv(out / "repetition_investigator.csv").to_string())
    print(pd.DataFrame(identical).to_string())


if __name__ == "__main__":
    main()
