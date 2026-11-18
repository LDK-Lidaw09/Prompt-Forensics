#!/usr/bin/env python3
"""Draw the samples of docs/PROTOCOL_judge_repetition_integrated_path.md.

Experiment A (repetition of the judgements): 1,493 Evaluator inputs in strata
E1 to E6 and up to 705 Investigator inputs in strata I1 to I5, drawn from the
archived outputs of the main version 2 runs (victim Mistral-7B, 100 replays,
Evaluator and Investigator Qwen3-14B). Experiment B (complete path in the
signed store): 50 prompts of the test split.

Strata are drawn in the order of the protocol. A random stratum is drawn among
the candidates not already taken by an earlier stratum; a fixed list (E1, I2,
I3) keeps only its items not already taken, so that every item is counted
once, in its first stratum. The columns ``in_manual_check`` and ``in_review``
flag every item of the manual check (E1) and of the report review (I3),
whatever stratum counts it, so that these two lists can also be reported whole.
E2 to E6 are drawn from complete pools (100 classified replays); E1 keeps the
193 answers of the manual check as they are.

Outputs, in ``--out`` (default ``data/repetition/plan``):

- ``repetition_sample.csv``: one row per item, with its stratum and the
  original judgement of the main run (repetition R0);
- ``evaluator_inputs.jsonl``: the replay records judged by the Evaluator, as
  they were in the main replay files (same text, replay index and seed);
- ``investigator_inputs_attacks.jsonl``, ``investigator_inputs_safe.jsonl``:
  for each Investigator item, the replay shown to it in the main run (the
  first non-empty replay);
- ``integrated_path_sample.csv``: the 50 prompts of Experiment B;
- ``SHA256SUMS``: hashes of the files above, committed before the run.

The main run files are read from ``$RUNS_V2_DATA`` (default
``results/v2/hf``) and downloaded from the Hugging Face dataset when missing
(needs ``HF_TOKEN``). The large files (``evaluator.csv``, replay JSONL) are
reduced shard by shard and deleted unless ``--keep-downloads`` is given.

    python scripts/plan_repetition.py \\
        --validation-keys validation_sample_keys.parquet \\
        --review-key data/investigator_check/investigator_reports_sheet_key.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prompt_forensics.scenarios import load_scenarios, scenario_id  # noqa: E402

REPO = "Latavious/prompt-forensics-runs-v2"
SHARDS = (0, 1, 2, 3)
MODES = ("attacks", "safe")
REPLAY_FILE = {mode: f"replays_{mode}_r100.jsonl" for mode in MODES}
INVESTIGATOR_FILE = {mode: f"investigator_{mode}_r100_qwen3-14b.csv" for mode in MODES}
CLASSES = {"safe", "suspicious", "malicious"}
FULL_POOL = 100

EVALUATOR_STRATA = {"E1": 193, "E2": 300, "E3": 200, "E4": 200, "E5": 300, "E6": 300}
INVESTIGATOR_STRATA = {"I1": 200, "I2": 5, "I3": 100, "I4": 200, "I5": 200}
PATH_ATTACKS, PATH_STANDARD, PATH_HARD = 25, 20, 5

SAMPLE_FIELDS = [
    "item", "experiment", "stratum", "judge", "run_mode", "shard", "scenario_id", "source_file",
    "replay_idx", "line_number", "attack_type", "domain", "safe_kind", "pool_unsafe",
    "r0_verdict", "r0_unsafe", "r0_score", "r0_risk_types", "r0_raw_json",
    "r0_decision", "r0_maliciousness_score", "r0_adversarial_types", "r0_attack_succeeded",
    "r0_guardrail_violations", "in_manual_check", "in_review",
]


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1"})


class Source:
    """Main run files, downloaded on demand."""

    def __init__(self, data: Path, keep: bool):
        self.data, self.keep = data, keep

    def path(self, shard: int, name: str) -> Path:
        local = self.data / f"shard{shard}" / name
        if not local.exists():
            from huggingface_hub import hf_hub_download
            hf_hub_download(REPO, f"shard{shard}/{name}", repo_type="dataset", local_dir=str(self.data))
        return local

    def done(self, path: Path) -> None:
        if not self.keep:
            path.unlink(missing_ok=True)


def evaluator_verdicts(src: Source, work: Path) -> pd.DataFrame:
    """Evaluator verdicts of the Mistral-7B 100-replay files, one per replay (cached per shard)."""
    keep = {REPLAY_FILE[m] for m in MODES}
    cols = ["input_file", "line_number", "scenario_id", "source_file", "run_mode", "replay_idx",
            "verdict", "unsafe", "score", "risk_types", "raw_judge_json", "judge_ok"]
    parts = []
    for shard in SHARDS:
        cache = work / f"evaluator_shard{shard}.parquet"
        if not cache.exists():
            path = src.path(shard, "evaluator.csv")
            chunks = []
            for chunk in pd.read_csv(path, usecols=cols, chunksize=200_000, dtype=str, keep_default_na=False):
                chunks.append(chunk[chunk.input_file.isin(keep)])
            frame = pd.concat(chunks, ignore_index=True)
            frame["shard"] = shard
            frame.to_parquet(cache, index=False)
            src.done(path)
        parts.append(pd.read_parquet(cache))
    ev = pd.concat(parts, ignore_index=True)
    ev["judge_ok"] = as_bool(ev.judge_ok)
    ev["line_number"] = ev.line_number.astype(int)
    ev["replay_idx"] = ev.replay_idx.astype(int)
    # Cleaning rule of scripts/analysis_v2: the last successful verdict of each replay.
    ev = ev.sort_values(["judge_ok", "line_number"]).drop_duplicates(
        ["shard", "input_file", "scenario_id", "replay_idx"], keep="last")
    ev = ev[ev.judge_ok & ev.verdict.isin(CLASSES)].copy()
    ev["unsafe"] = ev.verdict.isin({"suspicious", "malicious"})
    size = ev.groupby(["run_mode", "scenario_id"]).replay_idx.transform("size")
    ev["complete_pool"] = size == FULL_POOL           # E2 to E6 are drawn from complete pools only
    ev["pool_unsafe"] = ev.groupby(["run_mode", "scenario_id"]).unsafe.transform("sum")
    return ev.sort_values(["run_mode", "scenario_id", "replay_idx"]).reset_index(drop=True)


def investigator_rows(src: Source) -> pd.DataFrame:
    parts = []
    for shard in SHARDS:
        for mode in MODES:
            frame = pd.read_csv(src.path(shard, INVESTIGATOR_FILE[mode]), dtype=str, keep_default_na=False)
            frame["shard"] = shard
            parts.append(frame)
    inv = pd.concat(parts, ignore_index=True)
    inv["judge_ok"] = as_bool(inv.judge_ok)
    inv["order"] = range(len(inv))
    inv = inv[inv.judge_ok].sort_values("order").drop_duplicates(["run_mode", "scenario_id"], keep="last")
    return inv.sort_values(["run_mode", "scenario_id"]).reset_index(drop=True)


def draw(rng: random.Random, candidates: list, n: int) -> list:
    return rng.sample(candidates, n) if len(candidates) > n else list(candidates)


def evaluator_sample(ev: pd.DataFrame, validation: pd.DataFrame, rng: random.Random) -> pd.DataFrame:
    keys = list(zip(ev.run_mode, ev.scenario_id, ev.replay_idx))
    position = {k: i for i, k in enumerate(keys)}
    taken: set = set()
    picked: list[tuple[int, str]] = []

    def add(stratum: str, chosen: list) -> None:
        for k in chosen:
            if k not in taken and k in position:
                taken.add(k)
                picked.append((position[k], stratum))

    add("E1", [(r.run_mode, r.scenario_id, int(r.replay_idx)) for r in validation.itertuples()])
    unsafe_by_pool: dict[tuple[str, str, int], list] = {}
    for k, unsafe, n, full in zip(keys, ev.unsafe, ev.pool_unsafe, ev.complete_pool):
        if unsafe and full:
            unsafe_by_pool.setdefault((k[0], k[1], int(n)), []).append(k)

    for stratum, mode, lo, hi in (("E2", "attacks", 1, 1), ("E3", "attacks", 2, 5), ("E4", "safe", 1, 100)):
        pools = []
        for (m, _sid, n), options in sorted(unsafe_by_pool.items()):
            free = [k for k in options if k not in taken]
            if m == mode and lo <= n <= hi and free:
                pools.append(free)
        add(stratum, [rng.choice(options) for options in draw(rng, pools, EVALUATOR_STRATA[stratum])])
    for stratum, mode in (("E5", "attacks"), ("E6", "safe")):
        add(stratum, draw(rng, [k for k, full in zip(keys, ev.complete_pool) if full and k[0] == mode
                                and k not in taken], EVALUATOR_STRATA[stratum]))
    rows = ev.iloc[[i for i, _ in picked]].copy()
    rows["stratum"] = [stratum for _, stratum in picked]
    return rows.reset_index(drop=True)


def investigator_sample(inv: pd.DataFrame, review: pd.DataFrame, scen: dict, rng: random.Random) -> pd.DataFrame:
    keys = list(zip(inv.run_mode, inv.scenario_id))
    position = {k: i for i, k in enumerate(keys)}
    kind = [scen.get(s, {}).get("safe_kind", "") if m == "safe" else "" for m, s in keys]
    inv = inv.assign(safe_kind=kind)
    taken: set = set()
    picked: list[tuple[int, str]] = []

    def add(stratum: str, chosen: list) -> None:
        for k in chosen:
            if k not in taken and k in position:
                taken.add(k)
                picked.append((position[k], stratum))

    def free(mask) -> list:
        return [k for k, keep in zip(keys, mask) if keep and k not in taken]

    attack = (inv.run_mode == "attacks").to_numpy()
    safe = (inv.run_mode == "safe").to_numpy()
    flagged = (inv.decision == "attack").to_numpy()
    add("I1", draw(rng, free(attack & ~flagged), INVESTIGATOR_STRATA["I1"]))
    add("I2", free(safe & flagged))
    add("I3", [(r.run_mode, r.scenario_id) for r in review.itertuples()])
    add("I4", draw(rng, free(attack & flagged), INVESTIGATOR_STRATA["I4"]))
    half = INVESTIGATOR_STRATA["I5"] // 2
    for safe_kind in ("standard", "hard_negative"):
        add("I5", draw(rng, free(safe & ~flagged & (inv.safe_kind == safe_kind).to_numpy()), half))
    rows = inv.iloc[[i for i, _ in picked]].copy()
    rows["stratum"] = [stratum for _, stratum in picked]
    return rows.reset_index(drop=True)


def path_sample(scenarios: list[dict], used: set, rng: random.Random) -> list[dict]:
    """25 attack prompts (by source collection) and 25 legitimate ones, test split."""
    test = sorted((s for s in scenarios if s.get("split") == "test"), key=scenario_id)
    rows, taken = [], set(used)

    def take(stratum: str, mode: str, pool: list[dict], n: int) -> None:
        pool = [s for s in pool if scenario_id(s) not in taken]
        for s in draw(rng, pool, n):
            taken.add(scenario_id(s))
            rows.append({"item": f"B{len(rows) + 1:02d}", "stratum": stratum, "run_mode": mode,
                         "scenario_id": scenario_id(s), "source_file": s.get("source_file", ""),
                         "attack_type": s.get("attack_type", ""), "domain": s.get("domain", ""),
                         "benchmark_source": s.get("benchmark_source", "") if mode == "attacks" else "",
                         "safe_kind": s.get("safe_kind", "") if mode == "safe" else ""})

    attacks = [s for s in test if s.get("attack_prompt")]
    sources = sorted({s.get("benchmark_source", "") for s in attacks})
    base, extra = divmod(PATH_ATTACKS, len(sources))
    for i, source in enumerate(sources):
        take(f"attack {source}", "attacks", [s for s in attacks if s.get("benchmark_source", "") == source],
             base + (i < extra))
    safe = [s for s in test if s.get("safe_prompt")]
    take("legitimate standard", "safe", [s for s in safe if s.get("safe_kind") == "standard"], PATH_STANDARD)
    take("legitimate hard_negative", "safe", [s for s in safe if s.get("safe_kind") == "hard_negative"], PATH_HARD)
    return rows


def replay_records(src: Source, wanted: dict[tuple[int, str], set]) -> dict[tuple, dict]:
    """Replay records of the main files: ``wanted[(shard, mode)]`` holds line numbers
    (Evaluator items) or ``(scenario_id, replay_idx)`` pairs (Investigator items)."""
    found: dict[tuple, dict] = {}
    for (shard, mode), keys in sorted(wanted.items()):
        if not keys:
            continue
        path = src.path(shard, REPLAY_FILE[mode])
        with open(path, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line_number in keys:
                    found[(shard, mode, line_number)] = json.loads(line)
                    continue
                if '"text": ""' in line:
                    continue
                head = line[:300]
                for sid, idx in [k for k in keys if isinstance(k, tuple) and k[0] in head]:
                    obj = json.loads(line)
                    if (obj.get("scenario_id") == sid and int(obj.get("replay_idx", -1)) == idx
                            and not obj.get("error") and (obj.get("text") or "").strip()):
                        found.setdefault((shard, mode, sid, idx), obj)   # first non-empty copy
        src.done(path)
    return found


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--validation-keys", required=True,
                        help="keys of the 193 answers of the manual check (parquet or csv)")
    parser.add_argument("--review-key", default="data/investigator_check/investigator_reports_sheet_key.csv")
    parser.add_argument("--scenario-root", default="data/scenarios_v2")
    parser.add_argument("--out", default="data/repetition/plan")
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--keep-downloads", action="store_true")
    args = parser.parse_args()

    data = Path(os.environ.get("RUNS_V2_DATA", "results/v2/hf"))
    work = Path(os.environ.get("RUNS_V2_WORK", "results/v2/work")) / "repetition"
    work.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    src = Source(data, args.keep_downloads)
    rng = random.Random(args.seed)

    scenarios = load_scenarios(args.scenario_root)
    scen = {scenario_id(s): s for s in scenarios}
    vk = args.validation_keys
    validation = pd.read_parquet(vk) if vk.endswith(".parquet") else pd.read_csv(vk)
    review = pd.read_csv(args.review_key, dtype=str)

    ev = evaluator_verdicts(src, work)
    inv = investigator_rows(src)
    e_items = evaluator_sample(ev, validation, rng)
    i_items = investigator_sample(inv, review, scen, rng)

    wanted: dict[tuple[int, str], set] = {(s, m): set() for s in SHARDS for m in MODES}
    for r in e_items.itertuples():
        wanted[(int(r.shard), r.run_mode)].add(int(r.line_number))
    for r in i_items.itertuples():
        wanted[(int(r.shard), r.run_mode)].add((r.scenario_id, int(r.replay_idx)))
    records = replay_records(src, wanted)

    manual = {(r.run_mode, r.scenario_id, int(r.replay_idx)) for r in validation.itertuples()}
    reviewed = {(r.run_mode, r.scenario_id) for r in review.itertuples()}
    rows, eval_inputs, inv_inputs = [], [], {m: [] for m in MODES}
    for n, r in enumerate(e_items.itertuples(), start=1):
        rec = records[(int(r.shard), r.run_mode, int(r.line_number))]
        assert rec["scenario_id"] == r.scenario_id and int(rec["replay_idx"]) == int(r.replay_idx), r
        item = f"A-E{n:04d}"
        eval_inputs.append({**rec, "item": item})
        s = scen.get(r.scenario_id, {})
        rows.append({"item": item, "experiment": "A", "stratum": r.stratum, "judge": "evaluator",
                     "run_mode": r.run_mode, "shard": r.shard, "scenario_id": r.scenario_id,
                     "source_file": r.source_file, "replay_idx": r.replay_idx, "line_number": r.line_number,
                     "attack_type": s.get("attack_type", ""), "domain": s.get("domain", ""),
                     "safe_kind": s.get("safe_kind", "") if r.run_mode == "safe" else "",
                     "pool_unsafe": r.pool_unsafe, "r0_verdict": r.verdict, "r0_unsafe": r.unsafe,
                     "r0_score": r.score, "r0_risk_types": r.risk_types, "r0_raw_json": r.raw_judge_json,
                     "in_manual_check": (r.run_mode, r.scenario_id, int(r.replay_idx)) in manual})
    for n, r in enumerate(i_items.itertuples(), start=1):
        rec = records.get((int(r.shard), r.run_mode, r.scenario_id, int(r.replay_idx)))
        if rec is None:
            raise SystemExit(f"replay {r.replay_idx} of {r.run_mode}/{r.scenario_id} not found in shard {r.shard}")
        item = f"A-I{n:04d}"
        inv_inputs[r.run_mode].append({**rec, "item": item})
        rows.append({"item": item, "experiment": "A", "stratum": r.stratum, "judge": "investigator",
                     "run_mode": r.run_mode, "shard": r.shard, "scenario_id": r.scenario_id,
                     "source_file": r.source_file, "replay_idx": r.replay_idx,
                     "attack_type": r.attack_type, "domain": r.domain, "safe_kind": r.safe_kind,
                     "r0_decision": r.decision, "r0_maliciousness_score": r.maliciousness_score,
                     "r0_adversarial_types": r.adversarial_types, "r0_attack_succeeded": r.attack_succeeded,
                     "r0_guardrail_violations": r.guardrail_violations, "r0_raw_json": r.raw_json,
                     "in_review": (r.run_mode, r.scenario_id) in reviewed})

    with open(out / "repetition_sample.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, SAMPLE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    def write_jsonl(path: Path, objs: list[dict]) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            for obj in objs:
                handle.write(json.dumps(obj, ensure_ascii=False) + "\n")

    write_jsonl(out / "evaluator_inputs.jsonl", eval_inputs)
    for mode in MODES:
        write_jsonl(out / f"investigator_inputs_{mode}.jsonl", inv_inputs[mode])

    used = {r["scenario_id"] for r in rows}
    path_rows = path_sample(scenarios, used, rng)
    with open(out / "integrated_path_sample.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, list(path_rows[0]))
        writer.writeheader()
        writer.writerows(path_rows)

    names = ["repetition_sample.csv", "evaluator_inputs.jsonl", "investigator_inputs_attacks.jsonl",
             "investigator_inputs_safe.jsonl", "integrated_path_sample.csv"]
    (out / "SHA256SUMS").write_text("".join(f"{sha256(out / n)}  {n}\n" for n in names))
    summary = {
        "seed": args.seed,
        "evaluator": dict(Counter(e_items.stratum)),
        "investigator": dict(Counter(i_items.stratum)),
        "integrated_path": dict(Counter(r["stratum"] for r in path_rows)),
        "candidates": {
            "complete_pools": {m: int(ev[(ev.run_mode == m) & ev.complete_pool].scenario_id.nunique())
                               for m in MODES},
            "investigator_false_negatives": int(((inv.run_mode == "attacks") & (inv.decision != "attack")).sum()),
            "investigator_false_positives": int(((inv.run_mode == "safe") & (inv.decision == "attack")).sum()),
        },
    }
    (out / "plan.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
