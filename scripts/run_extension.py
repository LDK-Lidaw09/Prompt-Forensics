#!/usr/bin/env python3
"""Experiments C and D of docs/PROTOCOL_policies_verdict_chain_fidelity.md.

C, contrasted policies (``data/extension/plan/policy_pairs.csv``):

``policies-victim`` (victim served)
    one answer of the victim per pair (seed 1234, temperature 0.2, 512 tokens),
    with the role only (no rule), the document and the request in one user
    message, as in the main runs; the same answer is given to the judge under
    both policies, so that only the rule changes within a pair;
``policies-judge`` (judge served)
    the Investigator Qwen3-14B on each item, with the settings of the main runs,
    in an order shuffled with a fixed seed, one independent call per item. The
    judge sees the system prompt, the document, the request and the answer,
    never the condition, the pair or the reference label.

D, Fidelity against an answer recorded apart (``fidelity_sample.csv``):

``fidelity-record`` (victim served, first server process)
    the recorded answer of each prompt (seed 1234);
``fidelity-replay`` (victim served again, after a restart of the server)
    20 replays per prompt, seeds 1234 + i for i = 0 to 19;
``fidelity-judge`` (judge served)
    the Evaluator on the recorded answer and on every replay;
``fidelity-score`` (no model served; needs the ``ml`` extra)
    cosine and BERTScore of every replay against the recorded answer, against
    replay 0, and against the recorded answer of another prompt of the same
    domain and run mode (substitution).

Every stage can be run again after a stop: rows already written are skipped.

    python scripts/run_extension.py policies-victim --out results/v2/extension
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prompt_forensics.agents.evaluator import evaluate_output  # noqa: E402
from prompt_forensics.agents.investigator import investigate_pair  # noqa: E402
from prompt_forensics.llm import call_llm  # noqa: E402
from prompt_forensics.scenarios import build_victim_prompt, load_scenarios, scenario_id  # noqa: E402

VICTIM = "mistralai/Mistral-7B-Instruct-v0.3"
JUDGE = "Qwen/Qwen3-14B"
VICTIM_PARAMS = {"temperature": 0.2, "max_tokens": 512}
SEED_BASE, FIDELITY_REPLAYS = 1234, 20
JUDGE_ORDER_SEED = 7   # order of the Investigator calls of experiment C
SUBSTITUTION_SEED = 7
_lock = threading.Lock()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_csv(path: Path, row: dict[str, Any]) -> None:
    with _lock:
        new = not path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, list(row))
            if new:
                writer.writeheader()
            writer.writerow(row)


def failure(out: Path, step: str, item: str, exc: BaseException) -> None:
    append_csv(out / "failures.csv", {"step": step, "item": item, "error": f"{type(exc).__name__}: {exc}",
                                      "created_at": now()})
    traceback.print_exc()


def victim(prompt: str, model: str, seed: int) -> dict[str, Any]:
    response = call_llm(prompt, model, {**VICTIM_PARAMS, "seed": seed})
    return {"text": response.text, "finish_reason": getattr(response, "finish_reason", "") or "",
            "model_used": response.model}


def pool(workers: int, fn, jobs) -> None:
    with ThreadPoolExecutor(workers) as executor:
        list(executor.map(fn, jobs))


# ----------------------------------------------------------------------------- C

def policy_items(args) -> tuple[list[dict[str, str]], dict[str, dict]]:
    rows = read_csv(Path(args.plan) / "policy_pairs.csv")
    scenarios = {r["item"]: {"id": r["item"], "domain": r["domain"], "system_prompt": r["system_prompt"],
                             "dataset": r["dataset"], "safe_prompt": r["request"]} for r in rows}
    return rows, scenarios


def cmd_policies_victim(args) -> None:
    out = Path(args.out) / "C"
    rows, _ = policy_items(args)
    pairs = {r["pair"]: r for r in rows if r["condition"] == "permit"}
    done = {r["pair"] for r in read_csv(out / "answers.csv")}

    def run(row: dict[str, str]) -> None:
        if row["pair"] in done:
            return
        scenario = {"id": row["pair"], "domain": row["domain"], "system_prompt": row["role_prompt"],
                    "dataset": row["dataset"], "safe_prompt": row["request"]}
        try:
            answer = victim(build_victim_prompt(scenario, "safe"), args.victim, SEED_BASE)
            append_csv(out / "answers.csv", {"pair": row["pair"], "seed": SEED_BASE, **answer, "created_at": now()})
        except Exception as exc:   # noqa: BLE001 - every failure is recorded with its cause
            failure(out, "policies-victim", row["pair"], exc)

    pool(args.workers, run, list(pairs.values()))
    print(f"answers: {len(read_csv(out / 'answers.csv'))} of {len(pairs)}")


def cmd_policies_judge(args) -> None:
    out = Path(args.out) / "C"
    rows, scenarios = policy_items(args)
    answers = {r["pair"]: r for r in read_csv(out / "answers.csv")}
    done = {r["item"] for r in read_csv(out / "investigator.csv")}

    def run(row: dict[str, str]) -> None:
        if row["item"] in done or row["pair"] not in answers:
            return
        try:
            report, raw, timing = investigate_pair(
                scenarios[row["item"]], "safe", answers[row["pair"]]["text"], args.judge,
                params={"temperature": 0.0, "max_tokens": 640, "seed": 777, "repetition_penalty": 1.3},
                guided_json=True, guided_backend="xgrammar", dataset_max_chars=1000)
            d = report.as_dict()
            append_csv(out / "investigator.csv", {
                "item": row["item"], "judge_ok": report.ok, "judge_error": report.error or "",
                "decision": d["decision"], "maliciousness_score": d["maliciousness_score"],
                "adversarial_types": json.dumps(d["adversarial_types"]),
                "guardrail_violations": json.dumps(d["guardrail_violations"], ensure_ascii=False),
                "risk_markers": json.dumps(d["risk_markers"], ensure_ascii=False),
                "attack_succeeded": d["attack_succeeded"], "summary": d["summary"], "raw_text": raw,
                "timing_ms": timing, "created_at": now()})
        except Exception as exc:   # noqa: BLE001
            failure(out, "policies-judge", row["item"], exc)

    order = list(rows)
    random.Random(JUDGE_ORDER_SEED).shuffle(order)
    pool(args.workers, run, order)
    print(f"reports: {len(read_csv(out / 'investigator.csv'))} of {len(rows)}")


# ----------------------------------------------------------------------------- D

def fidelity_items(args) -> tuple[list[dict[str, str]], dict[str, dict]]:
    rows = read_csv(Path(args.plan) / "fidelity_sample.csv")
    scenarios = {scenario_id(s): s for s in load_scenarios(args.scenario_root)}
    return rows, scenarios


def cmd_fidelity_record(args) -> None:
    out = Path(args.out) / "D"
    rows, scenarios = fidelity_items(args)
    done = {r["item"] for r in read_csv(out / "recorded.csv")}

    def run(row: dict[str, str]) -> None:
        if row["item"] in done:
            return
        try:
            prompt = build_victim_prompt(scenarios[row["scenario_id"]], row["run_mode"])
            answer = victim(prompt, args.victim, SEED_BASE)
            append_csv(out / "recorded.csv", {"item": row["item"], "seed": SEED_BASE, **answer,
                                              "server_process": args.server_process, "created_at": now()})
        except Exception as exc:   # noqa: BLE001
            failure(out, "fidelity-record", row["item"], exc)

    pool(args.workers, run, rows)
    print(f"recorded: {len(read_csv(out / 'recorded.csv'))} of {len(rows)}")


def cmd_fidelity_replay(args) -> None:
    out = Path(args.out) / "D"
    rows, scenarios = fidelity_items(args)
    done = {(r["item"], r["replay"]) for r in read_csv(out / "replays.csv")}

    def run(job: tuple[dict[str, str], int]) -> None:
        row, i = job
        if (row["item"], str(i)) in done:
            return
        try:
            prompt = build_victim_prompt(scenarios[row["scenario_id"]], row["run_mode"])
            answer = victim(prompt, args.victim, SEED_BASE + i)
            append_csv(out / "replays.csv", {"item": row["item"], "replay": i, "seed": SEED_BASE + i, **answer,
                                             "server_process": args.server_process, "created_at": now()})
        except Exception as exc:   # noqa: BLE001
            failure(out, "fidelity-replay", f"{row['item']}#{i}", exc)

    pool(args.workers, run, [(r, i) for r in rows for i in range(FIDELITY_REPLAYS)])
    print(f"replays: {len(read_csv(out / 'replays.csv'))} of {len(rows) * FIDELITY_REPLAYS}")


def cmd_fidelity_judge(args) -> None:
    out = Path(args.out) / "D"
    texts = [("recorded", r["item"], "", r["text"]) for r in read_csv(out / "recorded.csv")]
    texts += [("replay", r["item"], r["replay"], r["text"]) for r in read_csv(out / "replays.csv")]
    done = {(r["source"], r["item"], r["replay"]) for r in read_csv(out / "evaluator.csv")}

    def run(job: tuple[str, str, str, str]) -> None:
        source, item, replay, text = job
        if (source, item, replay) in done:
            return
        try:
            # Same seed rule as the main runs: 777 + replay index; the recorded answer counts as index 0.
            verdict, raw, _ = evaluate_output(text, judge_model=args.judge, temperature=0.0, max_tokens=512,
                                              seed=777 + int(replay or 0), guided_json=True,
                                              guided_backend="xgrammar")
            append_csv(out / "evaluator.csv", {"source": source, "item": item, "replay": replay,
                                               "judge_ok": verdict.ok, "verdict": verdict.verdict,
                                               "unsafe": verdict.unsafe, "raw_text": raw, "created_at": now()})
        except Exception as exc:   # noqa: BLE001
            failure(out, "fidelity-judge", f"{source}:{item}#{replay}", exc)

    pool(args.workers, run, texts)
    print(f"verdicts: {len(read_csv(out / 'evaluator.csv'))} of {len(texts)}")


def substitutes(rows: list[dict[str, str]]) -> dict[str, str]:
    """For each item, another item of the same domain and run mode (fixed seed)."""
    rng = random.Random(SUBSTITUTION_SEED)
    groups: dict[tuple[str, str], list[str]] = {}
    for r in rows:
        groups.setdefault((r["domain"], r["run_mode"]), []).append(r["item"])
    mapping = {}
    for items in groups.values():
        shifted = items[:]
        rng.shuffle(shifted)
        # a derangement: each item takes the next one in a shuffled cycle
        for a, b in zip(shifted, shifted[1:] + shifted[:1]):
            mapping[a] = b
    return mapping


def cmd_fidelity_score(args) -> None:
    from prompt_forensics.metrics.robustness import map_cosine
    from prompt_forensics.metrics.similarity import bertscore, cosine_matrix

    out = Path(args.out) / "D"
    rows, _ = fidelity_items(args)
    recorded = {r["item"]: r["text"] for r in read_csv(out / "recorded.csv")}
    replays: dict[str, dict[int, str]] = {}
    for r in read_csv(out / "replays.csv"):
        replays.setdefault(r["item"], {})[int(r["replay"])] = r["text"]
    sub = substitutes(rows)
    scored, loo_rows = [], []
    for row in rows:
        item = row["item"]
        reps = replays.get(item, {})
        if item not in recorded or len(reps) < FIDELITY_REPLAYS or sub.get(item) not in recorded:
            continue
        texts = [recorded[item], recorded[sub[item]]] + [reps[i] for i in range(FIDELITY_REPLAYS)]
        cos = cosine_matrix(texts)
        cands, refs, keys = [], [], []
        for i in range(FIDELITY_REPLAYS):
            for ref_name, ref_text in (("recorded", recorded[item]), ("substitute", recorded[sub[item]]),
                                       ("replay0", reps[0])):
                if ref_name == "replay0" and i == 0:
                    continue
                cands.append(reps[i])
                refs.append(ref_text)
                keys.append((i, ref_name))
        bert = bertscore(cands, refs)
        bert_of = {k: f1 for k, (_, _, f1) in zip(keys, bert)}
        for i in range(FIDELITY_REPLAYS):
            scored.append({
                "item": item, "replay": i, "substitute_item": sub[item],
                "cos_recorded": map_cosine(cos[2 + i][0], "shifted"),
                "cos_substitute": map_cosine(cos[2 + i][1], "shifted"),
                "cos_replay0": "" if i == 0 else map_cosine(cos[2 + i][2], "shifted"),
                "bert_recorded": bert_of[(i, "recorded")], "bert_substitute": bert_of[(i, "substitute")],
                "bert_replay0": "" if i == 0 else bert_of[(i, "replay0")],
                "identical_recorded": reps[i] == recorded[item],
            })
        # leave-one-out Fidelity of each of the 21 answers (recorded first), for the rank of the recorded one
        pool_ids = [0] + list(range(2, 2 + FIDELITY_REPLAYS))
        for a in pool_ids:
            loo = [map_cosine(cos[a][b], "shifted") for b in pool_ids if b != a]
            loo_rows.append({"item": item, "answer": "recorded" if a == 0 else f"replay{a - 2}",
                             "fidelity_cos": sum(loo) / len(loo)})
    write(out / "scores.csv", scored)
    write(out / "loo_fidelity.csv", loo_rows)
    print(f"scored {len({r['item'] for r in scored})} prompts")


def write(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["policies-victim", "policies-judge", "fidelity-record", "fidelity-replay",
                                          "fidelity-judge", "fidelity-score"])
    parser.add_argument("--out", default="results/v2/extension")
    parser.add_argument("--plan", default="data/extension/plan")
    parser.add_argument("--scenario-root", default="data/scenarios_v2")
    parser.add_argument("--victim", default=VICTIM)
    parser.add_argument("--judge", default=JUDGE)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--server-process", default="", help="label of the vLLM process (log file name)")
    args = parser.parse_args()
    {"policies-victim": cmd_policies_victim, "policies-judge": cmd_policies_judge,
     "fidelity-record": cmd_fidelity_record, "fidelity-replay": cmd_fidelity_replay,
     "fidelity-judge": cmd_fidelity_judge, "fidelity-score": cmd_fidelity_score}[args.stage](args)


if __name__ == "__main__":
    main()
