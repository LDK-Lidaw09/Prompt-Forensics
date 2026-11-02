#!/usr/bin/env python3
"""Build the real examples shown by the demo interface (data/demo/examples_v2.json.gz).

Takes ten scenarios of the version 2 runs (shard 0): one attack prompt of each
of the six families, chosen among those whose 100 replays mix safe and unsafe
verdicts, and four legitimate prompts (two standard and one resembling an
attack with no unsafe replay, and one standard prompt with unsafe replays,
which shows a false alarm of the Evaluator). For each, it keeps the prompt as
sent to Mistral-7B, the 100 replayed responses and the Evaluator verdict of
each response.

The raw runs are in a private dataset on the Hugging Face Hub; HF_TOKEN must
give read access. The files are streamed, nothing large is written to disk.

    HF_TOKEN=hf_... python scripts/build_demo_examples.py
"""

import argparse
import csv
import gzip
import json
import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prompt_forensics.scenarios import build_victim_prompt, load_scenarios, main_prompt, scenario_id  # noqa: E402

REPO = "Latavious/prompt-forensics-runs-v2"
SHARD = "shard0"
# (run mode, scenario id, source file): chosen with pools.parquet of the
# analysis (Mistral-7B, shard 0), random_state 7, see the module docstring.
EXAMPLES = [
    ("attacks", "AUDIT_BE_0493", "audit_scenarios.json"),
    ("attacks", "HEALTH_DL_0583", "health_scenarios.json"),
    ("attacks", "EDU_DD_0880", "education_scenarios.json"),
    ("attacks", "IP_JK_0731", "ip_scenarios.json"),
    ("attacks", "AUDIT_PI_0618", "audit_scenarios.json"),
    ("attacks", "HEALTH_RP_0372", "health_scenarios.json"),
    ("safe", "HEALTH_PI_0099", "health_scenarios.json"),
    ("safe", "IP_JK_S00034", "ip_scenarios.json"),
    ("safe", "AUDIT_DD_0273", "audit_scenarios.json"),
    ("safe", "HEALTH_DD_0305", "health_scenarios.json"),
]
EVALUATOR_FIELDS = ("verdict", "score", "risk_types", "analysis_summary")


def stream(path: str):
    from huggingface_hub import hf_hub_url

    url = hf_hub_url(REPO, path, repo_type="dataset")
    response = requests.get(url, headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}"},
                            stream=True, timeout=120)
    response.raise_for_status()
    # Lines are yielded with their line ending, as the csv module expects.
    with response:
        pending = b""
        for chunk in response.iter_content(chunk_size=1 << 20):
            pending += chunk
            *lines, pending = pending.split(b"\n")
            for line in lines:
                yield line.decode("utf-8") + "\n"
        if pending:
            yield pending.decode("utf-8")


def fetch(wanted: set[tuple[str, str, str]]) -> tuple[dict, dict]:
    replays: dict[tuple, dict[int, dict]] = {}
    for mode in ("attacks", "safe"):
        for line in stream(f"{SHARD}/replays_{mode}_r100.jsonl"):
            row = json.loads(line)
            key = (row.get("run_mode"), row.get("scenario_id"), row.get("source_file"))
            if key in wanted and (row.get("text") or "").strip():
                replays.setdefault(key, {})[int(row["replay_idx"])] = row
    verdicts: dict[tuple, dict[int, dict]] = {}
    csv.field_size_limit(10**9)
    for row in csv.DictReader(stream(f"{SHARD}/evaluator.csv")):
        if row["input_file"] not in ("replays_attacks_r100.jsonl", "replays_safe_r100.jsonl"):
            continue
        key = (row["run_mode"], row["scenario_id"], row["source_file"])
        if key in wanted and row["judge_ok"] == "True":  # last successful judgement wins
            verdicts.setdefault(key, {})[int(row["replay_idx"])] = {f: row[f] for f in EVALUATOR_FIELDS}
    return replays, verdicts


def build(replays: dict, verdicts: dict) -> list[dict]:
    scenarios = {(scenario_id(s), s["source_file"]): s for s in load_scenarios(ROOT / "data" / "scenarios_v2")}
    examples = []
    for mode, sid, source in EXAMPLES:
        scenario = scenarios[(sid, source)]
        pool = replays.get((mode, sid, source), {})
        judged = verdicts.get((mode, sid, source), {})
        first = pool[min(pool)] if pool else {}
        items = []
        for idx in sorted(pool):
            verdict = judged.get(idx, {})
            items.append({
                "replay_idx": idx, "seed": int(pool[idx]["seed"]), "text": pool[idx]["text"],
                "verdict": verdict.get("verdict", "not judged"),
                "score": float(verdict["score"]) if verdict.get("score") else None,
                "risk_types": verdict.get("risk_types", ""),
                "summary": verdict.get("analysis_summary", ""),
            })
        examples.append({
            "scenario_id": sid, "source_file": source, "run_mode": mode,
            "domain": scenario.get("domain"), "attack_type": scenario.get("attack_type"),
            "scenario_name": scenario.get("scenario_name"),
            "safe_kind": scenario.get("safe_kind") if mode == "safe" else None,
            "benchmark_source": scenario.get("benchmark_source") if mode == "attacks" else None,
            "system_prompt": scenario.get("system_prompt"), "document": scenario.get("dataset"),
            "prompt": main_prompt(scenario, mode), "model_input": build_victim_prompt(scenario, mode),
            "model": first.get("target_model"), "temperature": float(first.get("temperature", 0.2)),
            "max_tokens": int(first.get("max_tokens", 512)), "replays": items,
        })
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(ROOT / "data" / "demo" / "examples_v2.json.gz"))
    args = parser.parse_args()
    replays, verdicts = fetch(set(EXAMPLES))
    examples = build(replays, verdicts)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8") as handle:
        json.dump(examples, handle, ensure_ascii=False)
    for ex in examples:
        unsafe = sum(r["verdict"] in ("suspicious", "malicious") for r in ex["replays"])
        print(f"{ex['run_mode']:7s} {ex['scenario_id']:15s} {len(ex['replays'])} replays, {unsafe} unsafe")


if __name__ == "__main__":
    main()
