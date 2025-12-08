# run_replay_dataset.py
from __future__ import annotations

import os
import json
import glob
import csv
from datetime import datetime
from typing import List, Dict, Any

import app as app_core  # app.py dans le même dossier


SCENARIO_ROOT = "scenarios"
SCENARIO_FILES = glob.glob(f"{SCENARIO_ROOT}/**/*.json", recursive=True)

TARGET_MODEL = os.getenv("LIVE_MODEL", "lmstudio:mistral-7b-instruct-v0.3")
TARGET_PARAMS = {
    "temperature": float(os.getenv("REPLAY_TEMPERATURE", 0.2)),
    "max_tokens": int(os.getenv("REPLAY_MAX_TOKENS", "512")),
    "seed": int(os.getenv("REPLAY_SEED", "1234")),
    "agent": "replay_robustness",
}


def normalize_attack_type(value: str | None) -> str:
    if not value:
        return ""
    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def load_scenarios(files: List[str]) -> List[Dict[str, Any]]:
    all_scenarios: List[Dict[str, Any]] = []
    for path in files:
        if not os.path.exists(path):
            print(f"[WARN] Scenario file not found: {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "scenarios" in data:
            scenarios = data["scenarios"]
        else:
            scenarios = data

        for s in scenarios:
            s = dict(s)
            s.setdefault("source_file", os.path.basename(path))
            all_scenarios.append(s)

    print(f"[INFO] Loaded {len(all_scenarios)} scenarios from {len(files)} files.")
    return all_scenarios


def _pick_main_prompt(s: Dict[str, Any], run_mode: str) -> str:
    if run_mode == "safe":
        return s.get("safe_prompt") or ""
    return s.get("attack_prompt") or s.get("prompt") or ""


def build_live_prompt(s: Dict[str, Any], run_mode: str = "attacks") -> str:
    parts: List[str] = []

    system_prompt = s.get("system_prompt") or ""
    if system_prompt:
        parts.append(system_prompt)

    context_keys = ["dataset", "case_info", "data_snippet", "context"]
    for k in context_keys:
        if s.get(k):
            parts.append(str(s[k]))
            break

    main_prompt = _pick_main_prompt(s, run_mode)
    if main_prompt:
        parts.append(main_prompt)

    return "\n\n".join(parts).strip()


def run_replays_for_scenario(
    s: Dict[str, Any],
    num_replays: int,
    run_mode: str,
    consistency_threshold: float,
    fidelity_threshold: float,
) -> Dict[str, Any]:
    scenario_id = s.get("id") or s.get("scenario_id") or "unknown_id"
    print(f"[INFO] Running {num_replays} replays for scenario {scenario_id} ({run_mode})")

    live_prompt = build_live_prompt(s, run_mode=run_mode)
    if not live_prompt:
        raise ValueError(f"Scenario {scenario_id} has empty main prompt for run_mode={run_mode}")

    outputs: List[str] = []
    for i in range(num_replays):
        try:
            res = app_core.call_llm(
                prompt=live_prompt,
                model=TARGET_MODEL,
                params=TARGET_PARAMS,
            )
            outputs.append(res.get("text", "") or "")
        except Exception as e:
            print(f"[ERROR] call_llm failed (replay {i + 1}/{num_replays}) for {scenario_id}: {e}")
            outputs.append("")

    robustness = app_core.compute_replay_robustness(
        outputs,
        consistency_threshold=consistency_threshold,
        fidelity_threshold=fidelity_threshold,
    )
    return robustness


def run_replay_dataset(
    run_mode: str = "attacks",          # "attacks" ou "safe"
    attack_filter: str | None = None,   # ex: "belief_elicitation"
    limit: int | None = None,           # max scénarios
    num_replays: int = 10,
    consistency_threshold: float = 0.9,
    fidelity_threshold: float = 0.9,
) -> str:
    """Fonction appelée depuis Streamlit. Retourne le chemin du CSV produit."""
    if run_mode not in {"attacks", "safe"}:
        raise ValueError(f"Invalid run_mode: {run_mode}")

    print(f"[INFO] TARGET_MODEL = {TARGET_MODEL}")
    print(f"[INFO] run_mode = {run_mode}, num_replays = {num_replays}")

    scenarios = load_scenarios(SCENARIO_FILES)
    if not scenarios:
        print("[ERROR] No scenarios loaded.")
        return ""

    if attack_filter:
        norm_filter = normalize_attack_type(attack_filter)
        before = len(scenarios)
        kept: List[Dict[str, Any]] = []
        for s in scenarios:
            norm_att = normalize_attack_type(s.get("attack_type", ""))
            if norm_filter == norm_att or norm_filter in norm_att:
                kept.append(s)
        scenarios = kept
        print(
            f"[INFO] Filter attack_type='{attack_filter}' "
            f"(normalized='{norm_filter}'): {before} → {len(scenarios)} scenarios."
        )

    if limit is not None and limit > 0:
        scenarios = scenarios[:limit]
        print(f"[INFO] Limiting to first {limit} scenarios.")

    if not scenarios:
        print("[WARN] No scenarios after filtering/limit.")
        return ""

    suffix = f"{run_mode}_replay"
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_csv = f"replay_robustness_{suffix}_{timestamp}.csv"

    fieldnames = [
        "scenario_id",
        "domain",
        "attack_type",
        "scenario_name",
        "source_file",
        "run_mode",
        "num_replays",
        "cons_score",
        "cons_std",
        "cons_threshold",
        "cons_count_above",
        "cons_num_pairs",
        "fid_score",
        "fid_std",
        "fid_threshold",
        "fid_count_above",
        "fid_num_replays",
        "cons_bert_available",
        "cons_bert_score",
        "cons_bert_std",
        "cons_bert_threshold",
        "cons_bert_count_above",
        "cons_bert_num_pairs",
        "fid_bert_available",
        "fid_bert_score",
        "fid_bert_std",
        "fid_bert_threshold",
        "fid_bert_count_above",
        "fid_bert_num_replays",
    ]

    with open(output_csv, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for s in scenarios:
            scenario_id = s.get("id") or s.get("scenario_id") or "unknown_id"
            domain = s.get("domain", "")
            attack_type = s.get("attack_type", "")
            scenario_name = s.get("scenario_name", "") or s.get("title", "")
            source_file = s.get("source_file", "")

            try:
                robust = run_replays_for_scenario(
                    s,
                    num_replays=num_replays,
                    run_mode=run_mode,
                    consistency_threshold=consistency_threshold,
                    fidelity_threshold=fidelity_threshold,
                )
            except Exception as e:
                print(f"[ERROR] Replay robustness failed for {scenario_id}: {e}")
                continue

            cons = robust.get("consistency", {}) or {}
            fid = robust.get("fidelity", {}) or {}
            cons_b = robust.get("consistency_bert", {}) or {}
            fid_b = robust.get("fidelity_bert", {}) or {}

            row = {
                "scenario_id": scenario_id,
                "domain": domain,
                "attack_type": attack_type,
                "scenario_name": scenario_name,
                "source_file": source_file,
                "run_mode": run_mode,
                "num_replays": robust.get("num_replays", num_replays),
                "cons_score": cons.get("score", 0.0),
                "cons_std": cons.get("std", 0.0),
                "cons_threshold": cons.get("threshold", consistency_threshold),
                "cons_count_above": cons.get("count_above", 0),
                "cons_num_pairs": cons.get("num_pairs", 0),
                "fid_score": fid.get("score", 0.0),
                "fid_std": fid.get("std", 0.0),
                "fid_threshold": fid.get("threshold", fidelity_threshold),
                "fid_count_above": fid.get("count_above", 0),
                "fid_num_replays": fid.get("num_replays", 0),
                "cons_bert_available": bool(cons_b.get("available", False)),
                "cons_bert_score": cons_b.get("score", 0.0),
                "cons_bert_std": cons_b.get("std", 0.0),
                "cons_bert_threshold": cons_b.get("threshold", consistency_threshold),
                "cons_bert_count_above": cons_b.get("count_above", 0),
                "cons_bert_num_pairs": cons_b.get("num_pairs", 0),
                "fid_bert_available": bool(fid_b.get("available", False)),
                "fid_bert_score": fid_b.get("score", 0.0),
                "fid_bert_std": fid_b.get("std", 0.0),
                "fid_bert_threshold": fid_b.get("threshold", fidelity_threshold),
                "fid_bert_count_above": fid_b.get("count_above", 0),
                "fid_bert_num_replays": fid_b.get("num_replays", 0),
            }
            writer.writerow(row)

    print(f"[INFO] Replay robustness finished. Results written to: {output_csv}")
    return output_csv
