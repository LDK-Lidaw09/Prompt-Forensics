# run_replay_dataset.py
# ------------------------------------------------------------
#
# - Filtrage strict selon run_mode :
#     attacks -> uniquement les scénarios avec attack_prompt
#     safe    -> uniquement les scénarios avec safe_prompt
# - 10 replays par prompt
# - Replays parallèles par scénario
# - Scénarios parallèles contrôlés
# - Limite globale d'appels LLM simultanés
# - Chargement JSON parallèle
# - JSONL thread-safe avec flush immédiat
# - CSV thread-safe avec flush progressif
# - Resume possible depuis JSONL/CSV si output paths fixes + resume=True
# - Calcul des métriques protégé par un lock pour éviter les problèmes
#   de BERTScore / embeddings en parallèle
# ------------------------------------------------------------

from __future__ import annotations

import os
import json
import glob
import csv
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple, Optional

import app as app_core  # app.py dans le même dossier


# ------------------------------------------------------------
# Configuration globale
# ------------------------------------------------------------

SCENARIO_ROOT = os.getenv("SCENARIO_ROOT", "scenarios")
SCENARIO_FILES = glob.glob(f"{SCENARIO_ROOT}/**/*.json", recursive=True)

TARGET_MODEL = os.getenv("LIVE_MODEL", "lmstudio:mistral-7b-instruct-v0.3")
TARGET_PARAMS = {
    "temperature": float(os.getenv("REPLAY_TEMPERATURE", 0.2)),
    "max_tokens": int(os.getenv("REPLAY_MAX_TOKENS", "512")),
    "seed": int(os.getenv("REPLAY_SEED", "1234")),
    "agent": "replay_robustness",
}

DEFAULT_SCENARIO_WORKERS = int(os.getenv("SCENARIO_WORKERS", "1"))
DEFAULT_REPLAY_WORKERS = int(os.getenv("REPLAY_WORKERS", "10"))
DEFAULT_MAX_INFLIGHT_LLM_CALLS = int(os.getenv("MAX_INFLIGHT_LLM_CALLS", "8"))
DEFAULT_JSON_LOAD_WORKERS = int(os.getenv("JSON_LOAD_WORKERS", "8"))


class JsonlWriter:
    """Écriture JSONL thread-safe avec flush immédiat."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._f = open(path, "a", encoding="utf-8", buffering=1)

    def write(self, obj: Dict[str, Any]) -> None:
        line = json.dumps(obj, ensure_ascii=False)
        with self._lock:
            self._f.write(line + "\n")
            self._f.flush()

    def close(self) -> None:
        with self._lock:
            self._f.flush()
            self._f.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class CsvWriter:
    """Écriture CSV thread-safe avec flush progressif."""

    def __init__(self, path: str, fieldnames: List[str], append: bool = False):
        self.path = path
        self.fieldnames = fieldnames
        self._lock = threading.Lock()

        file_exists = os.path.exists(path) and os.path.getsize(path) > 0
        mode = "a" if append else "w"
        self._f = open(path, mode, encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._f, fieldnames=fieldnames)

        if not append or not file_exists:
            self._writer.writeheader()
            self._f.flush()

    def write_row(self, row: Dict[str, Any]) -> None:
        clean_row = {k: row.get(k, "") for k in self.fieldnames}
        with self._lock:
            self._writer.writerow(clean_row)
            self._f.flush()

    def close(self) -> None:
        with self._lock:
            self._f.flush()
            self._f.close()

    def __enter__(self) -> "CsvWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def normalize_attack_type(value: str | None) -> str:
    if not value:
        return ""
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def _load_one_scenario_file(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        print(f"[WARN] Scenario file not found: {path}")
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[ERROR] Cannot load scenario file {path}: {e}")
        return []

    if isinstance(data, dict) and "scenarios" in data:
        scenarios = data["scenarios"]
    else:
        scenarios = data

    if not isinstance(scenarios, list):
        print(f"[WARN] Scenario file has unsupported structure: {path}")
        return []

    loaded: List[Dict[str, Any]] = []
    for s in scenarios:
        if not isinstance(s, dict):
            continue
        s = dict(s)
        s.setdefault("source_file", os.path.basename(path))
        loaded.append(s)

    return loaded


def load_scenarios(files: List[str], json_load_workers: int = DEFAULT_JSON_LOAD_WORKERS) -> List[Dict[str, Any]]:
    """Chargement parallèle des JSON de scénarios."""
    if not files:
        return []

    all_scenarios: List[Dict[str, Any]] = []
    max_workers = max(1, min(int(json_load_workers), len(files)))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_load_one_scenario_file, path) for path in files]
        for future in as_completed(futures):
            all_scenarios.extend(future.result())

    print(f"[INFO] Loaded {len(all_scenarios)} scenarios from {len(files)} files.")
    return all_scenarios


def filter_scenarios_by_run_mode(scenarios: List[Dict[str, Any]], run_mode: str) -> List[Dict[str, Any]]:
    before = len(scenarios)

    if run_mode == "attacks":
        filtered = [s for s in scenarios if s.get("attack_prompt")]
    elif run_mode == "safe":
        filtered = [s for s in scenarios if s.get("safe_prompt")]
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}")

    print(f"[INFO] Run mode filter '{run_mode}': {before} → {len(filtered)} scenarios.")
    return filtered


def _pick_main_prompt(s: Dict[str, Any], run_mode: str) -> str:
    if run_mode == "safe":
        return s.get("safe_prompt") or ""
    if run_mode == "attacks":
        return s.get("attack_prompt") or ""
    return ""


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


def _scenario_id(s: Dict[str, Any]) -> str:
    return str(s.get("id") or s.get("scenario_id") or "unknown_id")


def _scenario_key(s: Dict[str, Any], run_mode: str, num_replays: int) -> Tuple[str, str, str, str, int]:
    return (
        _scenario_id(s),
        str(s.get("source_file", "")),
        str(run_mode),
        str(TARGET_MODEL),
        int(num_replays),
    )


def _replay_key(
    scenario_id: str,
    source_file: str,
    run_mode: str,
    replay_idx: int,
    seed: int,
) -> Tuple[str, str, str, str, int, int]:
    return (
        str(scenario_id),
        str(source_file or ""),
        str(run_mode),
        str(TARGET_MODEL),
        int(replay_idx),
        int(seed),
    )


def load_existing_replays(jsonl_path: str, run_mode: str) -> Dict[Tuple[str, str, str, str, int, int], str]:
    existing: Dict[Tuple[str, str, str, str, int, int], str] = {}
    if not jsonl_path or not os.path.exists(jsonl_path):
        return existing

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            if obj.get("error"):
                continue
            if obj.get("run_mode") != run_mode:
                continue
            if obj.get("target_model") != TARGET_MODEL:
                continue
            if "text" not in obj:
                continue

            try:
                key = _replay_key(
                    scenario_id=str(obj.get("scenario_id", "unknown_id")),
                    source_file=str(obj.get("source_file", "")),
                    run_mode=str(obj.get("run_mode", run_mode)),
                    replay_idx=int(obj.get("replay_idx")),
                    seed=int(obj.get("seed")),
                )
                text = obj.get("text", "") or ""
                if text.strip():
                    existing[key] = text
            except Exception:
                continue

    print(f"[INFO] Resume: loaded {len(existing)} existing successful replays from {jsonl_path}")
    return existing


def load_completed_csv_keys(csv_path: str, run_mode: str, num_replays: int) -> set[Tuple[str, str, str, str, int]]:
    completed: set[Tuple[str, str, str, str, int]] = set()
    if not csv_path or not os.path.exists(csv_path):
        return completed

    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("run_mode") != run_mode:
                    continue
                if row.get("target_model") != TARGET_MODEL:
                    continue

                try:
                    if int(row.get("num_replays", 0)) != int(num_replays):
                        continue
                except Exception:
                    continue

                completed.add(
                    (
                        str(row.get("scenario_id", "unknown_id")),
                        str(row.get("source_file", "")),
                        str(row.get("run_mode", run_mode)),
                        str(row.get("target_model", TARGET_MODEL)),
                        int(num_replays),
                    )
                )
    except Exception as e:
        print(f"[WARN] Could not read existing CSV for resume: {e}")

    print(f"[INFO] Resume: loaded {len(completed)} completed scenario rows from {csv_path}")
    return completed


def _run_one_replay(
    *,
    s: Dict[str, Any],
    live_prompt: str,
    run_mode: str,
    num_replays: int,
    replay_idx: int,
    seed_base: int,
    jsonl_writer: JsonlWriter,
    llm_semaphore: threading.Semaphore,
    existing_replays: Optional[Dict[Tuple[str, str, str, str, int, int], str]] = None,
) -> Tuple[int, str]:
    scenario_id = _scenario_id(s)
    source_file = str(s.get("source_file", ""))

    params = dict(TARGET_PARAMS)
    params["seed"] = int(seed_base) + int(replay_idx)

    key = _replay_key(
        scenario_id=scenario_id,
        source_file=source_file,
        run_mode=run_mode,
        replay_idx=replay_idx,
        seed=params["seed"],
    )

    if existing_replays and key in existing_replays:
        return replay_idx, existing_replays[key]

    try:
        with llm_semaphore:
            res = app_core.call_llm(
                prompt=live_prompt,
                model=TARGET_MODEL,
                params=params,
            )

        text = res.get("text", "") or ""
        jsonl_writer.write(
            {
                "scenario_id": scenario_id,
                "domain": s.get("domain", ""),
                "attack_type": s.get("attack_type", ""),
                "scenario_name": s.get("scenario_name", "") or s.get("title", ""),
                "source_file": source_file,
                "run_mode": run_mode,
                "num_replays": num_replays,
                "replay_idx": replay_idx,
                "seed": params["seed"],
                "target_model": TARGET_MODEL,
                "temperature": params.get("temperature"),
                "max_tokens": params.get("max_tokens"),
                "agent": params.get("agent", "replay_robustness"),
                "text": text,
            }
        )
        return replay_idx, text

    except Exception as e:
        print(f"[ERROR] call_llm failed (replay {replay_idx + 1}/{num_replays}) for {scenario_id}: {e}")
        jsonl_writer.write(
            {
                "scenario_id": scenario_id,
                "domain": s.get("domain", ""),
                "attack_type": s.get("attack_type", ""),
                "scenario_name": s.get("scenario_name", "") or s.get("title", ""),
                "source_file": source_file,
                "run_mode": run_mode,
                "num_replays": num_replays,
                "replay_idx": replay_idx,
                "seed": params["seed"],
                "target_model": TARGET_MODEL,
                "temperature": params.get("temperature"),
                "max_tokens": params.get("max_tokens"),
                "agent": params.get("agent", "replay_robustness"),
                "error": str(e),
                "text": "",
            }
        )
        return replay_idx, ""


def run_replays_for_scenario(
    s: Dict[str, Any],
    num_replays: int,
    run_mode: str,
    consistency_threshold: float,
    fidelity_threshold: float,
    jsonl_writer: JsonlWriter,
    seed_base: int,
    replay_workers: int,
    llm_semaphore: threading.Semaphore,
    metrics_lock: threading.Lock,
    existing_replays: Optional[Dict[Tuple[str, str, str, str, int, int], str]] = None,
) -> Dict[str, Any]:
    scenario_id = _scenario_id(s)
    print(f"[INFO] Running {num_replays} replays for scenario {scenario_id} ({run_mode})")

    live_prompt = build_live_prompt(s, run_mode=run_mode)
    if not live_prompt:
        raise ValueError(f"Scenario {scenario_id} has empty main prompt for run_mode={run_mode}")

    outputs_by_idx: List[str] = [""] * num_replays
    max_workers = max(1, min(int(replay_workers), int(num_replays)))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _run_one_replay,
                s=s,
                live_prompt=live_prompt,
                run_mode=run_mode,
                num_replays=num_replays,
                replay_idx=i,
                seed_base=seed_base,
                jsonl_writer=jsonl_writer,
                llm_semaphore=llm_semaphore,
                existing_replays=existing_replays,
            )
            for i in range(num_replays)
        ]

        for future in as_completed(futures):
            idx, text = future.result()
            outputs_by_idx[idx] = text

    with metrics_lock:
        robustness = app_core.compute_replay_robustness(
            outputs_by_idx,
            consistency_threshold=consistency_threshold,
            fidelity_threshold=fidelity_threshold,
        )

    return robustness


def _build_result_row(
    s: Dict[str, Any],
    robust: Dict[str, Any],
    run_mode: str,
    num_replays: int,
    seed_base: int,
    output_jsonl: str,
    consistency_threshold: float,
    fidelity_threshold: float,
) -> Dict[str, Any]:
    scenario_id = _scenario_id(s)
    cons = robust.get("consistency", {}) or {}
    fid = robust.get("fidelity", {}) or {}
    cons_b = robust.get("consistency_bert", {}) or {}
    fid_b = robust.get("fidelity_bert", {}) or {}

    return {
        "scenario_id": scenario_id,
        "domain": s.get("domain", ""),
        "attack_type": s.get("attack_type", ""),
        "scenario_name": s.get("scenario_name", "") or s.get("title", ""),
        "source_file": s.get("source_file", ""),
        "run_mode": run_mode,
        "num_replays": robust.get("num_replays", num_replays),
        "seed_base": seed_base,
        "target_model": TARGET_MODEL,
        "temperature": TARGET_PARAMS.get("temperature", 0.2),
        "max_tokens": TARGET_PARAMS.get("max_tokens", 512),
        "outputs_jsonl": output_jsonl,
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


def run_replay_dataset(
    run_mode: str = "attacks",
    attack_filter: str | None = None,
    limit: int | None = None,
    num_replays: int = 10,
    consistency_threshold: float = 0.9,
    fidelity_threshold: float = 0.9,
    scenario_workers: int | None = None,
    replay_workers: int | None = None,
    max_inflight_llm_calls: int | None = None,
    json_load_workers: int | None = None,
    resume: bool = False,
    output_csv: str | None = None,
    output_jsonl: str | None = None,
) -> Tuple[str, str]:
    """
    Fonction appelée depuis Streamlit ou depuis le shell.
    Retourne (csv_path, jsonl_path).
    """
    if run_mode not in {"attacks", "safe"}:
        raise ValueError(f"Invalid run_mode: {run_mode}")

    seed_base = int(TARGET_PARAMS.get("seed", 1234))
    scenario_workers = int(scenario_workers or DEFAULT_SCENARIO_WORKERS)
    replay_workers = int(replay_workers or DEFAULT_REPLAY_WORKERS)
    max_inflight_llm_calls = int(max_inflight_llm_calls or DEFAULT_MAX_INFLIGHT_LLM_CALLS)
    json_load_workers = int(json_load_workers or DEFAULT_JSON_LOAD_WORKERS)

    print(f"[INFO] TARGET_MODEL = {TARGET_MODEL}")
    print(f"[INFO] TARGET_PARAMS = {TARGET_PARAMS}")
    print(f"[INFO] run_mode = {run_mode}, num_replays = {num_replays}, seed_base = {seed_base}")
    print(
        "[INFO] Parallelism: "
        f"scenario_workers={scenario_workers}, replay_workers={replay_workers}, "
        f"max_inflight_llm_calls={max_inflight_llm_calls}, json_load_workers={json_load_workers}"
    )

    scenarios = load_scenarios(SCENARIO_FILES, json_load_workers=json_load_workers)
    if not scenarios:
        print("[ERROR] No scenarios loaded.")
        return "", ""

    scenarios = filter_scenarios_by_run_mode(scenarios, run_mode)

    if attack_filter:
        norm_filter = normalize_attack_type(attack_filter)
        before = len(scenarios)
        scenarios = [
            s for s in scenarios
            if norm_filter == normalize_attack_type(s.get("attack_type", ""))
            or norm_filter in normalize_attack_type(s.get("attack_type", ""))
        ]
        print(
            f"[INFO] Filter attack_type='{attack_filter}' "
            f"(normalized='{norm_filter}'): {before} → {len(scenarios)} scenarios."
        )

    if limit is not None and limit > 0:
        scenarios = scenarios[:limit]
        print(f"[INFO] Limiting to first {limit} scenarios.")

    if not scenarios:
        print("[WARN] No scenarios after filtering/limit.")
        return "", ""

    suffix = f"{run_mode}_replay"
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    if output_csv is None:
        output_csv = f"replay_robustness_{suffix}_{timestamp}.csv"
    if output_jsonl is None:
        output_jsonl = f"replay_outputs_{suffix}_{timestamp}.jsonl"

    fieldnames = [
        "scenario_id",
        "domain",
        "attack_type",
        "scenario_name",
        "source_file",
        "run_mode",
        "num_replays",
        "seed_base",
        "target_model",
        "temperature",
        "max_tokens",
        "outputs_jsonl",
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

    existing_replays = load_existing_replays(output_jsonl, run_mode) if resume else {}
    completed_csv_keys = load_completed_csv_keys(output_csv, run_mode, num_replays) if resume else set()

    llm_semaphore = threading.Semaphore(max(1, max_inflight_llm_calls))
    metrics_lock = threading.Lock()

    def process_one_scenario(s: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        scenario_id = _scenario_id(s)
        key = _scenario_key(s, run_mode, num_replays)

        if resume and key in completed_csv_keys:
            print(f"[INFO] Resume: skipping completed scenario {scenario_id}")
            return None

        try:
            robust = run_replays_for_scenario(
                s,
                num_replays=num_replays,
                run_mode=run_mode,
                consistency_threshold=consistency_threshold,
                fidelity_threshold=fidelity_threshold,
                jsonl_writer=jsonl_writer,
                seed_base=seed_base,
                replay_workers=replay_workers,
                llm_semaphore=llm_semaphore,
                metrics_lock=metrics_lock,
                existing_replays=existing_replays,
            )
            return _build_result_row(
                s=s,
                robust=robust,
                run_mode=run_mode,
                num_replays=num_replays,
                seed_base=seed_base,
                output_jsonl=output_jsonl,
                consistency_threshold=consistency_threshold,
                fidelity_threshold=fidelity_threshold,
            )
        except Exception as e:
            print(f"[ERROR] Replay robustness failed for {scenario_id}: {e}")
            return None

    append_csv = bool(resume and os.path.exists(output_csv))

    with JsonlWriter(output_jsonl) as jsonl_writer, CsvWriter(output_csv, fieldnames, append=append_csv) as csv_writer:
        max_scenario_workers = max(1, min(int(scenario_workers), len(scenarios)))

        with ThreadPoolExecutor(max_workers=max_scenario_workers) as executor:
            futures = [executor.submit(process_one_scenario, s) for s in scenarios]

            completed = 0
            total = len(futures)
            for future in as_completed(futures):
                row = future.result()
                completed += 1

                if row is not None:
                    csv_writer.write_row(row)

                print(f"[INFO] Progress: {completed}/{total} scenarios processed")

    print(f"[INFO] Replay robustness finished. Results written to: {output_csv}")
    print(f"[INFO] Replay outputs JSONL written to: {output_jsonl}")
    return output_csv, output_jsonl