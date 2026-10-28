"""Replay pools and their robustness metrics (thesis sections 4.3.1, 5.2 and 6.1).

For every selected scenario, the victim prompt is sent ``num_replays`` times
(10 in the thesis) with seeds ``seed_base + i`` (1234 + i), temperature 0.2 and
at most 512 new tokens. Two files are written:

- a **JSONL** file with one line per replay (the raw outputs, with their seed
  and model, or the error if the call failed). This is the evidence the other
  experiments (Evaluator, Investigator) read.
- a **CSV** file with one row per scenario: Consistency and Fidelity with
  embedding cosine and with BERTScore, their standard deviations and the
  number of pairs/replays above the similarity threshold.
- a **pairs JSONL** file (``output_pairs``, by default the JSONL name with
  ``_pairs``) with one line per scenario: the raw cosine and the BERTScore F1
  of every pair of replays, in the order (0, 1), (0, 2), ..., (n-2, n-1) of
  ``indices``. The pairs with replay 0 give Fidelity. Chapter 6 needs these
  values for the analyses that compare pairs (same Evaluator verdict or not,
  sensitivity to the threshold), which the CSV summaries cannot answer.

A pool with failed replays is still written, with ``complete = False`` and the
failed indices, so that the analysis can exclude it explicitly. With
``resume=True`` successful replays already in the JSONL are reused and complete
scenarios already in the CSV are skipped.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from ..config import get_settings
from ..llm import LLMError, call_llm
from ..metrics.robustness import compute_replay_robustness
from ..scenarios import build_victim_prompt, load_scenarios, scenario_id, select_scenarios
from .io import CsvWriter, JsonlWriter, read_jsonl

CSV_FIELDS = [
    # Scenario
    "scenario_id", "domain", "attack_type", "scenario_name", "source_file", "run_mode",
    # Generation
    "num_requested", "num_replays", "missing_indices", "complete",
    "seed_base", "target_model", "temperature", "max_tokens", "outputs_jsonl",
    # Embedding cosine (see cosine_mapping)
    "cosine_mapping",
    "cons_score", "cons_std", "cons_raw_cosine_mean", "cons_threshold",
    "cons_count_above", "cons_num_pairs", "cons_min",
    "fid_score", "fid_std", "fid_raw_cosine_mean", "fid_threshold",
    "fid_count_above", "fid_num_replays", "fid_min",
    # BERTScore F1
    "cons_bert_available", "cons_bert_score", "cons_bert_std", "cons_bert_threshold",
    "cons_bert_count_above", "cons_bert_num_pairs", "cons_bert_min",
    "fid_bert_available", "fid_bert_score", "fid_bert_std", "fid_bert_threshold",
    "fid_bert_count_above", "fid_bert_num_replays", "fid_bert_min",
    "bert_error",
]


@dataclass
class ReplayConfig:
    """Parameters of one replay run. Defaults are the thesis settings."""

    run_mode: str = "attacks"
    attack_filter: str | None = None
    limit: int | None = None
    split: str | None = None               # "dev" or "test" (dataset v2 only)
    num_replays: int = 10
    target_model: str | None = None        # default: LIVE_MODEL
    temperature: float | None = None       # default: LIVE_TEMPERATURE (0.2)
    max_tokens: int | None = None          # default: LIVE_MAX_TOKENS (512)
    seed_base: int | None = None           # default: LIVE_SEED (1234)
    threshold: float = 0.9
    cosine_mapping: str = "shifted"
    use_bertscore: bool = True
    scenario_root: str | None = None       # default: SCENARIO_ROOT
    scenario_workers: int = 1
    replay_workers: int = 10
    max_inflight_calls: int = 8
    resume: bool = False
    output_csv: str | None = None
    output_jsonl: str | None = None
    output_pairs: str | None = None

    def resolved(self) -> "ReplayConfig":
        """Copy with every ``None`` default filled from the settings."""
        s = get_settings()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_jsonl = self.output_jsonl or f"results/replay_outputs_{self.run_mode}_{stamp}.jsonl"
        return ReplayConfig(
            **{
                **self.__dict__,
                "target_model": self.target_model or s.live_model,
                "temperature": s.live_temperature if self.temperature is None else self.temperature,
                "max_tokens": s.live_max_tokens if self.max_tokens is None else self.max_tokens,
                "seed_base": s.seed_base if self.seed_base is None else self.seed_base,
                "scenario_root": self.scenario_root or s.scenario_root,
                "output_csv": self.output_csv
                or f"results/replay_robustness_{self.run_mode}_{stamp}.csv",
                "output_jsonl": output_jsonl,
                "output_pairs": self.output_pairs or _pairs_path(output_jsonl),
            }
        )


def _pairs_path(output_jsonl: str) -> str:
    root, ext = os.path.splitext(output_jsonl)
    return f"{root}_pairs{ext or '.jsonl'}"


def _pairs_row(scenario: dict, robust: dict, cfg: ReplayConfig) -> dict:
    """Pair similarities of one pool, rounded to 4 decimals to keep the file small."""
    cons, cons_b = robust["consistency"], robust["consistency_bert"]
    pairs = cons.get("pairwise", [])
    indices = sorted({d["i"] for d in pairs} | {d["j"] for d in pairs})
    return {
        "scenario_id": scenario_id(scenario),
        "source_file": scenario.get("source_file", ""),
        "run_mode": cfg.run_mode,
        "target_model": cfg.target_model,
        "indices": indices,
        "cosine": [round(d["raw_cosine"], 4) for d in pairs],
        "bert_f1": [round(d["f1"], 4) for d in cons_b.get("pairwise", [])],
    }


def _replay_key(sid: str, source: str, run_mode: str, model: str, idx: int, seed: int) -> tuple:
    return (sid, source, run_mode, model, int(idx), int(seed))


def load_existing_replays(cfg: ReplayConfig) -> dict[tuple, str]:
    """Successful replays already present in the JSONL (for ``resume``)."""
    if not os.path.exists(cfg.output_jsonl):
        return {}
    existing = {}
    for _, _, obj in read_jsonl([cfg.output_jsonl]):
        if obj.get("error") or obj.get("run_mode") != cfg.run_mode:
            continue
        if obj.get("target_model") != cfg.target_model or not (obj.get("text") or "").strip():
            continue
        key = _replay_key(obj["scenario_id"], obj.get("source_file", ""), cfg.run_mode,
                          cfg.target_model, obj["replay_idx"], obj["seed"])
        existing[key] = obj["text"]
    return existing


def load_completed_scenarios(cfg: ReplayConfig) -> set[tuple[str, str]]:
    """``(scenario_id, source_file)`` of complete rows already in the CSV."""
    import csv

    if not os.path.exists(cfg.output_csv):
        return set()
    done = set()
    with open(cfg.output_csv, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if (row.get("run_mode") == cfg.run_mode
                    and row.get("target_model") == cfg.target_model
                    and row.get("num_requested") == str(cfg.num_replays)
                    and row.get("complete") == "True"):
                done.add((row["scenario_id"], row.get("source_file", "")))
    return done


def _run_one_replay(scenario: dict, prompt: str, idx: int, cfg: ReplayConfig,
                    jsonl: JsonlWriter, semaphore: threading.Semaphore,
                    existing: dict[tuple, str]) -> str | None:
    """Generate (or reuse) replay ``idx``. Returns its text, or None on failure."""
    sid, source = scenario_id(scenario), scenario.get("source_file", "")
    seed = cfg.seed_base + idx
    key = _replay_key(sid, source, cfg.run_mode, cfg.target_model, idx, seed)
    if key in existing:
        return existing[key]

    record = {
        "scenario_id": sid,
        "domain": scenario.get("domain", ""),
        "attack_type": scenario.get("attack_type", ""),
        "scenario_name": scenario.get("scenario_name", ""),
        "source_file": source,
        "run_mode": cfg.run_mode,
        "num_replays": cfg.num_replays,
        "replay_idx": idx,
        "seed": seed,
        "target_model": cfg.target_model,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
    }
    try:
        with semaphore:
            response = call_llm(prompt, cfg.target_model, {
                "temperature": cfg.temperature, "max_tokens": cfg.max_tokens, "seed": seed,
            })
    except LLMError as exc:
        jsonl.write({**record, "text": "", "error": str(exc)})
        return None
    jsonl.write({**record, "text": response.text, "finish_reason": response.finish_reason,
                 "usage": response.usage, "timing_ms": response.timing_ms})
    return response.text


def _result_row(scenario: dict, robust: dict, cfg: ReplayConfig) -> dict[str, Any]:
    cons, fid = robust["consistency"], robust["fidelity"]
    cons_b, fid_b = robust["consistency_bert"], robust["fidelity_bert"]
    return {
        "scenario_id": scenario_id(scenario),
        "domain": scenario.get("domain", ""),
        "attack_type": scenario.get("attack_type", ""),
        "scenario_name": scenario.get("scenario_name", ""),
        "source_file": scenario.get("source_file", ""),
        "run_mode": cfg.run_mode,
        "num_requested": cfg.num_replays,
        "num_replays": robust["num_replays"],
        "missing_indices": robust["missing_indices"],
        "complete": robust["complete"],
        "seed_base": cfg.seed_base,
        "target_model": cfg.target_model,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "outputs_jsonl": cfg.output_jsonl,
        "cosine_mapping": robust["cosine_mapping"],
        "cons_score": cons["score"], "cons_std": cons["std"],
        "cons_raw_cosine_mean": cons["raw_cosine_mean"], "cons_threshold": cons["threshold"],
        "cons_count_above": cons["count_above"], "cons_num_pairs": cons["num_pairs"],
        "cons_min": cons["min"],
        "fid_score": fid["score"], "fid_std": fid["std"],
        "fid_raw_cosine_mean": fid["raw_cosine_mean"], "fid_threshold": fid["threshold"],
        "fid_count_above": fid["count_above"], "fid_num_replays": fid["num_replays"],
        "fid_min": fid["min"],
        "cons_bert_available": cons_b["available"], "cons_bert_score": cons_b["score"],
        "cons_bert_std": cons_b["std"], "cons_bert_threshold": cons_b["threshold"],
        "cons_bert_count_above": cons_b["count_above"], "cons_bert_num_pairs": cons_b["num_pairs"],
        "cons_bert_min": cons_b["min"],
        "fid_bert_available": fid_b["available"], "fid_bert_score": fid_b["score"],
        "fid_bert_std": fid_b["std"], "fid_bert_threshold": fid_b["threshold"],
        "fid_bert_count_above": fid_b["count_above"], "fid_bert_num_replays": fid_b["num_replays"],
        "fid_bert_min": fid_b["min"],
        "bert_error": cons_b["error"],
    }


def run_replays(cfg: ReplayConfig, progress: Callable[[str], None] = print) -> tuple[str, str]:
    """Run the replay experiment. Returns ``(csv_path, jsonl_path)``."""
    cfg = cfg.resolved()
    scenarios = select_scenarios(load_scenarios(cfg.scenario_root), cfg.run_mode,
                                 cfg.attack_filter, cfg.limit, cfg.split)
    if not scenarios:
        raise ValueError("no scenario matches the selection")

    existing = load_existing_replays(cfg) if cfg.resume else {}
    completed = load_completed_scenarios(cfg) if cfg.resume else set()
    todo = [s for s in scenarios if (scenario_id(s), s.get("source_file", "")) not in completed]
    progress(f"{len(scenarios)} scenarios selected, {len(todo)} to run, "
             f"{len(existing)} replays reused; model={cfg.target_model}")

    semaphore = threading.Semaphore(max(1, cfg.max_inflight_calls))
    # The embedding and BERTScore models are shared and not thread-safe.
    metrics_lock = threading.Lock()

    with JsonlWriter(cfg.output_jsonl) as jsonl, JsonlWriter(cfg.output_pairs) as pair_file, \
            CsvWriter(cfg.output_csv, CSV_FIELDS, append=cfg.resume) as table:

        def process(scenario: dict) -> None:
            prompt = build_victim_prompt(scenario, cfg.run_mode)
            outputs: list[str | None] = [None] * cfg.num_replays
            with ThreadPoolExecutor(max_workers=max(1, min(cfg.replay_workers, cfg.num_replays))) as pool:
                futures = {pool.submit(_run_one_replay, scenario, prompt, i, cfg, jsonl,
                                       semaphore, existing): i for i in range(cfg.num_replays)}
                for future in as_completed(futures):
                    outputs[futures[future]] = future.result()
            if all(text is None for text in outputs):
                # Every call failed: the model server is most likely down. Stop
                # instead of filling the table with empty pools.
                raise RuntimeError(f"all {cfg.num_replays} replays of {scenario_id(scenario)} failed; "
                                   f"is {cfg.target_model} still served? (see the error field in "
                                   f"{cfg.output_jsonl})")
            with metrics_lock:
                robust = compute_replay_robustness(
                    outputs, cfg.threshold, cfg.threshold,
                    cosine_mapping=cfg.cosine_mapping, use_bertscore=cfg.use_bertscore,
                )
            # Pairs first: a row in the CSV means the pool is fully written.
            pair_file.write(_pairs_row(scenario, robust, cfg))
            table.write(_result_row(scenario, robust, cfg))

        with ThreadPoolExecutor(max_workers=max(1, cfg.scenario_workers)) as pool:
            futures = [pool.submit(process, s) for s in todo]
            for done, future in enumerate(as_completed(futures), start=1):
                future.result()  # re-raise unexpected errors instead of hiding them
                progress(f"{done}/{len(todo)} scenarios done")

    return cfg.output_csv, cfg.output_jsonl
