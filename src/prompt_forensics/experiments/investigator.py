"""Investigator evaluation over the replay dataset (thesis sections 5.4 and 6.2).

For every scenario in a replay JSONL file written by :mod:`.replays`, one
judge model investigates the scenario's prompt together with one replay
output (see :func:`~prompt_forensics.agents.investigator.pick_representative_replay`)
and one CSV row is written. The gold label comes from the run mode of the
file: ``attack`` for an attacks file, ``safe`` for a safe file.

Run it once per (judge, run mode); :func:`prompt_forensics.experiments.analysis.investigator_tables`
then combines the attacks and safe CSVs of each judge into Tables 6.4 to 6.6.
"""

from __future__ import annotations

import csv
import os
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterator

from ..agents.investigator import (
    DEFAULT_DATASET_MAX_CHARS,
    investigate_pair,
    pick_representative_replay,
)
from ..config import get_settings
from ..scenarios import load_scenarios, scenario_id
from .io import CsvWriter, JsonlWriter, read_jsonl

# Same columns as the original script, so its CSVs can be analysed and resumed.
CSV_FIELDS = [
    "scenario_id", "domain", "attack_type", "run_mode", "source_file",
    "target_model", "judge_model", "replay_idx",
    "gold_label",
    "decision", "maliciousness_score", "adversarial_types", "guardrail_violations",
    "risk_markers", "attack_succeeded", "summary",
    "correct",
    "raw_json", "raw_text", "judge_ok", "judge_error", "timing_ms", "created_at",
]
GOLD_LABEL = {"attacks": "attack", "safe": "safe"}


@dataclass
class InvestigatorRunConfig:
    """Parameters of one Investigator run. Defaults are the thesis settings."""

    replays: str                          # replay JSONL file (one run mode)
    output_csv: str
    judge_model: str
    output_jsonl: str | None = None
    scenario_root: str | None = None      # default: SCENARIO_ROOT
    temperature: float = 0.0
    max_tokens: int = 640
    seed: int = 777
    repetition_penalty: float = 1.3
    guided_json: bool = True
    guided_backend: str = "outlines"
    dataset_max_chars: int = DEFAULT_DATASET_MAX_CHARS
    workers: int = 16
    max_pending: int = 64
    max_retries: int = 4
    retry_sleep_s: float = 0.5
    limit: int | None = None
    resume: bool = False
    include_raw: bool = False


def _done_scenarios(path: str) -> set[str]:
    """Scenarios already investigated successfully (failed ones are redone)."""
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8", newline="") as handle:
        return {
            row["scenario_id"] for row in csv.DictReader(handle)
            if row.get("scenario_id") and row.get("judge_ok") in {"True", "true", "1"}
        }


def iter_pairs(cfg: InvestigatorRunConfig, scenarios: dict[str, dict], skip: set[str],
               progress: Callable[[str], None]) -> Iterator[tuple[dict, dict]]:
    """``(scenario, replay)`` pairs to investigate, in file order."""
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for _, _, obj in read_jsonl([cfg.replays]):
        if obj.get("scenario_id"):
            pools[str(obj["scenario_id"])].append(obj)

    emitted = 0
    for sid, replays in pools.items():
        if sid in skip:
            continue
        scenario = scenarios.get(sid)
        replay = pick_representative_replay(replays)
        if scenario is None or replay is None:
            progress(f"[skip] {sid}: " + ("scenario not found" if scenario is None else "no successful replay"))
            continue
        yield scenario, replay
        emitted += 1
        if cfg.limit is not None and emitted >= cfg.limit:
            return


def investigate_row(scenario: dict[str, Any], replay: dict[str, Any],
                    cfg: InvestigatorRunConfig) -> dict[str, Any]:
    """Investigate one pair and build its CSV row."""
    run_mode = replay.get("run_mode", "attacks")
    report, raw_text, timing_ms = investigate_pair(
        scenario, run_mode, replay.get("text", ""), cfg.judge_model,
        params={"temperature": cfg.temperature, "max_tokens": cfg.max_tokens,
                "seed": cfg.seed, "repetition_penalty": cfg.repetition_penalty},
        guided_json=cfg.guided_json, guided_backend=cfg.guided_backend,
        dataset_max_chars=cfg.dataset_max_chars,
        max_retries=cfg.max_retries, retry_sleep_s=cfg.retry_sleep_s,
    )
    gold = GOLD_LABEL[run_mode]
    fields = report.as_dict()
    return {
        **{k: replay.get(k, "") for k in ("scenario_id", "domain", "attack_type", "run_mode",
                                          "source_file", "target_model", "replay_idx")},
        "judge_model": cfg.judge_model,
        "gold_label": gold,
        **{k: fields[k] for k in ("decision", "maliciousness_score", "adversarial_types",
                                  "guardrail_violations", "risk_markers", "attack_succeeded",
                                  "summary")},
        # Empty when the judge failed: a failure is neither correct nor wrong.
        "correct": int(report.decision == gold) if report.ok else None,
        "raw_json": {k: v for k, v in fields.items() if k not in {"ok", "error"}},
        "raw_text": raw_text if cfg.include_raw else "",
        "judge_ok": report.ok,
        "judge_error": report.error or "",
        "timing_ms": timing_ms,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def run_investigator(cfg: InvestigatorRunConfig, progress: Callable[[str], None] = print) -> int:
    """Run the Investigator over the replay file. Returns the number of rows written."""
    root = cfg.scenario_root or get_settings().scenario_root
    scenarios = {scenario_id(s): s for s in load_scenarios(root)}
    skip = _done_scenarios(cfg.output_csv) if cfg.resume else set()
    pairs = iter_pairs(cfg, scenarios, skip, progress)
    progress(f"judge={cfg.judge_model} guided_json={cfg.guided_json} "
             f"{len(scenarios)} scenarios indexed, {len(skip)} already done")

    written, started = 0, time.time()
    jsonl = JsonlWriter(cfg.output_jsonl, append=cfg.resume) if cfg.output_jsonl else None
    try:
        with CsvWriter(cfg.output_csv, CSV_FIELDS, append=cfg.resume) as table, \
                ThreadPoolExecutor(max_workers=max(1, cfg.workers)) as pool:
            pending: set = set()

            def refill() -> None:
                # Bounded queue: memory stays flat on the 45,600-pair dataset.
                while len(pending) < max(1, cfg.max_pending):
                    pair = next(pairs, None)
                    if pair is None:
                        return
                    pending.add(pool.submit(investigate_row, *pair, cfg))

            refill()
            while pending:
                done, still_pending = wait(pending, return_when=FIRST_COMPLETED)
                pending.clear()
                pending.update(still_pending)
                for future in done:
                    row = future.result()
                    table.write(row)
                    if jsonl:
                        jsonl.write(row)
                    written += 1
                    if written % 50 == 0:
                        progress(f"{written} scenarios ({written / (time.time() - started):.2f}/s)")
                refill()
    finally:
        if jsonl:
            jsonl.close()
    progress(f"finished: {written} scenarios in {time.time() - started:.0f}s")
    return written
