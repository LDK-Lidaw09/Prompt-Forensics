"""Evaluator Agent over replay outputs (thesis sections 5.3 and 6.4).

Reads the replay JSONL files written by :mod:`.replays` and judges every
successful replay output with :func:`prompt_forensics.agents.evaluator.evaluate_output`.
One CSV row (and optionally one detailed JSONL line) is written per output.

The tail-risk analysis then groups these rows by scenario: a pool is unsafe
if at least one of its replays is unsafe (see :mod:`.analysis`).

Judging is **blind** by default (no scenario metadata in the prompt). Use
``include_metadata=True`` to reproduce the prompt used for the thesis runs.
"""

from __future__ import annotations

import csv
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterator

from ..agents.evaluator import METADATA_FIELDS, evaluate_output
from ..canonical import sha256_hex
from ..config import get_settings
from .io import CsvWriter, JsonlWriter, read_jsonl

CSV_FIELDS = [
    "input_file", "line_number", "output_key", "scenario_id", "source_file", "domain",
    "attack_type", "scenario_name", "run_mode", "replay_idx", "seed", "target_model",
    "temperature", "max_tokens", "judge_model", "judge_temperature", "judge_max_tokens",
    "judge_blind", "verdict", "malicious", "unsafe", "score", "risk_types", "issues",
    "analysis_summary", "output_hash", "output_chars", "output_preview", "raw_judge_json",
    "raw_judge_text",
    "judge_ok", "judge_error", "timing_ms", "created_at",
]


@dataclass
class EvaluatorConfig:
    """Parameters of one Evaluator run. Defaults are the thesis settings."""

    inputs: list[str]
    output_csv: str
    output_jsonl: str | None = None
    judge_model: str | None = None      # default: JUDGE_MODEL (Qwen/Qwen3-14B)
    temperature: float = 0.0
    max_tokens: int = 512
    seed: int = 777                     # the replay index is added to it
    include_metadata: bool = False      # True reproduces the original (non-blind) prompt
    guided_json: bool = True
    guided_backend: str = "outlines"
    workers: int = 2
    max_pending: int | None = None
    limit: int | None = None
    run_mode: str | None = None         # keep only "attacks" or "safe" outputs
    resume: bool = False
    max_retries: int = 2
    retry_sleep_s: float = 2.0
    preview_chars: int = 500
    include_output_text: bool = False
    include_raw_judge_text: bool = False


def output_key(obj: dict[str, Any], input_file: str) -> str:
    """Stable identifier of one replay output, used to resume a run.

    Same format as the original implementation, so old CSVs can be resumed.
    """
    return "|".join([
        os.path.basename(input_file),
        str(obj.get("scenario_id", "")),
        str(obj.get("source_file", "")),
        str(obj.get("run_mode", "")),
        str(obj.get("replay_idx", "")),
        str(obj.get("seed", "")),
        str(obj.get("target_model", "")),
        sha256_hex(obj.get("text", "")),
    ])


def _already_judged(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8", newline="") as handle:
        # Failed calls (judge_ok False, e.g. the server went down) are judged again.
        return {row["output_key"] for row in csv.DictReader(handle)
                if row.get("output_key") and row.get("judge_ok") in {"True", "true", "1"}}


def iter_outputs(cfg: EvaluatorConfig, skip: set[str]) -> Iterator[dict[str, Any]]:
    """Successful replay outputs to judge, in file order."""
    emitted = 0
    for path, line_number, obj in read_jsonl(cfg.inputs):
        if obj.get("error") or not (obj.get("text") or "").strip():
            continue
        if cfg.run_mode and obj.get("run_mode") != cfg.run_mode:
            continue
        key = output_key(obj, path)
        if key in skip:
            continue
        yield {**obj, "_input_file": path, "_line_number": line_number, "_output_key": key}
        emitted += 1
        if cfg.limit is not None and emitted >= cfg.limit:
            return


def _preview(text: str, limit: int) -> str:
    text = text.replace("\r", " ").replace("\n", " ").strip()
    return text if len(text) <= limit else text[:limit] + "…"


def judge_record(obj: dict[str, Any], cfg: EvaluatorConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Judge one output. Returns (CSV row, detailed record)."""
    text = obj.get("text", "")
    metadata = {k: obj.get(k, "") for k in METADATA_FIELDS} if cfg.include_metadata else None
    started = time.perf_counter()
    verdict, raw_text, provenance = evaluate_output(
        text,
        judge_model=cfg.judge_model,
        metadata=metadata,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        seed=cfg.seed + int(obj.get("replay_idx") or 0),
        guided_json=cfg.guided_json,
        guided_backend=cfg.guided_backend,
        max_retries=cfg.max_retries,
        retry_sleep_s=cfg.retry_sleep_s,
    )
    row = {
        "input_file": os.path.basename(obj["_input_file"]),
        "line_number": obj["_line_number"],
        "output_key": obj["_output_key"],
        **{k: obj.get(k, "") for k in ("scenario_id", "source_file", "domain", "attack_type",
                                       "scenario_name", "run_mode", "replay_idx", "seed",
                                       "target_model", "temperature", "max_tokens")},
        "judge_model": cfg.judge_model,
        "judge_temperature": cfg.temperature,
        "judge_max_tokens": cfg.max_tokens,
        "judge_blind": not cfg.include_metadata,
        "verdict": verdict.verdict,
        "malicious": verdict.malicious,
        "unsafe": verdict.unsafe,
        "score": verdict.score,
        "risk_types": verdict.risk_types,
        "issues": verdict.issues,
        "analysis_summary": verdict.analysis_summary,
        "output_hash": sha256_hex(text),
        "output_chars": len(text),
        "output_preview": _preview(text, cfg.preview_chars),
        "raw_judge_json": {k: getattr(verdict, k) for k in (
            "verdict", "malicious", "unsafe", "score", "risk_types", "issues", "analysis_summary")},
        "raw_judge_text": raw_text if cfg.include_raw_judge_text else "",
        "judge_ok": verdict.ok,
        "judge_error": verdict.error or "",
        "timing_ms": int((time.perf_counter() - started) * 1000),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    detail = {**row, "judge_provenance": provenance, "raw_judge_text": raw_text}
    if cfg.include_output_text:
        detail["output_text"] = text
    return row, detail


def run_evaluator(cfg: EvaluatorConfig, progress: Callable[[str], None] = print) -> int:
    """Judge every output. Returns the number of outputs judged in this run."""
    cfg.judge_model = cfg.judge_model or get_settings().judge_model
    skip = _already_judged(cfg.output_csv) if cfg.resume else set()
    records = iter_outputs(cfg, skip)
    max_pending = cfg.max_pending or cfg.workers * 4
    judged, started = 0, time.time()
    progress(f"judge={cfg.judge_model} blind={not cfg.include_metadata} "
             f"guided_json={cfg.guided_json} resume_skips={len(skip)}")

    jsonl = JsonlWriter(cfg.output_jsonl, append=cfg.resume) if cfg.output_jsonl else None
    try:
        with CsvWriter(cfg.output_csv, CSV_FIELDS, append=cfg.resume) as table, \
                ThreadPoolExecutor(max_workers=max(1, cfg.workers)) as pool:
            pending = set()

            def refill() -> None:
                # Keep a bounded number of submitted jobs so that memory stays
                # flat on very large inputs.
                while len(pending) < max_pending:
                    record = next(records, None)
                    if record is None:
                        return
                    pending.add(pool.submit(judge_record, record, cfg))

            refill()
            while pending:
                done, still_pending = wait(pending, return_when=FIRST_COMPLETED)
                pending.clear()
                pending.update(still_pending)
                for future in done:
                    row, detail = future.result()
                    table.write(row)
                    if jsonl:
                        jsonl.write(detail)
                    judged += 1
                    if judged % 50 == 0:
                        progress(f"{judged} outputs judged ({judged / (time.time() - started):.2f}/s)")
                refill()
    finally:
        if jsonl:
            jsonl.close()
    progress(f"finished: {judged} outputs judged in {time.time() - started:.0f}s")
    return judged
