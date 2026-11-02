"""Repository Investigator over the scenario dataset (pilot study).

For every selected scenario:

1. the victim answers the scenario prompt once (seed ``seed_base``); answers
   are cached in a JSONL file so that several judges can be run one after the
   other on the same answers, loading one judge model at a time;
2. each judge model runs :func:`prompt_forensics.agents.repository_investigator.investigate`
   on the scenario's system prompt and main prompt plus the victim's answer;
3. one CSV row per (scenario, judge) is appended, with ``gold_label`` taken
   from the run mode (``malicious`` for attacks, ``benign`` for safe prompts)
   and ``detected_as_malicious`` from :func:`~prompt_forensics.agents.repository_investigator.is_detected`.

:mod:`.analysis` turns that CSV into precision, recall and F1 per judge.

This reproduces the pilot experiments (``results/pilot_2025-12/investigator``).
It is not the five-judge protocol of thesis section 5.4.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from ..agents.repository_investigator import investigate, is_detected
from ..config import get_settings
from ..llm import LLMError, call_llm
from ..scenarios import build_victim_prompt, load_scenarios, main_prompt, scenario_id, select_scenarios
from .io import CsvWriter, JsonlWriter, read_jsonl

CSV_FIELDS = [
    "scenario_id", "domain", "attack_type", "scenario_name", "source_file", "target_model",
    "judge_model", "judge_ok", "judge_error", "verdict", "score", "score_imputed", "risk_types",
    "issues", "malicious_prompt_ids", "num_malicious_prompts", "detected_as_malicious",
    "gold_label", "run_mode", "analysis_summary", "live_output",
]
GOLD_LABEL = {"attacks": "malicious", "safe": "benign"}


@dataclass
class InvestigatorConfig:
    run_mode: str = "attacks"
    judge_models: list[str] = field(default_factory=list)
    attack_filter: str | None = None
    limit: int | None = None
    target_model: str | None = None         # default: LIVE_MODEL
    output_csv: str | None = None
    live_cache: str = "results/live_cache.jsonl"
    skip_judging: bool = False              # only build the live cache
    max_tokens: int = 640
    disable_thinking: bool = False          # Qwen3-style chat_template_kwargs
    scenario_root: str | None = None


def investigator_prompts(scenario: dict[str, Any], run_mode: str) -> list[dict[str, str]]:
    """What the Investigator sees: the system prompt (role system) and the main prompt."""
    sid = scenario_id(scenario)
    prompts = []
    if scenario.get("system_prompt"):
        prompts.append({"id": f"{sid}_system", "role": "system", "text": scenario["system_prompt"]})
    prompts.append({"id": sid, "role": "user", "text": main_prompt(scenario, run_mode)})
    return prompts


def _load_cache(path: str) -> dict[tuple[str, str, str], str]:
    if not os.path.exists(path):
        return {}
    return {
        (o["scenario_id"], o["run_mode"], o["target_model"]): o["live_output"]
        for _, _, o in read_jsonl([path])
        if isinstance(o.get("live_output"), str)
    }


def run_investigator(cfg: InvestigatorConfig, progress: Callable[[str], None] = print) -> str:
    """Run the experiment. Returns the CSV path."""
    settings = get_settings()
    target = cfg.target_model or settings.live_model
    if not cfg.judge_models and not cfg.skip_judging:
        raise ValueError("give at least one judge model")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_csv = cfg.output_csv or f"results/investigator_{cfg.run_mode}_{stamp}.csv"
    scenarios = select_scenarios(load_scenarios(cfg.scenario_root or settings.scenario_root),
                                 cfg.run_mode, cfg.attack_filter, cfg.limit)
    cache = _load_cache(cfg.live_cache)
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if cfg.disable_thinking else None
    victim_params = {"temperature": settings.live_temperature,
                     "max_tokens": settings.live_max_tokens, "seed": settings.seed_base}
    progress(f"{len(scenarios)} scenarios, {len(cache)} cached victim answers, "
             f"judges={cfg.judge_models}")

    with JsonlWriter(cfg.live_cache) as cache_writer, \
            CsvWriter(output_csv, CSV_FIELDS, append=os.path.exists(output_csv)) as table:
        for n, scenario in enumerate(scenarios, start=1):
            sid = scenario_id(scenario)
            key = (sid, cfg.run_mode, target)
            if key not in cache:
                try:
                    answer = call_llm(build_victim_prompt(scenario, cfg.run_mode), target, victim_params)
                except LLMError as exc:
                    progress(f"[{sid}] victim call failed, scenario skipped: {exc}")
                    continue
                cache[key] = answer.text
                cache_writer.write({"scenario_id": sid, "run_mode": cfg.run_mode,
                                    "target_model": target, "live_output": answer.text})
            if cfg.skip_judging:
                continue

            for judge in cfg.judge_models:
                result = investigate(investigator_prompts(scenario, cfg.run_mode), cache[key],
                                     judge, max_tokens=cfg.max_tokens, extra_body=extra)
                report = result["json"]
                table.write({
                    "scenario_id": sid,
                    "domain": scenario.get("domain", ""),
                    "attack_type": scenario.get("attack_type", ""),
                    "scenario_name": scenario.get("scenario_name", ""),
                    "source_file": scenario.get("source_file", ""),
                    "target_model": target,
                    "judge_model": judge,
                    "judge_ok": result["ok"],
                    "judge_error": result["error"] or "",
                    "verdict": report["verdict"],
                    "score": report["score"],
                    "score_imputed": report["score_imputed"],
                    "risk_types": "|".join(report["risk_types"]),
                    "issues": "|".join(report["issues"]),
                    "malicious_prompt_ids": "|".join(report["malicious_prompt_ids"]),
                    "num_malicious_prompts": sum(1 for p in report["malicious_prompts"] if p),
                    "detected_as_malicious": int(is_detected(report)),
                    "gold_label": GOLD_LABEL[cfg.run_mode],
                    "run_mode": cfg.run_mode,
                    "analysis_summary": report["analysis_summary"],
                    "live_output": cache[key],
                })
            if n % 20 == 0:
                progress(f"{n}/{len(scenarios)} scenarios done")
    return output_csv
