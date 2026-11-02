"""Generation rounds, filtering, deduplication and output files.

One **round** sends one generation request (``per_call`` prompts) for every
context that has fewer kept prompts than its quota (``target`` divided
evenly over the contexts, which keeps the domain x family design balanced).
Each new prompt then goes through:

1. the **filters** (length, language, blatant attack markers);
2. the **deduplication** against every prompt kept so far.

The run stops when the target is reached, when a round keeps less than
``min_yield`` of what it generated (the generator is saturated: more rounds
would mostly produce duplicates), or after ``max_rounds``.

Everything is written to ``out_dir`` as it happens, so the run can be
interrupted and resumed (on resume the kept prompts are checked again, so a
change of the deduplication rule prunes them and the next rounds top up), and every kept prompt can be traced back to the
request and the answer that produced it:

=====================  ======================================================
``requests.jsonl``     every generation request (context, slot, full prompt)
``responses.jsonl``    every raw answer (serving model, usage, stop reason)
``candidates.jsonl``   every generated prompt and what happened to it
``safe_prompts.jsonl`` the prompts kept, with their ids
``pruned.jsonl``       kept prompts removed on resume by a stricter deduplication
``rounds.csv``         per round: generated, kept, yield (saturation curve)
``manifest.json``      settings, model, code version, counts and token usage
=====================  ======================================================
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import re
import subprocess
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from ..canonical import sha256_hex
from .contexts import Context
from .dedup import KEPT, DedupIndex, EmbedFn, mask_document
from .generator import PROMPT_TEMPLATE, Generation, build_prompt, parse_prompts
from .taxonomy import STANDARD, sample_slot

Generator = Callable[[Sequence[tuple[str, str]]], list[Generation]]

ROUND_FIELDS = [
    "round", "requests", "failed_requests", "generated", "filtered",
    "exact_duplicates", "near_duplicates", "kept", "yield", "cumulative_kept",
]

# Phrases that only appear in attacks. A candidate containing one is dropped,
# hard negatives included: a "legitimate" prompt asking to ignore previous
# instructions would make the label itself questionable.
BLATANT_ATTACK = re.compile(
    r"ignore (all |any )?(previous|prior|above|your) (instructions|rules)|developer mode|"
    r"\bDAN\b|do anything now|jailbreak|unrestricted (mode|ai)|without (any )?restrictions|"
    r"reveal (your|the) (system prompt|instructions)",
    re.IGNORECASE,
)
# Additional markers for standard prompts, which should not even look like an attack.
ATTACK_VOCABULARY = re.compile(
    r"system prompt|bypass|override|pretend (you are|to be)|role-?play|hypothetically|"
    r"confidential (data|information) (of|about)|verbatim",
    re.IGNORECASE,
)


@dataclass
class GenerationConfig:
    out_dir: Path
    target: int = 20_000
    per_call: int = 10
    hard_negative_rate: float = 0.2
    near_threshold: float = 0.90
    min_yield: float = 0.10
    max_rounds: int = 12
    seed: int = 2026
    avoid_examples: int = 15
    min_words: int = 6
    max_words: int = 200
    # Recorded in the manifest only; the generator object does the calls.
    generator_info: dict[str, Any] = field(default_factory=dict)


def filter_reason(text: str, kind: str, min_words: int, max_words: int) -> str | None:
    """Why a candidate is dropped before deduplication, or None if it passes."""
    words = len(text.split())
    if words < min_words:
        return "too_short"
    if words > max_words:
        return "too_long"
    if sum(ord(c) > 127 for c in text) > 0.05 * len(text):
        return "not_english"
    if BLATANT_ATTACK.search(text):
        return "attack_marker"
    if kind == STANDARD and ATTACK_VOCABULARY.search(text):
        return "attack_vocabulary"
    return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        lines = [line for line in handle if line.strip()]
    for n, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A run killed while appending leaves a truncated last line; the
            # round it belongs to is redone on resume. Anything else is an error.
            if n != len(lines) - 1:
                raise
    return rows


def _append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Replace ``path`` atomically, so an interruption never leaves it half written."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    os.replace(tmp, path)


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class GenerationRun:
    """State of one generation run, rebuilt from ``out_dir`` when resuming."""

    def __init__(self, config: GenerationConfig, contexts: Sequence[Context], embed: EmbedFn | None,
                 log: Callable[[str], None] = print):
        self.config = config
        self.contexts = list(contexts)
        self.by_id = {c.context_id: c for c in self.contexts}
        self.log = log
        self.out = Path(config.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.quota = math.ceil(config.target / max(len(self.contexts), 1))
        self.index = DedupIndex(embed, config.near_threshold)
        self.kept: list[dict[str, Any]] = []
        self.kept_per_context: Counter[str] = Counter()
        self.recent_per_doc: dict[str, deque[str]] = defaultdict(lambda: deque(maxlen=config.avoid_examples))
        self.rounds: list[dict[str, Any]] = []
        self.responses: dict[str, dict[str, Any]] = {}
        self.next_number = 1          # SP2_<number> of the next kept prompt
        self.pruned_after_round = 0   # last round before a prune on resume
        self._load()

    # ------------------------------------------------------------------ files
    def path(self, name: str) -> Path:
        return self.out / name

    def _load(self) -> None:
        """Rebuild the state of an interrupted run (completed rounds only)."""
        if self.path("rounds.csv").exists():
            with self.path("rounds.csv").open(newline="", encoding="utf-8") as handle:
                self.rounds = list(csv.DictReader(handle))
        done = len(self.rounds)
        # Rows of a round that did not finish are dropped; the round is redone,
        # reusing the answers already received.
        kept = [r for r in _read_jsonl(self.path("safe_prompts.jsonl")) if r["round"] <= done]
        _write_jsonl(self.path("safe_prompts.jsonl"), kept)
        candidates = [r for r in _read_jsonl(self.path("candidates.jsonl")) if r["round"] <= done]
        _write_jsonl(self.path("candidates.jsonl"), candidates)
        self.responses = {r["request_id"]: r for r in _read_jsonl(self.path("responses.jsonl"))}
        pruned = _read_jsonl(self.path("pruned.jsonl"))
        numbers = [int(r["id"].rsplit("_", 1)[1]) for r in kept + pruned]
        self.next_number = max(numbers, default=0) + 1
        if kept:
            self.log(f"resuming after round {done}: {len(kept)} prompts kept")
            verdicts = self.index.add_batch([self._dedup_item(r) for r in kept])
            removed = []
            for row, verdict in zip(kept, verdicts):
                if verdict.status == KEPT:
                    self._register(row)
                else:
                    removed.append({**row, "status": verdict.status, "duplicate_of": verdict.duplicate_of,
                                    "similarity": verdict.similarity, "pruned_after_round": done})
            if removed:
                # The deduplication rule changed since these prompts were kept.
                self.log(f"pruned {len(removed)} kept prompts that are duplicates under the current rule")
                _append_jsonl(self.path("pruned.jsonl"), removed)
                _write_jsonl(self.path("safe_prompts.jsonl"), self.kept)
                self.pruned_after_round = done

    def _dedup_item(self, row: dict[str, Any]) -> tuple[str, str, str]:
        """(id, text for the exact key, text for the embedding); see DedupIndex.add_batch."""
        masked = mask_document(row["text"], self.by_id[row["context_id"]].mask_terms())
        return row.get("id") or row["candidate_id"], masked, row["text"]

    def _register(self, row: dict[str, Any]) -> None:
        self.kept.append(row)
        self.kept_per_context[row["context_id"]] += 1
        self.recent_per_doc[self.by_id[row["context_id"]].doc_key].append(row["text"])

    # ----------------------------------------------------------------- rounds
    def open_contexts(self) -> list[Context]:
        return [c for c in self.contexts if self.kept_per_context[c.context_id] < self.quota]

    def build_requests(self, round_no: int) -> list[dict[str, Any]]:
        """At most one request per open context. Deterministic given the seed and the state.

        A round never asks for much more than what is still missing: when the
        remaining target is smaller than ``per_call`` x open contexts (a pilot,
        or the last round), only ``ceil(remaining / per_call)`` contexts are
        served, those with the fewest kept prompts first, ties broken at random.
        """
        remaining = self.config.target - len(self.kept)
        wanted = math.ceil(remaining / self.config.per_call)
        order = random.Random(f"{self.config.seed}:{round_no}:order")
        contexts = self.open_contexts()
        tiebreak = {c.context_id: order.random() for c in contexts}
        contexts.sort(key=lambda c: (self.kept_per_context[c.context_id], tiebreak[c.context_id]))
        requests = []
        for context in contexts[: max(wanted, 0)]:
            rng = random.Random(f"{self.config.seed}:{round_no}:{context.context_id}")
            slot = sample_slot(rng, context.domain, context.family, self.config.hard_negative_rate)
            avoid = list(self.recent_per_doc[context.doc_key])
            prompt = build_prompt(context, slot, self.config.per_call, avoid)
            requests.append({
                "request_id": f"r{round_no:02d}-{context.context_id}",
                "round": round_no,
                "context_id": context.context_id,
                "slot": slot,
                "prompt": prompt,
                "prompt_sha256": sha256_hex(prompt),
            })
        return requests

    def _reusable(self, request: dict[str, Any]) -> bool:
        """An earlier successful answer to exactly this prompt (resumed run)."""
        answer = self.responses.get(request["request_id"], {})
        return bool(answer.get("ok")) and answer.get("prompt_sha256") == request["prompt_sha256"]

    def run_round(self, round_no: int, generate: Generator) -> dict[str, Any]:
        requests = self.build_requests(round_no)
        todo = [r for r in requests if not self._reusable(r)]
        _append_jsonl(self.path("requests.jsonl"), (
            {**{k: v for k, v in r.items() if k != "slot"}, "slot": r["slot"].as_dict()} for r in todo
        ))
        if todo:
            self.log(f"round {round_no}: {len(todo)} generation requests")
            answers = generate([(r["request_id"], r["prompt"]) for r in todo])
            rows = [{**asdict(a), "ok": a.ok, "round": round_no, "prompt_sha256": r["prompt_sha256"]}
                    for a, r in zip(answers, todo)]
            _append_jsonl(self.path("responses.jsonl"), rows)
            self.responses.update({r["request_id"]: r for r in rows})

        stats = Counter()
        candidates: list[dict[str, Any]] = []
        for request in requests:
            answer = self.responses.get(request["request_id"], {})
            if not self._reusable(request):
                stats["failed_requests"] += 1
                continue
            try:
                texts = parse_prompts(answer["text"])
            except ValueError:
                stats["failed_requests"] += 1
                continue
            slot = request["slot"]
            for k, text in enumerate(texts):
                candidates.append({
                    "candidate_id": f"{request['request_id']}-{k:02d}",
                    "request_id": request["request_id"],
                    "round": round_no,
                    "context_id": request["context_id"],
                    **slot.as_dict(),
                    "model": answer.get("model"),
                    "text": text,
                })
        stats["generated"] = len(candidates)

        to_check = []
        for cand in candidates:
            reason = filter_reason(cand["text"], cand["kind"], self.config.min_words, self.config.max_words)
            if reason:
                cand.update(status="filtered", reason=reason)
                stats["filtered"] += 1
            else:
                to_check.append(cand)
        verdicts = self.index.add_batch([self._dedup_item(c) for c in to_check])

        new_rows = []
        for cand, verdict in zip(to_check, verdicts):
            cand.update(status=verdict.status, duplicate_of=verdict.duplicate_of, similarity=verdict.similarity)
            if verdict.status != KEPT:
                stats[verdict.status + "s"] += 1
                continue
            # A context over its quota still contributes what it generated in
            # this round: dropping valid prompts would waste them.
            row = {
                "id": f"SP2_{self.next_number:06d}",
                "candidate_id": cand["candidate_id"],
                "round": round_no,
                "context_id": cand["context_id"],
                "domain": self.by_id[cand["context_id"]].domain,
                "family": cand["family"],
                "doc_name": self.by_id[cand["context_id"]].doc_name,
                "kind": cand["kind"],
                "task": cand["task"],
                "persona": cand["persona"],
                "answer_format": cand["answer_format"],
                "model": cand["model"],
                "text": cand["text"],
                "sha256": sha256_hex(cand["text"]),
            }
            self.next_number += 1
            self._register(row)
            new_rows.append(row)
        kept_ids = {r["candidate_id"]: r["id"] for r in new_rows}
        for cand in candidates:
            if cand["candidate_id"] in kept_ids:
                cand["kept_id"] = kept_ids[cand["candidate_id"]]

        _append_jsonl(self.path("candidates.jsonl"), candidates)
        _append_jsonl(self.path("safe_prompts.jsonl"), new_rows)
        summary = {
            "round": round_no,
            "requests": len(requests),
            "failed_requests": stats["failed_requests"],
            "generated": stats["generated"],
            "filtered": stats["filtered"],
            "exact_duplicates": stats["exact_duplicates"],
            "near_duplicates": stats["near_duplicates"],
            "kept": len(new_rows),
            "yield": round(len(new_rows) / stats["generated"], 4) if stats["generated"] else "",
            "cumulative_kept": len(self.kept),
        }
        self.rounds.append(summary)
        with self.path("rounds.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=ROUND_FIELDS)
            writer.writeheader()
            writer.writerows(self.rounds)
        return summary

    def stop_reason(self) -> str | None:
        last = self.rounds[-1] if self.rounds else None
        if last and int(last["requests"]) and int(last["failed_requests"]) == int(last["requests"]):
            errors = Counter(
                (r.get("error") or r.get("stop_reason") or "unparseable answer")[:120]
                for r in self.responses.values() if r.get("round") == int(self.rounds[-1]["round"])
            )
            return f"every request of the last round failed: {errors.most_common(1)[0][0] if errors else '?'}"
        if len(self.kept) >= self.config.target:
            return "target reached"
        if not self.open_contexts():
            return "every context reached its quota"
        if len(self.rounds) >= self.config.max_rounds:
            return "max_rounds reached"
        # A prune on resume reopens contexts: the yield of the rounds before it
        # no longer says whether the generator is saturated.
        if self.rounds and len(self.rounds) > self.pruned_after_round:
            last = self.rounds[-1]
            if last["yield"] != "" and float(last["yield"]) < self.config.min_yield:
                return f"saturation: last round kept {float(last['yield']):.1%} of what it generated"
        return None

    def run(self, generate: Generator) -> str:
        reason = self.stop_reason()
        while reason is None:
            summary = self.run_round(len(self.rounds) + 1, generate)
            self.log(
                f"round {summary['round']}: generated {summary['generated']}, kept {summary['kept']} "
                f"(yield {summary['yield']}), total {summary['cumulative_kept']}"
            )
            self.write_manifest(None)
            reason = self.stop_reason()
        self.write_manifest(reason)
        self.log(f"stopped: {reason}")
        return reason

    def write_manifest(self, stop_reason: str | None) -> None:
        usage: Counter[str] = Counter()
        models: Counter[str] = Counter()
        for r in self.responses.values():
            for key, value in (r.get("usage") or {}).items():
                if isinstance(value, (int, float)):
                    usage[key] += value
            if r.get("model"):
                models[r["model"]] += 1
        manifest = {
            "dataset": "Prompt Forensics scenario dataset v2, safe prompts",
            "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "code_commit": _git_commit(),
            "stop_reason": stop_reason,
            "settings": {
                k: (str(v) if isinstance(v, Path) else v)
                for k, v in asdict(self.config).items()
            },
            "generation_prompt_sha256": sha256_hex(PROMPT_TEMPLATE),
            "contexts": len(self.contexts),
            "quota_per_context": self.quota,
            "rounds": len(self.rounds),
            "kept": len(self.kept),
            "pruned_on_resume": len(_read_jsonl(self.path("pruned.jsonl"))),
            "kept_by_kind": dict(Counter(r["kind"] for r in self.kept)),
            "kept_by_domain": dict(Counter(r["domain"] for r in self.kept)),
            "kept_by_family": dict(Counter(r["family"] for r in self.kept)),
            "serving_models": dict(models),
            "token_usage": dict(usage),
        }
        self.path("manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
