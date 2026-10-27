"""Version 2 scenario files, dev/test split, annotation sample and audit.

**Assembly.** Each version 1 scenario keeps its context and its attack
prompt; its safe prompt is replaced by a new one generated for the same
context (``safe_prompt_v1`` keeps the old text). Every new safe prompt is
used once. When a context has more new safe prompts than scenarios, the
extra ones become safe-only scenarios; when it has fewer, the remaining
scenarios have no safe prompt (they stay in the attack runs). A scenario
left with neither an attack nor a safe prompt is dropped.

**Split.** ``split`` is ``dev`` or ``test`` per *document*, so that the
prompts used to choose a threshold never share a document with the prompts
used to report results. Attack and safe prompts of the same document are in
the same split.

**Annotation sample.** A stratified random sample (domain x kind) written as
two CSV files: the sheet given to the annotators, which shows only what the
victim model sees, and the key, which links each item to its prompt and its
kind. Annotators therefore do not know which items are hard negatives.

**Agreement.** Two filled sheets are compared with Cohen's kappa over the
three labels, overall and per kind and domain (through the key).
"""

from __future__ import annotations

import csv
import io
import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..canonical import sha256_hex
from .contexts import Context, scenario_context_id
from .dedup import DedupIndex, EmbedFn, mask_document

SHEET_FIELDS = ["item", "system_prompt", "document", "prompt", "label", "comment"]
KEY_FIELDS = ["item", "safe_prompt_id", "domain", "family", "kind", "task"]
_SAFE_NAME = re.compile(r"[A-Za-z0-9_-]+")
LABELS = ("legitimate", "not_legitimate", "unclear")
LABELS_HELP = "label: " + " | ".join(LABELS)


def split_of(doc_key: str, dev_fraction: float, seed: int) -> str:
    """``dev`` or ``test``, fixed for a document given the seed."""
    u = int(sha256_hex(f"{seed}:{doc_key}")[:8], 16) / 0x1_0000_0000
    return "dev" if u < dev_fraction else "test"


def _doc_key(scenario: dict[str, Any]) -> str:
    doc = str(scenario.get("dataset", "")).strip().split("\n", 1)[0].strip()
    return f"{scenario.get('domain', '')}/{doc}"


def assemble_scenarios(scenarios: Sequence[dict[str, Any]], kept: Sequence[dict[str, Any]],
                       dev_fraction: float = 0.2, seed: int = 2026) -> list[dict[str, Any]]:
    """Version 2 scenarios (see the module docstring)."""
    pool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in kept:
        pool[row["context_id"]].append(row)
    for rows in pool.values():
        rows.reverse()  # pop() then returns them in the order they were kept

    out: list[dict[str, Any]] = []
    first_of_context: dict[str, dict[str, Any]] = {}
    for scenario in scenarios:
        cid = scenario_context_id(scenario)
        first_of_context.setdefault(cid, scenario)
        new = dict(scenario)
        new.pop("source_file", None)
        new["context_id"] = cid
        new["safe_prompt_v1"] = scenario.get("safe_prompt")
        new["split"] = split_of(_doc_key(scenario), dev_fraction, seed)
        if pool.get(cid):
            row = pool[cid].pop()
            new.update(safe_prompt=row["text"], safe_prompt_id=row["id"], safe_kind=row["kind"])
        elif scenario.get("attack_prompt"):
            new.update(safe_prompt=None, safe_prompt_id=None, safe_kind=None)
        else:
            continue  # nothing left to run
        out.append(new)

    counters: Counter[str] = Counter()
    for cid, rows in pool.items():
        base = first_of_context[cid]
        prefix = str(base.get("id", "SCENARIO")).rsplit("_", 1)[0]
        while rows:
            row = rows.pop()
            counters[prefix] += 1
            out.append({
                "id": f"{prefix}_S{counters[prefix]:05d}",
                "domain": base.get("domain"),
                "attack_type": base.get("attack_type"),
                "scenario_name": base.get("scenario_name"),
                "description": base.get("description"),
                "system_prompt": base.get("system_prompt"),
                "dataset": base.get("dataset"),
                "attack_prompt": None,
                "benchmark_source": None,
                "benchmark_id": None,
                "context_id": cid,
                "safe_prompt_v1": None,
                "split": split_of(_doc_key(base), dev_fraction, seed),
                "safe_prompt": row["text"],
                "safe_prompt_id": row["id"],
                "safe_kind": row["kind"],
            })
    return out


def write_scenarios(scenarios: Iterable[dict[str, Any]], root: str | Path, overwrite: bool = False) -> list[Path]:
    """Write ``<root>/<family>/<domain>_scenarios.json`` (the version 1 layout).

    The files are written to ``<root>.partial`` and the directory is renamed
    to ``root`` only once every file is complete, so an interrupted assembly
    never leaves a half-written dataset where the runners look for one. An
    existing ``root`` is replaced only with ``overwrite=True``.
    """
    root = Path(root)
    if root.exists() and not overwrite:
        raise FileExistsError(f"{root} exists; choose another directory or pass overwrite=True")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for s in scenarios:
        family, domain = str(s["attack_type"]), str(s["domain"])
        for name in (family, domain):
            # These values become path components.
            if not _SAFE_NAME.fullmatch(name):
                raise ValueError(f"scenario {s.get('id')}: invalid family or domain name {name!r}")
        groups[(family, domain)].append(s)

    staging = root.with_name(root.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    relative = []
    for (family, domain), items in sorted(groups.items()):
        rel = Path(family) / f"{domain}_scenarios.json"
        (staging / family).mkdir(parents=True, exist_ok=True)
        (staging / rel).write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        relative.append(rel)
    if root.exists():
        shutil.rmtree(root)
    os.replace(staging, root)
    return [root / rel for rel in relative]


def annotation_sample(kept: Sequence[dict[str, Any]], contexts: dict[str, Context], size: int = 500,
                      seed: int = 2026) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Stratified sample (domain x kind, proportional allocation) -> (sheet, key)."""
    rng = random.Random(seed)
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in kept:
        strata[(row["domain"], row["kind"])].append(row)
    total = len(kept)
    chosen: list[dict[str, Any]] = []
    # Largest-remainder allocation, so the strata sizes add up to `size`.
    quotas = {k: size * len(v) / total for k, v in strata.items()} if total else {}
    alloc = {k: int(q) for k, q in quotas.items()}
    for k in sorted(quotas, key=lambda k: quotas[k] - alloc[k], reverse=True)[: size - sum(alloc.values())]:
        alloc[k] += 1
    for key in sorted(strata):
        chosen += rng.sample(strata[key], min(alloc[key], len(strata[key])))
    rng.shuffle(chosen)

    sheet, answer_key = [], []
    for n, row in enumerate(chosen, 1):
        ctx = contexts[row["context_id"]]
        item = f"A{n:04d}"
        sheet.append({"item": item, "system_prompt": ctx.system_prompt, "document": ctx.document,
                      "prompt": row["text"], "label": "", "comment": ""})
        answer_key.append({"item": item, "safe_prompt_id": row["id"], "domain": row["domain"],
                           "family": row["family"], "kind": row["kind"], "task": row["task"]})
    return sheet, answer_key


def write_csv(rows: Sequence[dict[str, Any]], fields: Sequence[str], path: str | Path) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def audit_prompts(rows: Sequence[tuple[str, str, Sequence[str]]], embed: EmbedFn | None = None,
                  threshold: float = 0.90) -> dict[str, Any]:
    """Duplicate audit of ``(id, text, document-name spellings)`` rows (Table 5.2).

    Returns the number of rows, of distinct raw texts, of distinct texts once
    normalised and with the document name masked, and, with an embedder, of
    texts left after removing near-duplicates.
    """
    report: dict[str, Any] = {
        "rows": len(rows),
        "distinct_raw": len({text for _, text, _ in rows}),
    }
    exact = DedupIndex(None)
    exact.add_batch([(i, mask_document(t, terms)) for i, t, terms in rows])
    report["distinct_normalised_masked"] = len(exact)
    if embed is not None:
        near = DedupIndex(embed, threshold)
        step = 2000
        for start in range(0, len(rows), step):
            near.add_batch([(i, mask_document(t, terms)) for i, t, terms in rows[start:start + step]])
        report[f"distinct_after_near_dedup_{threshold}"] = len(near)
    return report


def read_csv(path: str | Path) -> list[dict[str, str]]:
    """Rows of a CSV file, also when a spreadsheet saved it with ';' and a BOM."""
    text = Path(path).read_text(encoding="utf-8-sig")
    header = text.split("\n", 1)[0]
    delimiter = max(",;\t", key=header.count)
    # StringIO, not splitlines(): cells (the documents) span several lines.
    return list(csv.DictReader(io.StringIO(text, newline=""), delimiter=delimiter))


def _label(value: str | None) -> str | None:
    """Normalised label, ``None`` if empty; spaces, hyphens and case ignored."""
    label = re.sub(r"[\s-]+", "_", (value or "").strip().lower())
    if not label:
        return None
    if label not in LABELS:
        raise ValueError(f"unknown label {value!r} (expected {LABELS_HELP})")
    return label


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Cohen's kappa of two equally long label sequences."""
    n = len(a)
    if n == 0:
        return float("nan")
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def _agreement_stats(pairs: Sequence[tuple[str, str]]) -> dict[str, Any]:
    a, b = [x for x, _ in pairs], [y for _, y in pairs]
    n = len(pairs)
    return {
        "items": n,
        "observed_agreement": round(sum(x == y for x, y in pairs) / n, 4) if n else None,
        "cohen_kappa": round(cohen_kappa(a, b), 4) if n else None,
        "legitimate_a": round(a.count("legitimate") / n, 4) if n else None,
        "legitimate_b": round(b.count("legitimate") / n, 4) if n else None,
        # Both annotators say legitimate: the share reported as validated.
        "legitimate_both": round(sum(x == y == "legitimate" for x, y in pairs) / n, 4) if n else None,
    }


def agreement(sheet_a: Sequence[dict[str, str]], sheet_b: Sequence[dict[str, str]],
              key: Sequence[dict[str, str]]) -> dict[str, Any]:
    """Agreement of two filled annotation sheets, overall and per kind and domain.

    Items left empty by either annotator are counted and left out.
    """
    labels_a = {row["item"]: _label(row.get("label")) for row in sheet_a}
    labels_b = {row["item"]: _label(row.get("label")) for row in sheet_b}
    if set(labels_a) != set(labels_b):
        raise ValueError("the two sheets do not have the same items")
    meta = {row["item"]: row for row in key}
    pairs: dict[str, tuple[str, str]] = {}
    missing = 0
    for item in labels_a:
        if labels_a[item] is None or labels_b[item] is None:
            missing += 1
        else:
            pairs[item] = (labels_a[item], labels_b[item])

    groups: dict[str, dict[str, list[tuple[str, str]]]] = {"kind": defaultdict(list), "domain": defaultdict(list)}
    for item, pair in pairs.items():
        for field, by_value in groups.items():
            by_value[meta.get(item, {}).get(field, "?")].append(pair)
    confusion = Counter(f"{x}/{y}" for x, y in pairs.values())
    return {
        "overall": _agreement_stats(list(pairs.values())),
        "unlabelled_items": missing,
        "by_kind": {k: _agreement_stats(v) for k, v in sorted(groups["kind"].items())},
        "by_domain": {k: _agreement_stats(v) for k, v in sorted(groups["domain"].items())},
        "confusion_a_b": dict(sorted(confusion.items())),
        "disagreements": sorted(item for item, (x, y) in pairs.items() if x != y),
    }
