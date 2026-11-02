"""Split the version 2 runs into deep and shallow pools and into GPU shards.

Replaying every prompt 100 times would take about 450 GPU hours, so the runs
use two pool sizes:

* **deep pools**: a stratified random sample of prompts replayed
  ``deep_replays`` times (100). They give the tail-risk curve up to 100
  replays and the statistics that need many replays per prompt;
* **shallow pools**: every other prompt, replayed ``replays`` times (10).

A *prompt* is one scenario in one run mode (its attack or its safe prompt),
so a scenario with both prompts counts twice. The sample is stratified by
run mode x attack family x domain x split x safe-prompt kind, with
proportional allocation (largest remainder), so it has the composition of
the whole dataset.

The work is then cut into ``shards`` parts of equal cost, one per GPU: in
each stratum, deep and shallow prompts are dealt in turn to the shards.
Every shard gets one scenario root per run mode and pool size, in the
version 1 layout, holding only the prompt it must replay (the other prompt
is set to ``None``), so ``run_replays.py --scenario-root`` runs it as is.
"""

from __future__ import annotations

import csv
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from ..canonical import normalize_text
from ..datagen.assemble import write_scenarios
from ..scenarios import RUN_MODES, main_prompt, scenario_id

PROMPT_KEY = {"attacks": "attack_prompt", "safe": "safe_prompt"}
PLAN_FIELDS = ["scenario_id", "run_mode", "pool", "replays", "shard",
               "attack_type", "domain", "split", "safe_kind"]


def _stratum(scenario: dict[str, Any], run_mode: str) -> tuple[str, ...]:
    kind = (scenario.get("safe_kind") or "") if run_mode == "safe" else ""
    return (run_mode, str(scenario.get("attack_type")), str(scenario.get("domain")),
            str(scenario.get("split")), kind)


def _allocate(sizes: dict[tuple, int], total: int) -> dict[tuple, int]:
    """Proportional allocation of ``total`` over strata, largest remainder."""
    population = sum(sizes.values())
    if total > population:
        raise ValueError(f"asked for {total} deep prompts, the dataset has {population}")
    exact = {k: total * n / population for k, n in sizes.items()}
    alloc = {k: int(v) for k, v in exact.items()}
    # Ties broken by stratum name, so the plan does not depend on dict order.
    by_remainder = sorted(exact, key=lambda k: (-(exact[k] - alloc[k]), k))
    for k in by_remainder[:total - sum(alloc.values())]:
        alloc[k] += 1
    return alloc


def plan_runs(scenarios: Sequence[dict[str, Any]], deep: int = 5_000, deep_replays: int = 100,
              replays: int = 10, shards: int = 2, seed: int = 2026) -> list[dict[str, Any]]:
    """One row per prompt: its pool (deep or shallow), replay count and shard."""
    strata: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for scenario in scenarios:
        for mode in RUN_MODES:
            if main_prompt(scenario, mode):
                strata[_stratum(scenario, mode)].append(scenario)
    alloc = _allocate({k: len(v) for k, v in strata.items()}, deep)

    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for key in sorted(strata):
        members = sorted(strata[key], key=scenario_id)
        rng.shuffle(members)
        mode = key[0]
        for pool, part in (("deep", members[:alloc[key]]), ("shallow", members[alloc[key]:])):
            # Dealt in turn within each stratum and pool, from a random first
            # shard so that the odd prompts of the strata do not all land on
            # shard 0: every shard gets the same work, give or take a prompt.
            offset = rng.randrange(shards)
            for i, scenario in enumerate(part):
                rows.append({
                    "scenario_id": scenario_id(scenario), "run_mode": mode, "pool": pool,
                    "replays": deep_replays if pool == "deep" else replays,
                    "shard": (offset + i) % shards,
                    "attack_type": scenario.get("attack_type"), "domain": scenario.get("domain"),
                    "split": scenario.get("split"),
                    "safe_kind": scenario.get("safe_kind") if mode == "safe" else None,
                })
    return rows


def write_plan(scenarios: Sequence[dict[str, Any]], rows: Sequence[dict[str, Any]],
               out_dir: str | Path, overwrite: bool = False) -> list[Path]:
    """Write ``plan.csv`` and the scenario roots ``shard<k>/<run_mode>_r<replays>``.

    Returns the scenario roots written.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "plan.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PLAN_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    by_id = {scenario_id(s): s for s in scenarios}
    groups: dict[tuple[int, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        scenario = dict(by_id[row["scenario_id"]])
        scenario.pop("source_file", None)
        other = "safe" if row["run_mode"] == "attacks" else "attacks"
        scenario[PROMPT_KEY[other]] = None  # replay only the prompt of this run mode
        groups[(row["shard"], row["run_mode"], row["replays"])].append(scenario)

    roots = []
    for (shard, mode, n), items in sorted(groups.items()):
        root = out / f"shard{shard}" / f"{mode}_r{n}"
        write_scenarios(items, root, overwrite=overwrite)
        roots.append(root)
    return roots


# ---- Whitespace experiment -------------------------------------------------------
#
# The Prompt ID is computed after whitespace normalisation (canonical.py), so
# two prompts that differ only in spacing share it. Is that safe, that is, does
# the victim model answer them alike? A sample of deep-pool prompts is replayed
# a second time as is, and once with whitespace that the normalisation removes
# (double spaces, tabs, trailing spaces, CRLF line ends), with the same seeds.

_QUOTED = re.compile(r'("[^"\n]*")')


def perturb_whitespace(text: str, rng: random.Random) -> str:
    """A copy of ``text`` with extra whitespace that ``normalize_text`` removes.

    Quoted spans are left alone, since the normalisation keeps their spacing.
    The result always differs from ``text`` and always normalises to the same
    string.
    """
    parts = _QUOTED.split(text)
    out = []
    for i, part in enumerate(parts):
        if i % 2 == 1:  # quoted span
            out.append(part)
            continue
        lines = part.split("\n")
        new_lines = []
        for line in lines:
            line = re.sub(r" ", lambda _: rng.choice([" ", " ", " ", "  ", "   ", "\t"]), line)
            if line and rng.random() < 0.5:
                line += rng.choice([" ", "  ", "\t"])
            new_lines.append(line)
        out.append("".join(
            line + ("" if j == len(new_lines) - 1 else rng.choice(["\n", "\r\n"]))
            for j, line in enumerate(new_lines)))
    result = "".join(out)
    if result == text:  # nothing was changed by chance: add a trailing space
        result = text + " "
    if normalize_text(result) != normalize_text(text):
        raise AssertionError("whitespace perturbation changed the normalised text")
    return result


def whitespace_sample(rows: Sequence[dict[str, Any]], size: int = 500,
                      seed: int = 2026) -> list[dict[str, Any]]:
    """``size`` deep-pool rows drawn at random, each keeping its shard."""
    deep = [r for r in rows if r["pool"] == "deep"]
    return random.Random(seed).sample(deep, min(size, len(deep)))


def write_whitespace_roots(scenarios: Sequence[dict[str, Any]], sample: Sequence[dict[str, Any]],
                           out_dir: str | Path, seed: int = 2026,
                           overwrite: bool = False) -> list[Path]:
    """Scenario roots ``shard<k>/ws_<mode>_orig`` and ``ws_<mode>_pert``."""
    rng = random.Random(seed)
    by_id = {scenario_id(s): s for s in scenarios}
    groups: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in sample:
        base = dict(by_id[row["scenario_id"]])
        base.pop("source_file", None)
        mode = row["run_mode"]
        other = "safe" if mode == "attacks" else "attacks"
        base[PROMPT_KEY[other]] = None
        perturbed = dict(base)
        perturbed[PROMPT_KEY[mode]] = perturb_whitespace(base[PROMPT_KEY[mode]], rng)
        groups[(row["shard"], mode, "orig")].append(base)
        groups[(row["shard"], mode, "pert")].append(perturbed)
    roots = []
    for (shard, mode, variant), items in sorted(groups.items()):
        root = Path(out_dir) / f"shard{shard}" / f"ws_{mode}_{variant}"
        write_scenarios(items, root, overwrite=overwrite)
        roots.append(root)
    return roots
