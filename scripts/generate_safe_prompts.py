#!/usr/bin/env python3
"""Generate the safe prompts of the scenario dataset, version 2.

Subcommands, in the order they are used:

    audit     duplicate audit of the safe prompts of a scenario dataset (Table 5.2)
    preview   print the generation prompt of one context, without calling the API
    generate  generation rounds with filtering and deduplication (resumable)
    assemble  write the version 2 scenario files with the dev/test split
    sample    write the human annotation sheet and its key

Example (needs ANTHROPIC_API_KEY):

    python scripts/generate_safe_prompts.py audit --near
    python scripts/generate_safe_prompts.py generate --out data/safe_prompts_v2 --target 300 --max-rounds 1
    python scripts/generate_safe_prompts.py generate --out data/safe_prompts_v2 --target 20000 --backend batch
    python scripts/generate_safe_prompts.py assemble --out data/safe_prompts_v2 --scenario-out data/scenarios_v2
    python scripts/generate_safe_prompts.py sample --out data/safe_prompts_v2 --size 500
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Allow running from a fresh clone without "pip install -e .".
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prompt_forensics.config import get_settings  # noqa: E402
from prompt_forensics.datagen import assemble as asm  # noqa: E402
from prompt_forensics.datagen.contexts import contexts_from_scenarios, scenario_context_id  # noqa: E402
from prompt_forensics.datagen.dedup import default_embedder  # noqa: E402
from prompt_forensics.datagen.generator import (  # noqa: E402
    DEFAULT_MODEL,
    BatchGenerator,
    SyncGenerator,
    build_prompt,
)
from prompt_forensics.datagen.pipeline import GenerationConfig, GenerationRun, _read_jsonl  # noqa: E402
from prompt_forensics.datagen.taxonomy import sample_slot  # noqa: E402
from prompt_forensics.scenarios import load_scenarios  # noqa: E402


def cmd_audit(args) -> None:
    scenarios = load_scenarios(args.scenario_root)
    contexts = {c.context_id: c for c in contexts_from_scenarios(scenarios)}
    rows = [
        (str(s.get("id")), str(s["safe_prompt"]), contexts[scenario_context_id(s)].mask_terms())
        for s in scenarios if s.get("safe_prompt")
    ]
    report = asm.audit_prompts(rows, default_embedder() if args.near else None, args.threshold)
    report["contexts"] = len(contexts)
    report["documents"] = len({c.doc_key for c in contexts.values()})
    report["system_prompts"] = len({c.system_prompt for c in contexts.values()})
    print(json.dumps(report, indent=2))


def cmd_preview(args) -> None:
    import random

    contexts = contexts_from_scenarios(load_scenarios(args.scenario_root))
    context = contexts[args.index % len(contexts)]
    slot = sample_slot(random.Random(args.seed), context.domain, context.family,
                       1.0 if args.hard_negative else 0.0)
    print(build_prompt(context, slot, args.per_call))


def cmd_generate(args) -> None:
    scenarios = load_scenarios(args.scenario_root)
    contexts = contexts_from_scenarios(scenarios)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.backend == "batch":
        generate = BatchGenerator(out / "batches", args.model, args.max_tokens, args.effort)
    else:
        generate = SyncGenerator(args.model, args.max_tokens, args.effort, args.workers,
                                 fallbacks=not args.no_fallbacks)
    fallbacks = getattr(generate, "fallbacks", False)
    config = GenerationConfig(
        out_dir=out, target=args.target, per_call=args.per_call,
        hard_negative_rate=args.hard_negative_rate, near_threshold=args.threshold,
        min_yield=args.min_yield, max_rounds=args.max_rounds, seed=args.seed,
        generator_info={
            "model": args.model, "backend": args.backend, "effort": args.effort,
            "max_tokens": args.max_tokens, "fallbacks": fallbacks,
            "embedding_model": get_settings().embed_model_name if not args.exact_only else None,
            "scenario_root": args.scenario_root,
        },
    )
    embed = None if args.exact_only else default_embedder()
    GenerationRun(config, contexts, embed).run(generate)


def cmd_assemble(args) -> None:
    scenarios = load_scenarios(args.scenario_root)
    kept = _read_jsonl(Path(args.out) / "safe_prompts.jsonl")
    v2 = asm.assemble_scenarios(scenarios, kept, args.dev_fraction, args.seed)
    paths = asm.write_scenarios(v2, args.scenario_out, overwrite=args.overwrite)
    with_safe = [s for s in v2 if s.get("safe_prompt")]
    summary = {
        "scenario_files": len(paths),
        "scenarios": len(v2),
        "safe_prompts": len(with_safe),
        "attack_prompts": sum(1 for s in v2 if s.get("attack_prompt")),
        "scenarios_without_safe_prompt": len(v2) - len(with_safe),
        "safe_by_split": dict(Counter(s["split"] for s in with_safe)),
        "safe_by_kind": dict(Counter(s["safe_kind"] for s in with_safe)),
    }
    # Kept with the generation run: every .json under the scenario root is read
    # as a scenario file by the runners.
    summary["scenario_root"] = str(args.scenario_out)
    (Path(args.out) / "assembly.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def cmd_sample(args) -> None:
    contexts = {c.context_id: c for c in contexts_from_scenarios(load_scenarios(args.scenario_root))}
    kept = _read_jsonl(Path(args.out) / "safe_prompts.jsonl")
    sheet, key = asm.annotation_sample(kept, contexts, args.size, args.seed)
    out = Path(args.out)
    asm.write_csv(sheet, asm.SHEET_FIELDS, out / "annotation_sheet.csv")
    asm.write_csv(key, asm.KEY_FIELDS, out / "annotation_key.csv")
    print(f"{len(sheet)} items -> {out / 'annotation_sheet.csv'} ({asm.LABELS_HELP})")
    print(f"key (do not give to annotators) -> {out / 'annotation_key.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario-root", default=get_settings().scenario_root,
                        help="version 1 scenarios (default: SCENARIO_ROOT)")
    parser.add_argument("--seed", type=int, default=2026)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("audit", help="duplicate audit of the safe prompts")
    p.add_argument("--near", action="store_true", help="also count near-duplicates (needs the ml extra)")
    p.add_argument("--threshold", type=float, default=0.90, help="raw cosine for near-duplicates")
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("preview", help="print one generation prompt")
    p.add_argument("--index", type=int, default=0, help="context number")
    p.add_argument("--per-call", type=int, default=10)
    p.add_argument("--hard-negative", action="store_true")
    p.set_defaults(func=cmd_preview)

    p = sub.add_parser("generate", help="generation rounds (resumable)")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--target", type=int, default=20_000, help="distinct safe prompts wanted")
    p.add_argument("--per-call", type=int, default=10, help="prompts asked per request")
    p.add_argument("--hard-negative-rate", type=float, default=0.2)
    p.add_argument("--threshold", type=float, default=0.90, help="raw cosine for near-duplicates")
    p.add_argument("--exact-only", action="store_true", help="no near-duplicate detection")
    p.add_argument("--min-yield", type=float, default=0.10, help="stop when a round keeps less than this")
    p.add_argument("--max-rounds", type=int, default=12)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--effort", default="low", choices=["low", "medium", "high"])
    p.add_argument("--max-tokens", type=int, default=8000)
    p.add_argument("--backend", choices=["sync", "batch"], default="sync",
                   help="batch: Message Batches API, half price, slower")
    p.add_argument("--workers", type=int, default=8, help="concurrent requests (sync backend)")
    p.add_argument("--no-fallbacks", action="store_true",
                   help="sync backend: do not re-run declined requests on a fallback model")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("assemble", help="write the version 2 scenario files")
    p.add_argument("--out", required=True, help="directory of the generation run")
    p.add_argument("--scenario-out", required=True, help="root of the version 2 scenarios")
    p.add_argument("--dev-fraction", type=float, default=0.2, help="share of documents in the dev split")
    p.add_argument("--overwrite", action="store_true", help="replace an existing --scenario-out directory")
    p.set_defaults(func=cmd_assemble)

    p = sub.add_parser("sample", help="human annotation sample")
    p.add_argument("--out", required=True, help="directory of the generation run")
    p.add_argument("--size", type=int, default=500)
    p.set_defaults(func=cmd_sample)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
