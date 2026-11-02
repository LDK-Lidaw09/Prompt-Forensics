#!/usr/bin/env python3
"""Five-minute demonstration of PSI on a laptop (no GPU, no model server).

1. Records six real interactions of the version 2 runs (three attacks, three
   legitimate prompts, with the responses that Mistral-7B gave) in a signed,
   hash-chained evidence database, and anchors the head of the chain.
2. Verifies the database: no problem is found.
3. Tampers with copies of the database in four ways and shows which check
   detects each change.
4. Prints the main results of the version 2 runs from ``results/v2/tables``.

    python scripts/demo.py                 # writes into demo_output/
    python scripts/demo.py --out somewhere

The database it leaves behind (``demo_output/evidence.sqlite``) can be opened
with any SQLite viewer, and checked again with
``python scripts/verify_evidence.py demo_output/evidence.sqlite
--public-key demo_output/demo_key.pub.pem``.
"""

import argparse
import csv
import gzip
import json
import shutil
import sqlite3
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from prompt_forensics.canonical import build_interaction_payload  # noqa: E402
from prompt_forensics.scenarios import load_scenarios, scenario_id  # noqa: E402
from prompt_forensics.signing import load_or_create_key, public_key_path  # noqa: E402
from prompt_forensics.storage import EvidenceStore  # noqa: E402
from tamper_experiment import (  # noqa: E402
    a_edit_response, a_edit_response_rechain, a_edit_response_rehash, a_truncate_end, rows, write_rows,
)

SAMPLE = ROOT / "data" / "second_judge" / "sample_20k.jsonl.gz"
TABLES = ROOT / "results" / "v2" / "tables"


def title(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def short(text: str, width: int = 74) -> str:
    return textwrap.shorten(" ".join(str(text).split()), width=width, placeholder=" ...")


def pick_interactions(per_mode: int = 3) -> list[tuple[dict, dict]]:
    """First replays of the sample, ``per_mode`` attacks and ``per_mode`` safe prompts."""
    scenarios = {scenario_id(s): s for s in load_scenarios(ROOT / "data" / "scenarios_v2")}
    picked: dict[str, list] = {"attacks": [], "safe": []}
    with gzip.open(SAMPLE, "rt", encoding="utf-8") as handle:
        for line in handle:
            replay = json.loads(line)
            mode = replay["run_mode"]
            scenario = scenarios.get(replay["scenario_id"])
            if scenario is None or len(picked[mode]) >= per_mode or not replay["text"].strip():
                continue
            picked[mode].append((replay, scenario))
            if all(len(v) >= per_mode for v in picked.values()):
                break
    return picked["attacks"] + picked["safe"]


def record(out: Path) -> tuple[Path, str, dict]:
    title("1. Recording six interactions in a signed, hash-chained evidence database")
    db = out / "evidence.sqlite"
    for path in (db, out / "demo_key.pem", out / "demo_key.pub.pem"):
        path.unlink(missing_ok=True)
    key = load_or_create_key(out / "demo_key.pem")
    store = EvidenceStore(db, key)
    for replay, scenario in pick_interactions():
        mode = replay["run_mode"]
        prompt = scenario["attack_prompt"] if mode == "attacks" else scenario["safe_prompt"]
        payload = build_interaction_payload(
            prompt=prompt, system=scenario.get("system_prompt"), model=replay["target_model"],
            params={"temperature": float(replay["temperature"]), "seed": int(replay["seed"]),
                    "max_tokens": int(replay["max_tokens"])},
            timestamp="2026-11-01T12:00:00Z", session_id="demo")
        stored = store.append_interaction(
            payload=payload, response_text=replay["text"], record_type="replay",
            model_used=replay["target_model"], provider="vllm",
            metadata={"scenario_id": replay["scenario_id"], "run_mode": mode})
        kind = "attack" if mode == "attacks" else "legitimate"
        print(f"\n  #{stored.id} {kind:10s} {replay['scenario_id']}  "
              f"Prompt ID {stored.prompt_id[:16]}...")
        print(f"     prompt:   {short(prompt)}")
        print(f"     response: {short(replay['text'])}")
    anchor = store.anchor()
    (out / "anchor.json").write_text(json.dumps(anchor, indent=2), encoding="utf-8")
    print(f"\n  Anchor kept outside the database: {anchor['records']} records, "
          f"head {anchor['chain_hash'][:16]}... ({out / 'anchor.json'})")
    print("  (scripts/anchor_evidence.py can also have it time-stamped by an RFC 3161 authority)")
    return db, public_key_path(out / "demo_key.pem").read_text(), anchor


def verify(db: Path, public_key: str, anchor: dict) -> list[str]:
    return EvidenceStore(db).verify(public_key, anchors=[anchor]).problems


TAMPERING = [
    ("Change the text of response #2", a_edit_response),
    ("Change response #2 and recompute its SHA-256 hash", a_edit_response_rehash),
    ("Change response #2 and rebuild the whole hash chain (no key)", a_edit_response_rechain),
    ("Delete the last records (truncation)", a_truncate_end),
]


def tamper(db: Path, public_key: str, anchor: dict, out: Path) -> None:
    title("2. Verifying the untouched database")
    problems = verify(db, public_key, anchor)
    print(f"  {len(problems)} problem(s) found" + ("" if problems else ": the record is intact."))

    title("3. Tampering with copies of the database")
    print("  The attacker edits the SQLite file directly and knows every algorithm,")
    print("  but not the private signing key nor the external anchor.")

    class FixedRng:  # makes the truncation remove exactly two records
        def randint(self, a, b):
            return 2

    for label, attack in TAMPERING:
        copy = out / "tampered.sqlite"
        shutil.copy(db, copy)
        conn = sqlite3.connect(copy)
        try:
            conn.execute("DROP TRIGGER IF EXISTS interactions_no_update")
            conn.execute("DROP TRIGGER IF EXISTS interactions_no_delete")
            rs = rows(conn)
            attack(rs, 1, FixedRng(), None)
            write_rows(conn, rs)
            conn.commit()
        finally:
            conn.close()  # Windows cannot delete a file that is still open
        found = verify(copy, public_key, anchor)
        print(f"\n  {label}")
        print(f"    -> {'DETECTED' if found else 'NOT DETECTED'}: {found[0] if found else ''}")
        copy.unlink()


def results() -> None:
    title("4. Main results of the version 2 runs (results/v2/tables)")
    with open(TABLES / "evaluator_tail_risk.csv", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["victim"].startswith("Mistral") and row["group"] == "all":
                print(f"  Mistral-7B, {row['mode']:7s}: {int(row['pools'])} prompts x 100 replays, "
                      f"at least one unsafe replay in {float(row['tail_unsafe']):.1%} of the prompts "
                      f"(per replay: {float(row['rate_unsafe']):.1%})")
    print()
    with open(TABLES / "investigator_summary.csv", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["rule"] == "decision":
                print(f"  Investigator {row['judge']:24s} recall {float(row['recall']):6.1%}   "
                      f"false positives on legitimate prompts {float(row['fpr']):6.2%}")
    print(f"\n  Every table: {TABLES}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="demo_output", help="folder for the demo database and key")
    args = parser.parse_args()
    # Model responses can hold characters that a Windows console cannot show.
    sys.stdout.reconfigure(errors="replace")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    db, public_key, anchor = record(out)
    tamper(db, public_key, anchor, out)
    results()


if __name__ == "__main__":
    main()
