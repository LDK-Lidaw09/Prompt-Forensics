#!/usr/bin/env python3
"""Experiment B of docs/PROTOCOL_judge_repetition_integrated_path.md: complete
path in the signed evidence store.

Three stages, run with the right model served (see scripts/run_repetition.sh):

``capture`` (victim served)
    1. each prompt of ``integrated_path_sample.csv`` is sent to the victim
       through ``pipeline.capture_interaction`` (seed 1234, recorded);
    2. 10 replays per prompt through ``pipeline.replay_interaction`` with the
       recorded configuration and seed 1234 + i for replay i.
``judge`` (judge served)
    3. the Evaluator on every replay and the Investigator on the captured
       interaction, written with ``EvidenceStore.add_annotation``.
``finish`` (no model)
    4. one report per prompt, built from the store only;
    5. the head of the chain is exported (``EvidenceStore.anchor``);
    6. verification with the trusted public key and the anchor
       (``scripts/verify_evidence.py`` and ``scripts/anchor_evidence.py verify``);
    then the controlled alterations T1 to T6, each on a fresh copy of the
    store, edited directly in SQLite (the triggers are dropped first).

The store is ``<out>/evidence.sqlite``; the signing key is created in
``<out>/../keys/`` (outside the database and outside the copied results; the
public key is copied to ``<out>/public_key.pem``). Every stage can be run again after a
stop: what is already in the store is not redone.

    python scripts/run_integrated_path.py capture --out results/v2/repetition/B
    python scripts/run_integrated_path.py judge --out results/v2/repetition/B
    python scripts/run_integrated_path.py finish --out results/v2/repetition/B
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sqlite3
import subprocess
import sys
import threading
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prompt_forensics.agents.evaluator import evaluate_output  # noqa: E402
from prompt_forensics.agents.investigator import investigate_pair  # noqa: E402
from prompt_forensics.canonical import canonical_json, text_hash  # noqa: E402
from prompt_forensics.pipeline import capture_interaction, replay_interaction  # noqa: E402
from prompt_forensics.scenarios import build_victim_prompt, load_scenarios, scenario_id  # noqa: E402
from prompt_forensics.signing import load_or_create_key, public_key_path  # noqa: E402
from prompt_forensics.storage import EvidenceStore, chain_digest  # noqa: E402

VICTIM = "mistralai/Mistral-7B-Instruct-v0.3"
JUDGE = "Qwen/Qwen3-14B"
SEED_BASE, REPLAYS = 1234, 10
LAYERS = ("hash", "chain", "signature", "anchor")
_lock = threading.Lock()


# ----------------------------------------------------------------------------- helpers

def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def paths(out: Path, key: str | None = None) -> dict[str, Path]:
    return {"db": out / "evidence.sqlite", "key": Path(key) if key else out.parent / "keys" / "signing_key.pem",
            "items": out / "items.csv", "replays": out / "replays.csv", "failures": out / "failures.csv",
            "anchor": out / "anchor.json"}


def append_csv(path: Path, row: dict[str, Any]) -> None:
    with _lock:
        new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, list(row))
            if new:
                writer.writeheader()
            writer.writerow(row)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def failure(p: dict[str, Path], step: str, item: str, exc: BaseException) -> None:
    append_csv(p["failures"], {"step": step, "item": item, "error": f"{type(exc).__name__}: {exc}",
                               "created_at": now()})
    traceback.print_exc()


def sample(args) -> tuple[list[dict[str, str]], dict[str, dict]]:
    rows = read_csv(Path(args.sample))
    scenarios = {scenario_id(s): s for s in load_scenarios(args.scenario_root)}
    return rows, scenarios


def store_of(p: dict[str, Path]) -> EvidenceStore:
    return EvidenceStore(p["db"], load_or_create_key(p["key"]))


def interaction_id(store: EvidenceStore, pid: str) -> int:
    row = store.get_interaction(pid)
    if row is None:
        raise KeyError(pid)
    return int(row["id"])


# ----------------------------------------------------------------------------- stage 1-2

def cmd_capture(args) -> None:
    p = paths(Path(args.out), args.key)
    store = store_of(p)
    rows, scenarios = sample(args)
    captured = {r["item"]: r["prompt_id"] for r in read_csv(p["items"])}
    done = {(r["item"], r["replay"]) for r in read_csv(p["replays"])}

    def capture(row: dict[str, str]) -> None:
        if row["item"] in captured:
            return
        try:
            prompt = build_victim_prompt(scenarios[row["scenario_id"]], row["run_mode"])
            result = capture_interaction(prompt, model=args.victim, user_id="integrated-path",
                                         temperature=0.2, max_tokens=512, seed=SEED_BASE,
                                         run_self_audit=False, store=store)
            captured[row["item"]] = result["prompt_id"]
            append_csv(p["items"], {**row, "prompt_id": result["prompt_id"], "created_at": now()})
        except Exception as exc:   # noqa: BLE001 - every failure is recorded with its cause
            failure(p, "capture", row["item"], exc)

    def replay(item: str, i: int) -> None:
        if (item, str(i)) in done or item not in captured:
            return
        try:
            result = replay_interaction(captured[item], seed=SEED_BASE + i, store=store)
            append_csv(p["replays"], {"item": item, "replay": i, "source_prompt_id": captured[item],
                                      "prompt_id": result["prompt_id"], "created_at": now()})
        except Exception as exc:   # noqa: BLE001
            failure(p, "replay", f"{item}#{i}", exc)

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(capture, rows))
    print(f"captured {len(captured)} of {len(rows)} prompts")
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(lambda job: replay(*job), [(r["item"], i) for r in rows for i in range(REPLAYS)]))
    print(f"replays recorded: {len(read_csv(p['replays']))}")


# ----------------------------------------------------------------------------- stage 3

def cmd_judge(args) -> None:
    p = paths(Path(args.out), args.key)
    store = store_of(p)
    rows, scenarios = sample(args)
    by_item = {r["item"]: r for r in rows}

    def has(iid: int, kind: str) -> bool:
        return any(a["kind"] == kind for a in store.annotations_for(iid))

    def evaluator(rep: dict[str, str]) -> None:
        try:
            iid = interaction_id(store, rep["prompt_id"])
            if has(iid, "evaluator"):
                return
            text = store.get_interaction(rep["prompt_id"])["response_text"]
            verdict, raw, provenance = evaluate_output(
                text, judge_model=args.judge, temperature=0.0, max_tokens=512,
                seed=777 + int(rep["replay"]), guided_json=True, guided_backend="xgrammar")
            if not verdict.ok:
                raise RuntimeError(f"judge failed: {verdict.error}")
            store.add_annotation(iid, "evaluator", {**verdict.as_dict(), "raw_text": raw,
                                                    "provenance": provenance}, args.judge)
        except Exception as exc:   # noqa: BLE001
            failure(p, "evaluator", f"{rep['item']}#{rep['replay']}", exc)

    def investigator(item: dict[str, str]) -> None:
        try:
            iid = interaction_id(store, item["prompt_id"])
            if has(iid, "investigator"):
                return
            text = store.get_interaction(item["prompt_id"])["response_text"]
            report, raw, timing = investigate_pair(
                scenarios[item["scenario_id"]], item["run_mode"], text, args.judge,
                params={"temperature": 0.0, "max_tokens": 640, "seed": 777, "repetition_penalty": 1.3},
                guided_json=True, guided_backend="xgrammar", dataset_max_chars=1000)
            if not report.ok:
                raise RuntimeError(f"judge failed: {report.error}")
            store.add_annotation(iid, "investigator", {**report.as_dict(), "raw_text": raw}, args.judge)
        except Exception as exc:   # noqa: BLE001
            failure(p, "investigator", item["item"], exc)

    replays = [r for r in read_csv(p["replays"]) if r["item"] in by_item]
    items = [r for r in read_csv(p["items"]) if r["item"] in by_item]
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(evaluator, replays))
        list(pool.map(investigator, items))
    print(f"judged {len(replays)} replays and {len(items)} captured prompts")


# ----------------------------------------------------------------------------- stage 4-6

def build_reports(store: EvidenceStore, p: dict[str, Path]) -> list[dict[str, Any]]:
    """One report per prompt, read from the store only."""
    replays: dict[str, list[dict]] = {}
    for rep in read_csv(p["replays"]):
        replays.setdefault(rep["item"], []).append(rep)
    reports = []
    for item in read_csv(p["items"]):
        source = store.get_interaction(item["prompt_id"])
        inv = [json.loads(a["result_json"]) for a in store.annotations_for(int(source["id"]))
               if a["kind"] == "investigator"]
        per_replay = []
        for rep in sorted(replays.get(item["item"], []), key=lambda r: int(r["replay"])):
            rec = store.get_interaction(rep["prompt_id"])
            meta = json.loads(rec["metadata_json"])
            ev = [json.loads(a["result_json"]) for a in store.annotations_for(int(rec["id"]))
                  if a["kind"] == "evaluator"]
            config = meta.get("replay_config", {})
            per_replay.append({
                "replay": int(rep["replay"]), "prompt_id": rec["prompt_id"],
                "linked": meta.get("source_prompt_id") == item["prompt_id"],
                "verdict": ev[-1]["verdict"] if ev else None,
                "unsafe": ev[-1]["unsafe"] if ev else None,
                "divergences": config.get("divergences"),
                "faithful": config.get("faithful"),
            })
        report = {
            "item": item["item"], "run_mode": item["run_mode"], "scenario_id": item["scenario_id"],
            "prompt_id": item["prompt_id"],
            "recorded_config": json.loads(source["canonical_payload"]).get("params"),
            "replays": per_replay,
            "replays_judged": sum(r["verdict"] is not None for r in per_replay),
            "unsafe_answers": sum(bool(r["unsafe"]) for r in per_replay),
            "investigator_decision": inv[-1].get("decision") if inv else None,
            "investigator_attack_succeeded": inv[-1].get("attack_succeeded") if inv else None,
        }
        if not any(a["kind"] == "report" for a in store.annotations_for(int(source["id"]))):
            store.add_annotation(int(source["id"]), "report", report)
        reports.append(report)
    return reports


def layer_of(problem: str) -> str:
    if problem.startswith("anchor"):
        return "anchor"
    if "signature" in problem or "not signed" in problem:
        return "signature"
    if "chain" in problem:
        return "chain"
    return "hash"


def drop_triggers(conn: sqlite3.Connection) -> None:
    for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'").fetchall():
        conn.execute(f"DROP TRIGGER {name}")


def alter(case: str, db: Path, key_path: Path, rng: random.Random) -> str:
    """Apply one alteration to ``db`` directly; returns what was changed."""
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    drop_triggers(conn)
    replays = [dict(r) for r in conn.execute(
        "SELECT * FROM interactions WHERE record_type='replay' ORDER BY id").fetchall()]
    n = conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
    middle = [r for r in replays if 1 < r["id"] < n]
    target = rng.choice(middle)
    if case == "T1":   # one word of one victim answer
        words = target["response_text"].split(" ")
        i = rng.randrange(len(words))
        words[i] = "never" if words[i] != "never" else "always"
        conn.execute("UPDATE interactions SET response_text=? WHERE id=?", (" ".join(words), target["id"]))
        what = f"interaction {target['id']}: word {i} of the answer replaced"
    elif case == "T2":   # recorded seed of one replay
        payload = json.loads(target["canonical_payload"])
        payload["params"]["seed"] = int(payload["params"]["seed"]) + 1
        conn.execute("UPDATE interactions SET canonical_payload=? WHERE id=?",
                     (canonical_json(payload), target["id"]))
        what = f"interaction {target['id']}: recorded seed {payload['params']['seed'] - 1} -> {payload['params']['seed']}"
    elif case == "T3":   # one replay record in the middle
        conn.execute("DELETE FROM interactions WHERE id=?", (target["id"],))
        what = f"interaction {target['id']} deleted"
    elif case == "T4":   # the last records
        k = 10
        conn.execute("DELETE FROM interactions WHERE id > ?", (n - k,))
        what = f"last {k} interactions deleted"
    elif case == "T5":   # one Evaluator verdict, with its own hash recomputed
        ann = conn.execute("SELECT * FROM annotations WHERE kind='evaluator' ORDER BY id").fetchall()
        a = dict(rng.choice(ann))
        result = json.loads(a["result_json"])
        old = result["verdict"]
        result["verdict"] = "safe" if old != "safe" else "malicious"
        result["unsafe"] = result["verdict"] != "safe"
        result["malicious"] = result["verdict"] == "malicious"
        result_json = json.dumps(result, ensure_ascii=False, sort_keys=True)
        from prompt_forensics.canonical import sha256_hex
        conn.execute("UPDATE annotations SET result_json=?, result_hash=? WHERE id=?",
                     (result_json, sha256_hex(result_json), a["id"]))
        what = f"annotation {a['id']} (interaction {a['interaction_id']}): verdict {old} -> {result['verdict']}"
    elif case == "T6":   # forged answer, chain rebuilt and re-signed with the stolen key
        key = load_or_create_key(key_path)
        rows = [dict(r) for r in conn.execute("SELECT * FROM interactions ORDER BY id").fetchall()]
        start = next(i for i, r in enumerate(rows) if r["id"] == target["id"])
        rows[start]["response_text"] = "I cannot help with that request."
        rows[start]["response_hash"] = text_hash(rows[start]["response_text"])
        prev = rows[start - 1]["chain_hash"]
        for r in rows[start:]:
            r["prev_chain_hash"] = prev
            r["chain_hash"] = chain_digest(r)
            r["signature"] = key.sign_digest(r["chain_hash"])
            prev = r["chain_hash"]
            conn.execute("UPDATE interactions SET response_text=?, response_hash=?, prev_chain_hash=?, "
                         "chain_hash=?, signature=? WHERE id=?",
                         (r["response_text"], r["response_hash"], r["prev_chain_hash"], r["chain_hash"],
                          r["signature"], r["id"]))
        what = f"interaction {target['id']}: answer forged, chain rebuilt and re-signed from there"
    else:
        raise ValueError(case)
    conn.commit()
    conn.close()
    return what


def run_cli(args: list[str]) -> dict[str, Any]:
    done = subprocess.run([sys.executable, *args], capture_output=True, text=True, cwd=ROOT)
    return {"command": " ".join(args), "exit_code": done.returncode, "output": done.stdout.strip()}


def cmd_finish(args) -> None:
    out = Path(args.out)
    p = paths(out, args.key)
    store = store_of(p)
    rows, _ = sample(args)
    reports = build_reports(store, p)
    with open(out / "reports.jsonl", "w", encoding="utf-8") as handle:
        for report in reports:
            handle.write(json.dumps(report, ensure_ascii=False) + "\n")

    anchor = store.anchor()
    p["anchor"].write_text(json.dumps(anchor, indent=2) + "\n")
    public = out / "public_key.pem"          # archived with the results
    shutil.copy(public_key_path(p["key"]), public)
    trusted = public.read_text()

    counts = Counter(r["record_type"] for r in store.recent_interactions(10**6))
    with sqlite3.connect(p["db"]) as conn:
        annotations = dict(conn.execute("SELECT kind, COUNT(*) FROM annotations GROUP BY kind").fetchall())
    steps = {
        "prompts": len(rows), "captured": len(read_csv(p["items"])), "replays": len(read_csv(p["replays"])),
        "records": dict(counts), "annotations": annotations,
        "failures": dict(Counter(f["step"] for f in read_csv(p["failures"]))),
        "reports": len(reports), "anchor": anchor,
    }

    # Replay configuration: recorded vs used, by field and origin.
    config = Counter()
    for report in reports:
        for rep in report["replays"]:
            for d in rep["divergences"] or []:
                config[(d["field"], d["origin"], str(d["recorded"]), str(d["used"]))] += 1
    with open(out / "replay_config.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["field", "origin", "recorded", "used", "replays"])
        for (fieldname, origin, recorded, used), k in sorted(config.items()):
            writer.writerow([fieldname, origin, recorded, used, k])
    steps["replays_with_divergence"] = sum(bool(rep["divergences"]) for r in reports for rep in r["replays"])
    steps["replays_unlinked"] = sum(not rep["linked"] for r in reports for rep in r["replays"])

    # Verification of the unaltered store.
    clean = store.verify(trusted, [anchor])
    steps["verification"] = {"checked": clean.checked, "problems": clean.problems,
                             "cli": [run_cli(["scripts/verify_evidence.py", str(p["db"]), "--public-key", str(public)]),
                                     run_cli(["scripts/anchor_evidence.py", "verify", str(p["db"]),
                                              str(p["anchor"]), "--public-key", str(public)])]}

    # Controlled alterations, each on a fresh copy.
    rng = random.Random(args.seed)
    results = []
    copies = out / "altered"
    copies.mkdir(exist_ok=True)
    for case in ("T1", "T2", "T3", "T4", "T5", "T6"):
        copy = copies / f"{case}.sqlite"
        shutil.copy(p["db"], copy)
        what = alter(case, copy, p["key"], rng)
        problems = EvidenceStore(copy).verify(trusted, [anchor]).problems
        without_anchor = EvidenceStore(copy).verify(trusted).problems
        layers = Counter(layer_of(x) for x in problems)
        results.append({"case": case, "alteration": what, "detected": bool(problems),
                        "detected_without_anchor": bool(without_anchor),
                        **{f"problems_{layer}": layers.get(layer, 0) for layer in LAYERS},
                        "first_problems": " | ".join(problems[:3])})
    with open(out / "alterations.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    (out / "steps.json").write_text(json.dumps(steps, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in steps.items() if k != "verification"}, indent=2, default=str))
    print(f"verification: {clean.checked} records, {len(clean.problems)} problem(s)")
    for r in results:
        print(f"{r['case']}: detected={r['detected']} (without anchor: {r['detected_without_anchor']}) "
              + " ".join(f"{layer}={r[f'problems_{layer}']}" for layer in LAYERS))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["capture", "judge", "finish"])
    parser.add_argument("--out", default="results/v2/repetition/B")
    parser.add_argument("--sample", default="data/repetition/plan/integrated_path_sample.csv")
    parser.add_argument("--scenario-root", default="data/scenarios_v2")
    parser.add_argument("--victim", default=VICTIM)
    parser.add_argument("--judge", default=JUDGE)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261118, help="choice of the altered records")
    parser.add_argument("--key", help="private signing key, kept outside the store and the results "
                        "(default: <out>/../keys/signing_key.pem; created if missing)")
    args = parser.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    {"capture": cmd_capture, "judge": cmd_judge, "finish": cmd_finish}[args.stage](args)


if __name__ == "__main__":
    main()
