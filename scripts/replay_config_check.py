"""Demonstrate the replay of the recorded configuration on five cases (thesis Section 4.5).

Runs offline with the simulator provider in a temporary evidence store and
writes, for each case, the replay configuration that PSI stores with the
replay (metadata.replay_config):

1. replay of a record with its recorded configuration;
2. explicit change of one parameter (the seed);
3. record that holds no generation parameters;
4. record whose stored payload was changed after capture (the replay is refused);
5. server that declares another model than the one requested. The simulator
   always declares the requested model, so this case wraps it to declare
   "victim-v0.3", as a server serving another version would.

Usage: python scripts/replay_config_check.py [output.json]
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

tmp = Path(tempfile.mkdtemp(prefix="replay_check_"))
os.environ.update(DB_PATH=str(tmp / "evidence.sqlite"), SIGNING_KEY_PATH=str(tmp / "key.pem"),
                  LIVE_MODEL="sim:victim", JUDGE_MODEL="sim:judge")

from prompt_forensics import pipeline
from prompt_forensics.canonical import build_interaction_payload
from prompt_forensics.llm import LLMResponse
from prompt_forensics.pipeline import (
    capture_interaction,
    get_store,
    replay_interaction,
)

out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "replay_check" / "replay_config_check.json"
store = get_store()
cases = []


def record(name: str, description: str, fn) -> None:
    try:
        result = fn()
        cases.append({"case": name, "description": description, "outcome": "replayed",
                      "faithful": result["replay_config"]["faithful"],
                      "params_sent": result["params"],
                      "divergences": result["replay_config"]["divergences"],
                      "not_recorded": result["replay_config"]["not_recorded"]})
    except ValueError as exc:
        cases.append({"case": name, "description": description, "outcome": "refused", "error": str(exc)})


src = capture_interaction("Summarise the audit rules.", run_self_audit=False,
                          temperature=0.2, max_tokens=512, seed=1234)
record("1 recorded configuration", "replay without any argument",
       lambda: replay_interaction(src["prompt_id"]))
record("2 explicit change", "replay with seed=1235",
       lambda: replay_interaction(src["prompt_id"], seed=1235))

bare = store.append_interaction(
    payload=build_interaction_payload(prompt="Summarise the audit rules.", model="sim:victim", params={},
                                      timestamp="2026-11-17T00:00:00Z", user_id="analyst"),
    response_text="(recorded answer)", record_type="live")
record("3 parameters absent from the record", "record with params = {}",
       lambda: replay_interaction(bare.prompt_id))

altered = capture_interaction("Summarise the audit rules.", run_self_audit=False,
                              temperature=0.2, max_tokens=512, seed=1234)
with sqlite3.connect(store.db_path) as conn:
    for (trigger,) in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'").fetchall():
        conn.execute(f"DROP TRIGGER {trigger}")
    conn.execute("UPDATE interactions SET canonical_payload = replace(canonical_payload, '\"seed\":1234', "
                  "'\"seed\":99') WHERE prompt_id = ?", (altered["prompt_id"],))
record("4 altered record", "stored payload changed after capture (seed 1234 -> 99)",
       lambda: replay_interaction(altered["prompt_id"]))

real_call = pipeline.call_llm


def other_version(prompt, model, params):
    r = real_call(prompt, model, params)
    return LLMResponse(text=r.text, model="victim-v0.3", provider=r.provider, timing_ms=0)


pipeline.call_llm = other_version
record("5 served model differs", "server declares victim-v0.3 for the requested sim:victim",
       lambda: replay_interaction(src["prompt_id"]))
pipeline.call_llm = real_call

commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True, check=False).stdout.strip()
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps({"code_commit": commit, "cases": cases}, indent=2, ensure_ascii=False) + "\n")
for c in cases:
    detail = c.get("error") or "; ".join(f"{d['field']}: {d['recorded']} -> {d['used']} ({d['origin']})"
                                          for d in c["divergences"]) or "no difference"
    print(f"{c['case']:38} {c['outcome']:9} {detail}")
print("written", out_path)
