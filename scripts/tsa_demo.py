#!/usr/bin/env python3
"""End-to-end check of the external anchor with a real RFC 3161 authority (thesis 4.2.5).

Builds a small signed evidence database, anchors its head, has the anchor
time-stamped by a public time-stamping authority, then checks that:

1. the clean database and the token both verify;
2. deleting the last records is detected by the anchor;
3. rewriting a record and re-signing the chain with the stolen key is
   detected by the anchor, while the signatures alone accept it.

Everything it does is written to ``<out>/tsa_demo_log.txt``; the anchor, the
token (``.tsr``) and the authority's certificate are kept next to it. Only the
SHA-256 of the anchor file is sent to the authority.

    python scripts/tsa_demo.py --out results/tsa_demo
"""

import argparse
import importlib.util
import json
import shutil
import sqlite3
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from anchor_evidence import timestamp  # noqa: E402
from prompt_forensics.signing import generate_key  # noqa: E402
from prompt_forensics.storage import EvidenceStore  # noqa: E402


def load_tamper_module():
    spec = importlib.util.spec_from_file_location("tamper_experiment", ROOT / "scripts" / "tamper_experiment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tsa", default="https://freetsa.org/tsr")
    parser.add_argument("--tsa-ca-url", default="https://freetsa.org/files/cacert.pem")
    parser.add_argument("--out", default="results/tsa_demo")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log(line: str = "") -> None:
        print(line)
        log_lines.append(line)

    tamper = load_tamper_module()
    key = generate_key()
    (out / "public_key.pem").write_text(key.public_key_pem)
    db = out / "evidence.sqlite"
    db.unlink(missing_ok=True)
    tamper.build(db, args.records, key, args.seed)
    log(f"1. Built {db} with {args.records} signed records.")

    anchor = EvidenceStore(db).anchor()
    anchor_path = out / "anchor.json"
    anchor_path.write_text(json.dumps(anchor, indent=2) + "\n")
    log(f"2. Anchor: {anchor['records']} records, head {anchor['chain_hash']}")

    tsr = timestamp(anchor_path, args.tsa)
    ca = out / "tsa_cacert.pem"
    with urllib.request.urlopen(args.tsa_ca_url, timeout=30) as response:
        ca.write_bytes(response.read())
    token = subprocess.run(["openssl", "ts", "-reply", "-in", str(tsr), "-text"],
                           capture_output=True, text=True, check=True).stdout
    when = next((line.strip() for line in token.splitlines() if line.strip().startswith("Time stamp:")), "?")
    log(f"3. Time-stamped by {args.tsa}: {when}  (token {tsr.name}, {tsr.stat().st_size} bytes)")

    def check(path: Path) -> list[str]:
        problems = list(EvidenceStore(path).verify(key.public_key_pem, [anchor]).problems)
        result = subprocess.run(["openssl", "ts", "-verify", "-data", str(anchor_path), "-in", str(tsr),
                                 "-CAfile", str(ca)], capture_output=True, text=True)
        if result.returncode != 0:
            problems.append(f"time-stamp token does not verify: {result.stderr.strip()}")
        return problems

    clean = check(db)
    log(f"4. Clean database + token: {'OK' if not clean else clean}")

    for name, attack, n in [("delete the last 10 records", "truncate", 10),
                            ("rewrite record 50 and re-sign with the stolen key", "rewrite", 50)]:
        copy = out / f"tampered_{attack}.sqlite"
        shutil.copy(db, copy)
        with sqlite3.connect(copy) as conn:
            conn.execute("DROP TRIGGER interactions_no_update")
            conn.execute("DROP TRIGGER interactions_no_delete")
            rs = tamper.rows(conn)
            if attack == "truncate":
                del rs[-n:]
            else:
                tamper.forge_response(rs[n - 1])
                tamper.rechain(rs, n - 1, key)
            tamper.write_rows(conn, rs)
        without_anchor = EvidenceStore(copy).verify(key.public_key_pem, []).problems
        with_anchor = check(copy)
        log(f"5. Attack '{name}':")
        log(f"   signatures only -> {len(without_anchor)} problem(s)")
        log(f"   with the time-stamped anchor -> {len(with_anchor)} problem(s)")
        for p in with_anchor:
            log(f"     - {p}")

    (out / "tsa_demo_log.txt").write_text("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()
