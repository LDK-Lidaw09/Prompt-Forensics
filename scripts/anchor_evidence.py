#!/usr/bin/env python3
"""Anchor the head of an evidence database outside the system (thesis 4.2.3).

``create`` writes the current head of the hash chain (number of records and
last chain hash) to a JSON file. With ``--tsa``, the file is also sent to an
RFC 3161 time-stamping authority, whose signed reply (``.tsr``) proves that
this head existed at that time. Whoever later rewrites or truncates the
database, even with the signing key, can no longer match the anchor.

``verify`` checks the database against one or more anchors and, when a
``.tsr`` file sits next to an anchor and ``--tsa-ca`` is given, checks the
time-stamp too (``openssl ts -verify``).

    python scripts/anchor_evidence.py create psi_evidence.sqlite anchors/ --tsa https://freetsa.org/tsr
    python scripts/anchor_evidence.py verify psi_evidence.sqlite anchors/*.json --tsa-ca cacert.pem
"""

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prompt_forensics.storage import EvidenceStore  # noqa: E402


def timestamp(path: Path, tsa_url: str) -> Path:
    """Ask an RFC 3161 authority to time-stamp ``path``; returns the ``.tsr`` file."""
    query = subprocess.run(
        ["openssl", "ts", "-query", "-data", str(path), "-sha256", "-cert"],
        check=True, capture_output=True,
    ).stdout
    request = urllib.request.Request(
        tsa_url, data=query, headers={"Content-Type": "application/timestamp-query"})
    with urllib.request.urlopen(request, timeout=30) as response:
        reply = response.read()
    tsr = path.with_suffix(".tsr")
    tsr.write_bytes(reply)
    return tsr


def cmd_create(args) -> None:
    anchor = EvidenceStore(args.db).anchor()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"anchor_{anchor['created_at'].replace(':', '').replace('.', '_')}.json"
    path.write_text(json.dumps(anchor, indent=2) + "\n")
    print(f"{anchor['records']} records, head {anchor['chain_hash'][:16]}... -> {path}")
    if args.tsa:
        print(f"time-stamp -> {timestamp(path, args.tsa)}")


def cmd_verify(args) -> None:
    anchors = [json.loads(Path(p).read_text()) for p in args.anchors]
    trusted = Path(args.public_key).read_text() if args.public_key else None
    report = EvidenceStore(args.db).verify(trusted, anchors)
    problems = list(report.problems)
    for p in args.anchors:
        tsr = Path(p).with_suffix(".tsr")
        if args.tsa_ca and tsr.exists():
            result = subprocess.run(
                ["openssl", "ts", "-verify", "-data", p, "-in", str(tsr), "-CAfile", args.tsa_ca],
                capture_output=True, text=True)
            if result.returncode != 0:
                problems.append(f"{p}: time-stamp does not verify ({result.stderr.strip()})")
    print(f"{report.checked} interactions checked against {len(anchors)} anchor(s), "
          f"{len(problems)} problem(s)")
    for problem in problems:
        print(f"  - {problem}")
    sys.exit(0 if not problems else 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("create", help="write the head of the chain to an anchor file")
    p.add_argument("db")
    p.add_argument("out_dir")
    p.add_argument("--tsa", help="RFC 3161 time-stamping URL")
    p.set_defaults(func=cmd_create)
    p = sub.add_parser("verify", help="check the database against anchors")
    p.add_argument("db")
    p.add_argument("anchors", nargs="+")
    p.add_argument("--public-key", help="trusted public key (PEM)")
    p.add_argument("--tsa-ca", help="CA certificate of the time-stamping authority")
    p.set_defaults(func=cmd_verify)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
