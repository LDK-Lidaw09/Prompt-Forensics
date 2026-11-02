#!/usr/bin/env python3
"""Check the integrity of an evidence database (thesis 4.2.3).

Recomputes every Prompt ID, content hash and response hash, walks the hash
chain and verifies every signature. Exit code 0 means no problem was found.

    python scripts/verify_evidence.py psi_evidence.sqlite --public-key keys/psi_signing_key.pub.pem
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prompt_forensics.storage import EvidenceStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("db", help="SQLite evidence file")
    parser.add_argument("--public-key", help="trusted public key (PEM); without it, each "
                        "row's own key is used, which only proves internal consistency")
    args = parser.parse_args()

    if not Path(args.db).exists():
        sys.exit(f"{args.db} does not exist")
    trusted = Path(args.public_key).read_text() if args.public_key else None
    report = EvidenceStore(args.db).verify(trusted)
    print(f"{report.checked} interactions checked, {len(report.problems)} problem(s)")
    for problem in report.problems:
        print(f"  - {problem}")
    sys.exit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
