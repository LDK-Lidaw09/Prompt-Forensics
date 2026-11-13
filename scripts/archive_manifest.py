#!/usr/bin/env python3
"""Write a SHA-256 manifest of the result files archived on the Hugging Face Hub.

The raw outputs of the runs live in a dataset repository on the Hub. That
repository is an archive, not an evidence store: its files are not chained,
signed or anchored. The manifest written here is published with the archived
version of the thesis, so that a reader can check that the files have not
changed since the archiving date. It says nothing about the files before that
date.

The manifest is pinned to one commit of the repository. For a file stored
with LFS or Xet, the Hub already gives the SHA-256 of its content, so nothing
is downloaded; small files stored directly in git are downloaded one at a
time and hashed.

    python scripts/archive_manifest.py --repo Latavious/prompt-forensics-runs-v2 \\
        --revision main --out archive/MANIFEST

writes two files:

- ``MANIFEST.sha256``: one line per file, ``<sha256>  <path>``, the format
  of ``sha256sum``. A reader who downloads the repository at the same
  revision checks it with ``sha256sum -c MANIFEST.sha256`` from the
  repository folder.
- ``MANIFEST.json``: the repository, the resolved commit, the date, and the
  size and hash of each file.

``--local DIR`` hashes a local folder instead of the Hub repository (for
example a downloaded snapshot), and ``--check DIR`` verifies a folder
against an existing ``MANIFEST.sha256`` without needing ``sha256sum``.

The Hub mode needs ``huggingface_hub`` (``pip install huggingface_hub``,
already used by ``backup_to_hf.py``). The token (``HF_TOKEN``) needs read
access while the repository is private. A ``--local`` folder must not change
while it is hashed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

CHUNK = 1 << 20


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def hub_entries(repo: str, revision: str) -> tuple[str, list[dict]]:
    """Return the resolved commit and one entry per file of the repository."""
    try:
        from huggingface_hub import HfApi, hf_hub_download
        from huggingface_hub.hf_api import RepoFile
    except ImportError:
        sys.exit("huggingface_hub is required for the Hub mode: pip install huggingface_hub")

    api = HfApi()
    commit = api.dataset_info(repo, revision=revision).sha
    files = sorted(
        (f for f in api.list_repo_tree(repo, repo_type="dataset", recursive=True, revision=commit)
         if isinstance(f, RepoFile)),
        key=lambda f: f.path,
    )
    entries = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, f in enumerate(files, 1):
            if f.lfs is not None:
                digest, source = f.lfs.sha256, "lfs"
            else:
                local = hf_hub_download(repo, f.path, repo_type="dataset", revision=commit,
                                        local_dir=tmp)
                digest, source = sha256_file(local), "downloaded"
                os.remove(local)
            entries.append({"path": f.path, "size": f.size, "sha256": digest, "source": source})
            if i % 100 == 0 or i == len(files):
                print(f"{i}/{len(files)} files", file=sys.stderr)
    return commit, entries


def list_files(folder: Path, skip: set[Path] = frozenset()) -> list[tuple[str, Path]]:
    """Relative POSIX path and absolute path of every file under ``folder``."""
    if not folder.is_dir():
        sys.exit(f"not a folder: {folder}")
    found = []
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        rel = path.relative_to(folder).as_posix()
        if rel.startswith((".cache/", ".git/")) or path.resolve() in skip:
            continue
        found.append((rel, path))
    return found


def local_entries(folder: Path, skip: set[Path]) -> list[dict]:
    entries = []
    for rel, path in list_files(folder, skip):
        entries.append({"path": rel, "size": path.stat().st_size,
                        "sha256": sha256_file(path), "source": "local"})
    return entries


def write_manifest(out: Path, meta: dict, entries: list[dict]) -> None:
    for e in entries:
        # The sha256sum format has one name per line; such names would need escaping.
        if "\n" in e["path"] or "\\" in e["path"]:
            sys.exit(f"unsupported file name (newline or backslash): {e['path']!r}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".sha256"), "w", encoding="utf-8", newline="\n") as handle:
        for e in entries:
            handle.write(f"{e['sha256']}  {e['path']}\n")
    with open(out.with_suffix(".json"), "w", encoding="utf-8") as handle:
        json.dump({**meta, "files": entries}, handle, indent=1, ensure_ascii=False)
        handle.write("\n")
    total = sum(e["size"] for e in entries)
    print(f"{len(entries)} files, {total / 1e9:.2f} GB -> {out.with_suffix('.sha256')}, "
          f"{out.with_suffix('.json')}")


def check(manifest: Path, folder: Path, skip: set[Path]) -> int:
    bad = missing = 0
    lines = manifest.read_text(encoding="utf-8").splitlines()
    listed = set()
    for line in lines:
        digest, rel = line.split("  ", 1)
        listed.add(rel)
        path = folder / rel
        if not path.is_file():
            print(f"MISSING  {rel}")
            missing += 1
        elif sha256_file(path) != digest:
            print(f"CHANGED  {rel}")
            bad += 1
    extra = [rel for rel, _ in list_files(folder, skip) if rel not in listed]
    for rel in extra:
        print(f"EXTRA    {rel}")
    print(f"{len(lines)} files checked: {bad} changed, {missing} missing, {len(extra)} not in the manifest")
    return 0 if bad == missing == len(extra) == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", default="Latavious/prompt-forensics-runs-v2")
    parser.add_argument("--revision", default="main",
                        help="branch, tag or commit; the manifest records the resolved commit")
    parser.add_argument("--local", type=Path, help="hash this folder instead of the Hub repository")
    parser.add_argument("--out", type=Path, default=Path("archive/MANIFEST"),
                        help="output path without extension (default: archive/MANIFEST)")
    parser.add_argument("--check", type=Path, metavar="DIR",
                        help="verify DIR against <out>.sha256 instead of writing a manifest")
    args = parser.parse_args()

    # The manifest files themselves are never part of what they describe.
    skip = {args.out.with_suffix(ext).resolve() for ext in (".sha256", ".json")}
    if args.check:
        return check(args.out.with_suffix(".sha256"), args.check, skip)

    created = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    if args.local:
        meta = {"source": "local", "folder": str(args.local), "created_utc": created}
        entries = local_entries(args.local, skip)
    else:
        commit, entries = hub_entries(args.repo, args.revision)
        meta = {"source": "huggingface", "repo": args.repo, "repo_type": "dataset",
                "revision": commit, "created_utc": created}
        print(f"{args.repo} at {commit}", file=sys.stderr)
    write_manifest(args.out, meta, entries)
    return 0


if __name__ == "__main__":
    sys.exit(main())
