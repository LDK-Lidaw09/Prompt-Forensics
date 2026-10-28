#!/usr/bin/env python3
"""Copy a results folder to a private dataset repository on the Hugging Face Hub.

Used by ``run_gpu_shard.sh`` during the version 2 runs, so that the replays
and scores survive the loss of a pod. Every call uploads the files that
changed since the last one (the Hub stores files in content-defined chunks,
so a JSONL file that grew only sends its new part).

    python scripts/backup_to_hf.py results/v2/shard0 --path-in-repo shard0
    python scripts/backup_to_hf.py results/v2/shard0 --repo someone/prompt-forensics-runs-v2

The token (``HF_TOKEN``) needs the write permission. Without ``--repo`` the
repository is ``<token owner>/prompt-forensics-runs-v2``, created private on
first use.
"""

import argparse
import fnmatch
import os
import shutil
import sys
import tempfile

from huggingface_hub import HfApi

DEFAULT_NAME = "prompt-forensics-runs-v2"
# Markers and archives are rebuilt from the rest.
IGNORE = [".done_*", "*.tar.gz"]


def snapshot(folder: str, dest: str) -> None:
    """Copy ``folder`` to ``dest``, each file cut at the size it had when listed.

    The runs keep appending to their JSONL files; uploading a file that grows
    during the upload makes the Hub reject the commit ("LFS pointer pointed to
    a file that does not exist"). The copy is a fixed prefix of each file.
    """
    for root, _, files in os.walk(folder):
        for name in files:
            if any(fnmatch.fnmatch(name, pattern) for pattern in IGNORE):
                continue
            src = os.path.join(root, name)
            dst = os.path.join(dest, os.path.relpath(src, folder))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(src, "rb") as reader, open(dst, "wb") as writer:
                remaining = os.path.getsize(src)
                while remaining > 0:
                    chunk = reader.read(min(remaining, 1 << 24))
                    if not chunk:
                        break
                    writer.write(chunk)
                    remaining -= len(chunk)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", help="local folder to copy")
    parser.add_argument("--repo", help=f"dataset repository id (default: <user>/{DEFAULT_NAME})")
    parser.add_argument("--path-in-repo", default="", help="folder inside the repository")
    parser.add_argument("--message", default="Backup of the version 2 runs")
    args = parser.parse_args()

    api = HfApi()
    repo = args.repo or f"{api.whoami()['name']}/{DEFAULT_NAME}"
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    # The snapshot sits next to the folder (same disk, the pod volume).
    parent = os.path.dirname(os.path.abspath(args.folder))
    copy = tempfile.mkdtemp(prefix=".backup_", dir=parent)
    try:
        snapshot(args.folder, copy)
        api.upload_folder(
            repo_id=repo, repo_type="dataset", folder_path=copy,
            path_in_repo=args.path_in_repo, commit_message=args.message,
        )
    finally:
        shutil.rmtree(copy, ignore_errors=True)
    print(f"{args.folder} -> https://huggingface.co/datasets/{repo}/tree/main/{args.path_in_repo}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # a failed backup must not stop the runs
        sys.exit(f"backup failed: {type(exc).__name__}: {exc}")
