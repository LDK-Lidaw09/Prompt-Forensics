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
import sys

from huggingface_hub import HfApi

DEFAULT_NAME = "prompt-forensics-runs-v2"


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
    api.upload_folder(
        repo_id=repo, repo_type="dataset", folder_path=args.folder,
        path_in_repo=args.path_in_repo, commit_message=args.message,
        # Markers and archives are rebuilt from the rest.
        ignore_patterns=[".done_*", "*.tar.gz"],
    )
    print(f"{args.folder} -> https://huggingface.co/datasets/{repo}/tree/main/{args.path_in_repo}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # a failed backup must not stop the runs
        sys.exit(f"backup failed: {type(exc).__name__}: {exc}")
