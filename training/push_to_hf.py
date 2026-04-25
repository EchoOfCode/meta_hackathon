from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload trained checkpoint folder to Hugging Face Hub.")
    parser.add_argument("--repo-id", required=True, help="e.g. username/work-life-firewall-qwen")
    parser.add_argument("--folder", default="checkpoints/final", help="Local model folder")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(repo_id=args.repo_id, repo_type="model", folder_path=str(folder))
    print(f"Uploaded model folder to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
