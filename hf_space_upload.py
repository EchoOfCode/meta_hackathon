from pathlib import Path
import sys
import traceback

from huggingface_hub import HfApi


REPO_ID = "YUS200619/meta_hackathon-qwen"
ROOT = Path(r"e:\lalaland")


def upload_all() -> None:
    api = HfApi()

    # Upload Space metadata and dependency files explicitly.
    api.upload_file(
        path_or_fileobj=str(ROOT / "space" / "README.md"),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="space",
    )
    api.upload_file(
        path_or_fileobj=str(ROOT / "space" / "requirements.txt"),
        path_in_repo="requirements.txt",
        repo_id=REPO_ID,
        repo_type="space",
    )

    # Upload runtime entrypoint and environment package used by the app.
    api.upload_file(
        path_or_fileobj=str(ROOT / "app.py"),
        path_in_repo="app.py",
        repo_id=REPO_ID,
        repo_type="space",
    )
    api.upload_folder(
        folder_path=str(ROOT / "environment"),
        path_in_repo="environment",
        repo_id=REPO_ID,
        repo_type="space",
    )


try:
    upload_all()
    print("UPLOAD_SUCCEEDED")
except Exception as e:
    print("UPLOAD_FAILED")
    print(f"{type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)
