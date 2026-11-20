"""Upload / download the trained weights to / from the Hugging Face Hub (R7:
no large weight files in the git repository).

    python -m src.hub push   # after training (needs `huggingface-cli login`)
    python -m src.hub pull   # on a new machine (also done automatically by app.py)
"""
from __future__ import annotations

import sys

from .config import HF_REPO_ID, MODELS_DIR

REQUIRED = ["labels.json", "vocab.json", "student/config.json"]


def models_present() -> bool:
    return all((MODELS_DIR / f).exists() for f in REQUIRED)


def pull() -> None:
    from huggingface_hub import snapshot_download
    print(f"downloading weights from https://huggingface.co/{HF_REPO_ID} ...")
    snapshot_download(repo_id=HF_REPO_ID, local_dir=str(MODELS_DIR),
                      ignore_patterns=["student_int8_state_dict.pt"])


def push() -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(HF_REPO_ID, exist_ok=True)
    api.upload_folder(folder_path=str(MODELS_DIR), repo_id=HF_REPO_ID,
                      ignore_patterns=["student_int8_state_dict.pt", "*/checkpoint-*"])
    print(f"uploaded {MODELS_DIR} -> https://huggingface.co/{HF_REPO_ID}")


def ensure_models() -> None:
    if models_present():
        return
    if HF_REPO_ID.startswith("YOUR_HF_USERNAME"):
        raise SystemExit("Models not found in ./models. Train them (python -m src.train_all) or set "
                         "ROUTER_HF_REPO / HF_REPO_ID in src/config.py to the Hub repo with the weights.")
    pull()


if __name__ == "__main__":
    {"push": push, "pull": pull}[sys.argv[1] if len(sys.argv) > 1 else "pull"]()
