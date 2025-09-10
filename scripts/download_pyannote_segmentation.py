#!/usr/bin/env python3
"""
Download script for pyannote/segmentation-3.0
Usage: python ps06_system/scripts/download_pyannote_segmentation.py
"""
import os
import sys
from pathlib import Path
from datetime import datetime

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

from huggingface_hub import snapshot_download


def main() -> int:
    # Resolve paths relative to repo
    base_dir = Path(__file__).resolve().parent.parent  # ps06_system
    models_dir = base_dir / "models" / "pyannote" / "segmentation"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load env from ps06_system/.env if available
    if load_dotenv is not None:
        load_dotenv(dotenv_path=base_dir / ".env")

    token = (
        os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HF_TOKEN")
        or os.getenv("PYANNOTE_ACCESS_TOKEN")
    )

    print("Downloading: pyannote/segmentation-3.0")
    print(f"  Target dir: {models_dir}")
    if token:
        print(f"  Using token: {token[:8]}…")
    else:
        print("  No token found in env; gated access may fail (403)")

    try:
        snapshot_download(
            repo_id="pyannote/segmentation-3.0",
            local_dir=str(models_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
            token=token if token else None,
        )

        # Create completion marker
        marker = models_dir / ".download_complete"
        marker.write_text(f"{datetime.utcnow().isoformat()}\npyannote/segmentation-3.0\n")

        print("Success: pyannote/segmentation-3.0 downloaded.")
        return 0

    except Exception as e:
        print(f"Error: Failed to download pyannote/segmentation-3.0: {e}")
        print("Hint: This repository is gated on Hugging Face.")
        print("  Visit and request access: https://huggingface.co/pyannote/segmentation-3.0")
        print("  Set HUGGINGFACE_HUB_TOKEN (or HUGGINGFACE_TOKEN) and retry.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

