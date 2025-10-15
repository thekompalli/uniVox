"""
Create required runtime folder structure for the PS-06 system.

This helps on fresh clones where empty directories are ignored by Git.
It also drops `.gitkeep` files so the structure persists if desired.
"""
from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# Base directories from the app config layout
DIRS = [
    ROOT / "models",
    ROOT / "models" / "cache",
    ROOT / "data",
    ROOT / "data" / "audio",
    ROOT / "data" / "results",
    ROOT / "logs",
    ROOT / "logs" / "nginx",
    ROOT / "temp",
]


def touch_gitkeep(path: Path) -> None:
    keep = path / ".gitkeep"
    if not keep.exists():
        keep.write_text("\n", encoding="utf-8")


def main() -> None:
    created = []
    for d in DIRS:
        d.mkdir(parents=True, exist_ok=True)
        touch_gitkeep(d)
        created.append(str(d.relative_to(ROOT)))

    print("Ensured directories:")
    for c in created:
        print(f" - {c}")


if __name__ == "__main__":
    main()

