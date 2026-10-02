"""Falla si el árbol de git contiene videos, directorios de datos o archivos grandes.

El repo es público: los videos y datos de evaluación nunca se versionan (docs/data/DATASETS.md).
Uso: python scripts/check_no_media.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

MEDIA_EXT = re.compile(r"\.(mp4|mkv|mov|avi|webm|m4v)$", re.IGNORECASE)
DATA_PREFIXES = ("videos/", "data/", "training_data/", "svhn_data/", "annotations/")
MAX_BYTES = 10 * 1024 * 1024
MODEL_MAX_BYTES = 20 * 1024 * 1024  # ADR-0003: modelos .onnx versionados en models/


def violations(files: dict[str, int]) -> list[str]:
    bad = []
    for path, size in sorted(files.items()):
        if MEDIA_EXT.search(path):
            bad.append(f"{path}: archivo de video")
        elif path.startswith(DATA_PREFIXES):
            bad.append(f"{path}: directorio de datos")
        elif path.startswith("models/") and path.endswith(".onnx"):
            if size > MODEL_MAX_BYTES:
                bad.append(f"{path}: modelo de {size / 2**20:.1f} MB > 20 MB (ver ADR-0003)")
        elif size > MAX_BYTES:
            bad.append(f"{path}: {size / 2**20:.1f} MB > 10 MB")
    return bad


def tracked_files(root: Path) -> dict[str, int]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True).stdout
    files = {}
    for raw in out.decode("utf-8").split("\0"):
        if raw:
            p = root / raw
            files[raw] = p.stat().st_size if p.is_file() else 0
    return files


def main(argv: list[str]) -> int:
    root = Path(argv[0]) if argv else Path(__file__).resolve().parents[1]
    bad = violations(tracked_files(root))
    for b in bad:
        print(f"::error::{b}")
    print(f"check_no_media: {len(bad)} violaciones")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
