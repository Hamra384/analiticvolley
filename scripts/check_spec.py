"""Valida que cada spec en docs/specs/ tenga las secciones obligatorias y criterios verificables.

Uso: python scripts/check_spec.py [archivos...]   (sin argumentos: todas las specs del repo)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REQUIRED = [
    "Problema",
    "Objetivos y alcance",
    "Requerimientos funcionales",
    "Requerimientos no funcionales",
    "Criterios de aceptación",
    "Manejo de errores",
    "Seguridad",
    "Estrategia de testing",
    "Dependencias e impacto",
]
CHECKABLE = re.compile(r"^\s*-\s*\[[ xX]\]\s*AC-\d+", re.MULTILINE)
ISSUE = re.compile(r"\*\*Issue:\*\*.*#\d+")


def _sections(text: str) -> dict[str, str]:
    parts = re.split(r"^##\s+(.+?)\s*$", text, flags=re.MULTILINE)
    return {parts[i].strip(): parts[i + 1] for i in range(1, len(parts) - 1, 2)}


def problems(text: str) -> list[str]:
    found = _sections(text)
    out = [f"falta la sección '## {s}'" for s in REQUIRED if s not in found]
    ac = found.get("Criterios de aceptación", "")
    if "Criterios de aceptación" in found and not CHECKABLE.search(ac):
        out.append(
            "los criterios de aceptación deben ser verificables: '- [ ] AC-N: dado…, cuando…, entonces…'"
        )
    if not ISSUE.search(text):
        out.append("falta '**Issue:** #N' (trazabilidad)")
    return out


def main(argv: list[str]) -> int:
    root = Path(__file__).resolve().parents[1]
    paths = [Path(a) for a in argv] or sorted((root / "docs" / "specs").glob("*.md"))
    paths = [p for p in paths if not p.name.startswith("_")]
    failed = 0
    for p in paths:
        for prob in problems(p.read_text(encoding="utf-8")):
            print(f"::error file={p}::{prob}")
            failed += 1
    print(f"check_spec: {len(paths)} specs, {failed} problemas")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
