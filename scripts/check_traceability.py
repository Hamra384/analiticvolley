"""Trazabilidad de Pull Requests: PR → Issue, y cambios de producto → Spec.

Reglas:
1. El cuerpo del PR referencia al menos un Issue (Closes/Fixes/Resolves/Refs #N).
2. Si el PR cambia código de producto (src/volley_cv/**), el cuerpo cita la spec (docs/specs/...)
   o declara 'Spec: N/A (<motivo>)'.
Uso en CI: PR_BODY=... python scripts/check_traceability.py <archivo-con-lista-de-cambios>
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ISSUE = re.compile(r"\b(close[sd]?|fix(e[sd])?|resolve[sd]?|refs?)\s+#\d+", re.IGNORECASE)
SPEC = re.compile(r"docs/specs/[\w./-]+\.md")
SPEC_NA = re.compile(r"Spec:\s*N/A\s*\(([^)]{10,})\)", re.IGNORECASE)


def problems(body: str, changed: list[str]) -> list[str]:
    out = []
    if not ISSUE.search(body):
        out.append("el PR no referencia un Issue (usar 'Closes #N' o 'Refs #N')")
    touches_product = any(f.startswith("src/volley_cv/") and not f.endswith("__init__.py") for f in changed)
    if touches_product and not (SPEC.search(body) or SPEC_NA.search(body)):
        out.append("cambia src/volley_cv/ sin citar la spec (docs/specs/...) ni 'Spec: N/A (<motivo>)'")
    return out


def main(argv: list[str]) -> int:
    body = os.environ.get("PR_BODY", "")
    changed = Path(argv[0]).read_text(encoding="utf-8").split() if argv else []
    probs = problems(body, changed)
    for p in probs:
        print(f"::error::{p}")
    print(f"check_traceability: {len(changed)} archivos cambiados, {len(probs)} problemas")
    return 1 if probs else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
