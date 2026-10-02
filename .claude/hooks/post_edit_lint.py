"""PostToolUse (Edit|MultiEdit|Write): ruff format + ruff check --fix sobre el .py editado.

Si quedan errores que ruff no puede corregir, se devuelven al agente (decision=block) para que los
arregle antes de seguir. legacy/ queda excluido. Si uv no está disponible, no hace nada.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _hooklib import log_event, project_dir, read_payload

LINTED_ROOTS = ("src/", "tests/", "scripts/", ".claude/hooks/")


def lint(file_path: Path, root: Path) -> str | None:
    try:
        rel = file_path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return None
    if file_path.suffix != ".py" or not rel.startswith(LINTED_ROOTS) or not file_path.is_file():
        return None
    uv = shutil.which("uv")
    if uv is None:
        return None
    base = [uv, "run", "--no-sync", "--quiet", "ruff"]
    try:
        subprocess.run([*base, "check", "--fix", "--quiet", rel], cwd=root, capture_output=True, timeout=60)
        subprocess.run([*base, "format", rel], cwd=root, capture_output=True, timeout=60)
        r = subprocess.run(
            [*base, "check", "--output-format", "concise", rel],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as e:
        log_event({"event": "lint_error", "error": type(e).__name__})
        return None
    if r.returncode == 0:
        return None
    return (r.stdout + r.stderr).strip()[:2000]


def main() -> None:
    payload = read_payload() or {}
    fp = (payload.get("tool_input") or {}).get("file_path")
    if not isinstance(fp, str):
        return
    problems = lint(Path(fp), project_dir())
    if problems:
        log_event({"event": "lint_block", "session": payload.get("session_id")})
        sys.stdout.write(
            json.dumps(
                {"decision": "block", "reason": f"ruff encontró errores no auto-corregibles:\n{problems}"}
            )
        )


if __name__ == "__main__":
    main()
