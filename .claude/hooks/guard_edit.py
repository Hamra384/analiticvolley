"""PreToolUse (Edit|MultiEdit|Write|NotebookEdit): protege el harness, los secretos y los tests.

deny: escribir videos o archivos .env.
ask:  modificar la configuración del harness (settings, hooks, workflows, checks, baseline) —
      el agente no relaja sus propias restricciones sin aprobación humana—, o debilitar tests
      (agregar skip/xfail sin referencia a un Issue, quitar asserts, quitar funciones de test).
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _hooklib import Decision, emit_pre_tool_use, project_dir, read_payload

PROTECTED_EXACT = {".claude/settings.json", ".github/codeowners", "quality/baseline.json"}
PROTECTED_PREFIX = (".claude/hooks/", ".github/workflows/")
PROTECTED_RE = re.compile(r"^scripts/check_[\w-]+\.py$")
MEDIA_EXT = re.compile(r"\.(mp4|mkv|mov|avi|webm|m4v)$", re.IGNORECASE)
SKIP_RE = re.compile(r"pytest\.mark\.skip|pytest\.skip\(|xfail|unittest\.skip|@skip\b")
ISSUE_REF = re.compile(r"#\d+")
ASSERT_RE = re.compile(r"^\s*(assert\b|with pytest\.raises\b)", re.MULTILINE)
TEST_DEF_RE = re.compile(r"^\s*(async\s+)?def test_", re.MULTILINE)


def _relative(file_path: str, project_root: Path) -> str:
    fp = Path(file_path).as_posix()
    root = project_root.as_posix().rstrip("/")
    if fp.lower().startswith(root.lower() + "/"):
        return fp[len(root) + 1 :]
    return fp


def _is_test_file(rel: str) -> bool:
    name = rel.rsplit("/", 1)[-1]
    return name.endswith(".py") and (rel.startswith("tests/") or name.startswith("test_"))


def _old_new(tool: str, tool_input: dict[str, Any], path: Path) -> tuple[str, str]:
    if tool == "Edit":
        return str(tool_input.get("old_string", "")), str(tool_input.get("new_string", ""))
    if tool == "MultiEdit":
        edits = tool_input.get("edits") or []
        return (
            "\n".join(str(e.get("old_string", "")) for e in edits),
            "\n".join(str(e.get("new_string", "")) for e in edits),
        )
    try:
        current = path.read_text(encoding="utf-8") if path.is_file() else ""
    except OSError:
        current = ""
    return current, str(tool_input.get("content", ""))


def _weakens_tests(old: str, new: str) -> str | None:
    old_skips = Counter(line.strip() for line in old.splitlines() if SKIP_RE.search(line))
    new_skips = Counter(line.strip() for line in new.splitlines() if SKIP_RE.search(line))
    added = [line for line in (new_skips - old_skips).elements() if not ISSUE_REF.search(line)]
    if added:
        return f"agrega skip/xfail sin referencia a un Issue (#N): {added[0][:80]}"
    if len(ASSERT_RE.findall(new)) < len(ASSERT_RE.findall(old)):
        return "reduce la cantidad de asserts en un test"
    if len(TEST_DEF_RE.findall(new)) < len(TEST_DEF_RE.findall(old)):
        return "elimina funciones de test"
    return None


def evaluate(tool: str, tool_input: dict[str, Any], project_root: Path) -> Decision | None:
    file_path = tool_input.get("file_path") or tool_input.get("notebook_path")
    if not isinstance(file_path, str) or not file_path:
        return Decision("ask", "no se pudo determinar el archivo a modificar")
    rel = _relative(file_path, project_root)
    name = rel.rsplit("/", 1)[-1]
    if MEDIA_EXT.search(name):
        return Decision("deny", "los videos no se escriben dentro del repo (repo público)")
    if name.startswith(".env"):
        return Decision("deny", "archivos de variables de entorno: posible exposición de secretos")
    if rel.lower() in PROTECTED_EXACT or rel.startswith(PROTECTED_PREFIX) or PROTECTED_RE.match(rel):
        return Decision("ask", f"{rel} es parte de los controles del harness: requiere aprobación humana")
    if _is_test_file(rel):
        old, new = _old_new(tool, tool_input, Path(file_path))
        reason = _weakens_tests(old, new)
        if reason:
            return Decision("ask", f"posible debilitamiento de tests en {rel}: {reason}")
    return None


def main() -> None:
    payload = read_payload()
    tool = str((payload or {}).get("tool_name", ""))
    tool_input = (payload or {}).get("tool_input") or {}
    emit_pre_tool_use(evaluate(tool, tool_input, project_dir()), "guard_edit", payload)


if __name__ == "__main__":
    main()
