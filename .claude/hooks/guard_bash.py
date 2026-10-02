"""PreToolUse (Bash|PowerShell): bloquea o pide aprobación para comandos de riesgo.

deny: reescritura de historial, push a main, saltear controles, medios al repo, lectura de .env,
      borrado de recursos remotos. ask: borrados recursivos, reset --hard, merges, cambios de
      protección de rama, instalación de dependencias.
"""

from __future__ import annotations

import re
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _hooklib import Decision, emit_pre_tool_use, read_payload, strongest

PROTECTED_BRANCHES = {"main", "master"}
MEDIA_EXT = re.compile(r"\.(mp4|mkv|mov|avi|webm|m4v)$", re.IGNORECASE)
DATA_DIRS = ("videos/", "data/", "training_data/", "svhn_data/", "videos", "data")
READERS = {"cat", "type", "get-content", "gc", "less", "more", "head", "tail", "bat", "source", "."}
SEPARATORS = re.compile(r"&&|\|\||;|\||\r?\n")


def _tokens(segment: str) -> list[str]:
    try:
        return shlex.split(segment, posix=True)
    except ValueError:
        return segment.split()


def _git_sub(tokens: list[str]) -> tuple[str, list[str]] | None:
    """Devuelve (subcomando, argumentos) para un comando git, salteando opciones globales."""
    if not tokens or Path(tokens[0]).name.lower() not in {"git", "git.exe"}:
        return None
    i = 1
    while i < len(tokens) and tokens[i].startswith("-"):
        i += 2 if tokens[i] in {"-C", "-c"} else 1
    if i >= len(tokens):
        return None
    return tokens[i], tokens[i + 1 :]


def _short_flags(args: list[str]) -> str:
    return "".join(a[1:] for a in args if a.startswith("-") and not a.startswith("--"))


def _check_git(sub: str, args: list[str], current_branch: str | None) -> Decision | None:
    flags = set(args)
    short = _short_flags(args)
    if sub == "push":
        if flags & {"--force", "--force-with-lease", "--force-if-includes", "--mirror"} or "f" in short:
            return Decision("deny", "push forzado: no se reescribe historial compartido sin autorización")
        if any(a.startswith("--force-with-lease=") for a in args):
            return Decision("deny", "push forzado: no se reescribe historial compartido sin autorización")
        positional = [a for a in args if not a.startswith("-")]
        refspecs = positional[1:]
        if any(r.startswith("+") for r in refspecs):
            return Decision("deny", "refspec forzado (+): reescribe historial")
        if "--delete" in flags or "-d" in flags or any(r.startswith(":") for r in refspecs):
            return Decision("ask", "borrar una rama remota es irreversible para otros")
        targets = {r.split(":")[-1].removeprefix("refs/heads/") for r in refspecs}
        if targets & PROTECTED_BRANCHES:
            return Decision("deny", "push directo a main: todo cambio entra por Pull Request")
        if not refspecs and current_branch in PROTECTED_BRANCHES:
            return Decision("deny", "push directo a main (rama actual): todo cambio entra por Pull Request")
    elif sub == "commit":
        if "--no-verify" in flags or "n" in short or "--no-gpg-sign" in flags:
            return Decision("deny", "no se saltean hooks ni firma de commits")
    elif sub == "add":
        paths = [a for a in args if not a.startswith("-")]
        if any(MEDIA_EXT.search(p) for p in paths):
            return Decision("deny", "videos fuera del repo (repo público): ver docs/data/DATASETS.md")
        if any(p.replace("\\", "/").startswith(DATA_DIRS) for p in paths):
            return Decision("deny", "directorios de datos fuera del repo (repo público)")
        if "--force" in flags or "f" in short:
            return Decision("ask", "git add -f agrega archivos ignorados a propósito")
    elif sub == "reset" and "--hard" in flags:
        return Decision("ask", "reset --hard descarta cambios sin recuperación")
    elif sub == "clean" and ("f" in short or "--force" in flags):
        return Decision("ask", "git clean -f borra archivos no versionados")
    elif sub == "branch" and ("D" in short or ("--delete" in flags and "--force" in flags)):
        return Decision("ask", "borrado forzado de rama")
    return None


def _check_gh(args: list[str]) -> Decision | None:
    if args[:2] == ["repo", "delete"] or args[:2] == ["release", "delete"]:
        return Decision("deny", "borrado de recursos de GitHub: operación irreversible")
    if args[:1] == ["api"]:
        method = ""
        for i, a in enumerate(args):
            if a in {"-X", "--method"} and i + 1 < len(args):
                method = args[i + 1].upper()
            elif a.startswith("--method="):
                method = a.split("=", 1)[1].upper()
        if method == "DELETE":
            return Decision("deny", "DELETE contra la API de GitHub: operación irreversible")
        if any("/protection" in a for a in args) and method in {"PUT", "POST", "PATCH"}:
            return Decision("ask", "cambio de branch protection (control de seguridad)")
    if args[:2] == ["pr", "merge"]:
        if "--admin" in args:
            return Decision("deny", "merge con --admin saltea la protección de rama")
        return Decision("ask", "merge a main: checkpoint humano")
    return None


def _check_segment(tokens: list[str], current_branch: str | None) -> Decision | None:
    if not tokens:
        return None
    lowered = [t.lower() for t in tokens]
    # token exacto: mencionar la opción en un texto (p. ej. un mensaje de commit) no es usarla
    if "--dangerously-skip-permissions" in lowered or "bypasspermissions" in lowered:
        return Decision("deny", "no se usan permisos indiscriminados (bypassPermissions)")
    head = Path(lowered[0]).name
    if head in READERS and any(Path(t).name.startswith(".env") for t in tokens[1:]):
        return Decision("deny", "lectura de archivo de variables de entorno: puede exponer secretos")
    git = _git_sub(tokens)
    if git:
        return _check_git(git[0], git[1], current_branch)
    if head in {"gh", "gh.exe"}:
        return _check_gh(tokens[1:])
    if head == "rm" and ("r" in _short_flags(tokens[1:]).lower() or "--recursive" in lowered):
        return Decision("ask", "borrado recursivo")
    if head in {"remove-item", "ri", "rmdir", "rd", "del"} and any(
        t in {"-recurse", "-r", "/s"} for t in lowered[1:]
    ):
        return Decision("ask", "borrado recursivo")
    if (head == "uv" and lowered[1:2] in (["add"], ["pip"])) or (
        head in {"pip", "pip3", "npm", "pnpm", "yarn"} and lowered[1:2] in (["install"], ["add"], ["i"])
    ):
        return Decision("ask", "instalar dependencias requiere que estén en el plan aprobado")
    if head in {"python", "python3", "py"} and lowered[1:4] == ["-m", "pip", "install"]:
        return Decision("ask", "instalar dependencias requiere que estén en el plan aprobado")
    return None


def evaluate(command: str, current_branch: str | None = None) -> Decision | None:
    segments = [_tokens(s) for s in SEPARATORS.split(command)]
    return strongest([_check_segment(t, current_branch) for t in segments])


def _current_branch() -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return r.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def main() -> None:
    payload = read_payload()
    command = ((payload or {}).get("tool_input") or {}).get("command")
    if not isinstance(command, str) or not command.strip():
        emit_pre_tool_use(
            Decision("ask", "no se pudo leer el comando; se pide confirmación"), "guard_bash", payload
        )
        return
    bare_push = re.search(r"\bgit\s+push\s*($|[;&|])", command) is not None
    emit_pre_tool_use(evaluate(command, _current_branch() if bare_push else None), "guard_bash", payload)


if __name__ == "__main__":
    main()
